from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas.detection import DetectionRequest, DetectionResponse
from src.core.exceptions import GenerationError
from src.embedding.indexer import get_vector_store
from src.llm.context_builder import build_context, build_detection_retrieval_query
from src.llm.generator import analyze_image_for_defects, generate_answer_with_image
from src.retrieval.retriever import similarity_search
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/detect",
    response_model=DetectionResponse,
    summary="Detect printer defects or print quality issues from an image",
    description=(
        "Analyzes the provided image for printer part defects or print quality issues. "
        "The image drives both detection and retrieval — no text query is needed. "
        "Returns structured detection results with recommended actions grounded "
        "in the indexed knowledge base. "
        "Status 503 if the detection or generation LLM call fails. "
        "Status 500 if retrieval or context building fails."
    ),
)
async def detect(request: DetectionRequest) -> DetectionResponse:
    """Detection pipeline handler.

    Flow:
        DetectionRequest
            → analyze_image_for_defects()       gpt-4o vision → structured DetectionResult
            → build_detection_retrieval_query()  DetectionResult → retrieval query string
            → similarity_search()               retrieve top-k KB chunks
            → build_context()                   format chunks + extract sources
            → generate_answer_with_image()      gpt-4o-mini → grounded recommended actions
            → DetectionResponse                 return full structured result

    Args:
        request: Validated DetectionRequest containing image_base64, media_type,
                 detection_mode, and k.

    Returns:
        DetectionResponse with detected issues, severity, confidence,
        recommended actions, and source documents.

    Raises:
        HTTPException 503: Detection LLM call or generation failed.
        HTTPException 500: Retrieval or context building failed.
    """
    logger.info(
        f"Detection request received | "
        f"detection_mode={request.detection_mode} "
        f"media_type={request.media_type} "
        f"k={request.k}"
    )

    # --- Step 1: Vision analysis — image → structured DetectionResult ---
    try:
        detection_result = await analyze_image_for_defects(
            image_base64=request.image_base64,
            media_type=request.media_type,
            detection_mode=request.detection_mode,
        )
    except GenerationError as exc:
        logger.error(f"Detection analysis failed: {exc.message}")
        raise HTTPException(
            status_code=503,
            detail={
                "detail": exc.message,
                "error_code": "DETECTION_FAILED",
            },
        )

    # --- Step 2: Build retrieval query from detection result ---
    # If the LLM found nothing (both lists empty), skip retrieval entirely.
    # This is a valid outcome — not an error.
    try:
        retrieval_query = build_detection_retrieval_query(detection_result)
    except ValueError:
        logger.info("Detection found no issues — returning empty result.")
        return DetectionResponse(
            detected_issues=[],
            affected_components=[],
            severity=detection_result.severity,
            confidence=detection_result.confidence,
            recommended_actions="No issues detected in the provided image.",
            sources=[],
        )

    # --- Step 3: Retrieve relevant KB chunks ---
    try:
        vector_store = get_vector_store()
        documents = similarity_search(
            query=retrieval_query,
            vector_store=vector_store,
            k=request.k,
        )
    except Exception as exc:
        logger.error(f"Retrieval failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Document retrieval failed.",
        )

    if not documents:
        logger.warning(f"No relevant KB chunks found | query='{retrieval_query}'")
        return DetectionResponse(
            detected_issues=detection_result.detected_issues,
            affected_components=detection_result.affected_components,
            severity=detection_result.severity,
            confidence=detection_result.confidence,
            recommended_actions=(
                "Issues detected but no relevant documentation found in the knowledge base."
            ),
            sources=[],
        )

    logger.info(f"Retrieved {len(documents)} chunks.")

    # --- Step 4: Build context and extract sources ---
    try:
        context_str, sources = build_context(documents)
    except ValueError as exc:
        logger.error(f"Context building failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Failed to build context from retrieved documents.",
        )

    # --- Step 5: Generate grounded recommended actions ---
    try:
        recommended_actions = await generate_answer_with_image(
            query=retrieval_query,
            context=context_str,
            image_base64=request.image_base64,
            media_type=request.media_type,
        )
    except GenerationError as exc:
        logger.error(f"Recommendation generation failed: {exc.message}")
        raise HTTPException(
            status_code=503,
            detail={
                "detail": exc.message,
                "error_code": "GENERATION_FAILED",
            },
        )
    except ValueError as exc:
        logger.error(f"Generation input error: {exc}")
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )

    logger.info(
        f"Detection response ready | "
        f"issues={len(detection_result.detected_issues)} "
        f"severity={detection_result.severity} "
        f"sources={len(sources)}"
    )

    return DetectionResponse(
        detected_issues=detection_result.detected_issues,
        affected_components=detection_result.affected_components,
        severity=detection_result.severity,
        confidence=detection_result.confidence,
        recommended_actions=recommended_actions,
        sources=sources,
    )