from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas.chat import ChatResponse
from src.api.schemas.image_chat import ImageChatRequest
from src.core.exceptions import GenerationError
from src.embedding.indexer import get_vector_store
from src.llm.context_builder import build_context
from src.llm.generator import generate_answer_with_image
from src.retrieval.retriever import similarity_search
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/chat/image",
    response_model=ChatResponse,
    summary="Ask a question with an image against the document corpus",
    description=(
        "Accepts a text query and a base64-encoded image. Retrieves the top-k most "
        "relevant document chunks from the vector store using the text query, then "
        "generates a grounded answer using gpt-4o-mini with both the retrieved context "
        "and the image as visual input. Returns the answer with source citations. "
        "The image is visual context only — it does not affect retrieval. "
        "Status 422 if query or image_base64 is blank. "
        "Status 503 if generation fails. "
        "Status 404 if no relevant documents are found."
    ),
)
async def chat_with_image(request: ImageChatRequest) -> ChatResponse:
    """Vision-augmented RAG handler.

    Flow:
        ImageChatRequest
            → similarity_search()             retrieve top-k chunks via text query
            → build_context()                 format chunks + extract sources
            → generate_answer_with_image()    call gpt-4o-mini with context + image
            → ChatResponse                    return answer + sources

    Note: Query rewriting is intentionally skipped here. The /chat/image
    endpoint is a single-turn interaction — there is no conversation history
    to rewrite against. The query is used as-is for retrieval.

    Args:
        request: Validated ImageChatRequest containing query, image_base64,
                 media_type, and k.

    Returns:
        ChatResponse with answer string and list of source documents.

    Raises:
        HTTPException 404: No relevant documents found for the query.
        HTTPException 503: LLM generation failed.
        HTTPException 500: Unexpected internal error.
    """
    logger.info(
        f"Image chat request received | "
        f"query_length={len(request.query)} "
        f"media_type={request.media_type} "
        f"k={request.k}"
    )

    # --- Step 1: Retrieve relevant chunks ---
    try:
        vector_store = get_vector_store()
        documents = similarity_search(
            query=request.query,
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
        logger.warning(f"No relevant documents found | query='{request.query}'")
        return ChatResponse(
            query=request.query,
            answer="I'm sorry, I couldn't find any relevant information in the knowledge base to answer your question.",
            sources=[],
        )

    logger.info(f"Retrieved {len(documents)} chunks.")

    # --- Step 2: Build context and extract sources ---
    try:
        context_str, sources = build_context(documents)
    except ValueError as exc:
        logger.error(f"Context building failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Failed to build context from retrieved documents.",
        )

    # --- Step 3: Generate answer with image context ---
    try:
        answer = await generate_answer_with_image(
            query=request.query,
            context=context_str,
            image_base64=request.image_base64,
            media_type=request.media_type,
        )
    except GenerationError as exc:
        logger.error(f"Vision generation failed: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Answer generation failed. Please try again later.",
        )
    except ValueError as exc:
        logger.error(f"Vision generation input error: {exc}")
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )

    logger.info(
        f"Image chat response ready | "
        f"answer_length={len(answer)} sources={len(sources)}"
    )

    return ChatResponse(
        query=request.query,
        answer=answer,
        sources=sources,
    )