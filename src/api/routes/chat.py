# src/api/routes/chat.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas.chat import ChatRequest, ChatResponse
from src.core.exceptions import GenerationError
from src.embedding.indexer import get_vector_store
from src.llm.context_builder import build_context
from src.llm.generator import generate_answer, rewrite_query
from src.retrieval.retriever import similarity_search
from src.utils.logger import get_logger
from src.llm.vision import analyze_images

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question against the document corpus",
    description=(
        "Accepts a conversation history, retrieves the top-k most relevant "
        "document chunks from the vector store, generates a grounded answer "
        "using gpt-4o-mini, and returns the answer with source citations. "
        "Status 422 if last message is not from user. Status 503 if generation fails. "
        "Status 404 if no relevant documents are found."
    ),
)
async def chat(request: ChatRequest) -> ChatResponse:
    """End-to-end conversational RAG handler.

    Flow:
        ChatRequest
            → rewrite_query()             rewrite follow-up into standalone query
            → similarity_search()         retrieve top-k chunks
            → build_context()             format chunks + extract sources
            → generate_answer()           call LLM with context + history
            → ChatResponse                return answer + sources

    Args:
        request: Validated ChatRequest containing history and k.

    Returns:
        ChatResponse with answer string and list of source documents.

    Raises:
        HTTPException 404: No relevant documents found for the query.
        HTTPException 503: LLM generation or rewrite failed.
        HTTPException 500: Unexpected internal error.
    """
    logger.info(
        f"Chat request received | "
        f"query_length={len(request.query)} "
        f"history_turns={len(request.history)} "
        f"k={request.k}"
    )

    attachment_available = False

    # --- Step -1: If there are attachments, analyze first ---
    if len(request.history[-1].pictures) > 0:
        attachment_available = True

        try:
            analyze_images_result = await analyze_images(
                image_urls=[v.data for v in request.history[-1].pictures]
            )
            logger.info("Analyze images results: %s", analyze_images_result)
        except GenerationError as exc:
            logger.error(f"Analyze images failed: {exc}")
            raise HTTPException(
                status_code=503,
                detail="Analyze images failed. Please try again later.",
            )

    # --- Step 0: Rewrite query using conversation history ---
    try:
        retrieval_query = await rewrite_query(
            query=request.query,
            history=request.history,
        )
    except GenerationError as exc:
        logger.error(f"Query rewrite failed: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Query rewriting failed. Please try again later.",
        )

    # --- Step 1: Retrieve relevant chunks ---
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
        logger.warning(f"No relevant documents found | query='{retrieval_query}'")
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

    # --- Step 3: Generate answer ---
    try:
        answer = await generate_answer(
            original_question=request.query,
            query=retrieval_query,
            context=context_str,
            history=request.history,
        )
    except GenerationError as exc:
        logger.error(f"Generation failed: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Answer generation failed. Please try again later.",
        )
    except ValueError as exc:
        logger.error(f"Generation input error: {exc}")
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )

    logger.info(
        f"Chat response ready | "
        f"answer_length={len(answer)} sources={len(sources)}"
    )
    
    return ChatResponse(
        query=request.query,
        answer=answer,
        sources=sources,
    )