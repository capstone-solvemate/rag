from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas.request import ChatRequest
from src.api.schemas.response import ChatResponse
from src.core.exceptions import GenerationError
from src.embedding.indexer import get_vector_store
from src.llm.context_builder import build_context
from src.llm.generator import generate_answer
from src.retrieval.retriever import similarity_search
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question against the document corpus",
    description=(
        "Accepts a natural language query, retrieves the top-k most relevant "
        "document chunks from the vector store, generates a grounded answer "
        "using gpt-4o-mini, and returns the answer with source citations. "
        "Status 422 if query is empty. Status 503 if generation fails. "
        "Status 404 if no relevant documents are found."
    ),
)
async def chat(request: ChatRequest) -> ChatResponse:
    """End-to-end RAG handler.

    Flow:
        ChatRequest
            → similarity_search()         retrieve top-k chunks
            → build_context()             format chunks + extract sources
            → generate_answer()           call LLM with context
            → ChatResponse                return answer + sources

    Args:
        request: Validated ChatRequest containing query and k.

    Returns:
        ChatResponse with answer string and list of source documents.

    Raises:
        HTTPException 404: No relevant documents found for the query.
        HTTPException 503: LLM generation failed.
        HTTPException 500: Unexpected internal error.
    """
    logger.info(
        f"Chat request received | "
        f"query_length={len(request.query)} k={request.k}"
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
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found for the given query.",
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
        answer = generate_answer(query=request.query, context=context_str)
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