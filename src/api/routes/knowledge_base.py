"""
Knowledge Base management endpoints.

POST   /knowledge-base          — load, chunk, and index a single document
DELETE /knowledge-base/{doc_id} — remove all vectors for a document
"""
from pathlib import Path

from fastapi import APIRouter, HTTPException
from langchain_core.documents import Document

from src.api.schemas.knowledge_base import (
    AddKnowledgeBaseRequest,
    AddKnowledgeBaseResponse,
    DeleteKnowledgeBaseResponse,
)
from src.data.loader import load_single_document
from src.data.chunker import chunk_documents
from src.embedding.indexer import get_ids_by_doc_id, delete_by_doc_id, index_documents
from src.utils.logger import get_logger

logger = get_logger("src.api.routes.knowledge_base")

router = APIRouter()


@router.post(
    "/knowledge-base",
    response_model=AddKnowledgeBaseResponse,
    status_code=200,
    summary="Index a new document into the knowledge base",
)
def add_knowledge_base(request: AddKnowledgeBaseRequest) -> AddKnowledgeBaseResponse:
    """
    Load, chunk, and index a single document.

    The Express backend is responsible for:
    - Generating doc_id (UUID)
    - Saving the file to shared storage
    - Sending file_path to this endpoint

    Error codes:
        FILE_NOT_FOUND        — file_path does not exist on disk
        UNSUPPORTED_FILE_TYPE — file extension not in {.txt, .pdf, .docx}
        DOC_ID_CONFLICT       — doc_id is already present in Chroma
        INDEXING_FAILED       — unexpected failure during embedding/storage
    """
    logger.info(
        f"POST /knowledge-base | doc_id={request.doc_id!r} "
        f"file_name={request.file_name!r}"
    )

    # Guard 1: file must exist on disk
    if not Path(request.file_path).exists():
        logger.warning(f"File not found: {request.file_path!r}")
        raise HTTPException(
            status_code=400,
            detail={
                "detail": f"File not found: {request.file_path}",
                "error_code": "FILE_NOT_FOUND",
            },
        )

    # Guard 2: load document — empty return means unsupported extension
    ext = Path(request.file_path).suffix.lower()
    documents = load_single_document(request.file_path)
    if not documents:
        logger.warning(
            f"Unsupported file type for {request.file_name!r}. "
            f"Extension: {ext!r}"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "detail": (
                    f"Unsupported file type: {ext}. "
                    "Supported formats: .txt, .pdf, .docx"
                ),
                "error_code": "UNSUPPORTED_FILE_TYPE",
            },
        )

    # Guard 3: doc_id must not already be indexed
    try:
        existing_ids = get_ids_by_doc_id(request.doc_id)
    except Exception as exc:
        logger.exception(f"Chroma query failed during conflict check: {exc}")
        raise HTTPException(
            status_code=500,
            detail={
                "detail": "Unexpected failure while checking the knowledge base.",
                "error_code": "INDEXING_FAILED",
            },
        )

    if existing_ids:
        logger.warning(f"doc_id already indexed: {request.doc_id!r}")
        raise HTTPException(
            status_code=409,
            detail={
                "detail": f"doc_id already indexed: {request.doc_id}",
                "error_code": "DOC_ID_CONFLICT",
            },
        )

    # Happy path: chunk and index
    try:
        chunks: list[Document] = chunk_documents(documents)
        index_documents(chunks, doc_id=request.doc_id)
    except Exception as exc:
        logger.exception(
            f"Indexing failed for doc_id={request.doc_id!r}: {exc}"
        )
        raise HTTPException(
            status_code=500,
            detail={
                "detail": "Unexpected failure during document indexing.",
                "error_code": "INDEXING_FAILED",
            },
        )

    logger.info(
        f"Indexed doc_id={request.doc_id!r} | "
        f"{len(chunks)} chunks | file={request.file_name!r}"
    )

    return AddKnowledgeBaseResponse(
        doc_id=request.doc_id,
        file_name=request.file_name,
        chunks_indexed=len(chunks),
        status="indexed",
    )


@router.delete(
    "/knowledge-base/{doc_id}",
    response_model=DeleteKnowledgeBaseResponse,
    status_code=200,
    summary="Remove all vectors for a document from the knowledge base",
)
def delete_knowledge_base(doc_id: str) -> DeleteKnowledgeBaseResponse:
    """
    Delete every chunk vector associated with the given doc_id.

    Error codes:
        DOC_NOT_FOUND   — no vectors found for this doc_id in Chroma
        DELETION_FAILED — unexpected failure during vector removal
    """
    logger.info(f"DELETE /knowledge-base/{doc_id!r}")

    # Fetch count and delete in one adapter call
    try:
        chunks_deleted = delete_by_doc_id(doc_id)
    except Exception as exc:
        logger.exception(f"Deletion failed for doc_id={doc_id!r}: {exc}")
        raise HTTPException(
            status_code=500,
            detail={
                "detail": "Unexpected failure during vector deletion.",
                "error_code": "DELETION_FAILED",
            },
        )

    if chunks_deleted == 0:
        logger.warning(f"doc_id not found in Chroma: {doc_id!r}")
        raise HTTPException(
            status_code=404,
            detail={
                "detail": f"doc_id not found: {doc_id}",
                "error_code": "DOC_NOT_FOUND",
            },
        )

    return DeleteKnowledgeBaseResponse(
        doc_id=doc_id,
        chunks_deleted=chunks_deleted,
        status="deleted",
    )   