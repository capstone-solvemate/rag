from __future__ import annotations

from typing import List, Optional, Tuple

from langchain.schema import Document

from src.api.schemas.response import SourceDocument


def _get_meta(doc: Document, key: str, default: Optional[str] = None) -> Optional[str]:
    """Safely retrieve a metadata field from a Document.

    Args:
        doc:     LangChain Document object.
        key:     Metadata key to retrieve.
        default: Value to return if key is absent or None.

    Returns:
        The metadata value as a string, or default.
    """
    value = doc.metadata.get(key, default)
    if value is None:
        return default
    return str(value)


def build_context(
    documents: List[Document],
) -> Tuple[str, List[SourceDocument]]:
    """Format retrieved documents into a numbered context block for prompt injection.

    Each block is headed with a citation marker and key source metadata.
    Only fields guaranteed by the loader/chunker contract are used.
    PDF-specific fields (page, total_pages) are included only when present.

    Args:
        documents: List of LangChain Document objects returned by the retriever.

    Returns:
        context_str:  Formatted string ready for injection into the prompt.
        sources:      List of SourceDocument objects for the API response.

    Raises:
        ValueError: If documents list is empty.

    Example output (context_str):
        [1] Source: manual L3210.pdf | Type: .pdf | Chunk: 3 | Page: 4
        ...chunk text...

        [2] Source: policy.docx | Type: .docx | Chunk: 11
        ...chunk text...
    """
    if not documents:
        raise ValueError("Cannot build context from empty document list.")

    context_blocks: List[str] = []
    sources: List[SourceDocument] = []

    for i, doc in enumerate(documents, start=1):
        file_name = _get_meta(doc, "file_name", default="unknown")
        file_path = _get_meta(doc, "file_path", default="unknown")
        file_type = _get_meta(doc, "file_type", default="unknown")
        chunk_index = _get_meta(doc, "chunk_index", default="0")
        page = _get_meta(doc, "page")  # PDF-only, may be None

        # Build header — include page only when available
        header_parts = [
            f"Source: {file_name}",
            f"Type: {file_type}",
            f"Chunk: {chunk_index}",
        ]
        if page is not None:
            # page is 0-indexed in the metadata, display as 1-indexed
            try:
                display_page = int(page) + 1
            except (ValueError, TypeError):
                display_page = page
            header_parts.append(f"Page: {display_page}")

        header = f"[{i}] " + " | ".join(header_parts)
        block = f"{header}\n{doc.page_content.strip()}"
        context_blocks.append(block)

        # Build source record for API response
        sources.append(
            SourceDocument(
                file_name=file_name,
                file_path=file_path,
                chunk_index=int(chunk_index),
            )
        )

    context_str = "\n\n".join(context_blocks)
    return context_str, sources