import pytest
from langchain.schema import Document

from src.llm.context_builder import build_context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_doc(
    content: str,
    file_name: str = "test.pdf",
    file_path: str = "/data/raw/test.pdf",
    file_type: str = ".pdf",
    chunk_index: int = 0,
    page: int | None = None,
) -> Document:
    metadata = {
        "file_name": file_name,
        "file_path": file_path,
        "file_type": file_type,
        "chunk_index": chunk_index,
    }
    if page is not None:
        metadata["page"] = page
    return Document(page_content=content, metadata=metadata)


# ---------------------------------------------------------------------------
# build_context — happy path
# ---------------------------------------------------------------------------

def test_single_document_returns_one_block():
    docs = [make_doc("Hello world.", chunk_index=2)]
    context_str, sources = build_context(docs)

    assert "[1]" in context_str
    assert "Hello world." in context_str
    assert len(sources) == 1


def test_multiple_documents_are_numbered_sequentially():
    docs = [
        make_doc("First chunk.", chunk_index=0),
        make_doc("Second chunk.", chunk_index=1),
        make_doc("Third chunk.", chunk_index=2),
    ]
    context_str, sources = build_context(docs)

    assert "[1]" in context_str
    assert "[2]" in context_str
    assert "[3]" in context_str
    assert len(sources) == 3


def test_source_metadata_is_correct():
    docs = [make_doc("Content.", file_name="report.pdf", file_path="/raw/report.pdf", chunk_index=7)]
    _, sources = build_context(docs)

    assert sources[0].file_name == "report.pdf"
    assert sources[0].file_path == "/raw/report.pdf"
    assert sources[0].chunk_index == 7


def test_context_str_contains_file_name_and_chunk():
    docs = [make_doc("Content.", file_name="manual.pdf", chunk_index=3)]
    context_str, _ = build_context(docs)

    assert "manual.pdf" in context_str
    assert "Chunk: 3" in context_str


def test_file_path_not_exposed_in_context_str():
    """Absolute file paths must not appear in the LLM context string."""
    docs = [make_doc("Content.", file_path="C:\\Users\\usER\\secret\\path\\file.pdf")]
    context_str, _ = build_context(docs)

    assert "C:\\" not in context_str
    assert "usER" not in context_str


# ---------------------------------------------------------------------------
# Page number handling
# ---------------------------------------------------------------------------

def test_page_number_included_when_present():
    docs = [make_doc("Content.", page=0)]  # 0-indexed → should display as Page: 1
    context_str, _ = build_context(docs)

    assert "Page: 1" in context_str


def test_page_number_incremented_correctly():
    docs = [make_doc("Content.", page=4)]  # 0-indexed → should display as Page: 5
    context_str, _ = build_context(docs)

    assert "Page: 5" in context_str


def test_page_number_absent_when_not_in_metadata():
    docs = [make_doc("Content.", page=None)]
    context_str, _ = build_context(docs)

    assert "Page:" not in context_str


# ---------------------------------------------------------------------------
# Missing/partial metadata handling
# ---------------------------------------------------------------------------

def test_missing_file_name_falls_back_to_unknown():
    doc = Document(page_content="Content.", metadata={"chunk_index": 0})
    context_str, sources = build_context([doc])

    assert "unknown" in context_str
    assert sources[0].file_name == "unknown"


def test_whitespace_in_content_is_stripped():
    docs = [make_doc("   trimmed content   ")]
    context_str, _ = build_context(docs)

    assert "trimmed content" in context_str
    assert "   trimmed" not in context_str


def test_blocks_separated_by_double_newline():
    docs = [make_doc("First."), make_doc("Second.")]
    context_str, _ = build_context(docs)

    assert "\n\n" in context_str


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_empty_document_list_raises_value_error():
    with pytest.raises(ValueError, match="empty document list"):
        build_context([])