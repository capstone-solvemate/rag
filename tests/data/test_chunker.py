"""
test(chunker): unit tests for chunk_documents and save_chunks_to_json

Phase 3 — Commit 2

Coverage:
- chunk_documents:
    - Returns List[Document]
    - Chunk count grows with document length
    - chunk_index metadata is sequential and zero-based
    - chunk_size metadata matches actual content length
    - chunk_overlap produces shared text at chunk boundaries
    - Original document metadata is preserved in each chunk
    - Empty input returns empty list without crashing
    - Single short document produces exactly one chunk
    - Custom chunk_size / chunk_overlap params are forwarded correctly
    - Caller's input list is not mutated

- save_chunks_to_json:
    - Returns the output path as a string
    - File contains valid JSON
    - JSON shape: list of objects with chunk_index, content, content_length, metadata
    - content_length matches len(content) for every record
    - chunk_index in JSON matches metadata chunk_index
    - Defaults to config.DATA_PROCESSED_DIR/chunks.json when output_path=None
    - Creates the output directory if it does not exist
    - Handles empty chunk list gracefully (writes [])

No real LangChain splitter behaviour is mocked — chunk_documents is exercised
with real RecursiveCharacterTextSplitter so overlap and boundary logic is
verified against actual output. save_chunks_to_json uses tmp_path for all I/O.
"""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(content: str, **meta) -> Document:
    """Construct a Document with optional extra metadata."""
    return Document(page_content=content, metadata=dict(meta))


def _make_chunks(n: int, size: int = 80) -> list[Document]:
    """
    Return a list of pre-built chunk Documents with chunk_index and chunk_size
    already stamped — mirrors what chunk_documents produces.
    """
    chunks = []
    for i in range(n):
        content = f"chunk {i} " + ("x" * size)
        doc = Document(
            page_content=content,
            metadata={"chunk_index": i, "chunk_size": len(content)},
        )
        chunks.append(doc)
    return chunks


def _long_text(approx_chars: int) -> str:
    """
    Generate a paragraph-separated text long enough to guarantee
    multiple chunks at the default 1000-char chunk_size.
    Paragraphs contain natural word boundaries so the splitter behaves
    predictably.
    """
    sentence = "The quick brown fox jumps over the lazy dog. "
    paragraph = sentence * 10 + "\n\n"
    result = ""
    while len(result) < approx_chars:
        result += paragraph
    return result.strip()


# ---------------------------------------------------------------------------
# chunk_documents
# ---------------------------------------------------------------------------

class TestChunkDocumentsReturnType:
    def test_returns_list(self):
        from src.data.chunker import chunk_documents

        docs = [_doc("short text")]
        result = chunk_documents(docs)

        assert isinstance(result, list)

    def test_each_item_is_document(self):
        from src.data.chunker import chunk_documents

        docs = [_doc("short text")]
        result = chunk_documents(docs)

        for item in result:
            assert isinstance(item, Document)


class TestChunkDocumentsEmptyInput:
    def test_empty_list_returns_empty_list(self):
        from src.data.chunker import chunk_documents

        result = chunk_documents([])

        assert result == []

    def test_empty_list_does_not_raise(self):
        from src.data.chunker import chunk_documents

        # Should complete without any exception
        chunk_documents([])


class TestChunkDocumentsSingleShortDocument:
    def test_short_document_produces_one_chunk(self):
        """
        A document shorter than chunk_size should come back as a single chunk.
        """
        from src.data.chunker import chunk_documents

        text = "This is a short document."
        docs = [_doc(text)]
        chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)

        assert len(chunks) == 1

    def test_short_document_content_preserved(self):
        from src.data.chunker import chunk_documents

        text = "This is a short document."
        docs = [_doc(text)]
        chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)

        assert chunks[0].page_content == text


class TestChunkDocumentsChunkCount:
    def test_long_document_produces_multiple_chunks(self):
        """
        A document ~5× the chunk_size should yield several chunks.
        """
        from src.data.chunker import chunk_documents

        docs = [_doc(_long_text(5000))]
        chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)

        assert len(chunks) > 1

    def test_larger_chunk_size_produces_fewer_chunks(self):
        from src.data.chunker import chunk_documents

        text = _long_text(5000)
        docs = [_doc(text)]

        chunks_small = chunk_documents(docs, chunk_size=500,  chunk_overlap=50)
        chunks_large = chunk_documents(docs, chunk_size=1500, chunk_overlap=50)

        assert len(chunks_small) > len(chunks_large)


class TestChunkDocumentsChunkIndexMetadata:
    def test_chunk_index_starts_at_zero(self):
        from src.data.chunker import chunk_documents

        docs = [_doc(_long_text(3000))]
        chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)

        assert chunks[0].metadata["chunk_index"] == 0

    def test_chunk_index_is_sequential(self):
        from src.data.chunker import chunk_documents

        docs = [_doc(_long_text(5000))]
        chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)

        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i

    def test_chunk_index_present_on_single_chunk(self):
        from src.data.chunker import chunk_documents

        docs = [_doc("tiny")]
        chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)

        assert "chunk_index" in chunks[0].metadata


class TestChunkDocumentsChunkSizeMetadata:
    def test_chunk_size_metadata_equals_content_length(self):
        """chunk.metadata['chunk_size'] must always equal len(chunk.page_content)."""
        from src.data.chunker import chunk_documents

        docs = [_doc(_long_text(5000))]
        chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)

        for chunk in chunks:
            assert chunk.metadata["chunk_size"] == len(chunk.page_content)

    def test_chunk_size_metadata_present_on_all_chunks(self):
        from src.data.chunker import chunk_documents

        docs = [_doc(_long_text(3000))]
        chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)

        for chunk in chunks:
            assert "chunk_size" in chunk.metadata


class TestChunkDocumentsOverlap:
    def test_consecutive_chunks_share_text(self):
        """
        With non-zero overlap the end of chunk[n] should appear at the
        start of chunk[n+1]. We verify by checking that any character
        sequence from the tail of chunk[0] appears somewhere in chunk[1].
        """
        from src.data.chunker import chunk_documents

        # Build a text long enough to guarantee at least 2 chunks
        text = _long_text(3000)
        docs = [_doc(text)]
        chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)

        if len(chunks) < 2:
            pytest.skip("Text too short to produce 2 chunks at this size")

        tail = chunks[0].page_content[-50:]   # last 50 chars of first chunk
        assert tail in chunks[1].page_content, (
            "Expected overlap text from chunk[0] tail to appear in chunk[1]"
        )

    def test_zero_overlap_smaller_than_nonzero_overlap(self):
        """
        The correct invariant is behavioural: with overlap=0 the splitter
        produces MORE chunks than with a large overlap (because no content
        is repeated), or at minimum the same count — never fewer.

        Testing that 'tail not in next chunk' is unreliable because
        RecursiveCharacterTextSplitter may include separator whitespace
        in both chunks even at overlap=0. Instead we assert the
        monotonic relationship between overlap size and chunk count.
        """
        from src.data.chunker import chunk_documents

        paragraphs = [f"Paragraph {i}. " + ("word " * 30) for i in range(10)]
        text = "\n\n".join(paragraphs)
        docs_a = [_doc(text)]
        docs_b = [_doc(text)]

        chunks_no_overlap   = chunk_documents(docs_a, chunk_size=300, chunk_overlap=0)
        chunks_with_overlap = chunk_documents(docs_b, chunk_size=300, chunk_overlap=100)

        if len(chunks_no_overlap) < 2:
            pytest.skip("Text did not produce multiple chunks")

        # Overlap repeats content → more chunks needed to cover the same text
        assert len(chunks_no_overlap) <= len(chunks_with_overlap)


class TestChunkDocumentsOriginalMetadataPreserved:
    def test_source_metadata_carried_through(self):
        """
        Metadata stamped by the loader (file_name, file_type, etc.) must
        survive the splitting process unchanged.
        """
        from src.data.chunker import chunk_documents

        docs = [_doc(
            _long_text(3000),
            file_name="report.pdf",
            file_type=".pdf",
            file_path="/data/raw/report.pdf",
        )]
        chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)

        for chunk in chunks:
            assert chunk.metadata["file_name"] == "report.pdf"
            assert chunk.metadata["file_type"] == ".pdf"
            assert chunk.metadata["file_path"] == "/data/raw/report.pdf"

    def test_multiple_documents_metadata_not_mixed(self):
        """
        Chunks from doc A must not carry metadata from doc B.
        """
        from src.data.chunker import chunk_documents

        doc_a = _doc(_long_text(2000), source="A")
        doc_b = _doc(_long_text(2000), source="B")
        chunks = chunk_documents([doc_a, doc_b], chunk_size=1000, chunk_overlap=100)

        sources = {c.metadata["source"] for c in chunks}
        assert sources == {"A", "B"}

        for chunk in chunks:
            # Each chunk carries exactly one source, not both
            assert chunk.metadata["source"] in {"A", "B"}


class TestChunkDocumentsInputNotMutated:
    def test_original_documents_unchanged(self):
        """
        chunk_documents must not modify the caller's input list or documents.
        """
        from src.data.chunker import chunk_documents

        text = _long_text(3000)
        original_meta = {"file_name": "original.txt"}
        docs = [_doc(text, **original_meta)]
        docs_copy_content = docs[0].page_content

        chunk_documents(docs, chunk_size=1000, chunk_overlap=200)

        assert len(docs) == 1
        assert docs[0].page_content == docs_copy_content
        assert docs[0].metadata.get("file_name") == "original.txt"


# ---------------------------------------------------------------------------
# save_chunks_to_json
# ---------------------------------------------------------------------------

class TestSaveChunksToJsonReturnValue:
    def test_returns_string_path(self, tmp_path):
        from src.data.chunker import save_chunks_to_json

        chunks = _make_chunks(3)
        result = save_chunks_to_json(chunks, str(tmp_path / "out.json"))

        assert isinstance(result, str)

    def test_returned_path_exists(self, tmp_path):
        from src.data.chunker import save_chunks_to_json

        out = tmp_path / "out.json"
        result = save_chunks_to_json(_make_chunks(3), str(out))

        assert Path(result).exists()


class TestSaveChunksToJsonFileContent:
    def test_output_is_valid_json(self, tmp_path):
        from src.data.chunker import save_chunks_to_json

        out = tmp_path / "chunks.json"
        save_chunks_to_json(_make_chunks(3), str(out))

        with open(out, encoding="utf-8") as f:
            data = json.load(f)  # raises on invalid JSON

        assert isinstance(data, list)

    def test_record_count_matches_chunk_count(self, tmp_path):
        from src.data.chunker import save_chunks_to_json

        n = 5
        out = tmp_path / "chunks.json"
        save_chunks_to_json(_make_chunks(n), str(out))

        with open(out, encoding="utf-8") as f:
            data = json.load(f)

        assert len(data) == n

    def test_record_has_required_keys(self, tmp_path):
        from src.data.chunker import save_chunks_to_json

        out = tmp_path / "chunks.json"
        save_chunks_to_json(_make_chunks(2), str(out))

        with open(out, encoding="utf-8") as f:
            data = json.load(f)

        for record in data:
            assert "chunk_index"    in record
            assert "content"        in record
            assert "content_length" in record
            assert "metadata"       in record

    def test_content_length_matches_content(self, tmp_path):
        """content_length must equal len(content) for every record."""
        from src.data.chunker import save_chunks_to_json

        out = tmp_path / "chunks.json"
        save_chunks_to_json(_make_chunks(4, size=120), str(out))

        with open(out, encoding="utf-8") as f:
            data = json.load(f)

        for record in data:
            assert record["content_length"] == len(record["content"])

    def test_chunk_index_in_json_matches_metadata(self, tmp_path):
        from src.data.chunker import save_chunks_to_json

        out = tmp_path / "chunks.json"
        save_chunks_to_json(_make_chunks(4), str(out))

        with open(out, encoding="utf-8") as f:
            data = json.load(f)

        for record in data:
            assert record["chunk_index"] == record["metadata"]["chunk_index"]

    def test_content_field_matches_page_content(self, tmp_path):
        from src.data.chunker import save_chunks_to_json

        chunks = [
            Document(
                page_content="hello world",
                metadata={"chunk_index": 0, "chunk_size": 11},
            )
        ]
        out = tmp_path / "chunks.json"
        save_chunks_to_json(chunks, str(out))

        with open(out, encoding="utf-8") as f:
            data = json.load(f)

        assert data[0]["content"] == "hello world"


class TestSaveChunksToJsonEmptyInput:
    def test_empty_chunks_writes_empty_array(self, tmp_path):
        from src.data.chunker import save_chunks_to_json

        out = tmp_path / "empty.json"
        save_chunks_to_json([], str(out))

        with open(out, encoding="utf-8") as f:
            data = json.load(f)

        assert data == []

    def test_empty_chunks_file_is_created(self, tmp_path):
        from src.data.chunker import save_chunks_to_json

        out = tmp_path / "empty.json"
        save_chunks_to_json([], str(out))

        assert out.exists()


class TestSaveChunksToJsonDefaultPath:
    def test_uses_config_data_processed_dir_when_none(self, tmp_path):
        """
        When output_path=None the file must land inside config.DATA_PROCESSED_DIR.
        We redirect the config value to tmp_path so no real filesystem writes occur
        outside the test sandbox.
        """
        from src.data.chunker import save_chunks_to_json

        with patch("src.data.chunker.config") as mock_config, \
             patch("src.data.chunker.os.makedirs") as mock_makedirs:

            mock_config.DATA_PROCESSED_DIR = str(tmp_path)

            # Provide the real open() so the file is actually written
            result = save_chunks_to_json(_make_chunks(2), output_path=None)

        expected = os.path.join(str(tmp_path), "chunks.json")
        assert result == expected

    def test_creates_output_directory_if_missing(self, tmp_path):
        """
        os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True) must be called
        when output_path is None.
        """
        from src.data.chunker import save_chunks_to_json

        new_dir = str(tmp_path / "new" / "nested")

        with patch("src.data.chunker.config") as mock_config:
            mock_config.DATA_PROCESSED_DIR = new_dir
            save_chunks_to_json(_make_chunks(1), output_path=None)

        assert Path(new_dir).exists()


class TestSaveChunksToJsonUnicodeHandling:
    def test_non_ascii_content_written_correctly(self, tmp_path):
        """
        ensure_ascii=False must be respected — Bahasa Indonesia content
        should be readable without escape sequences.
        """
        from src.data.chunker import save_chunks_to_json

        content = "Ini adalah teks dalam Bahasa Indonesia dengan karakter: é, ñ, ü."
        chunks = [
            Document(
                page_content=content,
                metadata={"chunk_index": 0, "chunk_size": len(content)},
            )
        ]
        out = tmp_path / "unicode.json"
        save_chunks_to_json(chunks, str(out))

        raw = out.read_text(encoding="utf-8")
        # ensure_ascii=False means the actual characters appear, not \uXXXX escapes
        assert "é" in raw
        assert "ñ" in raw