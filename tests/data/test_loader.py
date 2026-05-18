"""
test(loader): unit tests for load_single_document and load_documents_from_directory

Phase 3 — Commit 1

Coverage:
- Supported extensions happy paths (.txt, .pdf, .docx)
- Unsupported extension returns []
- Missing file raises (propagated from loader)
- Metadata keys stamped correctly (file_name, file_path, file_type)
- Directory scan picks up nested files, skips unsupported extensions
- Directory defaults to config.DATA_RAW_DIR when None is passed

All external I/O (LangChain loaders, os.walk) is mocked — no real files required
except where tmp_path is used to test the actual os.walk traversal logic.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_docs(*contents: str) -> list[Document]:
    """Return bare Document objects with no metadata (loaders produce these)."""
    return [Document(page_content=c, metadata={}) for c in contents]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_text_loader():
    """Patch TextLoader so no real file is touched."""
    with patch("src.data.loader.TextLoader") as MockLoader:
        instance = MagicMock()
        instance.load.return_value = _make_docs("hello txt")
        MockLoader.return_value = instance
        yield MockLoader, instance


@pytest.fixture()
def mock_pdf_loader():
    with patch("src.data.loader.PyPDFLoader") as MockLoader:
        instance = MagicMock()
        instance.load.return_value = _make_docs("page 1 content", "page 2 content")
        MockLoader.return_value = instance
        yield MockLoader, instance


@pytest.fixture()
def mock_docx_loader():
    with patch("src.data.loader.Docx2txtLoader") as MockLoader:
        instance = MagicMock()
        instance.load.return_value = _make_docs("docx content")
        MockLoader.return_value = instance
        yield MockLoader, instance


# ---------------------------------------------------------------------------
# load_single_document — supported extensions
# ---------------------------------------------------------------------------

class TestLoadSingleDocumentTxt:
    def test_returns_documents(self, mock_text_loader):
        from src.data.loader import load_single_document

        docs = load_single_document("/fake/dir/readme.txt")

        assert len(docs) == 1
        assert docs[0].page_content == "hello txt"

    def test_uses_text_loader_with_utf8(self, mock_text_loader):
        from src.data.loader import load_single_document

        MockLoader, _ = mock_text_loader
        load_single_document("/fake/dir/readme.txt")

        MockLoader.assert_called_once_with("/fake/dir/readme.txt", encoding="utf-8")

    def test_metadata_file_name(self, mock_text_loader):
        from src.data.loader import load_single_document

        docs = load_single_document("/fake/dir/readme.txt")

        assert docs[0].metadata["file_name"] == "readme.txt"

    def test_metadata_file_path(self, mock_text_loader):
        from src.data.loader import load_single_document

        docs = load_single_document("/fake/dir/readme.txt")

        assert docs[0].metadata["file_path"] == "/fake/dir/readme.txt"

    def test_metadata_file_type(self, mock_text_loader):
        from src.data.loader import load_single_document

        docs = load_single_document("/fake/dir/readme.txt")

        assert docs[0].metadata["file_type"] == ".txt"


class TestLoadSingleDocumentPdf:
    def test_returns_all_pages_as_documents(self, mock_pdf_loader):
        from src.data.loader import load_single_document

        docs = load_single_document("/fake/dir/report.pdf")

        assert len(docs) == 2

    def test_uses_pypdf_loader(self, mock_pdf_loader):
        from src.data.loader import load_single_document

        MockLoader, _ = mock_pdf_loader
        load_single_document("/fake/dir/report.pdf")

        MockLoader.assert_called_once_with("/fake/dir/report.pdf")

    def test_metadata_stamped_on_all_pages(self, mock_pdf_loader):
        from src.data.loader import load_single_document

        docs = load_single_document("/fake/dir/report.pdf")

        for doc in docs:
            assert doc.metadata["file_name"] == "report.pdf"
            assert doc.metadata["file_type"] == ".pdf"

    def test_extension_case_insensitive(self, mock_pdf_loader):
        """Uppercase .PDF should still route to PyPDFLoader."""
        from src.data.loader import load_single_document

        MockLoader, _ = mock_pdf_loader
        # path.suffix.lower() normalises it — loader should still be called
        docs = load_single_document("/fake/dir/REPORT.PDF")

        MockLoader.assert_called_once()
        assert docs[0].metadata["file_type"] == ".pdf"


class TestLoadSingleDocumentDocx:
    def test_returns_documents(self, mock_docx_loader):
        from src.data.loader import load_single_document

        docs = load_single_document("/fake/dir/contract.docx")

        assert len(docs) == 1
        assert docs[0].page_content == "docx content"

    def test_uses_docx_loader(self, mock_docx_loader):
        from src.data.loader import load_single_document

        MockLoader, _ = mock_docx_loader
        load_single_document("/fake/dir/contract.docx")

        MockLoader.assert_called_once_with("/fake/dir/contract.docx")

    def test_metadata_file_type(self, mock_docx_loader):
        from src.data.loader import load_single_document

        docs = load_single_document("/fake/dir/contract.docx")

        assert docs[0].metadata["file_type"] == ".docx"


# ---------------------------------------------------------------------------
# load_single_document — unsupported extension
# ---------------------------------------------------------------------------

class TestLoadSingleDocumentUnsupported:
    def test_returns_empty_list_for_csv(self):
        from src.data.loader import load_single_document

        result = load_single_document("/fake/dir/data.csv")

        assert result == []

    def test_returns_empty_list_for_jpg(self):
        from src.data.loader import load_single_document

        result = load_single_document("/fake/dir/photo.jpg")

        assert result == []

    def test_returns_empty_list_for_no_extension(self):
        from src.data.loader import load_single_document

        result = load_single_document("/fake/dir/Makefile")

        assert result == []

    def test_no_loader_instantiated_for_unsupported(self):
        """Confirm we never try to call any LangChain loader for bad extensions."""
        from src.data.loader import load_single_document

        with patch("src.data.loader.TextLoader") as tl, \
             patch("src.data.loader.PyPDFLoader") as pl, \
             patch("src.data.loader.Docx2txtLoader") as dl:

            load_single_document("/fake/dir/data.csv")

            tl.assert_not_called()
            pl.assert_not_called()
            dl.assert_not_called()


# ---------------------------------------------------------------------------
# load_single_document — missing / unreadable file
# ---------------------------------------------------------------------------

class TestLoadSingleDocumentMissingFile:
    def test_propagates_file_not_found_for_txt(self):
        """
        We don't swallow errors — if the loader raises, so do we.
        TextLoader raises FileNotFoundError for missing files.
        """
        from src.data.loader import load_single_document

        with patch("src.data.loader.TextLoader") as MockLoader:
            MockLoader.return_value.load.side_effect = FileNotFoundError("no such file")

            with pytest.raises(FileNotFoundError):
                load_single_document("/nonexistent/path/readme.txt")

    def test_propagates_exception_for_pdf(self):
        from src.data.loader import load_single_document

        with patch("src.data.loader.PyPDFLoader") as MockLoader:
            MockLoader.return_value.load.side_effect = Exception("PDF corrupt")

            with pytest.raises(Exception, match="PDF corrupt"):
                load_single_document("/fake/dir/broken.pdf")


# ---------------------------------------------------------------------------
# load_documents_from_directory
# ---------------------------------------------------------------------------

class TestLoadDocumentsFromDirectory:

    def _patch_walk(self, structure: dict) -> list:
        """
        structure: { root: (dirs, files) }
        Returns os.walk-compatible list of (root, dirs, files) tuples.
        """
        return [(root, data[0], data[1]) for root, data in structure.items()]

    def test_loads_all_supported_files(self, tmp_path):
        """
        Use tmp_path + real (but tiny) files to exercise the actual os.walk path.
        Mocks only the individual loaders, not the filesystem traversal.
        """
        from src.data.loader import load_documents_from_directory

        # Create real files so os.walk finds them
        (tmp_path / "doc.txt").write_text("text content", encoding="utf-8")
        (tmp_path / "report.pdf").write_bytes(b"%PDF-1.4 fake")
        (tmp_path / "contract.docx").write_bytes(b"PK fake docx")

        with patch("src.data.loader.TextLoader") as tl, \
             patch("src.data.loader.PyPDFLoader") as pl, \
             patch("src.data.loader.Docx2txtLoader") as dl:

            tl.return_value.load.return_value = _make_docs("txt")
            pl.return_value.load.return_value = _make_docs("pdf")
            dl.return_value.load.return_value = _make_docs("docx")

            docs = load_documents_from_directory(str(tmp_path))

        assert len(docs) == 3

    def test_skips_unsupported_extensions(self, tmp_path):
        """Files like .csv, .jpg, .md must be silently skipped."""
        from src.data.loader import load_documents_from_directory

        (tmp_path / "data.csv").write_text("a,b,c")
        (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8 fake jpg")
        (tmp_path / "notes.md").write_text("# heading")

        docs = load_documents_from_directory(str(tmp_path))

        assert docs == []

    def test_recurses_into_subdirectories(self, tmp_path):
        """os.walk should reach nested folders."""
        from src.data.loader import load_documents_from_directory

        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("nested content", encoding="utf-8")

        with patch("src.data.loader.TextLoader") as tl:
            tl.return_value.load.return_value = _make_docs("nested content")
            docs = load_documents_from_directory(str(tmp_path))

        assert len(docs) == 1
        assert docs[0].page_content == "nested content"

    def test_empty_directory_returns_empty_list(self, tmp_path):
        from src.data.loader import load_documents_from_directory

        docs = load_documents_from_directory(str(tmp_path))

        assert docs == []

    def test_mixed_directory_only_loads_supported(self, tmp_path):
        """Mix of supported and unsupported — only supported come back."""
        from src.data.loader import load_documents_from_directory

        (tmp_path / "keep.txt").write_text("keep", encoding="utf-8")
        (tmp_path / "skip.csv").write_text("skip")

        with patch("src.data.loader.TextLoader") as tl:
            tl.return_value.load.return_value = _make_docs("keep")
            docs = load_documents_from_directory(str(tmp_path))

        assert len(docs) == 1

    def test_defaults_to_config_data_raw_dir_when_none(self):
        """
        Passing directory=None must use config.DATA_RAW_DIR, not crash.
        """
        from src.data.loader import load_documents_from_directory

        with patch("src.data.loader.config") as mock_config, \
             patch("src.data.loader.os.walk") as mock_walk:

            mock_config.DATA_RAW_DIR = "/mocked/raw"
            mock_walk.return_value = []  # no files — we only care about the path used

            load_documents_from_directory(None)

            mock_walk.assert_called_once_with("/mocked/raw")

    def test_metadata_stamped_correctly_in_directory_load(self, tmp_path):
        """
        Metadata injected by load_single_document must survive aggregation
        through load_documents_from_directory.
        """
        from src.data.loader import load_documents_from_directory

        (tmp_path / "sample.txt").write_text("hello", encoding="utf-8")

        with patch("src.data.loader.TextLoader") as tl:
            tl.return_value.load.return_value = _make_docs("hello")
            docs = load_documents_from_directory(str(tmp_path))

        assert docs[0].metadata["file_name"] == "sample.txt"
        assert docs[0].metadata["file_type"] == ".txt"
        assert "file_path" in docs[0].metadata

    def test_aggregates_multipage_pdf_correctly(self, tmp_path):
        """PDF with 3 pages → 3 Documents in the final list."""
        from src.data.loader import load_documents_from_directory

        (tmp_path / "big.pdf").write_bytes(b"%PDF-1.4 fake")

        with patch("src.data.loader.PyPDFLoader") as pl:
            pl.return_value.load.return_value = _make_docs("p1", "p2", "p3")
            docs = load_documents_from_directory(str(tmp_path))

        assert len(docs) == 3