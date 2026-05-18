# tests/test_api_knowledge_base.py
"""
Unit tests for POST /knowledge-base and DELETE /knowledge-base/{doc_id}.

All external calls (loader, chunker, indexer, Chroma) are mocked.
No OPENAI_API_KEY required.
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from src.api.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "doc_id": "doc-uuid-001",
    "file_path": "/data/uploads/report.pdf",
    "file_name": "report.pdf",
}

MOCK_CHUNKS = [
    Document(page_content=f"chunk {i}", metadata={"chunk_index": i})
    for i in range(3)
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _no_conflict_collection():
    """Chroma collection mock that reports doc_id as absent."""
    col = MagicMock()
    col.get.return_value = {"ids": []}
    return col


def _conflict_collection():
    """Chroma collection mock that reports doc_id as already present."""
    col = MagicMock()
    col.get.return_value = {"ids": ["some-vector-id"]}
    return col


def _mock_vector_store(collection_mock):
    vs = MagicMock()
    vs._collection = collection_mock
    return vs


# ── POST /knowledge-base ──────────────────────────────────────────────────────

def test_post_kb_file_not_found():
    with patch("src.api.routes.knowledge_base.Path") as MockPath:
        MockPath.return_value.exists.return_value = False
        MockPath.return_value.suffix.lower.return_value = ".pdf"

        resp = client.post("/knowledge-base", json=VALID_PAYLOAD)

    assert resp.status_code == 400
    assert resp.json()["error_code"] == "FILE_NOT_FOUND"


def test_post_kb_unsupported_file_type():
    with patch("src.api.routes.knowledge_base.Path") as MockPath, \
         patch("src.api.routes.knowledge_base.load_single_document", return_value=[]):

        MockPath.return_value.exists.return_value = True
        MockPath.return_value.suffix.lower.return_value = ".xyz"

        resp = client.post("/knowledge-base", json=VALID_PAYLOAD)

    assert resp.status_code == 400
    assert resp.json()["error_code"] == "UNSUPPORTED_FILE_TYPE"


def test_post_kb_doc_id_conflict():
    mock_vs = _mock_vector_store(_conflict_collection())

    with patch("src.api.routes.knowledge_base.Path") as MockPath, \
         patch("src.api.routes.knowledge_base.load_single_document",
               return_value=[Document(page_content="x", metadata={})]), \
         patch("src.api.routes.knowledge_base.get_vector_store",
               return_value=mock_vs):

        MockPath.return_value.exists.return_value = True
        MockPath.return_value.suffix.lower.return_value = ".pdf"

        resp = client.post("/knowledge-base", json=VALID_PAYLOAD)

    assert resp.status_code == 409
    assert resp.json()["error_code"] == "DOC_ID_CONFLICT"


def test_post_kb_indexing_failure():
    mock_vs = _mock_vector_store(_no_conflict_collection())

    with patch("src.api.routes.knowledge_base.Path") as MockPath, \
         patch("src.api.routes.knowledge_base.load_single_document",
               return_value=[Document(page_content="x", metadata={})]), \
         patch("src.api.routes.knowledge_base.get_vector_store",
               return_value=mock_vs), \
         patch("src.api.routes.knowledge_base.chunk_documents",
               return_value=MOCK_CHUNKS), \
         patch("src.api.routes.knowledge_base.index_documents",
               side_effect=Exception("Chroma exploded")):

        MockPath.return_value.exists.return_value = True
        MockPath.return_value.suffix.lower.return_value = ".pdf"

        resp = client.post("/knowledge-base", json=VALID_PAYLOAD)

    assert resp.status_code == 500
    assert resp.json()["error_code"] == "INDEXING_FAILED"


def test_post_kb_success():
    mock_vs = _mock_vector_store(_no_conflict_collection())

    with patch("src.api.routes.knowledge_base.Path") as MockPath, \
         patch("src.api.routes.knowledge_base.load_single_document",
               return_value=[Document(page_content="content", metadata={})]), \
         patch("src.api.routes.knowledge_base.get_vector_store",
               return_value=mock_vs), \
         patch("src.api.routes.knowledge_base.chunk_documents",
               return_value=MOCK_CHUNKS), \
         patch("src.api.routes.knowledge_base.index_documents",
               return_value=mock_vs):

        MockPath.return_value.exists.return_value = True
        MockPath.return_value.suffix.lower.return_value = ".pdf"

        resp = client.post("/knowledge-base", json=VALID_PAYLOAD)

    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_id"] == VALID_PAYLOAD["doc_id"]
    assert body["file_name"] == VALID_PAYLOAD["file_name"]
    assert body["chunks_indexed"] == len(MOCK_CHUNKS)
    assert body["status"] == "indexed"


# ── DELETE /knowledge-base/{doc_id} ──────────────────────────────────────────

def test_delete_kb_not_found():
    """404 when no vectors exist for the given doc_id."""
    mock_vs = _mock_vector_store(_no_conflict_collection())

    with patch("src.api.routes.knowledge_base.get_vector_store",
               return_value=mock_vs):
        resp = client.delete("/knowledge-base/doc-uuid-missing")

    assert resp.status_code == 404
    assert resp.json()["error_code"] == "DOC_NOT_FOUND"


def test_delete_kb_query_failure():
    """500 when Chroma raises during the initial ID fetch."""
    mock_vs = MagicMock()
    mock_vs._collection.get.side_effect = Exception("Chroma unavailable")

    with patch("src.api.routes.knowledge_base.get_vector_store",
               return_value=mock_vs):
        resp = client.delete("/knowledge-base/doc-uuid-001")

    assert resp.status_code == 500
    assert resp.json()["error_code"] == "DELETION_FAILED"


def test_delete_kb_deletion_failure():
    """500 when Chroma raises during the actual delete call."""
    mock_col = MagicMock()
    mock_col.get.return_value = {"ids": ["vec-1", "vec-2"]}
    mock_col.delete.side_effect = Exception("Write failed")
    mock_vs = _mock_vector_store(mock_col)

    with patch("src.api.routes.knowledge_base.get_vector_store",
               return_value=mock_vs):
        resp = client.delete("/knowledge-base/doc-uuid-001")

    assert resp.status_code == 500
    assert resp.json()["error_code"] == "DELETION_FAILED"


def test_delete_kb_success():
    """200 with correct chunks_deleted count on happy path."""
    mock_col = MagicMock()
    mock_col.get.return_value = {"ids": ["vec-1", "vec-2", "vec-3"]}
    mock_col.delete.return_value = None
    mock_vs = _mock_vector_store(mock_col)

    with patch("src.api.routes.knowledge_base.get_vector_store",
               return_value=mock_vs):
        resp = client.delete("/knowledge-base/doc-uuid-001")

    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_id"] == "doc-uuid-001"
    assert body["chunks_deleted"] == 3
    assert body["status"] == "deleted"

    mock_col.delete.assert_called_once_with(ids=["vec-1", "vec-2", "vec-3"])