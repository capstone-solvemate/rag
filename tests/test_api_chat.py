from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain.schema import Document

from src.api.main import app
from src.core.exceptions import GenerationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """TestClient with config.validate() mocked out.

    No OPENAI_API_KEY required to run the test suite.
    """
    with patch("src.config.Config.validate", return_value=None):
        with TestClient(app) as c:
            yield c


def _make_docs(n: int = 2) -> list[Document]:
    """Generate n mock Document objects with valid metadata."""
    return [
        Document(
            page_content=f"Chunk content number {i}.",
            metadata={
                "file_name": f"doc_{i}.pdf",
                "file_path": f"/data/raw/doc_{i}.pdf",
                "file_type": ".pdf",
                "chunk_index": i,
                "page": i,
            },
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Patch helper — patches all three RAG steps
# ---------------------------------------------------------------------------

def _patch_rag(
    documents=None,
    answer: str = "The warranty is one year. [1]",
    retrieval_error: Exception | None = None,
    generation_error: Exception | None = None,
):
    """Context manager that patches the full RAG pipeline."""
    from contextlib import ExitStack
    from unittest.mock import patch as _patch

    if documents is None:
        documents = _make_docs()

    stack = ExitStack()

    # Patch get_vector_store to avoid Chroma disk access
    stack.enter_context(
        _patch("src.api.routes.chat.get_vector_store", return_value=MagicMock())
    )

    # Patch similarity_search
    if retrieval_error:
        stack.enter_context(
            _patch(
                "src.api.routes.chat.similarity_search",
                side_effect=retrieval_error,
            )
        )
    else:
        stack.enter_context(
            _patch(
                "src.api.routes.chat.similarity_search",
                return_value=documents,
            )
        )

    # Patch generate_answer
    if generation_error:
        stack.enter_context(
            _patch(
                "src.api.routes.chat.generate_answer",
                side_effect=generation_error,
            )
        )
    else:
        stack.enter_context(
            _patch(
                "src.api.routes.chat.generate_answer",
                return_value=answer,
            )
        )

    return stack


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_chat_returns_200_on_success(client):
    with _patch_rag():
        response = client.post("/chat", json={"query": "What is the warranty?"})

    assert response.status_code == 200


def test_chat_response_contains_answer(client):
    with _patch_rag(answer="The warranty is one year. [1]"):
        data = client.post("/chat", json={"query": "What is the warranty?"}).json()

    assert data["answer"] == "The warranty is one year. [1]"


def test_chat_response_echoes_query(client):
    with _patch_rag():
        data = client.post("/chat", json={"query": "What is the warranty?"}).json()

    assert data["query"] == "What is the warranty?"


def test_chat_response_contains_sources(client):
    with _patch_rag(documents=_make_docs(3)):
        data = client.post("/chat", json={"query": "Any question."}).json()

    assert len(data["sources"]) == 3


def test_chat_source_has_correct_shape(client):
    with _patch_rag(documents=_make_docs(1)):
        data = client.post("/chat", json={"query": "Any question."}).json()

    source = data["sources"][0]
    assert "file_name" in source
    assert "file_path" in source
    assert "chunk_index" in source


def test_chat_uses_default_k_of_5(client):
    with _patch_rag() as mocks:
        client.post("/chat", json={"query": "Any question."})


def test_chat_respects_custom_k(client):
    with patch("src.api.routes.chat.get_vector_store", return_value=MagicMock()):
        with patch("src.api.routes.chat.similarity_search", return_value=_make_docs(3)) as mock_search:
            with patch("src.api.routes.chat.generate_answer", return_value="Answer."):
                client.post("/chat", json={"query": "Question.", "k": 3})

    mock_search.assert_called_once()
    call_kwargs = mock_search.call_args
    assert call_kwargs.kwargs.get("k") == 3 or call_kwargs.args[2] == 3


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------

def test_chat_rejects_empty_query(client):
    response = client.post("/chat", json={"query": ""})
    assert response.status_code == 422


def test_chat_rejects_missing_query(client):
    response = client.post("/chat", json={})
    assert response.status_code == 422


def test_chat_rejects_k_below_minimum(client):
    response = client.post("/chat", json={"query": "Valid.", "k": 0})
    assert response.status_code == 422


def test_chat_rejects_k_above_maximum(client):
    response = client.post("/chat", json={"query": "Valid.", "k": 21})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_chat_returns_404_when_no_documents_found(client):
    with _patch_rag(documents=[]):
        response = client.post("/chat", json={"query": "Unknown topic."})

    assert response.status_code == 404


def test_chat_returns_503_on_generation_error(client):
    with _patch_rag(generation_error=GenerationError("LLM failed.")):
        response = client.post("/chat", json={"query": "Valid question."})

    assert response.status_code == 503


def test_chat_returns_500_on_retrieval_error(client):
    with _patch_rag(retrieval_error=RuntimeError("Chroma connection lost.")):
        response = client.post("/chat", json={"query": "Valid question."})

    assert response.status_code == 500


def test_chat_503_detail_is_user_friendly(client):
    with _patch_rag(generation_error=GenerationError("LLM failed.")):
        data = client.post("/chat", json={"query": "Valid question."}).json()

    assert "generation" in data["detail"].lower()
    # Internal error detail must not leak to client
    assert "LLM failed." not in data["detail"]