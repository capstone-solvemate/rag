"""
Unit tests for POST /chat/image.

All OpenAI and Chroma calls are mocked — no OPENAI_API_KEY needed.

Coverage:
- Happy path: returns ChatResponse with answer and sources
- No documents found: returns 200 with empty-sources answer
- GenerationError from generate_answer_with_image → 503
- ValueError from generate_answer_with_image → 422
- Retrieval failure → 500
- Blank query → 422 (Pydantic validation)
- Blank image_base64 → 422 (Pydantic validation)
- Unsupported media_type → 422 (Pydantic validation)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from src.core.exceptions import GenerationError
from src.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
    "query": "What does the chart show?",
    "image_base64": "aGVsbG8=",  # base64("hello") — not a real image, fine for unit tests
    "media_type": "image/jpeg",
    "k": 3,
}

_MOCK_DOCUMENTS = [
    Document(
        page_content="Revenue increased by 20% in Q3.",
        metadata={
            "file_name": "report.pdf",
            "file_path": "/docs/report.pdf",
            "chunk_index": 0,
        },
    ),
]

_MOCK_SOURCES = [
    {
        "file_name": "report.pdf",
        "file_path": "/docs/report.pdf",
        "chunk_index": 0,
    }
]


def _make_vector_store_mock() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@patch("src.api.routes.image_chat.generate_answer_with_image", new_callable=AsyncMock)
@patch("src.api.routes.image_chat.build_context")
@patch("src.api.routes.image_chat.similarity_search")
@patch("src.api.routes.image_chat.get_vector_store")
def test_chat_with_image_happy_path(
    mock_get_vs,
    mock_similarity_search,
    mock_build_context,
    mock_generate,
):
    mock_get_vs.return_value = _make_vector_store_mock()
    mock_similarity_search.return_value = _MOCK_DOCUMENTS
    mock_build_context.return_value = ("Context string", _MOCK_SOURCES)
    mock_generate.return_value = "Revenue grew 20% per the chart. [1]"

    response = client.post("/chat/image", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == VALID_PAYLOAD["query"]
    assert body["answer"] == "Revenue grew 20% per the chart. [1]"
    assert body["sources"] == _MOCK_SOURCES

    mock_generate.assert_awaited_once_with(
        query=VALID_PAYLOAD["query"],
        context="Context string",
        image_base64=VALID_PAYLOAD["image_base64"],
        media_type=VALID_PAYLOAD["media_type"],
    )


# ---------------------------------------------------------------------------
# No documents found
# ---------------------------------------------------------------------------

@patch("src.api.routes.image_chat.similarity_search")
@patch("src.api.routes.image_chat.get_vector_store")
def test_chat_with_image_no_documents(mock_get_vs, mock_similarity_search):
    mock_get_vs.return_value = _make_vector_store_mock()
    mock_similarity_search.return_value = []

    response = client.post("/chat/image", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == []
    assert "couldn't find" in body["answer"].lower()


# ---------------------------------------------------------------------------
# GenerationError → 503
# ---------------------------------------------------------------------------

@patch("src.api.routes.image_chat.generate_answer_with_image", new_callable=AsyncMock)
@patch("src.api.routes.image_chat.build_context")
@patch("src.api.routes.image_chat.similarity_search")
@patch("src.api.routes.image_chat.get_vector_store")
def test_chat_with_image_generation_error(
    mock_get_vs,
    mock_similarity_search,
    mock_build_context,
    mock_generate,
):
    mock_get_vs.return_value = _make_vector_store_mock()
    mock_similarity_search.return_value = _MOCK_DOCUMENTS
    mock_build_context.return_value = ("Context string", _MOCK_SOURCES)
    mock_generate.side_effect = GenerationError(
        message="OpenAI upstream failure.", cause=None
    )

    response = client.post("/chat/image", json=VALID_PAYLOAD)

    assert response.status_code == 503
    assert "generation failed" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# ValueError from generator → 422
# ---------------------------------------------------------------------------

@patch("src.api.routes.image_chat.generate_answer_with_image", new_callable=AsyncMock)
@patch("src.api.routes.image_chat.build_context")
@patch("src.api.routes.image_chat.similarity_search")
@patch("src.api.routes.image_chat.get_vector_store")
def test_chat_with_image_value_error(
    mock_get_vs,
    mock_similarity_search,
    mock_build_context,
    mock_generate,
):
    mock_get_vs.return_value = _make_vector_store_mock()
    mock_similarity_search.return_value = _MOCK_DOCUMENTS
    mock_build_context.return_value = ("Context string", _MOCK_SOURCES)
    mock_generate.side_effect = ValueError("image_base64 must be a non-empty string.")

    response = client.post("/chat/image", json=VALID_PAYLOAD)

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Retrieval failure → 500
# ---------------------------------------------------------------------------

@patch("src.api.routes.image_chat.similarity_search")
@patch("src.api.routes.image_chat.get_vector_store")
def test_chat_with_image_retrieval_failure(mock_get_vs, mock_similarity_search):
    mock_get_vs.return_value = _make_vector_store_mock()
    mock_similarity_search.side_effect = RuntimeError("Chroma connection dropped.")

    response = client.post("/chat/image", json=VALID_PAYLOAD)

    assert response.status_code == 500
    assert "retrieval failed" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Pydantic validation — blank query
# ---------------------------------------------------------------------------

def test_chat_with_image_blank_query():
    payload = {**VALID_PAYLOAD, "query": "   "}
    response = client.post("/chat/image", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Pydantic validation — blank image_base64
# ---------------------------------------------------------------------------

def test_chat_with_image_blank_image_base64():
    payload = {**VALID_PAYLOAD, "image_base64": "   "}
    response = client.post("/chat/image", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Pydantic validation — unsupported media_type
# ---------------------------------------------------------------------------

def test_chat_with_image_invalid_media_type():
    payload = {**VALID_PAYLOAD, "media_type": "image/bmp"}
    response = client.post("/chat/image", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Pydantic validation — k out of range
# ---------------------------------------------------------------------------

def test_chat_with_image_k_out_of_range():
    payload = {**VALID_PAYLOAD, "k": 0}
    response = client.post("/chat/image", json=payload)
    assert response.status_code == 422