from __future__ import annotations

"""End-to-end smoke tests for the full RAG pipeline.

These tests validate that all layers compose correctly:
    ChatRequest
        → similarity_search     (retrieval)
        → build_context         (formatting)
        → generate_answer       (generation)
        → ChatResponse          (API response)

All external I/O is mocked. No OPENAI_API_KEY required.
No Chroma disk access. No network calls.

These tests are deliberately coarse-grained — they test the
seams between layers, not the internal logic of each layer
(which is covered by the unit tests in test_context_builder.py,
test_prompt_templates.py, and test_generator.py).
"""

from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import GenerationError
from tests.conftest import make_document

REFUSAL_PHRASE = "I could not find a relevant answer in the available documents."


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_docs():
    return [
        make_document(
            content="The warranty period for the L3210 Series is one year.",
            file_name="manual L3210.pdf",
            chunk_index=5,
            page=11,
        ),
        make_document(
            content="Contact Epson support for warranty claims.",
            file_name="manual L3210.pdf",
            chunk_index=6,
            page=12,
        ),
    ]


@pytest.fixture
def mock_vector_store():
    return MagicMock()


# ---------------------------------------------------------------------------
# Full pipeline — happy path
# ---------------------------------------------------------------------------

def test_full_pipeline_returns_200(api_client, sample_docs, mock_vector_store):
    """Full path from HTTP request to HTTP response returns 200."""
    with patch("src.api.routes.chat.get_vector_store", return_value=mock_vector_store), \
         patch("src.api.routes.chat.similarity_search", return_value=sample_docs), \
         patch("src.api.routes.chat.generate_answer", return_value="Warranty is one year. [1]"):

        response = api_client.post("/chat", json={"history": [{"role": "user", "content": "What is the warranty?"}], "k": 5})

    assert response.status_code == 200


def test_full_pipeline_response_shape(api_client, sample_docs, mock_vector_store):
    """Response contains all required top-level fields."""
    with patch("src.api.routes.chat.get_vector_store", return_value=mock_vector_store), \
         patch("src.api.routes.chat.similarity_search", return_value=sample_docs), \
         patch("src.api.routes.chat.generate_answer", return_value="Warranty is one year. [1]"):

        data = api_client.post(
            "/chat", json={"history": [{"role": "user", "content": "What is the warranty?"}], "k": 5}
        ).json()

    assert "query" in data
    assert "answer" in data
    assert "sources" in data


def test_full_pipeline_sources_match_retrieved_docs(api_client, sample_docs, mock_vector_store):
    """Number of sources in response matches number of retrieved documents."""
    with patch("src.api.routes.chat.get_vector_store", return_value=mock_vector_store), \
         patch("src.api.routes.chat.similarity_search", return_value=sample_docs), \
         patch("src.api.routes.chat.generate_answer", return_value="Answer. [1][2]"):

        data = api_client.post(
            "/chat", json={"history": [{"role": "user", "content": "What is the warranty?"}], "k": 5}
        ).json()

    assert len(data["sources"]) == len(sample_docs)


def test_full_pipeline_source_file_name_is_correct(api_client, sample_docs, mock_vector_store):
    """Source file names in response match the input document metadata."""
    with patch("src.api.routes.chat.get_vector_store", return_value=mock_vector_store), \
         patch("src.api.routes.chat.similarity_search", return_value=sample_docs), \
         patch("src.api.routes.chat.generate_answer", return_value="Answer. [1]"):

        data = api_client.post(
            "/chat", json={"history": [{"role": "user", "content": "What is the warranty?"}], "k": 5}
        ).json()

    source_names = [s["file_name"] for s in data["sources"]]
    assert all(name == "manual L3210.pdf" for name in source_names)


def test_full_pipeline_query_is_echoed(api_client, sample_docs, mock_vector_store):
    """Response echoes the original query back to the client."""
    query = "What is the warranty period for the L3210?"
    with patch("src.api.routes.chat.get_vector_store", return_value=mock_vector_store), \
         patch("src.api.routes.chat.similarity_search", return_value=sample_docs), \
         patch("src.api.routes.chat.generate_answer", return_value="One year. [1]"):

        data = api_client.post("/chat", json={"history": [{"role": "user", "content": "What is the warranty period for the L3210?"}], "k": 5}).json()

    assert data["query"] == query


def test_full_pipeline_refusal_phrase_passes_through(api_client, sample_docs, mock_vector_store):
    """Refusal phrase from LLM is returned as-is without modification."""
    with patch("src.api.routes.chat.get_vector_store", return_value=mock_vector_store), \
         patch("src.api.routes.chat.similarity_search", return_value=sample_docs), \
         patch("src.api.routes.chat.generate_answer", return_value=REFUSAL_PHRASE):

        data = api_client.post(
            "/chat", json={"history": [{"role": "user", "content": "Who is Viktor Frankl?"}], "k": 5}
        ).json()

    assert data["answer"] == REFUSAL_PHRASE
    assert data["sources"] is not None  # Sources still returned even on refusal


# ---------------------------------------------------------------------------
# Pipeline seam — retrieval to context boundary
# ---------------------------------------------------------------------------

def test_pipeline_handles_single_document_correctly(api_client, mock_vector_store):
    """Pipeline works correctly when retrieval returns exactly one document."""
    single_doc = [make_document(content="Single relevant chunk.", chunk_index=0)]

    with patch("src.api.routes.chat.get_vector_store", return_value=mock_vector_store), \
         patch("src.api.routes.chat.similarity_search", return_value=single_doc), \
         patch("src.api.routes.chat.generate_answer", return_value="Answer based on one source. [1]"):

        data = api_client.post("/chat", json={"history": [{"role": "user", "content": "What is the warranty?"}], "k": 5}).json()

    assert data["status_code"] if "status_code" in data else True
    assert len(data["sources"]) == 1


def test_pipeline_page_number_present_in_context_passed_to_generator(
    api_client, mock_vector_store
):
    """Page metadata from documents reaches the generator as part of context."""
    doc_with_page = [make_document(content="Content.", chunk_index=2, page=4)]

    captured_context = {}

    def capture_generate(query, context, history):
        captured_context["context"] = context
        return "Answer. [1]"

    with patch("src.api.routes.chat.get_vector_store", return_value=mock_vector_store), \
         patch("src.api.routes.chat.similarity_search", return_value=doc_with_page), \
         patch("src.api.routes.chat.generate_answer", side_effect=capture_generate):

        api_client.post("/chat", json={"history": [{"role": "user", "content": "What is the warranty?"}], "k": 5})

    # Page 4 (0-indexed) should appear as Page: 5 in the context string
    assert "Page: 5" in captured_context["context"]


# ---------------------------------------------------------------------------
# Pipeline seam — error propagation
# ---------------------------------------------------------------------------

def test_pipeline_empty_retrieval_returns_200_with_fallback(api_client, mock_vector_store):
    with patch("src.api.routes.chat.get_vector_store", return_value=mock_vector_store), \
         patch("src.api.routes.chat.similarity_search", return_value=[]):

        response = api_client.post("/chat", json={
            "history": [{"role": "user", "content": "What is the warranty?"}],
            "k": 5
        })

    assert response.status_code == 200
    assert response.json()["answer"].startswith("I'm sorry")
    assert response.json()["sources"] == []


def test_pipeline_503_does_not_leak_internal_error(api_client, sample_docs, mock_vector_store):
    """503 response body does not contain internal GenerationError message."""
    with patch("src.api.routes.chat.get_vector_store", return_value=mock_vector_store), \
         patch("src.api.routes.chat.similarity_search", return_value=sample_docs), \
         patch(
             "src.api.routes.chat.generate_answer",
             side_effect=GenerationError("Secret internal reason."),
         ):

        data = api_client.post("/chat", json={"history": [{"role": "user", "content": "What is the warranty?"}], "k": 5}).json()

    assert "Secret internal reason." not in str(data)


def test_pipeline_500_on_retrieval_failure_does_not_reach_generator(
    api_client, mock_vector_store
):
    """Retrieval failure stops the pipeline before context building or generation."""
    context_mock = MagicMock()
    generate_mock = MagicMock()

    with patch("src.api.routes.chat.get_vector_store", return_value=mock_vector_store), \
         patch("src.api.routes.chat.similarity_search", side_effect=RuntimeError("Chroma down.")), \
         patch("src.api.routes.chat.build_context", context_mock), \
         patch("src.api.routes.chat.generate_answer", generate_mock):

        response = api_client.post("/chat", json={"history": [{"role": "user", "content": "What is the warranty?"}], "k": 5})

    assert response.status_code == 500
    context_mock.assert_not_called()
    generate_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Health + chat coexistence
# ---------------------------------------------------------------------------

def test_health_and_chat_endpoints_coexist(api_client, sample_docs, mock_vector_store):
    """Both endpoints are reachable in the same app instance."""
    with patch("src.api.routes.health._check_openai_reachable", return_value=True), \
         patch("src.api.routes.health._get_chroma_doc_count", return_value=263):

        health_response = api_client.get("/health")

    with patch("src.api.routes.chat.get_vector_store", return_value=mock_vector_store), \
         patch("src.api.routes.chat.similarity_search", return_value=sample_docs), \
         patch("src.api.routes.chat.generate_answer", return_value="Answer. [1]"):

        chat_response = api_client.post("/chat", json={"history": [{"role": "user", "content": "What is the warranty?"}], "k": 5})

    assert health_response.status_code == 200
    assert chat_response.status_code == 200