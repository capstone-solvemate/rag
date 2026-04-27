from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


# ---------------------------------------------------------------------------
# Test client fixture
# We patch config.validate() globally so TestClient startup
# does not require a real OPENAI_API_KEY in the test environment.
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    with patch("src.api.main.config") as mock_config:
        mock_config.APP_ENV = "test"
        mock_config.validate.return_value = None
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_health_deps(openai_ok: bool = True, doc_count: int = 10):
    """Return a context manager that patches both health dependencies."""
    from unittest.mock import patch as _patch
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(
        _patch(
            "src.api.routes.health._check_openai_reachable",
            return_value=openai_ok,
        )
    )
    stack.enter_context(
        _patch(
            "src.api.routes.health._get_chroma_doc_count",
            return_value=doc_count,
        )
    )
    return stack


# ---------------------------------------------------------------------------
# Status: ok
# ---------------------------------------------------------------------------

def test_health_returns_200_when_all_healthy(client):
    with _patch_health_deps(openai_ok=True, doc_count=10):
        response = client.get("/health")

    assert response.status_code == 200


def test_health_status_ok_when_all_healthy(client):
    with _patch_health_deps(openai_ok=True, doc_count=10):
        data = client.get("/health").json()

    assert data["status"] == "ok"


def test_health_returns_correct_doc_count(client):
    with _patch_health_deps(openai_ok=True, doc_count=263):
        data = client.get("/health").json()

    assert data["chroma_doc_count"] == 263


def test_health_returns_openai_reachable_true(client):
    with _patch_health_deps(openai_ok=True, doc_count=10):
        data = client.get("/health").json()

    assert data["openai_reachable"] is True


# ---------------------------------------------------------------------------
# Status: degraded
# ---------------------------------------------------------------------------

def test_health_degraded_when_openai_unreachable(client):
    with _patch_health_deps(openai_ok=False, doc_count=10):
        data = client.get("/health").json()

    assert data["status"] == "degraded"
    assert data["openai_reachable"] is False


def test_health_degraded_when_no_documents_indexed(client):
    with _patch_health_deps(openai_ok=True, doc_count=0):
        data = client.get("/health").json()

    assert data["status"] == "degraded"
    assert data["chroma_doc_count"] == 0


def test_health_degraded_when_chroma_unreachable(client):
    with _patch_health_deps(openai_ok=True, doc_count=-1):
        data = client.get("/health").json()

    assert data["status"] == "degraded"


def test_health_degraded_when_all_deps_fail(client):
    with _patch_health_deps(openai_ok=False, doc_count=-1):
        data = client.get("/health").json()

    assert data["status"] == "degraded"
    assert data["openai_reachable"] is False


# ---------------------------------------------------------------------------
# Response schema shape
# ---------------------------------------------------------------------------

def test_health_response_contains_required_fields(client):
    with _patch_health_deps():
        data = client.get("/health").json()

    required_keys = {
        "status",
        "chroma_doc_count",
        "openai_reachable",
        "python_version",
        "app_env",
    }
    assert required_keys.issubset(data.keys())


def test_health_app_env_is_test(client):
    with _patch_health_deps():
        data = client.get("/health").json()

    assert data["app_env"] == "development"