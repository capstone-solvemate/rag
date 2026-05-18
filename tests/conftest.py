from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from src.api.main import app


# ---------------------------------------------------------------------------
# Shared document factory
# ---------------------------------------------------------------------------

def make_document(
    content: str = "Sample chunk content.",
    file_name: str = "sample.pdf",
    file_path: str = "/data/raw/sample.pdf",
    file_type: str = ".pdf",
    chunk_index: int = 0,
    page: int | None = 0,
) -> Document:
    """Create a LangChain Document with a full valid metadata contract.

    Use this factory in any test that needs a realistic Document object.
    Keeps metadata shape consistent with the loader/chunker contract
    across the entire test suite.
    """
    metadata = {
        "file_name": file_name,
        "file_path": file_path,
        "file_type": file_type,
        "chunk_index": chunk_index,
        "chunk_size": len(content),
    }
    if page is not None:
        metadata["page"] = page
    return Document(page_content=content, metadata=metadata)


# ---------------------------------------------------------------------------
# Shared API client fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def api_client():
    """FastAPI TestClient with startup validation mocked out.

    Centralizes the client fixture so individual test files don't
    need to repeat the patch. Any test file can request api_client
    instead of defining its own client fixture.
    """
    with patch("src.config.Config.validate", return_value=None):
        with TestClient(app) as c:
            yield c