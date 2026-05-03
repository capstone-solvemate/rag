from __future__ import annotations

from src.api.schemas.health import HealthResponse
from src.embedding.embedder import embed_single_text
from src.embedding.indexer import get_collection_count, get_vector_store
from src.utils.logger import get_logger

import platform

logger = get_logger(__name__)


def _check_openai_reachable() -> bool:
    """Probe OpenAI reachability by embedding a short sentinel string.

    Reuses the existing embed_single_text path rather than introducing
    a new API call pattern. A successful embedding confirms both network
    reachability and API key validity.

    Returns:
        True if the call succeeded, False otherwise.
    """
    try:
        embed_single_text("health check")
        return True
    except Exception as exc:
        logger.warning(f"OpenAI reachability check failed: {type(exc).__name__}: {exc}")
        return False


def _get_chroma_doc_count() -> int:
    """Return the current document count from the Chroma collection.

    Returns:
        Document count, or -1 if Chroma is unreachable.
    """
    try:
        vector_store = get_vector_store()
        return get_collection_count(vector_store)
    except Exception as exc:
        logger.warning(f"Chroma count check failed: {type(exc).__name__}: {exc}")
        return -1


async def health_check() -> HealthResponse:
    """Run all dependency checks and return service health status.

    Checks performed:
    - OpenAI API reachability (live embed call)
    - Chroma vector store document count

    Overall status is 'ok' only when all checks pass and at least
    one document is indexed. Otherwise status is 'degraded'.

    Returns:
        HealthResponse with status, counts, and runtime metadata.
    """
    from src.config import config

    logger.info("Health check requested.")

    openai_reachable = _check_openai_reachable()
    chroma_doc_count = _get_chroma_doc_count()

    # Degraded if OpenAI is unreachable or no documents are indexed
    is_healthy = openai_reachable and chroma_doc_count > 0

    status = "ok" if is_healthy else "degraded"

    if not is_healthy:
        logger.warning(
            f"Service degraded | openai_reachable={openai_reachable} "
            f"chroma_doc_count={chroma_doc_count}"
        )
    else:
        logger.info(
            f"Service healthy | chroma_doc_count={chroma_doc_count}"
        )

    return HealthResponse(
        status=status,
        chroma_doc_count=chroma_doc_count,
        openai_reachable=openai_reachable,
        python_version=platform.python_version(),
        app_env=config.APP_ENV,
    )