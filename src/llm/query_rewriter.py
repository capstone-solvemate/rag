from src.config import config
from src.utils.logger import get_logger

logger = get_logger("src.llm.query_rewriter")


def _build_rewrite_prompt(query: str) -> str:
    """Compose an instruction prompt to rewrite and translate the query to English."""
    ...


def rewrite_query(query: str) -> str:
    """Clean up, translate to English, and return the reformatted query."""
    ...