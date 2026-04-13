from src.api.schemas.response import ChatResponse
from src.config import config
from src.utils.logger import get_logger

logger = get_logger("src.pipeline")


def _build_prompt(query: str, context: str) -> str:
    """Compose the final prompt from the system instruction, context, and the original user query."""
    ...


def _call_llm(prompt: str) -> str:
    """Send the prompt to OpenAI and return the response as a string."""
    ...


def run_rag_pipeline(query: str, k: int = config.RETRIEVAL_K) -> ChatResponse:
    """Run the full pipeline: rewrite → retrieve → build context → generate answer."""
    ...