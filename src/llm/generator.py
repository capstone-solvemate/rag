from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from src.config import config
from src.core.exceptions import GenerationError
from src.llm.prompt_templates import SYSTEM_PROMPT, build_user_prompt
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _get_chat_model() -> ChatOpenAI:
    """Instantiate the ChatOpenAI client.

    Kept as a private factory function so it can be patched
    cleanly in tests without touching module-level state.

    Returns:
        Configured ChatOpenAI instance using gpt-4o-mini.
    """
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=config.OPENAI_API_KEY,
    )


def generate_answer(query: str, context: str) -> str:
    """Generate a grounded answer from a query and retrieved context.

    Calls the LLM with a system prompt enforcing context-only answering
    and numbered citation format. Stateless — no conversation history
    is maintained between calls.

    Args:
        query:   The user's question. Must be non-empty.
        context: Formatted context string from build_context().
                 Must be non-empty.

    Returns:
        Answer string from the LLM. May be the explicit refusal phrase
        defined in SYSTEM_PROMPT if context is insufficient.

    Raises:
        GenerationError: If the LLM call fails for any reason (network
                         error, invalid API key, rate limit, etc.).
        ValueError:      If query or context is empty (propagated from
                         build_user_prompt).
    """
    logger.info(f"Generating answer | query_length={len(query)} context_length={len(context)}")

    user_prompt = build_user_prompt(query, context)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    try:
        model = _get_chat_model()
        response = model.invoke(messages)
        answer = response.content.strip()
    except ValueError:
        # Re-raise ValueError from build_user_prompt without wrapping
        raise
    except Exception as exc:
        logger.error(f"LLM generation failed: {type(exc).__name__}: {exc}")
        raise GenerationError(
            message="LLM generation failed. The upstream API call did not succeed.",
            cause=exc,
        ) from exc

    if not answer:
        logger.warning("LLM returned an empty response.")
        raise GenerationError(
            message="LLM returned an empty response.",
            cause=None,
        )

    logger.info(f"Answer generated successfully | answer_length={len(answer)}")
    return answer