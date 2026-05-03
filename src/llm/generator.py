# src/llm/generator.py
from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from src.api.schemas.chat import Message
from src.config import config
from src.core.exceptions import GenerationError
from src.llm.prompt_templates import (
    QUERY_REWRITE_TEMPLATE,
    SYSTEM_PROMPT,
    build_user_prompt,
    _TRANSLATE_QUERY_TEMPLATE
)
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


def _history_to_langchain(history: list[Message]) -> list[HumanMessage | AIMessage]:
    """Convert OpenAI-format history to LangChain message objects.

    Excludes the last message (the current query) since it will be
    handled separately in the prompt.

    Args:
        history: Conversation history from ChatRequest.

    Returns:
        List of LangChain HumanMessage / AIMessage objects.
    """
    role_map = {
        "user": HumanMessage,
        "assistant": AIMessage,
    }
    return [role_map[msg.role](content=msg.content) for msg in history[:-1]]

async def generate_answer(query: str, context: str, history: list[Message]) -> str:
    """Generate a grounded answer from a query, context, and conversation history.

    Calls the LLM with:
    - A system prompt enforcing context-only answering
    - Prior conversation history for continuity
    - A user prompt containing the query and retrieved context

    Args:
        query:   The user's original question (not the rewritten one).
        context: Formatted context string from build_context().
        history: Full conversation history from ChatRequest.

    Returns:
        Answer string from the LLM.

    Raises:
        GenerationError: If the LLM call fails or returns an empty response.
        ValueError:      If query or context is empty.
    """
    logger.info(
        f"Generating answer | "
        f"query_length={len(query)} "
        f"context_length={len(context)} "
        f"history_turns={len(history) - 1}"
    )

    user_prompt = build_user_prompt(query, context)
    prior_messages = _history_to_langchain(history)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *prior_messages,
        HumanMessage(content=user_prompt),
    ]

    try:
        model = _get_chat_model()
        response = await model.ainvoke(messages)
        answer = response.content.strip()
    except ValueError:
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

async def rewrite_query(query: str, history: list[Message]) -> str:
    """Rewrite and/or translate a query for vector store retrieval.

    - If history exists: rewrite follow-up into standalone question + translate to English.
    - If no history: translate query to English only.

    Args:
        query:   The user's latest question.
        history: Full conversation history including the current query.

    Returns:
        A standalone English query string suitable for vector store retrieval.

    Raises:
        GenerationError: If the LLM call fails.
    """
    model = _get_chat_model()

    # --- Kasus 1: Ada history — rewrite + translate ---
    if len(history) > 1:
        prior_history = history[:-1]
        history_str = "\n".join(
            f"{msg.role.capitalize()}: {msg.content}" for msg in prior_history
        )
        prompt = QUERY_REWRITE_TEMPLATE.format(
            history=history_str,
            question=query,
        )
        logger.info(f"Rewriting and translating query | history_turns={len(prior_history)}")

    # --- Kasus 2: Tidak ada history — translate saja ---
    else:
        prompt = _TRANSLATE_QUERY_TEMPLATE.format(question=query)
        logger.info("No prior history — translating query only.")

    try:
        response = await model.ainvoke([HumanMessage(content=prompt)])
        rewritten = response.content.strip()
    except Exception as exc:
        logger.error(f"Query rewrite failed: {type(exc).__name__}: {exc}")
        raise GenerationError(
            message="Query rewriting failed. The upstream API call did not succeed.",
            cause=exc,
        ) from exc

    if not rewritten:
        logger.warning("LLM returned empty rewrite — falling back to original query.")
        return query

    logger.info(f"Query rewritten | original='{query}' rewritten='{rewritten}'")
    return rewritten