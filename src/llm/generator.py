# src/llm/generator.py
from __future__ import annotations

import openai
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.api.schemas.chat import Message
from src.api.schemas.detection import DetectionResult
from src.config import config
from src.core.exceptions import GenerationError
from src.llm.prompt_templates import (
    QUERY_REWRITE_TEMPLATE,
    SYSTEM_PROMPT,
    build_user_prompt,
    build_vision_messages,
    _TRANSLATE_QUERY_TEMPLATE,
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


def _get_openai_client() -> openai.AsyncOpenAI:
    """Instantiate a raw AsyncOpenAI client for vision calls.

    LangChain's multimodal support requires extra wrapping for image_url
    content blocks. Using the raw client here keeps vision messages as plain
    dicts (built by build_vision_messages) without any LangChain adaptation
    layer.

    Returns:
        Configured AsyncOpenAI instance.
    """
    return openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)


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


async def generate_answer_with_image(
    query: str,
    context: str,
    image_base64: str,
    media_type: str,
) -> str:
    """Generate a grounded answer using both retrieved context and an image.

    Uses the raw AsyncOpenAI client (not LangChain) to send a vision-capable
    messages array built by build_vision_messages(). Retrieval still runs on
    the text query — the image is additional visual context for the LLM only.

    The model is always gpt-4o-mini, which supports vision input.

    Args:
        query:        The user's question.
        context:      Formatted context string from build_context().
        image_base64: Base64-encoded image without the data URI prefix.
        media_type:   MIME type, e.g. "image/jpeg".

    Returns:
        Answer string from the LLM.

    Raises:
        GenerationError: If the LLM call fails or returns an empty response.
        ValueError:      If query, context, or image_base64 is empty.
    """
    logger.info(
        f"Generating vision answer | "
        f"query_length={len(query)} "
        f"context_length={len(context)} "
        f"media_type={media_type}"
    )

    # Raises ValueError on empty inputs — let it propagate to the route.
    messages = build_vision_messages(
        query=query,
        context=context,
        image_base64=image_base64,
        media_type=media_type,
    )

    try:
        client = _get_openai_client()
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,  # type: ignore[arg-type]
            max_tokens=1024,
            temperature=0,
        )
        answer = response.choices[0].message.content or ""
        answer = answer.strip()
    except ValueError:
        raise
    except Exception as exc:
        logger.error(f"Vision LLM generation failed: {type(exc).__name__}: {exc}")
        raise GenerationError(
            message="Vision LLM generation failed. The upstream API call did not succeed.",
            cause=exc,
        ) from exc

    if not answer:
        logger.warning("Vision LLM returned an empty response.")
        raise GenerationError(
            message="Vision LLM returned an empty response.",
            cause=None,
        )

    logger.info(f"Vision answer generated successfully | answer_length={len(answer)}")
    return answer

async def analyze_image_for_defects(
    image_base64: str,
    media_type: str,
    detection_mode: str,
) -> DetectionResult:
    """Analyze an image for printer defects or print quality issues.

    Step 1 of the Phase 5 detection pipeline. Sends the image to gpt-4o
    with a structured JSON instruction prompt and parses the response into
    a DetectionResult.

    Retrieval and answer generation are NOT performed here — this function
    returns the structured detection output only. The caller orchestrates
    the full pipeline.

    Args:
        image_base64:   Base64-encoded image data without the data URI prefix.
        media_type:     MIME type, e.g. "image/jpeg".
        detection_mode: One of "quality", "defect", "both".

    Returns:
        Parsed DetectionResult from the LLM's JSON response.

    Raises:
        GenerationError: If the LLM call fails, returns empty content,
                         or returns malformed JSON.
        ValueError:      If image_base64 is empty or detection_mode is unrecognized
                         (raised by build_detection_analysis_prompt).
    """
    logger.info(
        f"Running detection analysis | "
        f"detection_mode={detection_mode} "
        f"media_type={media_type}"
    )

    messages = build_detection_analysis_prompt(
        image_base64=image_base64,
        media_type=media_type,
        detection_mode=detection_mode,
    )

    try:
        client = _get_openai_client()
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,  # type: ignore[arg-type]
            max_tokens=512,
            temperature=0,
        )
        raw = response.choices[0].message.content or ""
        raw = raw.strip()
    except ValueError:
        raise
    except Exception as exc:
        logger.error(f"Detection LLM call failed: {type(exc).__name__}: {exc}")
        raise GenerationError(
            message="Detection LLM call failed. The upstream API call did not succeed.",
            cause=exc,
        ) from exc

    if not raw:
        logger.warning("Detection LLM returned empty content.")
        raise GenerationError(
            message="Detection LLM returned an empty response.",
            cause=None,
        )

    try:
        result = DetectionResult.model_validate_json(raw)
    except Exception as exc:
        logger.error(f"Detection JSON parse failed | raw='{raw[:200]}' error={exc}")
        raise GenerationError(
            message="Detection LLM returned malformed JSON. Could not parse DetectionResult.",
            cause=exc,
        ) from exc

    logger.info(
        f"Detection analysis complete | "
        f"issues={len(result.detected_issues)} "
        f"severity={result.severity} "
        f"confidence={result.confidence:.2f}"
    )
    return result

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