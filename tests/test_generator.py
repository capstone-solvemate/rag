from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from src.core.exceptions import GenerationError
from src.llm.generator import generate_answer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_QUERY = "What is the warranty period for the L3210?"
VALID_CONTEXT = (
    "[1] Source: manual L3210.pdf | Type: .pdf | Chunk: 5 | Page: 12\n"
    "The warranty period for the L3210 Series is one year from the date of purchase."
)
VALID_HISTORY = {"history": [{"role": "user", "content": "What is the warranty?"}], "k": 5}


def _make_mock_model(response_text: str) -> MagicMock:
    """Return a mock ChatOpenAI that returns response_text on invoke()."""
    mock_model = MagicMock()
    mock_model.invoke.return_value = AIMessage(content=response_text)
    return mock_model


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

async def test_returns_string_on_success():
    with patch("src.llm.generator._get_chat_model") as mock_factory:
        mock_factory.return_value = _make_mock_model("The warranty is one year. [1]")
        result = await generate_answer(VALID_QUERY, VALID_CONTEXT, VALID_HISTORY)

    assert isinstance(result, str)
    assert len(result) > 0


async def test_returns_stripped_answer():
    with patch("src.llm.generator._get_chat_model") as mock_factory:
        mock_factory.return_value = _make_mock_model("  Answer with whitespace.  ")
        result = await generate_answer(VALID_QUERY, VALID_CONTEXT, VALID_HISTORY)

    assert result == "Answer with whitespace."


async def test_model_receives_system_and_human_messages():
    from langchain_core.messages import HumanMessage, SystemMessage
    from src.llm.prompt_templates import SYSTEM_PROMPT

    with patch("src.llm.generator._get_chat_model") as mock_factory:
        mock_model = _make_mock_model("An answer.")
        mock_factory.return_value = mock_model
        await generate_answer(VALID_QUERY, VALID_CONTEXT, VALID_HISTORY)

    call_args = mock_model.invoke.call_args[0][0]
    assert isinstance(call_args[0], SystemMessage)
    assert isinstance(call_args[1], HumanMessage)
    assert call_args[0].content == SYSTEM_PROMPT


async def test_refusal_phrase_is_passed_through():
    """If the LLM returns the refusal phrase it must be returned as-is."""
    refusal = "I could not find a relevant answer in the available documents."
    with patch("src.llm.generator._get_chat_model") as mock_factory:
        mock_factory.return_value = _make_mock_model(refusal)
        result = await generate_answer(VALID_QUERY, VALID_CONTEXT, VALID_HISTORY)

    assert result == refusal


# ---------------------------------------------------------------------------
# GenerationError cases
# ---------------------------------------------------------------------------

async def test_api_exception_raises_generation_error():
    with patch("src.llm.generator._get_chat_model") as mock_factory:
        mock_model = MagicMock()
        mock_model.invoke.side_effect = RuntimeError("connection timeout")
        mock_factory.return_value = mock_model

        with pytest.raises(GenerationError) as exc_info:
           await generate_answer(VALID_QUERY, VALID_CONTEXT, VALID_HISTORY)

    assert "LLM generation failed" in str(exc_info.value)
    assert isinstance(exc_info.value.cause, RuntimeError)


async def test_empty_llm_response_raises_generation_error():
    with patch("src.llm.generator._get_chat_model") as mock_factory:
        mock_factory.return_value = _make_mock_model("")
        with pytest.raises(GenerationError, match="empty response"):
            await generate_answer(VALID_QUERY, VALID_CONTEXT, VALID_HISTORY)


async def test_generation_error_wraps_cause():
    original = ConnectionError("API unreachable")
    with patch("src.llm.generator._get_chat_model") as mock_factory:
        mock_model = MagicMock()
        mock_model.invoke.side_effect = original
        mock_factory.return_value = mock_model

        with pytest.raises(GenerationError) as exc_info:
            await generate_answer(VALID_QUERY, VALID_CONTEXT, VALID_HISTORY)

    assert exc_info.value.cause is original


# ---------------------------------------------------------------------------
# ValueError propagation (contract violations)
# ---------------------------------------------------------------------------

async def test_empty_query_raises_value_error_not_generation_error():
    with pytest.raises(ValueError, match="query"):
        await generate_answer("", VALID_CONTEXT, VALID_HISTORY)


async def test_empty_context_raises_value_error_not_generation_error():
    with pytest.raises(ValueError, match="context"):
        await generate_answer(VALID_QUERY, "", VALID_HISTORY)


async def test_whitespace_query_raises_value_error():
    with pytest.raises(ValueError, match="query"):
        await generate_answer("   ", VALID_CONTEXT, VALID_HISTORY)