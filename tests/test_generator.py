from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain.schema import AIMessage

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


def _make_mock_model(response_text: str) -> MagicMock:
    """Return a mock ChatOpenAI that returns response_text on invoke()."""
    mock_model = MagicMock()
    mock_model.invoke.return_value = AIMessage(content=response_text)
    return mock_model


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_returns_string_on_success():
    with patch("src.llm.generator._get_chat_model") as mock_factory:
        mock_factory.return_value = _make_mock_model("The warranty is one year. [1]")
        result = generate_answer(VALID_QUERY, VALID_CONTEXT)

    assert isinstance(result, str)
    assert len(result) > 0


def test_returns_stripped_answer():
    with patch("src.llm.generator._get_chat_model") as mock_factory:
        mock_factory.return_value = _make_mock_model("  Answer with whitespace.  ")
        result = generate_answer(VALID_QUERY, VALID_CONTEXT)

    assert result == "Answer with whitespace."


def test_model_receives_system_and_human_messages():
    from langchain.schema import HumanMessage, SystemMessage
    from src.llm.prompt_templates import SYSTEM_PROMPT

    with patch("src.llm.generator._get_chat_model") as mock_factory:
        mock_model = _make_mock_model("An answer.")
        mock_factory.return_value = mock_model
        generate_answer(VALID_QUERY, VALID_CONTEXT)

    call_args = mock_model.invoke.call_args[0][0]
    assert isinstance(call_args[0], SystemMessage)
    assert isinstance(call_args[1], HumanMessage)
    assert call_args[0].content == SYSTEM_PROMPT


def test_refusal_phrase_is_passed_through():
    """If the LLM returns the refusal phrase it must be returned as-is."""
    refusal = "I could not find a relevant answer in the available documents."
    with patch("src.llm.generator._get_chat_model") as mock_factory:
        mock_factory.return_value = _make_mock_model(refusal)
        result = generate_answer(VALID_QUERY, VALID_CONTEXT)

    assert result == refusal


# ---------------------------------------------------------------------------
# GenerationError cases
# ---------------------------------------------------------------------------

def test_api_exception_raises_generation_error():
    with patch("src.llm.generator._get_chat_model") as mock_factory:
        mock_model = MagicMock()
        mock_model.invoke.side_effect = RuntimeError("connection timeout")
        mock_factory.return_value = mock_model

        with pytest.raises(GenerationError) as exc_info:
            generate_answer(VALID_QUERY, VALID_CONTEXT)

    assert "LLM generation failed" in str(exc_info.value)
    assert isinstance(exc_info.value.cause, RuntimeError)


def test_empty_llm_response_raises_generation_error():
    with patch("src.llm.generator._get_chat_model") as mock_factory:
        mock_factory.return_value = _make_mock_model("")
        with pytest.raises(GenerationError, match="empty response"):
            generate_answer(VALID_QUERY, VALID_CONTEXT)


def test_generation_error_wraps_cause():
    original = ConnectionError("API unreachable")
    with patch("src.llm.generator._get_chat_model") as mock_factory:
        mock_model = MagicMock()
        mock_model.invoke.side_effect = original
        mock_factory.return_value = mock_model

        with pytest.raises(GenerationError) as exc_info:
            generate_answer(VALID_QUERY, VALID_CONTEXT)

    assert exc_info.value.cause is original


# ---------------------------------------------------------------------------
# ValueError propagation (contract violations)
# ---------------------------------------------------------------------------

def test_empty_query_raises_value_error_not_generation_error():
    with pytest.raises(ValueError, match="query"):
        generate_answer("", VALID_CONTEXT)


def test_empty_context_raises_value_error_not_generation_error():
    with pytest.raises(ValueError, match="context"):
        generate_answer(VALID_QUERY, "")


def test_whitespace_query_raises_value_error():
    with pytest.raises(ValueError, match="query"):
        generate_answer("   ", VALID_CONTEXT)