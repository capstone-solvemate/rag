import pytest

from src.llm.prompt_templates import (
    SYSTEM_PROMPT,
    _USER_PROMPT_TEMPLATE,
    build_user_prompt,
)


# ---------------------------------------------------------------------------
# SYSTEM_PROMPT — structural integrity
# ---------------------------------------------------------------------------

def test_system_prompt_is_non_empty():
    assert SYSTEM_PROMPT and SYSTEM_PROMPT.strip()


def test_system_prompt_contains_citation_instruction():
    """Model must be told to cite sources by reference number."""
    assert "[1]" in SYSTEM_PROMPT or "reference number" in SYSTEM_PROMPT.lower()


def test_system_prompt_contains_refusal_phrase():
    """Exact refusal phrase must be present for programmatic detection."""
    assert "I could not find a relevant answer in the available documents." in SYSTEM_PROMPT


def test_system_prompt_contains_no_fabrication_rule():
    assert "fabricate" in SYSTEM_PROMPT.lower() or "do not fabricate" in SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# build_user_prompt — happy path
# ---------------------------------------------------------------------------

def test_output_contains_query():
    result = build_user_prompt("What is the warranty period?", "Some context.")
    assert "What is the warranty period?" in result


def test_output_contains_context():
    result = build_user_prompt("Any question.", "[1] Source: doc.pdf\nSome content.")
    assert "[1] Source: doc.pdf" in result
    assert "Some content." in result


def test_output_contains_answer_label():
    """Prompt must end with 'Answer:' to direct the model."""
    result = build_user_prompt("Question?", "Context.")
    assert result.strip().endswith("Answer:")


def test_output_uses_template_structure():
    """Context block must appear before the question."""
    result = build_user_prompt("My question.", "My context.")
    context_pos = result.index("My context.")
    question_pos = result.index("My question.")
    assert context_pos < question_pos


def test_whitespace_in_inputs_is_stripped():
    result = build_user_prompt("  What is X?  ", "  Context here.  ")
    assert result.startswith("Context:\nContext here.")
    assert "What is X?" in result


# ---------------------------------------------------------------------------
# build_user_prompt — validation / error cases
# ---------------------------------------------------------------------------

def test_empty_query_raises_value_error():
    with pytest.raises(ValueError, match="query"):
        build_user_prompt("", "Some context.")


def test_whitespace_only_query_raises_value_error():
    with pytest.raises(ValueError, match="query"):
        build_user_prompt("   ", "Some context.")


def test_empty_context_raises_value_error():
    with pytest.raises(ValueError, match="context"):
        build_user_prompt("Valid query.", "")


def test_whitespace_only_context_raises_value_error():
    with pytest.raises(ValueError, match="context"):
        build_user_prompt("Valid query.", "   ")


# ---------------------------------------------------------------------------
# Template integrity
# ---------------------------------------------------------------------------

def test_template_contains_required_placeholders():
    """Catch accidental edits that break the template format."""
    assert "{context}" in _USER_PROMPT_TEMPLATE
    assert "{query}" in _USER_PROMPT_TEMPLATE