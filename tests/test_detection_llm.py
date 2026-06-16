from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.schemas.detection import DetectionResult
from src.core.exceptions import GenerationError
from src.llm.context_builder import build_detection_retrieval_query
from src.llm.prompt_templates import build_detection_analysis_prompt


# ---------------------------------------------------------------------------
# build_detection_analysis_prompt
# ---------------------------------------------------------------------------

class TestBuildDetectionAnalysisPrompt:
    def test_quality_mode_vocabulary(self):
        messages = build_detection_analysis_prompt(
            image_base64="abc123",
            media_type="image/jpeg",
            detection_mode="quality",
        )
        system_content = messages[0]["content"]
        assert "banding" in system_content
        assert "streaking" in system_content
        # defect vocabulary must not bleed into quality mode
        assert "roller wear" not in system_content

    def test_defect_mode_vocabulary(self):
        messages = build_detection_analysis_prompt(
            image_base64="abc123",
            media_type="image/jpeg",
            detection_mode="defect",
        )
        system_content = messages[0]["content"]
        assert "roller wear" in system_content
        assert "head clog" in system_content
        # quality vocabulary must not bleed into defect mode
        assert "banding" not in system_content

    def test_both_mode_includes_all_vocabulary(self):
        messages = build_detection_analysis_prompt(
            image_base64="abc123",
            media_type="image/jpeg",
            detection_mode="both",
        )
        system_content = messages[0]["content"]
        assert "banding" in system_content
        assert "roller wear" in system_content

    def test_data_uri_constructed_correctly(self):
        messages = build_detection_analysis_prompt(
            image_base64="abc123",
            media_type="image/png",
            detection_mode="quality",
        )
        image_block = messages[1]["content"][0]
        assert image_block["image_url"]["url"] == "data:image/png;base64,abc123"

    def test_raises_on_empty_image(self):
        with pytest.raises(ValueError, match="image_base64"):
            build_detection_analysis_prompt(
                image_base64="",
                media_type="image/jpeg",
                detection_mode="quality",
            )

    def test_raises_on_unknown_mode(self):
        with pytest.raises(ValueError, match="detection_mode"):
            build_detection_analysis_prompt(
                image_base64="abc123",
                media_type="image/jpeg",
                detection_mode="unknown",
            )


# ---------------------------------------------------------------------------
# build_detection_retrieval_query
# ---------------------------------------------------------------------------

class TestBuildDetectionRetrievalQuery:
    def _make_result(self, issues, components) -> DetectionResult:
        return DetectionResult(
            detected_issues=issues,
            affected_components=components,
            severity="low",
            confidence=0.9,
        )

    def test_both_issues_and_components(self):
        result = self._make_result(
            issues=["horizontal banding", "faded cyan"],
            components=["print head", "cyan cartridge"],
        )
        query = build_detection_retrieval_query(result)
        assert "horizontal banding" in query
        assert "faded cyan" in query
        assert "print head" in query
        assert "cyan cartridge" in query
        assert "affecting" in query

    def test_issues_only(self):
        result = self._make_result(issues=["streaking"], components=[])
        query = build_detection_retrieval_query(result)
        assert "streaking" in query
        assert "affecting" not in query

    def test_components_only(self):
        result = self._make_result(issues=[], components=["paper feed roller"])
        query = build_detection_retrieval_query(result)
        assert "paper feed roller" in query
        assert "affecting" in query

    def test_both_empty_raises(self):
        result = self._make_result(issues=[], components=[])
        with pytest.raises(ValueError):
            build_detection_retrieval_query(result)


# ---------------------------------------------------------------------------
# analyze_image_for_defects
# ---------------------------------------------------------------------------

_VALID_DETECTION_JSON = json.dumps({
    "detected_issues": ["horizontal banding"],
    "affected_components": ["print head"],
    "severity": "medium",
    "confidence": 0.85,
})


class TestAnalyzeImageForDefects:
    """All LLM calls are mocked — we test parsing and error handling logic only."""

    async def test_happy_path_returns_detection_result(self):
        from src.llm.generator import analyze_image_for_defects

        mock_response = MagicMock()
        mock_response.choices[0].message.content = _VALID_DETECTION_JSON

        with patch("src.llm.generator._get_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(
                return_value=mock_response
            )
            result = await analyze_image_for_defects(
                image_base64="abc123",
                media_type="image/jpeg",
                detection_mode="quality",
            )

        assert isinstance(result, DetectionResult)
        assert result.detected_issues == ["horizontal banding"]
        assert result.severity == "medium"
        assert result.confidence == 0.85

    async def test_malformed_json_raises_generation_error(self):
        from src.llm.generator import analyze_image_for_defects

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "not json at all"

        with patch("src.llm.generator._get_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(
                return_value=mock_response
            )
            with pytest.raises(GenerationError, match="malformed JSON"):
                await analyze_image_for_defects(
                    image_base64="abc123",
                    media_type="image/jpeg",
                    detection_mode="quality",
                )

    async def test_empty_llm_response_raises_generation_error(self):
        from src.llm.generator import analyze_image_for_defects

        mock_response = MagicMock()
        mock_response.choices[0].message.content = ""

        with patch("src.llm.generator._get_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(
                return_value=mock_response
            )
            with pytest.raises(GenerationError, match="empty response"):
                await analyze_image_for_defects(
                    image_base64="abc123",
                    media_type="image/jpeg",
                    detection_mode="quality",
                )

    async def test_api_failure_raises_generation_error(self):
        from src.llm.generator import analyze_image_for_defects

        with patch("src.llm.generator._get_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(
                side_effect=Exception("OpenAI timeout")
            )
            with pytest.raises(GenerationError, match="did not succeed"):
                await analyze_image_for_defects(
                    image_base64="abc123",
                    media_type="image/jpeg",
                    detection_mode="quality",
                )