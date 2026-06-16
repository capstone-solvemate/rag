from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.schemas.detection import DetectionResult
from src.core.exceptions import GenerationError
from src.api.main import app

client = TestClient(app)

_BASE_PAYLOAD = {
    "image_base64": "abc123",
    "media_type": "image/jpeg",
    "detection_mode": "quality",
    "k": 3,
}

_DETECTION_RESULT = DetectionResult(
    detected_issues=["horizontal banding"],
    affected_components=["print head"],
    severity="medium",
    confidence=0.85,
)

_MOCK_DOCUMENTS = [
    MagicMock(
        page_content="Clean the print head using the maintenance menu.",
        metadata={
            "file_name": "epson_manual.pdf",
            "file_path": "/docs/epson_manual.pdf",
            "file_type": ".pdf",
            "chunk_index": "0",
        },
    )
]


def _patch_pipeline(
    detection_result=_DETECTION_RESULT,
    documents=_MOCK_DOCUMENTS,
    recommended_actions="Clean the print head immediately.",
):
    """Helper: patch all three external calls in the detection pipeline."""
    return (
        patch(
            "src.api.routes.detection.analyze_image_for_defects",
            new=AsyncMock(return_value=detection_result),
        ),
        patch(
            "src.api.routes.detection.get_vector_store",
            return_value=MagicMock(),
        ),
        patch(
            "src.api.routes.detection.similarity_search",
            return_value=documents,
        ),
        patch(
            "src.api.routes.detection.generate_answer_with_image",
            new=AsyncMock(return_value=recommended_actions),
        ),
    )


class TestDetectRoute:
    def test_happy_path_returns_detection_response(self):
        with (
            patch("src.api.routes.detection.analyze_image_for_defects", new=AsyncMock(return_value=_DETECTION_RESULT)),
            patch("src.api.routes.detection.get_vector_store", return_value=MagicMock()),
            patch("src.api.routes.detection.similarity_search", return_value=_MOCK_DOCUMENTS),
            patch("src.api.routes.detection.generate_answer_with_image", new=AsyncMock(return_value="Clean the print head.")),
        ):
            response = client.post("/detect", json=_BASE_PAYLOAD)

        assert response.status_code == 200
        body = response.json()
        assert body["detected_issues"] == ["horizontal banding"]
        assert body["affected_components"] == ["print head"]
        assert body["severity"] == "medium"
        assert 0.0 <= body["confidence"] <= 1.0
        assert body["recommended_actions"] == "Clean the print head."
        assert len(body["sources"]) == 1

    def test_no_issues_detected_returns_empty_response(self):
        empty_result = DetectionResult(
            detected_issues=[],
            affected_components=[],
            severity="low",
            confidence=0.1,
        )
        with patch(
            "src.api.routes.detection.analyze_image_for_defects",
            new=AsyncMock(return_value=empty_result),
        ):
            response = client.post("/detect", json=_BASE_PAYLOAD)

        assert response.status_code == 200
        body = response.json()
        assert body["detected_issues"] == []
        assert body["affected_components"] == []
        assert body["recommended_actions"] == "No issues detected in the provided image."
        assert body["sources"] == []

    def test_no_kb_docs_returns_ungrounded_response(self):
        with (
            patch("src.api.routes.detection.analyze_image_for_defects", new=AsyncMock(return_value=_DETECTION_RESULT)),
            patch("src.api.routes.detection.get_vector_store", return_value=MagicMock()),
            patch("src.api.routes.detection.similarity_search", return_value=[]),  # empty KB
        ):
            response = client.post("/detect", json=_BASE_PAYLOAD)

        assert response.status_code == 200
        body = response.json()
        assert body["detected_issues"] == ["horizontal banding"]
        assert "no relevant documentation" in body["recommended_actions"]
        assert body["sources"] == []

    def test_detection_llm_failure_returns_503(self):
        with patch(
            "src.api.routes.detection.analyze_image_for_defects",
            new=AsyncMock(side_effect=GenerationError("Detection failed", cause=None)),
        ):
            response = client.post("/detect", json=_BASE_PAYLOAD)

        assert response.status_code == 503
        assert response.json()["error_code"] == "DETECTION_FAILED"

    def test_generation_llm_failure_returns_503(self):
        with (
            patch("src.api.routes.detection.analyze_image_for_defects", new=AsyncMock(return_value=_DETECTION_RESULT)),
            patch("src.api.routes.detection.get_vector_store", return_value=MagicMock()),
            patch("src.api.routes.detection.similarity_search", return_value=_MOCK_DOCUMENTS),
            patch(
                "src.api.routes.detection.generate_answer_with_image",
                new=AsyncMock(side_effect=GenerationError("Generation failed", cause=None)),
            ),
        ):
            response = client.post("/detect", json=_BASE_PAYLOAD)

        assert response.status_code == 503
        assert response.json()["error_code"] == "GENERATION_FAILED"

    def test_retrieval_failure_returns_500(self):
        with (
            patch("src.api.routes.detection.analyze_image_for_defects", new=AsyncMock(return_value=_DETECTION_RESULT)),
            patch("src.api.routes.detection.get_vector_store", return_value=MagicMock()),
            patch(
                "src.api.routes.detection.similarity_search",
                side_effect=Exception("Chroma connection error"),
            ),
        ):
            response = client.post("/detect", json=_BASE_PAYLOAD)

        assert response.status_code == 500