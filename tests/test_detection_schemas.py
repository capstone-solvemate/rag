from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas.detection import DetectionRequest, DetectionResult


class TestDetectionRequest:
    def test_valid_request(self):
        req = DetectionRequest(
            image_base64="abc123",
            media_type="image/jpeg",
            detection_mode="quality",
        )
        assert req.k == 5  # default

    def test_invalid_detection_mode(self):
        with pytest.raises(ValidationError):
            DetectionRequest(
                image_base64="abc123",
                media_type="image/jpeg",
                detection_mode="unknown",
            )

    def test_invalid_media_type(self):
        # gif is excluded from DetectionRequest (unlike ImageChatRequest)
        with pytest.raises(ValidationError):
            DetectionRequest(
                image_base64="abc123",
                media_type="image/gif",
                detection_mode="defect",
            )

    def test_k_out_of_bounds(self):
        with pytest.raises(ValidationError):
            DetectionRequest(
                image_base64="abc123",
                media_type="image/jpeg",
                detection_mode="both",
                k=0,
            )

    def test_empty_image_base64(self):
        with pytest.raises(ValidationError):
            DetectionRequest(
                image_base64="",
                media_type="image/jpeg",
                detection_mode="quality",
            )


class TestDetectionResult:
    def test_confidence_out_of_bounds(self):
        with pytest.raises(ValidationError):
            DetectionResult(
                detected_issues=["banding"],
                affected_components=["print head"],
                severity="low",
                confidence=1.5,
            )

    def test_invalid_severity(self):
        with pytest.raises(ValidationError):
            DetectionResult(
                detected_issues=[],
                affected_components=[],
                severity="critical",  # not in Literal
                confidence=0.9,
            )