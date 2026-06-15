from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.api.schemas.chat import SourceDocument


class DetectionRequest(BaseModel):
    image_base64: str = Field(
        ...,
        min_length=1,
        description=(
            "Base64-encoded image data (no data URI prefix). "
            "The image drives both detection and retrieval — no text query is needed."
        ),
    )
    media_type: Literal["image/jpeg", "image/png", "image/webp"] = Field(
        description="MIME type of the image. Used to construct the data URI for the OpenAI vision API.",
    )
    detection_mode: Literal["quality", "defect", "both"] = Field(
        description=(
            "'quality' — detect print quality issues (banding, streaking, ghosting, fading, color bleed). "
            "'defect' — detect physical part defects (wear, damage, contamination). "
            "'both' — run both analyses."
        ),
    )
    k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of knowledge base chunks to retrieve. Must be between 1 and 20.",
    )


class DetectionResult(BaseModel):
    """Internal model: structured output parsed from the vision LLM's JSON response.

    Not returned directly to the client — consumed by the retrieval and generation steps.
    """

    detected_issues: list[str] = Field(
        description="List of specific issues or defects identified in the image.",
    )
    affected_components: list[str] = Field(
        description="Printer components or print areas implicated by the detected issues.",
    )
    severity: Literal["low", "medium", "high"] = Field(
        description="Overall severity of the detected issues.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model's confidence in the detection result, in the range [0.0, 1.0].",
    )


class DetectionResponse(BaseModel):
    detected_issues: list[str] = Field(
        description="List of specific issues or defects identified in the image.",
    )
    affected_components: list[str] = Field(
        description="Printer components or print areas implicated by the detected issues.",
    )
    severity: Literal["low", "medium", "high"] = Field(
        description="Overall severity of the detected issues.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model's confidence in the detection result, in the range [0.0, 1.0].",
    )
    recommended_actions: str = Field(
        description="LLM-generated recommended actions grounded in the knowledge base.",
    )
    sources: list[SourceDocument] = Field(
        description="Knowledge base chunks used to generate recommended_actions.",
    )