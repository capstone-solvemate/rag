from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ImageChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="The user's question. Retrieval runs on this text query.",
    )
    image_base64: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded image data (no data URI prefix). "
                    "The image is sent to the LLM as visual context alongside "
                    "the retrieved document chunks.",
    )
    media_type: Literal["image/jpeg", "image/png", "image/webp", "image/gif"] = Field(
        description="MIME type of the image. Used to construct the data URI "
                    "for the OpenAI vision API.",
    )
    k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve. Must be between 1 and 20.",
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank or whitespace-only.")
        return v

    @field_validator("image_base64")
    @classmethod
    def image_base64_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("image_base64 must not be blank or whitespace-only.")
        return v