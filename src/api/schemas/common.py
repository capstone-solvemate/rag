# src/api/schemas/common.py
from typing import Optional
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    detail: str = Field(description="Human-readable error message.")
    error_code: Optional[str] = Field(
        default=None,
        description="Machine-readable error code for client handling.",
    )