# src/api/schemas/health.py
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(description="Overall service status: 'ok' or 'degraded'.")
    chroma_doc_count: int = Field(description="Number of documents indexed in Chroma.")
    openai_reachable: bool = Field(description="Whether OpenAI API is reachable.")
    python_version: str = Field(description="Runtime Python version.")
    app_env: str = Field(description="Current APP_ENV value from config.")