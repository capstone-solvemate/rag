from typing import List, Optional
from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    file_name: str = Field(description="Original file name the chunk came from.")
    file_path: str = Field(description="Full path to the source file.")
    chunk_index: int = Field(description="Index of this chunk within its source document.")


class ChatResponse(BaseModel):
    query: str = Field(description="The original query as received.")
    answer: str = Field(description="LLM-generated answer grounded in retrieved context.")
    sources: List[SourceDocument] = Field(
        description="Source chunks used to generate the answer."
    )


class HealthResponse(BaseModel):
    status: str = Field(description="Overall service status: 'ok' or 'degraded'.")
    chroma_doc_count: int = Field(description="Number of documents indexed in Chroma.")
    openai_reachable: bool = Field(description="Whether OpenAI API is reachable.")
    python_version: str = Field(description="Runtime Python version.")
    app_env: str = Field(description="Current APP_ENV value from config.")


class ErrorResponse(BaseModel):
    detail: str = Field(description="Human-readable error message.")
    error_code: Optional[str] = Field(
        default=None,
        description="Machine-readable error code for client handling.",
    )