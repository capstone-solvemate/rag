# src/api/schemas/chat.py
from typing import Literal
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question to answer from the document corpus.",
    )
    k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve. Must be between 1 and 20.",
    )
    history: list[Message] = Field(
        default_factory=list,
        description="Conversation history in OpenAI message format.",
    )


class SourceDocument(BaseModel):
    file_name: str = Field(description="Original file name the chunk came from.")
    file_path: str = Field(description="Full path to the source file.")
    chunk_index: int = Field(description="Index of this chunk within its source document.")


class ChatResponse(BaseModel):
    query: str = Field(description="The original query as received.")
    answer: str = Field(description="LLM-generated answer grounded in retrieved context.")
    sources: list[SourceDocument] = Field(
        description="Source chunks used to generate the answer."
    )