# src/api/schemas/chat.py
from typing import Literal
from pydantic import BaseModel, Field, model_validator

class PictureAttachment(BaseModel):
    data: str

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)
    pictures: list[PictureAttachment] = Field(default=[])


class ChatRequest(BaseModel):
    k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve. Must be between 1 and 20.",
    )
    history: list[Message] = Field(
        ...,
        min_length=1,
        description="Conversation history in OpenAI message format. Last message must be from user.",
    )

    @model_validator(mode="after")
    def last_message_must_be_user(self) -> "ChatRequest":
        if self.history[-1].role != "user":
            raise ValueError("The last message in history must have role 'user'.")
        return self

    @property
    def query(self) -> str:
        """Convenience accessor for the latest user query."""
        return self.history[-1].content


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