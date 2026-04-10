from pydantic import BaseModel, Field


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