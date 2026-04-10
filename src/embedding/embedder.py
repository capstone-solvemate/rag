# src/embedding/embedder.py
"""
Embedding Pipeline: converts text chunks into numerical vectors.

What is an embedding?
- A vector (array of floats) that represents the semantic meaning of a text
- Similar texts produce vectors that are close together in vector space
- This enables semantic search: find chunks by meaning, not just keywords

Model: text-embedding-3-small
- 1536 dimensions
- Cost-effective for prototyping
- Sufficient quality for enterprise document Q&A
"""

from typing import List

from langchain_openai import OpenAIEmbeddings

from src.config import config
from src.utils.logger import get_logger

logger = get_logger("src.embedding.embedder")


def get_embedding_model() -> OpenAIEmbeddings:
    """
    Initialize and return the OpenAI embedding model.

    Returns a singleton-like object — LangChain handles
    connection pooling internally.
    """
    logger.info(f"Initializing embedding model: {config.EMBEDDING_MODEL}")

    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
        dimensions=config.EMBEDDING_DIMENSIONS,
    )

    logger.info("✅ Embedding model ready.")
    return embeddings


def embed_single_text(text: str) -> List[float]:
    """
    Embed a single string into a vector.
    Useful for embedding a user query at retrieval time.

    Args:
        text: Raw text string to embed

    Returns:
        List of floats representing the text vector
    """
    model = get_embedding_model()
    vector = model.embed_query(text)
    logger.info(f"✅ Embedded query ({len(text)} chars) → vector ({len(vector)} dims)")
    return vector


if __name__ == "__main__":
    # Quick sanity check: embed a single sentence and inspect the vector
    config.validate()

    test_text = "How do I replace the ink cartridge?"
    print(f"\nEmbedding test text: '{test_text}'")

    vector = embed_single_text(test_text)

    print(f"\n{'='*50}")
    print(f"✅ Embedding successful!")
    print(f"   Model      : {config.EMBEDDING_MODEL}")
    print(f"   Dimensions : {len(vector)}")
    print(f"   Preview    : {vector[:5]}...")  # show first 5 values
    print(f"{'='*50}")