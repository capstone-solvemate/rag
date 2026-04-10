# src/retrieval/retriever.py
"""
Retriever: given a user query, finds the most relevant chunks
from the vector store using semantic similarity search.

How similarity search works:
1. Embed the user query into a vector
2. Compute cosine similarity between query vector and all stored vectors
3. Return top-K chunks with highest similarity scores

Two search methods available:
- similarity_search         : returns top-K by similarity score
- max_marginal_relevance    : returns diverse results (avoids redundancy)
  MMR balances relevance vs diversity — useful when chunks are repetitive
"""

from typing import List, Tuple

from langchain.schema import Document
from langchain_chroma import Chroma

from src.config import config
from src.utils.logger import get_logger


logger = get_logger("src.retrieval.retriever")


def similarity_search(
    query: str,
    vector_store: Chroma,
    k: int = 5,
) -> List[Document]:
    """
    Retrieve the top-K most semantically similar chunks for a query.

    Args:
        query       : User's question in natural language
        vector_store: Populated Chroma instance
        k           : Number of chunks to retrieve

    Returns:
        List of top-K Document chunks, ordered by relevance
    """
    logger.info(f"Searching for: '{query}' (top {k} results)")

    results = vector_store.similarity_search(query, k=k)

    logger.info(f"✅ Retrieved {len(results)} chunks.")
    return results


def similarity_search_with_score(
    query: str,
    vector_store: Chroma,
    k: int = 5,
) -> List[Tuple[Document, float]]:
    """
    Same as similarity_search but also returns similarity scores.
    Scores range from 0 to 1 — higher is more similar.
    Useful for debugging and setting confidence thresholds later.

    Returns:
        List of (Document, score) tuples, ordered by score descending
    """
    logger.info(f"Searching with scores: '{query}' (top {k} results)")

    results = vector_store.similarity_search_with_score(query, k=k)

    for doc, score in results:
        logger.info(
            f"   Score: {score:.4f} | "
            f"File: {doc.metadata.get('file_name', 'unknown')} | "
            f"Preview: {doc.page_content[:60]}..."
        )

    return results


def mmr_search(
    query: str,
    vector_store: Chroma,
    k: int = 5,
    fetch_k: int = 20,
) -> List[Document]:
    """
    Maximum Marginal Relevance search — retrieves relevant BUT diverse chunks.

    Why use MMR?
    If your document has repeated sections, pure similarity search
    might return 5 nearly identical chunks. MMR avoids this by
    penalizing results that are too similar to already-selected ones.

    Args:
        query       : User's question
        vector_store: Populated Chroma instance
        k           : Final number of chunks to return
        fetch_k     : Candidate pool size before diversity filtering
                      (fetch_k > k, typically 3-4x)

    Returns:
        List of k diverse and relevant Document chunks
    """
    logger.info(f"MMR search: '{query}' (k={k}, fetch_k={fetch_k})")

    results = vector_store.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=fetch_k,
    )

    logger.info(f"✅ MMR retrieved {len(results)} diverse chunks.")
    return results


if __name__ == "__main__":
    from src.embedding.indexer import get_collection_count, get_vector_store

    config.validate()

    # Load existing vector store (no re-indexing needed)
    print("\n🔄 Loading vector store...")
    vector_store = get_vector_store()

    total = get_collection_count(vector_store)
    print(f"   Vectors in store: {total}")

    # Test queries
    test_queries = [
        "How do I replace the ink cartridge?",
        "What to do when paper is jammed?",
        "How to clean the printer head?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        results = similarity_search_with_score(query, vector_store, k=3)

        for i, (doc, score) in enumerate(results, 1):
            print(f"\n  Result #{i} (score: {score:.4f})")
            print(f"  File  : {doc.metadata.get('file_name', 'unknown')}")
            print(f"  Page  : {doc.metadata.get('page', 'unknown')}")
            print(f"  Text  : {doc.page_content[:200]}...")