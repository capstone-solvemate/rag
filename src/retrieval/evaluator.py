# src/retrieval/evaluator.py
"""
Retrieval Evaluator: measures how well our retriever finds relevant chunks.

Metric: Precision@K
- For each test query, we manually define which keywords MUST appear
  in a relevant result (ground truth)
- We check how many of the top-K results contain those keywords
- Precision@K = relevant_found / K

This is a lightweight evaluation — no labeled dataset needed.
Good enough for Week 2. We'll use more rigorous evals in Week 5.

Target: Precision@K >= 0.6 (project success criterion)
"""

from typing import List, Dict

from langchain_chroma import Chroma

from src.retrieval.retriever import similarity_search_with_score
from src.utils.logger import get_logger

logger = get_logger("src.retrieval.evaluator")

# ---------------------------------------------------------------------------
# Test set: define queries and keywords that MUST appear in relevant results
# Adjust these based on YOUR document content
# ---------------------------------------------------------------------------
TEST_QUERIES: List[Dict] = [
    {
        # Based on: chunks about paper loading instructions
        "query"   : "How do I load paper into the printer?",
        "keywords": ["paper", "load", "tray", "sheet"],
    },
    {
        # Based on: chunks about scanning originals on glass
        "query"   : "How do I scan a document?",
        "keywords": ["scan", "original", "glass", "scanner"],
    },
    {
        # Based on: chunks about copying basics
        "query"   : "How do I make a copy?",
        "keywords": ["copy", "copies", "copying"],
    },
    {
        # Based on: chunks about control panel buttons
        "query"   : "What do the buttons on the control panel do?",
        "keywords": ["button", "control", "panel", "light"],
    },
    {
        # Based on: chunks about software updater
        "query"   : "How do I update the printer firmware?",
        "keywords": ["update", "firmware", "software", "epson"],
    },
]

def is_relevant(chunk_text: str, keywords: List[str]) -> bool:
    """
    Determine if a chunk is relevant by checking keyword presence.
    Case-insensitive. A chunk is relevant if ANY keyword matches.

    Args:
        chunk_text: The chunk's text content
        keywords  : List of keywords that indicate relevance

    Returns:
        True if at least one keyword is found in the chunk
    """
    text_lower = chunk_text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def evaluate_retriever(vector_store: Chroma, k: int = 5) -> Dict:
    """
    Run all test queries and compute Precision@K for each.

    Args:
        vector_store: Populated Chroma vector store
        k           : Number of results to retrieve per query

    Returns:
        Dict with per-query scores and overall average Precision@K
    """
    logger.info(f"Starting retrieval evaluation (k={k})...")

    results_summary = []

    print(f"\n{'='*65}")
    print(f"RETRIEVAL EVALUATION — Precision@{k}")
    print(f"{'='*65}")

    for test_case in TEST_QUERIES:
        query    = test_case["query"]
        keywords = test_case["keywords"]

        # Retrieve top-K chunks
        retrieved = similarity_search_with_score(query, vector_store, k=k)

        # Count how many retrieved chunks are relevant
        relevant_count = sum(
            1 for doc, _ in retrieved
            if is_relevant(doc.page_content, keywords)
        )

        precision_at_k = relevant_count / k if k > 0 else 0.0

        results_summary.append({
            "query"        : query,
            "precision_at_k": precision_at_k,
            "relevant"     : relevant_count,
            "retrieved"    : k,
        })

        # Visual indicator
        status = "✅" if precision_at_k >= 0.6 else "⚠️ "

        print(f"\n{status} Query   : {query}")
        print(f"   Keywords : {keywords}")
        print(f"   Relevant : {relevant_count}/{k}")
        print(f"   P@{k}     : {precision_at_k:.2f}")

    # Compute overall average
    avg_precision = sum(r["precision_at_k"] for r in results_summary) / len(results_summary)
    passed        = avg_precision >= 0.6

    print(f"\n{'='*65}")
    print(f"OVERALL Precision@{k} : {avg_precision:.2f}")
    print(f"Target              : >= 0.60")
    print(f"Status              : {'✅ PASSED' if passed else '❌ NEEDS IMPROVEMENT'}")
    print(f"{'='*65}")

    return {
        "avg_precision_at_k": avg_precision,
        "k"                 : k,
        "passed"            : passed,
        "details"           : results_summary,
    }


if __name__ == "__main__":
    from src.embedding.indexer import get_vector_store
    from src.config import config

    config.validate()

    vector_store = get_vector_store()
    evaluate_retriever(vector_store, k=5)