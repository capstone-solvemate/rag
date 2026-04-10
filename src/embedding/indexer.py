# src/embedding/indexer.py
"""
Indexer: embeds all document chunks and stores them in Chroma vector store.

This is a one-time (or on-update) operation:
1. Load chunks from the chunking pipeline
2. Send each chunk to OpenAI Embeddings API
3. Store the resulting vectors + text + metadata in Chroma

After indexing, the vector store can be queried instantly
without calling OpenAI Embeddings again.

Cost note:
- text-embedding-3-small costs $0.00002 per 1K tokens
- 263 chunks × ~250 tokens avg ≈ 65K tokens ≈ $0.0013 per full index
- Very cheap — safe to re-index during development
"""

import os
from typing import List

import chromadb
from langchain_chroma import Chroma
from langchain.schema import Document

from src.config import config
from src.embedding.embedder import get_embedding_model
from src.utils.logger import get_logger

logger = get_logger("src.embedding.indexer")


def get_vector_store(reset: bool = False) -> Chroma:
    """
    Initialize (or load existing) Chroma vector store from disk.

    Args:
        reset: If True, delete existing collection and start fresh.
               Use this when re-indexing after document updates.

    Returns:
        Chroma vector store instance ready for querying or inserting
    """
    os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)

    embedding_model = get_embedding_model()

    if reset:
        logger.warning("Reset flag is set — deleting existing collection.")
        client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        try:
            client.delete_collection(config.CHROMA_COLLECTION_NAME)
            logger.info("✅ Existing collection deleted.")
        except Exception:
            logger.info("No existing collection found, starting fresh.")

    vector_store = Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )

    logger.info(f"✅ Vector store ready at: {config.CHROMA_PERSIST_DIR}")
    return vector_store


def get_collection_count(vector_store: Chroma) -> int:
    """Return collection count through one adapter function.

    Chroma count access currently relies on an internal attribute.
    Keeping that dependency here makes future upgrades simpler and safer.
    """
    return int(vector_store._collection.count())


def index_documents(chunks: List[Document], reset: bool = False) -> Chroma:
    """
    Embed all chunks and store them in the Chroma vector store.

    Args:
        chunks: List of Document chunks from the chunking pipeline
        reset : Whether to wipe and re-index from scratch

    Returns:
        Populated Chroma vector store
    """
    logger.info(f"Starting indexing for {len(chunks)} chunks...")

    vector_store = get_vector_store(reset=reset)

    # Check how many documents are already indexed
    existing_count = get_collection_count(vector_store)
    logger.info(f"Existing documents in store: {existing_count}")

    if existing_count > 0 and not reset:
        logger.warning(
            "Vector store already contains documents. "
            "Use reset=True to re-index. Skipping indexing."
        )
        return vector_store

    # LangChain handles batching automatically
    # It will call OpenAI API in batches to avoid rate limits
    logger.info("Sending chunks to OpenAI Embeddings API...")
    logger.info("(This may take 30-60 seconds for ~263 chunks)")

    vector_store.add_documents(chunks)

    final_count = get_collection_count(vector_store)
    logger.info(f"✅ Indexing complete. Total vectors in store: {final_count}")

    return vector_store


if __name__ == "__main__":
    from src.data.loader import load_documents_from_directory
    from src.data.chunker import chunk_documents

    config.validate()

    # ── Step 1: Load & chunk (reuse Week 1 pipeline) ────────────
    print("\n🔄 Step 1: Loading and chunking documents...")
    documents = load_documents_from_directory()
    chunks    = chunk_documents(
        documents,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    print(f"   Ready to index: {len(chunks)} chunks")

    # ── Step 2: Index into Chroma ────────────────────────────────
    print("\n🔄 Step 2: Indexing chunks into Chroma...")
    print("   ⚠️  This will call OpenAI API — estimated cost: < $0.01")

    vector_store = index_documents(chunks, reset=False)

    # ── Summary ──────────────────────────────────────────────────
    total = get_collection_count(vector_store)
    print(f"\n{'='*50}")
    print(f"✅ INDEXING COMPLETE")
    print(f"   Chunks indexed : {total}")
    print(f"   Stored at      : {config.CHROMA_PERSIST_DIR}")
    print(f"{'='*50}")