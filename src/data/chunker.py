"""
Chunking Pipeline: splits long documents into smaller, semantically meaningful chunks.

Why do we need chunking?
- LLMs have a limited context window (max tokens they can process at once)
- We only want to send RELEVANT text to the LLM, not the entire document
- A good chunk retains complete meaning without being cut mid-sentence

Key parameters:
- chunk_size   : maximum number of characters per chunk (default: 1000)
- chunk_overlap : number of overlapping characters between consecutive chunks (default: 200)
                  Overlap ensures context is not lost at chunk boundaries.
"""
import json
import os
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import config
from src.utils.logger import get_logger

logger = get_logger("src.data.chunker")


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split a list of Documents into smaller chunks.

    The splitter tries to cut at the most natural boundaries first:
    paragraph breaks → line breaks → spaces → any character.
    This preserves semantic meaning better than fixed-size splitting.

    Args:
        documents    : List of Document objects from the loader
        chunk_size   : Maximum number of characters per chunk
        chunk_overlap: Number of characters shared between consecutive chunks

    Returns:
        List of chunk Documents, each with enriched metadata
    """
    logger.info(
        f"Starting chunking for {len(documents)} documents "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Enrich each chunk with positional and size metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_size"]  = len(chunk.page_content)

    # Compute basic statistics for observability
    sizes   = [len(c.page_content) for c in chunks]
    avg     = sum(sizes) / len(sizes) if sizes else 0
    minimum = min(sizes) if sizes else 0
    maximum = max(sizes) if sizes else 0

    logger.info(f"Chunking complete: {len(documents)} documents → {len(chunks)} chunks")
    logger.info(f"   Avg: {avg:.0f} | Min: {minimum} | Max: {maximum} characters")

    return chunks


def save_chunks_to_json(chunks: List[Document], output_path: str = None) -> str:
    """
    Persist chunks to a JSON file for manual inspection and debugging.

    This is useful to visually verify that chunking produces
    semantically coherent segments before moving to embedding.

    Args:
        chunks     : List of chunk Documents
        output_path: Destination file path.
                     Defaults to data/processed/chunks.json

    Returns:
        Absolute path of the saved file
    """
    if output_path is None:
        os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
        output_path = os.path.join(config.DATA_PROCESSED_DIR, "chunks.json")

    chunks_data = [
        {
            "chunk_index"   : chunk.metadata.get("chunk_index", i),
            "content"       : chunk.page_content,
            "content_length": len(chunk.page_content),
            "metadata"      : chunk.metadata,
        }
        for i, chunk in enumerate(chunks)
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Chunks saved to: {output_path}")
    return output_path


def explore_chunk_sizes(documents: List[Document]) -> None:
    """
    Run a quick benchmark across multiple chunk size configurations.

    Call this once to understand how different settings affect
    chunk count, average size, and content granularity.
    Use the output to make an informed decision before production.
    """
    configurations = [
        (500,  50,  "Small  — suitable for short FAQ documents"),
        (1000, 200, "Medium — general purpose default (recommended)"),
        (1500, 300, "Large  — suitable for long technical manuals"),
    ]

    print("\n" + "=" * 65)
    print("CHUNK SIZE EXPLORATION")
    print("=" * 65)

    for chunk_size, overlap, label in configurations:
        chunks = chunk_documents(documents, chunk_size, overlap)
        sizes  = [len(c.page_content) for c in chunks]
        avg    = sum(sizes) / len(sizes) if sizes else 0

        print(f"\n {label}")
        print(f"   chunk_size={chunk_size}, overlap={overlap}")
        print(f"   Total chunks  : {len(chunks)}")
        print(f"   Avg size      : {avg:.0f} characters")
        print(f"   Sample chunk  :")
        print(f"   '{chunks[0].page_content[:120].strip()}...'")

    print("\n" + "=" * 65)
    print("Selection guide:")
    print("   - Fewer, larger chunks → richer context, slower retrieval")
    print("   - More, smaller chunks → precise retrieval, risk of lost context")
    print("=" * 65)


if __name__ == "__main__":
    from src.data.loader import load_documents_from_directory

    print("\nStep 1: Loading documents...")
    documents = load_documents_from_directory()

    if not documents:
        print("No documents found. Add files to data/raw/ first.")
        raise SystemExit(1)

    print("\n Step 2: Exploring chunk size configurations...")
    explore_chunk_sizes(documents)

    # Adjust these values based on the exploration output above
    CHUNK_SIZE    = 1000
    CHUNK_OVERLAP = 200

    print(f"\n Step 3: Final chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = chunk_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)

    print("\n Step 4: Saving chunks to disk...")
    output_path = save_chunks_to_json(chunks)

    print(f"\n{'=' * 50}")
    print(f" Pipeline completed.")
    print(f"   Input  : {len(documents)} document pages")
    print(f"   Output : {len(chunks)} chunks")
    print(f"   Ratio  : {len(chunks)/len(documents):.1f} chunks per page on average")
    print(f"   Saved  : {output_path}")
    print(f"{'=' * 50}")