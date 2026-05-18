import os
import sys
from src.config import config
from pathlib import Path
from typing import List

"""
Document Loader: membaca berbagai format file dan mengubahnya jadi teks.

Mendukung:
- TXT: format paling sederhana
- PDF: format umum dokumen perusahaan  
- DOCX: dokumen Word

Output: List of LangChain Document objects
Setiap Document punya dua field:
- page_content: isi teksnya
- metadata: info tambahan (nama file, halaman, dll)
"""
from langchain_community.document_loaders import (
    TextLoader,        # .txt
    PyPDFLoader,       # .pdf
    Docx2txtLoader,    # .docx
)
from langchain_core.documents import Document

from src.utils.logger import get_logger

logger = get_logger("src.data.loader")


def load_single_document(file_path: str) -> List[Document]:
    """
    Load satu file dokumen berdasarkan ekstensinya.
    
    Args:
        file_path: Path lengkap ke file
        
    Returns:
        List of Document (biasanya satu file = satu atau beberapa Document)
    """
    path = Path(file_path)
    extension = path.suffix.lower()
    
    logger.info(f"Loading file: {path.name} (format: {extension})")
    
    # Pilih loader yang sesuai dengan format file
    if extension == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
        
    elif extension == ".pdf":
        # PyPDFLoader otomatis split per halaman
        loader = PyPDFLoader(file_path)
        
    elif extension == ".docx":
        loader = Docx2txtLoader(file_path)
        
    else:
        logger.warning(f"Format unsupported: {extension}. Skipped this file.")
        return []
    
    documents = loader.load()
    
    # Tambahkan metadata tambahan yang berguna
    for doc in documents:
        doc.metadata["file_name"] = path.name
        doc.metadata["file_path"] = str(file_path)
        doc.metadata["file_type"] = extension
    
    logger.info(f"Successfullly loaded {len(documents)} document from {path.name}")
    return documents


def load_documents_from_directory(directory: str = None) -> List[Document]:
    """
    Load semua dokumen dari sebuah folder.
    
    Args:
        directory: Path ke folder. Default: data/raw dari config.
        
    Returns:
        List of semua Document dari semua file
    """
    if directory is None:
        directory = config.DATA_RAW_DIR
    
    supported_extensions = {".txt", ".pdf", ".docx"}
    all_documents = []
    
    logger.info(f"Scanning folder: {directory}")
    
    # os.walk = jalan ke semua subfolder secara rekursif
    for root, dirs, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            extension = Path(filename).suffix.lower()
            
            if extension in supported_extensions:
                docs = load_single_document(file_path)
                all_documents.extend(docs)
    
    logger.info(f"Total: {len(all_documents)} documents successfully loaded")
    return all_documents


if __name__ == "__main__":
    # Quick test: jalankan file ini langsung untuk cek apakah loader bekerja
    docs = load_documents_from_directory()
    
    if docs:
        print(f"\n{'='*50}")
        print(f"Total docs: {len(docs)}")
        print(f"{'='*50}")
        print(f"Example first doc:")
        print(f"  File: {docs[0].metadata.get('file_name')}")
        print(f"  Text length: {len(docs[0].page_content)} characters")
        print(f"  Preview: {docs[0].page_content[:200]}...")
    else:
        print("No documents found, add the data to data/raw/")