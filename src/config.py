import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application settings with repo-root anchored path conventions.

    Deriving all storage directories from one base keeps script execution,
    package execution, and persisted artifacts aligned.
    """

    _BASE_PATH: Path = Path(__file__).resolve().parent.parent
    BASE_DIR: str = str(_BASE_PATH)
    DATA_RAW_DIR: str = str(_BASE_PATH / "data" / "raw")
    DATA_PROCESSED_DIR: str = str(_BASE_PATH / "data" / "processed")
    CHROMA_PERSIST_DIR: str = str(_BASE_PATH / "data" / "chroma")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536

    CHROMA_COLLECTION_NAME: str = "enterprise_docs"

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    APP_ENV: str = os.getenv("APP_ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG")

    def validate(self):
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY hasn't been set in .env!")
        print("All configs are valid.")

config = Config()