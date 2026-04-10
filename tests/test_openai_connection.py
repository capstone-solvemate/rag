import sys

from openai import OpenAI
from src.config import config
from src.utils.logger import get_logger

logger = get_logger("test.openai")

def test_openai_connection():
    logger.info("Ping started...")
    
    try:
        config.validate()
        
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        models = client.models.list()
        
        logger.info(f"Connected successfully! {len(list(models))} models are available.")
        logger.info("OpenAI API ready to use.")
        return True
        
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return False

if __name__ == "__main__":
    success = test_openai_connection()
    sys.exit(0 if success else 1)