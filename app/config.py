from pydantic_settings import BaseSettings
from typing import List
import logging
import sys


class Settings(BaseSettings):
    app_name: str = "Adress Validation API"
    app_description: str = "API to validate and complete addresses using RAG approach"
    app_version: str = "1.0.0"
    api_v1_prefix: str = "/api/v1"
    allowed_origins: List[str] = ["*"]
    debug: bool = False  # Default to False for production safety

    qdrant_host: str
    qdrant_port: int
    qdrant_url: str
    qdrant_api_key: str
    ollama_host: str
    chat_model: str
    embedding_model: str
    collection_name: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Initialize settings
settings = Settings()

# Configure global logging behavior
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],  # Explicit stdout
    force=True,  # Override any existing configuration
)

# Create a logger instance for config
logger = logging.getLogger(__name__)

if settings.debug:
    logger.debug(" Debug mode is ON — showing DEBUG, INFO, WARNING, ERROR logs")
else:
    logger.info(" Running in production mode — showing INFO, WARNING, ERROR logs")
