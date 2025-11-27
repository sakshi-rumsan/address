# app/services/health_service.py
import logging
from app.config import settings
from app.services.qdrant_service import qdrant_service
from app.services.embedding_service import embeddings
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

def check_health() -> dict:
    health = {
        "qdrant": False,
        "ollama": False,
        "embeddingModel": False,
        "chatModel": False
    }

    # 1. Qdrant — check collection exists
    try:
        collections = qdrant_service.client.get_collections()
        health["qdrant"] = settings.collection_name in [c.name for c in collections.collections]
    except Exception as e:
        logger.error(f"Qdrant health failed: {e}")

    # 2–4. Ollama — reuse the exact same clients your RAG uses (synchronous)
    try:
        # Test embedding model (this is what RAG actually uses)
        test_vec = embeddings.embed_query("health")
        if isinstance(test_vec, list) and len(test_vec) > 0:
            health["ollama"] = True
            health["embeddingModel"] = True
    except Exception as e:
        logger.error(f"Embedding model failed: {e}")

    try:
        # Test chat model (same as RAG)
        llm = ChatOllama(
            model=settings.chat_model,
            base_url=settings.ollama_host.rstrip("/"),
            temperature=0.0
        )
        response = llm.invoke("ping")
        if response and hasattr(response, "content"):
            health["chatModel"] = True
            health["ollama"] = True   
    except Exception as e:
        logger.error(f"Chat model failed: {e}")

    return {"data": health}