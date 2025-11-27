from langchain_community.embeddings import OllamaEmbeddings
from app.config import settings

embeddings = OllamaEmbeddings(
    model=settings.embedding_model,
    base_url=settings.ollama_host.rstrip("/")
)

def get_embedding(text: str) -> list[float]:
    return embeddings.embed_query(text)