# app/services/qdrant_service.py
from qdrant_client import QdrantClient
from app.config import settings


class QdrantService:
    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            timeout=60,
        )
        self.collection_name = settings.collection_name
        # Auto-detect client version and pick correct method
        self._use_modern_search = hasattr(self.client, "search")

        if not self.client.collection_exists(self.collection_name):
            raise ValueError(f"Collection '{self.collection_name}' does not exist!")

    def search(
        self, vector: list[float], top_k: int = 3, score_threshold: float = 0.75
    ):
        """
        Works with BOTH old (<1.6) and new (>=1.6) qdrant-client versions
        """
        try:
            if self._use_modern_search:
                # official method
                hits = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=vector,
                    limit=top_k,
                    score_threshold=score_threshold,
                )
            else:
                # fallback using scroll + manual cosine similarity
                from qdrant_client.http.models import Filter, FieldCondition, MatchValue
                from qdrant_client.http.models import PointStruct
                import numpy as np

                # Get all points (or up to 1000 if too many)
                points, _ = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    with_vectors=True,
                )

                if not points:
                    raise ValueError("No points in collection")

                vectors = np.array([p.vector for p in points])
                query_vec = np.array(vector).reshape(1, -1)
                vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
                query_vec = query_vec / np.linalg.norm(query_vec)

                similarities = (vectors @ query_vec.T).flatten()
                top_indices = np.argsort(similarities)[-top_k:][::-1]

                class ScoredPoint:
                    def __init__(self, score, payload):
                        self.score = score
                        self.payload = payload

                hits = [
                    ScoredPoint(score=float(similarities[i]), payload=points[i].payload)
                    for i in top_indices
                    if similarities[i] >= score_threshold
                ]

            return hits

        except Exception as e:
            raise ValueError(f"Search failed in '{self.collection_name}': {str(e)}")


# Keep singleton
qdrant_service = QdrantService()
