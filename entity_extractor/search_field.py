# qdrant_search_post.py
import json
import os
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from langchain_ollama import OllamaEmbeddings
from qdrant_client.models import Filter, FieldCondition, MatchText, MatchValue

from entity_extractor.cache_field import save_field_to_single_json


# -------------------
# Load environment variables
# -------------------
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("COLLECTION_NAME")

# -------------------
# Initialize clients
# -------------------
client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
embedder = OllamaEmbeddings(base_url=OLLAMA_URL, model="nomic-embed-text:latest")


# -------------------
# Async helper functions
# -------------------
async def get_collections():
    collections = await client.get_collections()
    print("Collections:", collections)


async def count_points_with_filter(filter_condition: Filter):
    """
    Count points in the collection that match the filter.
    """
    count_result = await client.count(
        collection_name=QDRANT_COLLECTION,
        count_filter=filter_condition,
    )
    return count_result.count


async def get_unique_towns(feild_name: str):
    """
    Scroll through the collection to extract all unique town names.
    """
    unique_towns = set()
    offset = None
    limit = 100  # Adjust limit for efficient pagination

    while True:
        scroll_result, next_page_offset = await client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in scroll_result:
            town_name = point.payload.get(feild_name)
            if town_name:
                unique_towns.add(town_name)

        if not next_page_offset:
            break
        offset = next_page_offset

    return list(unique_towns)


# -------------------
# Main async function
# -------------------


def load_fields_json(json_file="fields.json"):
    """Load existing JSON file or return empty dict."""
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


async def SearchFeilds(feild_name: str):
    json_file = "fields.json"
    existing_fields = load_fields_json(json_file)
    if feild_name in existing_fields:
        return existing_fields[feild_name]
    my_filter = Filter(
        must_not=[FieldCondition(key=feild_name, match=MatchValue(value=""))]
    )
    total_matching = await count_points_with_filter(my_filter)
    print(f"Total points with non-empty {feild_name}: {total_matching}")
    towns = await get_unique_towns(feild_name)
    save_field_to_single_json(feild_name, towns, json_file=json_file)
    print(f"Unique {feild_name}: {towns}")
    return towns


def get_embedding(text: str):
    """Get embedding vector for the query text."""
    return embedder.embed_query(text)


from qdrant_client.models import Filter, FieldCondition, MatchText


async def search_qdrant_by_filter(filter_dict: dict, query: str, limit: int = 1):
    print(filter_dict)
    query_vector = get_embedding(query)
    # Build Qdrant Filter, using best_match and skipping None/empty
    must_conditions = []
    for key, val in filter_dict.items():
        # Ensure val is a dict and best_match is not empty
        if isinstance(val, dict) and val.get("best_match"):
            must_conditions.append(
                FieldCondition(key=key, match=MatchText(text=val["best_match"]))
            )
    my_filter = Filter(must=must_conditions) if must_conditions else None
    # Perform the search
    search_result = await client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
        query_filter=my_filter,
        with_payload=True,
        with_vectors=False,
    )
    # search_result is a list of ScoredPoint
    points = search_result.points
    clean_results = [
        {"id": p.id, "score": p.score, "payload": p.payload} for p in points
    ]
    print(clean_results)
    return clean_results
