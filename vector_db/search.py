# qdrant_search_post.py
import json
import os
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import SearchParams
from langchain_ollama import OllamaEmbeddings

from vector_db.address_extractor import run_workflow


# -------------------
# Load environment variables
# -------------------
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION = os.getenv("COLLECTION_NAME")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)


# -------------------
# Ollama Embedder
# -------------------
embedder = OllamaEmbeddings(base_url=OLLAMA_URL, model="nomic-embed-text:latest")


def clean_qdrant_response(search_result):
    # Parse JSON if search_result.json() returns a string
    data = search_result.json()
    if isinstance(data, str):
        data = json.loads(data)  # convert string -> dict

    cleaned_points = []

    for point in data.get("points", []):
        payload = point.get("payload", {})
        cleaned_points.append(
            {
                "id": point.get("id"),
                "score": point.get("score"),
                "normalized_address": payload.get("normalized_address"),
                "address_type": payload.get("address_type"),
                "street_name": payload.get("street_name"),
                "locality": payload.get("locality"),
                "town": payload.get("town"),
                "postcode": payload.get("postcode"),
                "region": payload.get("region"),
                "tlc": payload.get("tlc"),
            }
        )

    return {"results": cleaned_points}


def get_embedding(text: str):
    """Get embedding vector for the query text."""
    return embedder.embed_query(text)


async def search_normalized_address(query: str, top_k: int = 1):
    """Search Qdrant collection using HTTP POST request."""
    query_vector = get_embedding(query)
    search_result = await client.query_points(
        collection_name="new-zealand",
        query=query_vector,  # type: ignore
        with_payload=True,
        with_vectors=False,
        limit=top_k,
        search_params=SearchParams(hnsw_ef=128),
    )
    cleaned = clean_qdrant_response(search_result)
    print(json.dumps(cleaned, indent=2))

    return cleaned


# -------------------
# Test
async def vector_search(text: str, top_k: int = 2):
    # 1️⃣ Extract addresses from text
    extracted_addresses = run_workflow(text)
    print(extracted_addresses)
    all_results = []
    query_result_array = []

    # Iterate over each extracted address
    for addr in extracted_addresses:
        print(f"Searching Qdrant for address: {addr}")
        results = await search_normalized_address(addr, top_k=top_k)
        all_results.append({"query": addr, "payload": results})
    query_result = await search_normalized_address(text, top_k=2)
    query_result_array.append(
        {
            "query": text,
            "payload": query_result,
        }
    )
    return all_results, query_result_array
