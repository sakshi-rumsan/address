# qdrant_search_post.py

import re
import os
import requests
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

from vector_db.address_extractor import run_workflow


# -------------------
# Load environment variables
# -------------------
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

# -------------------
# Ollama Embedder
# -------------------
embedder = OllamaEmbeddings(
    base_url=OLLAMA_URL,
    model="nomic-embed-text:latest"
)

def get_embedding(text: str):
    """Get embedding vector for the query text."""
    return embedder.embed_query(text)

def search_normalized_address(query: str, top_k: int = 5):
    """Search Qdrant collection using HTTP POST request."""
    query_vector = get_embedding(query)

    # Build Qdrant search URL from .env
    url = f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search"

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "vector": query_vector,
        "limit": top_k,
        "with_payload": True,
        "with_vectors": False
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()

    return data.get("result", [])

# -------------------
# Test
def vector_search(text: str, top_k: int = 2):
    # 1️⃣ Extract addresses from text
    extracted_addresses = run_workflow (text)
  # This is already a list
    print(extracted_addresses)

    # 2️⃣ Search Qdrant for each extracted address and collect all results

    # Container for all Qdrant search results
    all_results = []


    # Iterate over each extracted address
    for addr in extracted_addresses:
        print(f"Searching Qdrant for address: {addr}")
        results = search_normalized_address(addr, top_k=top_k)
        
        # Iterate over each result for the current address
        for hit in results:
            all_results.append({
                "query": addr,
                "score": hit.get("score"),
                "payload": hit.get("payload", {})
            })
    query_result = search_normalized_address(text, top_k=top_k)
    for hit in query_result:
            all_results.append({
                "query": addr,
                "score": hit.get("score"),
                "payload": hit.get("payload", {})
            })
    
    return all_results
