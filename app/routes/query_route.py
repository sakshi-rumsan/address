# app/routes/query_route.py
from fastapi import APIRouter
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from app.schemas import RAGQueryRequest
from entity_extractor.relevent_places import run_workflow
from llm.model import rag_address_query
from vector_db.search import vector_search

router = APIRouter(prefix="/query-address", tags=["Address RAG"])


class MultiMatchResponse(BaseModel):
    llm_response: str = Field(..., description="List of matching addresses with scores")
    extracted_address_matches: List[Dict[str, Any]] = Field(
        ..., description="List of matching addresses with scores"
    )


@router.post("", response_model=MultiMatchResponse)
async def query_address_endpoint(request: RAGQueryRequest):
    try:
        # Get best matches
        merged_best_matches = await run_workflow(request.query)  # returns dict
        result = merged_best_matches

        # Flatten dicts if needed
        if isinstance(result, dict):
            result = [result]  # convert single dict to list

        print("Final Qdrant Results:", result)

        # Check if any address was detected
        has_address = bool(
            result
            and any(isinstance(item, dict) and item.get("results") for item in result)
        )

        # If no address detected, have a conversation instead
        if not has_address:
            llm_response = rag_address_query("", request.query, request.session_id)
            return {
                "llm_response": str(llm_response).strip(),
                "extracted_address_matches": [],
            }

        # Fallback vector search if no results but address was detected
        if not result:
            results, query_result_array = await vector_search(request.query, 5)
            result = results + query_result_array  # assuming both are lists

        # Call your RAG/LLM query with address results
        llm_response = rag_address_query(str(result), request.query, request.session_id)

        # Return properly formatted dict
        return {
            "llm_response": str(llm_response).strip(),
            "extracted_address_matches": result,  # always list of dicts
        }

    except Exception as e:
        print("Error in query_address_endpoint:", e)
        raise  # re-raise to propagate
