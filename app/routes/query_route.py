# app/routes/query_route.py
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from app.schemas import RAGQueryRequest
from app.services.rag_service import rag_address_query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/query-address", tags=["Address RAG"])

class MultiMatchResponse(BaseModel):
    matches: List[Dict[str, Any]] = Field(..., description="List of matching addresses with scores")

@router.post("", response_model=MultiMatchResponse)
async def query_address_endpoint(request: RAGQueryRequest):
    try:
        results = rag_address_query(
            partial_address=request.query,
            top_k=request.top_k
        )
        return {"matches": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))