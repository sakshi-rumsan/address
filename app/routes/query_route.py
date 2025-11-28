# app/routes/query_route.py
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from app.schemas import RAGQueryRequest

from pydantic import BaseModel, Field

from vector_db.search import vector_search

router = APIRouter(prefix="/query-address", tags=["Address RAG"])

class MultiMatchResponse(BaseModel):
    matches: List[Dict[str, Any]] = Field(..., description="List of matching addresses with scores")

@router.post("", response_model=MultiMatchResponse)
async def query_address_endpoint(request: RAGQueryRequest):
    try:
        results =  vector_search(request.query,2)
        return {"matches": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))