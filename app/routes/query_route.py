# app/routes/query_route.py
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from app.schemas import RAGQueryRequest

from pydantic import BaseModel, Field


from entity_extractor.relevent_places import run_workflow
from entity_extractor.search_feild import search_qdrant_by_filter
from llm.model import rag_address_query
from vector_db.search import vector_search
from app.database import ConversationHistory, SessionLocal
from datetime import datetime

router = APIRouter(prefix="/query-address", tags=["Address RAG"])


class MultiMatchResponse(BaseModel):
    llm_response: str = Field(
        ..., description="List of matching addresses with scores"
    )
    extracted_address_matches: List[Dict[str, Any]] = Field(
        ..., description="List of matching addresses with scores"
    )


@router.post("", response_model=MultiMatchResponse)
async def query_address_endpoint(request: RAGQueryRequest):


    try:
        filter_dict = await run_workflow()
        result =await search_qdrant_by_filter(
   
    filter_dict,
    request.query

)
        if result ==[]:
        
            results, query_result_array = await vector_search(request.query, request.top_k)
            result =  results + query_result_array

        llm_response = rag_address_query(
    str( result ),request.query,  request.session_id
) 
        print(result)
       

        

        return {"llm_response": str(llm_response), "extracted_address_matches": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
