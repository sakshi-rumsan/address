# app/routes/query_route.py
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from app.schemas import RAGQueryRequest

from pydantic import BaseModel, Field

from vector_db.search import vector_search
from app.database import ConversationHistory, SessionLocal
from datetime import datetime

router = APIRouter(prefix="/query-address", tags=["Address RAG"])


class MultiMatchResponse(BaseModel):
    raw_query: List[Dict[str, Any]] = Field(
        ..., description="List of matching addresses with scores"
    )
    extracted_address_matches: List[Dict[str, Any]] = Field(
        ..., description="List of matching addresses with scores"
    )


@router.post("", response_model=MultiMatchResponse)
async def query_address_endpoint(request: RAGQueryRequest):
    try:
        results, query_result_array = vector_search(request.query, request.top_k)

        # Save to database if session_id is provided
        if request.session_id:
            db = SessionLocal()
            try:
                # Prepare response data to store
                response_data = {
                    "raw_query": query_result_array,
                    "extracted_address_matches": results,
                }

                # Get top score from results if available
                top_score = None
                if results and len(results) > 0:
                    top_score = str(results[0].get("score", ""))

                # Save to conversation history
                record = ConversationHistory(
                    session_id=request.session_id,
                    query=request.query,
                    response=response_data,
                    score=top_score,
                    timestamp=datetime.utcnow(),
                )
                db.add(record)
                db.commit()
            except Exception as db_error:
                print(f"Failed to save to database: {db_error}")
                db.rollback()
            finally:
                db.close()

        return {"raw_query": query_result_array, "extracted_address_matches": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
