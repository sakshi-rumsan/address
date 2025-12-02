# app/routes/history_route.py
from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any
from app.database import ConversationHistory, SessionLocal
from pydantic import BaseModel
from datetime import datetime

router = APIRouter(prefix="/history", tags=["Conversation History"])


class HistoryResponse(BaseModel):
    id: int
    session_id: str
    query: str
    response: Dict[str, Any]
    score: str | None
    timestamp: datetime

    class Config:
        from_attributes = True


@router.get("/{session_id}", response_model=List[HistoryResponse])
async def get_session_history(
    session_id: str, limit: int = Query(default=10, ge=1, le=100)
):
    """Retrieve conversation history for a specific session."""
    db = SessionLocal()
    try:
        history = (
            db.query(ConversationHistory)
            .filter(ConversationHistory.session_id == session_id)
            .order_by(ConversationHistory.timestamp.desc())
            .limit(limit)
            .all()
        )

        if not history:
            raise HTTPException(
                status_code=404, detail=f"No history found for session: {session_id}"
            )

        return history
    finally:
        db.close()


@router.delete("/{session_id}")
async def clear_session_history(session_id: str):
    """Clear conversation history for a specific session."""
    db = SessionLocal()
    try:
        deleted = (
            db.query(ConversationHistory)
            .filter(ConversationHistory.session_id == session_id)
            .delete()
        )
        db.commit()

        if deleted == 0:
            raise HTTPException(
                status_code=404, detail=f"No history found for session: {session_id}"
            )

        return {"message": f"Deleted {deleted} records for session {session_id}"}
    finally:
        db.close()
