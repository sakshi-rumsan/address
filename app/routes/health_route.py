# app/routes/health_route.py
from fastapi import APIRouter
from app.services.health_service import check_health

router = APIRouter(prefix="/rag", tags=["Health"])

@router.get("/health")
async def health_check():
    return check_health()