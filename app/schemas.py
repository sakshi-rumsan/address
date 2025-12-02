from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class RAGQueryRequest(BaseModel):
    """Request model for address RAG query"""

    query: str = Field(
        ...,
        description="Partial address to search for (e.g., '123 Main', 'ROAD', 'Suite 200')",
    )
    top_k: int = Field(
        default=1, ge=1, le=10, description="Number of similar addresses to retrieve"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="LLM temperature for response generation (lower = more precise)",
    )
    session_id: Optional[str] = Field(
        default=None, description="Session ID for conversation memory tracking"
    )


class RAGQueryResponse(BaseModel):
    """Response model for address RAG query - returns structured Mplify 150 address"""

    answer: Dict[str, Any] = Field(
        ..., description="Complete address in Mplify 150 JSON format or error message"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "answer": """{
  "street_number": "123",
  "street_name": "Main",
  "street_type": "Street",
  "city": "Springfield",
  "state_or_province": "CA",
  "postal_code": "12345",
  "country": "US",
  "language": "EN",
  "sub_units": [
    {
      "sub_unit_type": "SUITE",
      "sub_unit_name": "200"
    }
  ]
}"""
            }
        }
