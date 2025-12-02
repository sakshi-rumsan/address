# app/services/rag_service.py
import logging
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from app.services.embedding_service import get_embedding
from app.services.qdrant_service import qdrant_service
from app.config import settings
from app.database import ConversationHistory, SessionLocal
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

llm = ChatOllama(
    model=settings.chat_model,
    base_url=settings.ollama_host.rstrip("/"),
    temperature=0.0,
)

# FULL ORIGINAL PROMPT — ALL {} IN THE EXAMPLE ARE DOUBLED {{}} SO LANGCHAIN DOES NOT BREAK
MPLIFY_150_PROMPT = """Hello! I'm your AI assistant specialized in parsing and formatting addresses according to the Mplify 150 Installation Place and Service Site Management standard. I'm here to help you extract and structure address information into the exact Installation Place Fielded Address Representation format as defined in the specification.

Conversation History:
{conversation_history}

Core Directive
Parse any input address and return it in the exact Mplify 150 Installation Place Fielded Address Representation format, following all requirements and constraints specified in the standard. Use the conversation history above to understand context and user preferences from previous queries, and provide a personalized experience based on our previous interactions.

Required Output Format
Structure all addresses using these exact attributes from Table 3 and Table 4:
Primary Address Attributes:
Street Number: Number identifying a specific property on a public street
Street Number Suffix: The first street number suffix or suffix for single number
Street Number Last: Last number in a range of street numbers (optional)
Street Number Last Suffix: Suffix for Street Number Last (only when Street Number Last is present)
Street Pre-Direction: Direction appearing before Street Name
Street Name: Name of the street
Street Type: Type of street (alley, avenue, boulevard, crescent, drive, highway, lane, terrace, parade, place, tarn, way, wharf)
Street Post-Direction: Direction appearing after Street Name
PO Box Number: Post office box number
Locality: Area within local authority boundaries
City: City where address is located
Postal Code: Postal delivery area descriptor
Postal Code Extension: Extension used on postal code
State Or Province: State or Province location
Country: Two-character ISO 3166 country code (REQUIRED)
Building Name: Well-known building name
Private Street Number: Street number on private street
Private Street Name: Private street name
Language: Two-letter ISO 639-2023 language code

Sub Unit Attributes (if applicable):
Sub Unit Type: Type of sub unit (BERTH, FLAT, PIER, SUITE, SHOP, TOWER, UNIT, ROOM, LEVEL)
Sub Unit Name: Distinctive value for the sub unit

Mandatory Requirements:
MUST use ISO 3166 two-letter country codes for Country attribute
MUST use ISO 639-2023 two-letter codes for Language attribute
MUST include Sub Unit Name and Sub Unit Type together if any sub unit is specified
MUST use Street Number Last Suffix only when Street Number Last is present
MUST use Street Pre-Direction when direction comes before Street Name
MUST use Street Post-Direction when direction comes after Street Name

Processing Rules:
Parse addresses hierarchically from most specific to least specific
Handle range addresses using Street Number and Street Number Last
Separate directional indicators appropriately (pre/post)
Identify and categorize sub-units correctly
Maintain data integrity and completeness
Default to standard abbreviations when full forms are provided

Response Format:
Return a flat JSON object (no extra nesting) with only populated fields, using exactly these snake_case keys.

Example of correct output:
{{
  "street_number": "123",
  "street_name": "Main",
  "street_type": "Street",
  "city": "Anytown",
  "state_or_province": "CA",
  "postal_code": "12345",
  "country": "US",
  "language": "EN",
  "sub_units": [
    {{
      "sub_unit_type": "SUITE",
      "sub_unit_name": "200"
    }}
  ]
}}

CRITICAL:
- Return ONLY the JSON object
- Use flat structure (no primary_address_attributes wrapper)
- If no sub-units, use "sub_units": []
- Country = 2-letter ISO code
- Language = 2-letter code

Address to parse:
{retrieved_address}

Return only the JSON. No markdown. No extra text.
"""

prompt = PromptTemplate.from_template(MPLIFY_150_PROMPT)
parser = JsonOutputParser()
chain = prompt | llm | parser


def get_conversation_history(session_id: str, limit: int = 5) -> str:
    """Retrieve recent conversation history for a session."""
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
            return ""

        # Format history for context (most recent first, so reverse)
        history_text = []
        for record in reversed(history):
            history_text.append(f"User Query: {record.query}")
            if isinstance(record.response, dict):
                history_text.append(f"Response: {record.response}")

        return "\n".join(history_text[-10:])  # Last 10 lines max
    finally:
        db.close()


def save_to_history(
    session_id: str, query: str, response: Dict[str, Any], score: Optional[str] = None
):
    """Save query and response to conversation history."""
    db = SessionLocal()
    try:
        record = ConversationHistory(
            session_id=session_id,
            query=query,
            response=response,
            score=score,
            timestamp=datetime.utcnow(),
        )
        db.add(record)
        db.commit()
        logger.debug(f"Saved conversation to history: session={session_id}")
    except Exception as e:
        logger.error(f"Failed to save conversation history: {e}")
        db.rollback()
    finally:
        db.close()


def rag_address_query(
    partial_address: str, top_k: int = 3, session_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    RAG-based address query with optional conversation memory.

    Args:
        partial_address: The address query
        top_k: Number of results to return
        session_id: Optional session ID for conversation memory
    """
    logger.info(
        f"RAG Query: '{partial_address}' | top_k={top_k} | session={session_id}"
    )

    # Get conversation history if session_id provided
    history_context = ""
    if session_id:
        history_context = get_conversation_history(session_id, limit=3)
        if history_context:
            logger.debug(f"Retrieved conversation history for session {session_id}")

    query_vector = get_embedding(partial_address)
    hits = qdrant_service.search(
        vector=query_vector, top_k=top_k + 2, score_threshold=0.70
    )

    if not hits:
        if session_id:
            save_to_history(session_id, partial_address, {"error": "no_results"})
        return []

    results = []
    for hit in hits[:top_k]:  # Respect user's top_k
        payload = hit.payload
        raw_address = payload.get("normalized_address", "").strip()
        if not raw_address:
            continue

        try:
            structured = chain.invoke(
                {
                    "retrieved_address": raw_address,
                    "conversation_history": history_context
                    or "No previous conversation.",
                }
            )
            result = {"score": round(float(hit.score), 4), "address": structured}
            results.append(result)

            # Save first result to history
            if session_id and len(results) == 1:
                save_to_history(
                    session_id, partial_address, structured, score=str(result["score"])
                )

        except Exception as e:
            logger.warning(f"Failed to parse one result: {e}")
            # Still include raw version as fallback
            results.append(
                {
                    "score": round(float(hit.score), 4),
                    "address": {"raw_address": raw_address, "error": "parsing_failed"},
                }
            )

    return results
