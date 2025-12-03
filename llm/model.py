# app/services/rag_service.py
import logging
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from app.config import settings
from app.database import ConversationHistory, SessionLocal
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatOllama(
    model=settings.chat_model,
    base_url=settings.ollama_host.rstrip("/"),
    temperature=0.0,
)

# Prompt template with placeholders
MPLIFY_150_PROMPT = """Hello! I'm your AI assistant specialized and my role is to provide correct address to you.

Conversation History:
{conversation_history}

Correct address:
{retrieved_address}

User query:
{user_query}

Provide your response as a conversation-style answer.
"""

prompt_template = PromptTemplate.from_template(MPLIFY_150_PROMPT)


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
        history_text = []
        for record in reversed(history):
            history_text.append(f"User Query: {record.query}")
            if isinstance(record.response, str):
                history_text.append(f"Response: {record.response}")
        return "\n".join(history_text[-10:])  # Last 10 lines max
    finally:
        db.close()


def save_to_history(
    session_id: str, query: str, response: str, score: Optional[str] = None
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
    partial_address: str, user_query: str, session_id: Optional[str] = None
) -> str:
    """
    RAG-based address query returning conversation-style output.

    Args:
        partial_address: The address query
        user_query: The user's query/question about the address
        session_id: Optional session ID for conversation memory

    Returns:
        LLM-generated response as text.
    """
    # Retrieve conversation history if session_id provided
    history_context = (
        get_conversation_history(session_id, limit=3)
        if session_id
        else "No previous conversation."
    )

    logger.debug(f"Using conversation history:\n{history_context}")

    # Format the prompt
    prompt_text = prompt_template.format(
        retrieved_address=partial_address,
        conversation_history=history_context,
        user_query=user_query,
    )

    # Generate LLM response
    response = llm.invoke(prompt_text)
    if hasattr(response, "content"):
        response_text = response.content
    else:
        response_text = str(response)

    # Save response to conversation history
    print(response_text)
    if session_id:
        save_to_history(session_id, user_query, response_text, score="0")

    return response_text
