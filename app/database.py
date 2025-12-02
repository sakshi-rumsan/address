# app/database.py
from sqlalchemy import create_engine, Column, String, JSON, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# SQLite database file path
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./conversation_history.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class ConversationHistory(Base):
    """Model for storing conversation history with memory."""

    __tablename__ = "conversation_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String, index=True, nullable=False)
    query = Column(String, nullable=False)
    response = Column(JSON, nullable=False)
    score = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)


def init_db():
    """Initialize the database tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for getting database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
