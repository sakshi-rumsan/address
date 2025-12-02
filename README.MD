# Address Validation & Retrieval API

This project provides a FastAPI-based API for address validation and retrieval using Retrieval-Augmented Generation (RAG) and Qdrant vector search. It parses and returns addresses in the Mplify 150 Installation Place Fielded Address Representation format.

## Features

- Query partial addresses and receive full, standardized addresses
- Uses Qdrant vector database for fast similarity search
- Embedding generation via Ollama/OpenAI models
- LLM-based address parsing and formatting
- Health check endpoint

## Project Structure
python -m spacy download en_core_web_sm
```
app/
  config.py         # Loads settings from .env
  main.py           # FastAPI app entry point
  schemas.py        # Pydantic models for request/response
  routes/
    health_route.py # Health check endpoint
    query_route.py  # RAG query endpoint
  services/
    embedding_service.py # Embedding logic
    monitor_service.py   # Monitoring utilities
    qdrant_service.py    # Qdrant vector search
    rag_service.py       # RAG pipeline
.env                # Environment variables
qdrant_data/        # Qdrant storage (ignored by git)
venv/               # Python virtual environment (ignored by git)
```

## Setup

1. **Clone the repository**
2. **Create and activate a Python virtual environment**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Windows
   # or
   source venv/bin/activate      # Linux/Mac
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure `.env` file**
   - Set model names, API keys, Qdrant host, etc.
5. **Start Qdrant server**
   ```bash
   docker run -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
   ```
6. **Run the API server**
   ```bash
   uvicorn app.main:app --reload
   ```

- Update `.env` for your environment and model settings


## API USAGE

# How to Use Conversation Memory with SQLite

## Overview
Persistent conversation memory has been added to your RAG system using SQLite database.

## Features Added

### 1. Database Storage (`app/database.py`)
- SQLite database: `conversation_history.db`
- Stores: session_id, query, response, score, timestamp
- Auto-initializes on app startup

### 2. Memory Functions in RAG Service (`app/services/rag_service.py`)
- `get_conversation_history(session_id, limit)` - Retrieve past conversations
- `save_to_history(session_id, query, response, score)` - Save new interactions
- `rag_address_query()` - Now accepts optional `session_id` parameter

### 3. API Changes

#### Updated Request Schema
```json
{
  "query": "123 Main Street",
  "top_k": 3,
  "session_id": "user-abc-123"  // NEW: Optional
}
```

#### New History Endpoints
- `GET /api/v1/history/{session_id}` - Get conversation history
- `DELETE /api/v1/history/{session_id}` - Clear session history

## Usage Examples

### 1. Query with Memory
```bash
curl -X POST "http://localhost:8000/api/v1/query-address" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Wellington Street",
    "top_k": 2,
    "session_id": "user-123"
  }'
```

### 2. Get Conversation History
```bash
curl "http://localhost:8000/api/v1/history/user-123?limit=10"
```

### 3. Clear Session History
```bash
curl -X DELETE "http://localhost:8000/api/v1/history/user-123"
```

## How It Works

1. **First Request**: User sends query with `session_id`
   - System searches Qdrant for addresses
   - Saves query + top result to SQLite
   
2. **Subsequent Requests**: Same `session_id`
   - System retrieves last 3 conversations
   - Uses context for better understanding (future enhancement)
   - Saves new interaction

3. **History Retrieval**: 
   - View all past queries and responses
   - Helpful for debugging or user review

## Database Location
- File: `./conversation_history.db` (in project root)
- Auto-created on first run

## Benefits
- Track user interactions
- Improve responses with context
- Debug and analyze query patterns
- No external services needed

## Optional: Environment Variable
Add to `.env` if you want custom database path:
```
DATABASE_URL=sqlite:///./my_custom_db.db
```