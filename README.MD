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

## License

MIT
