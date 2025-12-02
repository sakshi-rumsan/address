# app/main.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from app.routes.query_route import router as rag_router
from app.routes.history_route import router as history_router
from app.config import settings
from app.database import init_db
import logging


logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
)


# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized successfully")


# Include routers
app.include_router(rag_router, prefix=settings.api_v1_prefix)
app.include_router(history_router, prefix=settings.api_v1_prefix)


# Exception handlers (order matters: specific first, general last)
@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    logger.warning(f"Request validation error for {request.url}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )


@app.exception_handler(ResponseValidationError)
async def response_validation_exception_handler(
    request: Request, exc: ResponseValidationError
):
    logger.warning(f"Response validation error for {request.url}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )


# Global handler for unhandled exceptions (500 only - after HTTPException)
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception for {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error. Please try again later.",
            "message": "An unexpected error occurred on the server.",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
