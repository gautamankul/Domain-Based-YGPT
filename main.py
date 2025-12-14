import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from api import initialize_bot, YILGPT
from utils.logger import get_logger

# --- Initialization ---
app = FastAPI(
    title="YILGPT RAG API",
    description="API for Document-Specific Question Answering using RAG and Intent Filtering."
)
logger = get_logger("fastapi_yil_gpt")

# Global state to hold the initialized bot
yil_gpt_bot: Optional[YILGPT] = None


# --- Request/Response Schemas ---

class QueryRequest(BaseModel):
    """Schema for the incoming JSON query."""
    question: str


class QueryResponse(BaseModel):
    """Schema for the outgoing JSON response."""
    answer: str


# --- Startup Event: Initialize the RAG Bot ---

@app.on_event("startup")
async def startup_event():
    """
    Called when the FastAPI application starts up.
    Initializes the YILGPT RAG pipeline. This runs synchronously 
    to ensure the embedding model and vector database are ready 
    before serving requests.
    """
    global yil_gpt_bot
    logger.info("Application startup initiated. Initializing YILGPT...")

    try:
        # Call the synchronous initialization function from yil_gpt.api
        # Note: If initialization is very long, you might need a background task, 
        # but for simplicity, we keep it synchronous as it was in the notebook.
        yil_gpt_bot = initialize_bot()
        logger.info("YILGPT Initialization complete. API is ready to serve.")
    except Exception as e:
        logger.error(f"FATAL ERROR during YILGPT initialization: {e}")
        # In a real app, you might want to stop the server here, 
        # but for development, we log and proceed with a null bot.
        # This will trigger the 503 error on the query endpoint.


# --- API Endpoint ---

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def rag_query(request: QueryRequest):
    """
    Receives a user question, processes it through the RAG pipeline, 
    and returns the generated answer.
    """
    if yil_gpt_bot is None:
        raise HTTPException(
            status_code=503,
            detail="YILGPT RAG system is not initialized or failed to start."
        )

    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )

    logger.info(f"Received query: '{question}'")

    try:
        # Call the core RAG function
        answer = yil_gpt_bot.query(question)

        # Clean up the output slightly before sending back
        cleaned_answer = answer.strip()

        return QueryResponse(answer=cleaned_answer)

    except Exception as e:
        logger.error(f"Error processing query '{question}': {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal error during RAG processing."
        )


# --- Health Check Endpoint ---

@app.get("/health", tags=["System"])
def health_check():
    """Checks the health of the application and the RAG bot."""
    status = "READY" if yil_gpt_bot is not None else "INITIALIZING"
    return {"status": status, "rag_bot_initialized": yil_gpt_bot is not None}
