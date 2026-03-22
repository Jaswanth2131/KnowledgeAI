import sys
import logging
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# We purposefully reuse the setup block from your query.py
from query import get_rag_chain

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global variable to hold the initialized RAG chain in memory
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager.
    Loads ChromaDB and the RAG pipeline ONCE at application startup.
    """
    logger.info("Starting up FastAPI background services...")
    try:
        # Load the heavy pipeline globally, once.
        app_state["rag_chain"] = get_rag_chain()
        logger.info("Successfully initialized RAG pipeline into memory.")
    except Exception as e:
        logger.error(f"Failed to load RAG pipeline: {e}")
        logger.error("System might not be able to answer queries. Ensure 'ingest.py' was successfully run first.")
    
    yield
    
    # Server exit cleanup
    app_state.clear()
    logger.info("Shutting down FastAPI services...")

app = FastAPI(
    title="Multi-Document AI API",
    description="Production-ready asynchronous FastAPI server for our RAG pipeline.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Pydantic Data Models ---
class QueryRequest(BaseModel):
    question: str = Field(..., description="The user's question to ask the AI", example="What is artificial intelligence?")

class SourceDocument(BaseModel):
    filename: str
    page: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]

# --- Endpoints ---
@app.post("/query", response_model=QueryResponse)
async def query_model(req: QueryRequest):
    """
    Endpoint to query the ChromaDB vector database using the RAG model.
    """
    rag_chain = app_state.get("rag_chain")
    
    if rag_chain is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The RAG pipeline is not initialized. Ensure ChromaDB embeddings exist and Ollama is running."
        )

    try:
        # ainvoke allows for asynchronous execution (non-blocking for production APIs)
        response = await rag_chain.ainvoke({"input": req.question})
        
        answer = response.get("answer", "I don't know")
        raw_sources = response.get("context", [])
        
        # Parse strictly unique sources
        parsed_sources = []
        seen = set()
        
        for doc in raw_sources:
            filename = doc.metadata.get("filename", "unknown file")
            page = str(doc.metadata.get("page", "?"))
            
            identifier = f"{filename}_{page}"
            if identifier not in seen:
                seen.add(identifier)
                parsed_sources.append(SourceDocument(filename=filename, page=page))
                
        return QueryResponse(answer=answer, sources=parsed_sources)

    except Exception as e:
        err_msg = str(e)
        logger.error(f"Error executing query: {err_msg}")
        
        if "actively refused" in err_msg or "Connection refused" in err_msg:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Could not connect to the local Ollama LLM. Make sure 'ollama serve' is running."
            )
            
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred during query execution: {err_msg}"
        )

# Root sanity check endpoint
@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Multi-Document AI API is running"}
