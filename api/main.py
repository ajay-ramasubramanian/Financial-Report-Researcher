"""
Financial Report Analyzer - FastAPI Application

RAG system for analyzing SEC 10-K and 10-Q filings.
"""

import logging
from datetime import datetime
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceInfo,
    HealthStatus,
    CollectionInfo,
    CollectionsResponse,
)

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.config import ollama_config, chroma_config
from src.sec_downloader import SECDownloader
from src.document_processor import DocumentProcessor
from src.embeddings import OllamaEmbeddings
from src.vector_store import VectorStore
from src.llm_interface import OllamaLLM
from src.query_engine import QueryEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
vector_store: VectorStore = None
query_engine: QueryEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan handler."""
    global vector_store, query_engine
    
    logger.info("Starting Financial Report Analyzer...")
    
    # Initialize components
    vector_store = VectorStore()
    query_engine = QueryEngine()
    
    logger.info("Components initialized successfully")
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")
    if query_engine:
        query_engine.close()


app = FastAPI(
    title="Financial Report Analyzer",
    description="RAG system for SEC 10-K and 10-Q filing analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthStatus)
async def health_check():
    """
    Check system health status.
    
    Returns Ollama availability and ChromaDB status.
    """
    # Check Ollama
    embedder = OllamaEmbeddings()
    llm = OllamaLLM()
    
    ollama_available = embedder.health_check() and llm.health_check()
    
    # Get collections count
    collections = vector_store.list_collections() if vector_store else []
    
    return HealthStatus(
        status="healthy" if ollama_available else "degraded",
        ollama_available=ollama_available,
        ollama_llm_model=ollama_config.llm_model,
        ollama_embed_model=ollama_config.embed_model,
        chromadb_path=chroma_config.persist_directory,
        collections_count=len(collections)
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_filings(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Download and ingest SEC filings for a company.
    
    Downloads 10-K and 10-Q filings within a 2-year lookback window,
    processes them into semantic chunks, and stores in ChromaDB.
    """
    ticker = request.ticker.upper()
    reference_date = datetime.combine(request.reference_date, datetime.min.time())
    
    logger.info(f"Starting ingestion for {ticker} (reference: {request.reference_date})")
    
    try:
        # Download filings
        downloader = SECDownloader()
        filings = downloader.download_filings(
            ticker=ticker,
            reference_date=reference_date,
            filing_types=request.filing_types
        )
        
        if not filings:
            return IngestResponse(
                ticker=ticker,
                filings_downloaded=0,
                chunks_created=0,
                status="warning",
                message="No filings found for the specified date range"
            )
        
        # Process documents
        processor = DocumentProcessor()
        all_chunks = []
        
        for filing in filings:
            chunks = processor.process_filing(filing)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            return IngestResponse(
                ticker=ticker,
                filings_downloaded=len(filings),
                chunks_created=0,
                status="warning",
                message="Filings downloaded but no content could be extracted"
            )
        
        # Store in vector database
        chunks_added = vector_store.add_chunks(all_chunks)
        
        return IngestResponse(
            ticker=ticker,
            filings_downloaded=len(filings),
            chunks_created=chunks_added,
            status="success",
            message=f"Successfully ingested {len(filings)} filings with {chunks_added} chunks"
        )
        
    except Exception as e:
        logger.error(f"Ingestion error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_filings(request: QueryRequest):
    """
    Query the RAG system with a natural language question.
    
    Retrieves relevant context from SEC filings and generates an answer.
    """
    ticker = request.ticker.upper()
    
    # Check if collection exists
    collection_info = query_engine.get_collection_status(ticker)
    if collection_info.get("count", 0) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No filings found for {ticker}. Please ingest filings first."
        )
    
    try:
        # Execute query
        result = query_engine.query(
            ticker=ticker,
            query=request.query,
            top_k=request.top_k
        )
        
        return QueryResponse(
            answer=result.answer,
            sources=[SourceInfo(**s) for s in result.sources],
            query=result.query,
            ticker=result.ticker,
            chunks_used=result.chunks_used
        )
        
    except Exception as e:
        logger.error(f"Query error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_filings_stream(request: QueryRequest):
    """
    Query with streaming response.
    
    Returns a streaming response for real-time answer generation.
    """
    ticker = request.ticker.upper()
    
    # Check if collection exists
    collection_info = query_engine.get_collection_status(ticker)
    if collection_info.get("count", 0) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No filings found for {ticker}. Please ingest filings first."
        )
    
    def generate():
        for chunk in query_engine.query_stream(
            ticker=ticker,
            query=request.query,
            top_k=request.top_k
        ):
            yield chunk
    
    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/collections", response_model=CollectionsResponse)
async def list_collections():
    """
    List all available ticker collections.
    
    Returns information about ingested tickers and their chunk counts.
    """
    tickers = vector_store.list_collections()
    
    collections = []
    for ticker in tickers:
        info = vector_store.get_collection_info(ticker)
        collections.append(CollectionInfo(
            ticker=ticker,
            chunk_count=info.get("count", 0),
            exists=True
        ))
    
    return CollectionsResponse(
        collections=collections,
        total=len(collections)
    )


@app.get("/collections/{ticker}", response_model=CollectionInfo)
async def get_collection(ticker: str):
    """
    Get information about a specific ticker collection.
    """
    ticker = ticker.upper()
    info = vector_store.get_collection_info(ticker)
    
    return CollectionInfo(
        ticker=ticker,
        chunk_count=info.get("count", 0),
        exists=info.get("count", 0) > 0
    )


@app.delete("/collections/{ticker}")
async def delete_collection(ticker: str):
    """
    Delete a ticker's collection.
    
    Removes all stored chunks for the specified ticker.
    """
    ticker = ticker.upper()
    
    if vector_store.delete_collection(ticker):
        return {"status": "success", "message": f"Deleted collection for {ticker}"}
    else:
        raise HTTPException(status_code=404, detail=f"Collection for {ticker} not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
