"""
API Schemas - Pydantic models for request/response validation.
"""

from datetime import date, datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Request to ingest SEC filings for a company."""
    ticker: str = Field(
        ..., 
        description="Company ticker symbol (e.g., 'AAPL') or CIK number",
        min_length=1,
        max_length=20
    )
    reference_date: date = Field(
        ..., 
        description="Reference date for 2-year lookback window (YYYY-MM-DD)"
    )
    filing_types: Optional[List[str]] = Field(
        default=["10-K", "10-Q"],
        description="Types of filings to download"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "reference_date": "2025-01-15",
                "filing_types": ["10-K", "10-Q"]
            }
        }


class IngestResponse(BaseModel):
    """Response from filing ingestion."""
    ticker: str
    filings_downloaded: int
    chunks_created: int
    status: str
    message: str


class QueryRequest(BaseModel):
    """Request to query the RAG system."""
    query: str = Field(
        ..., 
        description="Natural language query about the SEC filings",
        min_length=3
    )
    ticker: str = Field(
        ..., 
        description="Company ticker symbol to query"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of context chunks to retrieve"
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What was the revenue growth in Q2 2024 compared to Q2 2023?",
                "ticker": "AAPL",
                "top_k": 5,
                "stream": False
            }
        }


class SourceInfo(BaseModel):
    """Information about a source chunk used in the response."""
    filing_type: Optional[str]
    fiscal_year: Optional[str]
    fiscal_quarter: Optional[str]
    section: Optional[str]
    relevance_score: float


class QueryResponse(BaseModel):
    """Response from a RAG query."""
    answer: str
    sources: List[SourceInfo]
    query: str
    ticker: str
    chunks_used: int


class HealthStatus(BaseModel):
    """System health status."""
    status: str
    ollama_available: bool
    ollama_llm_model: str
    ollama_embed_model: str
    chromadb_path: str
    collections_count: int


class CollectionInfo(BaseModel):
    """Information about a ticker collection."""
    ticker: str
    chunk_count: int
    exists: bool


class CollectionsResponse(BaseModel):
    """Response listing all collections."""
    collections: List[CollectionInfo]
    total: int
