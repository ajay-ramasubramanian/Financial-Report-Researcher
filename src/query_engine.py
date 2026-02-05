"""
Query Engine Module

Orchestrates the RAG pipeline: query parsing, retrieval, context assembly, and generation.
"""

import re
import logging
from typing import List, Optional, Dict, Any, Generator
from dataclasses import dataclass

from .config import query_config
from .vector_store import VectorStore, create_metadata_filter
from .llm_interface import OllamaLLM

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result from a RAG query."""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    ticker: str
    chunks_used: int


class QueryEngine:
    """
    RAG query processing engine.
    
    Handles natural language queries by retrieving relevant context
    from the vector store and generating answers using the LLM.
    """
    
    def __init__(self):
        """Initialize the query engine."""
        self.vector_store = VectorStore()
        self.llm = OllamaLLM()
    
    def parse_query_metadata(self, query: str) -> Dict[str, Any]:
        """
        Extract metadata filters from a natural language query.
        
        Identifies mentions of:
        - Filing types (10-K, 10-Q)
        - Years (2023, 2024, etc.)
        - Quarters (Q1, Q2, Q3, Q4)
        - Sections (MD&A, Risk Factors, etc.)
        
        Args:
            query: Natural language query.
            
        Returns:
            Dictionary of extracted metadata filters.
        """
        filters = {}
        query_lower = query.lower()
        
        # Extract filing type
        if "10-k" in query_lower or "annual" in query_lower:
            filters["filing_type"] = "10-K"
        elif "10-q" in query_lower or "quarterly" in query_lower:
            filters["filing_type"] = "10-Q"
        
        # Extract years
        year_matches = re.findall(r'\b(20\d{2})\b', query)
        if year_matches:
            filters["years"] = list(set(year_matches))
        
        # Extract quarters
        quarter_matches = re.findall(r'\b[qQ]([1-4])\b', query)
        if quarter_matches:
            filters["quarters"] = [f"Q{q}" for q in set(quarter_matches)]
        
        # Extract sections
        section_keywords = {
            "risk": "Risk Factors",
            "mda": "MD&A",
            "md&a": "MD&A",
            "management discussion": "MD&A",
            "business": "Business",
            "financial statement": "Financial Statements",
            "revenue": "MD&A",  # Revenue discussion usually in MD&A
            "income": "MD&A",
            "earnings": "MD&A",
        }
        
        for keyword, section in section_keywords.items():
            if keyword in query_lower:
                filters["section_hint"] = section
                break
        
        return filters
    
    def retrieve_context(
        self,
        ticker: str,
        query: str,
        n_results: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context chunks from the vector store.
        
        Args:
            ticker: Company ticker symbol.
            query: Natural language query.
            n_results: Number of chunks to retrieve.
            metadata_filters: Optional metadata filters.
            
        Returns:
            List of context chunks with metadata.
        """
        # Build ChromaDB where filter from metadata
        where_filter = None
        
        if metadata_filters:
            conditions = []
            
            if "filing_type" in metadata_filters:
                conditions.append({"filing_type": {"$eq": metadata_filters["filing_type"]}})
            
            if "years" in metadata_filters and len(metadata_filters["years"]) == 1:
                conditions.append({"fiscal_year": {"$eq": metadata_filters["years"][0]}})
            
            if "quarters" in metadata_filters and len(metadata_filters["quarters"]) == 1:
                conditions.append({"fiscal_quarter": {"$eq": metadata_filters["quarters"][0]}})
            
            if conditions:
                where_filter = {"$and": conditions} if len(conditions) > 1 else conditions[0]
        
        # Query the vector store
        results = self.vector_store.query(
            ticker=ticker,
            query_text=query,
            n_results=n_results,
            where=where_filter
        )
        
        # Format results
        context_chunks = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                context_chunks.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0
                })
        
        return context_chunks
    
    def assemble_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Assemble retrieved chunks into a context string.
        
        Args:
            chunks: List of context chunks.
            
        Returns:
            Formatted context string.
        """
        if not chunks:
            return "No relevant context found in the SEC filings."
        
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            
            # Create a header for the chunk
            header = f"[Source {i}: {metadata.get('filing_type', 'Filing')} "
            header += f"({metadata.get('fiscal_year', '')})"
            if metadata.get('fiscal_quarter'):
                header += f" {metadata.get('fiscal_quarter')}"
            header += f" - {metadata.get('section', 'Document')}]"
            
            context_parts.append(f"{header}\n{chunk['content']}")
        
        return query_config.context_delimiter.join(context_parts)
    
    def build_prompt(self, context: str, query: str) -> str:
        """
        Build the full prompt for the LLM.
        
        Args:
            context: Assembled context string.
            query: User's question.
            
        Returns:
            Complete prompt string.
        """
        return f"""Context from SEC filings:

{context}

Question: {query}

Provide a concise, accurate answer with specific numbers and dates where available. If the information is not available in the provided context, state that clearly."""
    
    def query(
        self,
        ticker: str,
        query: str,
        top_k: int = 5,
        stream: bool = False
    ) -> QueryResult:
        """
        Execute a RAG query.
        
        Args:
            ticker: Company ticker symbol.
            query: Natural language query.
            top_k: Number of context chunks to retrieve.
            stream: Whether to stream the response.
            
        Returns:
            QueryResult with answer and sources.
        """
        logger.info(f"Processing query for {ticker}: {query[:100]}...")
        
        # Parse query for metadata filters
        metadata_filters = self.parse_query_metadata(query)
        logger.info(f"Extracted filters: {metadata_filters}")
        
        # Handle comparative queries (multiple years)
        if "compare" in query.lower() or "vs" in query.lower() or "versus" in query.lower():
            # For comparisons, get more chunks
            top_k = max(top_k, 10)
        
        # Retrieve context
        chunks = self.retrieve_context(
            ticker=ticker,
            query=query,
            n_results=top_k,
            metadata_filters=metadata_filters
        )
        
        if not chunks:
            return QueryResult(
                answer="No relevant information found in the SEC filings for this query. Please ensure the filings are ingested for this ticker.",
                sources=[],
                query=query,
                ticker=ticker,
                chunks_used=0
            )
        
        # Assemble context
        context = self.assemble_context(chunks)
        
        # Build prompt
        prompt = self.build_prompt(context, query)
        
        # Generate answer
        answer = self.llm.generate(
            prompt=prompt,
            system_prompt=query_config.system_prompt,
            stream=stream
        )
        
        # Prepare sources
        sources = [
            {
                "filing_type": c["metadata"].get("filing_type"),
                "fiscal_year": c["metadata"].get("fiscal_year"),
                "fiscal_quarter": c["metadata"].get("fiscal_quarter"),
                "section": c["metadata"].get("section"),
                "relevance_score": 1 - c.get("distance", 0)  # Convert distance to similarity
            }
            for c in chunks
        ]
        
        return QueryResult(
            answer=answer,
            sources=sources,
            query=query,
            ticker=ticker,
            chunks_used=len(chunks)
        )
    
    def query_stream(
        self,
        ticker: str,
        query: str,
        top_k: int = 5
    ) -> Generator[str, None, None]:
        """
        Execute a RAG query with streaming response.
        
        Args:
            ticker: Company ticker symbol.
            query: Natural language query.
            top_k: Number of context chunks to retrieve.
            
        Yields:
            Generated text chunks.
        """
        logger.info(f"Processing streaming query for {ticker}: {query[:100]}...")
        
        # Parse query for metadata filters
        metadata_filters = self.parse_query_metadata(query)
        
        # Retrieve context
        chunks = self.retrieve_context(
            ticker=ticker,
            query=query,
            n_results=top_k,
            metadata_filters=metadata_filters
        )
        
        if not chunks:
            yield "No relevant information found in the SEC filings for this query."
            return
        
        # Assemble context
        context = self.assemble_context(chunks)
        
        # Build prompt
        prompt = self.build_prompt(context, query)
        
        # Stream generation
        for chunk in self.llm.generate_stream(
            prompt=prompt,
            system_prompt=query_config.system_prompt
        ):
            yield chunk
    
    def get_collection_status(self, ticker: str) -> Dict[str, Any]:
        """
        Get the status of a ticker's collection.
        
        Args:
            ticker: Company ticker symbol.
            
        Returns:
            Collection info including chunk count.
        """
        return self.vector_store.get_collection_info(ticker)
    
    def close(self):
        """Close resources."""
        self.vector_store.close()
        self.llm.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Test mode
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 2:
        ticker = sys.argv[1]
        query = " ".join(sys.argv[2:])
        
        engine = QueryEngine()
        result = engine.query(ticker, query)
        
        print(f"\n{'='*60}")
        print(f"Query: {result.query}")
        print(f"Ticker: {result.ticker}")
        print(f"Chunks used: {result.chunks_used}")
        print(f"{'='*60}")
        print(f"\nAnswer:\n{result.answer}")
        print(f"\n{'='*60}")
        print("Sources:")
        for s in result.sources[:3]:
            print(f"  - {s['filing_type']} {s['fiscal_year']} {s.get('section', '')}")
    else:
        print("Usage: python -m src.query_engine TICKER 'your question'")
