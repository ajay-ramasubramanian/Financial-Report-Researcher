"""
Vector Store Module

Manages ChromaDB for storing and retrieving document embeddings with metadata filtering.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings

from .config import chroma_config, ollama_config
from .document_processor import DocumentChunk
from .embeddings import OllamaEmbeddings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB vector store for SEC filing chunks.
    
    Organizes collections by ticker symbol and supports metadata filtering
    for dates, filing types, and sections.
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Path to ChromaDB persistence directory.
        """
        self.persist_directory = persist_directory or chroma_config.persist_directory
        
        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embeddings
        self.embedder = OllamaEmbeddings()
    
    def _get_collection_name(self, ticker: str) -> str:
        """
        Generate collection name for a ticker.
        
        Args:
            ticker: Company ticker symbol.
            
        Returns:
            Collection name.
        """
        return f"{chroma_config.collection_prefix}{ticker.lower()}"
    
    def get_or_create_collection(self, ticker: str) -> chromadb.Collection:
        """
        Get or create a collection for a ticker.
        
        Args:
            ticker: Company ticker symbol.
            
        Returns:
            ChromaDB collection.
        """
        collection_name = self._get_collection_name(ticker)
        
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
    
    def add_chunks(
        self, 
        chunks: List[DocumentChunk],
        batch_size: int = 50
    ) -> int:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of document chunks to add.
            batch_size: Number of chunks to process at once.
            
        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0
        
        # Group chunks by ticker
        chunks_by_ticker: Dict[str, List[DocumentChunk]] = {}
        for chunk in chunks:
            if chunk.ticker not in chunks_by_ticker:
                chunks_by_ticker[chunk.ticker] = []
            chunks_by_ticker[chunk.ticker].append(chunk)
        
        total_added = 0
        
        for ticker, ticker_chunks in chunks_by_ticker.items():
            collection = self.get_or_create_collection(ticker)
            
            # Process in batches
            for i in range(0, len(ticker_chunks), batch_size):
                batch = ticker_chunks[i:i + batch_size]
                
                # Generate embeddings
                texts = [chunk.content for chunk in batch]
                embeddings = self.embedder.embed_texts(texts)
                
                # Prepare data for ChromaDB
                ids = [chunk.get_id() for chunk in batch]
                metadatas = [chunk.to_metadata() for chunk in batch]
                
                # Add to collection
                try:
                    collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=texts,
                        metadatas=metadatas
                    )
                    total_added += len(batch)
                    logger.info(f"Added {len(batch)} chunks to {ticker} collection")
                except Exception as e:
                    logger.error(f"Error adding chunks to {ticker}: {e}")
        
        return total_added
    
    def query(
        self,
        ticker: str,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the vector store for similar chunks.
        
        Args:
            ticker: Company ticker symbol.
            query_text: Natural language query.
            n_results: Number of results to return.
            where: Metadata filter conditions.
            where_document: Document content filter conditions.
            
        Returns:
            Query results with documents, distances, and metadata.
        """
        collection = self.get_or_create_collection(ticker)
        
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query_text)
        
        # Build query parameters
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, collection.count()) if collection.count() > 0 else n_results,
            "include": ["documents", "metadatas", "distances"]
        }
        
        if where:
            query_params["where"] = where
        
        if where_document:
            query_params["where_document"] = where_document
        
        try:
            results = collection.query(**query_params)
            return results
        except Exception as e:
            logger.error(f"Query error for {ticker}: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    def query_multiple_tickers(
        self,
        tickers: List[str],
        query_text: str,
        n_results_per_ticker: int = 3
    ) -> Dict[str, Any]:
        """
        Query multiple ticker collections and combine results.
        
        Args:
            tickers: List of ticker symbols to query.
            query_text: Natural language query.
            n_results_per_ticker: Results per ticker.
            
        Returns:
            Combined query results.
        """
        all_documents = []
        all_metadatas = []
        all_distances = []
        
        for ticker in tickers:
            results = self.query(ticker, query_text, n_results_per_ticker)
            
            if results["documents"] and results["documents"][0]:
                all_documents.extend(results["documents"][0])
                all_metadatas.extend(results["metadatas"][0])
                all_distances.extend(results["distances"][0])
        
        # Sort by distance and return top results
        if all_documents:
            combined = sorted(
                zip(all_documents, all_metadatas, all_distances),
                key=lambda x: x[2]
            )
            
            return {
                "documents": [[x[0] for x in combined]],
                "metadatas": [[x[1] for x in combined]],
                "distances": [[x[2] for x in combined]]
            }
        
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    def get_collection_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get information about a collection.
        
        Args:
            ticker: Company ticker symbol.
            
        Returns:
            Collection metadata and count.
        """
        collection_name = self._get_collection_name(ticker)
        
        try:
            collection = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "count": collection.count(),
                "metadata": collection.metadata
            }
        except Exception:
            return {"name": collection_name, "count": 0, "exists": False}
    
    def list_collections(self) -> List[str]:
        """
        List all ticker collections in the store.
        
        Returns:
            List of ticker symbols with collections.
        """
        collections = self.client.list_collections()
        tickers = []
        
        for collection in collections:
            name = collection.name
            if name.startswith(chroma_config.collection_prefix):
                ticker = name[len(chroma_config.collection_prefix):].upper()
                tickers.append(ticker)
        
        return tickers
    
    def delete_collection(self, ticker: str) -> bool:
        """
        Delete a ticker's collection.
        
        Args:
            ticker: Company ticker symbol.
            
        Returns:
            True if deleted, False otherwise.
        """
        collection_name = self._get_collection_name(ticker)
        
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            return False
    
    def close(self):
        """Close resources."""
        self.embedder.close()


def create_metadata_filter(
    filing_type: Optional[str] = None,
    fiscal_year: Optional[str] = None,
    fiscal_quarter: Optional[str] = None,
    section: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Create a ChromaDB metadata filter.
    
    Args:
        filing_type: Filter by filing type (10-K or 10-Q).
        fiscal_year: Filter by fiscal year.
        fiscal_quarter: Filter by fiscal quarter.
        section: Filter by section name.
        
    Returns:
        Metadata filter dictionary or None.
    """
    conditions = []
    
    if filing_type:
        conditions.append({"filing_type": {"$eq": filing_type}})
    
    if fiscal_year:
        conditions.append({"fiscal_year": {"$eq": fiscal_year}})
    
    if fiscal_quarter:
        conditions.append({"fiscal_quarter": {"$eq": fiscal_quarter}})
    
    if section:
        conditions.append({"section": {"$contains": section}})
    
    if not conditions:
        return None
    
    if len(conditions) == 1:
        return conditions[0]
    
    return {"$and": conditions}


if __name__ == "__main__":
    # Test mode
    logging.basicConfig(level=logging.INFO)
    
    store = VectorStore()
    
    print(f"ChromaDB initialized at: {store.persist_directory}")
    print(f"Existing collections: {store.list_collections()}")
