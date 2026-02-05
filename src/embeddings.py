"""
Embeddings Layer

Generates embeddings for document chunks using Ollama's nomic-embed-text model.
"""

import logging
from typing import List, Optional
import httpx

from .config import ollama_config

logger = logging.getLogger(__name__)


class OllamaEmbeddings:
    """
    Generates embeddings using Ollama's embedding models.
    
    Uses nomic-embed-text by default (768 dimensions).
    """
    
    def __init__(
        self, 
        host: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize the embeddings client.
        
        Args:
            host: Ollama API host URL.
            model: Embedding model name.
        """
        self.host = host or ollama_config.host
        self.model = model or ollama_config.embed_model
        self.embed_url = f"{self.host}/api/embeddings"
        self.client = httpx.Client(timeout=60.0)
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed.
            
        Returns:
            List of floats representing the embedding vector.
        """
        try:
            response = self.client.post(
                self.embed_url,
                json={
                    "model": self.model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            result = response.json()
            return result.get("embedding", [])
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        embeddings = []
        for i, text in enumerate(texts):
            try:
                embedding = self.embed_text(text)
                embeddings.append(embedding)
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated embeddings for {i + 1}/{len(texts)} chunks")
            except Exception as e:
                logger.error(f"Error embedding text {i}: {e}")
                # Return empty embedding on error
                embeddings.append([0.0] * ollama_config.embed_dimensions)
        
        return embeddings
    
    def health_check(self) -> bool:
        """
        Check if Ollama is running and the model is available.
        
        Returns:
            True if healthy, False otherwise.
        """
        try:
            response = self.client.get(f"{self.host}/api/tags")
            if response.status_code != 200:
                return False
            
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            
            return self.model.split(":")[0] in model_names
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Convenience function to generate embeddings.
    
    Args:
        texts: List of texts to embed.
        
    Returns:
        List of embedding vectors.
    """
    with OllamaEmbeddings() as embedder:
        return embedder.embed_texts(texts)


if __name__ == "__main__":
    # Test mode
    logging.basicConfig(level=logging.INFO)
    
    embedder = OllamaEmbeddings()
    
    # Health check
    if embedder.health_check():
        print(f"✓ Ollama is running with {embedder.model}")
        
        # Test embedding
        test_text = "Apple Inc. reported quarterly revenue of $94.9 billion."
        embedding = embedder.embed_text(test_text)
        print(f"✓ Generated embedding with {len(embedding)} dimensions")
    else:
        print(f"✗ Ollama not available or model {embedder.model} not found")
        print(f"  Run: ollama pull {embedder.model}")
