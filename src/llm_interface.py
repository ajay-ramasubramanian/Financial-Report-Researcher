"""
LLM Interface Module

Interfaces with Ollama for text generation using local LLMs.
"""

import logging
from typing import Optional, Generator, Dict, Any
import httpx

from .config import ollama_config

logger = logging.getLogger(__name__)


class OllamaLLM:
    """
    Interface to Ollama for text generation.
    
    Supports streaming responses and configurable generation parameters.
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize the LLM client.
        
        Args:
            host: Ollama API host URL.
            model: LLM model name.
            temperature: Generation temperature (0-1).
            max_tokens: Maximum tokens to generate.
        """
        self.host = host or ollama_config.host
        self.model = model or ollama_config.llm_model
        self.temperature = temperature if temperature is not None else ollama_config.temperature
        self.max_tokens = max_tokens or ollama_config.max_tokens
        
        self.generate_url = f"{self.host}/api/generate"
        self.client = httpx.Client(timeout=120.0)  # Longer timeout for generation
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            stream: Whether to stream the response.
            
        Returns:
            Generated text.
        """
        if stream:
            return "".join(self.generate_stream(prompt, system_prompt))
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = self.client.post(self.generate_url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Stream text generation from a prompt.
        
        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            
        Yields:
            Generated text chunks.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            with self.client.stream("POST", self.generate_url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done", False):
                            break
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            raise
    
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
            model_names = [m.get("name", "") for m in models]
            
            # Check if model is available (exact match or prefix match)
            model_base = self.model.split(":")[0]
            return any(model_base in name for name in model_names)
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return False
    
    def list_models(self) -> list:
        """
        List available Ollama models.
        
        Returns:
            List of model names.
        """
        try:
            response = self.client.get(f"{self.host}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            return [m.get("name", "") for m in models]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Test mode
    logging.basicConfig(level=logging.INFO)
    
    llm = OllamaLLM()
    
    print(f"Available models: {llm.list_models()}")
    
    if llm.health_check():
        print(f"✓ Ollama is running with {llm.model}")
        
        # Test generation
        response = llm.generate(
            "What is the purpose of a 10-K filing?",
            system_prompt="You are a helpful financial assistant. Be concise."
        )
        print(f"\nTest response:\n{response}")
    else:
        print(f"✗ Ollama not available or model {llm.model} not found")
        print(f"  Run: ollama pull {llm.model}")
