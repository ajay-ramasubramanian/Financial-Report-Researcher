"""
Financial Report Analyzer - Configuration Module

Centralized configuration for SEC EDGAR API, ChromaDB, Ollama, and chunking parameters.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FILINGS_DIR = DATA_DIR / "filings"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"


@dataclass
class SECConfig:
    """SEC EDGAR API configuration."""
    user_agent: str = field(
        default_factory=lambda: os.getenv(
            "SEC_USER_AGENT", 
            "FinanceAnalyzer contact@example.com"
        )
    )
    base_url: str = "https://www.sec.gov"
    rate_limit_delay: float = 0.1  # 10 requests per second max
    max_retries: int = 3
    filing_types: List[str] = field(default_factory=lambda: ["10-K", "10-Q"])


@dataclass
class OllamaConfig:
    """Ollama API configuration."""
    host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )
    embed_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_LLM_MODEL", "qwen3:8b-q4_K_M")
    )
    # Generation parameters
    temperature: float = 0.1
    max_tokens: int = 1024
    embed_dimensions: int = 768  # nomic-embed-text dimensions


@dataclass
class ChunkingConfig:
    """Document chunking configuration."""
    # Minimum chunk lengths by section type
    min_chunk_length_mda: int = 2500  # Item 7 - MD&A
    min_chunk_length_item1: int = 1500  # Item 1 - Business
    min_chunk_length_default: int = 1000
    
    # Section identifiers for 10-K/10-Q filings
    sections: List[str] = field(default_factory=lambda: [
        "Item 1",    # Business
        "Item 1A",   # Risk Factors
        "Item 1B",   # Unresolved Staff Comments
        "Item 2",    # Properties
        "Item 3",    # Legal Proceedings
        "Item 4",    # Mine Safety Disclosures
        "Item 5",    # Market for Registrant's Common Equity
        "Item 6",    # Selected Financial Data (removed in 2021)
        "Item 7",    # MD&A
        "Item 7A",   # Quantitative and Qualitative Disclosures About Market Risk
        "Item 8",    # Financial Statements
        "Item 9",    # Changes in and Disagreements with Accountants
        "Item 9A",   # Controls and Procedures
        "Item 9B",   # Other Information
        "Item 10",   # Directors, Executive Officers and Corporate Governance
        "Item 11",   # Executive Compensation
        "Item 12",   # Security Ownership
        "Item 13",   # Certain Relationships and Related Transactions
        "Item 14",   # Principal Accountant Fees and Services
        "Item 15",   # Exhibits and Financial Statement Schedules
    ])


@dataclass
class ChromaConfig:
    """ChromaDB configuration."""
    persist_directory: str = field(
        default_factory=lambda: os.getenv(
            "CHROMA_DB_PATH", 
            str(CHROMA_DB_DIR)
        )
    )
    collection_prefix: str = "collection_"
    default_n_results: int = 5


@dataclass
class QueryConfig:
    """Query processing configuration."""
    default_top_k: int = 5
    max_top_k: int = 20
    context_delimiter: str = "\n\n---\n\n"
    
    system_prompt: str = """You are a financial analyst assistant. Use ONLY the provided SEC filing excerpts to answer the user's question. If the information is not in the context, state that clearly.

When providing answers:
- Cite specific numbers, dates, and figures from the filings
- Reference the filing type (10-K/10-Q) and period when relevant
- Be precise and factual
- If comparing periods, clearly distinguish between them"""


# Global configuration instances
sec_config = SECConfig()
ollama_config = OllamaConfig()
chunking_config = ChunkingConfig()
chroma_config = ChromaConfig()
query_config = QueryConfig()


def ensure_directories():
    """Ensure all required directories exist."""
    FILINGS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)


# Ensure directories on import
ensure_directories()
