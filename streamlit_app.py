"""
Financial Report Analyzer - Streamlit GUI

Interactive interface for SEC filing ingestion, querying, and chunk monitoring.
"""

import streamlit as st
import requests
from datetime import date, datetime
import json
import pandas as pd

# API Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Financial Report Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stAlert {
        margin-top: 1rem;
    }
    .chunk-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .source-badge {
        background-color: #1f77b4;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if the API is available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}


def get_collections():
    """Get list of ingested collections."""
    try:
        response = requests.get(f"{API_URL}/collections", timeout=10)
        if response.status_code == 200:
            return response.json()
        return {"collections": [], "total": 0}
    except Exception:
        return {"collections": [], "total": 0}


def ingest_filings(ticker: str, reference_date: date, filing_types: list):
    """Ingest SEC filings for a ticker."""
    try:
        response = requests.post(
            f"{API_URL}/ingest",
            json={
                "ticker": ticker,
                "reference_date": reference_date.isoformat(),
                "filing_types": filing_types
            },
            timeout=300  # Long timeout for downloading
        )
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def query_filings(ticker: str, query: str, top_k: int):
    """Query the RAG system."""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={
                "ticker": ticker,
                "query": query,
                "top_k": top_k
            },
            timeout=120
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# Sidebar
with st.sidebar:
    st.title("📊 Finance RAG")
    st.markdown("---")
    
    # Health check
    is_healthy, health_data = check_api_health()
    
    if is_healthy:
        st.success("✓ API Connected")
        st.caption(f"LLM: {health_data.get('ollama_llm_model', 'N/A')}")
        st.caption(f"Embeddings: {health_data.get('ollama_embed_model', 'N/A')}")
    else:
        st.error("✗ API Unavailable")
        st.caption("Start the API server first:")
        st.code("uvicorn api.main:app --port 8000")
    
    st.markdown("---")
    
    # Collections info
    collections = get_collections()
    st.subheader("📁 Ingested Tickers")
    
    if collections["total"] > 0:
        for col in collections["collections"]:
            st.markdown(f"**{col['ticker']}** - {col['chunk_count']} chunks")
    else:
        st.caption("No filings ingested yet")
    
    st.markdown("---")
    st.caption("Built with Streamlit + FastAPI")

# Main content
st.title("📈 Financial Report Analyzer")
st.markdown("Analyze SEC 10-K and 10-Q filings using RAG")

# Tabs for different functions
tab1, tab2, tab3 = st.tabs(["🔍 Query", "📥 Ingest", "📊 Chunk Monitor"])

# Query Tab
with tab1:
    st.header("Query SEC Filings")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query_ticker = st.text_input(
            "Ticker Symbol",
            value="AAPL",
            placeholder="e.g., AAPL, MSFT, GOOGL",
            key="query_ticker"
        )
    
    with col2:
        top_k = st.slider("Context Chunks", 3, 15, 5, key="top_k")
    
    query_text = st.text_area(
        "Your Question",
        placeholder="e.g., What was the revenue in fiscal year 2024? Compare Q2 2024 to Q2 2023...",
        height=100,
        key="query_text"
    )
    
    if st.button("🔍 Ask Question", type="primary", key="query_btn"):
        if query_text.strip():
            with st.spinner("Analyzing filings..."):
                result = query_filings(query_ticker.upper(), query_text, top_k)
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
            elif "detail" in result:
                st.warning(result["detail"])
            else:
                # Display answer
                st.subheader("💡 Answer")
                st.markdown(result.get("answer", "No answer generated"))
                
                # Store for chunk monitoring
                st.session_state["last_query_result"] = result
                
                # Display sources summary
                st.subheader("📚 Sources Used")
                sources = result.get("sources", [])
                
                if sources:
                    source_data = []
                    for s in sources:
                        source_data.append({
                            "Filing": s.get("filing_type", "N/A"),
                            "Year": s.get("fiscal_year", "N/A"),
                            "Quarter": s.get("fiscal_quarter", "") or "Annual",
                            "Section": s.get("section", "N/A"),
                            "Relevance": f"{s.get('relevance_score', 0):.1%}"
                        })
                    
                    df = pd.DataFrame(source_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                
                st.info(f"📊 Used {result.get('chunks_used', 0)} context chunks for this response")
        else:
            st.warning("Please enter a question")

# Ingest Tab
with tab2:
    st.header("Ingest SEC Filings")
    st.markdown("Download and process 10-K and 10-Q filings from SEC EDGAR")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ingest_ticker = st.text_input(
            "Ticker Symbol",
            value="AAPL",
            placeholder="e.g., AAPL, MSFT, GOOGL",
            key="ingest_ticker"
        )
    
    with col2:
        reference_date = st.date_input(
            "Reference Date",
            value=date.today(),
            help="System will download filings from 2 years before this date",
            key="ref_date"
        )
    
    filing_types = st.multiselect(
        "Filing Types",
        options=["10-K", "10-Q"],
        default=["10-K", "10-Q"],
        key="filing_types"
    )
    
    st.info(f"📅 Will fetch filings from **{reference_date.year - 2}** to **{reference_date.year}**")
    
    if st.button("📥 Start Ingestion", type="primary", key="ingest_btn"):
        if ingest_ticker.strip() and filing_types:
            with st.spinner(f"Downloading and processing {ingest_ticker.upper()} filings... This may take several minutes."):
                result = ingest_filings(ingest_ticker.upper(), reference_date, filing_types)
            
            if result.get("status") == "success":
                st.success(f"✓ {result.get('message', 'Ingestion complete')}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Filings Downloaded", result.get("filings_downloaded", 0))
                with col2:
                    st.metric("Chunks Created", result.get("chunks_created", 0))
                
                # Store chunk info for monitoring
                st.session_state["last_ingest"] = result
                
            elif result.get("status") == "warning":
                st.warning(f"⚠️ {result.get('message', 'Warning during ingestion')}")
            else:
                st.error(f"❌ Error: {result.get('message', 'Unknown error')}")
        else:
            st.warning("Please enter a ticker and select filing types")

# Chunk Monitor Tab
with tab3:
    st.header("Chunk Monitor")
    st.markdown("View details of ingested and retrieved chunks")
    
    # Last Query Chunks
    st.subheader("🔍 Last Query - Retrieved Chunks")
    
    if "last_query_result" in st.session_state:
        result = st.session_state["last_query_result"]
        sources = result.get("sources", [])
        
        st.markdown(f"**Query:** {result.get('query', 'N/A')}")
        st.markdown(f"**Ticker:** {result.get('ticker', 'N/A')}")
        st.markdown(f"**Chunks Used:** {result.get('chunks_used', 0)}")
        
        st.markdown("---")
        
        for i, source in enumerate(sources, 1):
            with st.expander(
                f"📄 Chunk {i}: {source.get('filing_type', '')} {source.get('fiscal_year', '')} - {source.get('section', 'Unknown Section')}",
                expanded=(i == 1)
            ):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Filing Type", source.get("filing_type", "N/A"))
                with col2:
                    st.metric("Fiscal Year", source.get("fiscal_year", "N/A"))
                with col3:
                    st.metric("Relevance", f"{source.get('relevance_score', 0):.1%}")
                
                st.markdown(f"**Section:** {source.get('section', 'N/A')}")
                if source.get("fiscal_quarter"):
                    st.markdown(f"**Quarter:** {source.get('fiscal_quarter')}")
    else:
        st.info("No query results yet. Run a query first to see retrieved chunks.")
    
    st.markdown("---")
    
    # Collection Details
    st.subheader("📁 Collection Statistics")
    
    collections = get_collections()
    
    if collections["total"] > 0:
        data = []
        for col in collections["collections"]:
            data.append({
                "Ticker": col["ticker"],
                "Chunks": col["chunk_count"],
                "Status": "✓ Ready" if col["exists"] else "Empty"
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Total chunks
        total_chunks = sum(c["chunk_count"] for c in collections["collections"])
        st.metric("Total Chunks in Database", total_chunks)
    else:
        st.info("No collections found. Ingest some filings first.")

# Footer
st.markdown("---")
st.caption("💡 Tip: Start by ingesting filings in the 'Ingest' tab, then query them in the 'Query' tab.")
