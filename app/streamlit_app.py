"""
Streamlit web UI for the Hybrid RAG system.
Provides an interactive chat interface where users can type questions and
get answers with source citations. The sidebar shows retrieval settings
(top-K, context size, max tokens) and system info. Uses st.cache_resource
to load the heavy models only once across browser refreshes.
Run with: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st

from src.generation.llm import LLMGenerator
from src.rag.pipeline import HybridRAG
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import DenseIndex
from src.types import Chunk
from src.utils.io import read_jsonl


def load_chunks(path):
    records = read_jsonl(path)
    return [Chunk(**r) for r in records]


@st.cache_resource
def load_rag_system():
    chunks_path = project_root / "data" / "corpus" / "chunks.jsonl"
    dense_index_path = project_root / "indexes" / "dense"
    bm25_index_path = project_root / "indexes" / "bm25"
    
    chunks = load_chunks(str(chunks_path))
    dense_index = DenseIndex.load(chunks, str(dense_index_path))
    bm25_index = BM25Index.load(chunks, str(bm25_index_path))
    
    generator = LLMGenerator(model_name="google/flan-t5-base", device="cpu")
    generator.load()
    
    rag = HybridRAG(
        chunks=chunks,
        dense_index=dense_index,
        bm25_index=bm25_index,
        generator=generator,
        rrf_constant=60,
    )
    
    return rag, len(chunks)


def render_context(retrieved_chunks):
    for pos, retrieved in enumerate(retrieved_chunks, start=1):
        with st.expander(f"Source {pos}: {retrieved.chunk.title}", expanded=(pos <= 2)):
            st.write(f"URL: {retrieved.chunk.url}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Dense Rank", retrieved.dense_rank or "N/A")
            col2.metric("BM25 Rank", retrieved.bm25_rank or "N/A")
            col3.metric("RRF Score", f"{retrieved.rrf_score:.4f}")
            
            text = retrieved.chunk.text
            if len(text) > 500:
                text = text[:500] + "..."
            st.write(text)


def main():
    st.set_page_config(page_title="Hybrid RAG System", layout="wide")
    
    st.title("Hybrid RAG System")
    st.write("Dense + BM25 + Reciprocal Rank Fusion")
    
    st.sidebar.header("System Info")
    
    with st.spinner("Loading system..."):
        try:
            rag, num_chunks = load_rag_system()
            st.sidebar.success("Loaded")
            st.sidebar.metric("Chunks", num_chunks)
        except Exception as e:
            st.error(f"Failed to load: {e}")
            st.stop()
    
    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Retrieval Top-K", 10, 100, 50)
    context_size = st.sidebar.slider("Context Chunks", 3, 10, 6)
    max_tokens = st.sidebar.slider("Max Tokens", 50, 200, 100)
    
    st.sidebar.markdown("---")
    st.sidebar.write("Dense: all-mpnet-base-v2")
    st.sidebar.write("Sparse: BM25 Okapi")
    st.sidebar.write("Fusion: RRF (k=60)")
    st.sidebar.write("LLM: Flan-T5-Base")
    
    st.markdown("---")
    
    query = st.text_input("Enter your question:")
    
    ask = st.button("Ask")
    
    st.write("Examples:")
    examples = [
        "When was the Eiffel Tower built?",
        "What is the height of Burj Khalifa?",
        "Which country is Machu Picchu located in?",
    ]
    
    cols = st.columns(len(examples))
    for i, (col, ex) in enumerate(zip(cols, examples)):
        if col.button(f"Example {i+1}", help=ex):
            query = ex
            ask = True
    
    if ask and query.strip():
        st.markdown("---")
        
        with st.spinner("Processing..."):
            result = rag.answer(query, top_k=top_k, context_size=context_size, max_new_tokens=max_tokens)
        
        st.header("Answer")
        st.write(result.answer)
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Retrieval", f"{result.latency_ms['retrieve_total']:.0f} ms")
        col2.metric("Generation", f"{result.latency_ms['generate']:.0f} ms")
        col3.metric("Total", f"{result.latency_ms['total']:.0f} ms")
        
        st.markdown("---")
        st.header("Sources")
        render_context(result.context_chunks)
    
    st.markdown("---")
    st.write("Hybrid RAG System")


if __name__ == "__main__":
    main()
