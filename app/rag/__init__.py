"""Part 2 — Vector store & RAG retrieval.

text-embedding-004 → Chroma (persistent, cosine, native metadata filtering) →
retriever → a thin chain that produces grounded, cited answers. Part 4 fuses
this with knowledge-graph results (GraphRAG).
"""

from app.rag.factory import get_rag, get_retriever, get_vector_store
from app.rag.rag import RAGPipeline, RAGResponse
from app.rag.retriever import RetrievedChunk, Retriever
from app.rag.vector_store import VectorStore

__all__ = [
    "VectorStore",
    "Retriever",
    "RetrievedChunk",
    "RAGPipeline",
    "RAGResponse",
    "get_vector_store",
    "get_retriever",
    "get_rag",
]
