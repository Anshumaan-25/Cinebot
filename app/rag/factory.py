"""Cached factory helpers for the RAG stack."""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.kg.factory import get_graph_store
from app.llm.factory import get_embedder, get_llm
from app.rag.entities import EntityLinker
from app.rag.hybrid import HybridRAG, HybridRetriever
from app.rag.rag import RAGPipeline
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore


@lru_cache
def get_vector_store() -> VectorStore:
    s = get_settings()
    return VectorStore(s.chroma_dir, s.chroma_collection)


@lru_cache
def get_retriever() -> Retriever:
    return Retriever(get_vector_store(), get_embedder())


@lru_cache
def get_rag() -> RAGPipeline:
    s = get_settings()
    return RAGPipeline(get_retriever(), get_llm(), k=s.rag_top_k)


@lru_cache
def get_entity_linker() -> EntityLinker:
    return EntityLinker(get_graph_store())


@lru_cache
def get_hybrid_retriever() -> HybridRetriever:
    s = get_settings()
    return HybridRetriever(get_graph_store(), get_entity_linker(), get_retriever(), k=s.rag_top_k)


@lru_cache
def get_hybrid_rag() -> HybridRAG:
    return HybridRAG(get_hybrid_retriever(), get_llm())
