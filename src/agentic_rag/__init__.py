"""
Agentic RAG System - 5 Layer Architecture

This module implements a production-grade Agentic RAG system with:
1. Layer 1: Indexación (Semantic Chunking + Multi-representación)
2. Layer 2: Enrutamiento (Query classification & tool selection)
3. Layer 3: Construcción de Consultas (Decomposition & expansion)
4. Layer 4: Recuperación (Fusion & re-ranking)
5. Layer 5: Generación (Synthesis with citations)

Author: Glemes
Version: 2.0.0 - Agentic RAG Architecture
"""

from .orchestrator import AgenticRAGOrchestrator
from .layer1_indexing import IndexingLayer
from .layer2_routing import RoutingLayer
from .layer3_query_builder import QueryBuilderLayer
from .layer4_retrieval import RetrievalLayer
from .layer5_generation import GenerationLayer

__all__ = [
    "AgenticRAGOrchestrator",
    "IndexingLayer",
    "RoutingLayer",
    "QueryBuilderLayer",
    "RetrievalLayer",
    "GenerationLayer",
]

__version__ = "2.0.0"

