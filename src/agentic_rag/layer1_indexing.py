"""
Layer 1: Indexación Inteligente (The Knowledge Base)

Implementa:
- Semantic Chunking (chunking por ideas completas)
- Indexación Multi-representación (keywords, embeddings, metadata)
- Embeddings especializados con cache
- Jerarquías (summary vs detail)

Author: Glemes
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.graphs import Neo4jGraph
from langchain_huggingface import HuggingFaceEmbeddings

from ..monitoring.prompt_logger import get_prompt_logger
from ..monitoring.error_tracker import get_error_tracker


@dataclass
class DocumentChunk:
    """Representa un fragmento de documento con metadata enriquecida."""
    content: str
    summary: str
    metadata: Dict[str, Any]
    chunk_id: str
    chunk_type: str  # "summary", "detail", "full"
    embeddings: Optional[List[float]] = None


class IndexingLayer:
    """
    Capa 1: Gestión inteligente del conocimiento.
    
    Features:
    - Semantic chunking (no cortes arbitrarios)
    - Multi-representación (keywords + embeddings + metadata)
    - Jerarquías (resumen + detalle)
    """
    
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        cache_dir: str = "/vectors/.cache",
        vector_store_dir: str = "/vectors",
        graph: Optional[Neo4jGraph] = None,
    ):
        """
        Initialize indexing layer.
        
        Args:
            embedding_model: HuggingFace embedding model
            cache_dir: Cache directory for model
            vector_store_dir: Vector store persistence directory
            graph: Neo4j graph database instance
        """
        self.logger = get_prompt_logger()
        self.error_tracker = get_error_tracker()
        
        print("\n[Layer 1: Indexing] Initializing...")
        
        # Initialize embeddings
        try:
            start_time = time.time()
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                cache_folder=cache_dir,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            elapsed = (time.time() - start_time) * 1000
            print(f"   Embeddings loaded ({elapsed:.0f}ms): {embedding_model}")
        except Exception as e:
            self.error_tracker.track_error("layer1_indexing", e, {"step": "embedding_init"})
            raise
        
        # Initialize vector store
        try:
            self.vectorstore = Chroma(
                persist_directory=vector_store_dir,
                embedding_function=self.embeddings,
                collection_name="fintech-rag-demo"
            )
            count = self.vectorstore._collection.count()
            print(f"   Vector store connected: {count} documents")
        except Exception as e:
            self.error_tracker.track_error("layer1_indexing", e, {"step": "vectorstore_init"})
            raise
        
        # Graph database (optional)
        self.graph = graph
        if self.graph:
            print(f"   Graph database connected")
        else:
            print(f"   Graph database not available")
        
        # Semantic text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],  # Semantic boundaries
            length_function=len,
        )
        
        print("   Layer 1 initialized\n")
    
    def chunk_document(
        self,
        document: str,
        metadata: Dict[str, Any],
        create_hierarchy: bool = True
    ) -> List[DocumentChunk]:
        """
        Chunk document with semantic boundaries.
        
        Args:
            document: Full document text
            metadata: Document metadata (source, date, etc.)
            create_hierarchy: Whether to create summary + detail hierarchy
            
        Returns:
            List of DocumentChunk objects
        """
        print(f"\n[Layer 1] Chunking document...")
        print(f"   Source: {metadata.get('source', 'unknown')}")
        print(f"   Length: {len(document)} chars")
        print(f"   Hierarchy: {create_hierarchy}")
        
        start_time = time.time()
        
        try:
            # Split into semantic chunks
            raw_chunks = self.text_splitter.split_text(document)
            print(f"   Created {len(raw_chunks)} semantic chunks")
            
            chunks: List[DocumentChunk] = []
            
            # Create detail chunks
            for i, chunk_text in enumerate(raw_chunks):
                chunk = DocumentChunk(
                    content=chunk_text,
                    summary="",  # Will be generated if hierarchy=True
                    metadata={
                        **metadata,
                        "chunk_index": i,
                        "total_chunks": len(raw_chunks),
                    },
                    chunk_id=f"{metadata.get('source', 'doc')}_{i}",
                    chunk_type="detail",
                )
                chunks.append(chunk)
            
            # Create summary chunk if requested
            if create_hierarchy and len(document) > 2000:
                summary_chunk = self._create_summary_chunk(document, metadata)
                if summary_chunk:
                    chunks.insert(0, summary_chunk)
                    print(f"   Created summary chunk")
            
            elapsed = (time.time() - start_time) * 1000
            print(f"   ⏱Chunking completed ({elapsed:.0f}ms)")
            
            # Log operation
            self.logger.log_prompt(
                layer="layer1_indexing",
                prompt=f"chunk_document(source={metadata.get('source')}, len={len(document)})",
                response=f"Created {len(chunks)} chunks",
                metadata={"hierarchy": create_hierarchy, "chunks": len(chunks)},
                latency_ms=elapsed,
            )
            
            return chunks
            
        except Exception as e:
            self.error_tracker.track_error(
                "layer1_indexing",
                e,
                context={"operation": "chunk_document", "source": metadata.get('source')}
            )
            raise
    
    def _create_summary_chunk(
        self,
        document: str,
        metadata: Dict[str, Any]
    ) -> Optional[DocumentChunk]:
        """
        Create a summary chunk for long documents.
        
        For now, uses simple extraction. In production, would use LLM summarization.
        """
        # Simple summary: first 500 chars
        summary = document[:500] + "..." if len(document) > 500 else document
        
        return DocumentChunk(
            content=summary,
            summary=summary,
            metadata={
                **metadata,
                "chunk_index": -1,  # Summary chunk
                "total_chunks": 1,
            },
            chunk_id=f"{metadata.get('source', 'doc')}_summary",
            chunk_type="summary",
        )
    
    def index_chunks(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Index chunks into vector store and graph.
        
        Multi-representación:
        - Vector embeddings for semantic search
        - Graph nodes for entity relationships
        - Metadata for filtering
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            Indexing statistics
        """
        print(f"\n[Layer 1] Indexing {len(chunks)} chunks...")
        
        start_time = time.time()
        stats = {
            "chunks_indexed": 0,
            "vectors_created": 0,
            "graph_nodes_created": 0,
        }
        
        try:
            # Index in vector store
            texts = [chunk.content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            self.vectorstore.add_texts(
                texts=texts,
                metadatas=metadatas,
            )
            
            stats["chunks_indexed"] = len(chunks)
            stats["vectors_created"] = len(chunks)
            print(f"   Indexed {len(chunks)} vectors")
            
            # Index in graph (if available)
            if self.graph:
                graph_stats = self._index_in_graph(chunks)
                stats["graph_nodes_created"] = graph_stats.get("nodes", 0)
                print(f"   Created {stats['graph_nodes_created']} graph nodes")
            
            elapsed = (time.time() - start_time) * 1000
            print(f"   ⏱Indexing completed ({elapsed:.0f}ms)")
            
            # Log operation
            self.logger.log_prompt(
                layer="layer1_indexing",
                prompt=f"index_chunks(count={len(chunks)})",
                response=json.dumps(stats),
                latency_ms=elapsed,
            )
            
            return stats
            
        except Exception as e:
            self.error_tracker.track_error(
                "layer1_indexing",
                e,
                context={"operation": "index_chunks", "chunks": len(chunks)}
            )
            raise
    
    def _index_in_graph(self, chunks: List[DocumentChunk]) -> Dict[str, int]:
        """Index chunks in graph database."""
        # Placeholder: En producción, extraería entidades y relaciones
        return {"nodes": 0, "relationships": 0}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get indexing statistics."""
        try:
            vector_count = self.vectorstore._collection.count()
            
            stats = {
                "vector_count": vector_count,
                "embedding_model": "BAAI/bge-large-en-v1.5",
                "graph_available": self.graph is not None,
            }
            
            if self.graph:
                try:
                    result = self.graph.query("MATCH (n) RETURN count(n) as total")
                    stats["graph_nodes"] = result[0]['total'] if result else 0
                except:
                    stats["graph_nodes"] = 0
            
            return stats
            
        except Exception as e:
            self.error_tracker.track_error("layer1_indexing", e, {"operation": "get_stats"})
            return {"error": str(e)}


# For compatibility
import json

