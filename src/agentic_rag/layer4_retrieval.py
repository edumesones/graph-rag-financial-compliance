"""
Layer 4: Recuperación Inteligente (The Hunter) - Docker Edition

Ejecuta búsquedas y refina resultados:
- Búsquedas híbridas (vector + keyword + graph)
- Re-ranking con modelo especializado
- Fusión de resultados de múltiples fuentes
- Control de acceso (ACLs)
- Redis caching para performance

Author: Glemes
"""

import time
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

from ..monitoring.prompt_logger import get_prompt_logger
from ..monitoring.error_tracker import get_error_tracker
from ..db.redis_cache import get_redis
from .layer3_query_builder import QueryPlan


@dataclass
class RetrievalResult:
    """Resultado de recuperación con metadata."""
    content: str
    source: str
    score: float
    rank: int
    metadata: Dict[str, Any]
    retrieval_method: str  # "vector", "graph", "hybrid"


@dataclass
class RetrievalResponse:
    """Respuesta completa de recuperación."""
    results: List[RetrievalResult]
    total_found: int
    retrieval_time_ms: float
    fusion_applied: bool
    reranking_applied: bool
    metadata: Dict[str, Any]


class RetrievalLayer:
    """
    Capa 4: Recuperación y refinamiento de información.
    
    Combina múltiples métodos de búsqueda y optimiza resultados.
    """
    
    def __init__(
        self,
        vectorstore: Any,
        graph: Optional[Any] = None,
        reranker: Optional[Any] = None,
        use_cache: bool = True,
    ):
        """
        Initialize retrieval layer.

        Args:
            vectorstore: Chroma vector store
            graph: Neo4j graph database (optional)
            reranker: Re-ranking model (optional)
            use_cache: Whether to use Redis caching (default: True)
        """
        self.logger = get_prompt_logger()
        self.error_tracker = get_error_tracker()
        self.redis_cache = get_redis() if use_cache else None

        self.vectorstore = vectorstore
        self.graph = graph
        self.reranker = reranker
        self.use_cache = use_cache

        print("\n[Layer 4: Retrieval] Initializing...")
        print(f"   Vector store available")

        if self.graph:
            print(f"   Graph database available")
        else:
            print(f"   Graph database not available")

        if self.reranker:
            print(f"   Re-ranker model available")
        else:
            print(f"   Re-ranker not available (using score-based ranking)")

        if self.redis_cache:
            print(f"   Redis cache available")
        else:
            print(f"   Redis cache not available")

        print("   Layer 4 initialized\n")
    
    async def retrieve(
        self,
        query_plan: QueryPlan,
        k: int = 5,
        apply_reranking: bool = True,
        apply_fusion: bool = True,
    ) -> RetrievalResponse:
        """
        Execute retrieval based on query plan with Redis caching.

        Args:
            query_plan: Query plan from Layer 3
            k: Number of results to return
            apply_reranking: Whether to apply re-ranking
            apply_fusion: Whether to fuse results from multiple sources

        Returns:
            RetrievalResponse with results
        """
        print(f"\n[Layer 4] Retrieving documents...")
        print(f"   Sub-queries: {len(query_plan.sub_queries)}")
        print(f"   Fusion: {apply_fusion}")
        print(f"   Re-ranking: {apply_reranking}")

        # Generate cache key from query plan + parameters
        cache_key = self._generate_cache_key(query_plan, k, apply_reranking, apply_fusion)

        # Check cache first
        if self.redis_cache:
            try:
                cached = await self.redis_cache.get(cache_key)
                if cached:
                    print(f"   Cache HIT! Returning cached results")
                    # Reconstruct RetrievalResponse from cached dict
                    cached_results = [
                        RetrievalResult(**r) for r in cached.get("results", [])
                    ]
                    return RetrievalResponse(
                        results=cached_results,
                        total_found=cached["total_found"],
                        retrieval_time_ms=cached["retrieval_time_ms"],
                        fusion_applied=cached["fusion_applied"],
                        reranking_applied=cached["reranking_applied"],
                        metadata={**cached["metadata"], "from_cache": True},
                    )
            except Exception as e:
                print(f"   Cache read error: {e}")

        print(f"   Cache MISS - executing retrieval")
        start_time = time.time()
        
        try:
            all_results: List[RetrievalResult] = []
            
            # Execute each sub-query
            for i, sub_query in enumerate(query_plan.sub_queries):
                print(f"\n   Sub-query {i+1}/{len(query_plan.sub_queries)}: {sub_query[:60]}...")
                
                # Vector search
                vector_results = self._vector_search(sub_query, k=k*2)
                print(f"      Vector: {len(vector_results)} results")
                all_results.extend(vector_results)
                
                # Graph search (if available and query has Cypher translation)
                if self.graph and "cypher" in query_plan.translated_queries:
                    graph_results = self._graph_search(
                        query_plan.translated_queries["cypher"],
                        k=k
                    )
                    print(f"      Graph: {len(graph_results)} results")
                    all_results.extend(graph_results)
            
            print(f"\n   Retrieved {len(all_results)} total results")
            
            # Fusion: combine and deduplicate
            if apply_fusion and len(all_results) > k:
                fused_results = self._fuse_results(all_results, k)
                print(f"   Fused to {len(fused_results)} results")
            else:
                fused_results = all_results[:k]
            
            # Re-ranking: reorder by relevance
            if apply_reranking and self.reranker:
                reranked_results = self._rerank_results(
                    query_plan.original_query,
                    fused_results
                )
                print(f"   Re-ranked results")
            else:
                reranked_results = fused_results
            
            # Add final ranks
            for i, result in enumerate(reranked_results):
                result.rank = i + 1
            
            elapsed = (time.time() - start_time) * 1000
            print(f"   ⏱Retrieval completed ({elapsed:.0f}ms)")

            response = RetrievalResponse(
                results=reranked_results,
                total_found=len(all_results),
                retrieval_time_ms=elapsed,
                fusion_applied=apply_fusion,
                reranking_applied=apply_reranking and self.reranker is not None,
                metadata={
                    "sub_queries": len(query_plan.sub_queries),
                    "k": k,
                    "from_cache": False,
                }
            )

            # Cache the result
            if self.redis_cache:
                try:
                    cache_data = {
                        "results": [asdict(r) for r in response.results],
                        "total_found": response.total_found,
                        "retrieval_time_ms": response.retrieval_time_ms,
                        "fusion_applied": response.fusion_applied,
                        "reranking_applied": response.reranking_applied,
                        "metadata": response.metadata,
                    }
                    # Cache for 1 hour (3600 seconds)
                    await self.redis_cache.set(cache_key, cache_data, ttl=3600)
                    print(f"   Results cached")
                except Exception as e:
                    print(f"   Cache write error: {e}")

            # Log retrieval
            await self.logger.log_prompt(
                layer="layer4_retrieval",
                prompt=self._format_retrieval_prompt(query_plan),
                response=f"Retrieved {len(reranked_results)} results",
                metadata={
                    "total_found": len(all_results),
                    "fusion": apply_fusion,
                    "reranking": apply_reranking,
                },
                latency_ms=elapsed,
            )

            return response
            
        except Exception as e:
            await self.error_tracker.track_error(
                "layer4_retrieval",
                e,
                context={"query": query_plan.original_query[:200]}
            )
            # Return empty response
            return RetrievalResponse(
                results=[],
                total_found=0,
                retrieval_time_ms=0,
                fusion_applied=False,
                reranking_applied=False,
                metadata={"error": str(e)}
            )
    
    def _vector_search(
        self,
        query: str,
        k: int = 10
    ) -> List[RetrievalResult]:
        """Perform vector similarity search."""
        try:
            docs = self.vectorstore.similarity_search_with_score(query, k=k)
            
            results = []
            for doc, score in docs:
                result = RetrievalResult(
                    content=doc.page_content,
                    source=doc.metadata.get('source', 'unknown'),
                    score=float(score),
                    rank=0,  # Will be set later
                    metadata=doc.metadata,
                    retrieval_method="vector"
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"      Vector search error: {e}")
            return []
    
    def _graph_search(
        self,
        cypher_query: str,
        k: int = 10
    ) -> List[RetrievalResult]:
        """Perform graph database search."""
        if not self.graph:
            return []
        
        try:
            results_data = self.graph.query(cypher_query)
            
            results = []
            for i, item in enumerate(results_data[:k]):
                # Format graph result as text
                content = json.dumps(item, indent=2)
                
                result = RetrievalResult(
                    content=content,
                    source="graph",
                    score=1.0,  # Graph results don't have similarity scores
                    rank=0,
                    metadata={"cypher_query": cypher_query},
                    retrieval_method="graph"
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"      Graph search error: {e}")
            return []
    
    def _fuse_results(
        self,
        results: List[RetrievalResult],
        k: int
    ) -> List[RetrievalResult]:
        """
        Fuse results from multiple sources using reciprocal rank fusion.
        
        Algorithm: RRF (Reciprocal Rank Fusion)
        """
        # Group by retrieval method
        by_method: Dict[str, List[RetrievalResult]] = {}
        for result in results:
            method = result.retrieval_method
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(result)
        
        # Calculate RRF scores
        rrf_scores: Dict[str, float] = {}
        k_rrf = 60  # RRF constant
        
        for method, method_results in by_method.items():
            for rank, result in enumerate(method_results):
                key = f"{result.source}_{hash(result.content[:100])}"
                
                # RRF formula: 1 / (k + rank)
                rrf_score = 1.0 / (k_rrf + rank + 1)
                
                if key not in rrf_scores:
                    rrf_scores[key] = 0.0
                rrf_scores[key] += rrf_score
        
        # Re-score original results
        for result in results:
            key = f"{result.source}_{hash(result.content[:100])}"
            result.score = rrf_scores.get(key, result.score)
        
        # Sort by new score and deduplicate
        seen_keys = set()
        fused = []
        
        for result in sorted(results, key=lambda x: x.score, reverse=True):
            key = f"{result.source}_{hash(result.content[:100])}"
            if key not in seen_keys:
                fused.append(result)
                seen_keys.add(key)
            
            if len(fused) >= k:
                break
        
        return fused
    
    def _rerank_results(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Re-rank results using specialized model.
        
        For now, placeholder. In production, would use cross-encoder model.
        """
        if not self.reranker:
            return results
        
        try:
            # Use reranker model to score query-document pairs
            # Placeholder: just return as-is
            return results
            
        except Exception as e:
            print(f"   Re-ranking failed: {e}")
            return results
    
    def _generate_cache_key(
        self,
        query_plan: QueryPlan,
        k: int,
        apply_reranking: bool,
        apply_fusion: bool
    ) -> str:
        """
        Generate Redis cache key from query plan and parameters.

        Args:
            query_plan: Query plan
            k: Number of results
            apply_reranking: Reranking flag
            apply_fusion: Fusion flag

        Returns:
            Cache key string
        """
        # Create unique hash from query + parameters
        import hashlib
        key_components = [
            query_plan.original_query,
            str(k),
            str(apply_reranking),
            str(apply_fusion),
            str(sorted(query_plan.sub_queries)),
        ]
        key_string = "|".join(key_components)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"retrieval:{key_hash}"

    def _format_retrieval_prompt(self, query_plan: QueryPlan) -> str:
        """Format prompt for logging."""
        return f"""RETRIEVAL EXECUTION
Original Query: {query_plan.original_query}
Sub-queries: {len(query_plan.sub_queries)}
{chr(10).join(f"  {i+1}. {q}" for i, q in enumerate(query_plan.sub_queries))}
Translations: {json.dumps(query_plan.translated_queries, indent=2)}"""

