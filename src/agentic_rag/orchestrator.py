"""
Agentic RAG Orchestrator

Orquesta las 5 capas del sistema:
1. Indexación (ya ejecutada, usa store existente)
2. Enrutamiento → decide estrategia
3. Query Builder → descompone y optimiza
4. Retrieval → busca y fusiona
5. Generation → sintetiza respuesta

Author: Glemes
"""

import time
import json
from typing import Dict, Any, Optional

from .layer1_indexing import IndexingLayer
from .layer2_routing import RoutingLayer, QueryType
from .layer3_query_builder import QueryBuilderLayer
from .layer4_retrieval import RetrievalLayer
from .layer5_generation import GenerationLayer

from ..monitoring.prompt_logger import get_prompt_logger
from ..monitoring.error_tracker import get_error_tracker


class AgenticRAGOrchestrator:
    """
    Orchestrador principal del sistema Agentic RAG.
    
    Coordina el flujo a través de las 5 capas y proporciona
    visibilidad completa del proceso.
    """
    
    def __init__(
        self,
        vectorstore: Any,
        llm: Any,
        embeddings: Any,
        graph: Optional[Any] = None,
    ):
        """
        Initialize orchestrator with all components.
        
        Args:
            vectorstore: Chroma vector store
            llm: Language model
            embeddings: Embedding model
            graph: Neo4j graph (optional)
        """
        self.logger = get_prompt_logger()
        self.error_tracker = get_error_tracker()
        
        print("\n" + "="*80)
        print("INITIALIZING AGENTIC RAG SYSTEM")
        print("="*80)
        
        # Initialize layers
        self.layer1 = IndexingLayer(
            embedding_model="BAAI/bge-large-en-v1.5",
            vector_store_dir="/vectors",
            graph=graph,
        )
        
        self.layer2 = RoutingLayer(llm=llm)
        self.layer3 = QueryBuilderLayer(llm=llm)
        self.layer4 = RetrievalLayer(
            vectorstore=vectorstore,
            graph=graph,
            reranker=None,  # TODO: Add reranker model
        )
        self.layer5 = GenerationLayer(llm=llm)
        
        print("\nALL LAYERS INITIALIZED")
        print("="*80 + "\n")
    
    def query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        k: int = 5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a query through the full Agentic RAG pipeline.
        
        Args:
            query: User's question
            context: Additional context (company, etc.)
            k: Number of documents to retrieve
            verbose: Whether to print detailed logs
            
        Returns:
            Complete response with answer, citations, and full trace
        """
        if verbose:
            print("\n" + "="*80)
            print("AGENTIC RAG PIPELINE EXECUTION")
            print("="*80)
            print(f"Query: {query}")
            if context:
                print(f"Context: {json.dumps(context)}")
            print("="*80)
        
        start_time = time.time()
        pipeline_trace = []
        
        try:
            # === LAYER 2: ROUTING ===
            step_start = time.time()
            routing_decision = self.layer2.route_query(query, context)
            step_time = (time.time() - step_start) * 1000
            
            pipeline_trace.append({
                "layer": "2_routing",
                "duration_ms": step_time,
                "decision": {
                    "query_type": routing_decision.query_type.value,
                    "confidence": routing_decision.confidence,
                    "tools": routing_decision.recommended_tools,
                    "reasoning": routing_decision.reasoning,
                }
            })
            
            # === LAYER 3: QUERY BUILDING ===
            step_start = time.time()
            query_plan = self.layer3.build_query_plan(
                query,
                routing_decision.query_type,
                context
            )
            step_time = (time.time() - step_start) * 1000
            
            pipeline_trace.append({
                "layer": "3_query_builder",
                "duration_ms": step_time,
                "plan": {
                    "sub_queries": query_plan.sub_queries,
                    "expanded_terms": query_plan.expanded_terms,
                    "translations": list(query_plan.translated_queries.keys()),
                }
            })
            
            # === LAYER 4: RETRIEVAL ===
            step_start = time.time()
            retrieval_response = self.layer4.retrieve(
                query_plan,
                k=k,
                apply_reranking=True,
                apply_fusion=True,
            )
            step_time = (time.time() - step_start) * 1000
            
            pipeline_trace.append({
                "layer": "4_retrieval",
                "duration_ms": step_time,
                "stats": {
                    "total_found": retrieval_response.total_found,
                    "returned": len(retrieval_response.results),
                    "fusion_applied": retrieval_response.fusion_applied,
                    "reranking_applied": retrieval_response.reranking_applied,
                }
            })
            
            # === LAYER 5: GENERATION ===
            step_start = time.time()
            generation_result = self.layer5.generate_response(
                query,
                retrieval_response,
                context
            )
            step_time = (time.time() - step_start) * 1000
            
            pipeline_trace.append({
                "layer": "5_generation",
                "duration_ms": step_time,
                "output": {
                    "answer_length": len(generation_result.get("answer", "")),
                    "citations": len(generation_result.get("citations", [])),
                    "confidence": generation_result.get("confidence", 0.0),
                }
            })
            
            # === FINALIZE ===
            total_time = (time.time() - start_time) * 1000
            
            response = {
                "query": query,
                "answer": generation_result.get("answer", ""),
                "citations": generation_result.get("citations", []),
                "confidence": generation_result.get("confidence", 0.0),
                "sources": generation_result.get("sources", []),
                "pipeline_trace": pipeline_trace,
                "total_time_ms": total_time,
                "metadata": {
                    "routing_decision": routing_decision.query_type.value,
                    "sub_queries": len(query_plan.sub_queries),
                    "documents_retrieved": len(retrieval_response.results),
                    "context": context,
                }
            }
            
            if verbose:
                self._print_summary(response)
            
            # Log complete execution
            self.logger.log_prompt(
                layer="orchestrator",
                prompt=f"FULL PIPELINE\nQuery: {query}\nContext: {json.dumps(context)}",
                response=f"Answer generated ({len(response['answer'])} chars)",
                metadata={
                    "total_time_ms": total_time,
                    "confidence": response["confidence"],
                    "citations": len(response["citations"]),
                },
                latency_ms=total_time,
            )
            
            return response
            
        except Exception as e:
            self.error_tracker.track_error(
                "orchestrator",
                e,
                context={"query": query[:200], "context": context},
                severity="critical"
            )
            
            return {
                "query": query,
                "answer": f"Pipeline error: {str(e)}",
                "citations": [],
                "confidence": 0.0,
                "sources": [],
                "pipeline_trace": pipeline_trace,
                "total_time_ms": (time.time() - start_time) * 1000,
                "metadata": {"error": str(e)}
            }
    
    def _print_summary(self, response: Dict[str, Any]):
        """Print execution summary."""
        print("\n" + "="*80)
        print("PIPELINE COMPLETED")
        print("="*80)
        
        print(f"\nSUMMARY:")
        print(f"   Total Time: {response['total_time_ms']:.0f}ms")
        print(f"   Confidence: {response['confidence']:.2f}")
        print(f"   Citations: {len(response['citations'])}")
        print(f"   Documents: {response['metadata']['documents_retrieved']}")
        
        print(f"\n⏱LAYER TIMINGS:")
        for step in response['pipeline_trace']:
            layer_name = step['layer'].replace('_', ' ').title()
            print(f"   {layer_name}: {step['duration_ms']:.0f}ms")
        
        print(f"\nANSWER ({len(response['answer'])} chars):")
        print("-" * 80)
        print(response['answer'][:300])
        if len(response['answer']) > 300:
            print("... [truncated]")
        print("-" * 80)
        
        if response['citations']:
            print(f"\nCITATIONS:")
            for citation in response['citations'][:3]:
                print(f"   - Source {citation['source_id']}: {citation['source']}")
        
        print("\n" + "="*80 + "\n")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        
        Returns:
            Dictionary with stats from all layers and monitoring
        """
        prompt_logger = get_prompt_logger()
        error_tracker = get_error_tracker()
        
        return {
            "indexing": self.layer1.get_stats(),
            "prompt_logs": prompt_logger.get_stats(),
            "errors": error_tracker.get_stats(),
        }
    
    def export_session_logs(self, output_dir: str = "/tmp"):
        """Export all logs from current session."""
        prompt_logger = get_prompt_logger()
        error_tracker = get_error_tracker()
        
        print(f"\nExporting session logs to {output_dir}...")
        
        # Export prompt logs
        prompt_file = f"{output_dir}/prompt_logs.json"
        prompt_logger.export_logs(prompt_file)
        print(f"   Prompt logs: {prompt_file}")
        
        # Export error logs (errors are auto-persisted)
        error_stats = error_tracker.get_stats()
        print(f"   Error logs: {error_stats.get('session_file', 'N/A')}")
        
        # Export system stats
        stats_file = f"{output_dir}/system_stats.json"
        import json
        with open(stats_file, 'w') as f:
            json.dump(self.get_system_stats(), f, indent=2)
        print(f"   System stats: {stats_file}")
        
        print(f"Export completed\n")

