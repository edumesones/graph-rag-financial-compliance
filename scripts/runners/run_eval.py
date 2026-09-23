"""
Script para ejecutar evaluación de agentes

Uso:
    python src/run_eval.py
    python src/run_eval.py --plan fintech_basic
    python src/run_eval.py --test-case "What SEC violations occurred?"
"""

import os
import sys
import argparse
import json
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval import AgentEvaluator, EvaluationPlanner
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain_community.graphs import Neo4jGraph
from langchain_huggingface import HuggingFaceEmbeddings


def create_agent_endpoint(llm, vectorstore, graph):
    """Create a simple agent endpoint function"""
    def agent_query(query: str) -> str:
        """Execute agent query"""
        try:
            # Simple vector search
            docs = vectorstore.similarity_search(query, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Simple LLM call
            prompt = f"""Answer this question about financial compliance:
            
Question: {query}

Context from documents:
{context}

Answer:"""
            
            response = llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error: {str(e)}"
    
    return agent_query


def graph_query_fn(cypher_query: str) -> str:
    """Graph query function"""
    try:
        graph = Neo4jGraph(
            url=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            username=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "password")
        )
        result = graph.query(cypher_query)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Graph error: {str(e)}"


def vector_search_fn(query: str) -> str:
    """Vector search function"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vectorstore = Chroma(
            persist_directory="/vectors" if os.path.exists("/vectors") else "./vectors",
            embedding_function=embeddings,
            collection_name="fintech-rag-demo"
        )
        docs = vectorstore.similarity_search(query, k=5)
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"Vector search error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Run agent evaluation")
    parser.add_argument("--plan", type=str, default="fintech_basic", help="Evaluation plan name")
    parser.add_argument("--test-case", type=str, help="Single test case query")
    parser.add_argument("--output", type=str, help="Output JSON file")
    
    args = parser.parse_args()
    
    print("🔍 Initializing evaluation system...")
    
    # Initialize LLM
    print("  Loading LLM...")
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("❌ ERROR: HUGGINGFACE_TOKEN not set")
        print("   Set it with: $env:HUGGINGFACE_TOKEN = 'your_token'")
        return
    
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-70B-Instruct",
        temperature=0.1,
        max_new_tokens=512,
        huggingfacehub_api_token=hf_token,
        # Note: endpoint removed - only repo_id needed
    )
    
    # Initialize stores
    print("  Loading vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = None
    try:
        vectorstore = Chroma(
            persist_directory="/vectors" if os.path.exists("/vectors") else "./vectors",
            embedding_function=embeddings,
            collection_name="fintech-rag-demo"
        )
    except Exception as e:
        print(f"  ⚠️  Vector store not found: {e}")
    
    graph = None
    try:
        graph = Neo4jGraph(
            url=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            username=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "password")
        )
    except Exception as e:
        print(f"  ⚠️  Graph database not connected: {e}")
    
    # Create agent endpoint
    agent_fn = create_agent_endpoint(llm, vectorstore, graph) if vectorstore else lambda q: "Vector store not available"
    
    # Create evaluator
    evaluator = AgentEvaluator(
        llm_endpoint=agent_fn,
        graph_query_fn=graph_query_fn,
        vector_search_fn=vector_search_fn,
        memory_store=vectorstore,
        graph_store=graph,
    )
    
    # Run evaluation
    planner = EvaluationPlanner()
    
    if args.test_case:
        # Single test case
        print(f"\n📝 Running single test case: {args.test_case}")
        from src.eval.planning import TestCase, TestScenario
        test_case = TestCase(
            test_id="manual_test",
            input_query=args.test_case,
            expected_output="",
            scenario=TestScenario.BASIC_QUERY,
            min_confidence=0.7,
            max_latency_ms=10000,
        )
        result = evaluator.evaluate_test_case(test_case)
        print(f"\n✅ Result: {'PASSED' if result.passed else 'FAILED'}")
        print(f"   Latency: {result.metrics.latency_ms:.2f}ms")
        print(f"   Output: {result.actual_output[:200]}...")
    else:
        # Full evaluation plan
        print(f"\n📋 Running evaluation plan: {args.plan}")
        plan = planner.create_default_fintech_plan()
        results = evaluator.evaluate_with_plan(plan)
        
        # Summary
        summary = evaluator.get_summary()
        print(f"\n📊 Evaluation Summary:")
        print(f"   Total tests: {summary['total_tests']}")
        print(f"   Passed: {summary['passed_tests']}")
        print(f"   Failed: {summary['failed_tests']}")
        print(f"   Success rate: {summary['success_rate']:.2%}")
        print(f"   Avg latency: {summary['avg_latency_ms']:.2f}ms")
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({
                    "summary": summary,
                    "results": [
                        {
                            "test_id": r.test_case_id,
                            "passed": r.passed,
                            "latency_ms": r.metrics.latency_ms,
                            "input": r.input_query,
                            "output": r.actual_output[:500],
                        }
                        for r in results
                    ]
                }, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()

