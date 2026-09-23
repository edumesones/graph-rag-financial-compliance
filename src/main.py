"""
Fintech Agentic RAG - FastAPI Application (Docker Edition)

Migrated from Modal serverless to Docker deployment with:
- FastAPI + Gunicorn ASGI server
- PostgreSQL for structured logging
- Redis for caching
- Prometheus metrics
- Neo4j graph database
- ChromaDB vector store

5-Layer Agentic RAG Architecture:
1. Indexación: Semantic chunking + multi-representación
2. Enrutamiento: Análisis inteligente de consultas
3. Query Building: Descomposición y optimización
4. Recuperación: Fusion + re-ranking
5. Generación: Síntesis con citaciones

Author: Glemes
Version: 2.0.0 - Docker Edition
"""

import os
import sys
import time
import traceback
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.graphs import Neo4jGraph

# Import local modules
from src.agentic_rag.orchestrator import AgenticRAGOrchestrator
from src.db.postgres import PostgresManager, postgres_manager, get_postgres
from src.db.redis_cache import RedisCache, redis_cache, get_redis
from src.utils.config import config, validate_config, print_config_summary
from src.monitoring import metrics


# ================================
# Global State
# ================================

# These will be initialized in lifespan
embeddings = None
vectorstore = None
graph = None
llm = None
orchestrator = None


# ================================
# Pydantic Models
# ================================


class AnalyzeRequest(BaseModel):
    """Request model for analyze_compliance endpoint"""

    company: str = Field(..., description="Company name to analyze", example="Tesla Inc")
    query: str = Field(..., description="Analysis query", example="What were Q4 2023 results?")
    verbose: bool = Field(default=True, description="Enable verbose logging")


class AnalyzeResponse(BaseModel):
    """Response model for analyze_compliance endpoint"""

    query: str
    answer: str
    citations: List[Dict[str, Any]]
    confidence: float
    sources: List[str]
    pipeline_trace: List[Dict[str, Any]]
    company: str
    llm_model: str
    embedding_model: str
    architecture: str
    total_request_time_ms: float
    monitoring: Dict[str, Any]
    status: str


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    service: str
    version: str
    architecture: str
    databases: Dict[str, str]
    features: List[str]


class SystemStatsResponse(BaseModel):
    """System statistics response"""

    system: Dict[str, Any]
    indexing: Dict[str, Any]
    databases: Dict[str, Any]
    cache: Dict[str, Any]
    status: str


# ================================
# Application Lifespan
# ================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager

    Handles startup and shutdown:
    - Startup: Initialize databases, load models, create orchestrator
    - Shutdown: Close connections, cleanup resources
    """
    global embeddings, vectorstore, graph, llm, orchestrator
    global postgres_manager, redis_cache

    print("\n" + "=" * 80)
    print("🚀 STARTING FINTECH AGENTIC RAG (DOCKER EDITION)")
    print("=" * 80)

    # Validate configuration
    if not validate_config():
        raise RuntimeError("Configuration validation failed")

    print_config_summary()

    # === STARTUP ===

    try:
        # 1. Initialize PostgreSQL
        print("\n[Startup] Connecting to PostgreSQL...")
        postgres_manager = PostgresManager(
            host=config.postgres.host,
            port=config.postgres.port,
            database=config.postgres.database,
            user=config.postgres.user,
            password=config.postgres.password,
        )
        await postgres_manager.connect()

        # 2. Initialize Redis
        print("\n[Startup] Connecting to Redis...")
        redis_cache = RedisCache(
            host=config.redis.host,
            port=config.redis.port,
            db=config.redis.db,
            password=config.redis.password,
            default_ttl=config.redis.ttl,
        )
        await redis_cache.connect()

        # 3. Load embeddings (expensive, do once)
        print("\n[Startup] Loading embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name=config.llm.embedding_model,
            cache_folder="/app/vectors/.cache",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print(f"   ✅ Embeddings loaded: {config.llm.embedding_model}")

        # 4. Connect to vector store
        print("\n[Startup] Connecting to ChromaDB...")
        vectorstore = Chroma(
            persist_directory=config.vectorstore.persist_directory,
            embedding_function=embeddings,
            collection_name=config.vectorstore.collection_name,
        )
        doc_count = vectorstore._collection.count()
        print(f"   ✅ Vector store loaded: {doc_count} documents")

        # 5. Connect to Neo4j graph (optional)
        print("\n[Startup] Connecting to Neo4j...")
        try:
            graph = Neo4jGraph(
                url=config.neo4j.uri,
                username=config.neo4j.user,
                password=config.neo4j.password,
                database=config.neo4j.database,
            )
            print(f"   ✅ Graph database connected: {config.neo4j.uri}")
        except Exception as e:
            print(f"   ⚠️  Graph database not available: {e}")
            graph = None

        # 6. Configure LLM
        print("\n[Startup] Configuring LLM...")
        llm = ChatOpenAI(
            model=config.llm.model,
            base_url=config.llm.base_url,
            api_key=config.llm.huggingface_token,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )
        print(f"   ✅ LLM configured: {config.llm.model}")

        # 7. Create Agentic RAG Orchestrator
        print("\n[Startup] Creating Agentic RAG Orchestrator...")
        orchestrator = AgenticRAGOrchestrator(
            vectorstore=vectorstore,
            llm=llm,
            embeddings=embeddings,
            graph=graph,
        )
        print("   ✅ Orchestrator ready")

        # 8. Initialize Prometheus metrics
        print("\n[Startup] Initializing Prometheus metrics...")
        metrics.init_metrics(app_name=config.app_name, version=config.version)

        print("\n" + "=" * 80)
        print("✅ STARTUP COMPLETE")
        print("=" * 80 + "\n")

        yield  # Application runs here

    except Exception as e:
        print(f"\n❌ STARTUP FAILED: {e}")
        traceback.print_exc()
        raise

    finally:
        # === SHUTDOWN ===
        print("\n" + "=" * 80)
        print("🛑 SHUTTING DOWN")
        print("=" * 80)

        if postgres_manager:
            print("\n[Shutdown] Closing PostgreSQL connection...")
            await postgres_manager.disconnect()

        if redis_cache:
            print("\n[Shutdown] Closing Redis connection...")
            await redis_cache.disconnect()

        print("\n" + "=" * 80)
        print("✅ SHUTDOWN COMPLETE")
        print("=" * 80 + "\n")


# ================================
# FastAPI Application
# ================================

app = FastAPI(
    title=config.app_name,
    version=config.version,
    description=f"""
    {config.architecture} for financial compliance analysis.

    Features:
    - 5-layer Agentic RAG pipeline
    - Semantic search with ChromaDB
    - Graph-based knowledge with Neo4j
    - Redis caching for performance
    - PostgreSQL structured logging
    - Prometheus metrics
    """,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================
# Endpoints
# ================================


@app.post(
    "/analyze_compliance",
    response_model=AnalyzeResponse,
    summary="Analyze company compliance",
    description="Full 5-layer Agentic RAG pipeline for compliance analysis",
)
async def analyze_compliance(request: AnalyzeRequest) -> Dict[str, Any]:
    """
    Agentic RAG endpoint with full 5-layer pipeline.

    Args:
        request: Analysis request with company, query, and verbose flag

    Returns:
        Comprehensive analysis with answer, citations, and pipeline trace
    """
    metrics.rag_active_requests.inc()
    start_time = time.time()

    print("\n" + "=" * 80)
    print("🚀 AGENTIC RAG REQUEST")
    print("=" * 80)
    print(f"Company: {request.company}")
    print(f"Query: {request.query[:100]}...")
    print(f"Verbose: {request.verbose}")

    try:
        # Check cache first
        cache_key = f"query:{hash(f'{request.company}:{request.query}')}"
        if redis_cache:
            cached_result = await redis_cache.get(cache_key)
            if cached_result:
                print("   ✅ Cache hit!")
                metrics.record_cache_hit("redis")
                metrics.rag_active_requests.dec()
                return {
                    **cached_result,
                    "cached": True,
                    "status": "success",
                }
            else:
                metrics.record_cache_miss("redis")

        # Execute pipeline
        context = {"company": request.company}
        response = orchestrator.query(
            query=request.query,
            context=context,
            k=5,
            verbose=request.verbose,
        )

        # Calculate total time
        total_elapsed = (time.time() - start_time) * 1000

        # Get database stats
        postgres_stats = {"connected": postgres_manager.is_connected if postgres_manager else False}
        redis_stats = await redis_cache.get_stats() if redis_cache else {"status": "not connected"}

        # Prepare final response
        final_response = {
            **response,
            "company": request.company,
            "llm_model": config.llm.model,
            "embedding_model": config.llm.embedding_model,
            "architecture": config.architecture,
            "total_request_time_ms": total_elapsed,
            "monitoring": {
                "postgres": postgres_stats,
                "redis": redis_stats,
                "cached": False,
            },
            "status": "success",
        }

        # Cache the result
        if redis_cache:
            await redis_cache.set(cache_key, final_response, ttl=config.redis.ttl)

        # Log to PostgreSQL
        if postgres_manager:
            await postgres_manager.log_query_metrics(
                query=request.query,
                company=request.company,
                total_time_ms=total_elapsed,
                confidence=response.get("confidence", 0.0),
                citations_count=len(response.get("citations", [])),
                documents_retrieved=len(response.get("sources", [])),
                routing_decision=response.get("routing_strategy", "unknown"),
                metadata={"verbose": request.verbose},
            )

        # Record Prometheus metrics
        metrics.record_request("/analyze_compliance", request.company, "success")
        metrics.record_confidence(response.get("confidence", 0.0))
        metrics.record_citations(len(response.get("citations", [])))
        metrics.record_company_query(request.company)
        metrics.rag_request_duration_seconds.labels(
            endpoint="/analyze_compliance", layer="total"
        ).observe(total_elapsed / 1000)

        print(f"\n✅ Request completed in {total_elapsed:.0f}ms")
        print("=" * 80 + "\n")

        metrics.rag_active_requests.dec()
        return final_response

    except Exception as e:
        error_msg = str(e)
        trace = traceback.format_exc()

        print(f"\n❌ ERROR: {error_msg}")
        print(trace)

        # Log error to PostgreSQL
        if postgres_manager:
            await postgres_manager.log_error(
                layer="main_endpoint",
                error_type=type(e).__name__,
                error_message=error_msg,
                context={"query": request.query[:200], "company": request.company},
                severity="critical",
                traceback_str=trace,
            )

        # Record error metrics
        metrics.record_error("main_endpoint", type(e).__name__, "critical")
        metrics.record_request("/analyze_compliance", request.company, "failed")

        metrics.rag_active_requests.dec()

        raise HTTPException(
            status_code=500,
            detail={
                "error": error_msg,
                "traceback": trace,
                "query": request.query,
                "company": request.company,
                "status": "failed",
            },
        )


@app.get(
    "/get_system_stats",
    response_model=SystemStatsResponse,
    summary="Get system statistics",
    description="Comprehensive system stats including databases, cache, and monitoring",
)
async def get_system_stats() -> Dict[str, Any]:
    """
    Get system statistics and monitoring data.

    Returns comprehensive stats about:
    - System info (versions, models)
    - Indexing layer (document counts)
    - Database connections
    - Cache statistics
    """
    try:
        # Vector store stats
        doc_count = vectorstore._collection.count() if vectorstore else 0

        # Database stats
        postgres_stats = {"status": "disconnected"}
        if postgres_manager:
            postgres_stats = {
                "status": "connected" if postgres_manager.is_connected else "disconnected",
                "host": postgres_manager.host,
                "database": postgres_manager.database,
                "health": await postgres_manager.health_check(),
            }

        # Redis stats
        redis_stats = {"status": "disconnected"}
        if redis_cache:
            redis_stats = await redis_cache.get_stats()

        # Neo4j stats
        graph_stats = {"status": "not available"}
        if graph:
            graph_stats = {"status": "connected", "uri": config.neo4j.uri}

        return {
            "system": {
                "architecture": config.architecture,
                "version": config.version,
                "llm_model": config.llm.model,
                "embedding_model": config.llm.embedding_model,
                "environment": config.environment,
            },
            "indexing": {
                "document_count": doc_count,
                "vector_store": "ChromaDB",
                "collection_name": config.vectorstore.collection_name,
            },
            "databases": {
                "postgresql": postgres_stats,
                "neo4j": graph_stats,
            },
            "cache": redis_stats,
            "status": "healthy",
        }

    except Exception as e:
        print(f"⚠️  Error getting system stats: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "status": "error"},
        )


@app.get(
    "/get_prompt_logs",
    summary="Get prompt logs",
    description="Retrieve prompt logs from PostgreSQL with optional filtering",
)
async def get_prompt_logs(
    layer: Optional[str] = Query(None, description="Filter by layer"),
    limit: int = Query(10, ge=1, le=100, description="Max number of logs"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
) -> Dict[str, Any]:
    """
    Get prompt logs with optional filtering.

    Query params:
    - layer: Filter by specific layer (e.g., "layer2_routing")
    - limit: Max number of logs to return
    - offset: Offset for pagination

    Returns logs from PostgreSQL database.
    """
    try:
        if not postgres_manager:
            raise HTTPException(status_code=503, detail="PostgreSQL not connected")

        logs = await postgres_manager.get_prompt_logs(layer=layer, limit=limit, offset=offset)

        return {
            "logs": logs,
            "filtered_by": {"layer": layer, "limit": limit, "offset": offset},
            "count": len(logs),
            "status": "success",
        }

    except Exception as e:
        print(f"⚠️  Error getting prompt logs: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "status": "error"},
        )


@app.get(
    "/health_check",
    response_model=HealthResponse,
    summary="Health check",
    description="Service health and feature availability",
)
async def health_check() -> Dict[str, Any]:
    """Health check endpoint with database connectivity status."""

    # Check database connectivity
    postgres_status = "disconnected"
    if postgres_manager:
        postgres_healthy = await postgres_manager.health_check()
        postgres_status = "connected" if postgres_healthy else "error"

    redis_status = "disconnected"
    if redis_cache:
        redis_healthy = await redis_cache.health_check()
        redis_status = "connected" if redis_healthy else "error"

    neo4j_status = "not available" if graph is None else "connected"

    return {
        "status": "healthy",
        "service": config.app_name,
        "version": config.version,
        "architecture": config.architecture,
        "databases": {
            "postgresql": postgres_status,
            "redis": redis_status,
            "neo4j": neo4j_status,
            "chromadb": "connected" if vectorstore else "disconnected",
        },
        "features": [
            "Semantic Chunking",
            "Intelligent Routing",
            "Query Decomposition",
            "Hybrid Retrieval",
            "Citation Generation",
            "PostgreSQL Logging",
            "Redis Caching",
            "Prometheus Metrics",
        ],
    }


@app.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Metrics endpoint for Prometheus scraping",
    include_in_schema=False,
)
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=metrics.get_metrics(),
        media_type=metrics.get_metrics_content_type(),
    )


# ================================
# Root Endpoint
# ================================


@app.get("/", summary="API root", include_in_schema=False)
async def root():
    """API root with links to documentation."""
    return {
        "service": config.app_name,
        "version": config.version,
        "documentation": "/docs",
        "health": "/health_check",
        "metrics": "/metrics",
    }


# ================================
# Main Entry Point
# ================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 80)
    print("🚀 STARTING DEVELOPMENT SERVER")
    print("=" * 80)
    print(f"Service: {config.app_name} v{config.version}")
    print(f"Environment: {config.environment}")
    print(f"Server: http://0.0.0.0:8000")
    print(f"Docs: http://0.0.0.0:8000/docs")
    print("=" * 80 + "\n")

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info",
    )
