"""
Prometheus Metrics for Fintech Agentic RAG (Docker Edition)

Comprehensive metrics collection using prometheus-client library.
Integrates with Prometheus scraping from docker-compose.yml.

Metric Categories:
- Request metrics (total, duration, active)
- Layer execution metrics (routing, query building, retrieval, generation)
- Document retrieval metrics (count, relevance)
- Cache metrics (hits, misses, hit rate)
- Database metrics (connection pool, query time)
- LLM metrics (tokens, latency, errors)
- Error metrics (by layer, type, severity)
- Vector store metrics (search time, results)
- Graph database metrics (query time, results)
- Business metrics (companies queried, confidence scores)
- System metrics (memory, CPU, active connections)
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
)
from typing import Optional, Dict, Any
import time
from functools import wraps


# ================================
# Global Prometheus Registry
# ================================
# Use default registry for automatic collection
registry = CollectorRegistry()


# ================================
# Request Metrics
# ================================

# Total requests counter
rag_requests_total = Counter(
    "rag_requests_total",
    "Total RAG analysis requests",
    ["endpoint", "status", "company"],
    registry=registry,
)

# Request duration histogram
rag_request_duration_seconds = Histogram(
    "rag_request_duration_seconds",
    "RAG request duration in seconds",
    ["endpoint", "layer"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
    registry=registry,
)

# Active requests gauge
rag_active_requests = Gauge(
    "rag_active_requests",
    "Number of active RAG requests",
    registry=registry,
)


# ================================
# Layer Execution Metrics
# ================================

# Layer execution time
layer_execution_time = Histogram(
    "layer_execution_time_seconds",
    "Execution time for each RAG layer",
    ["layer"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=registry,
)

# Layer success/failure counter
layer_operations_total = Counter(
    "layer_operations_total",
    "Total operations per layer",
    ["layer", "status"],
    registry=registry,
)


# ================================
# Document Retrieval Metrics
# ================================

# Documents retrieved per query
documents_retrieved = Summary(
    "documents_retrieved",
    "Number of documents retrieved per query",
    registry=registry,
)

# Document relevance scores
document_relevance_score = Histogram(
    "document_relevance_score",
    "Relevance scores of retrieved documents",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=registry,
)

# Retrieval latency
retrieval_latency_seconds = Histogram(
    "retrieval_latency_seconds",
    "Document retrieval latency",
    ["retrieval_type"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=registry,
)


# ================================
# Cache Metrics
# ================================

# Cache hits/misses
cache_hits_total = Counter(
    "cache_hits_total",
    "Total cache hits",
    ["cache_type"],
    registry=registry,
)

cache_misses_total = Counter(
    "cache_misses_total",
    "Total cache misses",
    ["cache_type"],
    registry=registry,
)

# Cache operation latency
cache_operation_duration_seconds = Histogram(
    "cache_operation_duration_seconds",
    "Cache operation duration",
    ["operation", "cache_type"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    registry=registry,
)


# ================================
# Database Metrics
# ================================

# PostgreSQL connection pool
postgres_connections_active = Gauge(
    "postgres_connections_active",
    "Active PostgreSQL connections",
    registry=registry,
)

postgres_connections_idle = Gauge(
    "postgres_connections_idle",
    "Idle PostgreSQL connections",
    registry=registry,
)

# PostgreSQL query duration
postgres_query_duration_seconds = Histogram(
    "postgres_query_duration_seconds",
    "PostgreSQL query duration",
    ["query_type"],
    buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    registry=registry,
)

# Redis connection pool
redis_connections_active = Gauge(
    "redis_connections_active",
    "Active Redis connections",
    registry=registry,
)


# ================================
# LLM Metrics
# ================================

# LLM API calls
llm_api_calls_total = Counter(
    "llm_api_calls_total",
    "Total LLM API calls",
    ["provider", "model", "status"],
    registry=registry,
)

# LLM token usage
llm_tokens_used = Counter(
    "llm_tokens_used",
    "Total tokens used",
    ["provider", "model", "token_type"],
    registry=registry,
)

# LLM latency
llm_request_duration_seconds = Histogram(
    "llm_request_duration_seconds",
    "LLM request duration",
    ["provider", "model"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
    registry=registry,
)


# ================================
# Error Metrics
# ================================

# Errors by layer and type
errors_total = Counter(
    "errors_total",
    "Total errors",
    ["layer", "error_type", "severity"],
    registry=registry,
)

# Error rate gauge
error_rate = Gauge(
    "error_rate",
    "Current error rate (errors per minute)",
    ["layer"],
    registry=registry,
)


# ================================
# Vector Store Metrics
# ================================

# ChromaDB operations
vectorstore_operations_total = Counter(
    "vectorstore_operations_total",
    "Total vector store operations",
    ["operation", "status"],
    registry=registry,
)

# Vector search duration
vectorstore_search_duration_seconds = Histogram(
    "vectorstore_search_duration_seconds",
    "Vector search duration",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=registry,
)

# Vector search results count
vectorstore_search_results = Summary(
    "vectorstore_search_results",
    "Number of results from vector search",
    registry=registry,
)


# ================================
# Graph Database Metrics (Neo4j)
# ================================

# Neo4j operations
graph_operations_total = Counter(
    "graph_operations_total",
    "Total graph database operations",
    ["operation", "status"],
    registry=registry,
)

# Neo4j query duration
graph_query_duration_seconds = Histogram(
    "graph_query_duration_seconds",
    "Graph query duration",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    registry=registry,
)

# Neo4j results count
graph_query_results = Summary(
    "graph_query_results",
    "Number of results from graph queries",
    registry=registry,
)


# ================================
# Business Metrics
# ================================

# Unique companies queried
companies_queried_total = Counter(
    "companies_queried_total",
    "Total unique companies queried",
    ["company"],
    registry=registry,
)

# Confidence scores
confidence_score = Histogram(
    "confidence_score",
    "Confidence scores for RAG responses",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=registry,
)

# Citations per response
citations_count = Summary(
    "citations_count",
    "Number of citations per response",
    registry=registry,
)


# ================================
# System Metrics
# ================================

# System info (static labels)
system_info = Info(
    "rag_system",
    "RAG system information",
    registry=registry,
)

# Active sessions
active_sessions = Gauge(
    "active_sessions",
    "Number of active user sessions",
    registry=registry,
)


# ================================
# Helper Functions
# ================================


def record_request(endpoint: str, company: str, status: str = "success"):
    """Record a RAG request"""
    rag_requests_total.labels(endpoint=endpoint, status=status, company=company).inc()


def record_layer_execution(layer: str, duration_seconds: float, status: str = "success"):
    """Record layer execution metrics"""
    layer_execution_time.labels(layer=layer).observe(duration_seconds)
    layer_operations_total.labels(layer=layer, status=status).inc()


def record_documents_retrieved(count: int, relevance_scores: list):
    """Record document retrieval metrics"""
    documents_retrieved.observe(count)
    for score in relevance_scores:
        document_relevance_score.observe(score)


def record_cache_hit(cache_type: str = "redis"):
    """Record cache hit"""
    cache_hits_total.labels(cache_type=cache_type).inc()


def record_cache_miss(cache_type: str = "redis"):
    """Record cache miss"""
    cache_misses_total.labels(cache_type=cache_type).inc()


def record_llm_call(
    provider: str,
    model: str,
    duration_seconds: float,
    prompt_tokens: int,
    completion_tokens: int,
    status: str = "success",
):
    """Record LLM API call metrics"""
    llm_api_calls_total.labels(provider=provider, model=model, status=status).inc()
    llm_request_duration_seconds.labels(provider=provider, model=model).observe(duration_seconds)
    llm_tokens_used.labels(provider=provider, model=model, token_type="prompt").inc(prompt_tokens)
    llm_tokens_used.labels(provider=provider, model=model, token_type="completion").inc(
        completion_tokens
    )


def record_error(layer: str, error_type: str, severity: str = "error"):
    """Record an error"""
    errors_total.labels(layer=layer, error_type=error_type, severity=severity).inc()


def record_vectorstore_search(duration_seconds: float, results_count: int, status: str = "success"):
    """Record vector store search metrics"""
    vectorstore_operations_total.labels(operation="search", status=status).inc()
    vectorstore_search_duration_seconds.observe(duration_seconds)
    vectorstore_search_results.observe(results_count)


def record_graph_query(duration_seconds: float, results_count: int, status: str = "success"):
    """Record graph database query metrics"""
    graph_operations_total.labels(operation="query", status=status).inc()
    graph_query_duration_seconds.observe(duration_seconds)
    graph_query_results.observe(results_count)


def record_confidence(score: float):
    """Record confidence score"""
    confidence_score.observe(score)


def record_citations(count: int):
    """Record citations count"""
    citations_count.observe(count)


def record_company_query(company: str):
    """Record company query"""
    companies_queried_total.labels(company=company).inc()


# ================================
# Decorators for Automatic Tracking
# ================================


def track_time(metric: Histogram, labels: Optional[Dict[str, str]] = None):
    """Decorator to track execution time"""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)

        # Return appropriate wrapper based on whether function is async
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def track_errors(layer: str):
    """Decorator to track errors"""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                error_type = type(e).__name__
                record_error(layer, error_type)
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error_type = type(e).__name__
                record_error(layer, error_type)
                raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# ================================
# Metrics Endpoint Handler
# ================================


def get_metrics() -> bytes:
    """
    Get metrics in Prometheus format

    Returns:
        Metrics in Prometheus exposition format
    """
    return generate_latest(registry)


def get_metrics_content_type() -> str:
    """Get the content type for metrics endpoint"""
    return CONTENT_TYPE_LATEST


# ================================
# Initialization
# ================================


def init_metrics(app_name: str = "Fintech Agentic RAG", version: str = "2.0.0"):
    """
    Initialize metrics with system information

    Args:
        app_name: Application name
        version: Application version
    """
    system_info.info(
        {
            "app_name": app_name,
            "version": version,
            "architecture": "5-Layer Agentic RAG",
            "deployment": "Docker Compose",
        }
    )
    print(f"Prometheus metrics initialized: {app_name} v{version}")


# ================================
# Metrics Summary
# ================================


def get_metrics_summary() -> Dict[str, Any]:
    """
    Get a summary of current metrics (for debugging)

    Returns:
        Dictionary with metrics summary
    """
    return {
        "request_metrics": {
            "total_requests": "rag_requests_total",
            "active_requests": "rag_active_requests",
            "request_duration": "rag_request_duration_seconds",
        },
        "layer_metrics": {
            "execution_time": "layer_execution_time_seconds",
            "operations": "layer_operations_total",
        },
        "retrieval_metrics": {
            "documents_retrieved": "documents_retrieved",
            "relevance_scores": "document_relevance_score",
            "retrieval_latency": "retrieval_latency_seconds",
        },
        "cache_metrics": {
            "hits": "cache_hits_total",
            "misses": "cache_misses_total",
            "operation_duration": "cache_operation_duration_seconds",
        },
        "database_metrics": {
            "postgres_connections": "postgres_connections_active/idle",
            "postgres_query_duration": "postgres_query_duration_seconds",
            "redis_connections": "redis_connections_active",
        },
        "llm_metrics": {
            "api_calls": "llm_api_calls_total",
            "tokens_used": "llm_tokens_used",
            "request_duration": "llm_request_duration_seconds",
        },
        "error_metrics": {
            "total_errors": "errors_total",
            "error_rate": "error_rate",
        },
        "vectorstore_metrics": {
            "operations": "vectorstore_operations_total",
            "search_duration": "vectorstore_search_duration_seconds",
            "search_results": "vectorstore_search_results",
        },
        "graph_metrics": {
            "operations": "graph_operations_total",
            "query_duration": "graph_query_duration_seconds",
            "query_results": "graph_query_results",
        },
        "business_metrics": {
            "companies_queried": "companies_queried_total",
            "confidence_scores": "confidence_score",
            "citations": "citations_count",
        },
        "system_metrics": {
            "system_info": "rag_system",
            "active_sessions": "active_sessions",
        },
    }
