"""
Monitoring System for Fintech RAG Template

Based on Notion AGENTS monitoring concept:
- OpenTelemetry: Observability setup
- Prometheus: Metrics collection
- Grafana: Dashboards and visualization
- Tempo: Distributed tracing
"""

from .opentelemetry_setup import (
    OpenTelemetrySetup,
    TelemetryConfig,
    create_default_config,
)
from . import metrics
from .metrics import (
    init_metrics,
    get_metrics,
    get_metrics_content_type,
    get_metrics_summary,
    record_request,
    record_layer_execution,
    record_documents_retrieved,
    record_cache_hit,
    record_cache_miss,
    record_llm_call,
    record_error,
    record_vectorstore_search,
    record_graph_query,
    record_confidence,
    record_citations,
    record_company_query,
    track_time,
    track_errors,
)
from .dashboards import (
    DashboardGenerator,
    Dashboard,
    DashboardPanel,
)

__all__ = [
    "OpenTelemetrySetup",
    "TelemetryConfig",
    "create_default_config",
    "metrics",
    "init_metrics",
    "get_metrics",
    "get_metrics_content_type",
    "get_metrics_summary",
    "record_request",
    "record_layer_execution",
    "record_documents_retrieved",
    "record_cache_hit",
    "record_cache_miss",
    "record_llm_call",
    "record_error",
    "record_vectorstore_search",
    "record_graph_query",
    "record_confidence",
    "record_citations",
    "record_company_query",
    "track_time",
    "track_errors",
    "DashboardGenerator",
    "Dashboard",
    "DashboardPanel",
]

