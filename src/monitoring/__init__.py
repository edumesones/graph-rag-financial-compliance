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
from .metrics import (
    MetricsCollector,
    PerformanceTracker,
    Metric,
    MetricType,
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
    "MetricsCollector",
    "PerformanceTracker",
    "Metric",
    "MetricType",
    "DashboardGenerator",
    "Dashboard",
    "DashboardPanel",
]

