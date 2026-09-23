"""
OpenTelemetry Setup - Observability Integration

Based on Notion AGENTS monitoring concept:
- OpenTelemetry instrumentation
- Metrics, logs, and traces
- Integration with Prometheus, Grafana, Tempo

Author: Glemes
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import os
import json


@dataclass
class TelemetryConfig:
    """OpenTelemetry configuration"""
    service_name: str
    service_version: str = "1.0.0"
    environment: str = "production"
    otel_exporter_otlp_endpoint: Optional[str] = None
    otel_exporter_otlp_headers: Dict[str, str] = field(default_factory=dict)
    otel_resource_attributes: Dict[str, str] = field(default_factory=dict)
    enable_metrics: bool = True
    enable_traces: bool = True
    enable_logs: bool = True
    prometheus_port: int = 8000
    metadata: Dict[str, Any] = field(default_factory=dict)


class OpenTelemetrySetup:
    """
    Setup OpenTelemetry instrumentation.
    
    Provides:
    - Metrics export to Prometheus
    - Traces export to Tempo
    - Logs export to Grafana Loki
    - Resource attributes
    """

    def __init__(self, config: TelemetryConfig):
        self.config = config
        self.initialized = False

    def initialize(self) -> bool:
        """
        Initialize OpenTelemetry.
        
        Returns:
            True if initialization successful
        """
        try:
            # Set environment variables
            os.environ["OTEL_SERVICE_NAME"] = self.config.service_name
            os.environ["OTEL_SERVICE_VERSION"] = self.config.service_version
            os.environ["OTEL_RESOURCE_ATTRIBUTES"] = self._format_resource_attributes()
            
            if self.config.otel_exporter_otlp_endpoint:
                os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = self.config.otel_exporter_otlp_endpoint
            
            if self.config.otel_exporter_otlp_headers:
                headers_str = ",".join([f"{k}={v}" for k, v in self.config.otel_exporter_otlp_headers.items()])
                os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = headers_str
            
            # Initialize SDK (would use actual OpenTelemetry SDK in production)
            # from opentelemetry import trace, metrics
            # from opentelemetry.sdk.trace import TracerProvider
            # from opentelemetry.sdk.metrics import MeterProvider
            # from opentelemetry.sdk.resources import Resource
            # from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            # from opentelemetry.exporter.prometheus import PrometheusMetricReader
            
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"Failed to initialize OpenTelemetry: {e}")
            return False

    def _format_resource_attributes(self) -> str:
        """Format resource attributes as comma-separated key=value pairs"""
        attrs = {
            "service.name": self.config.service_name,
            "service.version": self.config.service_version,
            "deployment.environment": self.config.environment,
            **self.config.otel_resource_attributes,
        }
        
        return ",".join([f"{k}={v}" for k, v in attrs.items()])

    def get_prometheus_config(self) -> Dict[str, Any]:
        """Get Prometheus exporter configuration"""
        return {
            "port": self.config.prometheus_port,
            "endpoint": f"/metrics",
            "format": "prometheus",
        }

    def get_tempo_config(self) -> Dict[str, Any]:
        """Get Tempo exporter configuration"""
        return {
            "endpoint": self.config.otel_exporter_otlp_endpoint or "http://localhost:4317",
            "protocol": "grpc",
            "headers": self.config.otel_exporter_otlp_headers,
        }

    def get_grafana_config(self) -> Dict[str, Any]:
        """Get Grafana configuration"""
        return {
            "prometheus_url": f"http://localhost:{self.config.prometheus_port}/metrics",
            "tempo_url": self.config.otel_exporter_otlp_endpoint or "http://localhost:4317",
            "service_name": self.config.service_name,
        }

    def export_config(self, filepath: str):
        """Export configuration to JSON"""
        config_dict = {
            "service_name": self.config.service_name,
            "service_version": self.config.service_version,
            "environment": self.config.environment,
            "prometheus": self.get_prometheus_config(),
            "tempo": self.get_tempo_config(),
            "grafana": self.get_grafana_config(),
            "initialized": self.initialized,
        }
        
        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=2)


def create_default_config(service_name: str) -> TelemetryConfig:
    """Create default OpenTelemetry configuration"""
    return TelemetryConfig(
        service_name=service_name,
        service_version="1.0.0",
        environment=os.getenv("ENVIRONMENT", "production"),
        otel_exporter_otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        otel_resource_attributes={
            "service.namespace": "fintech-rag",
            "service.instance.id": os.getenv("HOSTNAME", "unknown"),
        },
    )

