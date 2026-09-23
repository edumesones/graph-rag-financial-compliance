"""
Grafana Dashboards - Monitoring Dashboards

Based on Notion AGENTS monitoring concept:
- Pre-configured Grafana dashboards
- Prometheus queries
- Tempo trace visualization
- Alert configurations

Author: Glemes
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class DashboardPanel:
    """Grafana dashboard panel"""
    title: str
    query: str  # PromQL query
    panel_type: str = "graph"  # graph, stat, table, etc.
    y_axis_label: Optional[str] = None
    unit: Optional[str] = None
    grid_pos: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0, "w": 12, "h": 8})
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dashboard:
    """Grafana dashboard configuration"""
    dashboard_id: str
    title: str
    description: str
    panels: List[DashboardPanel] = field(default_factory=list)
    refresh: str = "30s"
    time_range: str = "now-6h"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class DashboardGenerator:
    """
    Generates Grafana dashboards for monitoring.
    
    Features:
    - Pre-configured dashboards
    - Prometheus queries
    - Tempo integration
    - Alert rules
    """

    def __init__(self, service_name: str = "fintech-rag"):
        self.service_name = service_name
        self.dashboards: Dict[str, Dashboard] = {}

    def create_performance_dashboard(self) -> Dashboard:
        """Create performance monitoring dashboard"""
        dashboard_id = "performance"
        
        panels = [
            DashboardPanel(
                title="Request Latency (p95)",
                query=f'histogram_quantile(0.95, rate({self.service_name}_request_latency_ms_bucket[5m]))',
                panel_type="graph",
                y_axis_label="Latency (ms)",
                unit="ms",
                grid_pos={"x": 0, "y": 0, "w": 12, "h": 8},
            ),
            DashboardPanel(
                title="Request Rate",
                query=f'rate({self.service_name}_requests_total[5m])',
                panel_type="graph",
                y_axis_label="Requests/sec",
                unit="reqps",
                grid_pos={"x": 0, "y": 8, "w": 12, "h": 8},
            ),
            DashboardPanel(
                title="Error Rate",
                query=f'rate({self.service_name}_errors_total[5m])',
                panel_type="graph",
                y_axis_label="Errors/sec",
                unit="errps",
                grid_pos={"x": 0, "y": 16, "w": 12, "h": 8},
            ),
            DashboardPanel(
                title="Success Rate",
                query=f'rate({self.service_name}_success_total[5m]) / rate({self.service_name}_requests_total[5m])',
                panel_type="stat",
                unit="percent",
                grid_pos={"x": 0, "y": 24, "w": 6, "h": 4},
            ),
        ]
        
        dashboard = Dashboard(
            dashboard_id=dashboard_id,
            title="Performance Monitoring",
            description="Performance metrics for Fintech RAG service",
            panels=panels,
        )
        
        self.dashboards[dashboard_id] = dashboard
        return dashboard

    def create_business_dashboard(self) -> Dashboard:
        """Create business metrics dashboard"""
        dashboard_id = "business"
        
        panels = [
            DashboardPanel(
                title="Queries Processed",
                query=f'sum(increase({self.service_name}_queries_total[1h]))',
                panel_type="stat",
                unit="short",
                grid_pos={"x": 0, "y": 0, "w": 6, "h": 4},
            ),
            DashboardPanel(
                title="Average Response Time",
                query=f'avg({self.service_name}_response_time_ms)',
                panel_type="stat",
                unit="ms",
                grid_pos={"x": 6, "y": 0, "w": 6, "h": 4},
            ),
            DashboardPanel(
                title="User Satisfaction",
                query=f'avg({self.service_name}_user_satisfaction_score)',
                panel_type="gauge",
                unit="percent",
                grid_pos={"x": 0, "y": 4, "w": 12, "h": 8},
            ),
        ]
        
        dashboard = Dashboard(
            dashboard_id=dashboard_id,
            title="Business Metrics",
            description="Business KPIs for Fintech RAG service",
            panels=panels,
        )
        
        self.dashboards[dashboard_id] = dashboard
        return dashboard

    def create_trace_dashboard(self) -> Dashboard:
        """Create trace visualization dashboard"""
        dashboard_id = "traces"
        
        panels = [
            DashboardPanel(
                title="Trace Overview",
                query=f'rate({self.service_name}_traces_total[5m])',
                panel_type="graph",
                y_axis_label="Traces/sec",
                unit="tracesps",
                grid_pos={"x": 0, "y": 0, "w": 12, "h": 8},
            ),
            DashboardPanel(
                title="Span Duration",
                query=f'histogram_quantile(0.95, rate({self.service_name}_span_duration_ms_bucket[5m]))',
                panel_type="graph",
                y_axis_label="Duration (ms)",
                unit="ms",
                grid_pos={"x": 0, "y": 8, "w": 12, "h": 8},
            ),
        ]
        
        dashboard = Dashboard(
            dashboard_id=dashboard_id,
            title="Distributed Tracing",
            description="Trace metrics and visualization",
            panels=panels,
        )
        
        self.dashboards[dashboard_id] = dashboard
        return dashboard

    def to_grafana_json(self, dashboard_id: str) -> Dict[str, Any]:
        """Convert dashboard to Grafana JSON format"""
        if dashboard_id not in self.dashboards:
            raise ValueError(f"Dashboard {dashboard_id} not found")
        
        dashboard = self.dashboards[dashboard_id]
        
        grafana_dashboard = {
            "dashboard": {
                "id": None,
                "uid": dashboard_id,
                "title": dashboard.title,
                "description": dashboard.description,
                "tags": ["fintech-rag", "monitoring"],
                "timezone": "browser",
                "schemaVersion": 16,
                "version": 0,
                "refresh": dashboard.refresh,
                "time": {
                    "from": dashboard.time_range,
                    "to": "now",
                },
                "panels": [
                    {
                        "id": i + 1,
                        "title": panel.title,
                        "type": panel.panel_type,
                        "gridPos": panel.grid_pos,
                        "targets": [
                            {
                                "expr": panel.query,
                                "refId": "A",
                            }
                        ],
                        "yaxes": [
                            {
                                "label": panel.y_axis_label,
                                "format": panel.unit,
                            },
                            {
                                "format": panel.unit,
                            }
                        ],
                    }
                    for i, panel in enumerate(dashboard.panels)
                ],
            },
            "overwrite": False,
        }
        
        return grafana_dashboard

    def create_alert_rules(self) -> List[Dict[str, Any]]:
        """Create Prometheus alert rules"""
        alerts = [
            {
                "alert": "HighErrorRate",
                "expr": f'rate({self.service_name}_errors_total[5m]) > 0.1',
                "for": "5m",
                "labels": {
                    "severity": "warning",
                },
                "annotations": {
                    "summary": "High error rate detected",
                    "description": "Error rate is above 0.1 errors/sec",
                },
            },
            {
                "alert": "HighLatency",
                "expr": f'histogram_quantile(0.95, rate({self.service_name}_request_latency_ms_bucket[5m])) > 5000',
                "for": "5m",
                "labels": {
                    "severity": "warning",
                },
                "annotations": {
                    "summary": "High latency detected",
                    "description": "P95 latency is above 5000ms",
                },
            },
            {
                "alert": "ServiceDown",
                "expr": f'up{{job="{self.service_name}"}} == 0',
                "for": "1m",
                "labels": {
                    "severity": "critical",
                },
                "annotations": {
                    "summary": "Service is down",
                    "description": "Service has been down for more than 1 minute",
                },
            },
        ]
        
        return alerts

    def export_dashboard(self, dashboard_id: str, filepath: str):
        """Export dashboard to JSON file"""
        grafana_json = self.to_grafana_json(dashboard_id)
        
        with open(filepath, "w") as f:
            json.dump(grafana_json, f, indent=2)

    def export_alerts(self, filepath: str):
        """Export alert rules to YAML"""
        alerts = self.create_alert_rules()
        
        alert_config = {
            "groups": [
                {
                    "name": "fintech_rag_alerts",
                    "interval": "30s",
                    "rules": alerts,
                }
            ]
        }
        
        with open(filepath, "w") as f:
            json.dump(alert_config, f, indent=2)

