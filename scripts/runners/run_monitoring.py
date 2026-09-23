"""
Script para ejecutar monitoring y métricas

Uso:
    python src/run_monitoring.py
    python src/run_monitoring.py --export prometheus
    python src/run_monitoring.py --record-latency query 2340
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.monitoring import MetricsCollector, PerformanceTracker


def main():
    parser = argparse.ArgumentParser(description="Run monitoring and metrics")
    parser.add_argument("--export", type=str, choices=["prometheus", "json"], help="Export format")
    parser.add_argument("--record-latency", nargs=2, metavar=("METRIC", "VALUE"), help="Record latency metric")
    parser.add_argument("--record-error", nargs=2, metavar=("METRIC", "ERROR"), help="Record error")
    parser.add_argument("--list", action="store_true", help="List all metrics")
    
    args = parser.parse_args()
    
    print("📊 Initializing monitoring system...")
    
    collector = MetricsCollector()
    tracker = PerformanceTracker()
    
    if args.record_latency:
        metric_name, value = args.record_latency
        collector.record_latency(metric_name, float(value))
        print(f"✅ Recorded latency: {metric_name} = {value}ms")
    
    if args.record_error:
        metric_name, error = args.record_error
        collector.record_error(metric_name, error)
        print(f"✅ Recorded error: {metric_name} = {error}")
    
    if args.list:
        metrics = collector.get_all_metrics()
        print(f"\n📋 Current Metrics ({len(metrics)}):")
        for metric in metrics:
            print(f"   {metric.name}: {metric.value} ({metric.metric_type.value})")
    
    if args.export == "prometheus":
        prometheus_output = collector.export_prometheus()
        print("\n📤 Prometheus Export:")
        print(prometheus_output)
        
        # Save to file
        with open("metrics.prom", "w") as f:
            f.write(prometheus_output)
        print("\n💾 Saved to metrics.prom")
    
    elif args.export == "json":
        metrics_data = collector.get_all_metrics()
        output = {
            "timestamp": time.time(),
            "metrics": [
                {
                    "name": m.name,
                    "type": m.metric_type.value,
                    "value": m.value,
                    "labels": m.labels,
                }
                for m in metrics_data
            ]
        }
        print("\n📤 JSON Export:")
        print(json.dumps(output, indent=2))
        
        with open("metrics.json", "w") as f:
            json.dump(output, f, indent=2)
        print("\n💾 Saved to metrics.json")
    
    if not any([args.record_latency, args.record_error, args.list, args.export]):
        print("\n💡 Usage examples:")
        print("   python src/run_monitoring.py --record-latency query 2340")
        print("   python src/run_monitoring.py --record-error query timeout")
        print("   python src/run_monitoring.py --list")
        print("   python src/run_monitoring.py --export prometheus")
        print("   python src/run_monitoring.py --export json")


if __name__ == "__main__":
    main()

