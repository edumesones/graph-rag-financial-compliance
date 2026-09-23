"""
Inspect and exercise the Prometheus metrics registry.

Note: metrics live in an in-process Prometheus registry. Values recorded by
this script describe only this process and are gone when it exits; the
service's real metrics are scraped from its /metrics endpoint.

Usage:
    python scripts/runners/run_monitoring.py --list
    python scripts/runners/run_monitoring.py --export prometheus
    python scripts/runners/run_monitoring.py --record-layer retrieval 2.34
    python scripts/runners/run_monitoring.py --record-error retrieval timeout
"""

import os
import sys
import argparse
import json

# This file lives at <repo>/scripts/runners/, so the repo root is three levels up.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.monitoring import (
    init_metrics,
    get_metrics,
    get_metrics_summary,
    record_layer_execution,
    record_error,
)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and exercise the Prometheus metrics registry"
    )
    parser.add_argument(
        "--export", choices=["prometheus", "json"], help="Export format"
    )
    parser.add_argument(
        "--record-layer",
        nargs=2,
        metavar=("LAYER", "SECONDS"),
        help="Record a layer execution time, in seconds",
    )
    parser.add_argument(
        "--record-error",
        nargs=2,
        metavar=("LAYER", "ERROR_TYPE"),
        help="Record an error against a layer",
    )
    parser.add_argument(
        "--list", action="store_true", help="List the registered metric families"
    )

    args = parser.parse_args()

    if not any([args.export, args.record_layer, args.record_error, args.list]):
        parser.print_help()
        return

    init_metrics()

    if args.record_layer:
        layer, seconds = args.record_layer
        record_layer_execution(layer, float(seconds))
        print("Recorded layer execution: {} = {}s".format(layer, seconds))

    if args.record_error:
        layer, error_type = args.record_error
        record_error(layer, error_type)
        print("Recorded error: {} / {}".format(layer, error_type))

    if args.list:
        print("\nRegistered metric families:")
        print(json.dumps(get_metrics_summary(), indent=2))

    if args.export == "prometheus":
        print("\nPrometheus exposition format:\n")
        print(get_metrics().decode("utf-8"))
    elif args.export == "json":
        print("\nMetric families (names only; values are in the Prometheus export):\n")
        print(json.dumps(get_metrics_summary(), indent=2))


if __name__ == "__main__":
    main()
