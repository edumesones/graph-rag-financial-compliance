"""
Master script to run the operational support systems.

Usage:
    python scripts/runners/run_all_systems.py --monitoring
    python scripts/runners/run_all_systems.py --all
"""

import os
import sys
import argparse

# This file lives at <repo>/scripts/runners/, so the repo root is three levels up.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def run_monitoring():
    """Run monitoring system"""
    print("\n" + "=" * 60)
    print("MONITORING SYSTEM")
    print("=" * 60)
    from scripts.runners.run_monitoring import main as monitoring_main
    monitoring_main()


def run_rca():
    """Run RCA system"""
    print("\n" + "=" * 60)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 60)
    from scripts.runners.run_rca import main as rca_main
    rca_main()


def run_hitl():
    """Run HITL system"""
    print("\n" + "=" * 60)
    print("HUMAN-IN-THE-LOOP")
    print("=" * 60)
    from scripts.runners.run_hitl import main as hitl_main
    hitl_main()


def run_experimentation():
    """Run experimentation system"""
    print("\n" + "=" * 60)
    print("EXPERIMENTATION")
    print("=" * 60)
    from scripts.runners.run_experimentation import main as exp_main
    exp_main()


def main():
    parser = argparse.ArgumentParser(
        description="Run the monitoring, RCA, HITL and experimentation systems"
    )
    parser.add_argument("--monitoring", action="store_true", help="Run monitoring")
    parser.add_argument("--rca", action="store_true", help="Run RCA")
    parser.add_argument("--hitl", action="store_true", help="Run HITL")
    parser.add_argument(
        "--experimentation", action="store_true", help="Run experimentation"
    )
    parser.add_argument("--all", action="store_true", help="Run all systems")

    args = parser.parse_args()

    if args.all:
        run_monitoring()
        run_rca()
        run_hitl()
        run_experimentation()
    elif args.monitoring or args.rca or args.hitl or args.experimentation:
        if args.monitoring:
            run_monitoring()
        if args.rca:
            run_rca()
        if args.hitl:
            run_hitl()
        if args.experimentation:
            run_experimentation()
    else:
        parser.print_help()
        return

    print("\nDone.")


if __name__ == "__main__":
    main()
