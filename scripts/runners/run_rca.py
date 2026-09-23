"""
Script para ejecutar Root Cause Analysis (RCA)

Uso:
    python src/run_rca.py
    python src/run_rca.py --record-failure "timeout" "Query timeout after 30s"
    python src/run_rca.py --analyze
"""

import os
import sys
import argparse
import json

# This file lives at <repo>/scripts/runners/, so the repo root is three levels up.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.advanced.feedback import RootCauseAnalyzer, FailureCategory, Severity


def main():
    parser = argparse.ArgumentParser(description="Run Root Cause Analysis")
    parser.add_argument("--record-failure", nargs=2, metavar=("CATEGORY", "MESSAGE"), 
                       help="Record a failure event")
    parser.add_argument("--analyze", action="store_true", help="Analyze failures")
    parser.add_argument("--list", action="store_true", help="List all failures")
    parser.add_argument("--export", type=str, help="Export analysis to file")
    
    args = parser.parse_args()
    
    print("🔍 Initializing RCA system...")
    
    rca = RootCauseAnalyzer()
    
    if args.record_failure:
        category_str, message = args.record_failure
        try:
            category = FailureCategory[category_str.upper()]
        except KeyError:
            category = FailureCategory.UNKNOWN
        
        failure = rca.record_failure(
            category=category,
            error_message=message,
            severity=Severity.MEDIUM,
            context={"source": "manual"},
        )
        print(f"✅ Recorded failure: {failure.event_id}")
        print(f"   Category: {category.value}")
        print(f"   Message: {message}")
    
    if args.list:
        failures = rca.get_recent_failures(limit=10)
        print(f"\n📋 Recent Failures ({len(failures)}):")
        for failure in failures:
            print(f"   [{failure.category.value}] {failure.error_message[:60]}...")
    
    if args.analyze:
        print("\n🔬 Analyzing failures...")
        analysis = rca.analyze_failures()
        
        print(f"\n📊 RCA Analysis:")
        print(f"   Total failures: {analysis.total_failures}")
        print(f"   Time period: {analysis.analysis_period_hours}h")
        print(f"\n   Root Causes:")
        for cause in analysis.root_causes:
            print(f"      - {cause.category.value}: {cause.description}")
            print(f"        Frequency: {cause.frequency}")
            print(f"        Severity: {cause.severity.value}")
        
        print(f"\n   Recommendations:")
        for rec in analysis.recommendations:
            print(f"      - {rec}")
        
        if args.export:
            with open(args.export, 'w') as f:
                json.dump({
                    "total_failures": analysis.total_failures,
                    "root_causes": [
                        {
                            "category": c.category.value,
                            "description": c.description,
                            "frequency": c.frequency,
                            "severity": c.severity.value,
                        }
                        for c in analysis.root_causes
                    ],
                    "recommendations": analysis.recommendations,
                }, f, indent=2)
            print(f"\n💾 Analysis saved to: {args.export}")
    
    if not any([args.record_failure, args.analyze, args.list]):
        print("\n💡 Usage examples:")
        print("   python src/run_rca.py --record-failure TIMEOUT 'Query timeout'")
        print("   python src/run_rca.py --analyze")
        print("   python src/run_rca.py --list")
        print("   python src/run_rca.py --analyze --export analysis.json")


if __name__ == "__main__":
    main()

