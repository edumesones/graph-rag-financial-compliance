"""
Script para ejecutar experimentación (A/B Testing, Shadow Deployments)

Uso:
    python src/run_experimentation.py --ab-test
    python src/run_experimentation.py --shadow-deployment
"""

import os
import sys
import argparse
import json

# This file lives at <repo>/scripts/runners/, so the repo root is three levels up.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.advanced.experimentation import ABTestingManager, ShadowDeploymentManager


def main():
    parser = argparse.ArgumentParser(description="Run experimentation")
    parser.add_argument("--ab-test", action="store_true", help="Create A/B test")
    parser.add_argument("--shadow", action="store_true", help="Create shadow deployment")
    parser.add_argument("--list", action="store_true", help="List experiments")
    
    args = parser.parse_args()
    
    print("Initializing experimentation system...")
    
    if args.ab_test:
        print("\nCreating A/B test...")
        manager = ABTestingManager()
        
        test = manager.create_test(
            test_name="prompt_variation_test",
            variants=[
                {"variant_id": "prompt_v1", "traffic_percentage": 50},
                {"variant_id": "prompt_v2", "traffic_percentage": 50},
            ],
            success_metric="accuracy",
        )
        
        print(f"A/B Test created: {test.test_id}")
        print(f"   Variants: {len(test.variants)}")
        print(f"   Status: {test.status.value}")
    
    if args.shadow:
        print("\nCreating shadow deployment...")
        manager = ShadowDeploymentManager()
        
        deployment = manager.create_shadow_deployment(
            deployment_name="new_prompt_version",
            production_endpoint="https://production.modal.run",
            shadow_endpoint="https://shadow.modal.run",
        )
        
        print(f"Shadow deployment created: {deployment.deployment_id}")
        print(f"   Status: {deployment.status.value}")
    
    if args.list:
        print("\nActive Experiments:")
        # This would list from storage - simplified for demo
        print("   (No active experiments - create one with --ab-test or --shadow)")
    
    if not any([args.ab_test, args.shadow, args.list]):
        print("\nUsage examples:")
        print("   python src/run_experimentation.py --ab-test")
        print("   python src/run_experimentation.py --shadow")
        print("   python src/run_experimentation.py --list")


if __name__ == "__main__":
    main()

