"""
Script maestro para ejecutar todos los sistemas de evaluación e improvement

Uso:
    python src/run_all_systems.py --eval
    python src/run_all_systems.py --monitoring
    python src/run_all_systems.py --all
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_evaluation():
    """Run evaluation system"""
    print("\n" + "="*60)
    print("🔍 EVALUATION SYSTEM")
    print("="*60)
    from src.run_eval import main as eval_main
    eval_main()


def run_monitoring():
    """Run monitoring system"""
    print("\n" + "="*60)
    print("📊 MONITORING SYSTEM")
    print("="*60)
    from src.run_monitoring import main as monitoring_main
    monitoring_main()


def run_rca():
    """Run RCA system"""
    print("\n" + "="*60)
    print("🔬 ROOT CAUSE ANALYSIS")
    print("="*60)
    from src.run_rca import main as rca_main
    rca_main()


def run_hitl():
    """Run HITL system"""
    print("\n" + "="*60)
    print("👤 HUMAN-IN-THE-LOOP")
    print("="*60)
    from src.run_hitl import main as hitl_main
    hitl_main()


def run_experimentation():
    """Run experimentation system"""
    print("\n" + "="*60)
    print("🧪 EXPERIMENTATION")
    print("="*60)
    from src.run_experimentation import main as exp_main
    exp_main()


def main():
    parser = argparse.ArgumentParser(description="Run all evaluation and improvement systems")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--monitoring", action="store_true", help="Run monitoring")
    parser.add_argument("--rca", action="store_true", help="Run RCA")
    parser.add_argument("--hitl", action="store_true", help="Run HITL")
    parser.add_argument("--experimentation", action="store_true", help="Run experimentation")
    parser.add_argument("--all", action="store_true", help="Run all systems")
    
    args = parser.parse_args()
    
    if args.all:
        run_evaluation()
        run_monitoring()
        run_rca()
        run_hitl()
        run_experimentation()
    else:
        if args.eval:
            run_evaluation()
        if args.monitoring:
            run_monitoring()
        if args.rca:
            run_rca()
        if args.hitl:
            run_hitl()
        if args.experimentation:
            run_experimentation()
        
        if not any([args.eval, args.monitoring, args.rca, args.hitl, args.experimentation]):
            print("💡 Usage:")
            print("   python src/run_all_systems.py --eval")
            print("   python src/run_all_systems.py --monitoring")
            print("   python src/run_all_systems.py --rca")
            print("   python src/run_all_systems.py --hitl")
            print("   python src/run_all_systems.py --experimentation")
            print("   python src/run_all_systems.py --all")
    
    print("\n✅ All systems executed!")


if __name__ == "__main__":
    main()

