"""
Script para ejecutar Human-in-the-Loop (HITL) feedback

Uso:
    python src/run_hitl.py
    python src/run_hitl.py --submit-feedback "query" "agent_output" "human_correction"
    python src/run_hitl.py --list-feedback
"""

import os
import sys
import argparse
import json

# This file lives at <repo>/scripts/runners/, so the repo root is three levels up.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.advanced.feedback import HITLPipeline, FeedbackType


def main():
    parser = argparse.ArgumentParser(description="Run Human-in-the-Loop feedback")
    parser.add_argument("--submit-feedback", nargs=4, 
                       metavar=("QUERY", "AGENT_OUTPUT", "HUMAN_INPUT", "RATING"),
                       help="Submit human feedback")
    parser.add_argument("--list-feedback", action="store_true", help="List all feedback")
    parser.add_argument("--validate", type=str, help="Validate feedback by ID")
    parser.add_argument("--export", type=str, help="Export feedback to file")
    
    args = parser.parse_args()
    
    print("👤 Initializing HITL system...")
    
    hitl = HITLPipeline()
    
    if args.submit_feedback:
        query, agent_output, human_input, rating = args.submit_feedback
        
        feedback = hitl.submit_feedback(
            query=query,
            agent_output=agent_output,
            feedback_type=FeedbackType.CORRECTION,
            human_input=human_input,
            rating=int(rating) if rating.isdigit() else None,
        )
        
        print(f"✅ Feedback submitted: {feedback.feedback_id}")
        print(f"   Query: {query[:60]}...")
        print(f"   Rating: {feedback.rating}")
    
    if args.list_feedback:
        feedbacks = hitl.feedback_records[-10:]  # Last 10
        print(f"\n📋 Recent Feedback ({len(feedbacks)}):")
        for fb in feedbacks:
            print(f"   [{fb.feedback_id}] {fb.query[:50]}...")
            print(f"      Rating: {fb.rating}, Type: {fb.feedback_type.value}")
    
    if args.validate:
        # Find feedback by ID
        feedback = next((f for f in hitl.feedback_records if f.feedback_id == args.validate), None)
        if feedback:
            result = hitl.validate_output(feedback)
            print(f"\n✅ Validation Result:")
            print(f"   Valid: {result.is_valid}")
            print(f"   Confidence: {result.confidence_score:.2f}")
        else:
            print(f"❌ Feedback not found: {args.validate}")
    
    if args.export:
        feedbacks = hitl.feedback_records
        output = {
            "total_feedback": len(feedbacks),
            "feedback": [
                {
                    "id": fb.feedback_id,
                    "query": fb.query,
                    "agent_output": fb.agent_output,
                    "human_input": fb.human_input,
                    "rating": fb.rating,
                    "type": fb.feedback_type.value,
                    "timestamp": fb.timestamp,
                }
                for fb in feedbacks
            ]
        }
        with open(args.export, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n💾 Feedback exported to: {args.export}")
    
    if not any([args.submit_feedback, args.list_feedback, args.validate, args.export]):
        print("\n💡 Usage examples:")
        print("   python src/run_hitl.py --submit-feedback 'query' 'agent output' 'correction' '4'")
        print("   python src/run_hitl.py --list-feedback")
        print("   python src/run_hitl.py --validate feedback_123")
        print("   python src/run_hitl.py --export feedback.json")


if __name__ == "__main__":
    main()

