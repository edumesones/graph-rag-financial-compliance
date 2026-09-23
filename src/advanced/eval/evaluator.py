"""
Agent Evaluation System - Based on AGENTS EVAL concepts from Notion

Implements:
1. EVAL Planning - Strategy and metrics
2. EVAL Memory - Memory capacity and boundary testing  
3. EVAL Learning - Learning loop verification

Author: Glemes
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import time

from .planning import EvaluationPlan, EvaluationPlanner, TestCase, EvaluationCriteria
from .memory import MemoryEvaluator, MemoryTestResult
from .learning import LearningEvaluator, LearningTestResult


@dataclass
class EvaluationMetrics:
    """Metrics for agent evaluation"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    latency_ms: float = 0.0
    token_usage: int = 0
    tool_calls: int = 0
    success_rate: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class EvaluationResult:
    """Result of an evaluation run"""
    test_case_id: str
    input_query: str
    expected_output: str
    actual_output: str
    metrics: EvaluationMetrics
    passed: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentEvaluator:
    """
    Evaluates agent performance across multiple dimensions.
    
    Based on AGENTS EVAL from Notion:
    - Planning: Define what to measure
    - Memory: Test storage/retrieval accuracy
    - Learning: Verify parameter updates
    """
    
    def __init__(
        self,
        llm_endpoint: Callable,
        graph_query_fn: Callable,
        vector_search_fn: Callable,
        memory_store: Optional[Any] = None,
        graph_store: Optional[Any] = None,
    ):
        self.llm_endpoint = llm_endpoint
        self.graph_query_fn = graph_query_fn
        self.vector_search_fn = vector_search_fn
        
        # Initialize sub-evaluators
        self.planner = EvaluationPlanner()
        self.memory_evaluator = None
        self.learning_evaluator = None
        
        if memory_store and graph_store:
            self.memory_evaluator = MemoryEvaluator(memory_store, graph_store)
        
        self.results: List[EvaluationResult] = []

    def evaluate_with_plan(
        self,
        plan: EvaluationPlan,
    ) -> List[EvaluationResult]:
        """
        Run evaluation using an evaluation plan.
        
        Args:
            plan: Evaluation plan with test cases and criteria
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for test_case in plan.test_cases:
            result = self.evaluate_test_case(test_case)
            results.append(result)
        
        self.results.extend(results)
        return results

    def evaluate_test_case(
        self,
        test_case: TestCase,
    ) -> EvaluationResult:
        """
        Evaluate a single test case.
        
        Args:
            test_case: Test case to evaluate
            
        Returns:
            Evaluation result
        """
        start_time = time.time()
        
        try:
            # Execute agent query
            actual_output = self.llm_endpoint(test_case.input_query)
            
            # Calculate metrics
            execution_time = (time.time() - start_time) * 1000
            
            # Simple accuracy check (would use LLM evaluation in production)
            passed = self._check_accuracy(
                test_case.expected_output,
                actual_output,
                test_case.min_confidence,
            )
            
            metrics = EvaluationMetrics(
                accuracy=1.0 if passed else 0.0,
                latency_ms=execution_time,
                success_rate=1.0 if passed else 0.0,
            )
            
            result = EvaluationResult(
                test_case_id=test_case.test_id,
                input_query=test_case.input_query,
                expected_output=test_case.expected_output,
                actual_output=actual_output,
                metrics=metrics,
                passed=passed and execution_time <= test_case.max_latency_ms,
                metadata={
                    "scenario": test_case.scenario.value,
                    "max_latency_ms": test_case.max_latency_ms,
                }
            )
            
        except Exception as e:
            result = EvaluationResult(
                test_case_id=test_case.test_id,
                input_query=test_case.input_query,
                expected_output=test_case.expected_output,
                actual_output="",
                metrics=EvaluationMetrics(),
                passed=False,
                error=str(e),
            )
        
        return result

    def run_memory_tests(self) -> List[MemoryTestResult]:
        """Run memory evaluation tests"""
        if not self.memory_evaluator:
            raise ValueError("Memory evaluator not initialized")
        
        results = []
        
        # Test capacity
        capacity_result = self.memory_evaluator.test_storage_capacity(
            max_items=1000,
            item_generator=lambda i: {"id": f"test_{i}", "content": f"Test content {i}"},
        )
        results.append(capacity_result)
        
        # Test retrieval accuracy
        retrieval_result = self.memory_evaluator.test_retrieval_accuracy([
            ("Tesla Inc violations", ["entity_1", "entity_2"]),
            ("SEC regulations", ["reg_1", "reg_2"]),
        ])
        results.append(retrieval_result)
        
        # Test boundary cases
        boundary_results = self.memory_evaluator.test_boundary_cases()
        results.extend(boundary_results)
        
        # Test persistence
        persistence_result = self.memory_evaluator.test_persistence()
        results.append(persistence_result)
        
        return results

    def run_learning_tests(
        self,
        get_parameters_fn: Callable[[], Dict[str, Any]],
        evaluate_performance_fn: Callable[[], float],
        update_parameters_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[LearningTestResult]:
        """Run learning evaluation tests"""
        if not self.learning_evaluator:
            self.learning_evaluator = LearningEvaluator(
                get_parameters_fn=get_parameters_fn,
                evaluate_performance_fn=evaluate_performance_fn,
                update_parameters_fn=update_parameters_fn,
            )
        
        results = []
        
        # Test parameter updates
        param_result = self.learning_evaluator.test_parameter_updates(
            initial_params={"temperature": 0.1},
            update_params={"temperature": 0.2},
        )
        results.append(param_result)
        
        # Test learning improvement
        improvement_result = self.learning_evaluator.test_learning_improvement(
            num_iterations=10,
        )
        results.append(improvement_result)
        
        # Test feedback learning
        feedback_result = self.learning_evaluator.test_feedback_learning(
            feedback_samples=[
                {"query": "test", "rating": 5, "comment": "good"},
                {"query": "test2", "rating": 4, "comment": "ok"},
            ],
        )
        results.append(feedback_result)
        
        # Test degradation
        degradation_result = self.learning_evaluator.test_learning_degradation(
            num_iterations=20,
        )
        results.append(degradation_result)
        
        return results

    def _check_accuracy(
        self,
        expected: str,
        actual: str,
        min_confidence: float,
    ) -> bool:
        """
        Check if actual output matches expected output.
        Simple implementation - would use LLM evaluation in production.
        """
        # Simple keyword matching
        expected_keywords = set(expected.lower().split())
        actual_keywords = set(actual.lower().split())
        
        if len(expected_keywords) == 0:
            return True
        
        overlap = len(expected_keywords & actual_keywords)
        similarity = overlap / len(expected_keywords)
        
        return similarity >= min_confidence

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluation results"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        
        avg_latency = sum(r.metrics.latency_ms for r in self.results) / total_tests if total_tests > 0 else 0.0
        avg_accuracy = sum(r.metrics.accuracy for r in self.results) / total_tests if total_tests > 0 else 0.0
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
            "average_latency_ms": avg_latency,
            "average_accuracy": avg_accuracy,
            "results": [
                {
                    "test_case_id": r.test_case_id,
                    "passed": r.passed,
                    "latency_ms": r.metrics.latency_ms,
                    "accuracy": r.metrics.accuracy,
                    "error": r.error,
                }
                for r in self.results
            ]
        }
        
        # Add memory test summary if available
        if self.memory_evaluator:
            summary["memory_tests"] = self.memory_evaluator.get_summary()
        
        # Add learning test summary if available
        if self.learning_evaluator:
            summary["learning_tests"] = self.learning_evaluator.get_summary()
        
        return summary

    def export_results(self, filepath: str):
        """Export evaluation results to JSON"""
        summary = self.get_summary()
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)
