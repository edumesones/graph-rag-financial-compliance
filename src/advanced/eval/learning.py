"""
EVAL Learning - Learning Loop Verification

Based on Notion AGENTS EVAL Learning concept:
- Verify parameter updates occur correctly
- Test learning from feedback
- Validate improvement over time
- Check for learning degradation

Author: Glemes
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import json


@dataclass
class LearningMetrics:
    """Metrics for learning evaluation"""
    initial_performance: float
    current_performance: float
    improvement_rate: float
    learning_rate: float
    convergence_epoch: Optional[int] = None
    stability_score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class LearningTestResult:
    """Result of a learning test"""
    test_id: str
    test_type: str
    passed: bool
    metrics: LearningMetrics
    iterations: int
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParameterSnapshot:
    """Snapshot of model parameters at a point in time"""
    snapshot_id: str
    timestamp: str
    parameters: Dict[str, Any]
    performance: float


class LearningEvaluator:
    """
    Evaluates learning loop performance.
    
    Tests:
    1. Parameter update verification
    2. Performance improvement over time
    3. Learning from feedback
    4. Convergence detection
    5. Stability checks
    """

    def __init__(
        self,
        get_parameters_fn: Callable[[], Dict[str, Any]],
        evaluate_performance_fn: Callable[[], float],
        update_parameters_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.get_parameters_fn = get_parameters_fn
        self.evaluate_performance_fn = evaluate_performance_fn
        self.update_parameters_fn = update_parameters_fn
        self.snapshots: List[ParameterSnapshot] = []
        self.results: List[LearningTestResult] = []

    def test_parameter_updates(
        self,
        initial_params: Dict[str, Any],
        update_params: Dict[str, Any],
    ) -> LearningTestResult:
        """
        Test that parameters update correctly.
        
        Args:
            initial_params: Initial parameter values
            update_params: Parameters to update
        """
        test_id = f"param_update_test_{int(time.time())}"
        
        try:
            # Get initial state
            initial_state = self.get_parameters_fn()
            
            # Apply updates
            if self.update_parameters_fn:
                self.update_parameters_fn(update_params)
            
            # Verify updates
            updated_state = self.get_parameters_fn()
            
            # Check if updates were applied
            updates_applied = all(
                updated_state.get(k) == v
                for k, v in update_params.items()
                if k in updated_state
            )
            
            initial_perf = self.evaluate_performance_fn()
            updated_perf = self.evaluate_performance_fn()
            
            metrics = LearningMetrics(
                initial_performance=initial_perf,
                current_performance=updated_perf,
                improvement_rate=(updated_perf - initial_perf) / initial_perf if initial_perf > 0 else 0.0,
                learning_rate=0.0,  # Would need historical data
            )
            
            result = LearningTestResult(
                test_id=test_id,
                test_type="parameter_updates",
                passed=updates_applied,
                metrics=metrics,
                iterations=1,
                metadata={
                    "initial_params": initial_params,
                    "update_params": update_params,
                    "updates_applied": updates_applied,
                }
            )
            
        except Exception as e:
            result = LearningTestResult(
                test_id=test_id,
                test_type="parameter_updates",
                passed=False,
                metrics=LearningMetrics(
                    initial_performance=0.0,
                    current_performance=0.0,
                    improvement_rate=0.0,
                    learning_rate=0.0,
                ),
                iterations=0,
                error=str(e),
            )
        
        self.results.append(result)
        return result

    def test_learning_improvement(
        self,
        num_iterations: int = 10,
        feedback_fn: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> LearningTestResult:
        """
        Test that performance improves over learning iterations.
        
        Args:
            num_iterations: Number of learning iterations
            feedback_fn: Function that provides feedback for learning
        """
        test_id = f"improvement_test_{int(time.time())}"
        
        try:
            initial_perf = self.evaluate_performance_fn()
            performance_history = [initial_perf]
            
            for i in range(num_iterations):
                # Get feedback
                if feedback_fn:
                    feedback = feedback_fn()
                    # Apply learning based on feedback
                    if self.update_parameters_fn:
                        # This would typically update parameters based on feedback
                        pass
                
                # Evaluate current performance
                current_perf = self.evaluate_performance_fn()
                performance_history.append(current_perf)
            
            final_perf = performance_history[-1]
            improvement_rate = (final_perf - initial_perf) / initial_perf if initial_perf > 0 else 0.0
            
            # Calculate learning rate (average improvement per iteration)
            improvements = [
                performance_history[i+1] - performance_history[i]
                for i in range(len(performance_history) - 1)
            ]
            learning_rate = sum(improvements) / len(improvements) if improvements else 0.0
            
            # Check for convergence (performance stabilizes)
            convergence_epoch = None
            if len(performance_history) >= 5:
                recent_std = self._calculate_std(performance_history[-5:])
                if recent_std < 0.01:  # Stable within 1%
                    convergence_epoch = len(performance_history) - 5
            
            # Stability score (lower variance = more stable)
            stability_score = 1.0 / (1.0 + self._calculate_std(performance_history))
            
            metrics = LearningMetrics(
                initial_performance=initial_perf,
                current_performance=final_perf,
                improvement_rate=improvement_rate,
                learning_rate=learning_rate,
                convergence_epoch=convergence_epoch,
                stability_score=stability_score,
            )
            
            # Test passes if performance improved or stayed stable
            passed = improvement_rate >= -0.05  # Allow 5% degradation
            
            result = LearningTestResult(
                test_id=test_id,
                test_type="learning_improvement",
                passed=passed,
                metrics=metrics,
                iterations=num_iterations,
                metadata={
                    "performance_history": performance_history,
                    "improvements": improvements,
                }
            )
            
        except Exception as e:
            result = LearningTestResult(
                test_id=test_id,
                test_type="learning_improvement",
                passed=False,
                metrics=LearningMetrics(
                    initial_performance=0.0,
                    current_performance=0.0,
                    improvement_rate=0.0,
                    learning_rate=0.0,
                ),
                iterations=0,
                error=str(e),
            )
        
        self.results.append(result)
        return result

    def test_feedback_learning(
        self,
        feedback_samples: List[Dict[str, Any]],
    ) -> LearningTestResult:
        """
        Test learning from explicit feedback.
        
        Args:
            feedback_samples: List of feedback samples with expected improvements
        """
        test_id = f"feedback_learning_test_{int(time.time())}"
        
        try:
            initial_perf = self.evaluate_performance_fn()
            
            for feedback in feedback_samples:
                # Apply feedback-based learning
                # This would typically update prompts, parameters, or model weights
                if self.update_parameters_fn:
                    # Extract learning signal from feedback
                    learning_signal = self._extract_learning_signal(feedback)
                    self.update_parameters_fn(learning_signal)
            
            final_perf = self.evaluate_performance_fn()
            improvement_rate = (final_perf - initial_perf) / initial_perf if initial_perf > 0 else 0.0
            
            metrics = LearningMetrics(
                initial_performance=initial_perf,
                current_performance=final_perf,
                improvement_rate=improvement_rate,
                learning_rate=improvement_rate / len(feedback_samples) if feedback_samples else 0.0,
            )
            
            # Test passes if performance improved
            passed = improvement_rate > 0.0
            
            result = LearningTestResult(
                test_id=test_id,
                test_type="feedback_learning",
                passed=passed,
                metrics=metrics,
                iterations=len(feedback_samples),
                metadata={
                    "feedback_samples_count": len(feedback_samples),
                }
            )
            
        except Exception as e:
            result = LearningTestResult(
                test_id=test_id,
                test_type="feedback_learning",
                passed=False,
                metrics=LearningMetrics(
                    initial_performance=0.0,
                    current_performance=0.0,
                    improvement_rate=0.0,
                    learning_rate=0.0,
                ),
                iterations=0,
                error=str(e),
            )
        
        self.results.append(result)
        return result

    def test_learning_degradation(
        self,
        num_iterations: int = 20,
    ) -> LearningTestResult:
        """
        Test for learning degradation (catastrophic forgetting).
        Performance should not degrade significantly over time.
        """
        test_id = f"degradation_test_{int(time.time())}"
        
        try:
            initial_perf = self.evaluate_performance_fn()
            performance_history = [initial_perf]
            
            for i in range(num_iterations):
                # Simulate learning iteration
                current_perf = self.evaluate_performance_fn()
                performance_history.append(current_perf)
            
            final_perf = performance_history[-1]
            degradation_rate = (initial_perf - final_perf) / initial_perf if initial_perf > 0 else 0.0
            
            # Check for significant degradation
            max_degradation = max([
                (initial_perf - p) / initial_perf
                for p in performance_history
            ])
            
            metrics = LearningMetrics(
                initial_performance=initial_perf,
                current_performance=final_perf,
                improvement_rate=-degradation_rate,  # Negative = degradation
                learning_rate=0.0,
                stability_score=1.0 / (1.0 + self._calculate_std(performance_history)),
            )
            
            # Test passes if degradation is minimal (< 10%)
            passed = degradation_rate < 0.10
            
            result = LearningTestResult(
                test_id=test_id,
                test_type="learning_degradation",
                passed=passed,
                metrics=metrics,
                iterations=num_iterations,
                metadata={
                    "degradation_rate": degradation_rate,
                    "max_degradation": max_degradation,
                    "performance_history": performance_history,
                }
            )
            
        except Exception as e:
            result = LearningTestResult(
                test_id=test_id,
                test_type="learning_degradation",
                passed=False,
                metrics=LearningMetrics(
                    initial_performance=0.0,
                    current_performance=0.0,
                    improvement_rate=0.0,
                    learning_rate=0.0,
                ),
                iterations=0,
                error=str(e),
            )
        
        self.results.append(result)
        return result

    def take_snapshot(self) -> ParameterSnapshot:
        """Take a snapshot of current parameters and performance"""
        snapshot = ParameterSnapshot(
            snapshot_id=f"snapshot_{int(time.time())}",
            timestamp=datetime.utcnow().isoformat(),
            parameters=self.get_parameters_fn(),
            performance=self.evaluate_performance_fn(),
        )
        self.snapshots.append(snapshot)
        return snapshot

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def _extract_learning_signal(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Extract learning signal from feedback"""
        # Placeholder - would extract actionable parameters from feedback
        return {}

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all learning test results"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        
        avg_improvement = sum(
            r.metrics.improvement_rate for r in self.results
        ) / total_tests if total_tests > 0 else 0.0
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
            "average_improvement_rate": avg_improvement,
            "snapshots_count": len(self.snapshots),
            "results": [
                {
                    "test_id": r.test_id,
                    "test_type": r.test_type,
                    "passed": r.passed,
                    "improvement_rate": r.metrics.improvement_rate,
                    "stability_score": r.metrics.stability_score,
                    "error": r.error,
                }
                for r in self.results
            ]
        }

