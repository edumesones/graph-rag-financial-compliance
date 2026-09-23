"""
A/B Testing - Variant Comparison

Based on Notion AGENTS experimentation concept:
- Compare multiple variants
- Statistical significance testing
- Performance metrics tracking
- Winner selection

Author: Glemes
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import random
import math


class VariantStatus(Enum):
    """Variant status"""
    ACTIVE = "active"
    WINNER = "winner"
    LOSER = "loser"
    ARCHIVED = "archived"


@dataclass
class Variant:
    """A/B test variant"""
    variant_id: str
    name: str
    traffic_percentage: float  # 0-100
    status: VariantStatus
    endpoint: Callable
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Result of a test request"""
    test_id: str
    timestamp: str
    variant_id: str
    query: str
    result: Any
    latency_ms: float
    success: bool
    user_satisfaction: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTest:
    """A/B test configuration"""
    test_id: str
    name: str
    variants: List[Variant]
    start_date: str
    end_date: Optional[str] = None
    min_sample_size: int = 100
    confidence_level: float = 0.95
    results: List[TestResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ABTestingManager:
    """
    Manages A/B tests for comparing variants.
    
    Features:
    - Traffic splitting
    - Statistical significance testing
    - Performance tracking
    - Winner selection
    """

    def __init__(self):
        self.tests: Dict[str, ABTest] = {}

    def create_test(
        self,
        test_id: str,
        name: str,
        variants: List[Dict[str, Any]],
        min_sample_size: int = 100,
        confidence_level: float = 0.95,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ABTest:
        """
        Create a new A/B test.
        
        Args:
            test_id: Unique test ID
            name: Test name
            variants: List of variant configs with 'variant_id', 'name', 'traffic_percentage', 'endpoint'
            min_sample_size: Minimum samples per variant
            confidence_level: Confidence level for significance (0-1)
            metadata: Optional metadata
        """
        # Validate traffic percentages sum to 100
        total_traffic = sum(v.get("traffic_percentage", 0) for v in variants)
        if abs(total_traffic - 100.0) > 0.01:
            raise ValueError("Traffic percentages must sum to 100")
        
        variant_objects = [
            Variant(
                variant_id=v["variant_id"],
                name=v["name"],
                traffic_percentage=v["traffic_percentage"],
                status=VariantStatus.ACTIVE,
                endpoint=v["endpoint"],
                metadata=v.get("metadata", {}),
            )
            for v in variants
        ]
        
        test = ABTest(
            test_id=test_id,
            name=name,
            variants=variant_objects,
            start_date=datetime.utcnow().isoformat(),
            min_sample_size=min_sample_size,
            confidence_level=confidence_level,
            metadata=metadata or {},
        )
        
        self.tests[test_id] = test
        return test

    def select_variant(
        self,
        test_id: str,
    ) -> Optional[str]:
        """
        Select a variant based on traffic distribution.
        
        Args:
            test_id: Test ID
            
        Returns:
            Selected variant ID
        """
        if test_id not in self.tests:
            return None
        
        test = self.tests[test_id]
        active_variants = [v for v in test.variants if v.status == VariantStatus.ACTIVE]
        
        if not active_variants:
            return None
        
        # Weighted random selection based on traffic percentage
        rand = random.random() * 100
        cumulative = 0.0
        
        for variant in active_variants:
            cumulative += variant.traffic_percentage
            if rand <= cumulative:
                return variant.variant_id
        
        # Fallback to first variant
        return active_variants[0].variant_id

    def record_result(
        self,
        test_id: str,
        variant_id: str,
        query: str,
        result: Any,
        latency_ms: float,
        success: bool,
        user_satisfaction: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TestResult:
        """Record a test result"""
        if test_id not in self.tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.tests[test_id]
        
        test_result = TestResult(
            test_id=f"result_{int(datetime.utcnow().timestamp() * 1000)}",
            timestamp=datetime.utcnow().isoformat(),
            variant_id=variant_id,
            query=query,
            result=result,
            latency_ms=latency_ms,
            success=success,
            user_satisfaction=user_satisfaction,
            metadata=metadata or {},
        )
        
        test.results.append(test_result)
        return test_result

    def get_variant_metrics(
        self,
        test_id: str,
        variant_id: str,
    ) -> Dict[str, Any]:
        """Get metrics for a specific variant"""
        if test_id not in self.tests:
            return {}
        
        test = self.tests[test_id]
        variant_results = [r for r in test.results if r.variant_id == variant_id]
        
        if not variant_results:
            return {
                "variant_id": variant_id,
                "total_samples": 0,
            }
        
        total_samples = len(variant_results)
        success_count = sum(1 for r in variant_results if r.success)
        success_rate = success_count / total_samples if total_samples > 0 else 0.0
        
        avg_latency = sum(r.latency_ms for r in variant_results) / total_samples
        avg_satisfaction = (
            sum(r.user_satisfaction for r in variant_results if r.user_satisfaction is not None) /
            sum(1 for r in variant_results if r.user_satisfaction is not None)
            if any(r.user_satisfaction is not None for r in variant_results) else None
        )
        
        return {
            "variant_id": variant_id,
            "total_samples": total_samples,
            "success_rate": success_rate,
            "avg_latency_ms": avg_latency,
            "avg_satisfaction": avg_satisfaction,
        }

    def check_significance(
        self,
        test_id: str,
    ) -> Dict[str, Any]:
        """
        Check if test has reached statistical significance.
        
        Returns:
            Dictionary with significance results
        """
        if test_id not in self.tests:
            return {}
        
        test = self.tests[test_id]
        
        if len(test.variants) < 2:
            return {"significant": False, "reason": "Need at least 2 variants"}
        
        # Get metrics for all variants
        variant_metrics = {}
        for variant in test.variants:
            metrics = self.get_variant_metrics(test_id, variant.variant_id)
            variant_metrics[variant.variant_id] = metrics
        
        # Check if all variants have minimum samples
        all_have_samples = all(
            m["total_samples"] >= test.min_sample_size
            for m in variant_metrics.values()
        )
        
        if not all_have_samples:
            return {
                "significant": False,
                "reason": "Not enough samples",
                "variant_metrics": variant_metrics,
            }
        
        # Perform statistical test (simplified - would use proper t-test in production)
        variant_ids = list(variant_metrics.keys())
        if len(variant_ids) >= 2:
            # Compare first two variants
            v1_id, v2_id = variant_ids[0], variant_ids[1]
            v1_metrics = variant_metrics[v1_id]
            v2_metrics = variant_metrics[v2_id]
            
            # Simple comparison based on success rate
            diff = abs(v1_metrics["success_rate"] - v2_metrics["success_rate"])
            
            # Calculate standard error (simplified)
            se1 = math.sqrt(v1_metrics["success_rate"] * (1 - v1_metrics["success_rate"]) / v1_metrics["total_samples"])
            se2 = math.sqrt(v2_metrics["success_rate"] * (1 - v2_metrics["success_rate"]) / v2_metrics["total_samples"])
            se_diff = math.sqrt(se1**2 + se2**2)
            
            # Z-score
            z_score = diff / se_diff if se_diff > 0 else 0
            
            # Critical value for 95% confidence (1.96)
            critical_value = 1.96
            significant = abs(z_score) > critical_value
            
            winner_id = v1_id if v1_metrics["success_rate"] > v2_metrics["success_rate"] else v2_id
            
            return {
                "significant": significant,
                "z_score": z_score,
                "difference": diff,
                "winner": winner_id,
                "variant_metrics": variant_metrics,
            }
        
        return {"significant": False, "reason": "Unable to compare variants"}

    def select_winner(
        self,
        test_id: str,
    ) -> Optional[str]:
        """
        Select winning variant based on metrics.
        
        Returns:
            Winning variant ID or None
        """
        significance = self.check_significance(test_id)
        
        if not significance.get("significant", False):
            return None
        
        winner_id = significance.get("winner")
        if winner_id:
            # Update variant statuses
            test = self.tests[test_id]
            for variant in test.variants:
                if variant.variant_id == winner_id:
                    variant.status = VariantStatus.WINNER
                else:
                    variant.status = VariantStatus.LOSER
        
        return winner_id

    def get_test_summary(self, test_id: str) -> Dict[str, Any]:
        """Get summary of A/B test"""
        if test_id not in self.tests:
            return {}
        
        test = self.tests[test_id]
        significance = self.check_significance(test_id)
        
        variant_summaries = []
        for variant in test.variants:
            metrics = self.get_variant_metrics(test_id, variant.variant_id)
            variant_summaries.append({
                "variant_id": variant.variant_id,
                "name": variant.name,
                "status": variant.status.value,
                "traffic_percentage": variant.traffic_percentage,
                **metrics,
            })
        
        return {
            "test_id": test_id,
            "name": test.name,
            "start_date": test.start_date,
            "end_date": test.end_date,
            "total_results": len(test.results),
            "significance": significance,
            "variants": variant_summaries,
        }

    def export_test(self, test_id: str, filepath: str):
        """Export test data to JSON"""
        summary = self.get_test_summary(test_id)
        
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)

