"""
EVAL Planning - Evaluation Strategy and Metrics Definition

Based on Notion AGENTS EVAL Planning concept:
- Define evaluation strategy
- Set success criteria
- Plan test scenarios
- Define metrics to track

Author: Glemes
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class EvaluationDimension(Enum):
    """Dimensions to evaluate"""
    ACCURACY = "accuracy"
    LATENCY = "latency"
    COST = "cost"
    RELIABILITY = "reliability"
    MEMORY = "memory"
    LEARNING = "learning"


class TestScenario(Enum):
    """Types of test scenarios"""
    UNIT = "unit"
    INTEGRATION = "integration"
    END_TO_END = "end_to_end"
    STRESS = "stress"
    BOUNDARY = "boundary"


@dataclass
class EvaluationCriteria:
    """Success criteria for evaluation"""
    dimension: EvaluationDimension
    metric_name: str
    threshold: float
    operator: str = ">="  # >=, <=, ==, !=
    weight: float = 1.0  # Weight for composite scoring


@dataclass
class TestCase:
    """Individual test case definition"""
    test_id: str
    scenario: TestScenario
    input_query: str
    expected_output: str
    expected_entities: List[str] = field(default_factory=list)
    expected_relationships: List[str] = field(default_factory=list)
    max_latency_ms: float = 5000.0
    min_confidence: float = 0.7
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationPlan:
    """Complete evaluation plan"""
    plan_id: str
    name: str
    description: str
    test_cases: List[TestCase] = field(default_factory=list)
    criteria: List[EvaluationCriteria] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_test_case(self, test_case: TestCase):
        """Add a test case to the plan"""
        self.test_cases.append(test_case)

    def add_criteria(self, criteria: EvaluationCriteria):
        """Add evaluation criteria"""
        self.criteria.append(criteria)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize plan to dictionary"""
        return {
            "plan_id": self.plan_id,
            "name": self.name,
            "description": self.description,
            "test_cases": [
                {
                    "test_id": tc.test_id,
                    "scenario": tc.scenario.value,
                    "input_query": tc.input_query,
                    "expected_output": tc.expected_output,
                    "expected_entities": tc.expected_entities,
                    "expected_relationships": tc.expected_relationships,
                    "max_latency_ms": tc.max_latency_ms,
                    "min_confidence": tc.min_confidence,
                    "metadata": tc.metadata,
                }
                for tc in self.test_cases
            ],
            "criteria": [
                {
                    "dimension": c.dimension.value,
                    "metric_name": c.metric_name,
                    "threshold": c.threshold,
                    "operator": c.operator,
                    "weight": c.weight,
                }
                for c in self.criteria
            ],
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationPlan":
        """Deserialize plan from dictionary"""
        plan = cls(
            plan_id=data["plan_id"],
            name=data["name"],
            description=data["description"],
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            metadata=data.get("metadata", {}),
        )

        for tc_data in data.get("test_cases", []):
            plan.add_test_case(TestCase(
                test_id=tc_data["test_id"],
                scenario=TestScenario(tc_data["scenario"]),
                input_query=tc_data["input_query"],
                expected_output=tc_data["expected_output"],
                expected_entities=tc_data.get("expected_entities", []),
                expected_relationships=tc_data.get("expected_relationships", []),
                max_latency_ms=tc_data.get("max_latency_ms", 5000.0),
                min_confidence=tc_data.get("min_confidence", 0.7),
                metadata=tc_data.get("metadata", {}),
            ))

        for c_data in data.get("criteria", []):
            plan.add_criteria(EvaluationCriteria(
                dimension=EvaluationDimension(c_data["dimension"]),
                metric_name=c_data["metric_name"],
                threshold=c_data["threshold"],
                operator=c_data.get("operator", ">="),
                weight=c_data.get("weight", 1.0),
            ))

        return plan


class EvaluationPlanner:
    """
    Creates and manages evaluation plans.
    
    Based on AGENTS EVAL Planning:
    - Define what to measure
    - Set success thresholds
    - Plan comprehensive test scenarios
    """

    def __init__(self):
        self.plans: Dict[str, EvaluationPlan] = {}

    def create_plan(
        self,
        plan_id: str,
        name: str,
        description: str,
    ) -> EvaluationPlan:
        """Create a new evaluation plan"""
        plan = EvaluationPlan(
            plan_id=plan_id,
            name=name,
            description=description,
        )
        self.plans[plan_id] = plan
        return plan

    def get_plan(self, plan_id: str) -> Optional[EvaluationPlan]:
        """Get an evaluation plan by ID"""
        return self.plans.get(plan_id)

    def create_default_fintech_plan(self) -> EvaluationPlan:
        """Create a default evaluation plan for fintech RAG"""
        plan = self.create_plan(
            plan_id="fintech_rag_default",
            name="Default Fintech RAG Evaluation",
            description="Comprehensive evaluation plan for financial compliance RAG system",
        )

        # Add accuracy criteria
        plan.add_criteria(EvaluationCriteria(
            dimension=EvaluationDimension.ACCURACY,
            metric_name="f1_score",
            threshold=0.85,
            weight=0.4,
        ))

        plan.add_criteria(EvaluationCriteria(
            dimension=EvaluationDimension.ACCURACY,
            metric_name="entity_extraction_accuracy",
            threshold=0.90,
            weight=0.3,
        ))

        # Add latency criteria
        plan.add_criteria(EvaluationCriteria(
            dimension=EvaluationDimension.LATENCY,
            metric_name="p95_latency_ms",
            threshold=3000.0,
            operator="<=",
            weight=0.2,
        ))

        # Add reliability criteria
        plan.add_criteria(EvaluationCriteria(
            dimension=EvaluationDimension.RELIABILITY,
            metric_name="success_rate",
            threshold=0.95,
            weight=0.1,
        ))

        # Add test cases
        plan.add_test_case(TestCase(
            test_id="tc_001",
            scenario=TestScenario.END_TO_END,
            input_query="What SEC violations occurred for Tesla Inc in 2023?",
            expected_output="Should identify specific SEC violations with dates and regulations",
            expected_entities=["Tesla Inc", "SEC", "2023"],
            expected_relationships=["VIOLATED", "REGULATED_BY"],
            max_latency_ms=5000.0,
            min_confidence=0.8,
        ))

        plan.add_test_case(TestCase(
            test_id="tc_002",
            scenario=TestScenario.INTEGRATION,
            input_query="Find all companies that violated Regulation FD",
            expected_output="Should return list of companies with Regulation FD violations",
            expected_entities=["Regulation FD"],
            expected_relationships=["VIOLATED"],
            max_latency_ms=4000.0,
            min_confidence=0.75,
        ))

        plan.add_test_case(TestCase(
            test_id="tc_003",
            scenario=TestScenario.BOUNDARY,
            input_query="",  # Empty query
            expected_output="Should handle gracefully with error message",
            max_latency_ms=1000.0,
            min_confidence=0.0,
        ))

        plan.add_test_case(TestCase(
            test_id="tc_004",
            scenario=TestScenario.STRESS,
            input_query="Analyze compliance for 100 companies simultaneously",
            expected_output="Should handle batch processing without errors",
            max_latency_ms=60000.0,  # 1 minute for batch
            min_confidence=0.7,
        ))

        return plan

    def save_plan(self, plan: EvaluationPlan, filepath: str):
        """Save plan to JSON file"""
        with open(filepath, "w") as f:
            json.dump(plan.to_dict(), f, indent=2)

    def load_plan(self, filepath: str) -> EvaluationPlan:
        """Load plan from JSON file"""
        with open(filepath, "r") as f:
            data = json.load(f)
        plan = EvaluationPlan.from_dict(data)
        self.plans[plan.plan_id] = plan
        return plan

