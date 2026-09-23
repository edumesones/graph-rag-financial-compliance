"""
Prompt Refinement - Iterative Prompt Improvement

Based on Notion AGENTS feedback concept:
- Analyze prompt performance
- Generate prompt variations
- A/B test prompts
- Learn optimal prompts

Author: Glemes
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import hashlib


class PromptVersion(Enum):
    """Prompt version status"""
    DRAFT = "draft"
    TESTING = "testing"
    ACTIVE = "active"
    ARCHIVED = "archived"


@dataclass
class Prompt:
    """Prompt definition"""
    prompt_id: str
    version: int
    content: str
    status: PromptVersion
    created_at: str
    performance_score: float = 0.0
    usage_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize prompt to dictionary"""
        return {
            "prompt_id": self.prompt_id,
            "version": self.version,
            "content": self.content,
            "status": self.status.value,
            "created_at": self.created_at,
            "performance_score": self.performance_score,
            "usage_count": self.usage_count,
            "metadata": self.metadata,
        }


@dataclass
class PromptPerformance:
    """Performance metrics for a prompt"""
    prompt_id: str
    version: int
    accuracy: float = 0.0
    latency_ms: float = 0.0
    user_satisfaction: float = 0.0
    token_usage: int = 0
    error_rate: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class PromptRefiner:
    """
    Refines prompts iteratively based on performance.
    
    Features:
    - Track prompt versions
    - Measure performance
    - Generate variations
    - Select best prompts
    """

    def __init__(self):
        self.prompts: Dict[str, List[Prompt]] = {}  # prompt_id -> versions
        self.performance_history: List[PromptPerformance] = []
        self.active_prompts: Dict[str, Prompt] = {}  # prompt_id -> active version

    def create_prompt(
        self,
        prompt_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Prompt:
        """Create a new prompt"""
        prompt = Prompt(
            prompt_id=prompt_id,
            version=1,
            content=content,
            status=PromptVersion.DRAFT,
            created_at=datetime.utcnow().isoformat(),
            metadata=metadata or {},
        )
        
        if prompt_id not in self.prompts:
            self.prompts[prompt_id] = []
        
        self.prompts[prompt_id].append(prompt)
        return prompt

    def create_variation(
        self,
        prompt_id: str,
        variation_strategy: str = "improve_clarity",
        base_version: Optional[int] = None,
    ) -> Prompt:
        """
        Create a variation of an existing prompt.
        
        Args:
            prompt_id: ID of base prompt
            variation_strategy: Strategy for variation
            base_version: Version to base variation on (defaults to latest)
        """
        if prompt_id not in self.prompts:
            raise ValueError(f"Prompt {prompt_id} not found")
        
        base_prompt = self._get_prompt_version(prompt_id, base_version)
        if not base_prompt:
            raise ValueError(f"Prompt version not found")
        
        # Generate variation based on strategy
        new_content = self._generate_variation(
            base_prompt.content,
            variation_strategy,
        )
        
        new_version = Prompt(
            prompt_id=prompt_id,
            version=len(self.prompts[prompt_id]) + 1,
            content=new_content,
            status=PromptVersion.DRAFT,
            created_at=datetime.utcnow().isoformat(),
            metadata={
                "base_version": base_prompt.version,
                "variation_strategy": variation_strategy,
            },
        )
        
        self.prompts[prompt_id].append(new_version)
        return new_version

    def record_performance(
        self,
        prompt_id: str,
        version: int,
        accuracy: float,
        latency_ms: float,
        user_satisfaction: float = 0.0,
        token_usage: int = 0,
        error_rate: float = 0.0,
    ) -> PromptPerformance:
        """Record performance metrics for a prompt"""
        performance = PromptPerformance(
            prompt_id=prompt_id,
            version=version,
            accuracy=accuracy,
            latency_ms=latency_ms,
            user_satisfaction=user_satisfaction,
            token_usage=token_usage,
            error_rate=error_rate,
        )
        
        self.performance_history.append(performance)
        
        # Update prompt performance score
        prompt = self._get_prompt_version(prompt_id, version)
        if prompt:
            prompt.performance_score = self._calculate_performance_score(performance)
            prompt.usage_count += 1
        
        return performance

    def select_best_prompt(
        self,
        prompt_id: str,
        metric_weights: Optional[Dict[str, float]] = None,
    ) -> Optional[Prompt]:
        """
        Select the best performing prompt version.
        
        Args:
            prompt_id: ID of prompt to evaluate
            metric_weights: Weights for different metrics
        """
        if prompt_id not in self.prompts:
            return None
        
        versions = self.prompts[prompt_id]
        if not versions:
            return None
        
        # Calculate scores for each version
        version_scores = []
        
        for version in versions:
            # Get performance history for this version
            version_performances = [
                p for p in self.performance_history
                if p.prompt_id == prompt_id and p.version == version.version
            ]
            
            if version_performances:
                avg_performance = self._average_performance(version_performances)
                score = self._calculate_performance_score(
                    avg_performance,
                    metric_weights,
                )
                version_scores.append((version, score))
            else:
                # No performance data, use default score
                version_scores.append((version, 0.0))
        
        # Select best version
        best_version, best_score = max(version_scores, key=lambda x: x[1])
        
        return best_version

    def activate_prompt(
        self,
        prompt_id: str,
        version: int,
    ) -> Prompt:
        """Activate a prompt version"""
        prompt = self._get_prompt_version(prompt_id, version)
        if not prompt:
            raise ValueError(f"Prompt version not found")
        
        # Deactivate current active version
        if prompt_id in self.active_prompts:
            self.active_prompts[prompt_id].status = PromptVersion.ARCHIVED
        
        # Activate new version
        prompt.status = PromptVersion.ACTIVE
        self.active_prompts[prompt_id] = prompt
        
        return prompt

    def get_active_prompt(self, prompt_id: str) -> Optional[Prompt]:
        """Get the currently active prompt"""
        return self.active_prompts.get(prompt_id)

    def _generate_variation(
        self,
        base_content: str,
        strategy: str,
    ) -> str:
        """Generate a prompt variation"""
        strategies = {
            "improve_clarity": self._improve_clarity,
            "add_examples": self._add_examples,
            "simplify": self._simplify,
            "add_constraints": self._add_constraints,
            "enhance_formatting": self._enhance_formatting,
        }
        
        variation_fn = strategies.get(strategy, self._improve_clarity)
        return variation_fn(base_content)

    def _improve_clarity(self, content: str) -> str:
        """Improve clarity of prompt"""
        # Simple improvements - would use LLM in production
        improvements = [
            ("be concise", "be concise and specific"),
            ("find", "identify and extract"),
            ("analyze", "analyze and summarize"),
        ]
        
        result = content
        for old, new in improvements:
            if old in result.lower():
                result = result.replace(old, new)
        
        return result

    def _add_examples(self, content: str) -> str:
        """Add examples to prompt"""
        if "Example:" not in content:
            content += "\n\nExample: [Add example here]"
        return content

    def _simplify(self, content: str) -> str:
        """Simplify prompt"""
        # Remove complex phrases
        simplifications = [
            ("in order to", "to"),
            ("with the purpose of", "to"),
            ("it is important to note that", ""),
        ]
        
        result = content
        for old, new in simplifications:
            result = result.replace(old, new)
        
        return result.strip()

    def _add_constraints(self, content: str) -> str:
        """Add constraints to prompt"""
        if "Constraints:" not in content:
            content += "\n\nConstraints: Be factual and cite sources."
        return content

    def _enhance_formatting(self, content: str) -> str:
        """Enhance formatting"""
        # Add structure if missing
        if not content.startswith("#"):
            content = f"# Task\n\n{content}"
        return content

    def _get_prompt_version(
        self,
        prompt_id: str,
        version: Optional[int] = None,
    ) -> Optional[Prompt]:
        """Get a specific prompt version"""
        if prompt_id not in self.prompts:
            return None
        
        versions = self.prompts[prompt_id]
        
        if version is None:
            return versions[-1] if versions else None
        
        for v in versions:
            if v.version == version:
                return v
        
        return None

    def _calculate_performance_score(
        self,
        performance: PromptPerformance,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Calculate overall performance score"""
        if weights is None:
            weights = {
                "accuracy": 0.4,
                "user_satisfaction": 0.3,
                "latency": 0.2,
                "error_rate": 0.1,
            }
        
        # Normalize metrics (higher is better)
        accuracy_score = performance.accuracy
        satisfaction_score = performance.user_satisfaction
        latency_score = max(0, 1.0 - (performance.latency_ms / 10000.0))  # Normalize to 0-1
        error_score = max(0, 1.0 - performance.error_rate)
        
        score = (
            accuracy_score * weights["accuracy"] +
            satisfaction_score * weights["user_satisfaction"] +
            latency_score * weights["latency"] +
            error_score * weights["error_rate"]
        )
        
        return score

    def _average_performance(
        self,
        performances: List[PromptPerformance],
    ) -> PromptPerformance:
        """Calculate average performance from multiple records"""
        if not performances:
            return PromptPerformance(
                prompt_id="",
                version=0,
            )
        
        return PromptPerformance(
            prompt_id=performances[0].prompt_id,
            version=performances[0].version,
            accuracy=sum(p.accuracy for p in performances) / len(performances),
            latency_ms=sum(p.latency_ms for p in performances) / len(performances),
            user_satisfaction=sum(p.user_satisfaction for p in performances) / len(performances),
            token_usage=sum(p.token_usage for p in performances) // len(performances),
            error_rate=sum(p.error_rate for p in performances) / len(performances),
        )

    def get_refinement_summary(self, prompt_id: str) -> Dict[str, Any]:
        """Get summary of prompt refinement"""
        if prompt_id not in self.prompts:
            return {}
        
        versions = self.prompts[prompt_id]
        active = self.active_prompts.get(prompt_id)
        
        return {
            "prompt_id": prompt_id,
            "total_versions": len(versions),
            "active_version": active.version if active else None,
            "versions": [
                {
                    "version": v.version,
                    "status": v.status.value,
                    "performance_score": v.performance_score,
                    "usage_count": v.usage_count,
                }
                for v in versions
            ],
        }

    def export_prompts(self, filepath: str):
        """Export prompts to JSON"""
        data = {
            "prompts": {
                prompt_id: [v.to_dict() for v in versions]
                for prompt_id, versions in self.prompts.items()
            },
            "active_prompts": {
                prompt_id: prompt.to_dict()
                for prompt_id, prompt in self.active_prompts.items()
            },
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

