"""
Bayesian Bandits - Adaptive Optimization

Based on Notion AGENTS experimentation concept:
- Multi-armed bandit optimization
- Bayesian inference for variant selection
- Adaptive traffic allocation
- Continuous improvement

Author: Glemes
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import random
import math


class BanditStrategy(Enum):
    """Bandit strategies"""
    THOMPSON_SAMPLING = "thompson_sampling"
    EPSILON_GREEDY = "epsilon_greedy"
    UCB = "upper_confidence_bound"


@dataclass
class Arm:
    """Bandit arm (variant)"""
    arm_id: str
    name: str
    alpha: float = 1.0  # Beta distribution alpha (successes)
    beta: float = 1.0  # Beta distribution beta (failures)
    total_pulls: int = 0
    total_rewards: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_pulls == 0:
            return 0.5  # Prior
        return self.alpha / (self.alpha + self.beta)

    @property
    def expected_value(self) -> float:
        """Expected value (mean of beta distribution)"""
        return self.alpha / (self.alpha + self.beta)


@dataclass
class PullResult:
    """Result of pulling an arm"""
    pull_id: str
    timestamp: str
    arm_id: str
    reward: float  # 0-1 scale
    metadata: Dict[str, Any] = field(default_factory=dict)


class BayesianBandit:
    """
    Bayesian multi-armed bandit for adaptive optimization.
    
    Features:
    - Thompson Sampling
    - Epsilon-Greedy
    - Upper Confidence Bound (UCB)
    - Adaptive traffic allocation
    """

    def __init__(
        self,
        bandit_id: str,
        name: str,
        strategy: BanditStrategy = BanditStrategy.THOMPSON_SAMPLING,
        epsilon: float = 0.1,  # For epsilon-greedy
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.bandit_id = bandit_id
        self.name = name
        self.strategy = strategy
        self.epsilon = epsilon
        self.arms: Dict[str, Arm] = {}
        self.pull_history: List[PullResult] = []
        self.metadata = metadata or {}

    def add_arm(
        self,
        arm_id: str,
        name: str,
        alpha: float = 1.0,
        beta: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Arm:
        """Add an arm to the bandit"""
        arm = Arm(
            arm_id=arm_id,
            name=name,
            alpha=alpha,
            beta=beta,
            metadata=metadata or {},
        )
        self.arms[arm_id] = arm
        return arm

    def select_arm(self) -> str:
        """
        Select an arm based on the strategy.
        
        Returns:
            Selected arm ID
        """
        if not self.arms:
            raise ValueError("No arms available")
        
        if self.strategy == BanditStrategy.THOMPSON_SAMPLING:
            return self._thompson_sampling()
        elif self.strategy == BanditStrategy.EPSILON_GREEDY:
            return self._epsilon_greedy()
        elif self.strategy == BanditStrategy.UCB:
            return self._upper_confidence_bound()
        else:
            return random.choice(list(self.arms.keys()))

    def update_arm(
        self,
        arm_id: str,
        reward: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PullResult:
        """
        Update arm statistics after pulling.
        
        Args:
            arm_id: ID of arm that was pulled
            reward: Reward value (0-1)
            metadata: Optional metadata
            
        Returns:
            Pull result record
        """
        if arm_id not in self.arms:
            raise ValueError(f"Arm {arm_id} not found")
        
        arm = self.arms[arm_id]
        
        # Update beta distribution parameters
        if reward > 0.5:  # Success
            arm.alpha += 1.0
        else:  # Failure
            arm.beta += 1.0
        
        arm.total_pulls += 1
        arm.total_rewards += reward
        
        # Record pull result
        result = PullResult(
            pull_id=f"pull_{int(datetime.utcnow().timestamp() * 1000)}",
            timestamp=datetime.utcnow().isoformat(),
            arm_id=arm_id,
            reward=reward,
            metadata=metadata or {},
        )
        
        self.pull_history.append(result)
        return result

    def _thompson_sampling(self) -> str:
        """Thompson Sampling: Sample from posterior distributions"""
        samples = {}
        
        for arm_id, arm in self.arms.items():
            # Sample from beta distribution
            # Using approximation: sample = random.betavariate(alpha, beta)
            sample = random.betavariate(arm.alpha, arm.beta)
            samples[arm_id] = sample
        
        # Select arm with highest sample
        return max(samples.items(), key=lambda x: x[1])[0]

    def _epsilon_greedy(self) -> str:
        """Epsilon-Greedy: Explore with probability epsilon"""
        if random.random() < self.epsilon:
            # Explore: random arm
            return random.choice(list(self.arms.keys()))
        else:
            # Exploit: best arm
            return max(
                self.arms.items(),
                key=lambda x: x[1].expected_value
            )[0]

    def _upper_confidence_bound(self) -> str:
        """Upper Confidence Bound: Balance exploration and exploitation"""
        total_pulls = sum(arm.total_pulls for arm in self.arms.values())
        
        if total_pulls == 0:
            return random.choice(list(self.arms.keys()))
        
        ucb_scores = {}
        
        for arm_id, arm in self.arms.items():
            if arm.total_pulls == 0:
                ucb_scores[arm_id] = float('inf')
            else:
                # UCB formula: mean + c * sqrt(ln(total_pulls) / arm_pulls)
                c = 2.0  # Exploration constant
                exploration = c * math.sqrt(math.log(total_pulls) / arm.total_pulls)
                ucb_scores[arm_id] = arm.expected_value + exploration
        
        return max(ucb_scores.items(), key=lambda x: x[1])[0]

    def get_arm_metrics(self, arm_id: str) -> Dict[str, Any]:
        """Get metrics for an arm"""
        if arm_id not in self.arms:
            return {}
        
        arm = self.arms[arm_id]
        
        return {
            "arm_id": arm_id,
            "name": arm.name,
            "total_pulls": arm.total_pulls,
            "total_rewards": arm.total_rewards,
            "success_rate": arm.success_rate,
            "expected_value": arm.expected_value,
            "alpha": arm.alpha,
            "beta": arm.beta,
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all arms"""
        total_pulls = sum(arm.total_pulls for arm in self.arms.values())
        
        return {
            "bandit_id": self.bandit_id,
            "name": self.name,
            "strategy": self.strategy.value,
            "total_pulls": total_pulls,
            "arms": [
                self.get_arm_metrics(arm_id)
                for arm_id in self.arms.keys()
            ],
            "best_arm": self._get_best_arm(),
        }

    def _get_best_arm(self) -> Optional[str]:
        """Get the best performing arm"""
        if not self.arms:
            return None
        
        return max(
            self.arms.items(),
            key=lambda x: x[1].expected_value
        )[0]

    def get_traffic_allocation(self) -> Dict[str, float]:
        """
        Get recommended traffic allocation based on current estimates.
        Returns percentage allocation for each arm.
        """
        if not self.arms:
            return {}
        
        # Calculate allocation based on expected values
        expected_values = {
            arm_id: arm.expected_value
            for arm_id, arm in self.arms.items()
        }
        
        total_value = sum(expected_values.values())
        
        if total_value == 0:
            # Equal allocation if no data
            equal_pct = 100.0 / len(self.arms)
            return {arm_id: equal_pct for arm_id in self.arms.keys()}
        
        # Proportional allocation
        allocation = {
            arm_id: (value / total_value) * 100.0
            for arm_id, value in expected_values.items()
        }
        
        return allocation

    def export_bandit(self, filepath: str):
        """Export bandit data to JSON"""
        data = {
            "bandit_id": self.bandit_id,
            "name": self.name,
            "strategy": self.strategy.value,
            "epsilon": self.epsilon,
            "arms": [
                self.get_arm_metrics(arm_id)
                for arm_id in self.arms.keys()
            ],
            "traffic_allocation": self.get_traffic_allocation(),
            "total_pulls": sum(arm.total_pulls for arm in self.arms.values()),
            "pull_history": [
                {
                    "pull_id": p.pull_id,
                    "timestamp": p.timestamp,
                    "arm_id": p.arm_id,
                    "reward": p.reward,
                }
                for p in self.pull_history[-1000:]  # Last 1000 pulls
            ],
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


class BayesianBanditManager:
    """Manager for multiple Bayesian bandits"""

    def __init__(self):
        self.bandits: Dict[str, BayesianBandit] = {}

    def create_bandit(
        self,
        bandit_id: str,
        name: str,
        strategy: BanditStrategy = BanditStrategy.THOMPSON_SAMPLING,
        epsilon: float = 0.1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BayesianBandit:
        """Create a new bandit"""
        bandit = BayesianBandit(
            bandit_id=bandit_id,
            name=name,
            strategy=strategy,
            epsilon=epsilon,
            metadata=metadata,
        )
        self.bandits[bandit_id] = bandit
        return bandit

    def get_bandit(self, bandit_id: str) -> Optional[BayesianBandit]:
        """Get a bandit by ID"""
        return self.bandits.get(bandit_id)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all bandits"""
        return {
            "total_bandits": len(self.bandits),
            "bandits": [
                {
                    "bandit_id": b.bandit_id,
                    "name": b.name,
                    "strategy": b.strategy.value,
                    "total_arms": len(b.arms),
                    "total_pulls": sum(arm.total_pulls for arm in b.arms.values()),
                    "best_arm": b._get_best_arm(),
                }
                for b in self.bandits.values()
            ],
        }

