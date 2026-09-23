"""
Experimentation System for Fintech RAG Template

Based on Notion AGENTS experimentation concept:
- Shadow Deployments: Risk-free validation
- A/B Testing: Variant comparison
- Bayesian Bandits: Adaptive optimization
"""

from .shadow_deployment import (
    ShadowDeploymentManager,
    ShadowDeployment,
    ShadowRequest,
    DeploymentStatus,
)
from .ab_testing import (
    ABTestingManager,
    ABTest,
    Variant,
    TestResult,
    VariantStatus,
)
from .bayesian_bandits import (
    BayesianBandit,
    BayesianBanditManager,
    Arm,
    PullResult,
    BanditStrategy,
)

__all__ = [
    "ShadowDeploymentManager",
    "ShadowDeployment",
    "ShadowRequest",
    "DeploymentStatus",
    "ABTestingManager",
    "ABTest",
    "Variant",
    "TestResult",
    "VariantStatus",
    "BayesianBandit",
    "BayesianBanditManager",
    "Arm",
    "PullResult",
    "BanditStrategy",
]
