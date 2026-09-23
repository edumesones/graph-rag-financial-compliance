"""
Shadow Deployments - Risk-Free Validation

Based on Notion AGENTS experimentation concept:
- Deploy new versions alongside production
- Compare performance without risk
- Validate improvements before rollout

Author: Glemes
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import time


class DeploymentStatus(Enum):
    """Deployment status"""
    SHADOW = "shadow"
    ACTIVE = "active"
    ARCHIVED = "archived"


@dataclass
class ShadowRequest:
    """Request processed in shadow mode"""
    request_id: str
    timestamp: str
    query: str
    production_result: Any
    shadow_result: Any
    production_latency_ms: float
    shadow_latency_ms: float
    comparison_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShadowDeployment:
    """Shadow deployment configuration"""
    deployment_id: str
    name: str
    version: str
    status: DeploymentStatus
    created_at: str
    production_endpoint: Callable
    shadow_endpoint: Callable
    comparison_threshold: float = 0.95
    requests: List[ShadowRequest] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ShadowDeploymentManager:
    """
    Manages shadow deployments for risk-free validation.
    
    Features:
    - Deploy new versions alongside production
    - Compare results without affecting users
    - Validate improvements before rollout
    """

    def __init__(self):
        self.deployments: Dict[str, ShadowDeployment] = {}

    def create_shadow_deployment(
        self,
        deployment_id: str,
        name: str,
        version: str,
        production_endpoint: Callable,
        shadow_endpoint: Callable,
        comparison_threshold: float = 0.95,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ShadowDeployment:
        """Create a new shadow deployment"""
        deployment = ShadowDeployment(
            deployment_id=deployment_id,
            name=name,
            version=version,
            status=DeploymentStatus.SHADOW,
            created_at=datetime.utcnow().isoformat(),
            production_endpoint=production_endpoint,
            shadow_endpoint=shadow_endpoint,
            comparison_threshold=comparison_threshold,
            metadata=metadata or {},
        )
        
        self.deployments[deployment_id] = deployment
        return deployment

    def process_request(
        self,
        deployment_id: str,
        query: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ShadowRequest:
        """
        Process a request through both production and shadow endpoints.
        
        Args:
            deployment_id: ID of shadow deployment
            query: User query
            metadata: Optional metadata
            
        Returns:
            Shadow request with comparison
        """
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment = self.deployments[deployment_id]
        
        # Process through production
        prod_start = time.time()
        prod_result = deployment.production_endpoint(query)
        prod_latency = (time.time() - prod_start) * 1000
        
        # Process through shadow
        shadow_start = time.time()
        shadow_result = deployment.shadow_endpoint(query)
        shadow_latency = (time.time() - shadow_start) * 1000
        
        # Compare results
        comparison_score = self._compare_results(prod_result, shadow_result)
        
        request = ShadowRequest(
            request_id=f"req_{int(time.time() * 1000)}",
            timestamp=datetime.utcnow().isoformat(),
            query=query,
            production_result=prod_result,
            shadow_result=shadow_result,
            production_latency_ms=prod_latency,
            shadow_latency_ms=shadow_latency,
            comparison_score=comparison_score,
            metadata=metadata or {},
        )
        
        deployment.requests.append(request)
        return request

    def get_deployment_metrics(
        self,
        deployment_id: str,
    ) -> Dict[str, Any]:
        """Get metrics for a shadow deployment"""
        if deployment_id not in self.deployments:
            return {}
        
        deployment = self.deployments[deployment_id]
        requests = deployment.requests
        
        if not requests:
            return {
                "deployment_id": deployment_id,
                "total_requests": 0,
            }
        
        total_requests = len(requests)
        avg_comparison_score = sum(r.comparison_score for r in requests) / total_requests
        
        prod_avg_latency = sum(r.production_latency_ms for r in requests) / total_requests
        shadow_avg_latency = sum(r.shadow_latency_ms for r in requests) / total_requests
        
        latency_improvement = ((prod_avg_latency - shadow_avg_latency) / prod_avg_latency) * 100
        
        passing_requests = sum(
            1 for r in requests
            if r.comparison_score >= deployment.comparison_threshold
        )
        pass_rate = passing_requests / total_requests if total_requests > 0 else 0.0
        
        return {
            "deployment_id": deployment_id,
            "name": deployment.name,
            "version": deployment.version,
            "status": deployment.status.value,
            "total_requests": total_requests,
            "average_comparison_score": avg_comparison_score,
            "production_avg_latency_ms": prod_avg_latency,
            "shadow_avg_latency_ms": shadow_avg_latency,
            "latency_improvement_pct": latency_improvement,
            "pass_rate": pass_rate,
            "meets_threshold": pass_rate >= 0.95,
        }

    def promote_to_production(
        self,
        deployment_id: str,
        min_requests: int = 100,
        min_pass_rate: float = 0.95,
    ) -> bool:
        """
        Promote shadow deployment to production if metrics meet criteria.
        
        Args:
            deployment_id: ID of shadow deployment
            min_requests: Minimum requests to evaluate
            min_pass_rate: Minimum pass rate required
            
        Returns:
            True if promoted, False otherwise
        """
        if deployment_id not in self.deployments:
            return False
        
        deployment = self.deployments[deployment_id]
        metrics = self.get_deployment_metrics(deployment_id)
        
        if metrics["total_requests"] < min_requests:
            return False
        
        if metrics["pass_rate"] < min_pass_rate:
            return False
        
        # Promote to production
        deployment.status = DeploymentStatus.ACTIVE
        return True

    def _compare_results(
        self,
        prod_result: Any,
        shadow_result: Any,
    ) -> float:
        """
        Compare production and shadow results.
        Returns similarity score (0-1).
        """
        # Simple comparison - would use semantic similarity in production
        if isinstance(prod_result, str) and isinstance(shadow_result, str):
            # String similarity
            prod_words = set(prod_result.lower().split())
            shadow_words = set(shadow_result.lower().split())
            
            if not prod_words:
                return 1.0 if not shadow_words else 0.0
            
            overlap = len(prod_words & shadow_words)
            similarity = overlap / len(prod_words)
            return similarity
        
        elif isinstance(prod_result, dict) and isinstance(shadow_result, dict):
            # Dictionary comparison
            prod_keys = set(prod_result.keys())
            shadow_keys = set(shadow_result.keys())
            
            if not prod_keys:
                return 1.0 if not shadow_keys else 0.0
            
            key_overlap = len(prod_keys & shadow_keys) / len(prod_keys)
            
            # Compare values for common keys
            value_similarities = []
            for key in prod_keys & shadow_keys:
                if prod_result[key] == shadow_result[key]:
                    value_similarities.append(1.0)
                else:
                    value_similarities.append(0.5)  # Partial match
            
            value_similarity = sum(value_similarities) / len(value_similarities) if value_similarities else 0.0
            
            return (key_overlap + value_similarity) / 2.0
        
        else:
            # Direct comparison
            return 1.0 if prod_result == shadow_result else 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all shadow deployments"""
        return {
            "total_deployments": len(self.deployments),
            "shadow_deployments": len([
                d for d in self.deployments.values()
                if d.status == DeploymentStatus.SHADOW
            ]),
            "active_deployments": len([
                d for d in self.deployments.values()
                if d.status == DeploymentStatus.ACTIVE
            ]),
            "deployments": [
                {
                    "deployment_id": d.deployment_id,
                    "name": d.name,
                    "version": d.version,
                    "status": d.status.value,
                    "total_requests": len(d.requests),
                }
                for d in self.deployments.values()
            ],
        }

    def export_deployment(self, deployment_id: str, filepath: str):
        """Export deployment data to JSON"""
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment = self.deployments[deployment_id]
        metrics = self.get_deployment_metrics(deployment_id)
        
        data = {
            "deployment": {
                "deployment_id": deployment.deployment_id,
                "name": deployment.name,
                "version": deployment.version,
                "status": deployment.status.value,
                "created_at": deployment.created_at,
                "comparison_threshold": deployment.comparison_threshold,
            },
            "metrics": metrics,
            "sample_requests": [
                {
                    "request_id": r.request_id,
                    "timestamp": r.timestamp,
                    "comparison_score": r.comparison_score,
                    "production_latency_ms": r.production_latency_ms,
                    "shadow_latency_ms": r.shadow_latency_ms,
                }
                for r in deployment.requests[-100:]  # Last 100 requests
            ],
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

