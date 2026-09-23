"""
Tools Refinement - Tool Optimization Pipeline

Based on Notion AGENTS feedback concept:
- Analyze tool usage patterns
- Optimize tool selection
- Improve tool performance
- Remove unused tools

Author: Glemes
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json


class ToolStatus(Enum):
    """Tool status"""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    TESTING = "testing"
    DISABLED = "disabled"


@dataclass
class ToolUsage:
    """Record of tool usage"""
    tool_id: str
    timestamp: str
    query: str
    success: bool
    latency_ms: float
    error: Optional[str] = None
    result_quality: float = 0.0  # 0-1 scale
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolMetrics:
    """Aggregated metrics for a tool"""
    tool_id: str
    total_uses: int
    success_rate: float
    avg_latency_ms: float
    avg_result_quality: float
    error_rate: float
    last_used: str
    time_window_hours: int


@dataclass
class Tool:
    """Tool definition"""
    tool_id: str
    name: str
    description: str
    status: ToolStatus
    function: Optional[Callable] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metrics: Optional[ToolMetrics] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolsRefiner:
    """
    Refines and optimizes tool usage.
    
    Features:
    - Track tool usage
    - Analyze performance
    - Optimize tool selection
    - Deprecate unused tools
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.usage_history: List[ToolUsage] = []

    def register_tool(
        self,
        tool_id: str,
        name: str,
        description: str,
        function: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tool:
        """Register a new tool"""
        tool = Tool(
            tool_id=tool_id,
            name=name,
            description=description,
            status=ToolStatus.ACTIVE,
            function=function,
            metadata=metadata or {},
        )
        
        self.tools[tool_id] = tool
        return tool

    def record_usage(
        self,
        tool_id: str,
        query: str,
        success: bool,
        latency_ms: float,
        error: Optional[str] = None,
        result_quality: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ToolUsage:
        """Record tool usage"""
        usage = ToolUsage(
            tool_id=tool_id,
            timestamp=datetime.utcnow().isoformat(),
            query=query,
            success=success,
            latency_ms=latency_ms,
            error=error,
            result_quality=result_quality,
            metadata=metadata or {},
        )
        
        self.usage_history.append(usage)
        
        # Update tool metrics
        self._update_tool_metrics(tool_id)
        
        return usage

    def get_tool_metrics(
        self,
        tool_id: str,
        time_window_hours: int = 24,
    ) -> Optional[ToolMetrics]:
        """Get metrics for a tool"""
        if tool_id not in self.tools:
            return None
        
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        recent_usage = [
            u for u in self.usage_history
            if u.tool_id == tool_id and datetime.fromisoformat(u.timestamp) >= cutoff_time
        ]
        
        if not recent_usage:
            return ToolMetrics(
                tool_id=tool_id,
                total_uses=0,
                success_rate=0.0,
                avg_latency_ms=0.0,
                avg_result_quality=0.0,
                error_rate=0.0,
                last_used="",
                time_window_hours=time_window_hours,
            )
        
        total_uses = len(recent_usage)
        success_count = sum(1 for u in recent_usage if u.success)
        success_rate = success_count / total_uses if total_uses > 0 else 0.0
        
        avg_latency = sum(u.latency_ms for u in recent_usage) / total_uses
        avg_quality = sum(u.result_quality for u in recent_usage) / total_uses if total_uses > 0 else 0.0
        
        error_count = sum(1 for u in recent_usage if u.error)
        error_rate = error_count / total_uses if total_uses > 0 else 0.0
        
        last_used = max(recent_usage, key=lambda u: u.timestamp).timestamp
        
        return ToolMetrics(
            tool_id=tool_id,
            total_uses=total_uses,
            success_rate=success_rate,
            avg_latency_ms=avg_latency,
            avg_result_quality=avg_quality,
            error_rate=error_rate,
            last_used=last_used,
            time_window_hours=time_window_hours,
        )

    def recommend_tool(
        self,
        query: str,
        available_tools: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Recommend best tool for a query.
        
        Args:
            query: User query
            available_tools: List of available tool IDs (None = all active)
            
        Returns:
            Recommended tool ID
        """
        if available_tools is None:
            available_tools = [
                tool_id for tool_id, tool in self.tools.items()
                if tool.status == ToolStatus.ACTIVE
            ]
        
        if not available_tools:
            return None
        
        # Score tools based on metrics
        tool_scores = []
        
        for tool_id in available_tools:
            metrics = self.get_tool_metrics(tool_id)
            if metrics and metrics.total_uses > 0:
                # Score = success_rate * quality - error_rate - normalized_latency
                normalized_latency = min(1.0, metrics.avg_latency_ms / 5000.0)
                score = (
                    metrics.success_rate * 0.4 +
                    metrics.avg_result_quality * 0.4 -
                    metrics.error_rate * 0.1 -
                    normalized_latency * 0.1
                )
                tool_scores.append((tool_id, score))
            else:
                # New tool, give default score
                tool_scores.append((tool_id, 0.5))
        
        if not tool_scores:
            return None
        
        # Return highest scoring tool
        best_tool_id, _ = max(tool_scores, key=lambda x: x[1])
        return best_tool_id

    def deprecate_tool(
        self,
        tool_id: str,
        reason: str = "Low usage or poor performance",
    ) -> Tool:
        """Deprecate a tool"""
        if tool_id not in self.tools:
            raise ValueError(f"Tool {tool_id} not found")
        
        tool = self.tools[tool_id]
        tool.status = ToolStatus.DEPRECATED
        tool.metadata["deprecation_reason"] = reason
        tool.metadata["deprecated_at"] = datetime.utcnow().isoformat()
        
        return tool

    def identify_unused_tools(
        self,
        time_window_hours: int = 168,  # 1 week
        min_uses: int = 5,
    ) -> List[str]:
        """Identify tools that are rarely used"""
        unused_tools = []
        
        for tool_id, tool in self.tools.items():
            if tool.status != ToolStatus.ACTIVE:
                continue
            
            metrics = self.get_tool_metrics(tool_id, time_window_hours)
            if metrics and metrics.total_uses < min_uses:
                unused_tools.append(tool_id)
        
        return unused_tools

    def identify_poor_performing_tools(
        self,
        min_success_rate: float = 0.7,
        max_error_rate: float = 0.2,
        time_window_hours: int = 24,
    ) -> List[str]:
        """Identify tools with poor performance"""
        poor_tools = []
        
        for tool_id, tool in self.tools.items():
            if tool.status != ToolStatus.ACTIVE:
                continue
            
            metrics = self.get_tool_metrics(tool_id, time_window_hours)
            if metrics and metrics.total_uses > 10:  # Only consider tools with enough usage
                if metrics.success_rate < min_success_rate or metrics.error_rate > max_error_rate:
                    poor_tools.append(tool_id)
        
        return poor_tools

    def optimize_tool_selection(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Optimize tool selection for a query.
        Returns ordered list of recommended tools.
        """
        available_tools = [
            tool_id for tool_id, tool in self.tools.items()
            if tool.status == ToolStatus.ACTIVE
        ]
        
        # Score and rank tools
        tool_scores = []
        
        for tool_id in available_tools:
            tool = self.tools[tool_id]
            metrics = self.get_tool_metrics(tool_id)
            
            # Base score from metrics
            if metrics and metrics.total_uses > 0:
                base_score = (
                    metrics.success_rate * 0.5 +
                    metrics.avg_result_quality * 0.3 -
                    metrics.error_rate * 0.2
                )
            else:
                base_score = 0.5
            
            # Relevance score (simple keyword matching - would use embeddings in production)
            relevance_score = self._calculate_relevance(query, tool.description)
            
            # Combined score
            final_score = base_score * 0.6 + relevance_score * 0.4
            
            tool_scores.append((tool_id, final_score))
        
        # Sort by score descending
        tool_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [tool_id for tool_id, _ in tool_scores]

    def _calculate_relevance(self, query: str, description: str) -> float:
        """Calculate relevance score between query and tool description"""
        query_words = set(query.lower().split())
        desc_words = set(description.lower().split())
        
        if not query_words or not desc_words:
            return 0.0
        
        overlap = len(query_words & desc_words)
        relevance = overlap / len(query_words)
        
        return min(1.0, relevance)

    def _update_tool_metrics(self, tool_id: str):
        """Update tool metrics"""
        metrics = self.get_tool_metrics(tool_id)
        if metrics:
            tool = self.tools[tool_id]
            tool.metrics = metrics

    def get_refinement_summary(self) -> Dict[str, Any]:
        """Get summary of tool refinement"""
        active_tools = [
            tool_id for tool_id, tool in self.tools.items()
            if tool.status == ToolStatus.ACTIVE
        ]
        
        unused = self.identify_unused_tools()
        poor_performing = self.identify_poor_performing_tools()
        
        return {
            "total_tools": len(self.tools),
            "active_tools": len(active_tools),
            "deprecated_tools": len([
                t for t in self.tools.values() if t.status == ToolStatus.DEPRECATED
            ]),
            "unused_tools": unused,
            "poor_performing_tools": poor_performing,
            "total_usage_records": len(self.usage_history),
            "tools": [
                {
                    "tool_id": tool_id,
                    "name": tool.name,
                    "status": tool.status.value,
                    "metrics": {
                        "total_uses": tool.metrics.total_uses if tool.metrics else 0,
                        "success_rate": tool.metrics.success_rate if tool.metrics else 0.0,
                        "avg_latency_ms": tool.metrics.avg_latency_ms if tool.metrics else 0.0,
                    } if tool.metrics else None,
                }
                for tool_id, tool in self.tools.items()
            ],
        }

    def export_tools(self, filepath: str):
        """Export tools and usage to JSON"""
        data = {
            "tools": [
                {
                    "tool_id": tool.tool_id,
                    "name": tool.name,
                    "description": tool.description,
                    "status": tool.status.value,
                    "created_at": tool.created_at,
                    "metrics": {
                        "total_uses": tool.metrics.total_uses,
                        "success_rate": tool.metrics.success_rate,
                        "avg_latency_ms": tool.metrics.avg_latency_ms,
                        "avg_result_quality": tool.metrics.avg_result_quality,
                        "error_rate": tool.metrics.error_rate,
                    } if tool.metrics else None,
                }
                for tool in self.tools.values()
            ],
            "usage_history": [
                {
                    "tool_id": u.tool_id,
                    "timestamp": u.timestamp,
                    "success": u.success,
                    "latency_ms": u.latency_ms,
                    "result_quality": u.result_quality,
                    "error": u.error,
                }
                for u in self.usage_history[-1000:]  # Last 1000 records
            ],
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

