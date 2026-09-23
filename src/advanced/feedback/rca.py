"""
Root Cause Analysis (RCA) - Continuous Analysis Pipeline

Based on Notion AGENTS feedback concept:
- Analyze failures and errors
- Identify root causes
- Generate actionable insights
- Track patterns over time

Author: Glemes
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json


class FailureCategory(Enum):
    """Categories of failures"""
    LLM_ERROR = "llm_error"
    TOOL_ERROR = "tool_error"
    TIMEOUT = "timeout"
    DATA_QUALITY = "data_quality"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class Severity(Enum):
    """Severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FailureEvent:
    """Record of a failure event"""
    event_id: str
    timestamp: str
    category: FailureCategory
    severity: Severity
    error_message: str
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    user_query: Optional[str] = None
    agent_state: Optional[Dict[str, Any]] = None


@dataclass
class RootCause:
    """Identified root cause"""
    cause_id: str
    category: FailureCategory
    description: str
    confidence: float
    affected_events: List[str] = field(default_factory=list)
    frequency: int = 0
    first_seen: str = ""
    last_seen: str = ""
    suggested_fix: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RCAAnalysis:
    """Complete RCA analysis result"""
    analysis_id: str
    timestamp: str
    events_analyzed: int
    root_causes: List[RootCause]
    patterns: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RootCauseAnalyzer:
    """
    Performs root cause analysis on failure events.
    
    Features:
    - Pattern detection
    - Frequency analysis
    - Correlation identification
    - Actionable recommendations
    """

    def __init__(self):
        self.events: List[FailureEvent] = []
        self.root_causes: Dict[str, RootCause] = {}
        self.analyses: List[RCAAnalysis] = []

    def record_failure(
        self,
        error_message: str,
        category: FailureCategory,
        severity: Severity,
        context: Optional[Dict[str, Any]] = None,
        stack_trace: Optional[str] = None,
        user_query: Optional[str] = None,
        agent_state: Optional[Dict[str, Any]] = None,
    ) -> FailureEvent:
        """Record a failure event"""
        event = FailureEvent(
            event_id=f"event_{int(datetime.utcnow().timestamp())}",
            timestamp=datetime.utcnow().isoformat(),
            category=category,
            severity=severity,
            error_message=error_message,
            stack_trace=stack_trace,
            context=context or {},
            user_query=user_query,
            agent_state=agent_state,
        )
        self.events.append(event)
        return event

    def analyze_failures(
        self,
        time_window_hours: int = 24,
        min_frequency: int = 2,
    ) -> RCAAnalysis:
        """
        Analyze failures in a time window and identify root causes.
        
        Args:
            time_window_hours: Hours to look back
            min_frequency: Minimum frequency to consider a pattern
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        recent_events = [
            e for e in self.events
            if datetime.fromisoformat(e.timestamp) >= cutoff_time
        ]
        
        # Group events by category and error pattern
        event_groups: Dict[str, List[FailureEvent]] = {}
        
        for event in recent_events:
            # Create a key based on category and error pattern
            error_pattern = self._extract_error_pattern(event.error_message)
            key = f"{event.category.value}:{error_pattern}"
            
            if key not in event_groups:
                event_groups[key] = []
            event_groups[key].append(event)
        
        # Identify root causes
        identified_causes = []
        
        for key, events in event_groups.items():
            if len(events) >= min_frequency:
                category_str, pattern = key.split(":", 1)
                category = FailureCategory(category_str)
                
                cause = self._identify_root_cause(
                    category=category,
                    pattern=pattern,
                    events=events,
                )
                
                if cause.cause_id not in self.root_causes:
                    self.root_causes[cause.cause_id] = cause
                
                identified_causes.append(cause)
        
        # Detect patterns
        patterns = self._detect_patterns(recent_events)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(identified_causes, patterns)
        
        analysis = RCAAnalysis(
            analysis_id=f"rca_{int(datetime.utcnow().timestamp())}",
            timestamp=datetime.utcnow().isoformat(),
            events_analyzed=len(recent_events),
            root_causes=identified_causes,
            patterns=patterns,
            recommendations=recommendations,
        )
        
        self.analyses.append(analysis)
        return analysis

    def _extract_error_pattern(self, error_message: str) -> str:
        """Extract error pattern from error message"""
        # Simple pattern extraction - would use NLP in production
        error_lower = error_message.lower()
        
        # Common patterns
        if "timeout" in error_lower:
            return "timeout"
        elif "connection" in error_lower or "network" in error_lower:
            return "connection_error"
        elif "authentication" in error_lower or "unauthorized" in error_lower:
            return "auth_error"
        elif "not found" in error_lower or "404" in error_lower:
            return "not_found"
        elif "rate limit" in error_lower:
            return "rate_limit"
        elif "invalid" in error_lower or "malformed" in error_lower:
            return "invalid_input"
        else:
            # Use first 50 chars as pattern
            return error_message[:50].replace(" ", "_")

    def _identify_root_cause(
        self,
        category: FailureCategory,
        pattern: str,
        events: List[FailureEvent],
    ) -> RootCause:
        """Identify root cause from events"""
        cause_id = f"{category.value}_{pattern}"
        
        # Analyze events to determine root cause
        first_event = min(events, key=lambda e: e.timestamp)
        last_event = max(events, key=lambda e: e.timestamp)
        
        # Determine confidence based on frequency and consistency
        confidence = min(1.0, len(events) / 10.0)  # Max confidence at 10+ events
        
        # Generate suggested fix based on category
        suggested_fix = self._suggest_fix(category, pattern, events)
        
        cause = RootCause(
            cause_id=cause_id,
            category=category,
            description=f"{category.value}: {pattern}",
            confidence=confidence,
            affected_events=[e.event_id for e in events],
            frequency=len(events),
            first_seen=first_event.timestamp,
            last_seen=last_event.timestamp,
            suggested_fix=suggested_fix,
            metadata={
                "pattern": pattern,
                "severity_distribution": self._get_severity_distribution(events),
            }
        )
        
        return cause

    def _suggest_fix(
        self,
        category: FailureCategory,
        pattern: str,
        events: List[FailureEvent],
    ) -> str:
        """Generate suggested fix based on root cause"""
        suggestions = {
            FailureCategory.LLM_ERROR: "Check LLM endpoint configuration and API key validity",
            FailureCategory.TOOL_ERROR: "Verify tool configuration and dependencies",
            FailureCategory.TIMEOUT: "Increase timeout settings or optimize query complexity",
            FailureCategory.DATA_QUALITY: "Validate input data and check data source integrity",
            FailureCategory.CONFIGURATION: "Review configuration parameters and environment variables",
            FailureCategory.UNKNOWN: "Review logs and error traces for more details",
        }
        
        base_suggestion = suggestions.get(category, "Review error details and system logs")
        
        # Add pattern-specific suggestions
        if "timeout" in pattern:
            return f"{base_suggestion}. Consider implementing query caching or reducing query complexity."
        elif "connection" in pattern:
            return f"{base_suggestion}. Check network connectivity and service availability."
        elif "rate_limit" in pattern:
            return f"{base_suggestion}. Implement rate limiting and retry logic with exponential backoff."
        
        return base_suggestion

    def _get_severity_distribution(self, events: List[FailureEvent]) -> Dict[str, int]:
        """Get severity distribution for events"""
        distribution = {}
        for event in events:
            severity = event.severity.value
            distribution[severity] = distribution.get(severity, 0) + 1
        return distribution

    def _detect_patterns(self, events: List[FailureEvent]) -> List[Dict[str, Any]]:
        """Detect patterns in failure events"""
        patterns = []
        
        # Pattern 1: Time-based patterns
        if len(events) > 5:
            hourly_distribution = {}
            for event in events:
                hour = datetime.fromisoformat(event.timestamp).hour
                hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1
            
            peak_hour = max(hourly_distribution.items(), key=lambda x: x[1])
            if peak_hour[1] > len(events) * 0.3:  # More than 30% in one hour
                patterns.append({
                    "type": "temporal_clustering",
                    "description": f"Failures cluster around hour {peak_hour[0]}",
                    "confidence": peak_hour[1] / len(events),
                })
        
        # Pattern 2: Category patterns
        category_distribution = {}
        for event in events:
            category = event.category.value
            category_distribution[category] = category_distribution.get(category, 0) + 1
        
        dominant_category = max(category_distribution.items(), key=lambda x: x[1])
        if dominant_category[1] > len(events) * 0.5:  # More than 50% in one category
            patterns.append({
                "type": "category_dominance",
                "description": f"Most failures are {dominant_category[0]}",
                "confidence": dominant_category[1] / len(events),
            })
        
        return patterns

    def _generate_recommendations(
        self,
        root_causes: List[RootCause],
        patterns: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # High-frequency root causes
        high_freq_causes = [c for c in root_causes if c.frequency >= 5]
        if high_freq_causes:
            recommendations.append(
                f"Address {len(high_freq_causes)} high-frequency root causes: "
                f"{', '.join([c.description for c in high_freq_causes[:3]])}"
            )
        
        # Critical severity issues
        critical_events = [e for e in self.events if e.severity == Severity.CRITICAL]
        if critical_events:
            recommendations.append(
                f"Immediate attention required: {len(critical_events)} critical failures detected"
            )
        
        # Pattern-based recommendations
        for pattern in patterns:
            if pattern["type"] == "temporal_clustering":
                recommendations.append(
                    f"Investigate system load during peak failure hour: {pattern['description']}"
                )
        
        return recommendations

    def get_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get summary of RCA analysis"""
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        recent_events = [
            e for e in self.events
            if datetime.fromisoformat(e.timestamp) >= cutoff_time
        ]
        
        return {
            "total_events": len(self.events),
            "recent_events": len(recent_events),
            "root_causes_identified": len(self.root_causes),
            "analyses_performed": len(self.analyses),
            "recent_root_causes": [
                {
                    "cause_id": c.cause_id,
                    "description": c.description,
                    "frequency": c.frequency,
                    "confidence": c.confidence,
                }
                for c in list(self.root_causes.values())[-10:]  # Last 10
            ],
        }

    def export_analysis(self, analysis: RCAAnalysis, filepath: str):
        """Export analysis to JSON"""
        data = {
            "analysis_id": analysis.analysis_id,
            "timestamp": analysis.timestamp,
            "events_analyzed": analysis.events_analyzed,
            "root_causes": [
                {
                    "cause_id": c.cause_id,
                    "category": c.category.value,
                    "description": c.description,
                    "confidence": c.confidence,
                    "frequency": c.frequency,
                    "suggested_fix": c.suggested_fix,
                }
                for c in analysis.root_causes
            ],
            "patterns": analysis.patterns,
            "recommendations": analysis.recommendations,
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

