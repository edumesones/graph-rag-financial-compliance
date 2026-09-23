"""
Human-in-the-Loop (HITL) - Validation Pipeline

Based on Notion AGENTS feedback concept:
- Collect human feedback
- Validate agent outputs
- Learn from corrections
- Improve over time

Author: Glemes
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class FeedbackType(Enum):
    """Types of human feedback"""
    CORRECTION = "correction"
    RATING = "rating"
    COMMENT = "comment"
    APPROVAL = "approval"
    REJECTION = "rejection"


class FeedbackStatus(Enum):
    """Status of feedback"""
    PENDING = "pending"
    PROCESSED = "processed"
    APPLIED = "applied"
    REJECTED = "rejected"


@dataclass
class HumanFeedback:
    """Human feedback record"""
    feedback_id: str
    timestamp: str
    feedback_type: FeedbackType
    query: str
    agent_output: str
    human_input: str  # Corrected output or rating/comment
    rating: Optional[int] = None  # 1-5 scale
    comment: Optional[str] = None
    status: FeedbackStatus = FeedbackStatus.PENDING
    processed_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of human validation"""
    validation_id: str
    feedback_id: str
    validated: bool
    corrections: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class HITLPipeline:
    """
    Human-in-the-Loop validation pipeline.
    
    Features:
    - Collect human feedback
    - Validate agent outputs
    - Track validation metrics
    - Generate learning signals
    """

    def __init__(self):
        self.feedback_records: List[HumanFeedback] = []
        self.validation_results: List[ValidationResult] = []

    def submit_feedback(
        self,
        query: str,
        agent_output: str,
        feedback_type: FeedbackType,
        human_input: str,
        rating: Optional[int] = None,
        comment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HumanFeedback:
        """
        Submit human feedback.
        
        Args:
            query: Original user query
            agent_output: Agent's output
            feedback_type: Type of feedback
            human_input: Human's correction/rating/comment
            rating: Optional rating (1-5)
            comment: Optional comment
            metadata: Optional metadata
        """
        feedback = HumanFeedback(
            feedback_id=f"feedback_{int(datetime.utcnow().timestamp())}",
            timestamp=datetime.utcnow().isoformat(),
            feedback_type=feedback_type,
            query=query,
            agent_output=agent_output,
            human_input=human_input,
            rating=rating,
            comment=comment,
            status=FeedbackStatus.PENDING,
            metadata=metadata or {},
        )
        
        self.feedback_records.append(feedback)
        return feedback

    def validate_output(
        self,
        feedback: HumanFeedback,
    ) -> ValidationResult:
        """
        Validate agent output based on human feedback.
        
        Args:
            feedback: Human feedback record
            
        Returns:
            Validation result
        """
        validated = False
        corrections = []
        
        if feedback.feedback_type == FeedbackType.CORRECTION:
            # Compare agent output with human correction
            validated = feedback.agent_output != feedback.human_input
            if validated:
                corrections.append(feedback.human_input)
        
        elif feedback.feedback_type == FeedbackType.APPROVAL:
            validated = True
        
        elif feedback.feedback_type == FeedbackType.REJECTION:
            validated = False
            corrections.append(feedback.comment or "Output rejected by human validator")
        
        elif feedback.feedback_type == FeedbackType.RATING:
            # Rating >= 4 is considered validated
            validated = feedback.rating is not None and feedback.rating >= 4
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(feedback)
        
        result = ValidationResult(
            validation_id=f"validation_{int(datetime.utcnow().timestamp())}",
            feedback_id=feedback.feedback_id,
            validated=validated,
            corrections=corrections,
            confidence_score=confidence_score,
        )
        
        self.validation_results.append(result)
        
        # Update feedback status
        feedback.status = FeedbackStatus.PROCESSED
        
        return result

    def get_learning_signal(
        self,
        feedback: HumanFeedback,
        validation: ValidationResult,
    ) -> Dict[str, Any]:
        """
        Extract learning signal from feedback and validation.
        
        Returns:
            Learning signal dictionary
        """
        signal = {
            "feedback_id": feedback.feedback_id,
            "query": feedback.query,
            "agent_output": feedback.agent_output,
            "validated": validation.validated,
            "corrections": validation.corrections,
            "rating": feedback.rating,
            "comment": feedback.comment,
            "confidence": validation.confidence_score,
        }
        
        # Extract specific learning points
        if feedback.feedback_type == FeedbackType.CORRECTION:
            signal["learning_type"] = "output_correction"
            signal["correct_output"] = feedback.human_input
        
        elif feedback.feedback_type == FeedbackType.RATING:
            signal["learning_type"] = "quality_rating"
            signal["quality_score"] = feedback.rating / 5.0  # Normalize to 0-1
        
        elif feedback.comment:
            signal["learning_type"] = "comment_insight"
            signal["insight"] = feedback.comment
        
        return signal

    def batch_validate(
        self,
        feedback_ids: List[str],
    ) -> List[ValidationResult]:
        """Validate multiple feedback records"""
        results = []
        
        for feedback_id in feedback_ids:
            feedback = self._get_feedback_by_id(feedback_id)
            if feedback:
                result = self.validate_output(feedback)
                results.append(result)
        
        return results

    def get_validation_metrics(
        self,
        time_window_hours: int = 24,
    ) -> Dict[str, Any]:
        """Get validation metrics"""
        from datetime import timedelta
        
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        recent_validations = [
            v for v in self.validation_results
            if datetime.fromisoformat(v.timestamp) >= cutoff_time
        ]
        
        total = len(recent_validations)
        validated = sum(1 for v in recent_validations if v.validated)
        
        avg_confidence = (
            sum(v.confidence_score for v in recent_validations) / total
            if total > 0 else 0.0
        )
        
        return {
            "total_validations": total,
            "validated_count": validated,
            "rejected_count": total - validated,
            "validation_rate": validated / total if total > 0 else 0.0,
            "average_confidence": avg_confidence,
            "time_window_hours": time_window_hours,
        }

    def _calculate_confidence(self, feedback: HumanFeedback) -> float:
        """Calculate confidence score for feedback"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on rating
        if feedback.rating is not None:
            confidence += (feedback.rating - 3) * 0.1  # -0.2 to +0.2
        
        # Increase confidence if correction is detailed
        if feedback.feedback_type == FeedbackType.CORRECTION:
            if len(feedback.human_input) > len(feedback.agent_output) * 0.8:
                confidence += 0.2
        
        # Increase confidence if comment is provided
        if feedback.comment and len(feedback.comment) > 20:
            confidence += 0.1
        
        return min(1.0, max(0.0, confidence))

    def _get_feedback_by_id(self, feedback_id: str) -> Optional[HumanFeedback]:
        """Get feedback record by ID"""
        for feedback in self.feedback_records:
            if feedback.feedback_id == feedback_id:
                return feedback
        return None

    def export_feedback(self, filepath: str):
        """Export feedback records to JSON"""
        data = {
            "feedback_records": [
                {
                    "feedback_id": f.feedback_id,
                    "timestamp": f.timestamp,
                    "feedback_type": f.feedback_type.value,
                    "query": f.query,
                    "agent_output": f.agent_output,
                    "human_input": f.human_input,
                    "rating": f.rating,
                    "comment": f.comment,
                    "status": f.status.value,
                }
                for f in self.feedback_records
            ],
            "validation_results": [
                {
                    "validation_id": v.validation_id,
                    "feedback_id": v.feedback_id,
                    "validated": v.validated,
                    "corrections": v.corrections,
                    "confidence_score": v.confidence_score,
                    "timestamp": v.timestamp,
                }
                for v in self.validation_results
            ],
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

