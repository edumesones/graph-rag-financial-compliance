"""
Error Tracker - PostgreSQL-based error tracking system (Docker Edition)

Migrated from file-based logging to PostgreSQL for:
- Structured error analysis
- Better querying capabilities
- Centralized error management
- Easier monitoring

Tracks errors by layer with:
- Error type
- Layer where it occurred
- Context
- Stack trace
- Severity level
- Frequency
"""

import asyncio
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import defaultdict
from src.db.postgres import get_postgres


class ErrorTracker:
    """
    Async PostgreSQL-based error tracker.

    Categorizes errors by layer and type for analysis.
    Replaces file-based JSONL logging from Modal version.
    """

    def __init__(self, session_id: Optional[str] = None, verbose: bool = True):
        """
        Initialize error tracker.

        Args:
            session_id: Session identifier (default: timestamp-based)
            verbose: Whether to print errors to console
        """
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.verbose = verbose
        self.errors: List[Dict[str, Any]] = []  # In-memory cache for session
        self.error_counts = defaultdict(lambda: defaultdict(int))

    async def track_error(
        self,
        layer: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        severity: str = "error",
    ) -> str:
        """
        Track an error occurrence to PostgreSQL.

        Args:
            layer: Which layer the error occurred in
            error: The exception object
            context: Additional context (query, state, etc.)
            severity: "warning", "error", or "critical"

        Returns:
            error_id: Unique identifier for this error
        """
        error_id = f"err_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        traceback_str = traceback.format_exc()

        error_entry = {
            "error_id": error_id,
            "timestamp": datetime.now().isoformat(),
            "layer": layer,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "severity": severity,
            "context": context or {},
            "traceback": traceback_str,
            "session_id": self.session_id,
        }

        # Cache in memory
        self.errors.append(error_entry)
        self.error_counts[layer][error_entry["error_type"]] += 1

        # Persist to PostgreSQL
        postgres = get_postgres()
        if postgres and postgres.is_connected:
            try:
                await postgres.log_error(
                    layer=layer,
                    error_type=type(error).__name__,
                    error_message=str(error),
                    context=context or {},
                    severity=severity,
                    session_id=self.session_id,
                    traceback_str=traceback_str,
                )
            except Exception as e:
                print(f"Failed to log error to PostgreSQL: {e}")

        # Print to console for real-time monitoring
        if self.verbose:
            self._print_error(error_entry)

        return error_id

    def _print_error(self, error_entry: Dict[str, Any]):
        """Print error to console with formatting."""
        severity_emoji = {
            "warning": "",
            "error": "",
            "critical": "",
        }.get(error_entry["severity"], "")

        print(f"\n{'='*80}")
        print(f"{severity_emoji} ERROR TRACKED - {error_entry['layer'].upper()}")
        print(f"{'='*80}")
        print(f"Error ID: {error_entry['error_id']}")
        print(f"Timestamp: {error_entry['timestamp']}")
        print(f"Type: {error_entry['error_type']}")
        print(f"Severity: {error_entry['severity'].upper()}")
        print(f"Session: {error_entry['session_id']}")

        if error_entry.get("context"):
            print(f"\nContext:")
            for key, value in error_entry["context"].items():
                value_str = str(value)[:100]
                print(f"  {key}: {value_str}")

        print(f"\nERROR MESSAGE:")
        print("-" * 80)
        print(error_entry["error_message"])
        print("-" * 80)

        print(f"\nTRACEBACK:")
        print(error_entry["traceback"][:500])
        if len(error_entry["traceback"]) > 500:
            print("... [see PostgreSQL for full traceback]")

        print(f"{'='*80}\n")

    async def get_errors(
        self,
        layer: Optional[str] = None,
        error_type: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve errors from PostgreSQL with optional filtering.

        Args:
            layer: Filter by layer
            error_type: Filter by error type (not implemented in PostgreSQL query yet)
            severity: Filter by severity
            limit: Maximum number of errors to return

        Returns:
            List of error entries from PostgreSQL
        """
        postgres = get_postgres()
        if not postgres or not postgres.is_connected:
            # Fallback to in-memory cache if PostgreSQL not available
            filtered = self.errors

            if layer:
                filtered = [err for err in filtered if err["layer"] == layer]

            if error_type:
                filtered = [err for err in filtered if err["error_type"] == error_type]

            if severity:
                filtered = [err for err in filtered if err["severity"] == severity]

            return filtered[-limit:]

        # Query from PostgreSQL
        try:
            db_errors = await postgres.get_error_logs(
                layer=layer,
                severity=severity,
                limit=limit,
            )
            return db_errors
        except Exception as e:
            print(f"Failed to get errors from PostgreSQL: {e}")
            return self.errors[-limit:]

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get error statistics.

        Returns:
            Dictionary with error stats from session cache
        """
        total = len(self.errors)

        if total == 0:
            return {
                "total_errors": 0,
                "by_layer": {},
                "by_type": {},
                "by_severity": {},
                "session_id": self.session_id,
            }

        by_severity = defaultdict(int)
        by_type = defaultdict(int)

        for err in self.errors:
            by_severity[err["severity"]] += 1
            by_type[err["error_type"]] += 1

        # Most common errors
        most_common = sorted(
            [(error_type, count) for error_type, count in by_type.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        return {
            "total_errors": total,
            "by_layer": dict(self.error_counts),
            "by_severity": dict(by_severity),
            "most_common_errors": most_common,
            "session_id": self.session_id,
            "storage": "PostgreSQL",
        }

    async def get_failure_rate(self, layer: str) -> int:
        """
        Calculate failure count for a specific layer.

        Args:
            layer: Layer name

        Returns:
            Number of errors for this layer
        """
        return sum(self.error_counts[layer].values())

    async def export_errors(self, output_file: str):
        """Export session errors to a JSON file (from in-memory cache)."""
        import json

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.errors, f, indent=2, ensure_ascii=False)
        print(f"Exported {len(self.errors)} errors to {output_file}")


# Global singleton
_global_tracker: Optional[ErrorTracker] = None


def get_error_tracker(session_id: Optional[str] = None, verbose: bool = True) -> ErrorTracker:
    """
    Get or create global error tracker instance.

    Args:
        session_id: Optional session ID (only used on first call)
        verbose: Whether to print errors to console

    Returns:
        Global ErrorTracker instance
    """
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ErrorTracker(session_id=session_id, verbose=verbose)
    return _global_tracker


def reset_error_tracker():
    """Reset global tracker (useful for testing)."""
    global _global_tracker
    _global_tracker = None


# ================================
# Synchronous Wrapper (for backward compatibility)
# ================================


def track_error_sync(
    layer: str,
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    severity: str = "error",
) -> str:
    """
    Synchronous wrapper for track_error.

    This is for backward compatibility with sync code.
    Uses asyncio to run async in background.
    """
    tracker = get_error_tracker()

    # Create task without waiting
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, schedule the coroutine
            asyncio.create_task(
                tracker.track_error(
                    layer=layer,
                    error=error,
                    context=context,
                    severity=severity,
                )
            )
        else:
            # If no loop is running, run synchronously
            loop.run_until_complete(
                tracker.track_error(
                    layer=layer,
                    error=error,
                    context=context,
                    severity=severity,
                )
            )
    except RuntimeError:
        # Fallback: create new event loop
        asyncio.run(
            tracker.track_error(
                layer=layer,
                error=error,
                context=context,
                severity=severity,
            )
        )

    return f"err_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
