"""
Prompt Logger - PostgreSQL-based prompt logging system (Docker Edition)

Migrated from file-based logging to PostgreSQL for:
- Structured querying
- Better performance
- Centralized storage
- Easy analytics

Logs all LLM prompts with:
- Timestamp
- Layer that generated it
- Prompt text
- Response
- Tokens used
- Latency
- Result (success/error)
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from src.db.postgres import get_postgres


class PromptLogger:
    """
    Async PostgreSQL-based prompt logger.

    Thread-safe and with automatic database persistence.
    Replaces file-based JSONL logging from Modal version.
    """

    def __init__(self, session_id: Optional[str] = None, verbose: bool = True):
        """
        Initialize prompt logger.

        Args:
            session_id: Session identifier (default: timestamp-based)
            verbose: Whether to print logs to console
        """
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.verbose = verbose
        self.logs: List[Dict[str, Any]] = []  # In-memory cache for session

    async def log_prompt(
        self,
        layer: str,
        prompt: str,
        response: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        latency_ms: Optional[float] = None,
        tokens_used: Optional[int] = None,
    ) -> str:
        """
        Log a prompt execution to PostgreSQL.

        Args:
            layer: Which layer generated this prompt (e.g., "layer2_routing")
            prompt: The actual prompt text sent to LLM
            response: LLM response (if successful)
            metadata: Additional context (model, temperature, etc.)
            error: Error message if failed
            latency_ms: Time taken in milliseconds
            tokens_used: Token count

        Returns:
            log_id: Unique identifier for this log entry
        """
        log_id = f"{layer}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        log_entry = {
            "log_id": log_id,
            "timestamp": datetime.now().isoformat(),
            "layer": layer,
            "prompt": prompt,
            "response": response or "",
            "error": error,
            "latency_ms": latency_ms,
            "tokens_used": tokens_used,
            "metadata": metadata or {},
            "status": "success" if error is None else "error",
            "session_id": self.session_id,
        }

        # Cache in memory
        self.logs.append(log_entry)

        # Persist to PostgreSQL
        postgres = get_postgres()
        if postgres and postgres.is_connected:
            try:
                await postgres.log_prompt(
                    layer=layer,
                    prompt=prompt,
                    response=response or "",
                    latency_ms=latency_ms or 0.0,
                    metadata=metadata or {},
                    session_id=self.session_id,
                )
            except Exception as e:
                print(f"Failed to log prompt to PostgreSQL: {e}")

        # Print to console for real-time monitoring
        if self.verbose:
            self._print_log(log_entry)

        return log_id

    def _print_log(self, log_entry: Dict[str, Any]):
        """Print log entry to console with formatting."""
        status_emoji = "" if log_entry["status"] == "success" else ""
        layer = log_entry["layer"]

        print(f"\n{'='*80}")
        print(f"{status_emoji} PROMPT LOG - {layer.upper()}")
        print(f"{'='*80}")
        print(f"Timestamp: {log_entry['timestamp']}")
        print(f"Log ID: {log_entry['log_id']}")
        print(f"Session: {log_entry['session_id']}")

        if log_entry.get("metadata"):
            print(f"\nMetadata:")
            for key, value in log_entry["metadata"].items():
                print(f"  {key}: {value}")

        print(f"\nPROMPT ({len(log_entry['prompt'])} chars):")
        print("-" * 80)
        print(log_entry["prompt"][:500])
        if len(log_entry["prompt"]) > 500:
            print(f"... [truncated, full prompt in PostgreSQL]")
        print("-" * 80)

        if log_entry.get("response"):
            print(f"\nRESPONSE ({len(log_entry['response'])} chars):")
            print("-" * 80)
            print(log_entry["response"][:300])
            if len(log_entry["response"]) > 300:
                print(f"... [truncated, full response in PostgreSQL]")
            print("-" * 80)

        if log_entry.get("error"):
            print(f"\nERROR:")
            print(log_entry["error"])

        if log_entry.get("latency_ms"):
            print(f"\n⏱Latency: {log_entry['latency_ms']:.0f}ms")

        if log_entry.get("tokens_used"):
            print(f"Tokens: {log_entry['tokens_used']}")

        print(f"{'='*80}\n")

    async def get_logs(
        self, layer: Optional[str] = None, status: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve logs from PostgreSQL with optional filtering.

        Args:
            layer: Filter by layer
            status: Filter by status (success/error) - not implemented in PostgreSQL query yet
            limit: Maximum number of logs to return

        Returns:
            List of log entries from PostgreSQL
        """
        postgres = get_postgres()
        if not postgres or not postgres.is_connected:
            # Fallback to in-memory cache if PostgreSQL not available
            filtered = self.logs

            if layer:
                filtered = [log for log in filtered if log["layer"] == layer]

            if status:
                filtered = [log for log in filtered if log["status"] == status]

            return filtered[-limit:]

        # Query from PostgreSQL
        try:
            db_logs = await postgres.get_prompt_logs(layer=layer, limit=limit)
            return db_logs
        except Exception as e:
            print(f"Failed to get logs from PostgreSQL: {e}")
            return self.logs[-limit:]

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about logged prompts.

        Returns:
            Dictionary with stats (total, by layer, success rate, etc.)
        """
        # Use in-memory cache for session stats
        total = len(self.logs)

        if total == 0:
            return {
                "total_prompts": 0,
                "success_rate": 0.0,
                "by_layer": {},
                "total_errors": 0,
                "session_id": self.session_id,
            }

        success_count = sum(1 for log in self.logs if log["status"] == "success")

        by_layer = {}
        for log in self.logs:
            layer = log["layer"]
            if layer not in by_layer:
                by_layer[layer] = {"total": 0, "success": 0, "errors": 0}

            by_layer[layer]["total"] += 1
            if log["status"] == "success":
                by_layer[layer]["success"] += 1
            else:
                by_layer[layer]["errors"] += 1

        avg_latency = None
        latencies = [log["latency_ms"] for log in self.logs if log.get("latency_ms")]
        if latencies:
            avg_latency = sum(latencies) / len(latencies)

        return {
            "total_prompts": total,
            "success_rate": success_count / total if total > 0 else 0.0,
            "by_layer": by_layer,
            "total_errors": total - success_count,
            "avg_latency_ms": avg_latency,
            "session_id": self.session_id,
            "storage": "PostgreSQL",
        }

    async def export_logs(self, output_file: str):
        """Export session logs to a JSON file (from in-memory cache)."""
        import json

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False)
        print(f"Exported {len(self.logs)} logs to {output_file}")


# Global singleton instance
_global_logger: Optional[PromptLogger] = None


def get_prompt_logger(session_id: Optional[str] = None, verbose: bool = True) -> PromptLogger:
    """
    Get or create global prompt logger instance.

    Args:
        session_id: Optional session ID (only used on first call)
        verbose: Whether to print logs to console

    Returns:
        Global PromptLogger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = PromptLogger(session_id=session_id, verbose=verbose)
    return _global_logger


def reset_prompt_logger():
    """Reset global logger (useful for testing)."""
    global _global_logger
    _global_logger = None


# ================================
# Synchronous Wrapper (for backward compatibility)
# ================================


def log_prompt_sync(
    layer: str,
    prompt: str,
    response: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    latency_ms: Optional[float] = None,
    tokens_used: Optional[int] = None,
) -> str:
    """
    Synchronous wrapper for log_prompt.

    This is for backward compatibility with sync code.
    Uses asyncio.create_task to run async in background.
    """
    logger = get_prompt_logger()

    # Create task without waiting
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, schedule the coroutine
            asyncio.create_task(
                logger.log_prompt(
                    layer=layer,
                    prompt=prompt,
                    response=response,
                    metadata=metadata,
                    error=error,
                    latency_ms=latency_ms,
                    tokens_used=tokens_used,
                )
            )
        else:
            # If no loop is running, run synchronously
            loop.run_until_complete(
                logger.log_prompt(
                    layer=layer,
                    prompt=prompt,
                    response=response,
                    metadata=metadata,
                    error=error,
                    latency_ms=latency_ms,
                    tokens_used=tokens_used,
                )
            )
    except RuntimeError:
        # Fallback: create new event loop
        asyncio.run(
            logger.log_prompt(
                layer=layer,
                prompt=prompt,
                response=response,
                metadata=metadata,
                error=error,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
            )
        )

    return f"{layer}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
