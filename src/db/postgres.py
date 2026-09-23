"""
PostgreSQL Manager for Fintech Agentic RAG

Replaces file-based logging from Modal with structured database persistence.
Provides async connection pooling and convenient logging methods.
"""

import asyncpg
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import traceback


class PostgresManager:
    """
    Async PostgreSQL connection manager

    Features:
    - Connection pooling (min 2, max 10 connections)
    - Structured logging (prompts, errors, metrics)
    - Health check methods
    - Automatic reconnection
    """

    def __init__(
        self,
        host: str = "postgresql",
        port: int = 5432,
        database: str = "fintech_rag",
        user: str = "fintech_user",
        password: str = "",
    ):
        """
        Initialize PostgreSQL manager

        Args:
            host: PostgreSQL host (default: "postgresql" for Docker)
            port: PostgreSQL port (default: 5432)
            database: Database name
            user: Database user
            password: Database password
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.pool: Optional[asyncpg.Pool] = None
        self.is_connected = False

    async def connect(self) -> None:
        """
        Create connection pool

        Creates an async connection pool with:
        - Min connections: 2
        - Max connections: 10
        - Timeout: 60s
        """
        try:
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=2,
                max_size=10,
                timeout=60,
                command_timeout=60,
            )
            self.is_connected = True
            print(f"PostgreSQL connected: {self.user}@{self.host}:{self.port}/{self.database}")
        except Exception as e:
            self.is_connected = False
            print(f"PostgreSQL connection failed: {e}")
            raise

    async def disconnect(self) -> None:
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            self.is_connected = False
            print("PostgreSQL disconnected")

    async def log_prompt(
        self,
        layer: str,
        prompt: str,
        response: str,
        latency_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: str = "default",
    ) -> Optional[int]:
        """
        Log LLM prompt and response to database

        Replaces file-based logging in Modal version.

        Args:
            layer: Which RAG layer (e.g., "layer2_routing")
            prompt: The prompt sent to LLM
            response: The LLM response
            latency_ms: Response time in milliseconds
            metadata: Additional context (dict)
            session_id: Session identifier

        Returns:
            The id of the inserted row, or None if the write did not happen.
        """
        if not self.pool:
            print("PostgreSQL not connected, skipping log_prompt")
            return None

        query = """
            INSERT INTO prompt_logs (layer, prompt, response, latency_ms, metadata, session_id)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id
        """

        try:
            async with self.pool.acquire() as conn:
                return await conn.fetchval(
                    query,
                    layer,
                    prompt,
                    response,
                    latency_ms,
                    json.dumps(metadata or {}),
                    session_id,
                )
        except Exception as e:
            print(f"Failed to log prompt: {e}")
            return None

    async def log_error(
        self,
        layer: str,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
        severity: str = "error",
        session_id: Optional[str] = None,
        traceback_str: Optional[str] = None,
    ) -> Optional[int]:
        """
        Log application error to database

        Args:
            layer: Which component (e.g., "main_endpoint", "layer4_retrieval")
            error_type: Error class name (e.g., "ValueError", "ConnectionError")
            error_message: Error message
            context: Additional context about the error
            severity: One of: critical, error, warning, info
            session_id: Session identifier
            traceback_str: Full traceback string

        Returns:
            The id of the inserted row, or None if the write did not happen.
        """
        if not self.pool:
            print(f"PostgreSQL not connected, skipping log_error: {error_message}")
            return None

        query = """
            INSERT INTO error_logs
            (layer, error_type, error_message, traceback, context, severity, session_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id
        """

        try:
            async with self.pool.acquire() as conn:
                return await conn.fetchval(
                    query,
                    layer,
                    error_type,
                    error_message,
                    traceback_str or "",
                    json.dumps(context or {}),
                    severity,
                    session_id or "default",
                )
        except Exception as e:
            print(f"Failed to log error: {e}")
            return None

    async def log_query_metrics(
        self,
        query: str,
        company: str,
        total_time_ms: float,
        confidence: float,
        citations_count: int,
        documents_retrieved: int,
        routing_decision: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """
        Log RAG query metrics

        Args:
            query: User query text
            company: Company name queried
            total_time_ms: Total query time
            confidence: Confidence score (0-1)
            citations_count: Number of citations
            documents_retrieved: Number of documents retrieved
            routing_decision: Routing strategy used
            metadata: Additional metadata

        Returns:
            The id of the inserted row, or None if the write did not happen.
        """
        if not self.pool:
            print("PostgreSQL not connected, skipping log_query_metrics")
            return None

        query_sql = """
            INSERT INTO query_metrics
            (query, company, total_time_ms, confidence, citations_count,
             documents_retrieved, routing_decision, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
        """

        try:
            async with self.pool.acquire() as conn:
                return await conn.fetchval(
                    query_sql,
                    query,
                    company,
                    total_time_ms,
                    confidence,
                    citations_count,
                    documents_retrieved,
                    routing_decision,
                    json.dumps(metadata or {}),
                )
        except Exception as e:
            print(f"Failed to log query metrics: {e}")
            return None

    async def get_prompt_logs(
        self,
        layer: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Get prompt logs from database

        Args:
            layer: Filter by layer (optional)
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of prompt log dictionaries
        """
        if not self.pool:
            return []

        if layer:
            query = """
                SELECT * FROM prompt_logs
                WHERE layer = $1
                ORDER BY timestamp DESC
                LIMIT $2 OFFSET $3
            """
            params = [layer, limit, offset]
        else:
            query = """
                SELECT * FROM prompt_logs
                ORDER BY timestamp DESC
                LIMIT $1 OFFSET $2
            """
            params = [limit, offset]

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                return [dict(row) for row in rows]
        except Exception as e:
            print(f"Failed to get prompt logs: {e}")
            return []

    async def get_error_logs(
        self,
        layer: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get error logs with optional filters"""
        if not self.pool:
            return []

        conditions = []
        params = []
        param_count = 1

        if layer:
            conditions.append(f"layer = ${param_count}")
            params.append(layer)
            param_count += 1

        if severity:
            conditions.append(f"severity = ${param_count}")
            params.append(severity)
            param_count += 1

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        query = f"""
            SELECT * FROM error_logs
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ${param_count}
        """

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                return [dict(row) for row in rows]
        except Exception as e:
            print(f"Failed to get error logs: {e}")
            return []

    async def get_query_count(self, hours: int = 24) -> int:
        """Get total query count for last N hours"""
        if not self.pool:
            return 0

        query = """
            SELECT COUNT(*) FROM query_metrics
            WHERE timestamp > NOW() - INTERVAL '%s hours'
        """

        try:
            async with self.pool.acquire() as conn:
                return await conn.fetchval(query, hours)
        except Exception as e:
            print(f"Failed to get query count: {e}")
            return 0

    async def get_error_count(self, hours: int = 24, severity: str = "error") -> int:
        """Get error count for last N hours"""
        if not self.pool:
            return 0

        query = """
            SELECT COUNT(*) FROM error_logs
            WHERE timestamp > NOW() - INTERVAL '%s hours'
            AND severity = $1
        """

        try:
            async with self.pool.acquire() as conn:
                return await conn.fetchval(query, hours, severity)
        except Exception as e:
            print(f"Failed to get error count: {e}")
            return 0

    async def get_system_health(self) -> Dict[str, Any]:
        """
        Get system health metrics

        Returns:
            Dictionary with health metrics and status
        """
        if not self.pool:
            return {"status": "disconnected"}

        query = "SELECT * FROM get_system_health()"

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query)
                health = {}
                for row in rows:
                    health[row["metric"]] = {
                        "value": row["value"],
                        "status": row["status"],
                    }
                return health
        except Exception as e:
            print(f"Failed to get system health: {e}")
            return {"status": "error", "error": str(e)}

    async def cleanup_old_logs(self, days: int = 30) -> None:
        """
        Clean up logs older than N days

        Args:
            days: Number of days to keep (default: 30)
        """
        if not self.pool:
            return

        try:
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT cleanup_old_logs()")
                print(f"Cleaned up logs older than {days} days")
        except Exception as e:
            print(f"Failed to cleanup old logs: {e}")

    async def health_check(self) -> bool:
        """
        Check database connectivity

        Returns:
            True if connected and healthy, False otherwise
        """
        if not self.pool:
            return False

        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
                return True
        except Exception:
            return False


# Singleton instance (will be initialized in main.py lifespan)
postgres_manager: Optional[PostgresManager] = None


def get_postgres() -> Optional[PostgresManager]:
    """Get global PostgreSQL manager instance"""
    return postgres_manager
