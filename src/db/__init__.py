"""
Database module for Fintech Agentic RAG

This module provides database connectivity and management:
- PostgreSQL: For structured logging and metrics
- Redis: For caching and session management
"""

from .postgres import PostgresManager
from .redis_cache import RedisCache

__all__ = ["PostgresManager", "RedisCache"]
