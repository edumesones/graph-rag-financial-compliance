"""
Redis Cache Manager for Fintech Agentic RAG

Provides async Redis caching for:
- Query results
- Embeddings
- Routing decisions
- Retrieval responses
"""

import redis.asyncio as redis
from typing import Optional, Any, Dict
import json
import pickle
from datetime import timedelta


class RedisCache:
    """
    Async Redis cache manager

    Features:
    - Automatic serialization (JSON or pickle)
    - TTL support
    - Cache statistics
    - Health checks
    """

    def __init__(
        self,
        host: str = "redis",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: int = 3600,
    ):
        """
        Initialize Redis cache

        Args:
            host: Redis host (default: "redis" for Docker)
            port: Redis port (default: 6379)
            db: Redis database number (default: 0)
            password: Redis password (optional)
            default_ttl: Default TTL in seconds (default: 3600 = 1 hour)
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.default_ttl = default_ttl
        self.client: Optional[redis.Redis] = None
        self.is_connected = False

    async def connect(self) -> None:
        """
        Connect to Redis

        Creates async Redis client with connection pooling
        """
        try:
            url = f"redis://{self.host}:{self.port}/{self.db}"
            self.client = await redis.from_url(
                url,
                password=self.password,
                encoding="utf-8",
                decode_responses=False,  # We'll handle encoding ourselves
                max_connections=50,
                socket_timeout=5,
                socket_connect_timeout=5,
            )

            # Test connection
            await self.client.ping()
            self.is_connected = True
            print(f"✅ Redis connected: {self.host}:{self.port}/{self.db}")
        except Exception as e:
            self.is_connected = False
            print(f"❌ Redis connection failed: {e}")
            raise

    async def disconnect(self) -> None:
        """Close Redis connection"""
        if self.client:
            await self.client.close()
            self.is_connected = False
            print("✅ Redis disconnected")

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if not self.client:
            print("⚠️  Redis not connected, cache miss")
            return None

        try:
            value = await self.client.get(key)
            if value is None:
                return None

            # Try JSON first (faster), fallback to pickle
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return pickle.loads(value)

        except Exception as e:
            print(f"⚠️  Redis get error for key '{key}': {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set value in cache with TTL

        Args:
            key: Cache key
            value: Value to cache (will be serialized)
            ttl: Time to live in seconds (default: self.default_ttl)

        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            print("⚠️  Redis not connected, skipping set")
            return False

        try:
            # Try JSON serialization first (more efficient)
            try:
                serialized = json.dumps(value)
            except (TypeError, ValueError):
                # Fallback to pickle for complex objects
                serialized = pickle.dumps(value)

            ttl_seconds = ttl if ttl is not None else self.default_ttl

            await self.client.setex(
                key,
                timedelta(seconds=ttl_seconds),
                serialized,
            )
            return True

        except Exception as e:
            print(f"⚠️  Redis set error for key '{key}': {e}")
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete key from cache

        Args:
            key: Cache key to delete

        Returns:
            True if deleted, False otherwise
        """
        if not self.client:
            return False

        try:
            await self.client.delete(key)
            return True
        except Exception as e:
            print(f"⚠️  Redis delete error for key '{key}': {e}")
            return False

    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache

        Args:
            key: Cache key

        Returns:
            True if exists, False otherwise
        """
        if not self.client:
            return False

        try:
            return await self.client.exists(key) > 0
        except Exception as e:
            print(f"⚠️  Redis exists error for key '{key}': {e}")
            return False

    async def get_ttl(self, key: str) -> Optional[int]:
        """
        Get TTL for a key

        Args:
            key: Cache key

        Returns:
            TTL in seconds or None if key doesn't exist
        """
        if not self.client:
            return None

        try:
            ttl = await self.client.ttl(key)
            return ttl if ttl > 0 else None
        except Exception as e:
            print(f"⚠️  Redis get_ttl error for key '{key}': {e}")
            return None

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dictionary with cache stats (hits, misses, keys, memory, etc.)
        """
        if not self.client:
            return {"status": "disconnected"}

        try:
            info = await self.client.info("stats")
            memory_info = await self.client.info("memory")
            db_size = await self.client.dbsize()

            # Calculate hit rate
            hits = info.get("keyspace_hits", 0)
            misses = info.get("keyspace_misses", 0)
            total_requests = hits + misses
            hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0.0

            return {
                "status": "connected",
                "total_keys": db_size,
                "hits": hits,
                "misses": misses,
                "hit_rate_percent": round(hit_rate, 2),
                "used_memory": memory_info.get("used_memory_human", "unknown"),
                "used_memory_peak": memory_info.get("used_memory_peak_human", "unknown"),
                "evicted_keys": info.get("evicted_keys", 0),
                "expired_keys": info.get("expired_keys", 0),
            }
        except Exception as e:
            print(f"⚠️  Failed to get cache stats: {e}")
            return {"status": "error", "error": str(e)}

    async def clear_all(self) -> bool:
        """
        Clear all keys from current database

        ⚠️ WARNING: Use with caution! This deletes all keys.

        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False

        try:
            await self.client.flushdb()
            print("⚠️  Redis: All keys cleared from current database")
            return True
        except Exception as e:
            print(f"⚠️  Redis flushdb error: {e}")
            return False

    async def health_check(self) -> bool:
        """
        Check Redis connectivity

        Returns:
            True if connected and healthy, False otherwise
        """
        if not self.client:
            return False

        try:
            await self.client.ping()
            return True
        except Exception:
            return False

    async def get_memory_usage(self, key: str) -> Optional[int]:
        """
        Get memory usage for a specific key

        Args:
            key: Cache key

        Returns:
            Memory usage in bytes or None if error
        """
        if not self.client:
            return None

        try:
            return await self.client.memory_usage(key)
        except Exception as e:
            print(f"⚠️  Redis memory_usage error for key '{key}': {e}")
            return None

    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increment a counter

        Args:
            key: Counter key
            amount: Amount to increment (default: 1)

        Returns:
            New value or None if error
        """
        if not self.client:
            return None

        try:
            return await self.client.incrby(key, amount)
        except Exception as e:
            print(f"⚠️  Redis increment error for key '{key}': {e}")
            return None

    async def get_keys_by_pattern(self, pattern: str) -> list:
        """
        Get all keys matching pattern

        Args:
            pattern: Key pattern (e.g., "query:*", "embedding:*")

        Returns:
            List of matching keys
        """
        if not self.client:
            return []

        try:
            keys = []
            async for key in self.client.scan_iter(match=pattern):
                keys.append(key.decode("utf-8") if isinstance(key, bytes) else key)
            return keys
        except Exception as e:
            print(f"⚠️  Redis scan error for pattern '{pattern}': {e}")
            return []


# Singleton instance (will be initialized in main.py lifespan)
redis_cache: Optional[RedisCache] = None


def get_redis() -> Optional[RedisCache]:
    """Get global Redis cache instance"""
    return redis_cache
