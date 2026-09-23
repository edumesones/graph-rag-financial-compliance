"""
Tests for utility modules
"""
import pytest


def test_config_loading():
    """Test configuration loading"""
    from src.utils.config import get_config

    config = get_config()

    assert config is not None
    assert hasattr(config, "postgres")
    assert hasattr(config, "redis")
    assert hasattr(config, "neo4j")


@pytest.mark.asyncio
async def test_postgres_manager(postgres_manager):
    """Test PostgreSQL manager basic operations"""
    # Test connection
    assert postgres_manager.is_connected is True

    # Test logging (basic)
    log_id = await postgres_manager.log_prompt(
        layer="other",
        prompt="test prompt",
        response="test response",
        latency_ms=100.0,
        metadata={"test": True},
        session_id="test_session",
    )

    assert log_id is not None


@pytest.mark.asyncio
async def test_redis_cache(redis_cache):
    """Test Redis cache basic operations"""
    # Test set/get
    key = "test_key"
    value = {"test": "data", "number": 42}

    success = await redis_cache.set(key, value, ttl=60)
    assert success is True

    retrieved = await redis_cache.get(key)
    assert retrieved == value

    # Test delete
    deleted = await redis_cache.delete(key)
    assert deleted is True

    # Verify deletion
    retrieved_after_delete = await redis_cache.get(key)
    assert retrieved_after_delete is None
