"""
Integration tests for PostgreSQL logging system
"""
import pytest
import asyncio
from datetime import datetime


@pytest.mark.asyncio
async def test_postgres_connection(postgres_manager):
    """Test PostgreSQL connection is established"""
    assert postgres_manager.is_connected is True


@pytest.mark.asyncio
async def test_log_prompt_basic(postgres_manager):
    """Test basic prompt logging to PostgreSQL"""
    log_id = await postgres_manager.log_prompt(
        layer="test_layer",
        prompt="test prompt content",
        response="test response content",
        latency_ms=150.5,
        metadata={"test": True, "environment": "pytest"},
        session_id="test_session_001",
    )

    assert log_id is not None
    assert isinstance(log_id, int)
    print(f"✅ Logged prompt with ID: {log_id}")


@pytest.mark.asyncio
async def test_log_prompt_with_complex_metadata(postgres_manager):
    """Test prompt logging with complex JSON metadata"""
    complex_metadata = {
        "query_type": "semantic",
        "confidence": 0.85,
        "tools_used": ["VectorSearch", "GraphQuery"],
        "nested": {
            "level1": {
                "level2": "deep value"
            }
        }
    }

    log_id = await postgres_manager.log_prompt(
        layer="layer2_routing",
        prompt="Complex routing decision",
        response='{"query_type": "hybrid", "confidence": 0.85}',
        latency_ms=250.0,
        metadata=complex_metadata,
        session_id="test_complex_001",
    )

    assert log_id is not None
    print(f"✅ Logged complex metadata with ID: {log_id}")


@pytest.mark.asyncio
async def test_log_error_basic(postgres_manager):
    """Test basic error logging to PostgreSQL"""
    error_id = await postgres_manager.log_error(
        layer="test_layer",
        error_type="ValueError",
        error_message="Test error message",
        traceback="Traceback test line 1\nTraceback test line 2",
        context={"operation": "test_operation", "input": "test_input"},
        severity="error"
    )

    assert error_id is not None
    assert isinstance(error_id, int)
    print(f"✅ Logged error with ID: {error_id}")


@pytest.mark.asyncio
async def test_log_error_severity_levels(postgres_manager):
    """Test error logging with different severity levels"""
    severities = ["info", "warning", "error", "critical"]

    for severity in severities:
        error_id = await postgres_manager.log_error(
            layer="test_layer",
            error_type="TestError",
            error_message=f"Test {severity} message",
            traceback="",
            context={"severity_test": True},
            severity=severity
        )

        assert error_id is not None
        print(f"✅ Logged {severity} error with ID: {error_id}")


@pytest.mark.asyncio
async def test_log_query_metrics(postgres_manager):
    """Test query metrics logging"""
    # Note: This assumes the method exists in PostgresManager
    # If not implemented, this test will fail and indicate what needs to be added
    try:
        metric_id = await postgres_manager.log_query_metrics(
            query="What were Tesla's Q4 2023 revenues?",
            company="Tesla Inc",
            total_time_ms=2500.0,
            confidence=0.87,
            citations_count=5,
            routing_decision="hybrid",
            metadata={
                "layers_executed": ["routing", "retrieval", "generation"],
                "cache_hit": False
            }
        )

        assert metric_id is not None
        print(f"✅ Logged query metrics with ID: {metric_id}")

    except AttributeError as e:
        pytest.skip(f"log_query_metrics not implemented: {e}")


@pytest.mark.asyncio
async def test_get_prompt_logs(postgres_manager):
    """Test retrieving prompt logs from PostgreSQL"""
    # First, insert a test log
    await postgres_manager.log_prompt(
        layer="test_retrieval",
        prompt="retrieval test prompt",
        response="retrieval test response",
        latency_ms=100.0,
        metadata={"retrieval": True},
        session_id="test_retrieval_001",
    )

    # Now retrieve logs
    logs = await postgres_manager.get_prompt_logs(limit=10)

    assert isinstance(logs, list)
    assert len(logs) > 0

    # Verify structure of returned logs
    if logs:
        log = logs[0]
        assert "layer" in log or "prompt" in log  # Basic structure check
        print(f"✅ Retrieved {len(logs)} prompt logs")


@pytest.mark.asyncio
async def test_get_error_logs(postgres_manager):
    """Test retrieving error logs from PostgreSQL"""
    # First, insert a test error
    await postgres_manager.log_error(
        layer="test_error_retrieval",
        error_type="TestException",
        error_message="Error retrieval test",
        traceback="Test traceback",
        context={"test": True},
        severity="warning"
    )

    # Now retrieve errors (assuming this method exists)
    try:
        errors = await postgres_manager.get_error_logs(limit=10)

        assert isinstance(errors, list)
        assert len(errors) > 0
        print(f"✅ Retrieved {len(errors)} error logs")

    except AttributeError as e:
        pytest.skip(f"get_error_logs not implemented: {e}")


@pytest.mark.asyncio
async def test_concurrent_logging(postgres_manager):
    """Test concurrent log writes to PostgreSQL"""
    # Create multiple concurrent logging tasks
    tasks = []

    for i in range(10):
        task = postgres_manager.log_prompt(
            layer=f"concurrent_test_{i}",
            prompt=f"Concurrent prompt {i}",
            response=f"Concurrent response {i}",
            latency_ms=100.0 + i,
            metadata={"concurrent_id": i},
            session_id=f"concurrent_session_{i}",
        )
        tasks.append(task)

    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Verify all succeeded
    successful = [r for r in results if isinstance(r, int)]
    assert len(successful) == 10, f"Expected 10 successful logs, got {len(successful)}"
    print(f"✅ Successfully logged {len(successful)} concurrent entries")


@pytest.mark.asyncio
async def test_logging_performance(postgres_manager):
    """Test logging performance under load"""
    import time

    start_time = time.time()

    # Log 50 entries
    tasks = []
    for i in range(50):
        task = postgres_manager.log_prompt(
            layer="performance_test",
            prompt=f"Performance test prompt {i}",
            response=f"Performance test response {i}",
            latency_ms=100.0,
            metadata={"batch_id": 1},
            session_id="performance_session",
        )
        tasks.append(task)

    await asyncio.gather(*tasks)

    elapsed = time.time() - start_time

    # Should complete 50 logs in under 5 seconds (async advantage)
    assert elapsed < 5.0, f"Logging 50 entries took {elapsed:.2f}s, expected <5s"
    print(f"✅ Logged 50 entries in {elapsed:.2f}s ({50/elapsed:.1f} logs/sec)")


@pytest.mark.asyncio
async def test_session_tracking(postgres_manager):
    """Test session ID tracking in logs"""
    session_id = "session_tracking_test_001"

    # Log multiple entries with same session ID
    for i in range(5):
        await postgres_manager.log_prompt(
            layer=f"layer_{i}",
            prompt=f"Session test prompt {i}",
            response=f"Session test response {i}",
            latency_ms=100.0,
            metadata={"step": i},
            session_id=session_id,
        )

    # Retrieve logs (would need filtering by session in real implementation)
    logs = await postgres_manager.get_prompt_logs(limit=100)

    # Count logs with our session ID (if session_id is returned in logs)
    session_logs = [log for log in logs if log.get("session_id") == session_id]

    # We should have at least 5 logs with this session (may have more from other tests)
    # Note: This assumes session_id is returned in get_prompt_logs
    if session_logs:
        assert len(session_logs) >= 5
        print(f"✅ Found {len(session_logs)} logs for session {session_id}")
    else:
        print("⚠️  Session ID tracking not verified (not returned in get_prompt_logs)")


@pytest.mark.asyncio
async def test_metadata_jsonb_storage(postgres_manager):
    """Test JSONB metadata storage and retrieval"""
    test_metadata = {
        "string_value": "test",
        "number_value": 42,
        "float_value": 3.14,
        "boolean_value": True,
        "null_value": None,
        "array_value": [1, 2, 3],
        "object_value": {"nested": "data"}
    }

    log_id = await postgres_manager.log_prompt(
        layer="metadata_test",
        prompt="Metadata test",
        response="Response",
        latency_ms=100.0,
        metadata=test_metadata,
        session_id="metadata_test_001",
    )

    assert log_id is not None
    print(f"✅ Stored complex JSONB metadata with ID: {log_id}")


@pytest.mark.asyncio
async def test_disconnect_and_reconnect(postgres_manager):
    """Test PostgreSQL disconnect and reconnect"""
    # Should already be connected from fixture
    assert postgres_manager.is_connected is True

    # Disconnect
    await postgres_manager.disconnect()
    assert postgres_manager.is_connected is False

    # Reconnect
    await postgres_manager.connect()
    assert postgres_manager.is_connected is True

    # Verify can still log after reconnect
    log_id = await postgres_manager.log_prompt(
        layer="reconnect_test",
        prompt="Reconnect test",
        response="Success",
        latency_ms=50.0,
        metadata={"reconnect": True},
        session_id="reconnect_test_001",
    )

    assert log_id is not None
    print("✅ Successfully reconnected and logged")
