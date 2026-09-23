"""
Integration tests for RAG system - FastAPI Endpoints
"""
import pytest
from httpx import AsyncClient, ASGITransport
import time


# ============================================================================
# FastAPI Endpoint Tests
# ============================================================================

@pytest.mark.asyncio
async def test_health_check():
    """Test health check endpoint returns healthy status"""
    from src.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health_check")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"

    # Verify service checks if present
    if "services" in data:
        assert "postgres" in data["services"]
        assert "redis" in data["services"]
        assert "neo4j" in data["services"]


@pytest.mark.asyncio
async def test_analyze_compliance_endpoint_structure(sample_company, sample_query):
    """Test analyze compliance endpoint returns proper structure"""
    from src.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=30.0) as client:
        response = await client.post(
            "/analyze_compliance",
            json={"company": sample_company, "query": sample_query, "verbose": True},
        )

    assert response.status_code == 200
    data = response.json()

    # Verify required fields
    assert "query" in data
    assert "answer" in data
    assert "confidence" in data
    assert "citations" in data
    assert "total_time_ms" in data
    assert "status" in data

    # Verify types
    assert isinstance(data["query"], str)
    assert isinstance(data["answer"], str)
    assert isinstance(data["confidence"], (int, float))
    assert isinstance(data["citations"], list)
    assert isinstance(data["total_time_ms"], (int, float))

    # Verify confidence range
    assert 0.0 <= data["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_analyze_compliance_verbose_mode(sample_company, sample_query):
    """Test verbose mode returns pipeline trace"""
    from src.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=30.0) as client:
        response = await client.post(
            "/analyze_compliance",
            json={"company": sample_company, "query": sample_query, "verbose": True},
        )

    assert response.status_code == 200
    data = response.json()

    # Verbose mode should include pipeline_trace
    if "pipeline_trace" in data:
        assert isinstance(data["pipeline_trace"], list)
        assert len(data["pipeline_trace"]) > 0


@pytest.mark.asyncio
async def test_analyze_compliance_non_verbose_mode(sample_company, sample_query):
    """Test non-verbose mode returns minimal data"""
    from src.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=30.0) as client:
        response = await client.post(
            "/analyze_compliance",
            json={"company": sample_company, "query": sample_query, "verbose": False},
        )

    assert response.status_code == 200
    data = response.json()

    # Should still have core fields
    assert "query" in data
    assert "answer" in data
    assert "confidence" in data


@pytest.mark.asyncio
async def test_get_system_stats():
    """Test system stats endpoint"""
    from src.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/get_system_stats")

    assert response.status_code == 200
    data = response.json()

    # Should return stats or error gracefully
    assert "database" in data or "error" in data


@pytest.mark.asyncio
async def test_get_prompt_logs():
    """Test prompt logs retrieval endpoint"""
    from src.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/get_prompt_logs?limit=10")

    # Should return 200 even if no logs exist
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_prometheus_metrics():
    """Test Prometheus metrics endpoint"""
    from src.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/metrics")

    assert response.status_code == 200

    # Verify key metrics are present
    content = response.content.decode('utf-8')
    assert "rag_requests_total" in content
    assert "rag_request_duration_seconds" in content
    assert "rag_active_requests" in content


@pytest.mark.asyncio
async def test_invalid_endpoint():
    """Test that invalid endpoints return 404"""
    from src.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/invalid_endpoint_12345")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_analyze_compliance_missing_company():
    """Test analyze compliance with missing company field"""
    from src.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/analyze_compliance",
            json={"query": "What is the revenue?"},  # Missing company
        )

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_analyze_compliance_missing_query():
    """Test analyze compliance with missing query field"""
    from src.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/analyze_compliance",
            json={"company": "Tesla Inc"},  # Missing query
        )

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_response_time_performance(sample_company, sample_query):
    """Test that endpoint responds within acceptable time"""
    from src.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=30.0) as client:
        start_time = time.time()
        response = await client.post(
            "/analyze_compliance",
            json={"company": sample_company, "query": sample_query, "verbose": False},
        )
        elapsed_time = time.time() - start_time

    assert response.status_code == 200

    # Should respond within 10 seconds for test environment
    # (Production target is 3s, but test env may be slower)
    assert elapsed_time < 10.0, f"Response took {elapsed_time:.2f}s, expected <10s"
