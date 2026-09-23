"""
Integration tests for Redis caching system
"""
import pytest
import asyncio
import time
import json


@pytest.mark.asyncio
async def test_redis_connection(redis_cache):
    """Test Redis connection is established"""
    # Connection should be established by fixture
    # Try a simple ping operation
    try:
        await redis_cache.redis.ping()
        print("✅ Redis connection successful")
    except Exception as e:
        pytest.fail(f"Redis connection failed: {e}")


@pytest.mark.asyncio
async def test_basic_set_get(redis_cache):
    """Test basic set and get operations"""
    key = "test_key_001"
    value = {"message": "Hello Redis", "timestamp": time.time()}

    # Set value
    await redis_cache.set(key, value, ttl=60)

    # Get value
    retrieved = await redis_cache.get(key)

    assert retrieved is not None
    assert retrieved["message"] == value["message"]
    print(f"✅ Set and retrieved value: {retrieved}")


@pytest.mark.asyncio
async def test_set_with_ttl(redis_cache):
    """Test that TTL expiration works"""
    key = "test_ttl_key"
    value = {"data": "expires_soon"}

    # Set with 2 second TTL
    await redis_cache.set(key, value, ttl=2)

    # Should exist immediately
    retrieved = await redis_cache.get(key)
    assert retrieved is not None
    print("✅ Value exists immediately after set")

    # Wait 3 seconds for expiration
    await asyncio.sleep(3)

    # Should be expired now
    retrieved_after = await redis_cache.get(key)
    assert retrieved_after is None
    print("✅ Value expired after TTL")


@pytest.mark.asyncio
async def test_delete_key(redis_cache):
    """Test deleting a key from cache"""
    key = "test_delete_key"
    value = {"to_be": "deleted"}

    # Set value
    await redis_cache.set(key, value, ttl=60)

    # Verify it exists
    assert await redis_cache.get(key) is not None

    # Delete it
    await redis_cache.delete(key)

    # Verify it's gone
    assert await redis_cache.get(key) is None
    print("✅ Successfully deleted key")


@pytest.mark.asyncio
async def test_exists_check(redis_cache):
    """Test checking if key exists"""
    key = "test_exists_key"

    # Should not exist initially
    exists_before = await redis_cache.exists(key)
    assert exists_before is False

    # Set value
    await redis_cache.set(key, {"data": "test"}, ttl=60)

    # Should exist now
    exists_after = await redis_cache.exists(key)
    assert exists_after is True

    print("✅ Exists check working correctly")


@pytest.mark.asyncio
async def test_cache_hit_miss_stats(redis_cache):
    """Test cache statistics tracking"""
    # Get initial stats
    stats = await redis_cache.get_stats()

    assert isinstance(stats, dict)
    print(f"✅ Cache stats: {stats}")

    # Note: Stats tracking depends on implementation
    # This is a basic structure test


@pytest.mark.asyncio
async def test_complex_data_types(redis_cache):
    """Test caching complex Python data structures"""
    test_cases = [
        {
            "key": "test_dict",
            "value": {
                "nested": {"level": 2, "data": [1, 2, 3]},
                "string": "test",
                "number": 42,
                "float": 3.14,
                "boolean": True,
                "null": None
            }
        },
        {
            "key": "test_list",
            "value": [1, 2, {"nested": "list"}, [4, 5, 6]]
        },
        {
            "key": "test_string",
            "value": "Simple string value"
        },
        {
            "key": "test_number",
            "value": 12345
        }
    ]

    for test_case in test_cases:
        key = test_case["key"]
        value = test_case["value"]

        # Set and get
        await redis_cache.set(key, value, ttl=60)
        retrieved = await redis_cache.get(key)

        assert retrieved == value, f"Mismatch for {key}: {retrieved} != {value}"
        print(f"✅ Cached and retrieved {key} correctly")


@pytest.mark.asyncio
async def test_routing_cache_pattern(redis_cache):
    """Test typical routing cache pattern (from layer2_routing.py)"""
    import hashlib

    # Simulate routing decision cache
    query = "What are Tesla's Q4 2023 revenues?"
    context = {"company": "Tesla Inc", "domain": "finance"}

    # Generate cache key (similar to layer2_routing.py)
    key_components = [query, json.dumps(context, sort_keys=True)]
    key_string = "|".join(key_components)
    key_hash = hashlib.md5(key_string.encode()).hexdigest()
    cache_key = f"routing:{key_hash}"

    # Routing decision
    routing_decision = {
        "query_type": "semantic",
        "reasoning": "Financial data query",
        "confidence": 0.85,
        "recommended_tools": ["VectorSearch"],
        "metadata": {"from_cache": True}
    }

    # Cache it (1 hour TTL like production)
    await redis_cache.set(cache_key, routing_decision, ttl=3600)

    # Retrieve it
    cached_decision = await redis_cache.get(cache_key)

    assert cached_decision is not None
    assert cached_decision["query_type"] == "semantic"
    assert cached_decision["confidence"] == 0.85
    print(f"✅ Routing cache pattern working: {cache_key[:20]}...")


@pytest.mark.asyncio
async def test_retrieval_cache_pattern(redis_cache):
    """Test typical retrieval cache pattern (from layer4_retrieval.py)"""
    # Simulate retrieval result cache
    query = "Tesla battery innovations"
    k = 5

    cache_key = f"retrieval:{hash(query)}:{k}"

    retrieval_result = {
        "documents": [
            {"content": "Doc 1", "score": 0.95},
            {"content": "Doc 2", "score": 0.87}
        ],
        "metadata": {
            "total_retrieved": 2,
            "query": query,
            "k": k
        }
    }

    # Cache it
    await redis_cache.set(cache_key, retrieval_result, ttl=3600)

    # Retrieve it
    cached_result = await redis_cache.get(cache_key)

    assert cached_result is not None
    assert len(cached_result["documents"]) == 2
    assert cached_result["metadata"]["total_retrieved"] == 2
    print(f"✅ Retrieval cache pattern working")


@pytest.mark.asyncio
async def test_concurrent_cache_operations(redis_cache):
    """Test concurrent cache operations"""
    # Create multiple concurrent operations
    tasks = []

    for i in range(20):
        key = f"concurrent_key_{i}"
        value = {"id": i, "data": f"concurrent_data_{i}"}

        # Mix of set, get, delete operations
        if i % 3 == 0:
            task = redis_cache.set(key, value, ttl=60)
        elif i % 3 == 1:
            task = redis_cache.get(key)
        else:
            task = redis_cache.delete(key)

        tasks.append(task)

    # Execute all concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Should complete without errors
    errors = [r for r in results if isinstance(r, Exception)]
    assert len(errors) == 0, f"Got {len(errors)} errors in concurrent operations"

    print(f"✅ Completed {len(tasks)} concurrent operations successfully")


@pytest.mark.asyncio
async def test_cache_performance(redis_cache):
    """Test cache operation performance"""
    num_operations = 100

    # Test SET performance
    start = time.time()
    for i in range(num_operations):
        await redis_cache.set(f"perf_key_{i}", {"data": i}, ttl=60)
    set_time = time.time() - start

    # Test GET performance
    start = time.time()
    for i in range(num_operations):
        await redis_cache.get(f"perf_key_{i}")
    get_time = time.time() - start

    # Performance assertions (should be very fast for local Redis)
    assert set_time < 2.0, f"SET ops too slow: {set_time:.2f}s for {num_operations} ops"
    assert get_time < 2.0, f"GET ops too slow: {get_time:.2f}s for {num_operations} ops"

    print(f"✅ Performance: {num_operations} SETs in {set_time:.3f}s, GETs in {get_time:.3f}s")
    print(f"   SET: {num_operations/set_time:.0f} ops/sec, GET: {num_operations/get_time:.0f} ops/sec")


@pytest.mark.asyncio
async def test_cache_namespace_isolation(redis_cache):
    """Test that different cache namespaces don't collide"""
    # Same hash but different namespace
    hash_value = "abc123"

    routing_key = f"routing:{hash_value}"
    retrieval_key = f"retrieval:{hash_value}"

    routing_data = {"type": "routing", "decision": "semantic"}
    retrieval_data = {"type": "retrieval", "docs": [1, 2, 3]}

    # Set both
    await redis_cache.set(routing_key, routing_data, ttl=60)
    await redis_cache.set(retrieval_key, retrieval_data, ttl=60)

    # Retrieve both
    cached_routing = await redis_cache.get(routing_key)
    cached_retrieval = await redis_cache.get(retrieval_key)

    # Should be different
    assert cached_routing["type"] == "routing"
    assert cached_retrieval["type"] == "retrieval"
    assert cached_routing != cached_retrieval

    print("✅ Cache namespace isolation working")


@pytest.mark.asyncio
async def test_cache_hit_rate_simulation(redis_cache):
    """Test cache hit rate in realistic scenario"""
    # Simulate queries (some repeated)
    queries = [
        "Tesla Q4 revenues",
        "Apple stock price",
        "Tesla Q4 revenues",  # Repeat
        "Microsoft earnings",
        "Tesla Q4 revenues",  # Repeat
        "Apple stock price",  # Repeat
    ]

    hits = 0
    misses = 0

    for query in queries:
        cache_key = f"test_hit_rate:{hash(query)}"

        # Try to get from cache
        cached = await redis_cache.get(cache_key)

        if cached is None:
            # Cache MISS - simulate computation and cache result
            misses += 1
            result = {"query": query, "answer": f"Answer for {query}"}
            await redis_cache.set(cache_key, result, ttl=60)
        else:
            # Cache HIT
            hits += 1

    hit_rate = hits / len(queries)

    print(f"✅ Cache hit rate: {hit_rate:.1%} ({hits}/{len(queries)} hits)")

    # With 6 queries and 3 unique, we expect:
    # - 3 misses (first occurrence of each)
    # - 3 hits (repetitions)
    assert hits == 3
    assert misses == 3
    assert hit_rate == 0.5  # 50% hit rate


@pytest.mark.asyncio
async def test_large_value_caching(redis_cache):
    """Test caching large values (simulating document content)"""
    large_value = {
        "documents": [
            {"content": "X" * 10000, "metadata": {"id": i}}
            for i in range(10)
        ],
        "metadata": {"total": 10, "size": "large"}
    }

    key = "test_large_value"

    # Cache large value
    await redis_cache.set(key, large_value, ttl=60)

    # Retrieve it
    retrieved = await redis_cache.get(key)

    assert retrieved is not None
    assert len(retrieved["documents"]) == 10
    assert len(retrieved["documents"][0]["content"]) == 10000

    print("✅ Successfully cached and retrieved large value")


@pytest.mark.asyncio
async def test_reconnect_resilience(redis_cache):
    """Test Redis reconnection resilience"""
    # Set a value
    await redis_cache.set("reconnect_test", {"data": "test"}, ttl=60)

    # Disconnect
    await redis_cache.disconnect()

    # Reconnect
    await redis_cache.connect()

    # Should still be able to retrieve (if TTL hasn't expired)
    retrieved = await redis_cache.get("reconnect_test")
    assert retrieved is not None

    print("✅ Successfully reconnected to Redis")
