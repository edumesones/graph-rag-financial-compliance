"""
Pytest configuration and fixtures
"""
import pytest
import asyncio
from typing import AsyncGenerator

# Make all tests async-compatible
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def postgres_manager():
    """PostgreSQL connection fixture"""
    from src.db.postgres import PostgresManager
    from src.utils.config import get_config

    config = get_config()
    manager = PostgresManager(
        host=config.postgres.host,
        port=config.postgres.port,
        database=config.postgres.database,
        user=config.postgres.user,
        password=config.postgres.password,
    )

    await manager.connect()
    yield manager
    await manager.disconnect()


@pytest.fixture
async def redis_cache():
    """Redis cache fixture"""
    from src.db.redis_cache import RedisCache
    from src.utils.config import get_config

    config = get_config()
    cache = RedisCache(
        host=config.redis.host,
        port=config.redis.port,
        db=config.redis.db,
        ttl=config.redis.ttl,
    )

    await cache.connect()
    yield cache
    await cache.disconnect()


@pytest.fixture
def sample_query():
    """Sample query for testing"""
    return "What were Tesla's Q4 2023 revenue figures?"


@pytest.fixture
def sample_company():
    """Sample company for testing"""
    return "Tesla Inc"


@pytest.fixture
def neo4j_connection():
    """Neo4j connection fixture"""
    from neo4j import GraphDatabase
    from src.utils.config import get_config

    config = get_config()
    driver = GraphDatabase.driver(
        config.neo4j.uri,
        auth=(config.neo4j.user, config.neo4j.password)
    )

    yield driver
    driver.close()


@pytest.fixture
def vectorstore():
    """ChromaDB vectorstore fixture"""
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from src.utils.config import get_config

    config = get_config()

    # Initialize embeddings (same as production)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Load existing vectorstore
    vectorstore = Chroma(
        persist_directory=config.vectorstore.persist_directory,
        embedding_function=embeddings,
        collection_name=config.vectorstore.collection_name
    )

    yield vectorstore


@pytest.fixture
def sample_routing_query():
    """Sample query for routing tests"""
    return "What are the relationships between Tesla and regulatory agencies?"


@pytest.fixture
def sample_semantic_query():
    """Sample semantic search query"""
    return "Explain Tesla's battery technology innovations"
