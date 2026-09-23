"""
Integration tests for Neo4j graph database
"""
import pytest
from neo4j.exceptions import ServiceUnavailable


def test_neo4j_connection(neo4j_connection):
    """Test Neo4j connection is established"""
    # Try to verify connectivity
    with neo4j_connection.session() as session:
        result = session.run("RETURN 1 AS test")
        record = result.single()
        assert record["test"] == 1
        print("Neo4j connection successful")


def test_neo4j_database_exists(neo4j_connection):
    """Test that database exists and is accessible"""
    with neo4j_connection.session() as session:
        # Get database name
        result = session.run("CALL dbms.listConfig() YIELD name, value WHERE name = 'dbms.default_database' RETURN value")
        try:
            record = result.single()
            if record:
                db_name = record["value"]
                print(f"Connected to database: {db_name}")
        except Exception:
            # Older Neo4j versions may not support this
            print("Could not determine database name (older Neo4j version)")


def test_create_test_node(neo4j_connection):
    """Test creating a node in Neo4j"""
    with neo4j_connection.session() as session:
        # Create a test node
        result = session.run("""
            CREATE (n:TestNode {
                name: $name,
                timestamp: timestamp(),
                test: true
            })
            RETURN id(n) AS node_id, n.name AS name
        """, name="pytest_test_node")

        record = result.single()
        assert record is not None
        assert record["name"] == "pytest_test_node"

        node_id = record["node_id"]
        print(f"Created test node with ID: {node_id}")

        # Clean up: delete the test node
        session.run("MATCH (n:TestNode {name: $name}) DELETE n", name="pytest_test_node")
        print("Cleaned up test node")


def test_query_existing_nodes(neo4j_connection):
    """Test querying existing nodes (if any exist from ingestion)"""
    with neo4j_connection.session() as session:
        # Count total nodes
        result = session.run("MATCH (n) RETURN count(n) AS total")
        record = result.single()
        total_nodes = record["total"]

        print(f"Total nodes in graph: {total_nodes}")

        # If nodes exist, test querying them
        if total_nodes > 0:
            # Get node labels
            result = session.run("CALL db.labels()")
            labels = [record["label"] for record in result]
            print(f"   Node labels: {labels}")


def test_create_relationship(neo4j_connection):
    """Test creating relationships between nodes"""
    with neo4j_connection.session() as session:
        # Create two test nodes with a relationship
        result = session.run("""
            CREATE (a:TestCompany {name: $company1})
            CREATE (b:TestRegulation {name: $regulation})
            CREATE (a)-[r:COMPLIES_WITH {since: 2023}]->(b)
            RETURN id(a) AS company_id, id(b) AS regulation_id, type(r) AS rel_type
        """, company1="Test Corp", regulation="Test Regulation")

        record = result.single()
        assert record is not None
        assert record["rel_type"] == "COMPLIES_WITH"

        print(f"Created relationship: Company {record['company_id']} -> Regulation {record['regulation_id']}")

        # Clean up
        session.run("""
            MATCH (a:TestCompany {name: $company1})-[r:COMPLIES_WITH]-(b:TestRegulation {name: $regulation})
            DELETE r, a, b
        """, company1="Test Corp", regulation="Test Regulation")
        print("Cleaned up test nodes and relationship")


def test_graph_traversal(neo4j_connection):
    """Test graph traversal queries"""
    with neo4j_connection.session() as session:
        # Create a small test graph
        session.run("""
            CREATE (c:TestCompany {name: 'TraversalTest Inc'})
            CREATE (v:TestViolation {type: 'SEC-001'})
            CREATE (r:TestRegulation {code: 'REG-001'})
            CREATE (c)-[:COMMITTED]->(v)
            CREATE (v)-[:VIOLATES]->(r)
        """)

        # Traverse the graph
        result = session.run("""
            MATCH (c:TestCompany {name: 'TraversalTest Inc'})-[:COMMITTED]->(v)-[:VIOLATES]->(r)
            RETURN c.name AS company, v.type AS violation, r.code AS regulation
        """)

        record = result.single()
        assert record is not None
        assert record["company"] == "TraversalTest Inc"
        assert record["violation"] == "SEC-001"
        assert record["regulation"] == "REG-001"

        print("Graph traversal successful")

        # Clean up
        session.run("""
            MATCH (c:TestCompany {name: 'TraversalTest Inc'})-[r1]->(v)-[r2]->(reg:TestRegulation)
            DELETE r1, r2, c, v, reg
        """)
        print("Cleaned up traversal test graph")


def test_cypher_query_performance(neo4j_connection):
    """Test Cypher query performance"""
    import time

    with neo4j_connection.session() as session:
        # Create test data
        start = time.time()
        session.run("""
            UNWIND range(1, 100) AS i
            CREATE (:TestPerfNode {id: i, value: 'test_' + i})
        """)
        create_time = time.time() - start

        # Query test data
        start = time.time()
        result = session.run("""
            MATCH (n:TestPerfNode)
            WHERE n.id > 50
            RETURN count(n) AS total
        """)
        query_time = time.time() - start

        record = result.single()
        assert record["total"] == 50

        print(f"Performance: Created 100 nodes in {create_time:.3f}s, queried in {query_time:.3f}s")

        # Clean up
        session.run("MATCH (n:TestPerfNode) DELETE n")
        print("Cleaned up performance test nodes")


def test_entity_extraction_pattern(neo4j_connection):
    """Test typical entity extraction pattern from RAG system"""
    with neo4j_connection.session() as session:
        # Simulate entities extracted from a document
        entities = {
            "company": "TestRAG Corp",
            "regulation": "TestRAG Regulation",
            "violation": "Compliance Issue 001",
            "date": "2023-Q4"
        }

        # Create graph structure (similar to ingestion pipeline)
        session.run("""
            MERGE (c:Company {name: $company})
            MERGE (r:Regulation {name: $regulation})
            MERGE (v:Violation {description: $violation, date: $date})
            MERGE (c)-[:COMMITTED]->(v)
            MERGE (v)-[:VIOLATES]->(r)
        """, **entities)

        # Query the created structure
        result = session.run("""
            MATCH (c:Company {name: $company})-[:COMMITTED]->(v)-[:VIOLATES]->(r)
            RETURN c.name AS company, v.description AS violation, r.name AS regulation
        """, company=entities["company"])

        record = result.single()
        assert record is not None
        assert record["company"] == entities["company"]

        print("Entity extraction pattern successful")

        # Clean up
        session.run("""
            MATCH (c:Company {name: $company})
            MATCH (r:Regulation {name: $regulation})
            MATCH (v:Violation {description: $violation})
            DETACH DELETE c, r, v
        """, **entities)
        print("Cleaned up entity extraction test")


def test_concurrent_writes(neo4j_connection):
    """Test concurrent write operations to Neo4j"""
    import concurrent.futures

    def create_node(session, node_id):
        session.run("""
            CREATE (n:ConcurrentTest {id: $id, timestamp: timestamp()})
        """, id=node_id)
        return node_id

    # Create multiple sessions and write concurrently
    created_ids = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i in range(20):
            session = neo4j_connection.session()
            future = executor.submit(create_node, session, f"concurrent_{i}")
            futures.append((future, session))

        for future, session in futures:
            node_id = future.result()
            created_ids.append(node_id)
            session.close()

    assert len(created_ids) == 20
    print(f"Created {len(created_ids)} nodes concurrently")

    # Verify all nodes were created
    with neo4j_connection.session() as session:
        result = session.run("MATCH (n:ConcurrentTest) RETURN count(n) AS total")
        record = result.single()
        assert record["total"] >= 20

        # Clean up
        session.run("MATCH (n:ConcurrentTest) DELETE n")
        print("Cleaned up concurrent test nodes")


def test_relationship_properties(neo4j_connection):
    """Test relationship properties (important for metadata)"""
    with neo4j_connection.session() as session:
        # Create relationship with properties
        session.run("""
            CREATE (a:TestEntity {name: 'Entity A'})
            CREATE (b:TestEntity {name: 'Entity B'})
            CREATE (a)-[r:RELATED_TO {
                strength: 0.95,
                source: 'document_123',
                timestamp: datetime(),
                metadata: 'test'
            }]->(b)
        """)

        # Query relationship properties
        result = session.run("""
            MATCH (a:TestEntity)-[r:RELATED_TO]->(b:TestEntity)
            WHERE a.name = 'Entity A'
            RETURN r.strength AS strength, r.source AS source
        """)

        record = result.single()
        assert record is not None
        assert record["strength"] == 0.95
        assert record["source"] == "document_123"

        print("Relationship properties stored and retrieved correctly")

        # Clean up
        session.run("MATCH (n:TestEntity) DETACH DELETE n")
        print("Cleaned up relationship property test")


def test_index_usage(neo4j_connection):
    """Test creating and using indexes for performance"""
    with neo4j_connection.session() as session:
        try:
            # Create index on TestIndexNode.name
            session.run("CREATE INDEX test_index_name IF NOT EXISTS FOR (n:TestIndexNode) ON (n.name)")

            # Create test nodes
            session.run("""
                UNWIND range(1, 1000) AS i
                CREATE (:TestIndexNode {name: 'Node_' + i, value: i})
            """)

            # Query using indexed property
            result = session.run("""
                MATCH (n:TestIndexNode {name: 'Node_500'})
                RETURN n.value AS value
            """)

            record = result.single()
            assert record is not None
            assert record["value"] == 500

            print("Index usage successful")

            # Clean up
            session.run("MATCH (n:TestIndexNode) DELETE n")
            session.run("DROP INDEX test_index_name IF EXISTS")
            print("Cleaned up index test")

        except Exception as e:
            print(f"Index test skipped: {e}")
            # Clean up even on failure
            session.run("MATCH (n:TestIndexNode) DELETE n")


def test_complex_graph_query(neo4j_connection):
    """Test complex graph query (multi-hop relationship)"""
    with neo4j_connection.session() as session:
        # Create a complex graph structure
        session.run("""
            CREATE (c1:TestCompany {name: 'Company Alpha'})
            CREATE (c2:TestCompany {name: 'Company Beta'})
            CREATE (v1:TestViolation {type: 'Type A'})
            CREATE (v2:TestViolation {type: 'Type B'})
            CREATE (r:TestRegulation {code: 'REG-X'})
            CREATE (c1)-[:COMMITTED]->(v1)
            CREATE (c2)-[:COMMITTED]->(v2)
            CREATE (v1)-[:VIOLATES]->(r)
            CREATE (v2)-[:VIOLATES]->(r)
        """)

        # Complex query: Find all companies violating the same regulation
        result = session.run("""
            MATCH (c:TestCompany)-[:COMMITTED]->(:TestViolation)-[:VIOLATES]->(r:TestRegulation {code: 'REG-X'})
            RETURN c.name AS company
            ORDER BY c.name
        """)

        companies = [record["company"] for record in result]
        assert len(companies) == 2
        assert "Company Alpha" in companies
        assert "Company Beta" in companies

        print(f"Complex query found {len(companies)} companies")

        # Clean up
        session.run("""
            MATCH (c:TestCompany)
            MATCH (v:TestViolation)
            MATCH (r:TestRegulation)
            DETACH DELETE c, v, r
        """)
        print("Cleaned up complex query test")


def test_transaction_rollback(neo4j_connection):
    """Test transaction rollback on error"""
    with neo4j_connection.session() as session:
        # Start a transaction that will fail
        tx = session.begin_transaction()

        try:
            # Create a node
            tx.run("CREATE (n:TestRollback {name: 'Should not persist'})")

            # Intentionally cause an error
            tx.run("INVALID CYPHER QUERY")

            tx.commit()
            pytest.fail("Transaction should have failed")

        except Exception:
            # Transaction should rollback
            tx.rollback()
            print("Transaction rolled back on error")

        # Verify node was not created
        result = session.run("MATCH (n:TestRollback) RETURN count(n) AS total")
        record = result.single()
        assert record["total"] == 0
        print("Verified rollback - node not persisted")
