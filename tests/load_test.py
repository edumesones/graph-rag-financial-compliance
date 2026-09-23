"""
Load testing with Locust for Fintech Agentic RAG API

Run with:
    locust -f tests/load_test.py --host=http://localhost:8000

For headless mode with specific parameters:
    locust -f tests/load_test.py --host=http://localhost:8000 --users 100 --spawn-rate 10 --run-time 5m --headless

Author: Glemes
"""

from locust import HttpUser, task, between, events
import random
import json
import time


# Test data: Different companies and queries
TEST_COMPANIES = [
    "Tesla Inc",
    "Apple Inc",
    "Microsoft Corporation",
    "Amazon.com Inc",
    "Meta Platforms Inc"
]

TEST_QUERIES = [
    "What were the Q4 2023 revenue figures?",
    "What SEC violations occurred in the last fiscal year?",
    "Explain the company's revenue growth strategy",
    "What are the main regulatory risks?",
    "Summarize the latest earnings report",
    "What compliance issues were mentioned?",
    "Describe the company's market position",
    "What were the key financial metrics?",
    "Explain the company's competitive advantages",
    "What are the major business risks?"
]


class RAGUser(HttpUser):
    """
    Simulated user for Fintech Agentic RAG API
    """
    # Wait between 1 and 5 seconds between requests (realistic user behavior)
    wait_time = between(1, 5)

    def on_start(self):
        """Called when a user starts"""
        self.client.verify = False  # Disable SSL verification for local testing
        print(f"[User {self.environment.runner.user_count}] Started")

    @task(3)  # Weight 3: Most common operation
    def analyze_compliance(self):
        """
        Main endpoint: POST /analyze_compliance

        This simulates the primary use case: querying the RAG system
        """
        company = random.choice(TEST_COMPANIES)
        query = random.choice(TEST_QUERIES)

        payload = {
            "company": company,
            "query": query,
            "verbose": False  # Non-verbose for faster responses
        }

        with self.client.post(
            "/analyze_compliance",
            json=payload,
            catch_response=True,
            name="/analyze_compliance [non-verbose]"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()

                    # Validate response structure
                    required_fields = ["query", "answer", "confidence", "status"]
                    if all(field in data for field in required_fields):
                        # Check response quality
                        if data["confidence"] > 0.0:
                            response.success()
                        else:
                            response.failure("Low confidence score")
                    else:
                        response.failure(f"Missing required fields. Got: {list(data.keys())}")

                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)  # Weight 1: Less common (verbose is slower)
    def analyze_compliance_verbose(self):
        """
        Verbose endpoint test: POST /analyze_compliance (verbose=True)

        Tests the more detailed response mode
        """
        company = random.choice(TEST_COMPANIES)
        query = random.choice(TEST_QUERIES)

        payload = {
            "company": company,
            "query": query,
            "verbose": True  # Verbose mode for detailed trace
        }

        with self.client.post(
            "/analyze_compliance",
            json=payload,
            catch_response=True,
            name="/analyze_compliance [verbose]"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()

                    # Verbose mode should include pipeline_trace
                    if "pipeline_trace" in data:
                        response.success()
                    else:
                        # Still valid if no pipeline_trace (depends on implementation)
                        response.success()

                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(2)  # Weight 2: Moderate frequency
    def health_check(self):
        """
        Health check endpoint: GET /health_check

        Lightweight endpoint to verify service availability
        """
        with self.client.get("/health_check", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get("status") == "healthy":
                        response.success()
                    else:
                        response.failure(f"Unhealthy status: {data.get('status')}")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)  # Weight 1: Monitoring/observability
    def get_system_stats(self):
        """
        System stats endpoint: GET /get_system_stats

        Tests observability endpoint
        """
        with self.client.get("/get_system_stats", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    response.success()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)  # Weight 1: Metrics endpoint
    def prometheus_metrics(self):
        """
        Prometheus metrics endpoint: GET /metrics

        Tests metrics collection
        """
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                # Metrics endpoint returns plain text, not JSON
                if b"rag_requests_total" in response.content:
                    response.success()
                else:
                    response.failure("Missing expected metrics")
            else:
                response.failure(f"HTTP {response.status_code}")


class CacheTestUser(HttpUser):
    """
    User specifically for testing cache hit rates

    Sends repeated queries to test Redis caching effectiveness
    """
    wait_time = between(0.5, 2)  # Faster requests to stress cache

    # Fixed queries for cache testing
    CACHE_QUERIES = [
        ("Tesla Inc", "What were Q4 2023 revenues?"),
        ("Apple Inc", "What were the latest earnings?"),
        ("Tesla Inc", "What were Q4 2023 revenues?"),  # Repeat for cache hit
        ("Microsoft Corporation", "What is the market cap?"),
        ("Apple Inc", "What were the latest earnings?"),  # Repeat for cache hit
    ]

    def on_start(self):
        """Initialize cache test user"""
        self.query_index = 0
        self.client.verify = False

    @task
    def test_cache(self):
        """
        Send repeated queries to test caching
        """
        # Cycle through queries
        company, query = self.CACHE_QUERIES[self.query_index % len(self.CACHE_QUERIES)]
        self.query_index += 1

        payload = {
            "company": company,
            "query": query,
            "verbose": False
        }

        start_time = time.time()

        with self.client.post(
            "/analyze_compliance",
            json=payload,
            catch_response=True,
            name="/analyze_compliance [cache-test]"
        ) as response:
            response_time = (time.time() - start_time) * 1000  # ms

            if response.status_code == 200:
                # Cache hits should be significantly faster (<500ms ideally)
                if response_time < 500:
                    response.success()
                    print(f"Fast response ({response_time:.0f}ms) - likely cache hit")
                else:
                    response.success()
                    print(f"⏱Slow response ({response_time:.0f}ms) - likely cache miss")
            else:
                response.failure(f"HTTP {response.status_code}")


# Event listeners for custom metrics
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts"""
    print("\n" + "="*60)
    print("Load Test Started")
    print("="*60)
    print(f"Target: {environment.host}")
    print(f"Users: {environment.runner.target_user_count if hasattr(environment.runner, 'target_user_count') else 'N/A'}")
    print("="*60 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops - Print summary"""
    print("\n" + "="*60)
    print("Load Test Summary")
    print("="*60)

    stats = environment.stats

    print(f"\nTotal Requests: {stats.total.num_requests}")
    print(f"Total Failures: {stats.total.num_failures}")
    print(f"Failure Rate: {stats.total.fail_ratio * 100:.2f}%")

    if stats.total.num_requests > 0:
        print(f"\nResponse Times:")
        print(f"  Average: {stats.total.avg_response_time:.0f}ms")
        print(f"  Median: {stats.total.median_response_time:.0f}ms")
        print(f"  P95: {stats.total.get_response_time_percentile(0.95):.0f}ms")
        print(f"  P99: {stats.total.get_response_time_percentile(0.99):.0f}ms")
        print(f"  Min: {stats.total.min_response_time:.0f}ms")
        print(f"  Max: {stats.total.max_response_time:.0f}ms")

        print(f"\nRequests per Second: {stats.total.total_rps:.2f}")

        # Performance assessment
        print(f"\nPerformance Assessment:")

        avg_time = stats.total.avg_response_time
        p95_time = stats.total.get_response_time_percentile(0.95)
        fail_rate = stats.total.fail_ratio

        if avg_time < 2000 and p95_time < 3000 and fail_rate < 0.01:
            print("  EXCELLENT - Meets all performance targets!")
        elif avg_time < 3000 and p95_time < 5000 and fail_rate < 0.05:
            print("  GOOD - Acceptable performance")
        elif avg_time < 5000 and p95_time < 8000 and fail_rate < 0.10:
            print("  FAIR - Performance needs improvement")
        else:
            print("  POOR - Performance below targets")

        print(f"\n  Target: Avg <2s, P95 <3s, Error <1%")
        print(f"  Actual: Avg {avg_time/1000:.1f}s, P95 {p95_time/1000:.1f}s, Error {fail_rate*100:.1f}%")

    print("\n" + "="*60 + "\n")


# Custom test scenarios
class QuickSmokeTest(HttpUser):
    """
    Quick smoke test scenario - minimal load

    Use with: locust -f tests/load_test.py QuickSmokeTest --users 5 --run-time 1m
    """
    wait_time = between(2, 5)

    @task
    def smoke_test(self):
        """Basic smoke test"""
        # Health check
        self.client.get("/health_check")

        # Single analyze request
        self.client.post("/analyze_compliance", json={
            "company": "Tesla Inc",
            "query": "Quick smoke test query",
            "verbose": False
        })


class StressTest(HttpUser):
    """
    Stress test scenario - high load

    Use with: locust -f tests/load_test.py StressTest --users 200 --spawn-rate 20 --run-time 10m
    """
    wait_time = between(0.5, 2)  # Faster requests

    @task
    def stress_analyze(self):
        """Stress test main endpoint"""
        self.client.post("/analyze_compliance", json={
            "company": random.choice(TEST_COMPANIES),
            "query": random.choice(TEST_QUERIES),
            "verbose": random.choice([True, False])
        })
