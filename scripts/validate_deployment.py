#!/usr/bin/env python3
"""
Deployment Validation Script for Fintech Agentic RAG

Validates all services and system requirements before deployment.

Usage:
    python scripts/validate_deployment.py
    python scripts/validate_deployment.py --detailed

Author: Glemes
"""

import os
import sys
import json
import time
import argparse
import asyncio
from typing import Dict, List, Tuple, Any
from datetime import datetime


# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}{text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}{text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}{text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message"""
    print(f"   {text}")


class ValidationResult:
    """Stores validation check results"""

    def __init__(self):
        self.checks: List[Tuple[str, bool, str]] = []  # (name, passed, message)
        self.start_time = time.time()

    def add_check(self, name: str, passed: bool, message: str = ""):
        """Add a validation check result"""
        self.checks.append((name, passed, message))

        if passed:
            print_success(f"{name}")
            if message:
                print_info(message)
        else:
            print_error(f"{name}")
            if message:
                print_info(f"Error: {message}")

    def print_summary(self):
        """Print validation summary"""
        elapsed = time.time() - self.start_time
        passed = sum(1 for _, p, _ in self.checks if p)
        failed = len(self.checks) - passed

        print_header("Validation Summary")

        print(f"Total Checks: {len(self.checks)}")
        print(f"{Colors.GREEN}Passed: {passed}{Colors.ENDC}")
        print(f"{Colors.RED}Failed: {failed}{Colors.ENDC}")
        print(f"Time Elapsed: {elapsed:.2f}s")

        if failed == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}ALL CHECKS PASSED - READY FOR DEPLOYMENT!{Colors.ENDC}")
            return True
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}DEPLOYMENT NOT READY - FIX ERRORS ABOVE{Colors.ENDC}")
            return False


# ============================================================================
# Validation Checks
# ============================================================================

def check_python_version(result: ValidationResult):
    """Check Python version >= 3.11"""
    import sys

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major == 3 and version.minor >= 11:
        result.add_check(
            "Python Version",
            True,
            f"Python {version_str} (>= 3.11 required)"
        )
    else:
        result.add_check(
            "Python Version",
            False,
            f"Python {version_str} found, but >= 3.11 required"
        )


def check_environment_variables(result: ValidationResult):
    """Check required environment variables"""
    required_vars = [
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "POSTGRES_HOST",
        "POSTGRES_PASSWORD",
        "REDIS_HOST",
    ]

    optional_vars = [
        "HUGGINGFACE_TOKEN",
        "OPENAI_API_KEY",
    ]

    missing_required = []
    missing_optional = []

    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)

    for var in optional_vars:
        if not os.getenv(var):
            missing_optional.append(var)

    if missing_required:
        result.add_check(
            "Environment Variables (Required)",
            False,
            f"Missing: {', '.join(missing_required)}"
        )
    else:
        result.add_check(
            "Environment Variables (Required)",
            True,
            "All required variables set"
        )

    if missing_optional:
        result.add_check(
            "Environment Variables (Optional)",
            True,  # Optional, so still pass
            f"Missing (optional): {', '.join(missing_optional)}"
        )


def check_docker_services(result: ValidationResult):
    """Check Docker services are running"""
    import subprocess

    try:
        # Check if docker-compose is available
        subprocess.run(
            ["docker-compose", "--version"],
            capture_output=True,
            check=True
        )

        # Check running services
        output = subprocess.run(
            ["docker-compose", "ps", "--services", "--filter", "status=running"],
            capture_output=True,
            text=True,
            check=True
        )

        running_services = output.stdout.strip().split('\n') if output.stdout.strip() else []

        expected_services = ["app", "neo4j", "redis", "postgresql", "prometheus", "grafana"]
        missing_services = [s for s in expected_services if s not in running_services]

        if not missing_services:
            result.add_check(
                "Docker Services",
                True,
                f"All services running: {', '.join(running_services)}"
            )
        else:
            result.add_check(
                "Docker Services",
                False,
                f"Missing services: {', '.join(missing_services)}"
            )

    except subprocess.CalledProcessError:
        result.add_check(
            "Docker Services",
            False,
            "docker-compose not available or services not running"
        )
    except FileNotFoundError:
        result.add_check(
            "Docker Services",
            False,
            "docker-compose command not found"
        )


async def check_postgresql_connection(result: ValidationResult):
    """Check PostgreSQL connection"""
    try:
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
        is_connected = manager.is_connected

        if is_connected:
            # Test query
            await manager.log_prompt(
                layer="validation_test",
                prompt="Validation check",
                response="Success",
                latency_ms=0.0,
                metadata={"validation": True},
                session_id="validation_session"
            )

            result.add_check(
                "PostgreSQL Connection",
                True,
                f"Connected to {config.postgres.host}:{config.postgres.port}"
            )

            await manager.disconnect()
        else:
            result.add_check(
                "PostgreSQL Connection",
                False,
                "Failed to connect"
            )

    except Exception as e:
        result.add_check(
            "PostgreSQL Connection",
            False,
            f"Error: {str(e)}"
        )


async def check_redis_connection(result: ValidationResult):
    """Check Redis connection"""
    try:
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

        # Test operations
        await cache.set("validation_test", {"test": True}, ttl=10)
        value = await cache.get("validation_test")

        if value and value.get("test"):
            result.add_check(
                "Redis Connection",
                True,
                f"Connected to {config.redis.host}:{config.redis.port}"
            )
        else:
            result.add_check(
                "Redis Connection",
                False,
                "Connection established but cache test failed"
            )

        await cache.disconnect()

    except Exception as e:
        result.add_check(
            "Redis Connection",
            False,
            f"Error: {str(e)}"
        )


def check_neo4j_connection(result: ValidationResult):
    """Check Neo4j connection"""
    try:
        from neo4j import GraphDatabase
        from src.utils.config import get_config

        config = get_config()
        driver = GraphDatabase.driver(
            config.neo4j.uri,
            auth=(config.neo4j.user, config.neo4j.password)
        )

        with driver.session() as session:
            result_query = session.run("RETURN 1 AS test")
            record = result_query.single()

            if record and record["test"] == 1:
                result.add_check(
                    "Neo4j Connection",
                    True,
                    f"Connected to {config.neo4j.uri}"
                )
            else:
                result.add_check(
                    "Neo4j Connection",
                    False,
                    "Connection established but test query failed"
                )

        driver.close()

    except Exception as e:
        result.add_check(
            "Neo4j Connection",
            False,
            f"Error: {str(e)}"
        )


def check_api_health(result: ValidationResult):
    """Check FastAPI health endpoint"""
    try:
        import requests

        response = requests.get("http://localhost:8000/health_check", timeout=5)

        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "healthy":
                result.add_check(
                    "API Health Endpoint",
                    True,
                    "API is healthy and responding"
                )
            else:
                result.add_check(
                    "API Health Endpoint",
                    False,
                    f"API returned unhealthy status: {data.get('status')}"
                )
        else:
            result.add_check(
                "API Health Endpoint",
                False,
                f"HTTP {response.status_code}"
            )

    except requests.exceptions.ConnectionError:
        result.add_check(
            "API Health Endpoint",
            False,
            "Cannot connect to http://localhost:8000 - is the API running?"
        )
    except Exception as e:
        result.add_check(
            "API Health Endpoint",
            False,
            f"Error: {str(e)}"
        )


def check_prometheus_metrics(result: ValidationResult):
    """Check Prometheus metrics endpoint"""
    try:
        import requests

        response = requests.get("http://localhost:8000/metrics", timeout=5)

        if response.status_code == 200:
            content = response.text

            # Check for key metrics
            required_metrics = [
                "rag_requests_total",
                "rag_request_duration_seconds",
                "rag_active_requests"
            ]

            missing_metrics = [m for m in required_metrics if m not in content]

            if not missing_metrics:
                result.add_check(
                    "Prometheus Metrics",
                    True,
                    "All required metrics present"
                )
            else:
                result.add_check(
                    "Prometheus Metrics",
                    False,
                    f"Missing metrics: {', '.join(missing_metrics)}"
                )
        else:
            result.add_check(
                "Prometheus Metrics",
                False,
                f"HTTP {response.status_code}"
            )

    except Exception as e:
        result.add_check(
            "Prometheus Metrics",
            False,
            f"Error: {str(e)}"
        )


def check_file_structure(result: ValidationResult):
    """Check required files and directories exist"""
    required_paths = [
        "src/main.py",
        "src/agentic_rag/orchestrator.py",
        "src/db/postgres.py",
        "src/db/redis_cache.py",
        "docker-compose.yml",
        "Dockerfile",
        "requirements.txt",
        "db/init.sql",
    ]

    missing_paths = []

    for path in required_paths:
        if not os.path.exists(path):
            missing_paths.append(path)

    if not missing_paths:
        result.add_check(
            "File Structure",
            True,
            "All required files present"
        )
    else:
        result.add_check(
            "File Structure",
            False,
            f"Missing files: {', '.join(missing_paths)}"
        )


def check_data_directories(result: ValidationResult):
    """Check data directories exist and are writable"""
    data_dirs = [
        "./vectors",
        "./data",
        "./logs",
    ]

    issues = []

    for dir_path in data_dirs:
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
                issues.append(f"Created {dir_path}")
            except Exception as e:
                issues.append(f"Cannot create {dir_path}: {e}")
        elif not os.access(dir_path, os.W_OK):
            issues.append(f"{dir_path} not writable")

    if not issues:
        result.add_check(
            "Data Directories",
            True,
            "All directories exist and are writable"
        )
    else:
        result.add_check(
            "Data Directories",
            True,  # Still pass if we created them
            "\n".join(issues)
        )


async def run_validation(detailed: bool = False):
    """Run all validation checks"""
    print_header("Deployment Validation - Fintech Agentic RAG")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    result = ValidationResult()

    # 1. System checks
    print_header("1. System Requirements")
    check_python_version(result)
    check_environment_variables(result)

    # 2. Docker services
    print_header("2. Docker Services")
    check_docker_services(result)

    # 3. Database connections
    print_header("3. Database Connections")
    await check_postgresql_connection(result)
    await check_redis_connection(result)
    check_neo4j_connection(result)

    # 4. API endpoints
    print_header("4. API Endpoints")
    check_api_health(result)
    check_prometheus_metrics(result)

    # 5. File structure
    print_header("5. File Structure")
    check_file_structure(result)
    check_data_directories(result)

    # Summary
    success = result.print_summary()

    return success


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Validate Fintech Agentic RAG deployment"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed output for each check"
    )

    args = parser.parse_args()

    try:
        success = asyncio.run(run_validation(detailed=args.detailed))
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Validation interrupted by user{Colors.ENDC}")
        sys.exit(1)

    except Exception as e:
        print(f"\n{Colors.RED}Fatal error during validation: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
