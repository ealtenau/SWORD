# -*- coding: utf-8 -*-
"""
Centralized pytest fixtures for SWORD DuckDB tests.

Provides:
- Minimal test database fixture (~100 reaches, <100MB vs 9.9GB production)
- Read-only and writable SWORD instances
- Session-scoped fixture generation for fast test runs
"""

import os
import sys
import pytest
import shutil
import tempfile
from pathlib import Path

# Add project root to path
main_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(main_dir))

# Import after path setup
from src.updates.sword_duckdb import SWORD


# ==============================================================================
# Configuration
# ==============================================================================

# Minimal test database location
TEST_FIXTURES_DIR = Path(__file__).parent / "fixtures"
MINIMAL_DB_PATH = TEST_FIXTURES_DIR / "sword_test_minimal.duckdb"

# Production database for legacy tests (if available)
PRODUCTION_DB_PATH = main_dir / "data" / "duckdb" / "sword_v17b.duckdb"

# Test region and version
TEST_REGION = "NA"
TEST_VERSION = "v17b"


# ==============================================================================
# Minimal Test Database Fixtures
# ==============================================================================

@pytest.fixture(scope="session")
def test_db_path():
    """Path to the minimal test database."""
    return MINIMAL_DB_PATH


@pytest.fixture(scope="session")
def ensure_test_db(test_db_path):
    """
    Ensure minimal test database exists, creating it if necessary.

    This fixture runs once per test session and creates the test database
    if it doesn't exist or if force regeneration is requested via environment.
    """
    force_regen = os.environ.get("SWORD_REGEN_TEST_DB", "").lower() == "true"

    if not test_db_path.exists() or force_regen:
        # Import and run generator
        from tests.sword_duckdb.fixtures.create_test_db import create_minimal_test_db

        test_db_path.parent.mkdir(parents=True, exist_ok=True)
        create_minimal_test_db(str(test_db_path))

    return test_db_path


@pytest.fixture(scope="session")
def sword_readonly_session(ensure_test_db):
    """
    Session-scoped read-only SWORD instance using minimal test database.

    Use this for tests that only read data. Faster than function-scoped
    fixtures because the database is only loaded once per session.
    """
    sword = SWORD(str(ensure_test_db), TEST_REGION, TEST_VERSION)
    yield sword
    sword.close()


@pytest.fixture
def sword_readonly(sword_readonly_session):
    """
    Function-scoped read-only SWORD instance (wraps session-scoped).

    Use this in individual tests for cleaner fixture dependency.
    Note: Do NOT modify data through this fixture!
    """
    return sword_readonly_session


@pytest.fixture
def sword_writable(ensure_test_db, tmp_path):
    """
    Function-scoped writable SWORD instance with temporary database copy.

    Use this for tests that modify data. Each test gets a fresh copy
    of the database, ensuring test isolation.
    """
    # Copy to temp directory
    temp_db = tmp_path / "sword_test.duckdb"
    shutil.copy2(ensure_test_db, temp_db)

    sword = SWORD(str(temp_db), TEST_REGION, TEST_VERSION)
    yield sword
    sword.close()


# ==============================================================================
# Production Database Fixtures (Legacy Compatibility)
# ==============================================================================

@pytest.fixture(scope="module")
def sword_production():
    """
    Module-scoped SWORD instance using production database.

    Only use this when testing against real SWORD data is required.
    Skips if production database is not available.
    """
    if not PRODUCTION_DB_PATH.exists():
        pytest.skip(f"Production database not found: {PRODUCTION_DB_PATH}")

    sword = SWORD(str(PRODUCTION_DB_PATH), TEST_REGION, TEST_VERSION)
    yield sword
    sword.close()


@pytest.fixture
def temp_sword_production(tmp_path):
    """
    Function-scoped writable SWORD instance using production database copy.

    WARNING: This copies the 9.9GB production database - use sparingly!
    Prefer sword_writable (minimal database) for most tests.
    """
    if not PRODUCTION_DB_PATH.exists():
        pytest.skip(f"Production database not found: {PRODUCTION_DB_PATH}")

    temp_db = tmp_path / "sword_test.duckdb"
    shutil.copy2(PRODUCTION_DB_PATH, temp_db)

    sword = SWORD(str(temp_db), TEST_REGION, TEST_VERSION)
    yield sword
    sword.close()


# ==============================================================================
# Convenience Aliases (for backward compatibility)
# ==============================================================================

# Alias for existing tests that use 'sword' fixture
@pytest.fixture(scope="module")
def sword(sword_readonly_session):
    """Alias for backward compatibility with existing tests."""
    return sword_readonly_session


# Alias for existing tests that use 'temp_sword' fixture
@pytest.fixture
def temp_sword(sword_writable):
    """Alias for backward compatibility with existing tests."""
    return sword_writable
