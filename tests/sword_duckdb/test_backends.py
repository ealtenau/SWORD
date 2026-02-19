# -*- coding: utf-8 -*-
"""
Tests for Database Backend Abstraction Layer
============================================

Tests for DuckDB and PostgreSQL backends, including:
- Connection management
- Query execution
- Transaction handling
- Placeholder conversion
- UPSERT formatting
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.db

from src.updates.sword_duckdb.backends import (
    BackendType,
    DuckDBBackend,
    PostgresBackend,
    detect_backend_type,
    get_backend,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_duckdb():
    """Create a temporary DuckDB database."""
    # Use tempfile to get a unique path, then delete it so DuckDB can create fresh
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=True) as f:
        db_path = f.name

    # db_path now points to a non-existent file that DuckDB will create
    backend = DuckDBBackend(db_path)
    conn = backend.connect()

    # Create test table
    conn.execute("""
        CREATE TABLE test_reaches (
            reach_id BIGINT PRIMARY KEY,
            region VARCHAR(2),
            dist_out DOUBLE,
            facc DOUBLE
        )
    """)

    # Insert test data
    conn.execute("""
        INSERT INTO test_reaches VALUES
            (1, 'NA', 1000.0, 500.0),
            (2, 'NA', 2000.0, 1000.0),
            (3, 'EU', 3000.0, 1500.0)
    """)

    yield backend

    backend.close()
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def memory_duckdb():
    """Create an in-memory DuckDB backend."""
    backend = DuckDBBackend(":memory:")
    conn = backend.connect()

    # Create test table
    conn.execute("""
        CREATE TABLE test_data (
            id INTEGER PRIMARY KEY,
            name VARCHAR,
            value DOUBLE
        )
    """)

    yield backend

    backend.close()


# =============================================================================
# BACKEND TYPE DETECTION TESTS
# =============================================================================


class TestBackendTypeDetection:
    """Tests for backend type detection."""

    def test_detect_duckdb_from_path(self):
        """DuckDB detected from .duckdb file path."""
        assert detect_backend_type("data/sword.duckdb") == BackendType.DUCKDB
        assert detect_backend_type("/absolute/path/db.duckdb") == BackendType.DUCKDB
        assert detect_backend_type(Path("relative/db.duckdb")) == BackendType.DUCKDB

    def test_detect_duckdb_from_memory(self):
        """DuckDB detected from :memory: path."""
        assert detect_backend_type(":memory:") == BackendType.DUCKDB

    def test_detect_postgres_from_url(self):
        """PostgreSQL detected from connection URLs."""
        assert (
            detect_backend_type("postgresql://user:pass@host/db")
            == BackendType.POSTGRES
        )
        assert detect_backend_type("postgres://localhost/sword") == BackendType.POSTGRES
        assert detect_backend_type("psql://host:5432/db") == BackendType.POSTGRES

    def test_detect_default_to_duckdb(self):
        """Unknown paths default to DuckDB."""
        assert detect_backend_type("unknown.db") == BackendType.DUCKDB


# =============================================================================
# DUCKDB BACKEND TESTS
# =============================================================================


class TestDuckDBBackend:
    """Tests for DuckDB backend implementation."""

    def test_properties(self, memory_duckdb):
        """Backend properties are correct."""
        assert memory_duckdb.backend_type == BackendType.DUCKDB
        assert memory_duckdb.placeholder == "?"

    def test_connect(self, memory_duckdb):
        """Connection is established."""
        conn = memory_duckdb.connect()
        assert conn is not None
        assert memory_duckdb.is_connected

    def test_query(self, temp_duckdb):
        """Query returns DataFrame."""
        df = temp_duckdb.query("SELECT * FROM test_reaches")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "reach_id" in df.columns

    def test_query_with_params(self, temp_duckdb):
        """Parameterized query works."""
        df = temp_duckdb.query("SELECT * FROM test_reaches WHERE region = ?", ["NA"])
        assert len(df) == 2

    def test_execute(self, memory_duckdb):
        """Execute inserts data."""
        memory_duckdb.execute(
            "INSERT INTO test_data VALUES (?, ?, ?)", [1, "test", 123.45]
        )

        df = memory_duckdb.query("SELECT * FROM test_data")
        assert len(df) == 1
        assert df.iloc[0]["name"] == "test"

    def test_executemany(self, memory_duckdb):
        """Executemany inserts multiple rows."""
        data = [
            (1, "a", 1.0),
            (2, "b", 2.0),
            (3, "c", 3.0),
        ]
        memory_duckdb.executemany("INSERT INTO test_data VALUES (?, ?, ?)", data)

        df = memory_duckdb.query("SELECT COUNT(*) as cnt FROM test_data")
        assert df.iloc[0]["cnt"] == 3

    def test_transaction_commit(self, temp_duckdb):
        """Transaction commits on success."""
        with temp_duckdb.transaction() as ctx:
            temp_duckdb.execute(
                "INSERT INTO test_reaches VALUES (?, ?, ?, ?)",
                [4, "AS", 4000.0, 2000.0],
                connection=ctx.connection,
            )

        df = temp_duckdb.query("SELECT * FROM test_reaches WHERE reach_id = 4")
        assert len(df) == 1

    def test_transaction_rollback(self, temp_duckdb):
        """Transaction rolls back on error."""
        try:
            with temp_duckdb.transaction() as ctx:
                temp_duckdb.execute(
                    "INSERT INTO test_reaches VALUES (?, ?, ?, ?)",
                    [5, "AF", 5000.0, 2500.0],
                    connection=ctx.connection,
                )
                raise ValueError("Simulated error")
        except ValueError:
            pass

        df = temp_duckdb.query("SELECT * FROM test_reaches WHERE reach_id = 5")
        assert len(df) == 0

    def test_format_upsert(self, memory_duckdb):
        """UPSERT format is correct for DuckDB."""
        sql = memory_duckdb.format_upsert("test_table", ["id", "name", "value"], ["id"])
        assert "INSERT OR REPLACE" in sql
        assert "test_table" in sql

    def test_format_array_literal(self, memory_duckdb):
        """Array literal format is correct for DuckDB."""
        result = memory_duckdb.format_array_literal([1, 2, 3])
        assert result == "[1, 2, 3]"

        result = memory_duckdb.format_array_literal(["a", "b"])
        assert result == "['a', 'b']"

    def test_convert_placeholders(self, memory_duckdb):
        """DuckDB keeps ? placeholders unchanged."""
        sql = "SELECT * FROM t WHERE id = ? AND name = ?"
        assert memory_duckdb.convert_placeholders(sql) == sql


# =============================================================================
# POSTGRES BACKEND TESTS (require running PostgreSQL)
# =============================================================================


@pytest.mark.postgres
@pytest.mark.skipif(
    not os.environ.get("SWORD_TEST_POSTGRES_URL"),
    reason="SWORD_TEST_POSTGRES_URL not set",
)
class TestPostgresBackend:
    """Tests for PostgreSQL backend (requires live database)."""

    @pytest.fixture
    def pg_backend(self):
        """Create PostgreSQL backend from env."""
        backend = PostgresBackend(
            os.environ["SWORD_TEST_POSTGRES_URL"],
            min_connections=1,
            max_connections=5,
        )
        yield backend
        backend.close()

    def test_properties(self, pg_backend):
        """Backend properties are correct."""
        assert pg_backend.backend_type == BackendType.POSTGRES
        assert pg_backend.placeholder == "%s"

    def test_connect(self, pg_backend):
        """Connection is established."""
        conn = pg_backend.connect()
        assert conn is not None
        pg_backend.release_connection(conn)

    def test_convert_placeholders(self, pg_backend):
        """PostgreSQL converts ? to %s."""
        sql = "SELECT * FROM t WHERE id = ? AND name = ?"
        result = pg_backend.convert_placeholders(sql)
        assert result == "SELECT * FROM t WHERE id = %s AND name = %s"

    def test_format_upsert(self, pg_backend):
        """UPSERT format is correct for PostgreSQL."""
        sql = pg_backend.format_upsert("test_table", ["id", "name", "value"], ["id"])
        assert "ON CONFLICT" in sql
        assert "DO UPDATE SET" in sql

    def test_format_array_literal(self, pg_backend):
        """Array literal format is correct for PostgreSQL."""
        result = pg_backend.format_array_literal([1, 2, 3])
        assert "ARRAY[" in result

    def test_region_lock_id(self, pg_backend):
        """Region lock IDs are consistent."""
        id1 = pg_backend._region_lock_id("NA")
        id2 = pg_backend._region_lock_id("NA")
        id3 = pg_backend._region_lock_id("EU")

        assert id1 == id2  # Same region = same ID
        assert id1 != id3  # Different region = different ID


# =============================================================================
# FACTORY TESTS
# =============================================================================


class TestBackendFactory:
    """Tests for backend factory functions."""

    def test_get_backend_duckdb(self):
        """Factory creates DuckDB backend from path."""
        backend = get_backend(":memory:")
        assert isinstance(backend, DuckDBBackend)
        backend.close()

    def test_get_backend_duckdb_with_options(self):
        """Factory passes options to DuckDB backend."""
        backend = get_backend(":memory:", read_only=False, spatial=False)
        assert isinstance(backend, DuckDBBackend)
        assert backend.spatial is False
        backend.close()

    def test_get_backend_explicit_type(self):
        """Factory respects explicit backend type."""
        backend = get_backend(":memory:", backend_type=BackendType.DUCKDB)
        assert isinstance(backend, DuckDBBackend)
        backend.close()

    @pytest.mark.skipif(
        not os.environ.get("SWORD_TEST_POSTGRES_URL"),
        reason="SWORD_TEST_POSTGRES_URL not set",
    )
    def test_get_backend_postgres(self):
        """Factory creates PostgreSQL backend from URL."""
        backend = get_backend(os.environ["SWORD_TEST_POSTGRES_URL"])
        assert isinstance(backend, PostgresBackend)
        backend.close()


# =============================================================================
# SWORD DATABASE DELEGATION TESTS
# =============================================================================


class TestSWORDDatabaseDelegation:
    """Tests that SWORDDatabase delegates correctly to the backend."""

    def test_backend_type_duckdb(self):
        """SWORDDatabase(:memory:) reports DuckDB backend type."""
        from src.updates.sword_duckdb.sword_db import SWORDDatabase

        db = SWORDDatabase(":memory:")
        assert db.backend_type == BackendType.DUCKDB
        db.close()

    def test_is_duckdb_true(self):
        """is_duckdb is True for DuckDB backend."""
        from src.updates.sword_duckdb.sword_db import SWORDDatabase

        db = SWORDDatabase(":memory:")
        assert db.is_duckdb is True
        assert db.is_postgres is False
        db.close()

    def test_format_upsert_delegates(self):
        """format_upsert delegates to backend (INSERT OR REPLACE for DuckDB)."""
        from src.updates.sword_duckdb.sword_db import SWORDDatabase

        db = SWORDDatabase(":memory:")
        sql = db.format_upsert("test_table", ["id", "name"], ["id"])
        assert "INSERT OR REPLACE" in sql
        db.close()

    def test_format_array_literal_delegates(self):
        """format_array_literal delegates to backend."""
        from src.updates.sword_duckdb.sword_db import SWORDDatabase

        db = SWORDDatabase(":memory:")
        result = db.format_array_literal([1, 2, 3])
        assert result == "[1, 2, 3]"
        db.close()

    def test_convert_placeholders_identity(self):
        """convert_placeholders is identity for DuckDB (? stays ?)."""
        from src.updates.sword_duckdb.sword_db import SWORDDatabase

        db = SWORDDatabase(":memory:")
        sql = "SELECT * FROM t WHERE id = ? AND name = ?"
        # SWORDDatabase.query calls convert_placeholders internally;
        # verify via the backend directly
        converted = db._backend.convert_placeholders(sql)
        assert converted == sql
        db.close()
