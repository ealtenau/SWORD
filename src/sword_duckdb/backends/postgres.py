# -*- coding: utf-8 -*-
"""
PostgreSQL Backend Implementation
=================================

Provides PostgreSQL/PostGIS backend with connection pooling and
advisory locking for concurrent multi-user access.

PostgreSQL is optimized for:
- Concurrent multi-user access
- Region-level advisory locking
- PostGIS spatial operations
- Proper transaction isolation

Connection pooling uses psycopg2's ThreadedConnectionPool for
efficient connection reuse in multi-threaded environments.
"""

from __future__ import annotations

import hashlib
import logging
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Tuple

import pandas as pd

if TYPE_CHECKING:
    import psycopg2

from .base import (
    BackendType,
    BaseBackend,
    ConnectionType,
    IsolationLevel,
    TransactionContext,
)

logger = logging.getLogger(__name__)


class PostgresConnectionError(Exception):
    """Failed to connect to PostgreSQL."""

    pass


class PostgresAuthenticationError(Exception):
    """Authentication failed."""

    pass


class PostgresBackend(BaseBackend):
    """
    PostgreSQL backend implementation with connection pooling.

    Provides pooled connections to a PostgreSQL/PostGIS database
    with region-level advisory locking for concurrent access.

    Parameters
    ----------
    connection_string : str
        PostgreSQL connection string (e.g., "postgresql://user:pass@host/db")
    min_connections : int, optional
        Minimum connections in pool. Default is 1.
    max_connections : int, optional
        Maximum connections in pool. Default is 10.
    enable_postgis : bool, optional
        If True, enables PostGIS extension. Default is True.

    Example
    -------
    >>> backend = PostgresBackend('postgresql://user:pass@localhost/sword')
    >>> conn = backend.connect()
    >>> with backend.acquire_region_lock('NA'):
    ...     df = backend.query("SELECT * FROM reaches WHERE region = 'NA' LIMIT 10")
    >>> backend.close()
    """

    def __init__(
        self,
        connection_string: str,
        min_connections: int = 1,
        max_connections: int = 10,
        enable_postgis: bool = True,
    ):
        super().__init__()
        self.connection_string = connection_string
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.enable_postgis = enable_postgis
        self._pool = None
        self._postgis_enabled = False
        self._locked_regions: Dict[str, ConnectionType] = {}

    @property
    def backend_type(self) -> BackendType:
        """Return the backend type."""
        return BackendType.POSTGRES

    @property
    def placeholder(self) -> str:
        """Return the SQL placeholder ('%s' for PostgreSQL)."""
        return "%s"

    @property
    def is_connected(self) -> bool:
        """Check if the pool is initialized."""
        return self._pool is not None

    def _get_psycopg2(self):
        """Import psycopg2, raising helpful error if not installed."""
        try:
            import psycopg2
            from psycopg2 import pool

            return psycopg2, pool
        except ImportError:
            raise ImportError(
                "psycopg2 is required for PostgreSQL backend. "
                "Install with: pip install psycopg2-binary"
            )

    def connect(self) -> "psycopg2.extensions.connection":
        """
        Get a connection from the pool.

        Initializes the pool on first call.

        Returns
        -------
        psycopg2.connection
            Database connection from the pool.
        """
        if self._pool is None:
            self._init_pool()

        conn = self._pool.getconn()

        # Enable PostGIS if requested and not already done
        if self.enable_postgis and not self._postgis_enabled:
            self._enable_postgis(conn)

        return conn

    def _init_pool(self) -> None:
        """Initialize the connection pool with retry logic."""
        psycopg2, pool_module = self._get_psycopg2()

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                self._pool = pool_module.ThreadedConnectionPool(
                    self.min_connections,
                    self.max_connections,
                    self.connection_string,
                )
                logger.info(
                    f"PostgreSQL pool initialized: "
                    f"{self.min_connections}-{self.max_connections} connections"
                )
                return

            except psycopg2.OperationalError as e:
                last_error = e
                error_msg = str(e).lower()

                # Check for non-retryable errors
                if "authentication failed" in error_msg or "password" in error_msg:
                    raise PostgresAuthenticationError(
                        f"PostgreSQL authentication failed. Check credentials. "
                        f"Original error: {e}"
                    ) from e

                if "does not exist" in error_msg:
                    raise PostgresConnectionError(
                        f"Database does not exist. Create it first. Original error: {e}"
                    ) from e

                # Retryable error - wait and retry
                if attempt < max_retries - 1:
                    delay = 2**attempt
                    logger.warning(
                        f"Connection attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)

        raise PostgresConnectionError(
            f"Failed to connect after {max_retries} attempts. Last error: {last_error}"
        ) from last_error

    def _enable_postgis(self, conn) -> None:
        """Enable PostGIS extension on the connection."""
        try:
            cursor = conn.cursor()
            cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
            conn.commit()
            cursor.close()
            self._postgis_enabled = True
            logger.debug("PostGIS extension enabled")
        except Exception as e:
            conn.rollback()
            logger.warning(f"Could not enable PostGIS: {e}")

    def release_connection(self, conn) -> None:
        """Return a connection to the pool."""
        if self._pool and conn:
            self._pool.putconn(conn)

    def close(self) -> None:
        """Close all connections and the pool."""
        # Release any held region locks
        for region, conn in list(self._locked_regions.items()):
            try:
                self._release_advisory_lock(conn, region)
                self.release_connection(conn)
            except Exception as e:
                logger.warning(f"Error releasing lock for {region}: {e}")

        self._locked_regions.clear()

        if self._pool:
            self._pool.closeall()
            self._pool = None
            logger.info("PostgreSQL pool closed")

    def query(
        self,
        sql: str,
        params: Optional[List[Any]] = None,
        connection: Optional[ConnectionType] = None,
    ) -> pd.DataFrame:
        """
        Execute a SQL query and return results as a DataFrame.

        Parameters
        ----------
        sql : str
            SQL query to execute.
        params : list, optional
            Query parameters for parameterized queries.
        connection : optional
            Specific connection to use. If None, gets one from pool.

        Returns
        -------
        pd.DataFrame
            Query results.
        """
        conn = connection
        should_release = False

        if conn is None:
            conn = self.connect()
            should_release = True

        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            cursor.close()
            return pd.DataFrame(data, columns=columns)
        finally:
            if should_release:
                self.release_connection(conn)

    def execute(
        self,
        sql: str,
        params: Optional[List[Any]] = None,
        connection: Optional[ConnectionType] = None,
    ) -> Any:
        """
        Execute a SQL statement.

        Parameters
        ----------
        sql : str
            SQL statement to execute.
        params : list, optional
            Query parameters for parameterized queries.
        connection : optional
            Specific connection to use.

        Returns
        -------
        cursor
            Cursor after execution (for rowcount, etc).
        """
        conn = connection
        should_release = False

        if conn is None:
            conn = self.connect()
            should_release = True

        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            return cursor
        except Exception:
            conn.rollback()
            raise
        finally:
            if should_release:
                self.release_connection(conn)

    def executemany(
        self,
        sql: str,
        params_list: List[Tuple[Any, ...]],
        connection: Optional[ConnectionType] = None,
    ) -> None:
        """
        Execute a SQL statement with multiple parameter sets.

        Uses psycopg2's executemany for batch operations.

        Parameters
        ----------
        sql : str
            SQL statement to execute.
        params_list : list of tuples
            List of parameter tuples.
        connection : optional
            Specific connection to use.
        """
        conn = connection
        should_release = False

        if conn is None:
            conn = self.connect()
            should_release = True

        try:
            cursor = conn.cursor()
            cursor.executemany(sql, params_list)
            conn.commit()
            cursor.close()
        except Exception:
            conn.rollback()
            raise
        finally:
            if should_release:
                self.release_connection(conn)

    @contextmanager
    def transaction(
        self,
        isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED,
        connection: Optional[ConnectionType] = None,
    ) -> Generator[TransactionContext, None, None]:
        """
        Context manager for database transactions.

        Parameters
        ----------
        isolation_level : IsolationLevel
            Transaction isolation level.
        connection : optional
            Specific connection to use.

        Yields
        ------
        TransactionContext
            Context object with transaction details.
        """
        psycopg2, _ = self._get_psycopg2()
        from psycopg2 import extensions

        # Map our isolation levels to psycopg2 constants
        level_map = {
            IsolationLevel.READ_UNCOMMITTED: extensions.ISOLATION_LEVEL_READ_UNCOMMITTED,
            IsolationLevel.READ_COMMITTED: extensions.ISOLATION_LEVEL_READ_COMMITTED,
            IsolationLevel.REPEATABLE_READ: extensions.ISOLATION_LEVEL_REPEATABLE_READ,
            IsolationLevel.SERIALIZABLE: extensions.ISOLATION_LEVEL_SERIALIZABLE,
        }

        conn = connection
        should_release = False

        if conn is None:
            conn = self.connect()
            should_release = True

        # Save original isolation level
        original_level = conn.isolation_level

        try:
            # Set the requested isolation level
            conn.set_isolation_level(level_map[isolation_level])

            ctx = TransactionContext(
                backend_type=BackendType.POSTGRES,
                isolation_level=isolation_level,
                connection=conn,
            )

            yield ctx

            conn.commit()

        except Exception:
            conn.rollback()
            raise

        finally:
            # Restore original isolation level
            conn.set_isolation_level(original_level)

            if should_release:
                self.release_connection(conn)

    def commit(self, connection: Optional[ConnectionType] = None) -> None:
        """Commit the current transaction."""
        conn = connection or self._connection
        if conn:
            conn.commit()

    def rollback(self, connection: Optional[ConnectionType] = None) -> None:
        """Rollback the current transaction."""
        conn = connection or self._connection
        if conn:
            conn.rollback()

    def convert_placeholders(self, sql: str) -> str:
        """
        Convert '?' placeholders to '%s' for PostgreSQL.

        Parameters
        ----------
        sql : str
            SQL with '?' placeholders.

        Returns
        -------
        str
            SQL with '%s' placeholders.
        """
        # Simple replacement - be careful with ? in string literals
        # This handles the common case; complex cases may need more logic
        return sql.replace("?", "%s")

    def format_upsert(
        self,
        table: str,
        columns: List[str],
        key_columns: List[str],
    ) -> str:
        """
        Generate PostgreSQL UPSERT SQL using ON CONFLICT DO UPDATE.

        Parameters
        ----------
        table : str
            Table name.
        columns : list of str
            All columns to insert/update.
        key_columns : list of str
            Primary key columns for conflict detection.

        Returns
        -------
        str
            UPSERT SQL template with '%s' placeholders.
        """
        placeholders = ", ".join(["%s"] * len(columns))
        col_list = ", ".join(columns)
        key_list = ", ".join(key_columns)

        # Columns to update (exclude key columns)
        update_cols = [c for c in columns if c not in key_columns]
        update_set = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)

        return f"""
            INSERT INTO {table} ({col_list})
            VALUES ({placeholders})
            ON CONFLICT ({key_list}) DO UPDATE SET {update_set}
        """

    def format_array_literal(self, values: List[Any]) -> str:
        """
        Format a list as a PostgreSQL ARRAY literal.

        PostgreSQL uses ARRAY[v1, v2, v3] or '{v1,v2,v3}' syntax.

        Parameters
        ----------
        values : list
            Values to format as array.

        Returns
        -------
        str
            PostgreSQL array literal.
        """
        formatted = ", ".join(
            f"'{v}'" if isinstance(v, str) else str(v) for v in values
        )
        return f"ARRAY[{formatted}]"

    # =========================================================================
    # REGION LOCKING
    # =========================================================================

    def _region_lock_id(self, region: str) -> int:
        """
        Generate a consistent lock ID for a region.

        Uses a hash of the region name to get a consistent integer.
        """
        # Use hashtext equivalent - hash the region name
        hash_bytes = hashlib.md5(region.encode()).digest()
        # Take first 4 bytes as signed int32
        lock_id = int.from_bytes(hash_bytes[:4], byteorder="big", signed=True)
        return lock_id

    @contextmanager
    def acquire_region_lock(
        self,
        region: str,
        timeout_ms: int = 30000,
        connection: Optional[ConnectionType] = None,
    ) -> Generator[ConnectionType, None, None]:
        """
        Acquire an advisory lock for a region.

        This provides exclusive access to a region, preventing concurrent
        edits to the same region by different users.

        Parameters
        ----------
        region : str
            Region code (e.g., 'NA', 'EU').
        timeout_ms : int
            Lock acquisition timeout in milliseconds. Default is 30s.
        connection : optional
            Specific connection to use.

        Yields
        ------
        connection
            The connection holding the lock.

        Raises
        ------
        TimeoutError
            If lock cannot be acquired within timeout.
        """
        conn = connection
        should_release = False

        if conn is None:
            conn = self.connect()
            should_release = True

        lock_id = self._region_lock_id(region)

        try:
            # Try to acquire advisory lock with timeout
            cursor = conn.cursor()
            cursor.execute("SET lock_timeout = %s", [f"{timeout_ms}ms"])

            # pg_advisory_lock blocks until lock is acquired
            cursor.execute("SELECT pg_advisory_lock(%s)", [lock_id])
            cursor.close()

            self._locked_regions[region] = conn
            logger.debug(f"Acquired advisory lock for region {region}")

            yield conn

        except Exception as e:
            if "lock timeout" in str(e).lower():
                raise TimeoutError(
                    f"Could not acquire lock for region {region} within "
                    f"{timeout_ms}ms. Another user may be editing this region."
                ) from e
            raise

        finally:
            # Release the lock
            if region in self._locked_regions:
                self._release_advisory_lock(conn, region)
                del self._locked_regions[region]

            if should_release:
                self.release_connection(conn)

    def _release_advisory_lock(self, conn, region: str) -> None:
        """Release an advisory lock for a region."""
        lock_id = self._region_lock_id(region)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT pg_advisory_unlock(%s)", [lock_id])
            cursor.close()
            logger.debug(f"Released advisory lock for region {region}")
        except Exception as e:
            logger.warning(f"Error releasing advisory lock: {e}")

    def has_region_lock(self, region: str) -> bool:
        """Check if we hold the lock for a region."""
        return region in self._locked_regions

    # =========================================================================
    # SYNC TRACKING
    # =========================================================================

    def mark_synced(
        self,
        operation_ids: List[int],
        connection: Optional[ConnectionType] = None,
    ) -> int:
        """
        Mark operations as synced to DuckDB.

        Parameters
        ----------
        operation_ids : list of int
            Operation IDs to mark as synced.
        connection : optional
            Specific connection to use.

        Returns
        -------
        int
            Number of operations marked.
        """
        if not operation_ids:
            return 0

        sql = """
            UPDATE sword_operations
            SET synced_to_duckdb = TRUE
            WHERE operation_id = ANY(%s)
        """

        conn = connection
        should_release = False

        if conn is None:
            conn = self.connect()
            should_release = True

        try:
            cursor = conn.cursor()
            cursor.execute(sql, [operation_ids])
            count = cursor.rowcount
            conn.commit()
            cursor.close()
            return count
        finally:
            if should_release:
                self.release_connection(conn)

    def get_unsynced_operations(
        self,
        since_operation_id: int = 0,
        connection: Optional[ConnectionType] = None,
    ) -> pd.DataFrame:
        """
        Get operations that haven't been synced to DuckDB.

        Parameters
        ----------
        since_operation_id : int
            Only get operations after this ID.
        connection : optional
            Specific connection to use.

        Returns
        -------
        pd.DataFrame
            Unsynced operations.
        """
        sql = """
            SELECT operation_id, operation_type, table_name, entity_ids, region
            FROM sword_operations
            WHERE synced_to_duckdb = FALSE
              AND operation_id > %s
              AND status = 'COMPLETED'
            ORDER BY operation_id
        """
        return self.query(sql, [since_operation_id], connection)
