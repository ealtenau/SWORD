# -*- coding: utf-8 -*-
"""
Database Backend Protocol
=========================

Defines the interface that all database backends must implement.
This allows SWORD to work with DuckDB, PostgreSQL, or other databases
through a common interface.

The backend abstraction handles:
- Connection management (pooling for PG, single connection for DuckDB)
- SQL dialect differences (? vs %s placeholders, UPSERT syntax)
- Transaction management with isolation levels
- Locking mechanisms (advisory locks for PG)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

import pandas as pd

if TYPE_CHECKING:
    import duckdb
    import psycopg2


class BackendType(Enum):
    """Supported database backend types."""
    DUCKDB = "duckdb"
    POSTGRES = "postgres"


class IsolationLevel(Enum):
    """Transaction isolation levels."""
    READ_UNCOMMITTED = "READ UNCOMMITTED"
    READ_COMMITTED = "READ COMMITTED"
    REPEATABLE_READ = "REPEATABLE READ"
    SERIALIZABLE = "SERIALIZABLE"


@dataclass
class TransactionContext:
    """Context for an active transaction."""
    backend_type: BackendType
    isolation_level: IsolationLevel
    connection: Any  # duckdb.DuckDBPyConnection or psycopg2.connection
    savepoint_name: Optional[str] = None


# Type alias for connection objects
ConnectionType = Union['duckdb.DuckDBPyConnection', 'psycopg2.extensions.connection']


@runtime_checkable
class DatabaseBackend(Protocol):
    """
    Protocol defining the interface for database backends.

    All backend implementations must provide these methods to ensure
    SWORD can work with them interchangeably.
    """

    @property
    def backend_type(self) -> BackendType:
        """Return the backend type."""
        ...

    @property
    def placeholder(self) -> str:
        """
        Return the SQL placeholder for parameterized queries.

        Returns '?' for DuckDB, '%s' for PostgreSQL.
        """
        ...

    @property
    def is_connected(self) -> bool:
        """Check if the backend has an active connection."""
        ...

    def connect(self) -> ConnectionType:
        """
        Establish a database connection.

        For DuckDB: Creates a single connection to the file.
        For PostgreSQL: Gets a connection from the pool.

        Returns
        -------
        connection
            Database connection object.
        """
        ...

    def close(self) -> None:
        """
        Close all connections and release resources.

        For DuckDB: Closes the single connection.
        For PostgreSQL: Closes the connection pool.
        """
        ...

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
            Specific connection to use. If None, uses default.

        Returns
        -------
        pd.DataFrame
            Query results.
        """
        ...

    def execute(
        self,
        sql: str,
        params: Optional[List[Any]] = None,
        connection: Optional[ConnectionType] = None,
    ) -> Any:
        """
        Execute a SQL statement (INSERT, UPDATE, DELETE, etc).

        Parameters
        ----------
        sql : str
            SQL statement to execute.
        params : list, optional
            Query parameters for parameterized queries.
        connection : optional
            Specific connection to use. If None, uses default.

        Returns
        -------
        Any
            Backend-specific result object.
        """
        ...

    def executemany(
        self,
        sql: str,
        params_list: List[Tuple[Any, ...]],
        connection: Optional[ConnectionType] = None,
    ) -> None:
        """
        Execute a SQL statement with multiple parameter sets.

        Parameters
        ----------
        sql : str
            SQL statement to execute.
        params_list : list of tuples
            List of parameter tuples.
        connection : optional
            Specific connection to use.
        """
        ...

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
        ...

    def commit(self, connection: Optional[ConnectionType] = None) -> None:
        """Commit the current transaction."""
        ...

    def rollback(self, connection: Optional[ConnectionType] = None) -> None:
        """Rollback the current transaction."""
        ...

    def convert_placeholders(self, sql: str) -> str:
        """
        Convert SQL placeholders to the backend's format.

        Converts '?' placeholders to the appropriate format.
        For DuckDB: returns as-is
        For PostgreSQL: converts '?' to '%s'

        Parameters
        ----------
        sql : str
            SQL with '?' placeholders.

        Returns
        -------
        str
            SQL with backend-appropriate placeholders.
        """
        ...

    def format_upsert(
        self,
        table: str,
        columns: List[str],
        key_columns: List[str],
    ) -> str:
        """
        Generate backend-specific UPSERT SQL.

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
            UPSERT SQL template with placeholders.
        """
        ...

    def format_array_literal(self, values: List[Any]) -> str:
        """
        Format a list as a SQL array literal.

        Parameters
        ----------
        values : list
            Values to format as array.

        Returns
        -------
        str
            Backend-specific array literal.
        """
        ...


class BaseBackend(ABC):
    """
    Abstract base class providing common functionality for backends.

    Subclasses must implement the abstract methods to provide
    backend-specific behavior.
    """

    def __init__(self):
        self._connection: Optional[ConnectionType] = None

    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """Return the backend type."""
        pass

    @property
    @abstractmethod
    def placeholder(self) -> str:
        """Return the SQL placeholder character."""
        pass

    @property
    def is_connected(self) -> bool:
        """Check if the backend has an active connection."""
        return self._connection is not None

    @abstractmethod
    def connect(self) -> ConnectionType:
        """Establish a database connection."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close all connections."""
        pass

    @abstractmethod
    def query(
        self,
        sql: str,
        params: Optional[List[Any]] = None,
        connection: Optional[ConnectionType] = None,
    ) -> pd.DataFrame:
        """Execute a query and return DataFrame."""
        pass

    @abstractmethod
    def execute(
        self,
        sql: str,
        params: Optional[List[Any]] = None,
        connection: Optional[ConnectionType] = None,
    ) -> Any:
        """Execute a SQL statement."""
        pass

    @abstractmethod
    def executemany(
        self,
        sql: str,
        params_list: List[Tuple[Any, ...]],
        connection: Optional[ConnectionType] = None,
    ) -> None:
        """Execute a statement with multiple parameter sets."""
        pass

    @abstractmethod
    @contextmanager
    def transaction(
        self,
        isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED,
        connection: Optional[ConnectionType] = None,
    ) -> Generator[TransactionContext, None, None]:
        """Context manager for transactions."""
        pass

    @abstractmethod
    def commit(self, connection: Optional[ConnectionType] = None) -> None:
        """Commit the current transaction."""
        pass

    @abstractmethod
    def rollback(self, connection: Optional[ConnectionType] = None) -> None:
        """Rollback the current transaction."""
        pass

    def convert_placeholders(self, sql: str) -> str:
        """
        Convert '?' placeholders to backend-specific format.

        Default implementation returns SQL unchanged (for DuckDB).
        PostgreSQL backend overrides this.
        """
        return sql

    @abstractmethod
    def format_upsert(
        self,
        table: str,
        columns: List[str],
        key_columns: List[str],
    ) -> str:
        """Generate backend-specific UPSERT SQL."""
        pass

    def format_array_literal(self, values: List[Any]) -> str:
        """
        Format a list as a SQL array literal.

        Default implementation uses DuckDB syntax: [v1, v2, v3]
        """
        formatted = ', '.join(
            repr(v) if isinstance(v, str) else str(v)
            for v in values
        )
        return f"[{formatted}]"

    def __enter__(self) -> 'BaseBackend':
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
