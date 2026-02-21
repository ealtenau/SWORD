# -*- coding: utf-8 -*-
"""
SWORD DuckDB Backend
====================

This module provides DuckDB-based storage for the SWOT River Database (SWORD).
It offers SQL query capabilities, spatial queries, and improved performance
while maintaining compatibility with existing workflows.

Modules:
    sword_class: DuckDB-backed SWORD class (drop-in replacement for original)
    sword_db: Connection management and database operations
    schema: Table definitions and schema creation
    migrations: NetCDF to DuckDB migration utilities
    views: View wrapper classes for numpy-array-style access
    reactive: Automatic recalculation of derived attributes
    workflow: High-level workflow orchestration with provenance
    provenance: Operation logging and audit trail
    triggers: Change tracking for reactive updates
    export: PostgreSQL, GeoParquet, and GeoPackage export functions
    reconstruction: Attribute reconstruction from source data

Example Usage:
    from sword_duckdb import SWORD

    # Load from DuckDB (same interface as original SWORD class)
    sword = SWORD('data/duckdb/sword_v17b.duckdb', 'NA')

    # Access data
    print(sword.reaches.wse[:5])
    print(sword.nodes.facc[:5])

    # Reactive updates (automatic recalculation)
    from sword_duckdb import SWORDReactive, mark_geometry_changed
    reactive = SWORDReactive(sword)
    mark_geometry_changed(reactive, reach_ids=[12345678901])
    reactive.recalculate()
"""

# --- Core imports (always available) ---
from .column_order import (
    CENTERLINES_COLUMN_ORDER,
    NODES_COLUMN_ORDER,
    REACHES_COLUMN_ORDER,
    get_column_order,
    reorder_columns,
)
from .schema import create_schema, get_schema_sql, SCHEMA_VERSION
from .lint import (
    LintRunner,
    Severity,
    Category,
    CheckResult,
    get_registry as get_lint_registry,
)

# --- Optional imports (require full dependency tree) ---
# Wrapped so that lint-only / reviewer-only usage works without psycopg2, aiohttp, etc.
try:
    from .sword_db import SWORDDatabase, create_database
except ImportError:
    SWORDDatabase = create_database = None

try:
    from .migrations import (
        migrate_region,
        migrate_all_regions,
        validate_migration,
        build_all_geometry,
    )
except ImportError:
    migrate_region = migrate_all_regions = validate_migration = build_all_geometry = (
        None
    )

try:
    from .sword_class import SWORD
except ImportError:
    SWORD = None

try:
    from .views import CenterlinesView, NodesView, ReachesView, WritableArray
except ImportError:
    CenterlinesView = NodesView = ReachesView = WritableArray = None

try:
    from .reactive import (
        SWORDReactive,
        DependencyGraph,
        ChangeType,
        mark_geometry_changed,
        mark_topology_changed,
        full_recalculate,
    )
except ImportError:
    SWORDReactive = DependencyGraph = ChangeType = None
    mark_geometry_changed = mark_topology_changed = full_recalculate = None

try:
    from .export import (
        export_to_postgres,
        export_to_geoparquet,
        export_to_geopackage,
        sync_from_postgres,
        PostgresExportError,
        PgConnectionError,
        AuthenticationError,
        NetworkError,
    )
except ImportError:
    export_to_postgres = export_to_geoparquet = export_to_geopackage = None
    sync_from_postgres = None
    PostgresExportError = PgConnectionError = AuthenticationError = NetworkError = None

try:
    from .triggers import (
        install_triggers,
        remove_triggers,
        get_pending_changes,
        get_changed_entities,
        mark_changes_synced,
        get_trigger_sql,
    )
except ImportError:
    install_triggers = remove_triggers = get_pending_changes = None
    get_changed_entities = mark_changes_synced = get_trigger_sql = None

try:
    from .workflow import SWORDWorkflow
except ImportError:
    SWORDWorkflow = None

try:
    from .provenance import ProvenanceLogger, OperationType, OperationStatus
except ImportError:
    ProvenanceLogger = OperationType = OperationStatus = None

try:
    from .schema import (
        create_provenance_tables,
        add_v17c_columns,
        add_swot_obs_columns,
        add_sync_tracking_column,
    )
except ImportError:
    create_provenance_tables = add_v17c_columns = None
    add_swot_obs_columns = add_sync_tracking_column = None

try:
    from .reconstruction import (
        ReconstructionEngine,
        SourceDataset,
        DerivationMethod,
        AttributeSpec,
        ATTRIBUTE_SOURCES,
    )
except ImportError:
    ReconstructionEngine = SourceDataset = DerivationMethod = None
    AttributeSpec = ATTRIBUTE_SOURCES = None

try:
    from .backends import (
        DatabaseBackend,
        BackendType,
        DuckDBBackend,
        PostgresBackend,
        get_backend,
        detect_backend_type,
    )
    from .backends.base import IsolationLevel, TransactionContext
except ImportError:
    DatabaseBackend = BackendType = DuckDBBackend = PostgresBackend = None
    get_backend = detect_backend_type = None
    IsolationLevel = TransactionContext = None

__all__ = [
    # Column ordering
    "REACHES_COLUMN_ORDER",
    "NODES_COLUMN_ORDER",
    "CENTERLINES_COLUMN_ORDER",
    "get_column_order",
    "reorder_columns",
    # Main SWORD class (drop-in replacement)
    "SWORD",
    # Workflow orchestration (recommended entry point)
    "SWORDWorkflow",
    # View classes
    "CenterlinesView",
    "NodesView",
    "ReachesView",
    "WritableArray",
    # Reactive updates
    "SWORDReactive",
    "DependencyGraph",
    "ChangeType",
    "mark_geometry_changed",
    "mark_topology_changed",
    "full_recalculate",
    # Provenance logging
    "ProvenanceLogger",
    "OperationType",
    "OperationStatus",
    # Reconstruction
    "ReconstructionEngine",
    "SourceDataset",
    "DerivationMethod",
    "AttributeSpec",
    "ATTRIBUTE_SOURCES",
    # Connection management
    "SWORDDatabase",
    "create_database",
    # Schema
    "create_schema",
    "create_provenance_tables",
    "add_v17c_columns",
    "add_swot_obs_columns",
    "add_sync_tracking_column",
    "get_schema_sql",
    "SCHEMA_VERSION",
    # Migration
    "migrate_region",
    "migrate_all_regions",
    "validate_migration",
    "build_all_geometry",
    # Export functions
    "export_to_postgres",
    "export_to_geoparquet",
    "export_to_geopackage",
    "sync_from_postgres",
    "PostgresExportError",
    "PgConnectionError",
    "AuthenticationError",
    "NetworkError",
    # Trigger functions
    "install_triggers",
    "remove_triggers",
    "get_pending_changes",
    "get_changed_entities",
    "mark_changes_synced",
    "get_trigger_sql",
    # Lint framework
    "LintRunner",
    "Severity",
    "Category",
    "CheckResult",
    "get_lint_registry",
    # Database backends
    "DatabaseBackend",
    "BackendType",
    "DuckDBBackend",
    "PostgresBackend",
    "get_backend",
    "detect_backend_type",
    "IsolationLevel",
    "TransactionContext",
]
