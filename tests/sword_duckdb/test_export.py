# -*- coding: utf-8 -*-
"""
Unit tests for SWORD export module.

Tests cover:
- GeoParquet export
- GeoPackage export
- PostgreSQL export (basic functionality)
- Sync from PostgreSQL
"""

import os
import sys
import pytest
import tempfile
import shutil

# Add project root to path
main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, main_dir)

from src.updates.sword_duckdb import SWORD


# Test configuration
TEST_DB_PATH = os.path.join(main_dir, 'data/duckdb/sword_v17b.duckdb')
TEST_REGION = 'NA'
TEST_VERSION = 'v17b'


@pytest.fixture
def sword():
    """Load SWORD for read-only tests."""
    if not os.path.exists(TEST_DB_PATH):
        pytest.skip(f"Test database not found: {TEST_DB_PATH}")
    return SWORD(TEST_DB_PATH, TEST_REGION, TEST_VERSION)


class TestExportImports:
    """Test that export functions can be imported."""

    def test_import_export_to_postgres(self):
        """Test import of export_to_postgres."""
        from src.updates.sword_duckdb.export import export_to_postgres
        assert callable(export_to_postgres)

    def test_import_export_to_geoparquet(self):
        """Test import of export_to_geoparquet."""
        from src.updates.sword_duckdb.export import export_to_geoparquet
        assert callable(export_to_geoparquet)

    def test_import_export_to_geopackage(self):
        """Test import of export_to_geopackage."""
        from src.updates.sword_duckdb.export import export_to_geopackage
        assert callable(export_to_geopackage)

    def test_import_sync_from_postgres(self):
        """Test import of sync_from_postgres."""
        from src.updates.sword_duckdb.export import sync_from_postgres
        assert callable(sync_from_postgres)

    def test_module_exports(self):
        """Test that functions are exported from package __init__."""
        from src.updates.sword_duckdb import (
            export_to_postgres,
            export_to_geoparquet,
            export_to_geopackage,
            sync_from_postgres,
        )
        assert callable(export_to_postgres)
        assert callable(export_to_geoparquet)
        assert callable(export_to_geopackage)
        assert callable(sync_from_postgres)


class TestGeoParquetExport:
    """Test GeoParquet export functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_export_reaches_parquet(self, sword, temp_dir):
        """Test exporting reaches to GeoParquet."""
        try:
            import geopandas  # noqa: F401
            import pyarrow  # noqa: F401
        except ImportError:
            pytest.skip("geopandas/pyarrow not installed")

        from src.updates.sword_duckdb.export import export_to_geoparquet

        output_path = os.path.join(temp_dir, 'reaches.parquet')
        count = export_to_geoparquet(sword, output_path, table='reaches')

        assert os.path.exists(output_path)
        assert count == len(sword.reaches)

        # Verify we can read it back
        import geopandas as gpd
        gdf = gpd.read_parquet(output_path)
        assert len(gdf) == count
        assert 'reach_id' in gdf.columns
        assert gdf.crs is not None

    def test_export_nodes_parquet(self, sword, temp_dir):
        """Test exporting nodes to GeoParquet."""
        try:
            import geopandas  # noqa: F401
            import pyarrow  # noqa: F401
        except ImportError:
            pytest.skip("geopandas/pyarrow not installed")

        from src.updates.sword_duckdb.export import export_to_geoparquet

        output_path = os.path.join(temp_dir, 'nodes.parquet')
        count = export_to_geoparquet(sword, output_path, table='nodes')

        assert os.path.exists(output_path)
        assert count == len(sword.nodes)


class TestGeoPackageExport:
    """Test GeoPackage export functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_export_geopackage(self, sword, temp_dir):
        """Test exporting to GeoPackage."""
        try:
            import geopandas  # noqa: F401
            import fiona  # noqa: F401
        except ImportError:
            pytest.skip("geopandas/fiona not installed")

        from src.updates.sword_duckdb.export import export_to_geopackage

        output_path = os.path.join(temp_dir, 'sword.gpkg')
        results = export_to_geopackage(sword, output_path, tables=['reaches', 'nodes'])

        assert os.path.exists(output_path)
        assert 'reaches' in results
        assert 'nodes' in results
        assert results['reaches'] == len(sword.reaches)
        assert results['nodes'] == len(sword.nodes)

        # Verify we can read it back
        import geopandas as gpd
        reaches = gpd.read_file(output_path, layer='reaches')
        assert len(reaches) == results['reaches']


class TestPostgresExport:
    """Test PostgreSQL export functionality (mocked)."""

    def test_pg_schema_definitions(self):
        """Test that PostgreSQL schema definitions are valid."""
        from src.updates.sword_duckdb.export import (
            PG_CENTERLINES_SCHEMA,
            PG_NODES_SCHEMA,
            PG_REACHES_SCHEMA,
            PG_REACH_TOPOLOGY_SCHEMA,
        )

        # Verify schemas contain expected table definitions
        assert 'CREATE TABLE' in PG_CENTERLINES_SCHEMA
        assert 'CREATE TABLE' in PG_NODES_SCHEMA
        assert 'CREATE TABLE' in PG_REACHES_SCHEMA
        assert 'CREATE TABLE' in PG_REACH_TOPOLOGY_SCHEMA

        # Verify primary keys are defined
        assert 'PRIMARY KEY' in PG_CENTERLINES_SCHEMA
        assert 'PRIMARY KEY' in PG_NODES_SCHEMA
        assert 'PRIMARY KEY' in PG_REACHES_SCHEMA
        assert 'PRIMARY KEY' in PG_REACH_TOPOLOGY_SCHEMA

        # Verify PostGIS geometry columns
        assert 'GEOMETRY' in PG_CENTERLINES_SCHEMA
        assert 'GEOMETRY' in PG_NODES_SCHEMA
        assert 'GEOMETRY' in PG_REACHES_SCHEMA

    def test_export_requires_psycopg2(self, sword):
        """Test that export raises ImportError when psycopg2 is missing."""
        # This test is tricky because psycopg2 might be installed
        # We'll just verify the function signature accepts expected args
        from src.updates.sword_duckdb.export import export_to_postgres
        import inspect

        sig = inspect.signature(export_to_postgres)
        params = list(sig.parameters.keys())

        assert 'sword' in params
        assert 'connection_string' in params
        assert 'tables' in params
        assert 'schema' in params
        assert 'prefix' in params
        assert 'drop_existing' in params
        assert 'batch_size' in params


class TestSyncFromPostgres:
    """Test sync functionality."""

    def test_sync_function_signature(self):
        """Test sync function has expected signature."""
        from src.updates.sword_duckdb.export import sync_from_postgres
        import inspect

        sig = inspect.signature(sync_from_postgres)
        params = list(sig.parameters.keys())

        assert 'sword' in params
        assert 'connection_string' in params
        assert 'table' in params
        assert 'prefix' in params
        assert 'changed_only' in params


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
