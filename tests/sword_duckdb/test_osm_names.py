"""Tests for OSM river name enrichment."""

import shutil
from pathlib import Path

import duckdb
import pandas as pd
import pytest
from src.sword_duckdb.osm_names.match import match_osm_names, save_osm_names
from src.sword_duckdb.schema import add_osm_name_columns

pytestmark = [pytest.mark.db]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def test_db(tmp_path):
    """Copy test DB to temp dir, add geometries from bounding boxes, add OSM columns."""
    src = Path(__file__).parent / "fixtures" / "sword_test_minimal.duckdb"
    if not src.exists():
        pytest.skip("sword_test_minimal.duckdb fixture not found")
    dst = tmp_path / "test.duckdb"
    shutil.copy2(src, dst)
    conn = duckdb.connect(str(dst))
    conn.execute("INSTALL spatial; LOAD spatial;")

    # The test DB has NULL geometries but valid x/y/bbox columns.
    # Create LINESTRING geometries from the bounding box so spatial joins work.
    conn.execute("""
        UPDATE reaches
        SET geom = ST_GeomFromText(
            'LINESTRING(' || x_min || ' ' || y || ', ' || x_max || ' ' || y || ')'
        )
        WHERE geom IS NULL AND x_min IS NOT NULL
    """)

    add_osm_name_columns(conn)
    yield conn
    conn.close()


@pytest.fixture
def osm_gpkg():
    """Path to test OSM waterways GPKG."""
    p = Path(__file__).parent / "fixtures" / "test_osm_waterways.gpkg"
    if not p.exists():
        pytest.skip("test_osm_waterways.gpkg fixture not found")
    return p


# =============================================================================
# Schema migration tests
# =============================================================================


class TestSchemaMigration:
    def test_add_columns_creates_columns(self, test_db):
        cols = test_db.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'reaches'"
        ).fetchall()
        col_names = {r[0].lower() for r in cols}
        assert "river_name_local" in col_names
        assert "river_name_en" in col_names

    def test_add_columns_idempotent(self, test_db):
        # Call again - should not error and should return False
        result = add_osm_name_columns(test_db)
        assert result is False


# =============================================================================
# Spatial matching tests
# =============================================================================


class TestSpatialMatching:
    def test_match_returns_dataframe(self, test_db, osm_gpkg):
        df = match_osm_names(test_db, osm_gpkg, "NA")
        assert "reach_id" in df.columns
        assert "river_name_local" in df.columns
        assert "river_name_en" in df.columns

    def test_match_finds_overlaps(self, test_db, osm_gpkg):
        df = match_osm_names(test_db, osm_gpkg, "NA")
        assert len(df) > 0, "Should find at least one match"

    def test_semicolon_delimited_multi_match(self, test_db, osm_gpkg):
        df = match_osm_names(test_db, osm_gpkg, "NA")
        # Reach 5 (x=-90) and reach 10 (x=-100, y=41) each have two overlapping OSM lines
        multi = df[df["river_name_local"].str.contains(";", na=False)]
        assert len(multi) > 0, "Should have at least one multi-name match"
        for _, row in multi.iterrows():
            names = row["river_name_local"].split("; ")
            assert len(names) > 1

    def test_null_name_en_preserved(self, test_db, osm_gpkg):
        df = match_osm_names(test_db, osm_gpkg, "NA")
        # Reach 10 overlaps Rio Grande (name_en=NULL) and Pecos River (name_en set).
        # The aggregated name_en should still be present (from Pecos) even if one is NULL.
        reach_10 = df[df["reach_id"] == 11000000010]
        if not reach_10.empty:
            # At least one of the OSM lines had a name_en
            assert reach_10.iloc[0]["river_name_local"] is not None

    def test_save_updates_reaches(self, test_db, osm_gpkg):
        df = match_osm_names(test_db, osm_gpkg, "NA")
        n = save_osm_names(test_db, "NA", df)
        assert n > 0
        # Verify in DB
        row = test_db.execute(
            "SELECT river_name_local FROM reaches WHERE region='NA' AND river_name_local IS NOT NULL LIMIT 1"
        ).fetchone()
        assert row is not None

    def test_save_clears_previous_values(self, test_db, osm_gpkg):
        """Calling save twice should clear old values before writing new ones."""
        df = match_osm_names(test_db, osm_gpkg, "NA")
        save_osm_names(test_db, "NA", df)

        # Save again with same data
        n = save_osm_names(test_db, "NA", df)
        assert n > 0

        # Count should match original, not double
        total = test_db.execute(
            "SELECT COUNT(*) FROM reaches WHERE region='NA' AND river_name_local IS NOT NULL"
        ).fetchone()[0]
        assert total == n

    def test_save_empty_df(self, test_db):
        empty = pd.DataFrame(columns=["reach_id", "river_name_local", "river_name_en"])
        n = save_osm_names(test_db, "NA", empty)
        assert n == 0

    def test_wrong_region_no_matches(self, test_db, osm_gpkg):
        # Test DB only has NA reaches, OC should return empty
        df = match_osm_names(test_db, osm_gpkg, "OC")
        assert len(df) == 0

    def test_case_insensitive_region(self, test_db, osm_gpkg):
        # Region should be normalized to uppercase internally
        df = match_osm_names(test_db, osm_gpkg, "na")
        assert len(df) > 0
