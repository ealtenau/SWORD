"""Tests for spatial utility functions.

Ported from src/_legacy/updates/geo_utils.py.
"""

import shutil

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class TestMetersToDegrees:
    def test_equator(self):
        """At equator, 111320m ≈ 1 degree."""
        from src.sword_duckdb.spatial_utils import meters_to_degrees

        result = meters_to_degrees(111320, 0.0)
        assert abs(result - 1.0) < 0.01

    def test_high_latitude(self):
        """At 60°N, 1 degree longitude ≈ 55660m."""
        from src.sword_duckdb.spatial_utils import meters_to_degrees

        result = meters_to_degrees(55660, 60.0)
        assert abs(result - 1.0) < 0.01

    def test_zero_meters(self):
        from src.sword_duckdb.spatial_utils import meters_to_degrees

        assert meters_to_degrees(0, 45.0) == 0.0


class TestReprojectUtm:
    def test_single_point(self):
        """Known point: Washington DC ≈ UTM zone 18N."""
        from src.sword_duckdb.spatial_utils import reproject_utm

        lats = np.array([38.9072])
        lons = np.array([-77.0369])
        east, north, zone_num, zone_let = reproject_utm(lats, lons)
        assert zone_num == 18
        assert east[0] > 300000
        assert north[0] > 4000000

    def test_multiple_points_same_zone(self):
        from src.sword_duckdb.spatial_utils import reproject_utm

        lats = np.array([38.9, 39.0, 38.8])
        lons = np.array([-77.0, -77.1, -76.9])
        east, north, zone_num, zone_let = reproject_utm(lats, lons)
        assert len(east) == 3
        assert len(north) == 3


class TestBfsTraversal:
    """BFS tests need the test DB for reach_topology table."""

    @pytest.fixture
    def conn(self, ensure_test_db, tmp_path):
        import duckdb

        temp_db = tmp_path / "bfs_test.duckdb"
        shutil.copy2(ensure_test_db, temp_db)
        con = duckdb.connect(str(temp_db))
        yield con
        con.close()

    @pytest.mark.db
    def test_get_all_upstream_returns_set(self, conn):
        from src.sword_duckdb.spatial_utils import get_all_upstream

        row = conn.execute(
            "SELECT reach_id FROM reaches WHERE end_reach = 2 AND region = 'NA' LIMIT 1"
        ).fetchone()
        if row is None:
            pytest.skip("No outlet reach in test DB")
        result = get_all_upstream(conn, row[0], region="NA")
        assert isinstance(result, set)

    @pytest.mark.db
    def test_get_all_downstream_returns_set(self, conn):
        from src.sword_duckdb.spatial_utils import get_all_downstream

        row = conn.execute(
            "SELECT reach_id FROM reaches WHERE end_reach = 1 AND region = 'NA' LIMIT 1"
        ).fetchone()
        if row is None:
            pytest.skip("No headwater reach in test DB")
        result = get_all_downstream(conn, row[0], region="NA")
        assert isinstance(result, set)

    @pytest.mark.db
    def test_upstream_of_headwater_is_empty(self, conn):
        from src.sword_duckdb.spatial_utils import get_all_upstream

        row = conn.execute(
            "SELECT reach_id FROM reaches "
            "WHERE end_reach = 1 AND n_rch_up = 0 AND region = 'NA' LIMIT 1"
        ).fetchone()
        if row is None:
            pytest.skip("No isolated headwater in test DB")
        result = get_all_upstream(conn, row[0], region="NA")
        assert len(result) == 0

    @pytest.mark.db
    def test_downstream_of_outlet_is_empty(self, conn):
        from src.sword_duckdb.spatial_utils import get_all_downstream

        row = conn.execute(
            "SELECT reach_id FROM reaches "
            "WHERE end_reach = 2 AND n_rch_down = 0 AND region = 'NA' LIMIT 1"
        ).fetchone()
        if row is None:
            pytest.skip("No isolated outlet in test DB")
        result = get_all_downstream(conn, row[0], region="NA")
        assert len(result) == 0
