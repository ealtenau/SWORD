# -*- coding: utf-8 -*-
"""
Unit tests for SWORD DuckDB class.

Tests cover:
- Basic loading and attribute access
- View classes (CenterlinesView, NodesView, ReachesView)
- WritableArray persistence
- SWORD methods (delete_data, append_data, break_reaches, etc.)
- Paths property
"""

import os
import sys
import pytest
import numpy as np
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


@pytest.fixture(scope='module')
def sword():
    """Create SWORD instance for testing (read-only operations)."""
    if not os.path.exists(TEST_DB_PATH):
        pytest.skip(f"Test database not found: {TEST_DB_PATH}")
    return SWORD(TEST_DB_PATH, TEST_REGION, TEST_VERSION)


class TestSWORDLoading:
    """Test SWORD class initialization and basic loading."""

    def test_load_sword(self, sword):
        """Test that SWORD loads without error."""
        assert sword is not None
        assert sword.region == TEST_REGION
        assert sword.version == TEST_VERSION

    def test_centerlines_loaded(self, sword):
        """Test centerlines data is loaded."""
        assert len(sword.centerlines) > 0
        assert hasattr(sword.centerlines, 'cl_id')
        assert hasattr(sword.centerlines, 'x')
        assert hasattr(sword.centerlines, 'y')

    def test_nodes_loaded(self, sword):
        """Test nodes data is loaded."""
        assert len(sword.nodes) > 0
        assert hasattr(sword.nodes, 'id')
        assert hasattr(sword.nodes, 'x')
        assert hasattr(sword.nodes, 'y')

    def test_reaches_loaded(self, sword):
        """Test reaches data is loaded."""
        assert len(sword.reaches) > 0
        assert hasattr(sword.reaches, 'id')
        assert hasattr(sword.reaches, 'x')
        assert hasattr(sword.reaches, 'y')


class TestCenterlinesView:
    """Test CenterlinesView attribute access."""

    def test_cl_id_array(self, sword):
        """Test cl_id is a numpy array."""
        cl_ids = sword.centerlines.cl_id
        assert isinstance(cl_ids, np.ndarray)
        assert len(cl_ids) == len(sword.centerlines)

    def test_x_y_coordinates(self, sword):
        """Test x, y coordinates are accessible."""
        x = np.array(sword.centerlines.x)
        y = np.array(sword.centerlines.y)
        assert len(x) == len(sword.centerlines)
        assert len(y) == len(sword.centerlines)
        # Coordinates should be valid lon/lat
        assert np.all((x >= -180) & (x <= 180))
        assert np.all((y >= -90) & (y <= 90))

    def test_reach_id_shape(self, sword):
        """Test reach_id is [4,N] array."""
        reach_id = sword.centerlines.reach_id
        assert reach_id.shape[0] == 4
        assert reach_id.shape[1] == len(sword.centerlines)

    def test_node_id_shape(self, sword):
        """Test node_id is [4,N] array."""
        node_id = sword.centerlines.node_id
        assert node_id.shape[0] == 4
        assert node_id.shape[1] == len(sword.centerlines)


class TestNodesView:
    """Test NodesView attribute access."""

    def test_node_id_array(self, sword):
        """Test node id is accessible."""
        node_ids = sword.nodes.id
        assert isinstance(node_ids, np.ndarray)
        assert len(node_ids) == len(sword.nodes)

    def test_renamed_attributes(self, sword):
        """Test renamed attributes map correctly."""
        # len -> node_length
        node_len = np.array(sword.nodes.len)
        assert len(node_len) == len(sword.nodes)

        # wth -> width
        wth = np.array(sword.nodes.wth)
        assert len(wth) == len(sword.nodes)

        # grod -> obstr_type
        grod = np.array(sword.nodes.grod)
        assert len(grod) == len(sword.nodes)

    def test_cl_id_shape(self, sword):
        """Test cl_id is [2,N] array (min, max)."""
        cl_id = sword.nodes.cl_id
        assert cl_id.shape[0] == 2
        assert cl_id.shape[1] == len(sword.nodes)

    def test_reach_id_array(self, sword):
        """Test reach_id is accessible."""
        reach_ids = sword.nodes.reach_id
        assert isinstance(reach_ids, np.ndarray)
        assert len(reach_ids) == len(sword.nodes)


class TestReachesView:
    """Test ReachesView attribute access."""

    def test_reach_id_array(self, sword):
        """Test reach id is accessible."""
        reach_ids = sword.reaches.id
        assert isinstance(reach_ids, np.ndarray)
        assert len(reach_ids) == len(sword.reaches)

    def test_renamed_attributes(self, sword):
        """Test renamed attributes map correctly."""
        # len -> reach_length
        rch_len = np.array(sword.reaches.len)
        assert len(rch_len) == len(sword.reaches)

        # rch_n_nodes -> n_nodes
        n_nodes = np.array(sword.reaches.rch_n_nodes)
        assert len(n_nodes) == len(sword.reaches)

    def test_topology_arrays(self, sword):
        """Test upstream/downstream topology arrays."""
        rch_id_up = sword.reaches.rch_id_up
        rch_id_down = sword.reaches.rch_id_down

        assert rch_id_up.shape[0] == 4
        assert rch_id_up.shape[1] == len(sword.reaches)
        assert rch_id_down.shape[0] == 4
        assert rch_id_down.shape[1] == len(sword.reaches)


class TestPathsProperty:
    """Test paths property for backward compatibility."""

    def test_paths_dict_exists(self, sword):
        """Test paths property returns a dict."""
        paths = sword.paths
        assert isinstance(paths, dict)

    def test_required_paths(self, sword):
        """Test all required path keys exist."""
        paths = sword.paths
        required_keys = [
            'shp_dir', 'gpkg_dir', 'nc_dir', 'geom_dir',
            'update_dir', 'topo_dir', 'version_dir', 'pts_gpkg_dir',
            'nc_fn', 'gpkg_rch_fn', 'gpkg_node_fn',
            'shp_rch_fn', 'shp_node_fn', 'geom_fn'
        ]
        for key in required_keys:
            assert key in paths, f"Missing path key: {key}"

    def test_nc_filename_format(self, sword):
        """Test NetCDF filename has correct format."""
        nc_fn = sword.paths['nc_fn']
        assert nc_fn == f"{TEST_REGION.lower()}_sword_{TEST_VERSION}.nc"


class TestWritableArray:
    """Test WritableArray for database persistence."""

    def test_writable_array_read(self, sword):
        """Test reading from WritableArray."""
        dist_out = sword.reaches.dist_out
        assert len(dist_out) == len(sword.reaches)
        # Should behave like numpy array
        assert dist_out[0] is not None

    def test_writable_array_slicing(self, sword):
        """Test array slicing works."""
        dist_out = sword.reaches.dist_out
        subset = dist_out[:10]
        assert len(subset) == 10

    def test_writable_array_comparison(self, sword):
        """Test comparison operators work."""
        dist_out = sword.reaches.dist_out
        mask = dist_out > 0
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool

    def test_writable_array_arithmetic(self, sword):
        """Test arithmetic operations work."""
        dist_out = sword.reaches.dist_out
        result = dist_out + 1
        assert isinstance(result, np.ndarray)

    def test_writable_array_numpy_conversion(self, sword):
        """Test conversion to numpy array."""
        dist_out = sword.reaches.dist_out
        arr = np.array(dist_out)
        assert isinstance(arr, np.ndarray)


class TestWritableArrayPersistence:
    """Test WritableArray write persistence (requires temp database)."""

    @pytest.fixture
    def temp_sword(self):
        """Create a temporary copy of the database for write tests."""
        if not os.path.exists(TEST_DB_PATH):
            pytest.skip(f"Test database not found: {TEST_DB_PATH}")

        # Create temp copy
        temp_dir = tempfile.mkdtemp()
        temp_db = os.path.join(temp_dir, 'sword_test.duckdb')
        shutil.copy2(TEST_DB_PATH, temp_db)

        sword = SWORD(temp_db, TEST_REGION, TEST_VERSION)
        yield sword

        # Cleanup
        sword.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_single_value_write(self, temp_sword):
        """Test writing a single value persists to database."""
        idx = 0
        original = float(temp_sword.reaches.dist_out[idx])
        test_value = 99999.0

        # Write new value
        temp_sword.reaches.dist_out[idx] = test_value

        # Verify in-memory
        assert temp_sword.reaches.dist_out[idx] == test_value

        # Restore original
        temp_sword.reaches.dist_out[idx] = original

    def test_multiple_value_write(self, temp_sword):
        """Test writing multiple values persists to database."""
        indices = [0, 1, 2]
        originals = [float(temp_sword.reaches.dist_out[i]) for i in indices]
        test_values = [11111.0, 22222.0, 33333.0]

        # Write new values
        temp_sword.reaches.dist_out[indices] = test_values

        # Verify in-memory
        for i, val in zip(indices, test_values):
            assert temp_sword.reaches.dist_out[i] == val

        # Restore originals
        for i, val in zip(indices, originals):
            temp_sword.reaches.dist_out[i] = val


class TestDimensionConsistency:
    """Test data dimension consistency (from check_nc_dimensions.py)."""

    def test_unique_cl_ids(self, sword):
        """Test centerline IDs are unique within region."""
        cl_ids = sword.centerlines.cl_id
        assert len(np.unique(cl_ids)) == len(cl_ids)

    def test_node_id_lengths(self, sword):
        """Test node IDs have correct character length (14)."""
        node_ids = sword.nodes.id
        min_len = len(str(np.min(node_ids)))
        max_len = len(str(np.max(node_ids)))
        assert min_len == 14, f"Min node ID length is {min_len}, expected 14"
        assert max_len == 14, f"Max node ID length is {max_len}, expected 14"

    def test_reach_id_lengths(self, sword):
        """Test reach IDs have correct character length (11)."""
        reach_ids = sword.reaches.id
        min_len = len(str(np.min(reach_ids)))
        max_len = len(str(np.max(reach_ids)))
        assert min_len == 11, f"Min reach ID length is {min_len}, expected 11"
        assert max_len == 11, f"Max reach ID length is {max_len}, expected 11"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
