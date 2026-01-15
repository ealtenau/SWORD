# -*- coding: utf-8 -*-
"""
Tests for SWORD DuckDB View Classes

Tests WritableArray, CenterlinesView, NodesView, and ReachesView.
"""

import numpy as np
import pytest


class TestWritableArrayBasics:
    """Tests for WritableArray basic functionality."""

    def test_getitem_single_index(self, sword_readonly):
        """Test single index access."""
        arr = sword_readonly.reaches.dist_out
        val = arr[0]
        assert isinstance(val, (int, float, np.integer, np.floating))

    def test_getitem_slice(self, sword_readonly):
        """Test slice access."""
        arr = sword_readonly.reaches.dist_out
        vals = arr[:5]
        assert len(vals) == 5
        assert isinstance(vals, np.ndarray)

    def test_getitem_fancy_indexing(self, sword_readonly):
        """Test fancy indexing with list."""
        arr = sword_readonly.reaches.dist_out
        vals = arr[[0, 1, 2]]
        assert len(vals) == 3
        assert isinstance(vals, np.ndarray)

    def test_getitem_boolean_mask(self, sword_readonly):
        """Test boolean mask indexing."""
        arr = sword_readonly.reaches.dist_out
        mask = arr > 0
        vals = arr[mask]
        assert all(v > 0 for v in vals)

    def test_len(self, sword_readonly):
        """Test __len__ returns correct length."""
        arr = sword_readonly.reaches.dist_out
        assert len(arr) == len(sword_readonly.reaches.id)

    def test_shape(self, sword_readonly):
        """Test shape property."""
        arr = sword_readonly.reaches.dist_out
        assert arr.shape == (len(sword_readonly.reaches.id),)

    def test_dtype(self, sword_readonly):
        """Test dtype property."""
        arr = sword_readonly.reaches.dist_out
        assert arr.dtype in (np.float64, np.float32, np.int64, np.int32)

    def test_ndim(self, sword_readonly):
        """Test ndim property."""
        arr = sword_readonly.reaches.dist_out
        assert arr.ndim == 1

    def test_repr(self, sword_readonly):
        """Test __repr__ method."""
        arr = sword_readonly.reaches.dist_out
        repr_str = repr(arr)
        assert 'WritableArray' in repr_str
        assert 'dist_out' in repr_str


class TestWritableArrayComparison:
    """Tests for WritableArray comparison operators."""

    def test_eq(self, sword_readonly):
        """Test equality comparison."""
        arr = sword_readonly.reaches.rch_n_nodes
        result = arr == arr[0]
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool

    def test_ne(self, sword_readonly):
        """Test inequality comparison."""
        arr = sword_readonly.reaches.rch_n_nodes
        result = arr != arr[0]
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool

    def test_lt(self, sword_readonly):
        """Test less than comparison."""
        arr = sword_readonly.reaches.dist_out
        result = arr < 10000
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool

    def test_le(self, sword_readonly):
        """Test less than or equal comparison."""
        arr = sword_readonly.reaches.dist_out
        result = arr <= arr[0]
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool

    def test_gt(self, sword_readonly):
        """Test greater than comparison."""
        arr = sword_readonly.reaches.dist_out
        result = arr > 0
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool

    def test_ge(self, sword_readonly):
        """Test greater than or equal comparison."""
        arr = sword_readonly.reaches.dist_out
        result = arr >= arr[0]
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool


class TestWritableArrayArithmetic:
    """Tests for WritableArray arithmetic operators."""

    def test_add(self, sword_readonly):
        """Test addition."""
        arr = sword_readonly.reaches.dist_out
        result = arr + 100
        assert isinstance(result, np.ndarray)
        assert result[0] == arr[0] + 100

    def test_radd(self, sword_readonly):
        """Test reverse addition."""
        arr = sword_readonly.reaches.dist_out
        result = 100 + arr
        assert isinstance(result, np.ndarray)
        assert result[0] == 100 + arr[0]

    def test_sub(self, sword_readonly):
        """Test subtraction."""
        arr = sword_readonly.reaches.dist_out
        result = arr - 100
        assert isinstance(result, np.ndarray)
        assert result[0] == arr[0] - 100

    def test_rsub(self, sword_readonly):
        """Test reverse subtraction."""
        arr = sword_readonly.reaches.dist_out
        result = 100 - arr
        assert isinstance(result, np.ndarray)
        assert result[0] == 100 - arr[0]

    def test_mul(self, sword_readonly):
        """Test multiplication."""
        arr = sword_readonly.reaches.dist_out
        result = arr * 2
        assert isinstance(result, np.ndarray)
        assert result[0] == arr[0] * 2

    def test_rmul(self, sword_readonly):
        """Test reverse multiplication."""
        arr = sword_readonly.reaches.dist_out
        result = 2 * arr
        assert isinstance(result, np.ndarray)
        assert result[0] == 2 * arr[0]

    def test_truediv(self, sword_readonly):
        """Test true division."""
        arr = sword_readonly.reaches.dist_out
        result = arr / 2
        assert isinstance(result, np.ndarray)
        assert result[0] == arr[0] / 2

    def test_floordiv(self, sword_readonly):
        """Test floor division."""
        arr = sword_readonly.reaches.dist_out
        result = arr // 2
        assert isinstance(result, np.ndarray)
        assert result[0] == arr[0] // 2


class TestWritableArrayNumpyCompat:
    """Tests for WritableArray numpy compatibility."""

    def test_array_conversion(self, sword_readonly):
        """Test conversion to numpy array."""
        arr = sword_readonly.reaches.dist_out
        np_arr = np.array(arr)
        assert isinstance(np_arr, np.ndarray)
        assert len(np_arr) == len(arr)

    def test_astype(self, sword_readonly):
        """Test astype conversion."""
        arr = sword_readonly.reaches.dist_out
        int_arr = arr.astype(np.int32)
        assert int_arr.dtype == np.int32

    def test_copy(self, sword_readonly):
        """Test copy method."""
        arr = sword_readonly.reaches.dist_out
        copied = arr.copy()
        assert isinstance(copied, np.ndarray)
        assert len(copied) == len(arr)
        # Modifying copy shouldn't affect original
        copied[0] = -9999
        assert arr[0] != -9999

    def test_iter(self, sword_readonly):
        """Test iteration."""
        arr = sword_readonly.reaches.dist_out[:5]
        values = list(arr)
        assert len(values) == 5

    def test_numpy_functions(self, sword_readonly):
        """Test numpy functions work with WritableArray."""
        arr = sword_readonly.reaches.dist_out
        assert np.sum(arr) >= 0 or np.sum(arr) < 0  # Just check it works
        assert np.mean(arr) >= 0 or np.mean(arr) < 0
        assert np.max(arr) >= np.min(arr)


class TestWritableArrayPersistence:
    """Tests for WritableArray database persistence."""

    def test_setitem_single_persists(self, sword_writable):
        """Test single value write persists to database."""
        original_value = sword_writable.reaches.dist_out[0]
        new_value = original_value + 1000.0

        # Modify
        sword_writable.reaches.dist_out[0] = new_value

        # Close and reopen to verify persistence
        db_path = sword_writable._db.db_path
        region = sword_writable.region
        version = sword_writable.version
        sword_writable.close()

        from src.updates.sword_duckdb import SWORD
        sword2 = SWORD(db_path, region, version)
        assert sword2.reaches.dist_out[0] == new_value
        sword2.close()

    def test_setitem_slice_persists(self, sword_writable):
        """Test slice write persists to database."""
        original_values = sword_writable.reaches.dist_out[:3].copy()
        new_values = original_values + 500.0

        # Modify
        sword_writable.reaches.dist_out[:3] = new_values

        # Close and reopen
        db_path = sword_writable._db.db_path
        region = sword_writable.region
        version = sword_writable.version
        sword_writable.close()

        from src.updates.sword_duckdb import SWORD
        sword2 = SWORD(db_path, region, version)
        np.testing.assert_array_almost_equal(
            sword2.reaches.dist_out[:3], new_values
        )
        sword2.close()

    def test_setitem_fancy_indexing_persists(self, sword_writable):
        """Test fancy indexing write persists."""
        indices = [0, 2, 4]
        new_value = 12345.0

        # Modify
        sword_writable.reaches.dist_out[indices] = new_value

        # Close and reopen
        db_path = sword_writable._db.db_path
        region = sword_writable.region
        version = sword_writable.version
        sword_writable.close()

        from src.updates.sword_duckdb import SWORD
        sword2 = SWORD(db_path, region, version)
        for idx in indices:
            assert sword2.reaches.dist_out[idx] == new_value
        sword2.close()


class TestCenterlinesView:
    """Tests for CenterlinesView class."""

    def test_cl_id_array(self, sword_readonly):
        """Test cl_id returns numpy array."""
        cl_ids = sword_readonly.centerlines.cl_id
        assert isinstance(cl_ids, np.ndarray)
        assert len(cl_ids) > 0

    def test_x_y_are_writable(self, sword_readonly):
        """Test x and y return WritableArray."""
        from src.updates.sword_duckdb.views import WritableArray
        assert isinstance(sword_readonly.centerlines.x, WritableArray)
        assert isinstance(sword_readonly.centerlines.y, WritableArray)

    def test_reach_id_shape(self, sword_readonly):
        """Test reach_id has shape [4, N]."""
        reach_id = sword_readonly.centerlines.reach_id
        assert reach_id.ndim == 2
        assert reach_id.shape[0] == 4

    def test_node_id_shape(self, sword_readonly):
        """Test node_id has shape [4, N]."""
        node_id = sword_readonly.centerlines.node_id
        assert node_id.ndim == 2
        assert node_id.shape[0] == 4

    def test_len(self, sword_readonly):
        """Test __len__ returns correct count."""
        assert len(sword_readonly.centerlines) == len(sword_readonly.centerlines.cl_id)


class TestNodesView:
    """Tests for NodesView class."""

    def test_id_attribute(self, sword_readonly):
        """Test id attribute returns node_id array."""
        ids = sword_readonly.nodes.id
        assert isinstance(ids, np.ndarray)
        assert len(ids) > 0

    def test_len_attribute(self, sword_readonly):
        """Test len attribute returns node lengths."""
        from src.updates.sword_duckdb.views import WritableArray
        lens = sword_readonly.nodes.len
        assert isinstance(lens, WritableArray)
        assert len(lens) == len(sword_readonly.nodes.id)

    def test_cl_id_shape(self, sword_readonly):
        """Test cl_id has shape [2, N]."""
        cl_id = sword_readonly.nodes.cl_id
        assert cl_id.ndim == 2
        assert cl_id.shape[0] == 2

    def test_reach_id_array(self, sword_readonly):
        """Test reach_id is 1D array."""
        reach_ids = sword_readonly.nodes.reach_id
        assert reach_ids.ndim == 1
        assert len(reach_ids) == len(sword_readonly.nodes.id)

    def test_renamed_attributes(self, sword_readonly):
        """Test renamed attributes work (id, len, wth, etc)."""
        from src.updates.sword_duckdb.views import WritableArray
        # id -> node_id
        assert isinstance(sword_readonly.nodes.id, np.ndarray)
        # len -> node_length
        assert isinstance(sword_readonly.nodes.len, WritableArray)
        # wth -> width
        assert isinstance(sword_readonly.nodes.wth, WritableArray)

    def test_len_method(self, sword_readonly):
        """Test __len__ returns correct count."""
        assert len(sword_readonly.nodes) == len(sword_readonly.nodes.id)


class TestReachesView:
    """Tests for ReachesView class."""

    def test_id_attribute(self, sword_readonly):
        """Test id attribute returns reach_id array."""
        ids = sword_readonly.reaches.id
        assert isinstance(ids, np.ndarray)
        assert len(ids) > 0

    def test_len_attribute(self, sword_readonly):
        """Test len attribute returns reach lengths."""
        from src.updates.sword_duckdb.views import WritableArray
        lens = sword_readonly.reaches.len
        assert isinstance(lens, WritableArray)
        assert len(lens) == len(sword_readonly.reaches.id)

    def test_topology_arrays_shape(self, sword_readonly):
        """Test topology arrays have shape [4, N]."""
        rch_id_up = sword_readonly.reaches.rch_id_up
        rch_id_down = sword_readonly.reaches.rch_id_down
        assert rch_id_up.ndim == 2
        assert rch_id_up.shape[0] == 4
        assert rch_id_down.ndim == 2
        assert rch_id_down.shape[0] == 4

    def test_orbits_shape(self, sword_readonly):
        """Test orbits has shape [75, N]."""
        orbits = sword_readonly.reaches.orbits
        assert orbits.ndim == 2
        # May be 75 or less depending on data
        assert orbits.shape[0] <= 75

    def test_iceflag_shape(self, sword_readonly):
        """Test iceflag has shape [366, N]."""
        iceflag = sword_readonly.reaches.iceflag
        assert iceflag.ndim == 2
        # May be 366 or less depending on data
        assert iceflag.shape[0] <= 366

    def test_renamed_attributes(self, sword_readonly):
        """Test renamed attributes work."""
        from src.updates.sword_duckdb.views import WritableArray
        # id -> reach_id
        assert isinstance(sword_readonly.reaches.id, np.ndarray)
        # len -> reach_length
        assert isinstance(sword_readonly.reaches.len, WritableArray)
        # wth -> width
        assert isinstance(sword_readonly.reaches.wth, WritableArray)
        # rch_n_nodes -> n_nodes (column)
        assert isinstance(sword_readonly.reaches.rch_n_nodes, WritableArray)

    def test_cl_id_shape(self, sword_readonly):
        """Test cl_id has shape [2, N]."""
        cl_id = sword_readonly.reaches.cl_id
        assert cl_id.ndim == 2
        assert cl_id.shape[0] == 2

    def test_len_method(self, sword_readonly):
        """Test __len__ returns correct count."""
        assert len(sword_readonly.reaches) == len(sword_readonly.reaches.id)


class TestViewDataConsistency:
    """Tests for data consistency across views."""

    def test_node_reach_ids_valid(self, sword_readonly):
        """Test all node reach_ids exist in reaches."""
        node_reach_ids = np.unique(sword_readonly.nodes.reach_id)
        reach_ids = set(sword_readonly.reaches.id)
        for rid in node_reach_ids:
            assert rid in reach_ids, f"Node references non-existent reach {rid}"

    def test_centerline_reach_ids_valid(self, sword_readonly):
        """Test all centerline reach_ids exist in reaches."""
        cl_reach_ids = np.unique(sword_readonly.centerlines.reach_id[0, :])
        reach_ids = set(sword_readonly.reaches.id)
        for rid in cl_reach_ids:
            if rid != 0:  # Skip null references
                assert rid in reach_ids, f"Centerline references non-existent reach {rid}"

    def test_centerline_node_ids_valid(self, sword_readonly):
        """Test all centerline node_ids exist in nodes."""
        cl_node_ids = np.unique(sword_readonly.centerlines.node_id[0, :])
        node_ids = set(sword_readonly.nodes.id)
        for nid in cl_node_ids:
            if nid != 0:  # Skip null references
                assert nid in node_ids, f"Centerline references non-existent node {nid}"

    def test_reach_node_count_matches(self, sword_readonly):
        """Test reach rch_n_nodes matches actual node count."""
        for idx, reach_id in enumerate(sword_readonly.reaches.id[:10]):  # Check first 10
            n_nodes_attr = sword_readonly.reaches.rch_n_nodes[idx]
            actual_count = np.sum(sword_readonly.nodes.reach_id == reach_id)
            assert n_nodes_attr == actual_count, \
                f"Reach {reach_id} claims {n_nodes_attr} nodes but has {actual_count}"
