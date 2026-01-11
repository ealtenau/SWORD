# -*- coding: utf-8 -*-
"""
Unit tests for SWORD reactive update system.

Tests cover:
- Dependency graph construction
- Topological sort
- Dirty tracking
- Recalculation functions
"""

import os
import sys
import pytest
import numpy as np

# Add project root to path
main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, main_dir)


class TestDependencyGraph:
    """Test dependency graph construction."""

    def test_graph_construction(self):
        """Test that dependency graph builds correctly."""
        from src.updates.sword_duckdb.reactive import DependencyGraph

        graph = DependencyGraph()

        # Verify key nodes exist
        assert 'centerline.geometry' in graph.nodes
        assert 'reach.len' in graph.nodes
        assert 'reach.dist_out' in graph.nodes
        assert 'node.dist_out' in graph.nodes
        assert 'reach.end_rch' in graph.nodes
        assert 'node.end_rch' in graph.nodes

    def test_reach_len_depends_on_geometry(self):
        """Test that reach.len depends on centerline.geometry."""
        from src.updates.sword_duckdb.reactive import DependencyGraph

        graph = DependencyGraph()
        reach_len_node = graph.nodes['reach.len']

        assert 'centerline.geometry' in reach_len_node.depends_on

    def test_node_dist_out_depends_on_reach_dist_out(self):
        """Test that node.dist_out depends on reach.dist_out."""
        from src.updates.sword_duckdb.reactive import DependencyGraph

        graph = DependencyGraph()
        node_dist_out = graph.nodes['node.dist_out']

        assert 'reach.dist_out' in node_dist_out.depends_on

    def test_get_downstream_deps(self):
        """Test getting downstream dependencies."""
        from src.updates.sword_duckdb.reactive import DependencyGraph

        graph = DependencyGraph()

        # centerline.geometry should trigger many downstream updates
        deps = graph.get_downstream_deps('centerline.geometry')

        assert 'reach.len' in deps
        assert 'reach.bounds' in deps
        assert 'node.len' in deps
        assert 'node.xy' in deps

    def test_topological_sort_order(self):
        """Test that topological sort produces correct order."""
        from src.updates.sword_duckdb.reactive import DependencyGraph

        graph = DependencyGraph()

        # Mark geometry as dirty
        dirty_attrs = {'centerline.geometry'}
        sorted_attrs = graph.topological_sort(dirty_attrs)

        # reach.len should come before reach.dist_out
        if 'reach.len' in sorted_attrs and 'reach.dist_out' in sorted_attrs:
            len_idx = sorted_attrs.index('reach.len')
            dist_out_idx = sorted_attrs.index('reach.dist_out')
            assert len_idx < dist_out_idx, "reach.len should be recalculated before reach.dist_out"


class TestDirtySet:
    """Test dirty set tracking."""

    def test_dirty_set_creation(self):
        """Test creating a dirty set."""
        from src.updates.sword_duckdb.reactive import DirtySet

        dirty = DirtySet()

        assert len(dirty.reach_ids) == 0
        assert len(dirty.node_ids) == 0
        assert dirty.all_reaches is False

    def test_dirty_set_add_reach_ids(self):
        """Test adding reach IDs to dirty set."""
        from src.updates.sword_duckdb.reactive import DirtySet

        dirty = DirtySet()
        dirty.reach_ids.update([123, 456, 789])

        assert 123 in dirty.reach_ids
        assert 456 in dirty.reach_ids
        assert 789 in dirty.reach_ids


class TestSWORDReactive:
    """Test SWORDReactive class."""

    def test_mark_dirty(self):
        """Test marking attributes as dirty."""
        # This test requires a SWORD instance, so we skip if database not available
        from src.updates.sword_duckdb import SWORD

        db_path = os.path.join(main_dir, 'data/duckdb/sword_v17b.duckdb')
        if not os.path.exists(db_path):
            pytest.skip(f"Test database not found: {db_path}")

        from src.updates.sword_duckdb.reactive import SWORDReactive, ChangeType

        sword = SWORD(db_path, 'NA', 'v17b')
        reactive = SWORDReactive(sword)

        # Mark geometry as changed
        reactive.mark_dirty(
            'centerline.geometry',
            ChangeType.GEOMETRY,
            reach_ids=[sword.reaches.id[0]]
        )

        assert 'centerline.geometry' in reactive._dirty_attrs
        assert sword.reaches.id[0] in reactive.dirty.reach_ids

        sword.close()

    def test_get_recalc_plan(self):
        """Test getting recalculation plan."""
        from src.updates.sword_duckdb import SWORD

        db_path = os.path.join(main_dir, 'data/duckdb/sword_v17b.duckdb')
        if not os.path.exists(db_path):
            pytest.skip(f"Test database not found: {db_path}")

        from src.updates.sword_duckdb.reactive import SWORDReactive, ChangeType

        sword = SWORD(db_path, 'NA', 'v17b')
        reactive = SWORDReactive(sword)

        # Mark geometry as changed
        reactive.mark_dirty('centerline.geometry', ChangeType.GEOMETRY, all_entities=True)

        plan = reactive.get_recalc_plan()

        # Plan should include functions to recalculate
        assert len(plan) > 0
        attr_names = [attr for attr, _ in plan]

        # reach.len should be in the plan (depends on geometry)
        assert 'reach.len' in attr_names

        sword.close()

    def test_dry_run_recalculate(self):
        """Test dry run of recalculation."""
        from src.updates.sword_duckdb import SWORD

        db_path = os.path.join(main_dir, 'data/duckdb/sword_v17b.duckdb')
        if not os.path.exists(db_path):
            pytest.skip(f"Test database not found: {db_path}")

        from src.updates.sword_duckdb.reactive import SWORDReactive, ChangeType

        sword = SWORD(db_path, 'NA', 'v17b')
        reactive = SWORDReactive(sword)

        reactive.mark_dirty('reach.topology', ChangeType.TOPOLOGY, all_entities=True)

        # Dry run should not actually change anything
        recalculated = reactive.recalculate(dry_run=True)

        assert len(recalculated) > 0
        assert 'reach.end_rch' in recalculated

        sword.close()


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_mark_geometry_changed(self):
        """Test mark_geometry_changed function."""
        from src.updates.sword_duckdb import SWORD

        db_path = os.path.join(main_dir, 'data/duckdb/sword_v17b.duckdb')
        if not os.path.exists(db_path):
            pytest.skip(f"Test database not found: {db_path}")

        from src.updates.sword_duckdb.reactive import (
            SWORDReactive,
            mark_geometry_changed,
        )

        sword = SWORD(db_path, 'NA', 'v17b')
        reactive = SWORDReactive(sword)

        mark_geometry_changed(reactive, reach_ids=[sword.reaches.id[0]])

        assert 'centerline.geometry' in reactive._dirty_attrs

        sword.close()

    def test_mark_topology_changed(self):
        """Test mark_topology_changed function."""
        from src.updates.sword_duckdb import SWORD

        db_path = os.path.join(main_dir, 'data/duckdb/sword_v17b.duckdb')
        if not os.path.exists(db_path):
            pytest.skip(f"Test database not found: {db_path}")

        from src.updates.sword_duckdb.reactive import (
            SWORDReactive,
            mark_topology_changed,
        )

        sword = SWORD(db_path, 'NA', 'v17b')
        reactive = SWORDReactive(sword)

        mark_topology_changed(reactive, all_reaches=True)

        assert 'reach.topology' in reactive._dirty_attrs
        assert reactive.dirty.all_reaches is True

        sword.close()


class TestGeodesicDistance:
    """Test geodesic distance calculation."""

    def test_get_geodesic_distances(self):
        """Test geodesic distance calculation."""
        try:
            from geopy import distance  # noqa: F401
        except ImportError:
            pytest.skip("geopy not installed")

        from src.updates.sword_duckdb.reactive import _get_geodesic_distances

        # Test with simple coordinates (roughly 1 degree = 111km at equator)
        lon = np.array([0.0, 1.0, 2.0])
        lat = np.array([0.0, 0.0, 0.0])

        distances = _get_geodesic_distances(lon, lat)

        assert len(distances) == 3
        assert distances[0] == 0.0  # First point has no distance
        assert distances[1] > 100000  # About 111km
        assert distances[2] > 100000  # About 111km

    def test_single_point(self):
        """Test with single point."""
        try:
            from geopy import distance  # noqa: F401
        except ImportError:
            pytest.skip("geopy not installed")

        from src.updates.sword_duckdb.reactive import _get_geodesic_distances

        lon = np.array([0.0])
        lat = np.array([0.0])

        distances = _get_geodesic_distances(lon, lat)

        assert len(distances) == 1
        assert distances[0] == 0.0


class TestChangeType:
    """Test ChangeType enum."""

    def test_change_types(self):
        """Test that all change types exist."""
        from src.updates.sword_duckdb.reactive import ChangeType

        assert hasattr(ChangeType, 'GEOMETRY')
        assert hasattr(ChangeType, 'TOPOLOGY')
        assert hasattr(ChangeType, 'ATTRIBUTE')
        assert hasattr(ChangeType, 'STRUCTURE')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
