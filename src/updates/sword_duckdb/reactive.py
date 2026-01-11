# -*- coding: utf-8 -*-
"""
SWORD Reactive Update System
============================

Automatic recalculation of derived attributes when upstream values change.
Implements a dependency graph with topological ordering for cascading updates.

Architecture:
    1. Dependency Graph - Defines which attributes depend on others
    2. Change Tracker - Records which values have changed
    3. Recalculation Engine - Executes updates in dependency order

Example Usage:
    from sword_duckdb.reactive import SWORDReactive

    sword = SWORD('data/duckdb/sword_v17b.duckdb', 'NA', 'v17b')
    reactive = SWORDReactive(sword)

    # Edit geometry
    sword.centerlines.x[idx] = new_x
    sword.centerlines.y[idx] = new_y

    # Trigger recalculations
    reactive.mark_dirty('centerline.geometry', reach_ids=[12345678901])
    reactive.recalculate()  # Cascades to reach.len, node.len, dist_out, etc.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set, Tuple
import numpy as np

if TYPE_CHECKING:
    from .sword_class import SWORD


class ChangeType(Enum):
    """Types of changes that can trigger recalculations."""
    GEOMETRY = auto()      # Centerline x,y coordinates changed
    TOPOLOGY = auto()      # Reach connectivity (up/down) changed
    ATTRIBUTE = auto()     # Non-geometric attribute changed
    STRUCTURE = auto()     # Reaches/nodes added, deleted, or split


@dataclass
class DirtySet:
    """Tracks which entities need recalculation."""
    reach_ids: Set[int] = field(default_factory=set)
    node_ids: Set[int] = field(default_factory=set)
    cl_ids: Set[int] = field(default_factory=set)
    all_reaches: bool = False
    all_nodes: bool = False
    all_centerlines: bool = False


@dataclass
class DependencyNode:
    """A node in the dependency graph."""
    name: str
    table: str  # reaches, nodes, centerlines
    depends_on: List[str] = field(default_factory=list)
    recalc_func: Optional[str] = None  # Method name to call for recalc
    change_types: List[ChangeType] = field(default_factory=list)


class DependencyGraph:
    """
    Defines attribute dependencies and recalculation order.

    Based on analysis of SWORD codebase:
    - Geometry changes → lengths → dist_out → end_rch/main_side
    - Topology changes → counts → dist_out → end_rch/main_side
    """

    def __init__(self):
        self.nodes: Dict[str, DependencyNode] = {}
        self._build_graph()

    def _build_graph(self):
        """Build the dependency graph based on SWORD attribute relationships."""

        # === CENTERLINE ATTRIBUTES ===
        self.add_node(DependencyNode(
            name='centerline.geometry',
            table='centerlines',
            depends_on=[],  # Root - manual input
            change_types=[ChangeType.GEOMETRY]
        ))

        # === REACH ATTRIBUTES ===
        self.add_node(DependencyNode(
            name='reach.len',
            table='reaches',
            depends_on=['centerline.geometry'],
            recalc_func='_recalc_reach_lengths',
            change_types=[ChangeType.GEOMETRY]
        ))

        self.add_node(DependencyNode(
            name='reach.bounds',  # x, y, x_min, x_max, y_min, y_max
            table='reaches',
            depends_on=['centerline.geometry'],
            recalc_func='_recalc_reach_bounds',
            change_types=[ChangeType.GEOMETRY]
        ))

        self.add_node(DependencyNode(
            name='reach.topology',  # rch_id_up, rch_id_down, n_rch_up, n_rch_down
            table='reaches',
            depends_on=[],  # Root - manual input
            change_types=[ChangeType.TOPOLOGY]
        ))

        self.add_node(DependencyNode(
            name='reach.dist_out',
            table='reaches',
            depends_on=['reach.len', 'reach.topology'],
            recalc_func='_recalc_reach_dist_out',
            change_types=[ChangeType.GEOMETRY, ChangeType.TOPOLOGY]
        ))

        self.add_node(DependencyNode(
            name='reach.end_rch',
            table='reaches',
            depends_on=['reach.topology'],
            recalc_func='_recalc_reach_end_rch',
            change_types=[ChangeType.TOPOLOGY]
        ))

        self.add_node(DependencyNode(
            name='reach.main_side',
            table='reaches',
            depends_on=['reach.topology', 'reach.dist_out'],
            recalc_func='_recalc_reach_main_side',
            change_types=[ChangeType.TOPOLOGY]
        ))

        # === NODE ATTRIBUTES ===
        self.add_node(DependencyNode(
            name='node.len',
            table='nodes',
            depends_on=['centerline.geometry'],
            recalc_func='_recalc_node_lengths',
            change_types=[ChangeType.GEOMETRY]
        ))

        self.add_node(DependencyNode(
            name='node.xy',  # x, y coordinates
            table='nodes',
            depends_on=['centerline.geometry'],
            recalc_func='_recalc_node_xy',
            change_types=[ChangeType.GEOMETRY]
        ))

        self.add_node(DependencyNode(
            name='node.dist_out',
            table='nodes',
            depends_on=['reach.dist_out', 'node.len'],
            recalc_func='_recalc_node_dist_out',
            change_types=[ChangeType.GEOMETRY, ChangeType.TOPOLOGY]
        ))

        self.add_node(DependencyNode(
            name='node.end_rch',
            table='nodes',
            depends_on=['reach.end_rch'],
            recalc_func='_recalc_node_end_rch',
            change_types=[ChangeType.TOPOLOGY]
        ))

        self.add_node(DependencyNode(
            name='node.main_side',
            table='nodes',
            depends_on=['reach.main_side'],
            recalc_func='_recalc_node_main_side',
            change_types=[ChangeType.TOPOLOGY]
        ))

        self.add_node(DependencyNode(
            name='node.cl_id_bounds',  # cl_id[0] (min), cl_id[1] (max)
            table='nodes',
            depends_on=['centerline.geometry'],
            recalc_func='_recalc_node_cl_id_bounds',
            change_types=[ChangeType.GEOMETRY, ChangeType.STRUCTURE]
        ))

        # === CENTERLINE NEIGHBOR ATTRIBUTES ===
        self.add_node(DependencyNode(
            name='centerline.node_id_neighbors',  # node_id[1:4] via KDTree
            table='centerlines',
            depends_on=['centerline.geometry', 'node.xy'],
            recalc_func='_recalc_centerline_neighbors',
            change_types=[ChangeType.GEOMETRY]
        ))

    def add_node(self, node: DependencyNode):
        """Add a node to the dependency graph."""
        self.nodes[node.name] = node

    def get_downstream_deps(self, attr_name: str) -> List[str]:
        """Get all attributes that depend on the given attribute (directly or transitively)."""
        result = []
        visited = set()

        def dfs(name):
            for node_name, node in self.nodes.items():
                if name in node.depends_on and node_name not in visited:
                    visited.add(node_name)
                    result.append(node_name)
                    dfs(node_name)

        dfs(attr_name)
        return result

    def topological_sort(self, dirty_attrs: Set[str]) -> List[str]:
        """
        Return attributes in recalculation order (dependencies first).

        Uses Kahn's algorithm for topological sorting.
        """
        # Build subgraph of dirty attributes and their dependencies
        relevant = set()
        for attr in dirty_attrs:
            relevant.add(attr)
            relevant.update(self.get_downstream_deps(attr))

        # Calculate in-degree for relevant nodes
        in_degree = defaultdict(int)
        for name in relevant:
            node = self.nodes.get(name)
            if node:
                for dep in node.depends_on:
                    if dep in relevant:
                        in_degree[name] += 1

        # Start with nodes that have no dependencies (in relevant set)
        queue = [name for name in relevant if in_degree[name] == 0]
        result = []

        while queue:
            name = queue.pop(0)
            result.append(name)

            # Reduce in-degree of dependent nodes
            for node_name in relevant:
                node = self.nodes.get(node_name)
                if node and name in node.depends_on:
                    in_degree[node_name] -= 1
                    if in_degree[node_name] == 0:
                        queue.append(node_name)

        return result


class SWORDReactive:
    """
    Reactive update manager for SWORD database.

    Tracks changes and automatically recalculates dependent attributes.
    """

    def __init__(self, sword: 'SWORD'):
        self.sword = sword
        self.graph = DependencyGraph()
        self.dirty = DirtySet()
        self._dirty_attrs: Set[str] = set()
        self._pending_changes: List[Tuple[str, ChangeType, DirtySet]] = []

    def mark_dirty(
        self,
        attr_name: str,
        change_type: ChangeType = ChangeType.ATTRIBUTE,
        reach_ids: List[int] = None,
        node_ids: List[int] = None,
        cl_ids: List[int] = None,
        all_entities: bool = False
    ):
        """
        Mark an attribute as dirty (needing recalculation of dependents).

        Parameters
        ----------
        attr_name : str
            Name of the changed attribute (e.g., 'centerline.geometry', 'reach.topology')
        change_type : ChangeType
            Type of change that occurred
        reach_ids : list of int, optional
            Specific reach IDs affected
        node_ids : list of int, optional
            Specific node IDs affected
        cl_ids : list of int, optional
            Specific centerline IDs affected
        all_entities : bool
            If True, marks all entities as dirty
        """
        dirty_set = DirtySet()

        if all_entities:
            dirty_set.all_reaches = True
            dirty_set.all_nodes = True
            dirty_set.all_centerlines = True
        else:
            if reach_ids:
                dirty_set.reach_ids.update(reach_ids)
            if node_ids:
                dirty_set.node_ids.update(node_ids)
            if cl_ids:
                dirty_set.cl_ids.update(cl_ids)

        self._dirty_attrs.add(attr_name)
        self._pending_changes.append((attr_name, change_type, dirty_set))

        # Merge into global dirty set
        self.dirty.reach_ids.update(dirty_set.reach_ids)
        self.dirty.node_ids.update(dirty_set.node_ids)
        self.dirty.cl_ids.update(dirty_set.cl_ids)
        self.dirty.all_reaches = self.dirty.all_reaches or dirty_set.all_reaches
        self.dirty.all_nodes = self.dirty.all_nodes or dirty_set.all_nodes
        self.dirty.all_centerlines = self.dirty.all_centerlines or dirty_set.all_centerlines

    def get_recalc_plan(self) -> List[Tuple[str, str]]:
        """
        Get the planned recalculation order.

        Returns list of (attribute_name, recalc_function_name) tuples.
        """
        if not self._dirty_attrs:
            return []

        sorted_attrs = self.graph.topological_sort(self._dirty_attrs)
        plan = []

        for attr_name in sorted_attrs:
            node = self.graph.nodes.get(attr_name)
            if node and node.recalc_func:
                plan.append((attr_name, node.recalc_func))

        return plan

    def recalculate(self, dry_run: bool = False) -> List[str]:
        """
        Execute recalculations for all dirty attributes.

        Parameters
        ----------
        dry_run : bool
            If True, only return what would be recalculated without doing it.

        Returns
        -------
        list of str
            Names of attributes that were (or would be) recalculated.
        """
        plan = self.get_recalc_plan()
        recalculated = []

        if dry_run:
            return [attr for attr, _ in plan]

        for attr_name, func_name in plan:
            print(f"Recalculating: {attr_name}")
            func = getattr(self, func_name, None)
            if func:
                func()
                recalculated.append(attr_name)
            else:
                print(f"  Warning: No implementation for {func_name}")

        # Clear dirty state
        self._dirty_attrs.clear()
        self._pending_changes.clear()
        self.dirty = DirtySet()

        return recalculated

    # === RECALCULATION IMPLEMENTATIONS ===

    def _recalc_reach_lengths(self):
        """Recalculate reach lengths from centerline geometry."""
        # Implementation follows update_rch_node_lengths.py logic
        # For now, placeholder that shows the pattern
        print("  -> Recalculating reach.len from centerline x,y")
        # TODO: Implement actual calculation
        pass

    def _recalc_reach_bounds(self):
        """Recalculate reach bounding box from centerlines."""
        print("  -> Recalculating reach x, y, x_min, x_max, y_min, y_max")
        # TODO: Implement
        pass

    def _recalc_reach_dist_out(self):
        """Recalculate reach dist_out from topology and lengths."""
        print("  -> Recalculating reach.dist_out (outlet-to-headwater order)")
        # TODO: Implement - must traverse from outlets upstream
        pass

    def _recalc_reach_end_rch(self):
        """Recalculate reach end_rch flag from topology."""
        print("  -> Recalculating reach.end_rch from n_rch_up, n_rch_down")
        # TODO: Implement
        pass

    def _recalc_reach_main_side(self):
        """Recalculate reach main_side from topology."""
        print("  -> Recalculating reach.main_side")
        # TODO: Implement
        pass

    def _recalc_node_lengths(self):
        """Recalculate node lengths from centerline geometry."""
        print("  -> Recalculating node.len from centerline x,y")
        # TODO: Implement
        pass

    def _recalc_node_xy(self):
        """Recalculate node x,y from centerline geometry."""
        print("  -> Recalculating node x, y (centroid of centerlines)")
        # TODO: Implement
        pass

    def _recalc_node_dist_out(self):
        """Recalculate node dist_out from reach dist_out and node lengths."""
        print("  -> Recalculating node.dist_out from reach base + cumulative node.len")
        # TODO: Implement
        pass

    def _recalc_node_end_rch(self):
        """Propagate reach end_rch to nodes."""
        print("  -> Propagating reach.end_rch to node.end_rch")
        # TODO: Implement
        pass

    def _recalc_node_main_side(self):
        """Propagate reach main_side to nodes."""
        print("  -> Propagating reach.main_side to node.main_side")
        # TODO: Implement
        pass

    def _recalc_node_cl_id_bounds(self):
        """Recalculate node cl_id min/max from centerlines."""
        print("  -> Recalculating node.cl_id_min, node.cl_id_max")
        # TODO: Implement
        pass

    def _recalc_centerline_neighbors(self):
        """Recalculate centerline node_id neighbors using KDTree."""
        print("  -> Recalculating centerline.node_id[1:4] via spatial query")
        # TODO: Implement using scipy.spatial.KDTree
        pass


# === CONVENIENCE FUNCTIONS ===

def mark_geometry_changed(
    reactive: SWORDReactive,
    reach_ids: List[int] = None,
    all_reaches: bool = False
):
    """Mark geometry as changed, triggering full recalculation cascade."""
    reactive.mark_dirty(
        'centerline.geometry',
        ChangeType.GEOMETRY,
        reach_ids=reach_ids,
        all_entities=all_reaches
    )


def mark_topology_changed(
    reactive: SWORDReactive,
    reach_ids: List[int] = None,
    all_reaches: bool = False
):
    """Mark topology as changed, triggering topology recalculation cascade."""
    reactive.mark_dirty(
        'reach.topology',
        ChangeType.TOPOLOGY,
        reach_ids=reach_ids,
        all_entities=all_reaches
    )


def full_recalculate(reactive: SWORDReactive):
    """Trigger full recalculation of all derived attributes."""
    reactive.mark_dirty('centerline.geometry', ChangeType.GEOMETRY, all_entities=True)
    reactive.mark_dirty('reach.topology', ChangeType.TOPOLOGY, all_entities=True)
    reactive.recalculate()
