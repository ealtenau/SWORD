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

try:
    from geopy import distance as geodesic_distance
except ImportError:
    geodesic_distance = None

if TYPE_CHECKING:
    from .sword_class import SWORD


def _get_geodesic_distances(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """
    Calculate geodesic distances between consecutive points.

    Parameters
    ----------
    lon : np.ndarray
        Longitude values (WGS84)
    lat : np.ndarray
        Latitude values (WGS84)

    Returns
    -------
    np.ndarray
        Distances in meters, with leading 0 for first point
    """
    if geodesic_distance is None:
        raise ImportError("geopy is required for distance calculations. Install with: pip install geopy")

    n_points = len(lon)
    if n_points < 2:
        return np.array([0.0])

    distances = np.zeros(n_points - 1)
    for i in range(n_points - 1):
        start = (lat[i], lon[i])
        finish = (lat[i + 1], lon[i + 1])
        distances[i] = geodesic_distance.geodesic(start, finish).m

    return np.concatenate([[0.0], distances])


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
        """
        Recalculate reach lengths from centerline geometry.

        Logic: For each reach, get all centerlines sorted by cl_id,
        calculate geodesic distances between consecutive points,
        and set reach.len to the total distance.
        """
        print("  -> Recalculating reach.len from centerline x,y")

        sword = self.sword
        reaches = sword.reaches
        centerlines = sword.centerlines

        # Get affected reach IDs
        if self.dirty.all_reaches:
            reach_ids = reaches.id[:]
        else:
            reach_ids = list(self.dirty.reach_ids)

        for reach_id in reach_ids:
            # Find centerlines for this reach
            cl_mask = centerlines.reach_id[0, :] == reach_id
            cl_indices = np.where(cl_mask)[0]

            if len(cl_indices) == 0:
                continue

            # Sort by cl_id
            sorted_indices = cl_indices[np.argsort(centerlines.cl_id[cl_indices])]

            # Get coordinates
            x_coords = centerlines.x[sorted_indices]
            y_coords = centerlines.y[sorted_indices]

            # Calculate distances
            distances = _get_geodesic_distances(x_coords, y_coords)
            total_length = np.sum(distances)

            # Update reach length
            reach_idx = np.where(reaches.id == reach_id)[0]
            if len(reach_idx) > 0:
                reaches.len[reach_idx[0]] = total_length

    def _recalc_reach_bounds(self):
        """
        Recalculate reach bounding box from centerlines.

        Updates: x, y (centroid), x_min, x_max, y_min, y_max
        """
        print("  -> Recalculating reach x, y, x_min, x_max, y_min, y_max")

        sword = self.sword
        reaches = sword.reaches
        centerlines = sword.centerlines

        if self.dirty.all_reaches:
            reach_ids = reaches.id[:]
        else:
            reach_ids = list(self.dirty.reach_ids)

        for reach_id in reach_ids:
            cl_mask = centerlines.reach_id[0, :] == reach_id
            cl_indices = np.where(cl_mask)[0]

            if len(cl_indices) == 0:
                continue

            x_coords = centerlines.x[cl_indices]
            y_coords = centerlines.y[cl_indices]

            reach_idx = np.where(reaches.id == reach_id)[0]
            if len(reach_idx) > 0:
                idx = reach_idx[0]
                reaches.x[idx] = np.mean(x_coords)
                reaches.y[idx] = np.mean(y_coords)
                reaches.x_min[idx] = np.min(x_coords)
                reaches.x_max[idx] = np.max(x_coords)
                reaches.y_min[idx] = np.min(y_coords)
                reaches.y_max[idx] = np.max(y_coords)

    def _recalc_reach_dist_out(self):
        """
        Recalculate reach dist_out from topology and lengths.

        Algorithm: Traverse from outlets (n_rch_down=0) upstream.
        Outlet dist_out = reach.len
        Upstream dist_out = reach.len + max(downstream dist_out values)
        """
        print("  -> Recalculating reach.dist_out (outlet-to-headwater order)")

        sword = self.sword
        reaches = sword.reaches
        n_reaches = len(reaches.id)

        # Initialize dist_out array
        dist_out = np.full(n_reaches, np.nan)

        # Find outlets (no downstream neighbors)
        outlet_mask = reaches.n_rch_down[:] == 0
        outlet_indices = np.where(outlet_mask)[0]

        # Process queue - start from outlets
        processed = set()
        queue = list(outlet_indices)

        while queue:
            idx = queue.pop(0)
            reach_id = reaches.id[idx]

            if idx in processed:
                continue

            n_down = reaches.n_rch_down[idx]

            if n_down == 0:
                # Outlet reach - dist_out is just its own length
                dist_out[idx] = reaches.len[idx]
            else:
                # Get downstream neighbors
                dn_ids = reaches.rch_id_dn[:, idx]
                dn_ids = dn_ids[dn_ids > 0]

                # Check if all downstream reaches are processed
                all_dn_processed = True
                max_dn_dist = 0

                for dn_id in dn_ids:
                    dn_idx = np.where(reaches.id == dn_id)[0]
                    if len(dn_idx) > 0:
                        dn_idx = dn_idx[0]
                        if dn_idx not in processed:
                            all_dn_processed = False
                            break
                        max_dn_dist = max(max_dn_dist, dist_out[dn_idx])

                if not all_dn_processed:
                    # Put back in queue and try later
                    queue.append(idx)
                    continue

                dist_out[idx] = reaches.len[idx] + max_dn_dist

            processed.add(idx)

            # Add upstream neighbors to queue
            up_ids = reaches.rch_id_up[:, idx]
            up_ids = up_ids[up_ids > 0]
            for up_id in up_ids:
                up_idx = np.where(reaches.id == up_id)[0]
                if len(up_idx) > 0 and up_idx[0] not in processed:
                    queue.append(up_idx[0])

        # Update reach dist_out values
        for i in range(n_reaches):
            if not np.isnan(dist_out[i]):
                reaches.dist_out[i] = dist_out[i]

    def _recalc_reach_end_rch(self):
        """
        Recalculate reach end_rch flag from topology.

        Values:
        - 0: Normal reach
        - 1: Headwater (n_rch_up == 0)
        - 2: Outlet (n_rch_down == 0)
        - 3: Junction (n_rch_up > 1 or n_rch_down > 1)
        """
        print("  -> Recalculating reach.end_rch from n_rch_up, n_rch_down")

        sword = self.sword
        reaches = sword.reaches
        n_reaches = len(reaches.id)

        # Default all to normal (0)
        end_rch = np.zeros(n_reaches, dtype=np.int32)

        # Mark junctions first (can be overridden by headwater/outlet)
        junction_mask = (reaches.n_rch_up[:] > 1) | (reaches.n_rch_down[:] > 1)
        end_rch[junction_mask] = 3

        # Mark headwaters
        headwater_mask = reaches.n_rch_up[:] == 0
        end_rch[headwater_mask] = 1

        # Mark outlets
        outlet_mask = reaches.n_rch_down[:] == 0
        end_rch[outlet_mask] = 2

        # Update
        for i in range(n_reaches):
            reaches.end_rch[i] = end_rch[i]

    def _recalc_reach_main_side(self):
        """
        Recalculate reach main_side from topology.

        Note: This is a complex calculation that typically requires
        external data (GPKG with accumulation values). For now, we
        preserve existing values and only recalculate if topology changed.
        """
        print("  -> Recalculating reach.main_side (preserving existing values)")
        # main_side calculation is complex and requires accumulation data
        # from GeoPackage files. For now, we preserve existing values.
        # Full implementation would trace main channels using accumulated
        # upstream reach counts.
        pass

    def _recalc_node_lengths(self):
        """
        Recalculate node lengths from centerline geometry.

        For each node, calculate the cumulative distance of all
        centerlines belonging to that node.
        """
        print("  -> Recalculating node.len from centerline x,y")

        sword = self.sword
        nodes = sword.nodes
        centerlines = sword.centerlines
        reaches = sword.reaches

        if self.dirty.all_reaches:
            reach_ids = reaches.id[:]
        else:
            reach_ids = list(self.dirty.reach_ids)

        for reach_id in reach_ids:
            # Get centerlines for this reach, sorted by cl_id
            cl_mask = centerlines.reach_id[0, :] == reach_id
            cl_indices = np.where(cl_mask)[0]

            if len(cl_indices) == 0:
                continue

            sorted_indices = cl_indices[np.argsort(centerlines.cl_id[cl_indices])]

            x_coords = centerlines.x[sorted_indices]
            y_coords = centerlines.y[sorted_indices]
            distances = _get_geodesic_distances(x_coords, y_coords)

            # Get unique nodes in sorted order
            node_ids_sorted = centerlines.node_id[0, sorted_indices]
            unique_nodes = []
            seen = set()
            for nid in node_ids_sorted:
                if nid not in seen:
                    unique_nodes.append(nid)
                    seen.add(nid)

            # Calculate node lengths
            for node_id in unique_nodes:
                node_cl_mask = centerlines.node_id[0, sorted_indices] == node_id
                node_distances = distances[node_cl_mask]
                node_length = np.sum(node_distances)

                node_idx = np.where(nodes.id == node_id)[0]
                if len(node_idx) > 0:
                    nodes.len[node_idx[0]] = node_length

    def _recalc_node_xy(self):
        """
        Recalculate node x,y as centroid of associated centerlines.
        """
        print("  -> Recalculating node x, y (centroid of centerlines)")

        sword = self.sword
        nodes = sword.nodes
        centerlines = sword.centerlines

        if self.dirty.all_nodes:
            node_ids = nodes.id[:]
        else:
            node_ids = list(self.dirty.node_ids)

        for node_id in node_ids:
            cl_mask = centerlines.node_id[0, :] == node_id
            cl_indices = np.where(cl_mask)[0]

            if len(cl_indices) == 0:
                continue

            x_coords = centerlines.x[cl_indices]
            y_coords = centerlines.y[cl_indices]

            node_idx = np.where(nodes.id == node_id)[0]
            if len(node_idx) > 0:
                nodes.x[node_idx[0]] = np.mean(x_coords)
                nodes.y[node_idx[0]] = np.mean(y_coords)

    def _recalc_node_dist_out(self):
        """
        Recalculate node dist_out from reach dist_out and node lengths.

        node.dist_out = reach_base_dist + cumulative_node_length
        where reach_base_dist = reach.dist_out - reach.len
        """
        print("  -> Recalculating node.dist_out from reach base + cumulative node.len")

        sword = self.sword
        nodes = sword.nodes
        reaches = sword.reaches

        for r in range(len(reaches.id)):
            reach_id = reaches.id[r]

            # Find nodes in this reach
            node_mask = nodes.reach_id[:] == reach_id
            node_indices = np.where(node_mask)[0]

            if len(node_indices) == 0:
                continue

            # Sort nodes by node_id
            sorted_indices = node_indices[np.argsort(nodes.id[node_indices])]

            # Base distance (upstream of this reach)
            base_dist = reaches.dist_out[r] - reaches.len[r]

            # Cumulative node lengths
            cumsum = np.cumsum(nodes.len[sorted_indices])

            # Update node dist_out
            for i, idx in enumerate(sorted_indices):
                nodes.dist_out[idx] = base_dist + cumsum[i]

    def _recalc_node_end_rch(self):
        """
        Propagate reach end_rch to nodes.

        Each node inherits the end_rch value from its parent reach.
        """
        print("  -> Propagating reach.end_rch to node.end_rch")

        sword = self.sword
        nodes = sword.nodes
        reaches = sword.reaches

        for r in range(len(reaches.id)):
            reach_id = reaches.id[r]
            end_rch_val = reaches.end_rch[r]

            node_mask = nodes.reach_id[:] == reach_id
            node_indices = np.where(node_mask)[0]

            for idx in node_indices:
                nodes.end_rch[idx] = end_rch_val

    def _recalc_node_main_side(self):
        """
        Propagate reach main_side to nodes.

        Each node inherits the main_side value from its parent reach.
        """
        print("  -> Propagating reach.main_side to node.main_side")

        sword = self.sword
        nodes = sword.nodes
        reaches = sword.reaches

        for r in range(len(reaches.id)):
            reach_id = reaches.id[r]
            main_side_val = reaches.main_side[r]

            node_mask = nodes.reach_id[:] == reach_id
            node_indices = np.where(node_mask)[0]

            for idx in node_indices:
                nodes.main_side[idx] = main_side_val

    def _recalc_node_cl_id_bounds(self):
        """
        Recalculate node cl_id min/max from centerlines.

        Updates nodes.cl_id[0] (min) and nodes.cl_id[1] (max)
        """
        print("  -> Recalculating node.cl_id_min, node.cl_id_max")

        sword = self.sword
        nodes = sword.nodes
        centerlines = sword.centerlines

        for n in range(len(nodes.id)):
            node_id = nodes.id[n]

            cl_mask = centerlines.node_id[0, :] == node_id
            cl_indices = np.where(cl_mask)[0]

            if len(cl_indices) == 0:
                continue

            cl_ids = centerlines.cl_id[cl_indices]
            # Note: cl_id is a 2D array [2, N] where [0] is min and [1] is max
            # This requires special handling since cl_id may not be writable
            # For now, we skip this as it requires 2D WritableArray support

    def _recalc_centerline_neighbors(self):
        """
        Recalculate centerline node_id neighbors using KDTree.

        Note: This requires scipy.spatial.KDTree and modifies
        the node_id[1:4] array which is currently read-only.
        """
        print("  -> Recalculating centerline.node_id[1:4] via spatial query")
        # This requires 2D WritableArray support which is not yet implemented
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
