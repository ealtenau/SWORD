"""
Tests for v17c pipeline attribute computation functions.

Tests the core attribute computation functions from v17c_pipeline.py:
- compute_hydro_distances(G)
- compute_best_headwater_outlet(G)
- compute_mainstem(G, hw_out_attrs)

Uses the minimal test database with 100 reaches in a linear chain topology.
"""

import pytest
import duckdb
import networkx as nx
from pathlib import Path

from src.updates.sword_v17c_pipeline.v17c_pipeline import (
    build_reach_graph,
    compute_hydro_distances,
    compute_best_headwater_outlet,
    compute_mainstem,
    load_topology,
    load_reaches,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def test_db_path():
    """Path to the minimal test database."""
    return Path(__file__).parent / "fixtures" / "sword_test_minimal.duckdb"


@pytest.fixture
def db_connection(test_db_path):
    """DuckDB connection to test database."""
    if not test_db_path.exists():
        pytest.skip(f"Test database not found: {test_db_path}")
    conn = duckdb.connect(str(test_db_path), read_only=True)
    yield conn
    conn.close()


@pytest.fixture
def topology_df(db_connection):
    """Load topology DataFrame from test database."""
    return load_topology(db_connection, "NA")


@pytest.fixture
def reaches_df(db_connection):
    """Load reaches DataFrame from test database."""
    return load_reaches(db_connection, "NA")


@pytest.fixture
def reach_graph(topology_df, reaches_df):
    """Build reach graph from test data."""
    return build_reach_graph(topology_df, reaches_df)


@pytest.fixture
def hydro_distances(reach_graph):
    """Compute hydrologic distances."""
    return compute_hydro_distances(reach_graph)


@pytest.fixture
def hw_out_attrs(reach_graph):
    """Compute headwater/outlet attributes."""
    return compute_best_headwater_outlet(reach_graph)


@pytest.fixture
def mainstem(reach_graph, hw_out_attrs):
    """Compute mainstem classification."""
    return compute_mainstem(reach_graph, hw_out_attrs)


# =============================================================================
# Test Graph Construction
# =============================================================================

class TestBuildReachGraph:
    """Tests for build_reach_graph function."""

    def test_graph_is_directed(self, reach_graph):
        """Graph should be a DiGraph."""
        assert isinstance(reach_graph, nx.DiGraph)

    def test_graph_node_count(self, reach_graph):
        """Graph should have 100 nodes (reaches)."""
        assert reach_graph.number_of_nodes() == 100

    def test_graph_has_edges(self, reach_graph):
        """Graph should have edges representing flow connections."""
        assert reach_graph.number_of_edges() > 0

    def test_graph_nodes_have_attributes(self, reach_graph):
        """Nodes should have reach attributes."""
        for node in list(reach_graph.nodes())[:5]:
            attrs = reach_graph.nodes[node]
            assert 'reach_length' in attrs
            assert 'width' in attrs
            assert attrs['reach_length'] > 0

    def test_graph_is_dag(self, reach_graph):
        """Graph should be a directed acyclic graph (DAG)."""
        assert nx.is_directed_acyclic_graph(reach_graph)


# =============================================================================
# Test Hydrologic Distances
# =============================================================================

class TestComputeHydroDistances:
    """Tests for compute_hydro_distances function."""

    def test_returns_dict_for_all_nodes(self, reach_graph, hydro_distances):
        """Should return results for all nodes in graph."""
        assert len(hydro_distances) == reach_graph.number_of_nodes()

    def test_contains_required_keys(self, hydro_distances):
        """Each result should have hydro_dist_out and hydro_dist_hw."""
        for node, attrs in hydro_distances.items():
            assert 'hydro_dist_out' in attrs
            assert 'hydro_dist_hw' in attrs

    def test_hydro_dist_out_at_outlet_is_zero(self, reach_graph, hydro_distances):
        """hydro_dist_out should be 0 at outlets (no outgoing edges)."""
        outlets = [n for n in reach_graph.nodes() if reach_graph.out_degree(n) == 0]
        assert len(outlets) > 0, "Should have at least one outlet"

        for outlet in outlets:
            dist_out = hydro_distances[outlet]['hydro_dist_out']
            assert dist_out == 0, f"Outlet {outlet} should have hydro_dist_out=0, got {dist_out}"

    def test_hydro_dist_out_increases_upstream(self, reach_graph, hydro_distances):
        """hydro_dist_out should increase as we go upstream."""
        outlets = [n for n in reach_graph.nodes() if reach_graph.out_degree(n) == 0]
        headwaters = [n for n in reach_graph.nodes() if reach_graph.in_degree(n) == 0]

        for outlet in outlets:
            outlet_dist = hydro_distances[outlet]['hydro_dist_out']
            for hw in headwaters:
                hw_dist = hydro_distances[hw]['hydro_dist_out']
                # Headwater should have larger dist_out than outlet
                # (unless they're the same node in a single-node network)
                if hw != outlet:
                    assert hw_dist > outlet_dist, (
                        f"Headwater {hw} dist_out ({hw_dist}) should be > outlet {outlet} dist_out ({outlet_dist})"
                    )

    def test_hydro_dist_hw_at_headwater_is_zero(self, reach_graph, hydro_distances):
        """hydro_dist_hw should be 0 at headwaters (no incoming edges)."""
        headwaters = [n for n in reach_graph.nodes() if reach_graph.in_degree(n) == 0]
        assert len(headwaters) > 0, "Should have at least one headwater"

        for hw in headwaters:
            dist_hw = hydro_distances[hw]['hydro_dist_hw']
            assert dist_hw == 0, f"Headwater {hw} should have hydro_dist_hw=0, got {dist_hw}"

    def test_hydro_dist_hw_increases_downstream(self, reach_graph, hydro_distances):
        """hydro_dist_hw should increase as we go downstream."""
        outlets = [n for n in reach_graph.nodes() if reach_graph.out_degree(n) == 0]
        headwaters = [n for n in reach_graph.nodes() if reach_graph.in_degree(n) == 0]

        for hw in headwaters:
            hw_dist = hydro_distances[hw]['hydro_dist_hw']
            for outlet in outlets:
                outlet_dist = hydro_distances[outlet]['hydro_dist_hw']
                # Outlet should have larger dist_hw than headwater
                if outlet != hw:
                    assert outlet_dist > hw_dist, (
                        f"Outlet {outlet} dist_hw ({outlet_dist}) should be > headwater {hw} dist_hw ({hw_dist})"
                    )

    def test_hydro_dist_values_are_non_negative(self, hydro_distances):
        """All distance values should be non-negative."""
        for node, attrs in hydro_distances.items():
            dist_out = attrs['hydro_dist_out']
            dist_hw = attrs['hydro_dist_hw']

            # dist_out can be inf for unreachable nodes, but should not be negative
            assert dist_out >= 0 or dist_out == float('inf'), (
                f"Node {node}: hydro_dist_out should be >= 0, got {dist_out}"
            )
            assert dist_hw >= 0, f"Node {node}: hydro_dist_hw should be >= 0, got {dist_hw}"


# =============================================================================
# Test Best Headwater/Outlet Computation
# =============================================================================

class TestComputeBestHeadwaterOutlet:
    """Tests for compute_best_headwater_outlet function."""

    def test_returns_dict_for_all_nodes(self, reach_graph, hw_out_attrs):
        """Should return results for all nodes in graph."""
        assert len(hw_out_attrs) == reach_graph.number_of_nodes()

    def test_contains_required_keys(self, hw_out_attrs):
        """Each result should have required keys."""
        required_keys = ['best_headwater', 'best_outlet', 'pathlen_hw', 'pathlen_out', 'path_freq']
        for node, attrs in hw_out_attrs.items():
            for key in required_keys:
                assert key in attrs, f"Node {node} missing key: {key}"

    def test_best_headwater_is_valid_reach_id(self, reach_graph, hw_out_attrs):
        """best_headwater should be a valid node in the graph."""
        all_nodes = set(reach_graph.nodes())
        for node, attrs in hw_out_attrs.items():
            hw = attrs['best_headwater']
            if hw is not None:
                assert hw in all_nodes, f"best_headwater {hw} not in graph nodes"

    def test_best_outlet_is_valid_reach_id(self, reach_graph, hw_out_attrs):
        """best_outlet should be a valid node in the graph."""
        all_nodes = set(reach_graph.nodes())
        for node, attrs in hw_out_attrs.items():
            out = attrs['best_outlet']
            if out is not None:
                assert out in all_nodes, f"best_outlet {out} not in graph nodes"

    def test_headwater_best_headwater_is_itself(self, reach_graph, hw_out_attrs):
        """At a headwater node, best_headwater should be itself."""
        headwaters = [n for n in reach_graph.nodes() if reach_graph.in_degree(n) == 0]

        for hw in headwaters:
            assert hw_out_attrs[hw]['best_headwater'] == hw, (
                f"Headwater {hw} should have best_headwater == itself"
            )

    def test_outlet_best_outlet_is_itself(self, reach_graph, hw_out_attrs):
        """At an outlet node, best_outlet should be itself."""
        outlets = [n for n in reach_graph.nodes() if reach_graph.out_degree(n) == 0]

        for outlet in outlets:
            assert hw_out_attrs[outlet]['best_outlet'] == outlet, (
                f"Outlet {outlet} should have best_outlet == itself"
            )

    def test_pathlen_hw_at_headwater_is_zero(self, reach_graph, hw_out_attrs):
        """pathlen_hw should be 0 at headwaters."""
        headwaters = [n for n in reach_graph.nodes() if reach_graph.in_degree(n) == 0]

        for hw in headwaters:
            assert hw_out_attrs[hw]['pathlen_hw'] == 0, (
                f"Headwater {hw} should have pathlen_hw == 0"
            )

    def test_pathlen_out_at_outlet_is_zero(self, reach_graph, hw_out_attrs):
        """pathlen_out should be 0 at outlets."""
        outlets = [n for n in reach_graph.nodes() if reach_graph.out_degree(n) == 0]

        for outlet in outlets:
            assert hw_out_attrs[outlet]['pathlen_out'] == 0, (
                f"Outlet {outlet} should have pathlen_out == 0"
            )

    def test_pathlen_values_are_non_negative(self, hw_out_attrs):
        """All pathlen values should be non-negative."""
        for node, attrs in hw_out_attrs.items():
            assert attrs['pathlen_hw'] >= 0, (
                f"Node {node}: pathlen_hw should be >= 0, got {attrs['pathlen_hw']}"
            )
            assert attrs['pathlen_out'] >= 0, (
                f"Node {node}: pathlen_out should be >= 0, got {attrs['pathlen_out']}"
            )

    def test_path_freq_is_positive(self, hw_out_attrs):
        """path_freq should be at least 1 for all nodes."""
        for node, attrs in hw_out_attrs.items():
            assert attrs['path_freq'] >= 1, (
                f"Node {node}: path_freq should be >= 1, got {attrs['path_freq']}"
            )


# =============================================================================
# Test Mainstem Computation
# =============================================================================

class TestComputeMainstem:
    """Tests for compute_mainstem function."""

    def test_returns_dict_for_all_nodes(self, reach_graph, mainstem):
        """Should return results for all nodes in graph."""
        assert len(mainstem) == reach_graph.number_of_nodes()

    def test_returns_boolean_values(self, mainstem):
        """All values should be boolean."""
        for node, is_main in mainstem.items():
            assert isinstance(is_main, bool), f"Node {node}: is_mainstem should be bool, got {type(is_main)}"

    def test_at_least_one_mainstem_reach(self, mainstem):
        """At least one reach should be on the mainstem."""
        n_mainstem = sum(mainstem.values())
        assert n_mainstem >= 1, "Should have at least one mainstem reach"

    def test_mainstem_forms_connected_path(self, reach_graph, mainstem):
        """Mainstem reaches should form a connected path."""
        mainstem_nodes = [n for n, is_main in mainstem.items() if is_main]

        if len(mainstem_nodes) <= 1:
            # Single node or no mainstem is trivially connected
            return

        # Create subgraph of mainstem nodes
        subgraph = reach_graph.subgraph(mainstem_nodes)

        # Convert to undirected for connectivity check
        undirected = subgraph.to_undirected()

        # Should be connected
        assert nx.is_connected(undirected), "Mainstem should form a connected path"

    def test_mainstem_includes_outlet(self, reach_graph, mainstem, hw_out_attrs):
        """Mainstem should include the outlet of the main path."""
        # Find nodes where best_outlet equals their own outlet
        outlets = [n for n in reach_graph.nodes() if reach_graph.out_degree(n) == 0]

        # At least one outlet should be on mainstem
        outlet_on_mainstem = any(mainstem.get(o, False) for o in outlets)
        assert outlet_on_mainstem, "At least one outlet should be on the mainstem"

    def test_mainstem_includes_headwater(self, reach_graph, mainstem, hw_out_attrs):
        """Mainstem should include a headwater of the main path."""
        headwaters = [n for n in reach_graph.nodes() if reach_graph.in_degree(n) == 0]

        # At least one headwater should be on mainstem
        hw_on_mainstem = any(mainstem.get(hw, False) for hw in headwaters)
        assert hw_on_mainstem, "At least one headwater should be on the mainstem"


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests verifying consistency between computed attributes."""

    def test_hydro_dist_sum_approximates_total_length(self, reach_graph, hydro_distances):
        """hydro_dist_out + hydro_dist_hw should approximate total path length."""
        # For nodes on a linear path, dist_out + dist_hw should be roughly constant
        # (equal to total path length from HW to outlet)

        # Get headwater and outlet
        headwaters = [n for n in reach_graph.nodes() if reach_graph.in_degree(n) == 0]
        outlets = [n for n in reach_graph.nodes() if reach_graph.out_degree(n) == 0]

        if not headwaters or not outlets:
            pytest.skip("Need headwater and outlet for this test")

        hw = headwaters[0]
        outlet = outlets[0]

        # For linear network, the sum should be the same for all nodes
        # We test that the headwater dist_out equals the outlet dist_hw
        hw_dist_out = hydro_distances[hw]['hydro_dist_out']
        outlet_dist_hw = hydro_distances[outlet]['hydro_dist_hw']

        # These should be approximately equal (both represent total path length)
        # Allow for floating point differences
        if hw_dist_out < float('inf') and outlet_dist_hw > 0:
            ratio = hw_dist_out / outlet_dist_hw if outlet_dist_hw > 0 else 0
            assert 0.9 <= ratio <= 1.1, (
                f"Total path lengths should match: HW dist_out={hw_dist_out}, outlet dist_hw={outlet_dist_hw}"
            )

    def test_mainstem_path_exists_between_best_hw_and_outlet(self, reach_graph, hw_out_attrs, mainstem):
        """The mainstem should form a valid path from best_headwater to best_outlet."""
        mainstem_nodes = [n for n, is_main in mainstem.items() if is_main]

        if not mainstem_nodes:
            pytest.skip("No mainstem nodes found")

        # Find the headwater and outlet on the mainstem
        mainstem_headwaters = [n for n in mainstem_nodes if reach_graph.in_degree(n) == 0]
        mainstem_outlets = [n for n in mainstem_nodes if reach_graph.out_degree(n) == 0]

        if not mainstem_headwaters or not mainstem_outlets:
            # Mainstem might not reach the true ends in complex networks
            return

        hw = mainstem_headwaters[0]
        outlet = mainstem_outlets[0]

        # Verify a path exists
        try:
            path = nx.shortest_path(reach_graph, hw, outlet)
            # All nodes on this path should be on mainstem
            for node in path:
                assert mainstem[node], f"Node {node} on HW-outlet path should be mainstem"
        except nx.NetworkXNoPath:
            pytest.fail(f"No path found from mainstem HW {hw} to outlet {outlet}")

    def test_consistency_across_linear_chain(self, reach_graph, hydro_distances, hw_out_attrs, mainstem):
        """For a linear chain, verify consistent attribute progression."""
        # Get topological order
        try:
            topo_order = list(nx.topological_sort(reach_graph))
        except nx.NetworkXUnfeasible:
            pytest.skip("Graph has cycles")

        if len(topo_order) < 2:
            return

        # hydro_dist_out should decrease along topological order (upstream to downstream)
        prev_dist_out = float('inf')
        for node in topo_order:
            curr_dist_out = hydro_distances[node]['hydro_dist_out']
            if curr_dist_out < float('inf'):
                assert curr_dist_out <= prev_dist_out, (
                    f"hydro_dist_out should decrease downstream: {prev_dist_out} -> {curr_dist_out}"
                )
                prev_dist_out = curr_dist_out

        # hydro_dist_hw should increase along topological order
        prev_dist_hw = -1
        for node in topo_order:
            curr_dist_hw = hydro_distances[node]['hydro_dist_hw']
            assert curr_dist_hw >= prev_dist_hw, (
                f"hydro_dist_hw should increase downstream: {prev_dist_hw} -> {curr_dist_hw}"
            )
            prev_dist_hw = curr_dist_hw


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_node_graph(self):
        """Test with a single-node graph."""
        G = nx.DiGraph()
        G.add_node(1, reach_length=1000, width=50)

        hydro_dist = compute_hydro_distances(G)
        assert 1 in hydro_dist
        assert hydro_dist[1]['hydro_dist_out'] == 0
        assert hydro_dist[1]['hydro_dist_hw'] == 0

        hw_out = compute_best_headwater_outlet(G)
        assert hw_out[1]['best_headwater'] == 1
        assert hw_out[1]['best_outlet'] == 1
        assert hw_out[1]['pathlen_hw'] == 0
        assert hw_out[1]['pathlen_out'] == 0

        mainstem = compute_mainstem(G, hw_out)
        assert mainstem[1] is True

    def test_two_node_linear_graph(self):
        """Test with a simple two-node linear graph."""
        G = nx.DiGraph()
        G.add_node(1, reach_length=1000, width=50)
        G.add_node(2, reach_length=1500, width=60)
        G.add_edge(1, 2)

        hydro_dist = compute_hydro_distances(G)
        # Node 1 is headwater, node 2 is outlet
        assert hydro_dist[1]['hydro_dist_hw'] == 0
        assert hydro_dist[2]['hydro_dist_out'] == 0

        hw_out = compute_best_headwater_outlet(G)
        assert hw_out[1]['best_headwater'] == 1
        assert hw_out[2]['best_outlet'] == 2

        mainstem = compute_mainstem(G, hw_out)
        assert mainstem[1] is True
        assert mainstem[2] is True

    def test_y_shaped_network(self):
        """Test with a Y-shaped network (two tributaries merging)."""
        G = nx.DiGraph()
        # Two headwaters (1, 2) merge at 3, flow to outlet 4
        G.add_node(1, reach_length=1000, width=30)
        G.add_node(2, reach_length=1200, width=40)  # Wider tributary
        G.add_node(3, reach_length=800, width=60)
        G.add_node(4, reach_length=900, width=70)

        G.add_edge(1, 3)
        G.add_edge(2, 3)
        G.add_edge(3, 4)

        hydro_dist = compute_hydro_distances(G)
        # Both headwaters should have hydro_dist_hw = 0
        assert hydro_dist[1]['hydro_dist_hw'] == 0
        assert hydro_dist[2]['hydro_dist_hw'] == 0
        # Outlet should have hydro_dist_out = 0
        assert hydro_dist[4]['hydro_dist_out'] == 0

        hw_out = compute_best_headwater_outlet(G)
        # Node 3 should pick the wider tributary (node 2) as best_headwater
        # (width is used as tiebreaker)
        assert hw_out[3]['best_headwater'] == 2
        assert hw_out[4]['best_outlet'] == 4

        mainstem = compute_mainstem(G, hw_out)
        # At least nodes 3 and 4 should be on mainstem
        assert mainstem[3] is True
        assert mainstem[4] is True

    def test_empty_graph(self):
        """Test with an empty graph."""
        G = nx.DiGraph()

        hydro_dist = compute_hydro_distances(G)
        assert len(hydro_dist) == 0

        hw_out = compute_best_headwater_outlet(G)
        assert len(hw_out) == 0

        mainstem = compute_mainstem(G, hw_out)
        assert len(mainstem) == 0
