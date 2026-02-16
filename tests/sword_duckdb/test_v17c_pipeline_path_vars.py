"""
Tests for compute_path_variables in the v17c pipeline.

Tests path_freq, stream_order, path_segs, and path_order computation
across linear chains, confluences, bifurcations, side channels, and
the 100-reach test DB fixture.
"""

import math
from pathlib import Path

import duckdb
import networkx as nx
import pandas as pd
import pytest

from src.updates.sword_v17c_pipeline.v17c_pipeline import (
    build_reach_graph,
    build_section_graph,
    compute_path_variables,
    identify_junctions,
    load_reaches,
    load_topology,
)

pytestmark = [pytest.mark.pipeline, pytest.mark.unit]


# =============================================================================
# Helper
# =============================================================================


def _build_and_compute(topology_df, reaches_df):
    """Build graph, sections, and compute path variables."""
    G = build_reach_graph(topology_df, reaches_df)
    junctions = identify_junctions(G)
    _, sections_df = build_section_graph(G, junctions)
    return G, sections_df, compute_path_variables(G, sections_df)


def _make_reaches(reach_ids, *, dist_outs=None, **overrides):
    """Create a minimal reaches DataFrame for the given IDs."""
    n = len(reach_ids)
    if dist_outs is None:
        dist_outs = [float((n - i) * 1000) for i in range(n)]
    data = {
        "reach_id": reach_ids,
        "reach_length": [1000.0] * n,
        "width": [50.0 + i * 5 for i in range(n)],
        "slope": [0.001] * n,
        "facc": [100 * (i + 1) for i in range(n)],
        "n_rch_up": [0] * n,
        "n_rch_down": [0] * n,
        "dist_out": dist_outs,
        "path_freq": [1] * n,
        "stream_order": [1] * n,
        "lakeflag": [0] * n,
    }
    data.update(overrides)
    return pd.DataFrame(data)


# Linear topology: 0 -> 1 -> 2 -> 3
_LINEAR_TOPO = pd.DataFrame(
    {
        "reach_id": [0, 1, 1, 2, 2, 3],
        "direction": ["down", "down", "up", "down", "up", "up"],
        "neighbor_rank": [0, 0, 0, 0, 0, 0],
        "neighbor_reach_id": [1, 2, 0, 3, 1, 2],
    }
)

# Confluence topology: 0,1 -> 2 -> 3
_CONFLUENCE_TOPO = pd.DataFrame(
    {
        "reach_id": [0, 1, 2, 2, 2, 3],
        "direction": ["down", "down", "up", "up", "down", "up"],
        "neighbor_rank": [0, 0, 0, 1, 0, 0],
        "neighbor_reach_id": [2, 2, 0, 1, 3, 2],
    }
)

# Bifurcation topology: 0 -> 1, 0 -> 2
_BIFURCATION_TOPO = pd.DataFrame(
    {
        "reach_id": [0, 0, 1, 2],
        "direction": ["down", "down", "up", "up"],
        "neighbor_rank": [0, 1, 0, 0],
        "neighbor_reach_id": [1, 2, 0, 0],
    }
)

# Short chain topology: 0 -> 1 -> 2
_SHORT_TOPO = pd.DataFrame(
    {
        "reach_id": [0, 1, 1, 2],
        "direction": ["down", "up", "down", "up"],
        "neighbor_rank": [0, 0, 0, 0],
        "neighbor_reach_id": [1, 0, 2, 1],
    }
)


# =============================================================================
# TestPathFreqLinear
# =============================================================================


class TestPathFreqLinear:
    """Linear chain: 0 -> 1 -> 2 -> 3.  All main_side=0, type=1."""

    @pytest.fixture
    def result(self):
        reaches = _make_reaches(
            [0, 1, 2, 3],
            dist_outs=[3000.0, 2000.0, 1000.0, 0.0],
            n_rch_up=[0, 1, 1, 1],
            n_rch_down=[1, 1, 1, 0],
        )
        _, _, res = _build_and_compute(_LINEAR_TOPO, reaches)
        return res

    def test_all_path_freq_one(self, result):
        for rid in [0, 1, 2, 3]:
            assert result[rid]["path_freq"] == 1

    def test_stream_order_one(self, result):
        for rid in [0, 1, 2, 3]:
            assert result[rid]["stream_order"] == 1

    def test_monotonicity(self, result):
        for u, v in [(0, 1), (1, 2), (2, 3)]:
            assert result[v]["path_freq"] >= result[u]["path_freq"]

    def test_all_reaches_present(self, result):
        assert set(result.keys()) == {0, 1, 2, 3}


# =============================================================================
# TestPathFreqConfluence
# =============================================================================


class TestPathFreqConfluence:
    """Confluence: 0,1 -> 2 -> 3.  Expected pf = [1,1,2,2]."""

    @pytest.fixture
    def result(self):
        reaches = _make_reaches(
            [0, 1, 2, 3],
            dist_outs=[4000.0, 4200.0, 2000.0, 0.0],
            n_rch_up=[0, 0, 2, 1],
            n_rch_down=[1, 1, 1, 0],
        )
        _, _, res = _build_and_compute(_CONFLUENCE_TOPO, reaches)
        return res

    def test_headwaters_pf_one(self, result):
        assert result[0]["path_freq"] == 1
        assert result[1]["path_freq"] == 1

    def test_confluence_pf_sum(self, result):
        assert result[2]["path_freq"] == 2

    def test_downstream_carries_sum(self, result):
        assert result[3]["path_freq"] == 2

    def test_stream_order_at_confluence(self, result):
        assert result[2]["stream_order"] == 2
        assert result[3]["stream_order"] == 2


# =============================================================================
# TestPathFreqBifurcation
# =============================================================================


class TestPathFreqBifurcation:
    """Bifurcation: 0 -> 1, 0 -> 2.  Expected pf = [1,1,1]."""

    @pytest.fixture
    def result(self):
        reaches = _make_reaches(
            [0, 1, 2],
            dist_outs=[2000.0, 0.0, 0.0],
            n_rch_up=[0, 1, 1],
            n_rch_down=[2, 0, 0],
        )
        _, _, res = _build_and_compute(_BIFURCATION_TOPO, reaches)
        return res

    def test_all_pf_one(self, result):
        assert result[0]["path_freq"] == 1
        assert result[1]["path_freq"] == 1
        assert result[2]["path_freq"] == 1


# =============================================================================
# TestPathFreqSideChannel
# =============================================================================


class TestPathFreqSideChannel:
    """Side channel (main_side=1) gets computed pf but excluded from sums."""

    @pytest.fixture
    def result(self):
        reaches = _make_reaches(
            [0, 1, 2],
            dist_outs=[2000.0, 1000.0, 0.0],
            n_rch_up=[0, 1, 1],
            n_rch_down=[1, 1, 0],
        )
        G = build_reach_graph(_SHORT_TOPO, reaches)
        G.nodes[1]["main_side"] = 1
        junctions = identify_junctions(G)
        _, sections_df = build_section_graph(G, junctions)
        return compute_path_variables(G, sections_df)

    def test_side_channel_gets_computed_pf(self, result):
        """Side channels get real computed pf, not -9999."""
        assert result[1]["path_freq"] == 1

    def test_side_channel_gets_stream_order(self, result):
        assert result[1]["stream_order"] == 1

    def test_headwater_valid(self, result):
        assert result[0]["path_freq"] == 1

    def test_downstream_excludes_side_from_sums(self, result):
        """Downstream reach ignores side channel in sums, falls back to 1."""
        assert result[2]["path_freq"] == 1


class TestPathFreqGhostType:
    """Ghost reach (type=6) gets -9999 for all path variables."""

    @pytest.fixture
    def result(self):
        reaches = _make_reaches(
            [0, 1, 2],
            dist_outs=[2000.0, 1000.0, 0.0],
            n_rch_up=[0, 1, 1],
            n_rch_down=[1, 1, 0],
        )
        G = build_reach_graph(_SHORT_TOPO, reaches)
        G.nodes[1]["type"] = 6
        junctions = identify_junctions(G)
        _, sections_df = build_section_graph(G, junctions)
        return compute_path_variables(G, sections_df)

    def test_ghost_gets_sentinel(self, result):
        assert result[1]["path_freq"] == -9999
        assert result[1]["stream_order"] == -9999
        assert result[1]["path_segs"] == -9999
        assert result[1]["path_order"] == -9999


# =============================================================================
# TestStreamOrder
# =============================================================================


class TestStreamOrder:
    """Verify stream_order = round(log(pf)) + 1."""

    @pytest.mark.parametrize(
        "pf,expected_so",
        [
            (1, 1),
            (2, 2),
            (3, 2),
            (7, 3),
            (8, 3),
            (20, 4),
            (100, 6),
            (1000, 8),
        ],
    )
    def test_formula(self, pf, expected_so):
        assert int(round(math.log(pf))) + 1 == expected_so

    def test_side_channel_gets_valid_so(self):
        """Side channels get computed stream_order, not -9999."""
        G = nx.DiGraph()
        G.add_node(0, dist_out=0.0, main_side=1)
        sections_df = pd.DataFrame(
            columns=[
                "section_id",
                "upstream_junction",
                "downstream_junction",
                "reach_ids",
                "distance",
                "n_reaches",
            ]
        )
        result = compute_path_variables(G, sections_df)
        assert result[0]["stream_order"] == 1

    def test_ghost_gets_sentinel_so(self):
        """Ghost reaches get -9999 stream_order."""
        G = nx.DiGraph()
        G.add_node(0, dist_out=0.0, type=6)
        sections_df = pd.DataFrame(
            columns=[
                "section_id",
                "upstream_junction",
                "downstream_junction",
                "reach_ids",
                "distance",
                "n_reaches",
            ]
        )
        result = compute_path_variables(G, sections_df)
        assert result[0]["stream_order"] == -9999


# =============================================================================
# TestPathSegs
# =============================================================================


class TestPathSegs:
    """path_segs assigns section-based segment identifiers."""

    def test_linear_one_section(self):
        reaches = _make_reaches(
            [0, 1, 2, 3],
            dist_outs=[3000.0, 2000.0, 1000.0, 0.0],
            n_rch_up=[0, 1, 1, 1],
            n_rch_down=[1, 1, 1, 0],
        )
        _, _, result = _build_and_compute(_LINEAR_TOPO, reaches)
        for rid in [0, 1, 2, 3]:
            assert result[rid]["path_segs"] > 0

    def test_confluence_multiple_sections(self):
        reaches = _make_reaches(
            [0, 1, 2, 3],
            dist_outs=[4000.0, 4200.0, 2000.0, 0.0],
            n_rch_up=[0, 0, 2, 1],
            n_rch_down=[1, 1, 1, 0],
        )
        _, _, result = _build_and_compute(_CONFLUENCE_TOPO, reaches)
        segs = {rid: result[rid]["path_segs"] for rid in [0, 1, 2, 3]}
        for v in segs.values():
            assert v > 0

    def test_every_reach_has_path_segs(self):
        reaches = _make_reaches(
            [0, 1, 2],
            dist_outs=[2000.0, 1000.0, 0.0],
            n_rch_up=[0, 1, 1],
            n_rch_down=[1, 1, 0],
        )
        _, _, result = _build_and_compute(_SHORT_TOPO, reaches)
        for rid in [0, 1, 2]:
            assert result[rid]["path_segs"] > 0


# =============================================================================
# TestPathOrder
# =============================================================================


class TestPathOrder:
    """path_order ranks by dist_out ASC within path_freq groups."""

    def test_linear_chain_ranking(self):
        reaches = _make_reaches(
            [0, 1, 2, 3],
            dist_outs=[3000.0, 2000.0, 1000.0, 0.0],
            n_rch_up=[0, 1, 1, 1],
            n_rch_down=[1, 1, 1, 0],
        )
        _, _, result = _build_and_compute(_LINEAR_TOPO, reaches)
        assert result[3]["path_order"] == 1
        assert result[2]["path_order"] == 2
        assert result[1]["path_order"] == 3
        assert result[0]["path_order"] == 4

    def test_confluence_separate_groups(self):
        reaches = _make_reaches(
            [0, 1, 2, 3],
            dist_outs=[4000.0, 4200.0, 2000.0, 0.0],
            n_rch_up=[0, 0, 2, 1],
            n_rch_down=[1, 1, 1, 0],
        )
        _, _, result = _build_and_compute(_CONFLUENCE_TOPO, reaches)
        # pf=1 group
        assert result[0]["path_order"] < result[1]["path_order"]
        # pf=2 group
        assert result[3]["path_order"] < result[2]["path_order"]


# =============================================================================
# TestEmptyGraph / TestCyclicGraph
# =============================================================================


class TestEmptyGraph:
    def test_empty_returns_empty(self):
        G = nx.DiGraph()
        sections_df = pd.DataFrame(
            columns=[
                "section_id",
                "upstream_junction",
                "downstream_junction",
                "reach_ids",
                "distance",
                "n_reaches",
            ]
        )
        assert compute_path_variables(G, sections_df) == {}


class TestCyclicGraph:
    def test_cycle_does_not_hang(self):
        G = nx.DiGraph()
        for i in range(3):
            G.add_node(i, dist_out=float((2 - i) * 1000), reach_length=1000.0)
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_edge(2, 0)
        sections_df = pd.DataFrame(
            columns=[
                "section_id",
                "upstream_junction",
                "downstream_junction",
                "reach_ids",
                "distance",
                "n_reaches",
            ]
        )
        result = compute_path_variables(G, sections_df)
        assert len(result) == 3
        for rid in [0, 1, 2]:
            assert result[rid]["path_freq"] >= 1


# =============================================================================
# TestPathVarsIntegration (100-reach test DB)
# =============================================================================


class TestPathVarsIntegration:
    """Integration test using the 100-reach test database fixture."""

    @pytest.fixture
    def test_db_path(self):
        return Path(__file__).parent / "fixtures" / "sword_test_minimal.duckdb"

    @pytest.fixture
    def db_conn(self, test_db_path):
        conn = duckdb.connect(str(test_db_path), read_only=True)
        yield conn
        conn.close()

    @pytest.fixture
    def result(self, db_conn):
        topology_df = load_topology(db_conn, "NA")
        reaches_df = load_reaches(db_conn, "NA")
        G = build_reach_graph(topology_df, reaches_df)
        junctions = identify_junctions(G)
        _, sections_df = build_section_graph(G, junctions)
        return compute_path_variables(G, sections_df)

    def test_all_100_reaches_present(self, result):
        assert len(result) == 100

    def test_all_valid_path_freq(self, result):
        for rid, vals in result.items():
            assert vals["path_freq"] > 0, f"reach {rid} pf={vals['path_freq']}"

    def test_all_valid_stream_order(self, result):
        for rid, vals in result.items():
            assert vals["stream_order"] > 0, f"reach {rid} so={vals['stream_order']}"

    def test_all_have_path_segs(self, result):
        for rid, vals in result.items():
            assert vals["path_segs"] > 0, f"reach {rid} ps={vals['path_segs']}"

    def test_all_have_path_order(self, result):
        for rid, vals in result.items():
            assert vals["path_order"] > 0, f"reach {rid} po={vals['path_order']}"

    def test_linear_chain_all_pf_one(self, result):
        for rid, vals in result.items():
            assert vals["path_freq"] == 1, f"reach {rid} pf={vals['path_freq']}"

    def test_monotonicity(self, db_conn, result):
        topo = load_topology(db_conn, "NA")
        down_rows = topo[topo["direction"] == "down"]
        for _, row in down_rows.iterrows():
            rid = int(row["reach_id"])
            nid = int(row["neighbor_reach_id"])
            pf_up = result[rid]["path_freq"]
            pf_dn = result[nid]["path_freq"]
            if pf_up > 0 and pf_dn > 0:
                assert pf_dn >= pf_up, f"T002: {rid}(pf={pf_up})->{nid}(pf={pf_dn})"

    def test_headwaters_have_pf_ge_one(self, result, db_conn):
        reaches_df = load_reaches(db_conn, "NA")
        headwaters = reaches_df[reaches_df["n_rch_up"] == 0]["reach_id"].tolist()
        for hw in headwaters:
            hw = int(hw)
            assert result[hw]["path_freq"] >= 1, (
                f"T010: HW {hw} pf={result[hw]['path_freq']}"
            )

    def test_stream_order_formula_consistency(self, result):
        for rid, vals in result.items():
            pf = vals["path_freq"]
            so = vals["stream_order"]
            if pf > 0:
                expected = int(round(math.log(pf))) + 1
                assert so == expected, (
                    f"reach {rid}: pf={pf}, so={so}, expected={expected}"
                )
