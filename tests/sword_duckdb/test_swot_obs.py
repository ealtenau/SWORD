# -*- coding: utf-8 -*-
"""Tests for SWOT observation pipeline (#29, #30, #31)."""

import os
import sys

import duckdb
import numpy as np
import pandas as pd
import pytest

main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, main_dir)

from src.sword_duckdb.schema import add_swot_obs_columns
from src.sword_duckdb.swot_filters import (
    SENTINEL,
    SLOPE_REF_UNCERTAINTY,
    build_node_filter_sql,
    build_reach_filter_sql,
)

pytestmark = [pytest.mark.db, pytest.mark.unit]

_PCTS = ["p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90"]
NODE_SWOT_COLS = (
    [f"wse_obs_{s}" for s in _PCTS]
    + ["wse_obs_range", "wse_obs_mad"]
    + [f"width_obs_{s}" for s in _PCTS]
    + ["width_obs_range", "width_obs_mad"]
    + ["n_obs"]
)
REACH_SWOT_COLS = (
    NODE_SWOT_COLS[:-1]  # without n_obs
    + [f"slope_obs_{s}" for s in _PCTS]
    + ["slope_obs_range", "slope_obs_mad"]
    + [
        "slope_obs_adj",
        "slope_obs_slopeF",
        "slope_obs_reliable",
        "slope_obs_quality",
        "n_obs",
    ]
)
LEGACY_COLS = [
    f"{v}_obs_{s}" for v in ("wse", "width", "slope") for s in ("mean", "median", "std")
]


def _make_db(path: str) -> duckdb.DuckDBPyConnection:
    """Create a minimal DuckDB with all tables SWORD needs to load."""
    conn = duckdb.connect(path)
    conn.execute("""CREATE TABLE centerlines (
        cl_id BIGINT, region VARCHAR, x DOUBLE, y DOUBLE,
        PRIMARY KEY (cl_id, region))""")
    conn.execute("""CREATE TABLE nodes (
        node_id BIGINT, region VARCHAR, version VARCHAR DEFAULT '17c',
        PRIMARY KEY (node_id, region))""")
    conn.execute("""CREATE TABLE reaches (
        reach_id BIGINT PRIMARY KEY, region VARCHAR, version VARCHAR DEFAULT '17c')""")
    conn.execute("""CREATE TABLE reach_topology (
        reach_id BIGINT, direction VARCHAR, neighbor_rank INTEGER,
        neighbor_reach_id BIGINT, topology_suspect BOOLEAN DEFAULT FALSE,
        topology_approved BOOLEAN DEFAULT FALSE)""")
    conn.execute("""CREATE TABLE reach_swot_orbits (
        reach_id BIGINT, orbit_number INTEGER, orbit_cycle INTEGER)""")
    conn.execute("""CREATE TABLE reach_ice_flags (
        reach_id BIGINT, day_of_year INTEGER, ice_flag INTEGER)""")
    conn.execute("""CREATE TABLE sword_versions (
        version VARCHAR, region VARCHAR, created_at TIMESTAMP)""")
    conn.execute("INSERT INTO nodes VALUES (1001,'NA','17c'),(1002,'NA','17c')")
    conn.execute("INSERT INTO reaches VALUES (100,'NA','17c'),(200,'NA','17c')")
    return conn


def _node_parquet(path, n=20, nid=1001):
    rng = np.random.default_rng(42)
    pd.DataFrame(
        {
            "node_id": [nid] * n,
            "wse": 100.0 + rng.normal(0, 2, n),
            "width": 500.0 + rng.normal(0, 50, n),
            "wse_q": np.zeros(n, dtype=int),
            "dark_frac": np.full(n, 0.1),
            "xtrk_dist": np.full(n, 30000.0),
            "xovr_cal_q": np.zeros(n, dtype=int),
            "ice_clim_f": np.zeros(n, dtype=int),
            "time_str": ["2024-01-01T00:00:00"] * n,
        }
    ).to_parquet(path, index=False)


def _reach_parquet(path, n=20, rid=100, slope=0.001):
    rng = np.random.default_rng(42)
    pd.DataFrame(
        {
            "reach_id": [rid] * n,
            "wse": 100.0 + rng.normal(0, 2, n),
            "width": 500.0 + rng.normal(0, 50, n),
            "slope": [slope] * n if isinstance(slope, (int, float)) else slope[:n],
            "n_good_nod": rng.integers(3, 20, n),
            "reach_q": np.zeros(n, dtype=int),
            "dark_frac": np.full(n, 0.1),
            "xovr_cal_q": np.zeros(n, dtype=int),
            "ice_clim_f": np.zeros(n, dtype=int),
        }
    ).to_parquet(path, index=False)


# === swot_filters tests ===


class TestSwotFilters:
    def test_constants(self):
        assert SLOPE_REF_UNCERTAINTY == pytest.approx(0.000017)
        assert SENTINEL == -999_999_999_999

    def test_node_filter_wse_col(self):
        _, wse = build_node_filter_sql({"wse", "width"})
        assert wse == "wse"
        _, wse = build_node_filter_sql({"wse_sm", "width"})
        assert wse == "wse_sm"

    def test_node_filter_includes_sentinel(self):
        where, _ = build_node_filter_sql({"wse", "width"})
        assert str(int(SENTINEL)) in where

    def test_reach_filter_slope_bounds(self):
        where = build_reach_filter_sql({"wse", "width", "slope"})
        assert "slope > -1" in where
        assert "slope < 1" in where
        assert "1e10" not in where

    def test_reach_filter_quality_cols(self):
        where = build_reach_filter_sql(
            {"wse", "width", "slope", "reach_q", "dark_frac"}
        )
        assert "reach_q" in where
        assert "dark_frac" in where


# === add_swot_obs_columns tests ===


class TestAddSwotObsColumns:
    def test_adds_columns(self, tmp_path):
        conn = _make_db(str(tmp_path / "t.duckdb"))
        assert add_swot_obs_columns(conn) is True
        ncols = {
            r[0]
            for r in conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name='nodes'"
            ).fetchall()
        }
        for c in NODE_SWOT_COLS:
            assert c in ncols, f"Missing: {c}"
        conn.close()

    def test_idempotent(self, tmp_path):
        conn = _make_db(str(tmp_path / "t.duckdb"))
        add_swot_obs_columns(conn)
        assert add_swot_obs_columns(conn) is False
        conn.close()

    def test_drops_legacy(self, tmp_path):
        conn = _make_db(str(tmp_path / "t.duckdb"))
        for c in LEGACY_COLS:
            try:
                conn.execute(f"ALTER TABLE reaches ADD COLUMN {c} DOUBLE")
            except Exception:
                pass
        add_swot_obs_columns(conn)
        rcols = {
            r[0]
            for r in conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name='reaches'"
            ).fetchall()
        }
        for c in LEGACY_COLS:
            assert c not in rcols, f"Legacy not dropped: {c}"
        conn.close()

    def test_column_types(self, tmp_path):
        conn = _make_db(str(tmp_path / "t.duckdb"))
        add_swot_obs_columns(conn)
        types = dict(
            conn.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name='reaches'"
            ).fetchall()
        )
        assert types["slope_obs_reliable"] == "BOOLEAN"
        assert types["slope_obs_quality"] == "VARCHAR"
        assert types["n_obs"] == "INTEGER"
        assert types["wse_obs_p50"] == "DOUBLE"
        conn.close()


# === Node aggregation tests ===


def _run_node_agg(tmp_path, parquet_fn):
    """Set up DB + parquet, run node aggregation, return (result, conn)."""
    from src.sword_duckdb.workflow import SWORDWorkflow

    db = str(tmp_path / "sword.duckdb")
    conn = _make_db(db)
    add_swot_obs_columns(conn)
    ndir = tmp_path / "pq" / "nodes"
    ndir.mkdir(parents=True)
    parquet_fn(ndir)
    # Call the unbound method with a bare workflow (no load needed)
    wf = SWORDWorkflow.__new__(SWORDWorkflow)
    res = wf._aggregate_node_observations(conn, ndir)
    return res, conn


def _run_reach_agg(tmp_path, parquet_fn):
    """Set up DB + parquet, run reach aggregation, return (result, conn)."""
    from src.sword_duckdb.workflow import SWORDWorkflow

    db = str(tmp_path / "sword.duckdb")
    conn = _make_db(db)
    add_swot_obs_columns(conn)
    rdir = tmp_path / "pq" / "reaches"
    rdir.mkdir(parents=True)
    parquet_fn(rdir)
    wf = SWORDWorkflow.__new__(SWORDWorkflow)
    res = wf._aggregate_reach_observations(conn, rdir)
    return res, conn


class TestNodeAggregation:
    def test_basic(self, tmp_path):
        def pq(d):
            _node_parquet(str(d / "SWOT_t.parquet"), n=20, nid=1001)

        res, c = _run_node_agg(tmp_path, pq)
        assert res["node_total_obs"] == 20
        row = c.execute(
            "SELECT wse_obs_p50, wse_obs_mad, n_obs FROM nodes WHERE node_id=1001"
        ).fetchone()
        assert row[0] is not None and row[1] is not None and row[2] == 20
        pcts = c.execute(
            "SELECT wse_obs_p10, wse_obs_p50, wse_obs_p90 FROM nodes WHERE node_id=1001"
        ).fetchone()
        assert pcts[0] <= pcts[1] <= pcts[2]
        c.close()

    def test_sentinel_filtered(self, tmp_path):
        def pq(d):
            pd.DataFrame(
                {"node_id": [1001] * 5, "wse": [SENTINEL] * 5, "width": [500.0] * 5}
            ).to_parquet(str(d / "SWOT_s.parquet"), index=False)

        res, c = _run_node_agg(tmp_path, pq)
        assert res["node_total_obs"] == 0
        c.close()


# === Reach aggregation tests ===


class TestReachAggregation:
    def test_basic(self, tmp_path):
        def pq(d):
            _reach_parquet(str(d / "SWOT_t.parquet"), n=20, rid=100)

        res, c = _run_reach_agg(tmp_path, pq)
        assert res["reach_total_obs"] == 20
        row = c.execute(
            "SELECT slope_obs_p50, slope_obs_adj, slope_obs_slopeF, "
            "slope_obs_reliable, slope_obs_quality, n_obs FROM reaches WHERE reach_id=100"
        ).fetchone()
        p50, adj, slopeF, reliable, quality, n = row
        assert p50 is not None
        assert adj >= 0
        assert -1 <= slopeF <= 1
        assert quality in (
            "reliable",
            "below_ref_uncertainty",
            "high_uncertainty",
            "negative",
        )
        assert n == 20
        c.close()

    def test_reliable_slope(self, tmp_path):
        def pq(d):
            _reach_parquet(str(d / "SWOT_p.parquet"), n=30, rid=100, slope=0.001)

        res, c = _run_reach_agg(tmp_path, pq)
        row = c.execute(
            "SELECT slope_obs_p50, slope_obs_slopeF, slope_obs_reliable, slope_obs_quality "
            "FROM reaches WHERE reach_id=100"
        ).fetchone()
        assert row[0] == pytest.approx(0.001)
        assert row[1] == pytest.approx(1.0)
        assert row[2] is True
        assert row[3] == "reliable"
        c.close()

    def test_single_obs(self, tmp_path):
        def pq(d):
            _reach_parquet(str(d / "SWOT_1.parquet"), n=1, rid=100, slope=0.001)

        res, c = _run_reach_agg(tmp_path, pq)
        assert res["reach_total_obs"] == 1
        row = c.execute(
            "SELECT slope_obs_range, slope_obs_mad, n_obs FROM reaches WHERE reach_id=100"
        ).fetchone()
        assert row[0] == pytest.approx(0.0)
        assert row[1] == pytest.approx(0.0)
        assert row[2] == 1
        c.close()

    def test_all_filtered(self, tmp_path):
        def pq(d):
            pd.DataFrame(
                {
                    "reach_id": [100] * 5,
                    "wse": [SENTINEL] * 5,
                    "width": [SENTINEL] * 5,
                    "slope": [SENTINEL] * 5,
                    "n_good_nod": [5] * 5,
                }
            ).to_parquet(str(d / "SWOT_b.parquet"), index=False)

        res, c = _run_reach_agg(tmp_path, pq)
        assert res["reach_total_obs"] == 0
        c.close()

    def test_percentile_ordering(self, tmp_path):
        def pq(d):
            _reach_parquet(str(d / "SWOT_o.parquet"), n=50, rid=100)

        res, c = _run_reach_agg(tmp_path, pq)
        for var in ("wse", "width", "slope"):
            cols = ", ".join(f"{var}_obs_{p}" for p in _PCTS)
            row = c.execute(f"SELECT {cols} FROM reaches WHERE reach_id=100").fetchone()
            for i in range(len(row) - 1):
                assert row[i] <= row[i + 1], f"{var} p{(i + 1) * 10} > p{(i + 2) * 10}"
        c.close()
