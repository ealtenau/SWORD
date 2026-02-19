"""
Tests for topology count and dist_out monotonicity fix functions,
and the post-write lint gate.

Creates in-memory DuckDB databases with known-bad data, runs the fix
functions, and asserts correctness.
"""

import pytest
import duckdb

from src.updates.sword_v17c_pipeline.v17c_pipeline import (
    fix_dist_out_monotonicity,
    fix_topology_counts,
    run_lint_gate,
)

pytestmark = [pytest.mark.topology, pytest.mark.unit]


# =============================================================================
# Helpers
# =============================================================================


def _create_schema(conn):
    """Create minimal reaches + reach_topology tables."""
    conn.execute("""
        CREATE TABLE reaches (
            reach_id BIGINT,
            region VARCHAR,
            n_rch_up INTEGER,
            n_rch_down INTEGER,
            end_reach INTEGER,
            dist_out DOUBLE,
            reach_length DOUBLE,
            river_name VARCHAR DEFAULT '',
            x DOUBLE DEFAULT 0,
            y DOUBLE DEFAULT 0,
            width DOUBLE DEFAULT 100,
            slope DOUBLE DEFAULT 0.001,
            facc DOUBLE DEFAULT 1000,
            path_freq INTEGER DEFAULT 1,
            stream_order INTEGER DEFAULT 1,
            lakeflag INTEGER DEFAULT 0,
            trib_flag INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE reach_topology (
            reach_id BIGINT,
            region VARCHAR,
            neighbor_reach_id BIGINT,
            direction VARCHAR,
            neighbor_rank INTEGER DEFAULT 0
        )
    """)


# =============================================================================
# fix_topology_counts tests
# =============================================================================


class TestFixTopologyCounts:
    """Tests for fix_topology_counts()."""

    def test_corrects_stale_upstream_count(self):
        """n_rch_up=1 but actual upstream neighbors=0 → fixes to 0."""
        conn = duckdb.connect()
        _create_schema(conn)

        # Reach 100: claims n_rch_up=1, but has NO upstream in topology
        conn.execute("""
            INSERT INTO reaches (reach_id, region, n_rch_up, n_rch_down, end_reach,
                                 dist_out, reach_length)
            VALUES (100, 'NA', 1, 1, 0, 5000, 1000),
                   (200, 'NA', 0, 1, 1, 6000, 1000)
        """)
        # Only downstream topology for reach 100
        conn.execute("""
            INSERT INTO reach_topology (reach_id, region, neighbor_reach_id, direction)
            VALUES (100, 'NA', 200, 'down'),
                   (200, 'NA', 100, 'up')
        """)

        n_fixed = fix_topology_counts(conn, "NA")

        result = conn.execute(
            "SELECT n_rch_up, n_rch_down FROM reaches WHERE reach_id = 100"
        ).fetchone()
        # reach 100 should now have n_rch_up=1 (200 points up to it)
        # Wait — 200 has (200, NA, 100, 'up') meaning 200's upstream is 100.
        # reach 100 has (100, NA, 200, 'down') meaning 100's downstream is 200.
        # So actual_up for 100 = count of topology rows with reach_id=100, dir='up' = 0
        # actual_down for 100 = count with reach_id=100, dir='down' = 1
        assert result[0] == 0  # n_rch_up fixed from 1 → 0
        assert result[1] == 1  # n_rch_down stays 1
        assert n_fixed > 0
        conn.close()

    def test_corrects_stale_downstream_count(self):
        """n_rch_down=2 but actual=1 → fixes to 1."""
        conn = duckdb.connect()
        _create_schema(conn)

        conn.execute("""
            INSERT INTO reaches (reach_id, region, n_rch_up, n_rch_down, end_reach,
                                 dist_out, reach_length)
            VALUES (100, 'NA', 0, 2, 1, 5000, 1000),
                   (200, 'NA', 1, 0, 2, 2000, 1000)
        """)
        conn.execute("""
            INSERT INTO reach_topology (reach_id, region, neighbor_reach_id, direction)
            VALUES (100, 'NA', 200, 'down'),
                   (200, 'NA', 100, 'up')
        """)

        fix_topology_counts(conn, "NA")

        result = conn.execute(
            "SELECT n_rch_down FROM reaches WHERE reach_id = 100"
        ).fetchone()
        assert result[0] == 1
        conn.close()

    def test_fixes_end_reach_classification(self):
        """Headwater/outlet/junction/middle classifications updated."""
        conn = duckdb.connect()
        _create_schema(conn)

        # 3 reaches: headwater → middle → outlet
        conn.execute("""
            INSERT INTO reaches (reach_id, region, n_rch_up, n_rch_down, end_reach,
                                 dist_out, reach_length)
            VALUES (100, 'NA', 0, 1, 0, 5000, 1000),
                   (200, 'NA', 1, 1, 0, 3000, 1000),
                   (300, 'NA', 1, 0, 0, 1000, 1000)
        """)
        conn.execute("""
            INSERT INTO reach_topology (reach_id, region, neighbor_reach_id, direction)
            VALUES (100, 'NA', 200, 'down'),
                   (200, 'NA', 100, 'up'),
                   (200, 'NA', 300, 'down'),
                   (300, 'NA', 200, 'up')
        """)

        fix_topology_counts(conn, "NA")

        rows = conn.execute(
            "SELECT reach_id, end_reach FROM reaches ORDER BY reach_id"
        ).fetchall()
        result = {r[0]: r[1] for r in rows}
        assert result[100] == 1  # headwater
        assert result[200] == 0  # middle
        assert result[300] == 2  # outlet
        conn.close()

    def test_junction_end_reach(self):
        """Junction (n_up>1) gets end_reach=3."""
        conn = duckdb.connect()
        _create_schema(conn)

        # Reach 300 has 2 upstream reaches
        conn.execute("""
            INSERT INTO reaches (reach_id, region, n_rch_up, n_rch_down, end_reach,
                                 dist_out, reach_length)
            VALUES (100, 'NA', 0, 1, 1, 8000, 1000),
                   (200, 'NA', 0, 1, 1, 7000, 1000),
                   (300, 'NA', 2, 1, 0, 5000, 1000),
                   (400, 'NA', 1, 0, 2, 2000, 1000)
        """)
        conn.execute("""
            INSERT INTO reach_topology (reach_id, region, neighbor_reach_id, direction)
            VALUES (100, 'NA', 300, 'down'),
                   (200, 'NA', 300, 'down'),
                   (300, 'NA', 100, 'up'),
                   (300, 'NA', 200, 'up'),
                   (300, 'NA', 400, 'down'),
                   (400, 'NA', 300, 'up')
        """)

        fix_topology_counts(conn, "NA")

        end_reach = conn.execute(
            "SELECT end_reach FROM reaches WHERE reach_id = 300"
        ).fetchone()[0]
        assert end_reach == 3  # junction
        conn.close()

    def test_no_changes_when_correct(self):
        """Returns 0 when counts already match."""
        conn = duckdb.connect()
        _create_schema(conn)

        conn.execute("""
            INSERT INTO reaches (reach_id, region, n_rch_up, n_rch_down, end_reach,
                                 dist_out, reach_length)
            VALUES (100, 'NA', 0, 1, 1, 5000, 1000),
                   (200, 'NA', 1, 0, 2, 2000, 1000)
        """)
        conn.execute("""
            INSERT INTO reach_topology (reach_id, region, neighbor_reach_id, direction)
            VALUES (100, 'NA', 200, 'down'),
                   (200, 'NA', 100, 'up')
        """)

        n_fixed = fix_topology_counts(conn, "NA")
        assert n_fixed == 0
        conn.close()

    def test_orphan_reach_counts_zeroed(self):
        """Reach with no topology entries but nonzero counts gets zeroed."""
        conn = duckdb.connect()
        _create_schema(conn)

        # Reach claims n_rch_up=1 but has no topology entries
        conn.execute("""
            INSERT INTO reaches (reach_id, region, n_rch_up, n_rch_down, end_reach,
                                 dist_out, reach_length)
            VALUES (100, 'NA', 1, 1, 0, 5000, 1000)
        """)

        n_fixed = fix_topology_counts(conn, "NA")
        result = conn.execute(
            "SELECT n_rch_up, n_rch_down FROM reaches WHERE reach_id = 100"
        ).fetchone()
        assert result[0] == 0
        assert result[1] == 0
        assert n_fixed > 0
        conn.close()


# =============================================================================
# fix_dist_out_monotonicity tests
# =============================================================================


class TestFixDistOutMonotonicity:
    """Tests for fix_dist_out_monotonicity()."""

    def test_fixes_simple_violation(self):
        """Upstream dist_out < downstream dist_out → recalculated."""
        conn = duckdb.connect()
        _create_schema(conn)

        # Reach 100 (upstream) has dist_out=3000
        # Reach 200 (downstream) has dist_out=5000 — VIOLATION
        # Correct: dist_out(100) = reach_length(100) + max(downstream dist_out)
        #        = 1000 + 5000 = 6000
        conn.execute("""
            INSERT INTO reaches (reach_id, region, n_rch_up, n_rch_down, end_reach,
                                 dist_out, reach_length)
            VALUES (100, 'NA', 0, 1, 1, 3000, 1000),
                   (200, 'NA', 1, 0, 2, 5000, 1000)
        """)
        conn.execute("""
            INSERT INTO reach_topology (reach_id, region, neighbor_reach_id, direction)
            VALUES (100, 'NA', 200, 'down'),
                   (200, 'NA', 100, 'up')
        """)

        n_fixed = fix_dist_out_monotonicity(conn, "NA")

        dist_out_100 = conn.execute(
            "SELECT dist_out FROM reaches WHERE reach_id = 100"
        ).fetchone()[0]
        # Should be 1000 (reach_length) + 5000 (max downstream) = 6000
        assert dist_out_100 == 6000.0
        assert n_fixed == 1
        conn.close()

    def test_cascading_violations(self):
        """Chain of violations fixed iteratively."""
        conn = duckdb.connect()
        _create_schema(conn)

        # Chain: 100 → 200 → 300 (outlet)
        # 300: dist_out=5000 (outlet, correct)
        # 200: dist_out=3000 (violation: downstream 300 has 5000 > 3000)
        # 100: dist_out=2000 (violation after 200 is fixed)
        conn.execute("""
            INSERT INTO reaches (reach_id, region, n_rch_up, n_rch_down, end_reach,
                                 dist_out, reach_length)
            VALUES (100, 'NA', 0, 1, 1, 2000, 1000),
                   (200, 'NA', 1, 1, 0, 3000, 1000),
                   (300, 'NA', 1, 0, 2, 5000, 1000)
        """)
        conn.execute("""
            INSERT INTO reach_topology (reach_id, region, neighbor_reach_id, direction)
            VALUES (100, 'NA', 200, 'down'),
                   (200, 'NA', 100, 'up'),
                   (200, 'NA', 300, 'down'),
                   (300, 'NA', 200, 'up')
        """)

        n_fixed = fix_dist_out_monotonicity(conn, "NA")

        rows = conn.execute(
            "SELECT reach_id, dist_out FROM reaches ORDER BY reach_id"
        ).fetchall()
        dist = {r[0]: r[1] for r in rows}
        # 300 stays 5000
        # 200 = 1000 + 5000 = 6000
        # 100 = 1000 + 6000 = 7000
        assert dist[300] == 5000.0
        assert dist[200] == 6000.0
        assert dist[100] == 7000.0
        assert n_fixed >= 2  # at least 2 violations fixed
        conn.close()

    def test_no_violations_returns_zero(self):
        """Correct dist_out ordering → 0 fixes."""
        conn = duckdb.connect()
        _create_schema(conn)

        conn.execute("""
            INSERT INTO reaches (reach_id, region, n_rch_up, n_rch_down, end_reach,
                                 dist_out, reach_length)
            VALUES (100, 'NA', 0, 1, 1, 6000, 1000),
                   (200, 'NA', 1, 0, 2, 2000, 1000)
        """)
        conn.execute("""
            INSERT INTO reach_topology (reach_id, region, neighbor_reach_id, direction)
            VALUES (100, 'NA', 200, 'down'),
                   (200, 'NA', 100, 'up')
        """)

        n_fixed = fix_dist_out_monotonicity(conn, "NA")
        assert n_fixed == 0
        conn.close()

    def test_within_tolerance_not_fixed(self):
        """Violation within tolerance is ignored."""
        conn = duckdb.connect()
        _create_schema(conn)

        # dist_out difference of 50m < default tolerance 100m
        conn.execute("""
            INSERT INTO reaches (reach_id, region, n_rch_up, n_rch_down, end_reach,
                                 dist_out, reach_length)
            VALUES (100, 'NA', 0, 1, 1, 4960, 1000),
                   (200, 'NA', 1, 0, 2, 5000, 1000)
        """)
        conn.execute("""
            INSERT INTO reach_topology (reach_id, region, neighbor_reach_id, direction)
            VALUES (100, 'NA', 200, 'down'),
                   (200, 'NA', 100, 'up')
        """)

        n_fixed = fix_dist_out_monotonicity(conn, "NA")
        assert n_fixed == 0
        conn.close()

    def test_skips_sentinel_values(self):
        """Reaches with dist_out=-9999 or 0 are skipped."""
        conn = duckdb.connect()
        _create_schema(conn)

        conn.execute("""
            INSERT INTO reaches (reach_id, region, n_rch_up, n_rch_down, end_reach,
                                 dist_out, reach_length)
            VALUES (100, 'NA', 0, 1, 1, -9999, 1000),
                   (200, 'NA', 1, 0, 2, 5000, 1000)
        """)
        conn.execute("""
            INSERT INTO reach_topology (reach_id, region, neighbor_reach_id, direction)
            VALUES (100, 'NA', 200, 'down'),
                   (200, 'NA', 100, 'up')
        """)

        n_fixed = fix_dist_out_monotonicity(conn, "NA")
        # -9999 is excluded from checks, so no violation detected
        assert n_fixed == 0
        conn.close()

    def test_bifurcation_uses_max_downstream(self):
        """At bifurcation, uses MAX downstream dist_out."""
        conn = duckdb.connect()
        _create_schema(conn)

        # Reach 100 → (200, 300)
        # 200 has dist_out=3000, 300 has dist_out=5000
        # 100 has dist_out=4000 — violation because max(3000,5000)=5000 > 4000
        conn.execute("""
            INSERT INTO reaches (reach_id, region, n_rch_up, n_rch_down, end_reach,
                                 dist_out, reach_length)
            VALUES (100, 'NA', 0, 2, 1, 4000, 1000),
                   (200, 'NA', 1, 0, 2, 3000, 1000),
                   (300, 'NA', 1, 0, 2, 5000, 1000)
        """)
        conn.execute("""
            INSERT INTO reach_topology (reach_id, region, neighbor_reach_id, direction)
            VALUES (100, 'NA', 200, 'down'),
                   (100, 'NA', 300, 'down'),
                   (200, 'NA', 100, 'up'),
                   (300, 'NA', 100, 'up')
        """)

        n_fixed = fix_dist_out_monotonicity(conn, "NA")

        dist_out_100 = conn.execute(
            "SELECT dist_out FROM reaches WHERE reach_id = 100"
        ).fetchone()[0]
        # Should be 1000 + max(3000, 5000) = 6000
        assert dist_out_100 == 6000.0
        assert n_fixed == 1
        conn.close()


# =============================================================================
# Lint gate helpers
# =============================================================================


def _create_lint_db(db_path, reaches_data, topology_data):
    """Create a file-based DuckDB with reaches + reach_topology for lint tests.

    Parameters
    ----------
    db_path : str or Path
        File path for the DuckDB database.
    reaches_data : list of tuples
        (reach_id, region, n_rch_up, n_rch_down, end_reach, dist_out,
         reach_length, river_name, x, y, width, slope, facc, path_freq,
         stream_order, lakeflag, trib_flag)
    topology_data : list of tuples
        (reach_id, region, neighbor_reach_id, direction, neighbor_rank)
    """
    conn = duckdb.connect(str(db_path))
    _create_schema(conn)
    if reaches_data:
        conn.executemany(
            """
            INSERT INTO reaches
                (reach_id, region, n_rch_up, n_rch_down, end_reach,
                 dist_out, reach_length, river_name, x, y, width, slope,
                 facc, path_freq, stream_order, lakeflag, trib_flag)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            reaches_data,
        )
    if topology_data:
        conn.executemany(
            """
            INSERT INTO reach_topology
                (reach_id, region, neighbor_reach_id, direction, neighbor_rank)
            VALUES (?,?,?,?,?)
            """,
            topology_data,
        )
    conn.close()


# =============================================================================
# run_lint_gate tests
# =============================================================================


class TestRunLintGate:
    """Tests for run_lint_gate()."""

    def test_lint_gate_passes_clean_data(self, tmp_path):
        """Clean data passes T005 + T008 checks without raising."""
        db_path = tmp_path / "clean.duckdb"
        # Two reaches with correct counts and positive dist_out
        _create_lint_db(
            db_path,
            reaches_data=[
                # (id, region, n_up, n_down, end, dist_out, length,
                #  name, x, y, width, slope, facc, pf, so, lake, trib)
                (
                    100,
                    "NA",
                    0,
                    1,
                    1,
                    6000,
                    1000,
                    "",
                    0,
                    0,
                    100,
                    0.001,
                    1000,
                    1,
                    1,
                    0,
                    0,
                ),
                (
                    200,
                    "NA",
                    1,
                    0,
                    2,
                    2000,
                    1000,
                    "",
                    0,
                    0,
                    100,
                    0.001,
                    2000,
                    1,
                    1,
                    0,
                    0,
                ),
            ],
            topology_data=[
                (100, "NA", 200, "down", 0),
                (200, "NA", 100, "up", 0),
            ],
        )

        result = run_lint_gate(str(db_path), "NA", checks=["T005", "T008"])

        assert isinstance(result, dict)
        assert "T005" in result
        assert "T008" in result
        assert result["T005"]["passed"] is True
        assert result["T008"]["passed"] is True

    def test_lint_gate_fails_on_bad_data(self, tmp_path):
        """Negative dist_out triggers T008 failure → RuntimeError."""
        db_path = tmp_path / "bad.duckdb"
        _create_lint_db(
            db_path,
            reaches_data=[
                (100, "NA", 0, 1, 1, -5, 1000, "", 0, 0, 100, 0.001, 1000, 1, 1, 0, 0),
                (
                    200,
                    "NA",
                    1,
                    0,
                    2,
                    2000,
                    1000,
                    "",
                    0,
                    0,
                    100,
                    0.001,
                    2000,
                    1,
                    1,
                    0,
                    0,
                ),
            ],
            topology_data=[
                (100, "NA", 200, "down", 0),
                (200, "NA", 100, "up", 0),
            ],
        )

        with pytest.raises(RuntimeError, match="Lint gate FAILED"):
            run_lint_gate(str(db_path), "NA", checks=["T008"])

    def test_lint_gate_respects_checks_filter(self, tmp_path):
        """Only requested checks run; T008 failure ignored when not selected."""
        db_path = tmp_path / "filter.duckdb"
        # Data fails T008 (negative dist_out) but passes T005 (correct counts)
        _create_lint_db(
            db_path,
            reaches_data=[
                (100, "NA", 0, 1, 1, -5, 1000, "", 0, 0, 100, 0.001, 1000, 1, 1, 0, 0),
                (
                    200,
                    "NA",
                    1,
                    0,
                    2,
                    2000,
                    1000,
                    "",
                    0,
                    0,
                    100,
                    0.001,
                    2000,
                    1,
                    1,
                    0,
                    0,
                ),
            ],
            topology_data=[
                (100, "NA", 200, "down", 0),
                (200, "NA", 100, "up", 0),
            ],
        )

        # Only run T005 — should pass despite T008-failing data
        result = run_lint_gate(str(db_path), "NA", checks=["T005"])

        assert "T005" in result
        assert "T008" not in result
        assert result["T005"]["passed"] is True
