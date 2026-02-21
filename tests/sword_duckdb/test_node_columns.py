"""Tests for dn_node_id, up_node_id, and node_order columns (issue #149)."""

import shutil
import subprocess
import sys

import duckdb
import pytest

SCRIPT = "scripts/maintenance/add_node_columns.py"


def _run_script(db_path, *extra_args):
    """Run the add_node_columns script and return the result."""
    return subprocess.run(
        [sys.executable, SCRIPT, "--db", str(db_path), *extra_args],
        capture_output=True,
        text=True,
    )


def _setup_db(ensure_test_db, tmp_path):
    """Copy test DB and run the script on it."""
    db = tmp_path / "test.duckdb"
    shutil.copy2(ensure_test_db, db)
    result = _run_script(db)
    assert result.returncode == 0, result.stderr
    return db


@pytest.mark.db
class TestNodeBoundaryColumns:
    """Test the add_node_columns maintenance script."""

    def test_script_adds_columns_and_populates(self, ensure_test_db, tmp_path):
        """Script adds columns and populates them correctly."""
        db = _setup_db(ensure_test_db, tmp_path)

        con = duckdb.connect(str(db), read_only=True)
        try:
            reach_cols = {
                r[0]
                for r in con.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name='reaches'"
                ).fetchall()
            }
            assert "dn_node_id" in reach_cols
            assert "up_node_id" in reach_cols

            node_cols = {
                r[0]
                for r in con.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name='nodes'"
                ).fetchall()
            }
            assert "node_order" in node_cols

            assert (
                con.execute(
                    "SELECT COUNT(*) FROM reaches "
                    "WHERE dn_node_id IS NULL AND n_nodes > 0"
                ).fetchone()[0]
                == 0
            )
            assert (
                con.execute(
                    "SELECT COUNT(*) FROM nodes WHERE node_order IS NULL"
                ).fetchone()[0]
                == 0
            )
        finally:
            con.close()

    def test_node_order_range_matches_n_nodes(self, ensure_test_db, tmp_path):
        """max(node_order) per reach equals n_nodes."""
        db = _setup_db(ensure_test_db, tmp_path)

        con = duckdb.connect(str(db), read_only=True)
        try:
            mismatches = con.execute("""
                SELECT COUNT(*) FROM (
                    SELECT n.reach_id, MAX(n.node_order) AS mx, r.n_nodes
                    FROM nodes n JOIN reaches r ON r.reach_id = n.reach_id
                    GROUP BY n.reach_id, r.n_nodes
                    HAVING MAX(n.node_order) != r.n_nodes
                )
            """).fetchone()[0]
            assert mismatches == 0
        finally:
            con.close()

    def test_node_order_1_is_downstream(self, ensure_test_db, tmp_path):
        """node_order=1 has the lowest dist_out in each reach."""
        db = _setup_db(ensure_test_db, tmp_path)

        con = duckdb.connect(str(db), read_only=True)
        try:
            bad = con.execute("""
                WITH first_nodes AS (
                    SELECT reach_id, dist_out AS first_do
                    FROM nodes WHERE node_order = 1
                ),
                min_do AS (
                    SELECT reach_id, MIN(dist_out) AS min_do
                    FROM nodes GROUP BY reach_id
                )
                SELECT COUNT(*) FROM first_nodes f
                JOIN min_do m ON m.reach_id = f.reach_id
                WHERE f.first_do != m.min_do
            """).fetchone()[0]
            assert bad == 0, f"{bad} reaches where node_order=1 is not downstream"
        finally:
            con.close()

    def test_boundary_nodes_match_dist_out_extremes(self, ensure_test_db, tmp_path):
        """dn_node_id has min dist_out, up_node_id has max dist_out."""
        db = _setup_db(ensure_test_db, tmp_path)

        con = duckdb.connect(str(db), read_only=True)
        try:
            inverted = con.execute("""
                SELECT COUNT(*) FROM reaches r
                JOIN nodes n_dn ON n_dn.node_id = r.dn_node_id
                JOIN nodes n_up ON n_up.node_id = r.up_node_id
                WHERE n_dn.dist_out > n_up.dist_out
            """).fetchone()[0]
            assert inverted == 0
        finally:
            con.close()

    def test_idempotent(self, ensure_test_db, tmp_path):
        """Running the script twice produces the same result."""
        db = tmp_path / "test.duckdb"
        shutil.copy2(ensure_test_db, db)

        for _ in range(2):
            result = _run_script(db)
            assert result.returncode == 0, result.stderr

        con = duckdb.connect(str(db), read_only=True)
        try:
            assert (
                con.execute(
                    "SELECT COUNT(*) FROM nodes WHERE node_order IS NULL"
                ).fetchone()[0]
                == 0
            )
        finally:
            con.close()

    def test_verify_only(self, ensure_test_db, tmp_path):
        """--verify-only passes after script has run."""
        db = _setup_db(ensure_test_db, tmp_path)

        result = _run_script(db, "--verify-only")
        assert result.returncode == 0, result.stdout
