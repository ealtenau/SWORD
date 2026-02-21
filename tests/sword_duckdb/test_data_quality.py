"""Tests for data quality workflow methods.

Ported from src/_legacy/updates/formatting_scripts/.
"""

import pandas as pd
import pytest

pytestmark = [pytest.mark.db, pytest.mark.workflow]


# =========================================================================
# Task 7 — fill_zero_width_nodes
# =========================================================================


class TestFillZeroWidthNodes:
    """Test SWORDWorkflow.fill_zero_width_nodes."""

    def test_dry_run_returns_dataframe(self, temp_workflow):
        result = temp_workflow.fill_zero_width_nodes(region="NA", dry_run=True)
        assert isinstance(result, dict)
        assert "issues" in result
        assert isinstance(result["issues"], pd.DataFrame)

    def test_fills_zero_widths(self, temp_workflow):
        """Set a node width to 0, run fill, verify it gets filled with reach median."""
        conn = temp_workflow._sword.db.conn

        # Find a multi-node reach with positive widths
        row = conn.execute(
            """
            SELECT reach_id FROM nodes
            WHERE region = 'NA' AND width > 0
            GROUP BY reach_id
            HAVING COUNT(*) >= 3
            LIMIT 1
            """
        ).fetchone()
        assert row is not None, "No multi-node reach with positive widths in test DB"
        reach_id = row[0]

        # Get the first node and its expected median
        nodes = conn.execute(
            """
            SELECT node_id, width FROM nodes
            WHERE reach_id = ? AND region = 'NA'
            ORDER BY node_id
            """,
            [reach_id],
        ).fetchall()
        target_node = nodes[0][0]
        other_widths = [w for _, w in nodes[1:] if w > 0]
        import numpy as np

        expected_median = float(np.median(other_widths))

        # RTREE drop → set width to 0 → RTREE recreate
        conn.execute("INSTALL spatial; LOAD spatial;")
        indexes = conn.execute(
            "SELECT index_name, table_name, sql FROM duckdb_indexes() "
            "WHERE sql LIKE '%RTREE%' AND table_name = 'nodes'"
        ).fetchall()
        for idx_name, _, _ in indexes:
            conn.execute(f'DROP INDEX "{idx_name}"')
        conn.execute(
            "UPDATE nodes SET width = 0 WHERE node_id = ? AND region = 'NA'",
            [target_node],
        )
        for _, _, sql in indexes:
            conn.execute(sql)

        # Verify it's zero
        val = conn.execute(
            "SELECT width FROM nodes WHERE node_id = ? AND region = 'NA'",
            [target_node],
        ).fetchone()[0]
        assert val == 0.0

        # Fill
        result = temp_workflow.fill_zero_width_nodes(region="NA")
        assert result["filled_count"] >= 1

        # Verify filled
        new_val = conn.execute(
            "SELECT width FROM nodes WHERE node_id = ? AND region = 'NA'",
            [target_node],
        ).fetchone()[0]
        assert new_val > 0
        assert abs(new_val - expected_median) < 0.01


# =========================================================================
# Task 8 — remove_duplicate_centerline_points
# =========================================================================


class TestRemoveDuplicateCenterlinePoints:
    """Test SWORDWorkflow.remove_duplicate_centerline_points."""

    def test_dry_run_returns_dataframe(self, temp_workflow):
        result = temp_workflow.remove_duplicate_centerline_points(
            region="NA", dry_run=True
        )
        assert isinstance(result, dict)
        assert "duplicates" in result
        assert isinstance(result["duplicates"], pd.DataFrame)

    def test_removes_inserted_duplicate(self, temp_workflow):
        """Insert a duplicate centerline point, then remove it."""
        conn = temp_workflow._sword.db.conn

        # Get an existing centerline point
        row = conn.execute(
            """
            SELECT cl_id, reach_id, region, x, y, node_id, version
            FROM centerlines
            WHERE region = 'NA'
            LIMIT 1
            """
        ).fetchone()
        assert row is not None, "No centerlines in test DB"
        cl_id, reach_id, region, x, y, node_id, version = row

        # Get max cl_id to create a new one
        max_cl = conn.execute("SELECT MAX(cl_id) FROM centerlines").fetchone()[0]
        new_cl_id = max_cl + 1

        # RTREE drop → INSERT duplicate → RTREE recreate
        conn.execute("INSTALL spatial; LOAD spatial;")
        indexes = conn.execute(
            "SELECT index_name, table_name, sql FROM duckdb_indexes() "
            "WHERE sql LIKE '%RTREE%' AND table_name = 'centerlines'"
        ).fetchall()
        for idx_name, _, _ in indexes:
            conn.execute(f'DROP INDEX "{idx_name}"')

        # Insert a duplicate with same reach_id, x, y but different cl_id
        conn.execute(
            """
            INSERT INTO centerlines (cl_id, reach_id, region, x, y, node_id, version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [new_cl_id, reach_id, region, x, y, node_id, version],
        )
        for _, _, sql in indexes:
            conn.execute(sql)

        # Dry run should find it
        result = temp_workflow.remove_duplicate_centerline_points(
            region="NA", dry_run=True
        )
        assert result["duplicate_count"] >= 1

        # Actually remove
        result = temp_workflow.remove_duplicate_centerline_points(region="NA")
        assert result["removed_count"] >= 1

        # Verify: only one row with that x,y for that reach
        count = conn.execute(
            "SELECT COUNT(*) FROM centerlines WHERE reach_id = ? AND x = ? AND y = ? AND region = 'NA'",
            [reach_id, x, y],
        ).fetchone()[0]
        assert count == 1


# =========================================================================
# Task 9 — find_and_merge_single_node_reaches
# =========================================================================


class TestFindAndMergeSingleNodeReaches:
    """Test SWORDWorkflow.find_and_merge_single_node_reaches."""

    def test_dry_run_returns_dataframe(self, temp_workflow):
        result = temp_workflow.find_and_merge_single_node_reaches(
            region="NA", dry_run=True
        )
        assert isinstance(result, dict)
        assert "candidates" in result
        assert isinstance(result["candidates"], pd.DataFrame)
        # DataFrame should have expected columns (even if empty)
        expected_cols = {"reach_id", "merge_target", "direction", "status"}
        assert expected_cols.issubset(set(result["candidates"].columns))


# =========================================================================
# Task 10 — rederive_nodes
# =========================================================================


class TestRederiveNodes:
    """Test SWORDWorkflow.rederive_nodes."""

    def test_dry_run_returns_dataframe(self, temp_workflow):
        conn = temp_workflow._sword.db.conn
        row = conn.execute(
            "SELECT reach_id FROM nodes WHERE region = 'NA' GROUP BY reach_id HAVING COUNT(*) >= 3 LIMIT 1"
        ).fetchone()
        assert row is not None
        result = temp_workflow.rederive_nodes(
            reach_ids=[row[0]], region="NA", dry_run=True
        )
        assert isinstance(result, dict)
        assert "reaches_processed" in result

    def test_rederive_preserves_node_count(self, temp_workflow):
        """Rederive nodes for a reach and verify node count is preserved."""
        conn = temp_workflow._sword.db.conn

        # Pick a reach with >= 3 nodes
        row = conn.execute(
            """
            SELECT reach_id, COUNT(*) as nc
            FROM nodes WHERE region = 'NA'
            GROUP BY reach_id HAVING nc >= 3
            LIMIT 1
            """
        ).fetchone()
        assert row is not None
        reach_id, original_count = row

        # Rederive
        result = temp_workflow.rederive_nodes(reach_ids=[reach_id], region="NA")
        assert result["reaches_processed"] == 1

        # Verify node count unchanged
        new_count = conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE reach_id = ? AND region = 'NA'",
            [reach_id],
        ).fetchone()[0]
        assert new_count == original_count
