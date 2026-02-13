# -*- coding: utf-8 -*-
"""
Integration tests for SWORD workflow.

Tests end-to-end workflows including:
- Load-modify-save roundtrips
- Data persistence verification
- Topology consistency after modifications
- Batch operations
"""

import sys
import pytest
import numpy as np
import shutil
from pathlib import Path

# Add project root to path
main_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(main_dir))

from src.updates.sword_duckdb import SWORD
from src.updates.sword_duckdb.workflow import SWORDWorkflow

pytestmark = [pytest.mark.integration, pytest.mark.db]


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture(scope="session")
def test_db_path():
    """Path to the minimal test database."""
    return (
        Path(__file__).parent.parent
        / "sword_duckdb"
        / "fixtures"
        / "sword_test_minimal.duckdb"
    )


@pytest.fixture(scope="session")
def ensure_test_db(test_db_path):
    """Ensure minimal test database exists."""
    if not test_db_path.exists():
        from tests.sword_duckdb.fixtures.create_test_db import create_minimal_test_db

        test_db_path.parent.mkdir(parents=True, exist_ok=True)
        create_minimal_test_db(str(test_db_path))
    return test_db_path


@pytest.fixture
def workflow_db(ensure_test_db, tmp_path):
    """Writable workflow with temporary database copy."""
    temp_db = tmp_path / "sword_workflow_test.duckdb"
    shutil.copy2(ensure_test_db, temp_db)
    return str(temp_db)


# ==============================================================================
# Test Classes
# ==============================================================================


class TestLoadModifySaveRoundtrip:
    """Test complete load-modify-save cycles."""

    def test_simple_modify_persist(self, workflow_db):
        """Test that a simple modification persists after reload."""
        # Load and modify
        sword = SWORD(workflow_db, "NA", "v17b")
        original_value = float(sword.reaches.dist_out[0])
        test_value = 123456.789

        sword.reaches.dist_out[0] = test_value
        assert sword.reaches.dist_out[0] == test_value, "In-memory value not updated"

        sword.close()

        # Reload and verify
        sword2 = SWORD(workflow_db, "NA", "v17b")
        reloaded_value = float(sword2.reaches.dist_out[0])
        sword2.close()

        assert reloaded_value == test_value, (
            f"Value not persisted: expected {test_value}, got {reloaded_value}"
        )

    def test_multiple_modifications_persist(self, workflow_db):
        """Test multiple modifications persist correctly."""
        sword = SWORD(workflow_db, "NA", "v17b")

        # Modify multiple reaches
        test_values = [111.0, 222.0, 333.0]
        for i, val in enumerate(test_values):
            sword.reaches.dist_out[i] = val

        sword.close()

        # Reload and verify all
        sword2 = SWORD(workflow_db, "NA", "v17b")
        for i, expected in enumerate(test_values):
            actual = float(sword2.reaches.dist_out[i])
            assert actual == expected, f"Reach {i}: expected {expected}, got {actual}"
        sword2.close()


class TestDataConsistency:
    """Test data consistency after operations."""

    def test_reach_count_unchanged_after_modify(self, workflow_db):
        """Test reach count doesn't change after modifications."""
        sword = SWORD(workflow_db, "NA", "v17b")
        original_count = len(sword.reaches)

        # Modify some values
        sword.reaches.dist_out[0] = 99999.0
        sword.reaches.wse[1] = 50.0

        # Verify count unchanged
        assert len(sword.reaches) == original_count
        sword.close()

        # Verify after reload
        sword2 = SWORD(workflow_db, "NA", "v17b")
        assert len(sword2.reaches) == original_count
        sword2.close()

    def test_node_reach_association(self, workflow_db):
        """Test nodes are correctly associated with reaches."""
        sword = SWORD(workflow_db, "NA", "v17b")

        # Get all unique reach IDs from nodes
        node_reach_ids = np.unique(sword.nodes.reach_id)

        # Verify all referenced reaches exist
        reach_ids = sword.reaches.id
        for rid in node_reach_ids:
            assert rid in reach_ids, f"Node references non-existent reach {rid}"

        sword.close()


class TestTopologyConsistency:
    """Test topology consistency."""

    def test_topology_arrays_shape(self, workflow_db):
        """Test topology arrays have correct shape."""
        sword = SWORD(workflow_db, "NA", "v17b")

        # rch_id_up and rch_id_down should be [4, N]
        rch_id_up = sword.reaches.rch_id_up
        rch_id_down = sword.reaches.rch_id_down

        assert rch_id_up.shape[0] == 4, (
            f"rch_id_up should have 4 rows, got {rch_id_up.shape[0]}"
        )
        assert rch_id_up.shape[1] == len(sword.reaches), (
            "rch_id_up column count mismatch"
        )

        assert rch_id_down.shape[0] == 4, (
            f"rch_id_down should have 4 rows, got {rch_id_down.shape[0]}"
        )
        assert rch_id_down.shape[1] == len(sword.reaches), (
            "rch_id_down column count mismatch"
        )

        sword.close()

    def test_topology_references_valid(self, workflow_db):
        """Test topology references point to valid reaches or zero."""
        sword = SWORD(workflow_db, "NA", "v17b")

        reach_ids = set(sword.reaches.id)
        rch_id_up = sword.reaches.rch_id_up
        rch_id_down = sword.reaches.rch_id_down

        # All non-zero references should be valid reach IDs
        for i in range(rch_id_up.shape[1]):
            for j in range(4):
                up_ref = rch_id_up[j, i]
                down_ref = rch_id_down[j, i]

                if up_ref != 0:
                    assert up_ref in reach_ids, (
                        f"Invalid upstream ref {up_ref} for reach {i}"
                    )
                if down_ref != 0:
                    assert down_ref in reach_ids, (
                        f"Invalid downstream ref {down_ref} for reach {i}"
                    )

        sword.close()


class TestWorkflowIntegration:
    """Test SWORDWorkflow integration."""

    def test_workflow_load_close(self, workflow_db):
        """Test basic workflow load and close."""
        workflow = SWORDWorkflow(user_id="test_user")

        sword = workflow.load(workflow_db, "NA")
        assert sword is not None
        assert len(sword.reaches) > 0

        workflow.close()

    def test_workflow_context_manager(self, workflow_db):
        """Test workflow as context manager."""
        with SWORDWorkflow(user_id="test_user") as workflow:
            sword = workflow.load(workflow_db, "NA")
            assert sword is not None
            assert len(sword.reaches) == 100  # Minimal fixture has 100 reaches


class TestBatchOperations:
    """Test batch modification operations."""

    def test_batch_modify_array_slice(self, workflow_db):
        """Test batch modification via array slicing."""
        sword = SWORD(workflow_db, "NA", "v17b")

        # Modify first 10 reaches at once
        indices = list(range(10))
        new_values = [i * 100.0 for i in range(10)]

        sword.reaches.dist_out[indices] = new_values

        # Verify in-memory
        for i, expected in zip(indices, new_values):
            assert sword.reaches.dist_out[i] == expected

        sword.close()

        # Verify persistence
        sword2 = SWORD(workflow_db, "NA", "v17b")
        for i, expected in zip(indices, new_values):
            assert float(sword2.reaches.dist_out[i]) == expected
        sword2.close()


class TestMinimalFixtureProperties:
    """Test properties of the minimal test fixture."""

    def test_fixture_reach_count(self, workflow_db):
        """Test minimal fixture has expected reach count."""
        sword = SWORD(workflow_db, "NA", "v17b")
        assert len(sword.reaches) == 100, (
            f"Expected 100 reaches, got {len(sword.reaches)}"
        )
        sword.close()

    def test_fixture_node_count(self, workflow_db):
        """Test minimal fixture has expected node count."""
        sword = SWORD(workflow_db, "NA", "v17b")
        # 5 nodes per reach * 100 reaches = 500 nodes
        assert len(sword.nodes) == 500, f"Expected 500 nodes, got {len(sword.nodes)}"
        sword.close()

    def test_fixture_centerline_count(self, workflow_db):
        """Test minimal fixture has expected centerline count."""
        sword = SWORD(workflow_db, "NA", "v17b")
        # 20 centerlines per reach * 100 reaches = 2000 centerlines
        assert len(sword.centerlines) == 2000, (
            f"Expected 2000 centerlines, got {len(sword.centerlines)}"
        )
        sword.close()

    def test_fixture_has_topology(self, workflow_db):
        """Test minimal fixture has topology data."""
        sword = SWORD(workflow_db, "NA", "v17b")

        # Should have some upstream/downstream connections
        rch_id_up = sword.reaches.rch_id_up
        rch_id_down = sword.reaches.rch_id_down

        # At least some reaches should have connections
        has_upstream = np.any(rch_id_up != 0)
        has_downstream = np.any(rch_id_down != 0)

        assert has_upstream, "No upstream topology connections found"
        assert has_downstream, "No downstream topology connections found"

        sword.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
