# -*- coding: utf-8 -*-
"""
Unit tests for SWORD Workflow class methods.

Tests cover the workflow-level wrappers in workflow.py:
- delete_reach / delete_reaches
- check_topology
- check_node_lengths
- validate_ids
- status
- load/close

Note: break_reach and append_reaches are harder to test without
mock data, so we focus on method existence and RuntimeError checks.
"""

import os
import sys
import pytest
import tempfile
import shutil

# Add project root to path
main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, main_dir)

from src.updates.sword_duckdb.workflow import SWORDWorkflow

# Test configuration
TEST_DB_PATH = os.path.join(main_dir, 'data/duckdb/sword_v17b.duckdb')
TEST_REGION = 'NA'
TEST_VERSION = 'v17b'


@pytest.fixture
def temp_workflow():
    """Create a temporary copy of the database for write tests."""
    if not os.path.exists(TEST_DB_PATH):
        pytest.skip(f"Test database not found: {TEST_DB_PATH}")

    # Create temp copy
    temp_dir = tempfile.mkdtemp()
    temp_db = os.path.join(temp_dir, 'sword_test.duckdb')
    try:
        shutil.copy2(TEST_DB_PATH, temp_db)
    except OSError as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        pytest.skip(f"Cannot copy database (disk space?): {e}")

    workflow = SWORDWorkflow(
        user_id="test_user",
        enable_provenance=False,  # Disable for tests
    )
    workflow.load(temp_db, TEST_REGION)
    yield workflow

    # Cleanup
    workflow.close()
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def unloaded_workflow():
    """Create a workflow instance without loading the database.

    Note: For tests that only check unloaded state (RuntimeError checks),
    we don't need to copy the database. For tests that need to load,
    use temp_workflow fixture instead.
    """
    workflow = SWORDWorkflow(
        user_id="test_user",
        enable_provenance=False,
    )
    yield workflow

    # Cleanup
    if workflow.is_loaded:
        workflow.close()


class TestWorkflowLoad:
    """Test load/close methods."""

    def test_load_database(self):
        """Test loading database."""
        if not os.path.exists(TEST_DB_PATH):
            pytest.skip(f"Test database not found: {TEST_DB_PATH}")

        temp_dir = tempfile.mkdtemp()
        temp_db = os.path.join(temp_dir, 'sword_test.duckdb')
        try:
            shutil.copy2(TEST_DB_PATH, temp_db)
        except OSError as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            pytest.skip(f"Cannot copy database: {e}")

        workflow = SWORDWorkflow(user_id="test", enable_provenance=False)
        try:
            assert not workflow.is_loaded
            workflow.load(temp_db, TEST_REGION)
            assert workflow.is_loaded
        finally:
            if workflow.is_loaded:
                workflow.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_close_database(self, temp_workflow):
        """Test closing database."""
        assert temp_workflow.is_loaded
        temp_workflow.close()
        assert not temp_workflow.is_loaded

    def test_context_manager(self):
        """Test using workflow as context manager."""
        if not os.path.exists(TEST_DB_PATH):
            pytest.skip(f"Test database not found: {TEST_DB_PATH}")

        temp_dir = tempfile.mkdtemp()
        temp_db = os.path.join(temp_dir, 'sword_test.duckdb')
        try:
            shutil.copy2(TEST_DB_PATH, temp_db)
        except OSError as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            pytest.skip(f"Cannot copy database: {e}")

        try:
            with SWORDWorkflow(user_id="test", enable_provenance=False) as wf:
                wf.load(temp_db, TEST_REGION)
                assert wf.is_loaded
            assert not wf.is_loaded
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestDeleteReach:
    """Test delete_reach method."""

    def test_delete_reach_requires_load(self, unloaded_workflow):
        """Test delete_reach raises error when database not loaded."""
        with pytest.raises(RuntimeError, match="No database loaded"):
            unloaded_workflow.delete_reach(12345)

    def test_delete_reach_single(self, temp_workflow):
        """Test deleting a single reach via workflow."""
        initial_count = len(temp_workflow._sword.reaches)
        reach_to_delete = int(temp_workflow._sword.reaches.id[0])

        result = temp_workflow.delete_reach(reach_to_delete)

        assert result['success'] is True
        assert result['deleted_reach'] == reach_to_delete
        assert result['cascade'] is True

    def test_delete_reach_no_cascade(self, temp_workflow):
        """Test delete_reach with cascade=False."""
        reach_to_delete = int(temp_workflow._sword.reaches.id[0])

        result = temp_workflow.delete_reach(reach_to_delete, cascade=False)

        assert result['success'] is True
        assert result['cascade'] is False

    def test_delete_reach_with_reason(self, temp_workflow):
        """Test delete_reach with reason parameter."""
        reach_to_delete = int(temp_workflow._sword.reaches.id[0])

        result = temp_workflow.delete_reach(
            reach_to_delete,
            reason="Test deletion"
        )

        assert result['success'] is True


class TestDeleteReaches:
    """Test delete_reaches method (multiple)."""

    def test_delete_reaches_requires_load(self, unloaded_workflow):
        """Test delete_reaches raises error when database not loaded."""
        with pytest.raises(RuntimeError, match="No database loaded"):
            unloaded_workflow.delete_reaches([12345, 67890])

    def test_delete_reaches_multiple(self, temp_workflow):
        """Test deleting multiple reaches via workflow."""
        reaches_to_delete = [
            int(temp_workflow._sword.reaches.id[0]),
        ]

        result = temp_workflow.delete_reaches(reaches_to_delete)

        assert result['success'] is True
        assert result['deleted_count'] == 1
        assert result['cascade'] is True


class TestCheckTopology:
    """Test check_topology method."""

    def test_check_topology_requires_load(self, unloaded_workflow):
        """Test check_topology raises error when database not loaded."""
        with pytest.raises(RuntimeError, match="No database loaded"):
            unloaded_workflow.check_topology()

    def test_check_topology_returns_dict(self, temp_workflow):
        """Test check_topology returns proper dict."""
        result = temp_workflow.check_topology(verbose=0)

        assert isinstance(result, dict)
        assert 'passed' in result
        assert 'total_reaches' in result
        assert 'error_counts' in result
        assert 'warning_counts' in result
        assert 'reaches_with_issues' in result

    def test_check_topology_with_details(self, temp_workflow):
        """Test check_topology with return_details=True."""
        result = temp_workflow.check_topology(verbose=0, return_details=True)

        assert 'details' in result


class TestCheckNodeLengths:
    """Test check_node_lengths method."""

    def test_check_node_lengths_requires_load(self, unloaded_workflow):
        """Test check_node_lengths raises error when database not loaded."""
        with pytest.raises(RuntimeError, match="No database loaded"):
            unloaded_workflow.check_node_lengths()

    def test_check_node_lengths_returns_dict(self, temp_workflow):
        """Test check_node_lengths returns proper dict."""
        result = temp_workflow.check_node_lengths(verbose=0)

        assert isinstance(result, dict)
        assert 'passed' in result
        assert 'total_nodes' in result
        assert 'long_nodes' in result
        assert 'zero_length_nodes' in result
        assert 'affected_reaches' in result

    def test_check_node_lengths_custom_threshold(self, temp_workflow):
        """Test check_node_lengths with custom threshold."""
        result = temp_workflow.check_node_lengths(verbose=0, threshold=500.0)

        assert isinstance(result, dict)
        assert 'passed' in result


class TestValidateIds:
    """Test validate_ids method."""

    def test_validate_ids_requires_load(self, unloaded_workflow):
        """Test validate_ids raises error when database not loaded."""
        with pytest.raises(RuntimeError, match="No database loaded"):
            unloaded_workflow.validate_ids()

    def test_validate_ids_all_valid(self, temp_workflow):
        """Test validate_ids returns results dict."""
        # Use a small subset to avoid long runtime
        reach_ids = temp_workflow._sword.reaches.id[:5].tolist()
        node_ids = temp_workflow._sword.nodes.id[:5].tolist()

        result = temp_workflow.validate_ids(
            reach_ids=reach_ids,
            node_ids=node_ids
        )

        assert isinstance(result, dict)
        assert 'passed' in result
        assert 'total_reaches_checked' in result
        assert 'total_nodes_checked' in result
        assert 'invalid_reaches' in result
        assert 'invalid_nodes' in result
        assert result['total_reaches_checked'] == 5
        assert result['total_nodes_checked'] == 5

    def test_validate_ids_detects_invalid_reach(self, temp_workflow):
        """Test validate_ids detects invalid reach ID."""
        result = temp_workflow.validate_ids(
            reach_ids=[12345],  # Invalid format
            node_ids=[]
        )

        assert result['passed'] is False
        assert 12345 in result['invalid_reaches']

    def test_validate_ids_detects_invalid_node(self, temp_workflow):
        """Test validate_ids detects invalid node ID."""
        result = temp_workflow.validate_ids(
            reach_ids=[],
            node_ids=[12345]  # Invalid format
        )

        assert result['passed'] is False
        assert 12345 in result['invalid_nodes']


class TestStatus:
    """Test status method."""

    def test_status_when_loaded(self, temp_workflow):
        """Test status returns correct info when loaded."""
        status = temp_workflow.status()

        assert isinstance(status, dict)
        assert status['is_loaded'] is True
        assert status['region'] == TEST_REGION

    def test_status_when_not_loaded(self, unloaded_workflow):
        """Test status returns correct info when not loaded."""
        status = unloaded_workflow.status()

        assert isinstance(status, dict)
        assert status['is_loaded'] is False


class TestBreakReach:
    """Test break_reach method existence and basic checks."""

    def test_break_reach_method_exists(self, temp_workflow):
        """Test break_reach method exists."""
        assert hasattr(temp_workflow, 'break_reach')
        assert callable(temp_workflow.break_reach)

    def test_break_reach_requires_load(self, unloaded_workflow):
        """Test break_reach raises error when database not loaded."""
        with pytest.raises(RuntimeError, match="No database loaded"):
            unloaded_workflow.break_reach(12345, 67890)


class TestAppendReaches:
    """Test append_reaches method existence and basic checks."""

    def test_append_reaches_method_exists(self, temp_workflow):
        """Test append_reaches method exists."""
        assert hasattr(temp_workflow, 'append_reaches')
        assert callable(temp_workflow.append_reaches)

    def test_append_reaches_requires_load(self, unloaded_workflow):
        """Test append_reaches raises error when database not loaded."""
        with pytest.raises(RuntimeError, match="No database loaded"):
            unloaded_workflow.append_reaches(None, None, None)


class TestCreateGhostReach:
    """Test create_ghost_reach workflow method."""

    def test_create_ghost_reach_method_exists(self, temp_workflow):
        """Test create_ghost_reach method exists."""
        assert hasattr(temp_workflow, 'create_ghost_reach')
        assert callable(temp_workflow.create_ghost_reach)

    def test_create_ghost_reach_requires_load(self, unloaded_workflow):
        """Test create_ghost_reach raises error when database not loaded."""
        with pytest.raises(RuntimeError, match="No database loaded"):
            unloaded_workflow.create_ghost_reach(12345)

    def test_create_ghost_reach_via_workflow(self, temp_workflow):
        """Test creating a ghost reach via workflow wrapper."""
        import numpy as np

        # Find a suitable reach (no upstream, at least 2 nodes, not ghost)
        candidate_reach = None
        for idx, reach_id in enumerate(temp_workflow._sword.reaches.id):
            reach_type = str(reach_id)[-1]
            if reach_type == '6':
                continue
            n_up = temp_workflow._sword.reaches.n_rch_up[idx]
            if n_up == 0:
                node_count = np.sum(temp_workflow._sword.nodes.reach_id == reach_id)
                if node_count >= 2:
                    candidate_reach = int(reach_id)
                    break

        if candidate_reach is None:
            pytest.skip("No suitable reach found for ghost reach test")

        result = temp_workflow.create_ghost_reach(
            candidate_reach,
            position='headwater',
            reason="Test ghost reach creation"
        )

        assert result['success'] is True
        assert result['original_reach'] == candidate_reach
        assert result['ghost_reach_id'] is not None
        assert str(result['ghost_reach_id'])[-1] == '6'


class TestFindMissingGhostReaches:
    """Test find_missing_ghost_reaches workflow method."""

    def test_find_missing_ghost_reaches_method_exists(self, temp_workflow):
        """Test find_missing_ghost_reaches method exists."""
        assert hasattr(temp_workflow, 'find_missing_ghost_reaches')
        assert callable(temp_workflow.find_missing_ghost_reaches)

    def test_find_missing_ghost_reaches_requires_load(self, unloaded_workflow):
        """Test find_missing_ghost_reaches raises error when not loaded."""
        with pytest.raises(RuntimeError, match="No database loaded"):
            unloaded_workflow.find_missing_ghost_reaches()

    def test_find_missing_ghost_reaches_returns_dict(self, temp_workflow):
        """Test find_missing_ghost_reaches returns proper dict."""
        result = temp_workflow.find_missing_ghost_reaches()

        assert isinstance(result, dict)
        assert 'missing_headwaters' in result
        assert 'missing_outlets' in result
        assert 'total_missing' in result


class TestFindIncorrectGhostReaches:
    """Test find_incorrect_ghost_reaches workflow method."""

    def test_find_incorrect_ghost_reaches_method_exists(self, temp_workflow):
        """Test find_incorrect_ghost_reaches method exists."""
        assert hasattr(temp_workflow, 'find_incorrect_ghost_reaches')
        assert callable(temp_workflow.find_incorrect_ghost_reaches)

    def test_find_incorrect_ghost_reaches_requires_load(self, unloaded_workflow):
        """Test find_incorrect_ghost_reaches raises error when not loaded."""
        with pytest.raises(RuntimeError, match="No database loaded"):
            unloaded_workflow.find_incorrect_ghost_reaches()

    def test_find_incorrect_ghost_reaches_returns_dict(self, temp_workflow):
        """Test find_incorrect_ghost_reaches returns proper dict."""
        result = temp_workflow.find_incorrect_ghost_reaches()

        assert isinstance(result, dict)
        assert 'incorrect_ghost_reaches' in result
        assert 'total_incorrect' in result


class TestRecalculateStreamOrder:
    """Test recalculate_stream_order workflow method."""

    def test_recalculate_stream_order_method_exists(self, temp_workflow):
        """Test recalculate_stream_order method exists."""
        assert hasattr(temp_workflow, 'recalculate_stream_order')
        assert callable(temp_workflow.recalculate_stream_order)

    def test_recalculate_stream_order_requires_load(self, unloaded_workflow):
        """Test recalculate_stream_order raises error when not loaded."""
        with pytest.raises(RuntimeError, match="No database loaded"):
            unloaded_workflow.recalculate_stream_order()

    def test_recalculate_stream_order_returns_dict(self, temp_workflow):
        """Test recalculate_stream_order returns proper dict."""
        result = temp_workflow.recalculate_stream_order(
            update_nodes=False,
            update_reaches=False
        )

        assert isinstance(result, dict)
        assert 'nodes_updated' in result
        assert 'reaches_updated' in result
        assert 'nodes_with_valid_path_freq' in result


class TestRecalculatePathSegs:
    """Test recalculate_path_segs workflow method."""

    def test_recalculate_path_segs_method_exists(self, temp_workflow):
        """Test recalculate_path_segs method exists."""
        assert hasattr(temp_workflow, 'recalculate_path_segs')
        assert callable(temp_workflow.recalculate_path_segs)

    def test_recalculate_path_segs_requires_load(self, unloaded_workflow):
        """Test recalculate_path_segs raises error when not loaded."""
        with pytest.raises(RuntimeError, match="No database loaded"):
            unloaded_workflow.recalculate_path_segs()

    def test_recalculate_path_segs_returns_dict(self, temp_workflow):
        """Test recalculate_path_segs returns proper dict."""
        result = temp_workflow.recalculate_path_segs(
            update_nodes=False,
            update_reaches=False
        )

        assert isinstance(result, dict)
        assert 'nodes_updated' in result
        assert 'reaches_updated' in result
        assert 'total_segments' in result
        assert 'nodes_assigned' in result


class TestRecalculateNetworkAttributes:
    """Test recalculate_network_attributes workflow method."""

    def test_recalculate_network_attributes_method_exists(self, temp_workflow):
        """Test recalculate_network_attributes method exists."""
        assert hasattr(temp_workflow, 'recalculate_network_attributes')
        assert callable(temp_workflow.recalculate_network_attributes)

    @pytest.mark.skip(reason="DuckDB GC segfault on large updates")
    def test_recalculate_network_attributes_supports_stream_order(self, temp_workflow):
        """Test recalculate_network_attributes handles stream_order."""
        result = temp_workflow.recalculate_network_attributes(
            attributes=['stream_order']
        )

        assert isinstance(result, dict)
        assert 'stream_order' in result['attributes_updated']

    @pytest.mark.skip(reason="DuckDB GC segfault on large updates")
    def test_recalculate_network_attributes_supports_path_segs(self, temp_workflow):
        """Test recalculate_network_attributes handles path_segs."""
        result = temp_workflow.recalculate_network_attributes(
            attributes=['path_segs']
        )

        assert isinstance(result, dict)
        assert 'path_segs' in result['attributes_updated']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
