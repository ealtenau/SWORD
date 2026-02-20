# -*- coding: utf-8 -*-
"""
Unit tests for SWORD PostgreSQL trigger module.

Tests cover:
- Trigger SQL generation
- Import validation
- Function signatures
"""

import os
import sys
import pytest

# Add project root to path
main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, main_dir)

pytestmark = pytest.mark.unit


class TestTriggerImports:
    """Test that trigger functions can be imported."""

    def test_import_install_triggers(self):
        """Test import of install_triggers."""
        from src.sword_duckdb.triggers import install_triggers

        assert callable(install_triggers)

    def test_import_remove_triggers(self):
        """Test import of remove_triggers."""
        from src.sword_duckdb.triggers import remove_triggers

        assert callable(remove_triggers)

    def test_import_get_pending_changes(self):
        """Test import of get_pending_changes."""
        from src.sword_duckdb.triggers import get_pending_changes

        assert callable(get_pending_changes)

    def test_import_get_changed_entities(self):
        """Test import of get_changed_entities."""
        from src.sword_duckdb.triggers import get_changed_entities

        assert callable(get_changed_entities)

    def test_import_mark_changes_synced(self):
        """Test import of mark_changes_synced."""
        from src.sword_duckdb.triggers import mark_changes_synced

        assert callable(mark_changes_synced)

    def test_import_get_trigger_sql(self):
        """Test import of get_trigger_sql."""
        from src.sword_duckdb.triggers import get_trigger_sql

        assert callable(get_trigger_sql)

    def test_module_exports(self):
        """Test that functions are exported from package __init__."""
        from src.sword_duckdb import (
            install_triggers,
            remove_triggers,
            get_pending_changes,
            get_changed_entities,
            mark_changes_synced,
            get_trigger_sql,
        )

        assert callable(install_triggers)
        assert callable(remove_triggers)
        assert callable(get_pending_changes)
        assert callable(get_changed_entities)
        assert callable(mark_changes_synced)
        assert callable(get_trigger_sql)


class TestTriggerSQL:
    """Test trigger SQL generation."""

    def test_get_trigger_sql_no_prefix(self):
        """Test SQL generation without prefix."""
        from src.sword_duckdb.triggers import get_trigger_sql

        sql = get_trigger_sql()

        # Verify SQL contains expected elements
        assert "CREATE TABLE IF NOT EXISTS" in sql
        assert "sword_changes" in sql
        assert "CREATE OR REPLACE FUNCTION" in sql
        assert "reaches_change_trigger" in sql
        assert "nodes_change_trigger" in sql
        assert "TRIGGER" in sql

    def test_get_trigger_sql_with_prefix(self):
        """Test SQL generation with prefix."""
        from src.sword_duckdb.triggers import get_trigger_sql

        sql = get_trigger_sql(prefix="na_")

        # Verify prefix is applied
        assert "na_sword_changes" in sql
        assert "na_reaches_change_trigger" in sql
        assert "na_nodes_change_trigger" in sql

    def test_trigger_sql_tracks_all_change_types(self):
        """Test that trigger SQL handles INSERT, UPDATE, DELETE."""
        from src.sword_duckdb.triggers import get_trigger_sql

        sql = get_trigger_sql()

        # Verify all change types are handled
        assert "'INSERT'" in sql
        assert "'UPDATE'" in sql
        assert "'DELETE'" in sql

    def test_change_tracking_table_has_required_columns(self):
        """Test that change tracking table has all required columns."""
        from src.sword_duckdb.triggers import CHANGE_TRACKING_TABLE

        required_columns = [
            "change_id",
            "table_name",
            "entity_id",
            "region",
            "change_type",
            "changed_columns",
            "old_values",
            "new_values",
            "changed_at",
            "synced",
        ]

        for col in required_columns:
            assert col in CHANGE_TRACKING_TABLE, f"Missing column: {col}"


class TestTriggerFunctionSignatures:
    """Test function signatures."""

    def test_install_triggers_signature(self):
        """Test install_triggers has expected parameters."""
        from src.sword_duckdb.triggers import install_triggers
        import inspect

        sig = inspect.signature(install_triggers)
        params = list(sig.parameters.keys())

        assert "connection_string" in params
        assert "prefix" in params
        assert "verbose" in params

    def test_get_changed_entities_signature(self):
        """Test get_changed_entities has expected parameters."""
        from src.sword_duckdb.triggers import get_changed_entities
        import inspect

        sig = inspect.signature(get_changed_entities)
        params = list(sig.parameters.keys())

        assert "connection_string" in params
        assert "prefix" in params
        assert "table" in params
        assert "since" in params

    def test_mark_changes_synced_signature(self):
        """Test mark_changes_synced has expected parameters."""
        from src.sword_duckdb.triggers import mark_changes_synced
        import inspect

        sig = inspect.signature(mark_changes_synced)
        params = list(sig.parameters.keys())

        assert "connection_string" in params
        assert "change_ids" in params
        assert "all_changes" in params


class TestTriggerSQLSafety:
    """Test that generated SQL is safe."""

    def test_sql_uses_parameterized_placeholders(self):
        """Test SQL uses %s for parameters, not string formatting for user input."""
        from src.sword_duckdb.triggers import (
            REACHES_TRIGGER_FUNCTION,
            NODES_TRIGGER_FUNCTION,
        )

        # These should not have any %s (user data is from QGIS, not Python params)
        # The SQL uses column references like NEW.reach_id, not string interpolation
        assert "%s" not in REACHES_TRIGGER_FUNCTION
        assert "%s" not in NODES_TRIGGER_FUNCTION

    def test_drop_triggers_is_safe(self):
        """Test DROP statements use IF EXISTS."""
        from src.sword_duckdb.triggers import DROP_TRIGGERS

        assert "DROP TRIGGER IF EXISTS" in DROP_TRIGGERS
        assert "DROP FUNCTION IF EXISTS" in DROP_TRIGGERS
        assert "DROP TABLE IF EXISTS" in DROP_TRIGGERS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
