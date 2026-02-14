"""
Tests for SWORD Lint Framework

Tests core functionality, check registration, runners, and formatters.
"""

import json
import pytest
from pathlib import Path

import pandas as pd

pytestmark = pytest.mark.lint

from src.updates.sword_duckdb.lint import (
    Severity,
    Category,
    CheckResult,
    LintRunner,
    get_registry,
    get_check,
    get_checks_by_category,
    get_checks_by_severity,
    list_check_ids,
)
from src.updates.sword_duckdb.lint.formatters import (
    ConsoleFormatter,
    JsonFormatter,
    MarkdownFormatter,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def test_db_path():
    """Path to test database."""
    return Path(__file__).parent / "fixtures" / "sword_test_minimal.duckdb"


@pytest.fixture
def sample_result():
    """Sample CheckResult for testing formatters."""
    return CheckResult(
        check_id="T001",
        name="test_check",
        severity=Severity.ERROR,
        passed=False,
        total_checked=100,
        issues_found=5,
        issue_pct=5.0,
        details=pd.DataFrame(
            {
                "reach_id": [1, 2, 3, 4, 5],
                "region": ["NA"] * 5,
                "issue": ["a", "b", "c", "d", "e"],
            }
        ),
        description="Test check description",
        threshold=10.0,
        elapsed_ms=123.45,
    )


@pytest.fixture
def sample_results(sample_result):
    """List of sample results for testing."""
    return [
        sample_result,
        CheckResult(
            check_id="T002",
            name="passing_check",
            severity=Severity.WARNING,
            passed=True,
            total_checked=50,
            issues_found=0,
            issue_pct=0.0,
            details=pd.DataFrame(),
            description="A passing check",
            elapsed_ms=50.0,
        ),
        CheckResult(
            check_id="A001",
            name="info_check",
            severity=Severity.INFO,
            passed=False,
            total_checked=200,
            issues_found=10,
            issue_pct=5.0,
            details=pd.DataFrame({"reach_id": range(10)}),
            description="An info check",
            elapsed_ms=75.0,
        ),
    ]


# =============================================================================
# Core Types Tests
# =============================================================================


class TestSeverity:
    """Tests for Severity enum."""

    def test_severity_values(self):
        assert Severity.ERROR.value == "error"
        assert Severity.WARNING.value == "warning"
        assert Severity.INFO.value == "info"

    def test_severity_from_string(self):
        assert Severity("error") == Severity.ERROR
        assert Severity("warning") == Severity.WARNING
        assert Severity("info") == Severity.INFO


class TestCategory:
    """Tests for Category enum."""

    def test_category_values(self):
        assert Category.TOPOLOGY.value == "topology"
        assert Category.ATTRIBUTES.value == "attributes"
        assert Category.GEOMETRY.value == "geometry"
        assert Category.CLASSIFICATION.value == "classification"


class TestCheckResult:
    """Tests for CheckResult dataclass."""

    def test_check_result_creation(self, sample_result):
        assert sample_result.check_id == "T001"
        assert sample_result.severity == Severity.ERROR
        assert sample_result.passed is False
        assert sample_result.issues_found == 5

    def test_check_result_repr(self, sample_result):
        repr_str = repr(sample_result)
        assert "T001" in repr_str
        assert "FAIL" in repr_str


# =============================================================================
# Registry Tests
# =============================================================================


class TestRegistry:
    """Tests for check registry."""

    def test_registry_not_empty(self):
        registry = get_registry()
        assert len(registry) > 0

    def test_check_ids_unique(self):
        """All check IDs must be unique."""
        ids = list_check_ids()
        assert len(ids) == len(set(ids))

    def test_check_id_format(self):
        """Check IDs should follow pattern: letter + digits."""
        for check_id in list_check_ids():
            assert len(check_id) >= 2
            assert check_id[0].isalpha()
            assert check_id[1:].isdigit()

    def test_get_check(self):
        spec = get_check("T001")
        assert spec is not None
        assert spec.check_id == "T001"
        assert spec.category == Category.TOPOLOGY

    def test_get_nonexistent_check(self):
        assert get_check("Z999") is None

    def test_checks_by_category(self):
        topology_checks = get_checks_by_category(Category.TOPOLOGY)
        assert len(topology_checks) > 0
        for spec in topology_checks:
            assert spec.category == Category.TOPOLOGY

    def test_checks_by_severity(self):
        error_checks = get_checks_by_severity(Severity.ERROR)
        for spec in error_checks:
            assert spec.severity == Severity.ERROR

    def test_required_checks_registered(self):
        """Verify all planned checks are registered."""
        required = [
            "T001",
            "T002",
            "T003",
            "T004",
            "T005",
            "T006",
            "A001",
            "A002",
            "A003",
            "A004",
            "A005",
            "G001",
            "C001",
        ]
        registry = get_registry()
        for check_id in required:
            assert check_id in registry, f"Missing required check: {check_id}"


# =============================================================================
# Runner Tests
# =============================================================================


@pytest.mark.db
class TestLintRunner:
    """Tests for LintRunner."""

    def test_runner_init_nonexistent_db(self):
        with pytest.raises(FileNotFoundError):
            LintRunner("/nonexistent/path.duckdb")

    @pytest.mark.skipif(
        not (Path(__file__).parent / "fixtures" / "sword_test_minimal.duckdb").exists(),
        reason="Test database not found",
    )
    def test_runner_init_valid_db(self, test_db_path):
        runner = LintRunner(test_db_path)
        assert runner.conn is not None
        runner.close()

    @pytest.mark.skipif(
        not (Path(__file__).parent / "fixtures" / "sword_test_minimal.duckdb").exists(),
        reason="Test database not found",
    )
    def test_runner_context_manager(self, test_db_path):
        with LintRunner(test_db_path) as runner:
            assert runner.conn is not None

    @pytest.mark.skipif(
        not (Path(__file__).parent / "fixtures" / "sword_test_minimal.duckdb").exists(),
        reason="Test database not found",
    )
    def test_runner_run_all_checks(self, test_db_path):
        with LintRunner(test_db_path) as runner:
            results = runner.run()
            assert len(results) > 0
            for result in results:
                assert isinstance(result, CheckResult)

    @pytest.mark.skipif(
        not (Path(__file__).parent / "fixtures" / "sword_test_minimal.duckdb").exists(),
        reason="Test database not found",
    )
    def test_runner_run_specific_checks(self, test_db_path):
        with LintRunner(test_db_path) as runner:
            results = runner.run(checks=["T001", "T002"])
            assert len(results) == 2
            check_ids = {r.check_id for r in results}
            assert check_ids == {"T001", "T002"}

    @pytest.mark.skipif(
        not (Path(__file__).parent / "fixtures" / "sword_test_minimal.duckdb").exists(),
        reason="Test database not found",
    )
    def test_runner_run_category_prefix(self, test_db_path):
        with LintRunner(test_db_path) as runner:
            results = runner.run(checks=["T"])
            for result in results:
                assert result.check_id.startswith("T")

    @pytest.mark.skipif(
        not (Path(__file__).parent / "fixtures" / "sword_test_minimal.duckdb").exists(),
        reason="Test database not found",
    )
    def test_runner_run_severity_filter(self, test_db_path):
        with LintRunner(test_db_path) as runner:
            results = runner.run(severity=Severity.ERROR)
            for result in results:
                assert result.severity == Severity.ERROR

    @pytest.mark.skipif(
        not (Path(__file__).parent / "fixtures" / "sword_test_minimal.duckdb").exists(),
        reason="Test database not found",
    )
    def test_runner_threshold_override(self, test_db_path):
        with LintRunner(test_db_path) as runner:
            runner.set_threshold("A002", 200.0)
            result = runner.run_check("A002")
            assert result.threshold == 200.0

    def test_runner_invalid_check_id(self, test_db_path):
        if not test_db_path.exists():
            pytest.skip("Test database not found")
        with LintRunner(test_db_path) as runner:
            with pytest.raises(ValueError):
                runner.run(checks=["Z999"])

    @pytest.mark.skipif(
        not (Path(__file__).parent / "fixtures" / "sword_test_minimal.duckdb").exists(),
        reason="Test database not found",
    )
    def test_runner_get_summary(self, test_db_path):
        with LintRunner(test_db_path) as runner:
            results = runner.run()
            summary = runner.get_summary(results)
            assert "total_checks" in summary
            assert "passed" in summary
            assert "failed" in summary
            assert "by_severity" in summary


# =============================================================================
# Formatter Tests
# =============================================================================


class TestConsoleFormatter:
    """Tests for ConsoleFormatter."""

    def test_format_with_results(self, sample_results):
        formatter = ConsoleFormatter(use_color=False)
        output = formatter.format(sample_results)
        assert "SWORD LINT REPORT" in output
        assert "T001" in output
        assert "T002" in output
        assert "A001" in output

    def test_format_with_color_disabled(self, sample_results):
        formatter = ConsoleFormatter(use_color=False)
        output = formatter.format(sample_results)
        # No ANSI codes when color disabled
        assert "\033[" not in output

    def test_format_verbose(self, sample_results):
        formatter = ConsoleFormatter(use_color=False, verbose=True)
        output = formatter.format(sample_results)
        assert "Sample issues" in output

    def test_format_summary(self, sample_results):
        formatter = ConsoleFormatter(use_color=False)
        summary = formatter.format_summary(sample_results)
        assert "FAIL" in summary or "WARN" in summary


class TestJsonFormatter:
    """Tests for JsonFormatter."""

    def test_format_valid_json(self, sample_results):
        formatter = JsonFormatter()
        output = formatter.format(sample_results)
        data = json.loads(output)
        assert "metadata" in data
        assert "summary" in data
        assert "results" in data

    def test_format_with_details(self, sample_results):
        formatter = JsonFormatter(include_details=True)
        output = formatter.format(sample_results)
        data = json.loads(output)
        # Should have details for at least one result
        has_details = any("details" in r for r in data["results"])
        assert has_details

    def test_format_summary_counts(self, sample_results):
        formatter = JsonFormatter()
        output = formatter.format(sample_results)
        data = json.loads(output)
        assert data["summary"]["total_checks"] == 3
        assert data["summary"]["passed"] == 1
        assert data["summary"]["failed"] == 2

    def test_format_exit_code(self, sample_results):
        formatter = JsonFormatter()
        output = formatter.format(sample_results)
        data = json.loads(output)
        # Has error, so exit_code should be 2
        assert data["exit_code"] == 2


class TestMarkdownFormatter:
    """Tests for MarkdownFormatter."""

    def test_format_valid_markdown(self, sample_results):
        formatter = MarkdownFormatter()
        output = formatter.format(sample_results)
        assert "# SWORD Lint Report" in output
        assert "## Summary" in output
        assert "## Check Results" in output

    def test_format_has_table(self, sample_results):
        formatter = MarkdownFormatter()
        output = formatter.format(sample_results)
        # Check for markdown table structure
        assert "| Status |" in output
        assert "|:------:|" in output

    def test_format_with_details(self, sample_results):
        formatter = MarkdownFormatter(include_details=True)
        output = formatter.format(sample_results)
        assert "## Issue Details" in output

    def test_format_summary(self, sample_results):
        formatter = MarkdownFormatter()
        summary = formatter.format_summary(sample_results)
        assert "FAIL" in summary or "WARN" in summary


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.db
@pytest.mark.skipif(
    not (Path(__file__).parent / "fixtures" / "sword_test_minimal.duckdb").exists(),
    reason="Test database not found",
)
class TestIntegration:
    """Integration tests running real checks."""

    def test_full_lint_run(self, test_db_path):
        """Run all checks and verify output."""
        with LintRunner(test_db_path) as runner:
            results = runner.run()

            # Should have results
            assert len(results) > 0

            # All results should be CheckResult
            for result in results:
                assert isinstance(result, CheckResult)
                assert result.elapsed_ms >= 0

            # Format with all formatters
            console = ConsoleFormatter(use_color=False).format(results)
            assert len(console) > 0

            json_out = JsonFormatter().format(results)
            json.loads(json_out)  # Should parse

            md_out = MarkdownFormatter().format(results)
            assert "# SWORD Lint Report" in md_out

    def test_region_filter(self, test_db_path):
        """Test region filtering."""
        with LintRunner(test_db_path) as runner:
            results = runner.run(region="NA")
            # Should still get results (test DB has NA data)
            assert len(results) > 0

    def test_check_timing(self, test_db_path):
        """Verify timing is captured."""
        with LintRunner(test_db_path) as runner:
            results = runner.run(checks=["T001"])
            assert len(results) == 1
            assert results[0].elapsed_ms > 0


# =============================================================================
# V011 OSM Name Continuity Tests
# =============================================================================


def _create_v011_test_data(conn, rows):
    """Create minimal reaches table for V011 testing.

    rows: list of (reach_id, region, x, y, river_name_osm, rch_id_dn_main, n_rch_down, n_rch_up)
    """

    conn.execute("""
        CREATE TABLE IF NOT EXISTS reaches (
            reach_id BIGINT,
            region VARCHAR,
            x DOUBLE,
            y DOUBLE,
            river_name_osm VARCHAR,
            rch_id_dn_main BIGINT,
            n_rch_down INTEGER,
            n_rch_up INTEGER
        )
    """)
    for row in rows:
        conn.execute(
            "INSERT INTO reaches VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            list(row),
        )


@pytest.mark.db
@pytest.mark.skipif(
    not (Path(__file__).parent / "fixtures" / "sword_test_minimal.duckdb").exists(),
    reason="Test database not found",
)
class TestV011OsmNameContinuity:
    """Tests for V011 osm_name_continuity check."""

    def test_v011_registered(self):
        """V011 should be in the registry."""
        check = get_check("V011")
        assert check is not None
        assert check.name == "osm_name_continuity"
        assert check.category == Category.V17C
        assert check.severity == Severity.WARNING

    def test_v011_missing_columns_passes(self, test_db_path):
        """V011 should pass gracefully when columns don't exist."""
        with LintRunner(test_db_path) as runner:
            results = runner.run(checks=["V011"])
            assert len(results) == 1
            r = results[0]
            assert r.check_id == "V011"
            assert r.passed is True
            assert r.total_checked == 0

    def test_v011_same_name_no_flag(self, tmp_path):
        """Same river_name_osm on 1:1 link should not be flagged."""
        import duckdb as _duckdb  # noqa: F811

        db_path = tmp_path / "v011_test.duckdb"
        conn = _duckdb.connect(str(db_path))
        _create_v011_test_data(
            conn,
            [
                # reach_id, region, x, y, river_name_osm, rch_id_dn_main, n_rch_down, n_rch_up
                (1001, "NA", -75.0, 45.0, "St. Lawrence", 1002, 1, 0),
                (1002, "NA", -75.1, 45.1, "St. Lawrence", None, 0, 1),
            ],
        )
        from src.updates.sword_duckdb.lint.checks.v17c import check_osm_name_continuity

        result = check_osm_name_continuity(conn, region="NA")
        assert result.passed is True
        assert result.issues_found == 0
        conn.close()

    def test_v011_different_name_1to1_flags(self, tmp_path):
        """Different river_name_osm on 1:1 link should be flagged."""
        import duckdb as _duckdb  # noqa: F811

        db_path = tmp_path / "v011_test.duckdb"
        conn = _duckdb.connect(str(db_path))
        _create_v011_test_data(
            conn,
            [
                (1001, "NA", -75.0, 45.0, "St. Lawrence", 1002, 1, 0),
                (1002, "NA", -75.1, 45.1, "Ottawa River", None, 0, 1),
            ],
        )
        from src.updates.sword_duckdb.lint.checks.v17c import check_osm_name_continuity

        result = check_osm_name_continuity(conn, region="NA")
        assert result.passed is False
        assert result.issues_found == 1
        conn.close()

    def test_v011_junction_not_flagged(self, tmp_path):
        """Different name at junction (n_rch_up > 1) should NOT be flagged."""
        import duckdb as _duckdb  # noqa: F811

        db_path = tmp_path / "v011_test.duckdb"
        conn = _duckdb.connect(str(db_path))
        _create_v011_test_data(
            conn,
            [
                (1001, "NA", -75.0, 45.0, "St. Lawrence", 1003, 1, 0),
                (1002, "NA", -74.9, 45.0, "Ottawa River", 1003, 1, 0),
                (
                    1003,
                    "NA",
                    -75.1,
                    45.1,
                    "St. Lawrence",
                    None,
                    0,
                    2,
                ),  # junction: n_rch_up=2
            ],
        )
        from src.updates.sword_duckdb.lint.checks.v17c import check_osm_name_continuity

        result = check_osm_name_continuity(conn, region="NA")
        assert result.passed is True
        assert result.issues_found == 0
        conn.close()

    def test_v011_null_name_skipped(self, tmp_path):
        """NULL river_name_osm should not be flagged."""
        import duckdb as _duckdb  # noqa: F811

        db_path = tmp_path / "v011_test.duckdb"
        conn = _duckdb.connect(str(db_path))
        _create_v011_test_data(
            conn,
            [
                (1001, "NA", -75.0, 45.0, None, 1002, 1, 0),
                (1002, "NA", -75.1, 45.1, "St. Lawrence", None, 0, 1),
            ],
        )
        from src.updates.sword_duckdb.lint.checks.v17c import check_osm_name_continuity

        result = check_osm_name_continuity(conn, region="NA")
        assert result.passed is True
        assert result.issues_found == 0
        conn.close()
