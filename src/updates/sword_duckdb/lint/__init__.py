"""
SWORD Lint Framework

Comprehensive, extensible linting framework for SWORD DuckDB databases.

Usage:
    from sword_duckdb.lint import LintRunner, Severity

    runner = LintRunner("sword_v17c.duckdb")
    results = runner.run()  # Run all checks
    results = runner.run(checks=["T001", "T002"])  # Specific checks
    results = runner.run(checks=["T"])  # All topology checks
    results = runner.run(region="NA", severity=Severity.ERROR)
    runner.close()

CLI Usage:
    python -m src.updates.sword_duckdb.lint.cli --db sword_v17c.duckdb
    python -m src.updates.sword_duckdb.lint.cli --db sword_v17c.duckdb --region NA
    python -m src.updates.sword_duckdb.lint.cli --db sword_v17c.duckdb --format json -o report.json

Check IDs:
    Topology (T0xx):
        T001: dist_out_monotonicity - dist_out must decrease downstream
        T002: path_freq_monotonicity - path_freq should increase toward outlets
        T003: facc_monotonicity - facc should increase downstream
        T004: orphan_reaches - Reaches with no neighbors
        T005: neighbor_count_consistency - n_rch_up/down must match topology
        T006: connected_components - Network connectivity analysis
        T007: topology_reciprocity - A→B implies B→A

    Attributes (A0xx):
        A001: wse_monotonicity - WSE must decrease downstream
        A002: slope_reasonableness - Slope must be non-negative and <100 m/km
        A003: width_trend - Width generally increases downstream
        A004: attribute_completeness - Required attributes present
        A005: trib_flag_consistency - trib_flag matches tributary count
        A006: attribute_outliers - Extreme values in key attributes

    Geometry (G0xx):
        G001: reach_length_bounds - Length between 100m-50km (excl end_reach)
        G002: node_length_consistency - Node sum ≈ reach length
        G003: zero_length_reaches - Reaches with zero length

    Classification (C0xx):
        C001: lake_sandwich - River between lakes
        C002: lakeflag_distribution - Lakeflag value distribution
        C003: type_distribution - Type field distribution
        C004: lakeflag_type_consistency - Lakeflag/type agreement

    Facc Anomaly (F0xx):
        F001: facc_width_ratio_anomaly - Facc/width > 5000 (MERIT corruption)
        F002: facc_jump_ratio - Facc >> upstream sum (entry points)
        F003: bifurcation_facc_divergence - Bifurcations with facc divergence > 0.9
        F004: facc_reach_acc_ratio - Facc >> expected from topology
        F005: facc_composite_anomaly - Composite anomaly score > 0.5
"""

from .core import (
    Severity,
    Category,
    CheckResult,
    CheckSpec,
    register_check,
    get_registry,
    get_check,
    get_checks_by_category,
    get_checks_by_severity,
    list_check_ids,
)
from .runner import LintRunner
from .formatters import ConsoleFormatter, JsonFormatter, MarkdownFormatter

# Import checks to register them
from . import checks  # noqa: F401

__all__ = [
    # Core types
    "Severity",
    "Category",
    "CheckResult",
    "CheckSpec",
    # Registry functions
    "register_check",
    "get_registry",
    "get_check",
    "get_checks_by_category",
    "get_checks_by_severity",
    "list_check_ids",
    # Runner
    "LintRunner",
    # Formatters
    "ConsoleFormatter",
    "JsonFormatter",
    "MarkdownFormatter",
]
