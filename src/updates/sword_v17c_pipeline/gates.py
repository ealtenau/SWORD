"""Lint-backed validation gates for v17c pipeline.

Gates run SQL-based lint checks between pipeline stages and fail fast on ERROR.
Uses existing LintRunner (read-only connection) from sword_duckdb.lint.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

from ..sword_duckdb.lint.core import CheckResult, Severity
from ..sword_duckdb.lint.runner import LintRunner

from .stages._logging import log


class GateFailure(Exception):
    """Raised when a gate detects failing lint checks."""

    def __init__(
        self,
        label: str,
        check_id: str,
        issues_found: int,
        *,
        failed_checks: Optional[List[str]] = None,
    ):
        self.label = label
        self.check_id = check_id
        self.issues_found = issues_found
        self.failed_checks = failed_checks or [check_id]
        super().__init__(
            f"Gate '{label}' failed: checks {self.failed_checks} "
            f"({issues_found} issues in {check_id})"
        )


@dataclass
class GateResult:
    """Result of running a validation gate."""

    label: str
    passed: bool
    results: List[CheckResult]
    failed_checks: List[str] = field(default_factory=list)


def run_gate(
    db_path: str,
    region: str,
    check_ids: List[str],
    label: str,
    *,
    fail_on: Optional[Set[Severity]] = None,
    artifact_dir: Optional[str] = None,
) -> GateResult:
    """Run lint checks as a pipeline gate.

    Parameters
    ----------
    db_path : str
        Path to DuckDB database.
    region : str
        Region code (e.g. "NA").
    check_ids : list[str]
        Lint check IDs to run (e.g. ["T005", "T012"]).
    label : str
        Human-readable gate name for logging/artifacts.
    fail_on : set[Severity], optional
        Severities that cause gate failure. Defaults to {ERROR}.
    artifact_dir : str, optional
        If set, write results JSON to ``artifact_dir/label.json``.

    Returns
    -------
    GateResult

    Raises
    ------
    GateFailure
        If any check at a ``fail_on`` severity fails.
    """
    if fail_on is None:
        fail_on = {Severity.ERROR}

    log(f"Gate '{label}': running checks {check_ids} for {region}...")

    with LintRunner(db_path) as runner:
        results = runner.run(checks=check_ids, region=region.upper())

    # Write artifact if requested (non-fatal — don't let I/O errors abort the gate)
    if artifact_dir:
        try:
            _write_artifact(artifact_dir, label, results)
        except OSError as e:
            log(f"Gate '{label}': WARNING — could not write artifact: {e}")

    # Evaluate pass/fail
    failed = []
    for r in results:
        if not r.passed and r.severity in fail_on:
            failed.append(r.check_id)

    passed = len(failed) == 0
    gate_result = GateResult(
        label=label, passed=passed, results=results, failed_checks=failed
    )

    if passed:
        log(f"Gate '{label}': PASSED ({len(results)} checks)")
    else:
        log(f"Gate '{label}': FAILED — {failed}")
        first = next(r for r in results if r.check_id == failed[0])
        raise GateFailure(
            label, first.check_id, first.issues_found, failed_checks=failed
        )

    return gate_result


def gate_source_data(db_path: str, region: str, **kwargs) -> GateResult:
    """Gate after loading source data + facc correction, before graph build.

    Checks:
        T005 — neighbor_count_consistency (ERROR)
        T012 — topology_referential_integrity (ERROR)
    """
    return run_gate(db_path, region, ["T005", "T012"], "source_data", **kwargs)


def gate_post_save(db_path: str, region: str, **kwargs) -> GateResult:
    """Gate after save_to_duckdb, verifying v17c output integrity.

    Checks:
        V001 — hydro_dist_out_monotonicity (ERROR)
        V005 — hydro_dist_out_coverage (ERROR)
        V007 — best_headwater_validity (WARNING)
        V008 — best_outlet_validity (WARNING)
        T001 — dist_out_monotonicity (ERROR)
        T002 — path_freq_monotonicity (WARNING)
    """
    return run_gate(
        db_path,
        region,
        ["V001", "V005", "V007", "V008", "T001", "T002"],
        "post_save",
        **kwargs,
    )


def _write_artifact(artifact_dir: str, label: str, results: List[CheckResult]) -> None:
    """Write gate results to JSON artifact file."""
    path = Path(artifact_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / f"{label}.json"

    records = []
    for r in results:
        records.append(
            {
                "check_id": r.check_id,
                "name": r.name,
                "severity": r.severity.value,
                "passed": r.passed,
                "total_checked": r.total_checked,
                "issues_found": r.issues_found,
                "issue_pct": r.issue_pct,
                "description": r.description,
                "elapsed_ms": r.elapsed_ms,
            }
        )

    out.write_text(json.dumps(records, indent=2))
    log(f"Gate artifact written to {out}")
