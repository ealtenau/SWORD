"""
SWORD Lint Framework - Runner

Orchestrates lint check execution against DuckDB databases.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import duckdb

from .core import (
    CheckResult,
    CheckSpec,
    Category,
    Severity,
    get_registry,
    get_check,
    get_checks_by_category,
    get_checks_by_severity,
)


class LintRunner:
    """
    Orchestrates lint check execution against SWORD DuckDB databases.

    Usage:
        runner = LintRunner("sword_v17c.duckdb")
        results = runner.run()  # Run all checks
        results = runner.run(checks=["T001", "T002"])  # Specific checks
        results = runner.run(checks=["T"])  # All topology checks
        results = runner.run(region="NA", severity=Severity.ERROR)
        runner.close()
    """

    def __init__(self, db_path: Union[str, Path]):
        """
        Initialize the lint runner.

        Args:
            db_path: Path to SWORD DuckDB database
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        self.conn = duckdb.connect(str(self.db_path), read_only=True)
        self._threshold_overrides: Dict[str, float] = {}

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def set_threshold(self, check_id: str, threshold: float):
        """
        Override the threshold for a specific check.

        Args:
            check_id: Check ID (e.g., "A002")
            threshold: New threshold value
        """
        self._threshold_overrides[check_id] = threshold

    def _resolve_checks(
        self,
        checks: Optional[List[str]] = None,
        severity: Optional[Severity] = None,
    ) -> List[CheckSpec]:
        """
        Resolve check specifiers to CheckSpec objects.

        Args:
            checks: List of check IDs or category prefixes (e.g., ["T001", "A"])
            severity: Filter by severity

        Returns:
            List of CheckSpec objects to run
        """
        registry = get_registry()

        if checks is None:
            # All checks
            specs = list(registry.values())
        else:
            specs = []
            for check in checks:
                check = check.upper()
                if check in registry:
                    # Exact match
                    specs.append(registry[check])
                elif len(check) == 1:
                    # Category prefix (T, A, G, C)
                    prefix_map = {
                        "T": Category.TOPOLOGY,
                        "A": Category.ATTRIBUTES,
                        "G": Category.GEOMETRY,
                        "C": Category.CLASSIFICATION,
                    }
                    if check in prefix_map:
                        specs.extend(get_checks_by_category(prefix_map[check]))
                else:
                    # Unknown check
                    raise ValueError(f"Unknown check ID or prefix: {check}")

        # Filter by severity
        if severity is not None:
            specs = [s for s in specs if s.severity == severity]

        # Remove duplicates while preserving order
        seen = set()
        unique_specs = []
        for spec in specs:
            if spec.check_id not in seen:
                seen.add(spec.check_id)
                unique_specs.append(spec)

        return sorted(unique_specs, key=lambda s: s.check_id)

    def run(
        self,
        checks: Optional[List[str]] = None,
        region: Optional[str] = None,
        severity: Optional[Severity] = None,
    ) -> List[CheckResult]:
        """
        Run lint checks.

        Args:
            checks: List of check IDs or prefixes to run (default: all)
            region: Optional region filter (e.g., "NA")
            severity: Optional severity filter

        Returns:
            List of CheckResult objects
        """
        specs = self._resolve_checks(checks, severity)
        results = []

        for spec in specs:
            # Determine threshold
            threshold = self._threshold_overrides.get(
                spec.check_id, spec.default_threshold
            )

            # Run check with timing
            start = time.perf_counter()
            try:
                result = spec.check_fn(
                    conn=self.conn,
                    region=region,
                    threshold=threshold,
                )
                result.elapsed_ms = (time.perf_counter() - start) * 1000
            except Exception as e:
                # Create error result
                result = CheckResult(
                    check_id=spec.check_id,
                    name=spec.name,
                    severity=spec.severity,
                    passed=False,
                    total_checked=0,
                    issues_found=0,
                    issue_pct=0.0,
                    details=None,
                    description=f"ERROR: {str(e)}",
                    threshold=threshold,
                    elapsed_ms=(time.perf_counter() - start) * 1000,
                )

            results.append(result)

        return results

    def run_check(
        self,
        check_id: str,
        region: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> CheckResult:
        """
        Run a single check by ID.

        Args:
            check_id: Check ID (e.g., "T001")
            region: Optional region filter
            threshold: Optional threshold override

        Returns:
            CheckResult
        """
        spec = get_check(check_id)
        if spec is None:
            raise ValueError(f"Unknown check ID: {check_id}")

        # Use provided threshold, override, or default
        if threshold is None:
            threshold = self._threshold_overrides.get(
                check_id, spec.default_threshold
            )

        start = time.perf_counter()
        result = spec.check_fn(conn=self.conn, region=region, threshold=threshold)
        result.elapsed_ms = (time.perf_counter() - start) * 1000

        return result

    def get_summary(self, results: List[CheckResult]) -> Dict:
        """
        Generate a summary of lint results.

        Args:
            results: List of CheckResult objects

        Returns:
            Summary dict with counts by severity and pass/fail
        """
        summary = {
            "total_checks": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "by_severity": {
                "error": {"total": 0, "passed": 0, "failed": 0},
                "warning": {"total": 0, "passed": 0, "failed": 0},
                "info": {"total": 0, "passed": 0, "failed": 0},
            },
            "total_issues": sum(r.issues_found for r in results),
            "total_elapsed_ms": sum(r.elapsed_ms for r in results),
        }

        for r in results:
            sev = r.severity.value
            summary["by_severity"][sev]["total"] += 1
            if r.passed:
                summary["by_severity"][sev]["passed"] += 1
            else:
                summary["by_severity"][sev]["failed"] += 1

        return summary
