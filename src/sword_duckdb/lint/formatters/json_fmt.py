"""
SWORD Lint - JSON Formatter

Machine-readable JSON output for CI/CD integration.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, TextIO

from ..core import CheckResult, Severity


class JsonFormatter:
    """
    Format lint results as JSON for machine processing.
    """

    def __init__(self, include_details: bool = False, pretty: bool = True):
        """
        Initialize JSON formatter.

        Args:
            include_details: Whether to include full issue details (can be large)
            pretty: Whether to pretty-print JSON with indentation
        """
        self.include_details = include_details
        self.pretty = pretty

    def _result_to_dict(self, result: CheckResult) -> Dict[str, Any]:
        """Convert CheckResult to dictionary."""
        d = {
            "check_id": result.check_id,
            "name": result.name,
            "severity": result.severity.value,
            "passed": result.passed,
            "total_checked": result.total_checked,
            "issues_found": result.issues_found,
            "issue_pct": round(result.issue_pct, 4),
            "description": result.description,
            "elapsed_ms": round(result.elapsed_ms, 2),
        }

        if result.threshold is not None:
            d["threshold"] = result.threshold

        if self.include_details and result.details is not None:
            if len(result.details) > 0:
                # Convert DataFrame to list of dicts
                d["details"] = result.details.to_dict(orient="records")
            else:
                d["details"] = []

        return d

    def format(
        self,
        results: List[CheckResult],
        output: Optional[TextIO] = None,
        db_path: Optional[str] = None,
        region: Optional[str] = None,
    ) -> str:
        """
        Format lint results as JSON.

        Args:
            results: List of CheckResult objects
            output: Optional file-like object to write to
            db_path: Database path (for metadata)
            region: Region filter (for metadata)

        Returns:
            JSON string if output is None
        """
        # Build summary stats
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        by_severity = {
            "error": {"total": 0, "passed": 0, "failed": 0},
            "warning": {"total": 0, "passed": 0, "failed": 0},
            "info": {"total": 0, "passed": 0, "failed": 0},
        }

        for r in results:
            sev = r.severity.value
            by_severity[sev]["total"] += 1
            if r.passed:
                by_severity[sev]["passed"] += 1
            else:
                by_severity[sev]["failed"] += 1

        # Build output structure
        data = {
            "metadata": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "database": str(db_path) if db_path else None,
                "region": region,
                "version": "1.0",
            },
            "summary": {
                "total_checks": total,
                "passed": passed,
                "failed": failed,
                "by_severity": by_severity,
                "total_issues": sum(r.issues_found for r in results),
                "total_elapsed_ms": round(sum(r.elapsed_ms for r in results), 2),
            },
            "results": [self._result_to_dict(r) for r in results],
        }

        # Determine exit code
        has_errors = by_severity["error"]["failed"] > 0
        has_warnings = by_severity["warning"]["failed"] > 0
        data["exit_code"] = 2 if has_errors else (1 if has_warnings else 0)

        # Format JSON
        indent = 2 if self.pretty else None
        text = json.dumps(data, indent=indent, default=str)

        if output:
            output.write(text)
            return ""
        return text

    def format_summary(self, results: List[CheckResult]) -> str:
        """Format a brief JSON summary."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        errors = sum(
            1 for r in results if not r.passed and r.severity == Severity.ERROR
        )
        warnings = sum(
            1 for r in results if not r.passed and r.severity == Severity.WARNING
        )

        data = {
            "passed": passed,
            "total": total,
            "errors": errors,
            "warnings": warnings,
            "status": "error" if errors > 0 else ("warning" if warnings > 0 else "ok"),
        }

        return json.dumps(data)
