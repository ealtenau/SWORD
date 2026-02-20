"""
SWORD Lint - Markdown Formatter

Generate markdown reports for documentation and GitHub.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, TextIO

from ..core import CheckResult, Severity


class MarkdownFormatter:
    """
    Format lint results as Markdown for reports and documentation.
    """

    def __init__(self, include_details: bool = True, max_detail_rows: int = 20):
        """
        Initialize Markdown formatter.

        Args:
            include_details: Whether to include issue detail tables
            max_detail_rows: Maximum rows to show in detail tables
        """
        self.include_details = include_details
        self.max_detail_rows = max_detail_rows

    def _severity_badge(self, severity: Severity, passed: bool) -> str:
        """Generate severity badge."""
        if passed:
            return "✅"
        return {
            Severity.ERROR: "❌",
            Severity.WARNING: "⚠️",
            Severity.INFO: "ℹ️",
        }.get(severity, "❓")

    def format(
        self,
        results: List[CheckResult],
        output: Optional[TextIO] = None,
        db_path: Optional[str] = None,
        region: Optional[str] = None,
    ) -> str:
        """
        Format lint results as Markdown.

        Args:
            results: List of CheckResult objects
            output: Optional file-like object to write to
            db_path: Database path (for metadata)
            region: Region filter (for metadata)

        Returns:
            Markdown string if output is None
        """
        lines = []

        # Header
        lines.append("# SWORD Lint Report")
        lines.append("")

        # Metadata
        lines.append("## Metadata")
        lines.append("")
        lines.append(f"- **Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        if db_path:
            lines.append(f"- **Database:** `{Path(db_path).name}`")
        if region:
            lines.append(f"- **Region:** {region}")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        errors = sum(1 for r in results if not r.passed and r.severity == Severity.ERROR)
        warnings = sum(1 for r in results if not r.passed and r.severity == Severity.WARNING)
        infos = sum(1 for r in results if not r.passed and r.severity == Severity.INFO)
        total_issues = sum(r.issues_found for r in results)
        total_time = sum(r.elapsed_ms for r in results)

        # Status badge
        if errors > 0:
            status = "❌ **FAILED**"
        elif warnings > 0:
            status = "⚠️ **WARNINGS**"
        else:
            status = "✅ **PASSED**"

        lines.append(f"**Status:** {status}")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Checks | {total} |")
        lines.append(f"| Passed | {passed} |")
        lines.append(f"| Failed | {failed} |")
        lines.append(f"| Errors | {errors} |")
        lines.append(f"| Warnings | {warnings} |")
        lines.append(f"| Info | {infos} |")
        lines.append(f"| Total Issues | {total_issues:,} |")
        lines.append(f"| Total Time | {total_time:.0f}ms |")
        lines.append("")

        # Results table
        lines.append("## Check Results")
        lines.append("")
        lines.append("| Status | ID | Name | Severity | Issues | % |")
        lines.append("|:------:|:---|:-----|:---------|-------:|--:|")

        for result in sorted(results, key=lambda r: r.check_id):
            badge = self._severity_badge(result.severity, result.passed)
            pct = f"{result.issue_pct:.2f}%" if result.total_checked > 0 else "N/A"
            lines.append(
                f"| {badge} | {result.check_id} | {result.name} | "
                f"{result.severity.value} | {result.issues_found:,} | {pct} |"
            )

        lines.append("")

        # Detailed results for failures
        if self.include_details:
            failed_results = [r for r in results if not r.passed and r.issues_found > 0]

            if failed_results:
                lines.append("## Issue Details")
                lines.append("")

                for result in sorted(failed_results, key=lambda r: r.check_id):
                    lines.append(f"### {result.check_id}: {result.name}")
                    lines.append("")
                    lines.append(f"**{result.description}**")
                    lines.append("")
                    lines.append(f"- **Severity:** {result.severity.value}")
                    lines.append(f"- **Issues:** {result.issues_found:,} / {result.total_checked:,}")
                    if result.threshold is not None:
                        lines.append(f"- **Threshold:** {result.threshold}")
                    lines.append("")

                    # Detail table
                    if result.details is not None and len(result.details) > 0:
                        # Get columns (limit to avoid huge tables)
                        df = result.details.head(self.max_detail_rows)
                        cols = list(df.columns)[:8]  # Max 8 columns

                        # Header
                        lines.append("| " + " | ".join(cols) + " |")
                        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

                        # Rows
                        for _, row in df.iterrows():
                            values = []
                            for col in cols:
                                val = row[col]
                                if isinstance(val, float):
                                    values.append(f"{val:.2f}")
                                else:
                                    values.append(str(val))
                            lines.append("| " + " | ".join(values) + " |")

                        if len(result.details) > self.max_detail_rows:
                            lines.append(
                                f"\n*... and {len(result.details) - self.max_detail_rows} more issues*"
                            )

                        lines.append("")

        # Footer
        lines.append("---")
        lines.append(f"*Report generated by SWORD Lint Framework*")
        lines.append("")

        text = "\n".join(lines)

        if output:
            output.write(text)
            return ""
        return text

    def format_summary(self, results: List[CheckResult]) -> str:
        """Format a brief Markdown summary."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        errors = sum(1 for r in results if not r.passed and r.severity == Severity.ERROR)
        warnings = sum(1 for r in results if not r.passed and r.severity == Severity.WARNING)

        if errors > 0:
            status = "❌ FAIL"
        elif warnings > 0:
            status = "⚠️ WARN"
        else:
            status = "✅ OK"

        return f"{status}: {passed}/{total} checks passed ({errors} errors, {warnings} warnings)"
