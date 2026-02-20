"""
SWORD Lint - Console Formatter

Pretty terminal output with colors and symbols.
"""

import sys
from typing import List, Optional, TextIO

from ..core import CheckResult, Severity


class ConsoleFormatter:
    """
    Format lint results for terminal output with colors and symbols.
    """

    # ANSI color codes
    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "cyan": "\033[36m",
        "dim": "\033[2m",
    }

    # Symbols
    SYMBOLS = {
        "pass": "✓",
        "fail": "✗",
        "warn": "⚠",
        "info": "ℹ",
    }

    def __init__(self, use_color: bool = True, verbose: bool = False):
        """
        Initialize console formatter.

        Args:
            use_color: Whether to use ANSI colors (auto-disabled if not a TTY)
            verbose: Whether to show detailed issue information
        """
        self.use_color = use_color and sys.stdout.isatty()
        self.verbose = verbose

    def _color(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if self.use_color:
            return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"
        return text

    def _severity_color(self, severity: Severity) -> str:
        """Get color for severity level."""
        return {
            Severity.ERROR: "red",
            Severity.WARNING: "yellow",
            Severity.INFO: "blue",
        }.get(severity, "reset")

    def _severity_symbol(self, severity: Severity, passed: bool) -> str:
        """Get symbol for result."""
        if passed:
            return self._color(self.SYMBOLS["pass"], "green")
        return {
            Severity.ERROR: self._color(self.SYMBOLS["fail"], "red"),
            Severity.WARNING: self._color(self.SYMBOLS["warn"], "yellow"),
            Severity.INFO: self._color(self.SYMBOLS["info"], "blue"),
        }.get(severity, self.SYMBOLS["info"])

    def format(
        self,
        results: List[CheckResult],
        output: Optional[TextIO] = None,
    ) -> str:
        """
        Format lint results for console output.

        Args:
            results: List of CheckResult objects
            output: Optional file-like object to write to (default: return string)

        Returns:
            Formatted string if output is None
        """
        lines = []

        # Header
        lines.append("")
        lines.append(self._color("=" * 70, "dim"))
        lines.append(self._color("SWORD LINT REPORT", "bold"))
        lines.append(self._color("=" * 70, "dim"))

        # Results by category
        current_category = None
        for result in sorted(results, key=lambda r: r.check_id):
            # Category header
            category = result.check_id[0]  # T, A, G, C
            if category != current_category:
                current_category = category
                cat_name = {
                    "T": "TOPOLOGY",
                    "A": "ATTRIBUTES",
                    "G": "GEOMETRY",
                    "C": "CLASSIFICATION",
                }.get(category, category)
                lines.append("")
                lines.append(self._color(f"── {cat_name} ──", "cyan"))

            # Result line
            symbol = self._severity_symbol(result.severity, result.passed)
            sev_label = f"[{result.severity.value.upper()}]"
            sev_colored = self._color(
                sev_label, self._severity_color(result.severity)
            )

            status = self._color("PASS", "green") if result.passed else self._color(
                "FAIL", self._severity_color(result.severity)
            )

            lines.append(
                f"  {symbol} {result.check_id} {result.name}: {status} "
                f"{sev_colored}"
            )

            # Stats line
            stats = f"     Checked: {result.total_checked:,} | Issues: {result.issues_found:,}"
            if result.total_checked > 0:
                stats += f" ({result.issue_pct:.2f}%)"
            if result.elapsed_ms > 0:
                stats += f" | {result.elapsed_ms:.0f}ms"
            lines.append(self._color(stats, "dim"))

            # Description
            lines.append(f"     {result.description}")

            # Verbose: show sample issues
            if self.verbose and not result.passed and result.details is not None:
                if len(result.details) > 0:
                    lines.append(self._color("     Sample issues:", "dim"))
                    sample = result.details.head(5)
                    for _, row in sample.iterrows():
                        reach_id = row.get("reach_id", "?")
                        region = row.get("region", "?")
                        lines.append(
                            self._color(f"       - {reach_id} ({region})", "dim")
                        )
                    if len(result.details) > 5:
                        lines.append(
                            self._color(
                                f"       ... and {len(result.details) - 5} more",
                                "dim",
                            )
                        )

        # Summary
        lines.append("")
        lines.append(self._color("=" * 70, "dim"))

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        errors = sum(1 for r in results if not r.passed and r.severity == Severity.ERROR)
        warnings = sum(1 for r in results if not r.passed and r.severity == Severity.WARNING)
        infos = sum(1 for r in results if not r.passed and r.severity == Severity.INFO)

        total_issues = sum(r.issues_found for r in results)
        total_time = sum(r.elapsed_ms for r in results)

        summary_parts = []
        if errors > 0:
            summary_parts.append(self._color(f"{errors} errors", "red"))
        if warnings > 0:
            summary_parts.append(self._color(f"{warnings} warnings", "yellow"))
        if infos > 0:
            summary_parts.append(self._color(f"{infos} info", "blue"))

        if summary_parts:
            lines.append(f"SUMMARY: {', '.join(summary_parts)}")
        else:
            lines.append(self._color("SUMMARY: All checks passed!", "green"))

        lines.append(
            f"Checks: {passed}/{total} passed | "
            f"Issues: {total_issues:,} | "
            f"Time: {total_time:.0f}ms"
        )
        lines.append(self._color("=" * 70, "dim"))
        lines.append("")

        text = "\n".join(lines)

        if output:
            output.write(text)
            return ""
        return text

    def format_summary(self, results: List[CheckResult]) -> str:
        """Format a brief one-line summary."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        errors = sum(1 for r in results if not r.passed and r.severity == Severity.ERROR)
        warnings = sum(1 for r in results if not r.passed and r.severity == Severity.WARNING)

        if errors > 0:
            return self._color(f"FAIL: {errors} errors, {warnings} warnings ({passed}/{total} passed)", "red")
        elif warnings > 0:
            return self._color(f"WARN: {warnings} warnings ({passed}/{total} passed)", "yellow")
        else:
            return self._color(f"OK: All {total} checks passed", "green")
