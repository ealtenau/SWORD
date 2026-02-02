"""
SWORD Lint - Geometry Checks (G0xx)

Validates reach geometry properties like length bounds.
"""

from typing import Optional

import duckdb
import pandas as pd

from ..core import (
    register_check,
    Category,
    Severity,
    CheckResult,
)


@register_check(
    "G001",
    Category.GEOMETRY,
    Severity.INFO,
    "Reach length should be between 100m and 50km (excl end_reach)",
    default_threshold=None,
)
def check_reach_length_bounds(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check that reach lengths are within expected bounds.

    Flags:
    - Too short: <100m (excluding end_reach=1 which are expected to be short)
    - Too long: >50km (unusual, may indicate missing junctions)
    """
    where_clause = f"AND r.region = '{region}'" if region else ""

    # Check for out-of-bounds reach lengths
    # Note: end_reach=1 reaches are excluded from "too short" check
    query = f"""
    SELECT
        r.reach_id, r.region, r.river_name, r.x, r.y,
        r.reach_length, r.end_reach, r.lakeflag,
        CASE
            WHEN r.reach_length < 100 AND COALESCE(r.end_reach, 0) != 1 THEN 'too_short'
            WHEN r.reach_length > 50000 THEN 'too_long'
        END as issue_type
    FROM reaches r
    WHERE r.reach_length IS NOT NULL
        AND r.reach_length > 0
        AND r.reach_length != -9999
        AND ((r.reach_length < 100 AND COALESCE(r.end_reach, 0) != 1)
             OR r.reach_length > 50000)
        {where_clause}
    ORDER BY
        CASE WHEN r.reach_length < 100 THEN r.reach_length ELSE 999999 - r.reach_length END
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r
    WHERE reach_length IS NOT NULL AND reach_length > 0 AND reach_length != -9999
    {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    # Separate counts for reporting
    too_short = len(issues[issues["issue_type"] == "too_short"]) if len(issues) > 0 else 0
    too_long = len(issues[issues["issue_type"] == "too_long"]) if len(issues) > 0 else 0

    return CheckResult(
        check_id="G001",
        name="reach_length_bounds",
        severity=Severity.INFO,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=f"Reaches with unusual length ({too_short} too short, {too_long} too long)",
    )


@register_check(
    "G002",
    Category.GEOMETRY,
    Severity.WARNING,
    "Node length sum should approximate reach length",
    default_threshold=0.1,  # 10% tolerance
)
def check_node_length_consistency(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check that sum of node lengths approximately equals reach length.

    Large discrepancies may indicate missing nodes or geometry issues.
    """
    tolerance = threshold if threshold is not None else 0.1
    where_clause = f"AND r.region = '{region}'" if region else ""
    where_clause_n = f"AND n.region = '{region}'" if region else ""

    query = f"""
    WITH node_sums AS (
        SELECT
            n.reach_id,
            n.region,
            SUM(n.node_length) as sum_node_length
        FROM nodes n
        WHERE n.node_length > 0 AND n.node_length != -9999
            {where_clause_n}
        GROUP BY n.reach_id, n.region
    )
    SELECT
        r.reach_id, r.region, r.river_name, r.x, r.y,
        r.reach_length,
        ns.sum_node_length,
        ABS(r.reach_length - ns.sum_node_length) as length_diff,
        ABS(r.reach_length - ns.sum_node_length) / r.reach_length as pct_diff
    FROM reaches r
    JOIN node_sums ns ON r.reach_id = ns.reach_id AND r.region = ns.region
    WHERE r.reach_length > 0 AND r.reach_length != -9999
        AND ABS(r.reach_length - ns.sum_node_length) / r.reach_length > {tolerance}
        {where_clause}
    ORDER BY pct_diff DESC
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(DISTINCT r.reach_id)
    FROM reaches r
    JOIN nodes n ON r.reach_id = n.reach_id AND r.region = n.region
    WHERE r.reach_length > 0 AND r.reach_length != -9999
    {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="G002",
        name="node_length_consistency",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=f"Reaches where node length sum differs from reach length by >{tolerance*100:.0f}%",
        threshold=tolerance,
    )


@register_check(
    "G003",
    Category.GEOMETRY,
    Severity.INFO,
    "Check for zero-length reaches",
)
def check_zero_length_reaches(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check for reaches with zero or negative length.

    These are geometry errors that need investigation.
    """
    where_clause = f"AND r.region = '{region}'" if region else ""

    query = f"""
    SELECT
        r.reach_id, r.region, r.river_name, r.x, r.y,
        r.reach_length, r.lakeflag, r.end_reach
    FROM reaches r
    WHERE (r.reach_length <= 0 AND r.reach_length != -9999)
       OR r.reach_length IS NULL
        {where_clause}
    ORDER BY r.reach_id
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r WHERE 1=1 {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="G003",
        name="zero_length_reaches",
        severity=Severity.INFO,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Reaches with zero or negative length (geometry error)",
    )
