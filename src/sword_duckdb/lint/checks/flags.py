"""
SWORD Lint - Flag Checks (FLxxx)

Validates iceflag, low_slope_flag, edit_flag, and swot_obs coverage.
"""

from typing import Optional

import duckdb

from ..core import (
    register_check,
    Category,
    Severity,
    CheckResult,
)


@register_check(
    "FL001",
    Category.FLAGS,
    Severity.INFO,
    "SWOT observation coverage statistics",
)
def check_swot_obs_coverage(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Report SWOT observation coverage across reaches.

    swot_obs indicates whether a reach has SWOT satellite observations.
    Reports the distribution and percentage of reaches with observations.
    """
    where_clause = f"AND r.region = '{region}'" if region else ""

    query = f"""
    SELECT
        swot_obs,
        COUNT(*) as reach_count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) as pct
    FROM reaches r
    WHERE swot_obs IS NOT NULL
        {where_clause}
    GROUP BY swot_obs
    ORDER BY swot_obs
    """

    stats = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r WHERE 1=1 {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    # Count reaches without SWOT observations
    no_obs_query = f"""
    SELECT COUNT(*) FROM reaches r
    WHERE (swot_obs IS NULL OR swot_obs = 0 OR swot_obs = -9999)
        {where_clause}
    """
    no_obs = conn.execute(no_obs_query).fetchone()[0]

    return CheckResult(
        check_id="FL001",
        name="swot_obs_coverage",
        severity=Severity.INFO,
        passed=True,
        total_checked=total,
        issues_found=no_obs,
        issue_pct=100 * no_obs / total if total > 0 else 0,
        details=stats,
        description=f"Reaches without SWOT observations: {no_obs} ({100 * no_obs / total:.1f}%)"
        if total > 0
        else "No data",
    )


@register_check(
    "FL002",
    Category.FLAGS,
    Severity.WARNING,
    "iceflag must be in {-9999, 0, 1, 2}",
)
def check_iceflag_values(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Validate iceflag values.

    Valid values:
    - -9999: no data
    - 0: no ice
    - 1: seasonal ice
    - 2: permanent ice
    """
    where_clause = f"AND r.region = '{region}'" if region else ""

    query = f"""
    SELECT
        r.reach_id, r.region, r.river_name, r.x, r.y,
        r.iceflag
    FROM reaches r
    WHERE r.iceflag IS NOT NULL
        AND r.iceflag NOT IN (-9999, 0, 1, 2)
        {where_clause}
    ORDER BY r.reach_id
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r
    WHERE iceflag IS NOT NULL
    {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="FL002",
        name="iceflag_values",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Reaches with invalid iceflag (not in {-9999, 0, 1, 2})",
    )


@register_check(
    "FL003",
    Category.FLAGS,
    Severity.WARNING,
    "low_slope_flag should be consistent with slope",
)
def check_low_slope_flag_consistency(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check low_slope_flag consistency with actual slope values.

    low_slope_flag=1 should correspond to reaches with very low slope.
    Flags cases where:
    - low_slope_flag=1 but slope is not low (>1e-4 m/m)
    - low_slope_flag=0 but slope is extremely low (<1e-6 m/m)
    """
    slope_high = threshold if threshold is not None else 1e-4
    where_clause = f"AND r.region = '{region}'" if region else ""

    query = f"""
    SELECT
        r.reach_id, r.region, r.river_name, r.x, r.y,
        r.low_slope_flag, r.slope, r.lakeflag,
        CASE
            WHEN r.low_slope_flag = 1 AND r.slope > {slope_high}
                THEN 'flagged_but_not_low'
            WHEN r.low_slope_flag = 0 AND r.slope >= 0 AND r.slope < 1e-6
                AND r.slope != -9999
                THEN 'unflagged_but_very_low'
        END as issue_type
    FROM reaches r
    WHERE r.slope IS NOT NULL
        AND r.slope != -9999
        AND r.lakeflag = 0
        AND (
            (r.low_slope_flag = 1 AND r.slope > {slope_high})
            OR (r.low_slope_flag = 0 AND r.slope >= 0 AND r.slope < 1e-6)
        )
        {where_clause}
    ORDER BY r.slope ASC
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r
    WHERE slope IS NOT NULL AND slope != -9999 AND lakeflag = 0
    {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="FL003",
        name="low_slope_flag_consistency",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Reaches where low_slope_flag disagrees with slope value",
        threshold=slope_high,
    )


@register_check(
    "FL004",
    Category.FLAGS,
    Severity.INFO,
    "edit_flag format validation and distribution",
)
def check_edit_flag_format(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Report edit_flag distribution and validate format.

    edit_flag is a comma-separated string of edit tags (e.g. 'facc_denoise_v3',
    'lake_sandwich', '7'). Reports the distribution of distinct values.
    """
    where_clause = f"AND r.region = '{region}'" if region else ""

    query = f"""
    SELECT
        edit_flag,
        COUNT(*) as reach_count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) as pct
    FROM reaches r
    WHERE edit_flag IS NOT NULL
        AND edit_flag != 'NaN'
        {where_clause}
    GROUP BY edit_flag
    ORDER BY reach_count DESC
    """

    stats = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r WHERE 1=1 {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    edited = int(stats["reach_count"].sum()) if len(stats) > 0 else 0

    return CheckResult(
        check_id="FL004",
        name="edit_flag_format",
        severity=Severity.INFO,
        passed=True,
        total_checked=total,
        issues_found=edited,
        issue_pct=100 * edited / total if total > 0 else 0,
        details=stats,
        description=f"Reaches with edit_flag set: {edited} ({100 * edited / total:.1f}%)"
        if total > 0
        else "No data",
    )
