"""
SWORD Lint - Attribute Checks (A0xx)

Validates attribute values, monotonicity, and completeness.
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
    "A001",
    Category.ATTRIBUTES,
    Severity.ERROR,
    "WSE must decrease downstream (flow direction)",
    default_threshold=0.5,  # meters tolerance
)
def check_wse_monotonicity(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check that water surface elevation (WSE) decreases downstream.

    Flags reaches where downstream neighbor has higher WSE,
    which suggests flow direction error.
    """
    tolerance = threshold if threshold is not None else 0.5
    where_clause = f"AND r1.region = '{region}'" if region else ""

    query = f"""
    WITH reach_pairs AS (
        SELECT
            r1.reach_id,
            r1.region,
            r1.wse as wse_up,
            r2.wse as wse_down,
            r1.river_name,
            r1.x, r1.y,
            r1.lakeflag
        FROM reaches r1
        JOIN reach_topology rt ON r1.reach_id = rt.reach_id AND r1.region = rt.region
        JOIN reaches r2 ON rt.neighbor_reach_id = r2.reach_id AND rt.region = r2.region
        WHERE rt.direction = 'down'
            AND r1.wse > 0 AND r1.wse != -9999
            AND r2.wse > 0 AND r2.wse != -9999
            AND r1.lakeflag = 0  -- rivers only
            AND r2.lakeflag = 0
            {where_clause}
    )
    SELECT
        reach_id, region, river_name, x, y,
        wse_up, wse_down,
        (wse_down - wse_up) as wse_increase
    FROM reach_pairs
    WHERE wse_down > wse_up + {tolerance}
    ORDER BY wse_increase DESC
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r1
    WHERE wse > 0 AND wse != -9999 AND lakeflag = 0
    {where_clause.replace('r1.', '')}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="A001",
        name="wse_monotonicity",
        severity=Severity.ERROR,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Reaches where WSE increases downstream (potential flow direction error)",
        threshold=tolerance,
    )


@register_check(
    "A002",
    Category.ATTRIBUTES,
    Severity.WARNING,
    "Slope must be non-negative and reasonable (<100 m/km)",
    default_threshold=100.0,  # m/km
)
def check_slope_reasonableness(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check that slope values are reasonable.

    Flags:
    - Negative slopes (physically impossible for flow)
    - Extremely high slopes (>100 m/km by default)
    """
    max_slope = threshold if threshold is not None else 100.0
    where_clause = f"AND r.region = '{region}'" if region else ""

    # Check for negative slopes OR unreasonably high slopes
    query = f"""
    SELECT
        r.reach_id, r.region, r.river_name, r.x, r.y,
        r.slope, r.reach_length, r.lakeflag,
        CASE
            WHEN r.slope < 0 THEN 'negative'
            WHEN r.slope > {max_slope} THEN 'too_high'
        END as issue_type
    FROM reaches r
    WHERE r.slope IS NOT NULL
        AND r.slope != -9999
        AND (r.slope < 0 OR r.slope > {max_slope})
        AND r.lakeflag = 0  -- rivers only
        {where_clause}
    ORDER BY ABS(r.slope) DESC
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r
    WHERE slope IS NOT NULL AND slope != -9999 AND lakeflag = 0
    {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="A002",
        name="slope_reasonableness",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=f"Reaches with negative or excessive (>{max_slope} m/km) slope",
        threshold=max_slope,
    )


@register_check(
    "A003",
    Category.ATTRIBUTES,
    Severity.INFO,
    "Width generally increases downstream",
    default_threshold=0.3,  # 30% of upstream width
)
def check_width_trend(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check if width generally increases downstream.

    Flags reaches where downstream width is less than threshold * upstream width.
    Some decrease is normal, but dramatic decreases suggest issues.
    """
    ratio_threshold = threshold if threshold is not None else 0.3
    where_clause = f"AND r1.region = '{region}'" if region else ""

    query = f"""
    WITH reach_pairs AS (
        SELECT
            r1.reach_id,
            r1.region,
            r1.width as width_up,
            r2.width as width_down,
            r1.river_name,
            r1.x, r1.y,
            r1.lakeflag as lakeflag_up,
            r2.lakeflag as lakeflag_down
        FROM reaches r1
        JOIN reach_topology rt ON r1.reach_id = rt.reach_id AND r1.region = rt.region
        JOIN reaches r2 ON rt.neighbor_reach_id = r2.reach_id AND rt.region = r2.region
        WHERE rt.direction = 'down'
            AND r1.width > 0 AND r1.width != -9999
            AND r2.width > 0 AND r2.width != -9999
            AND r1.lakeflag = 0 AND r2.lakeflag = 0  -- rivers only
            {where_clause}
    )
    SELECT
        reach_id, region, river_name, x, y,
        width_up, width_down,
        ROUND(width_down / width_up, 3) as width_ratio
    FROM reach_pairs
    WHERE width_down < {ratio_threshold} * width_up
        AND width_up > 100  -- ignore small streams
    ORDER BY width_ratio ASC
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r1
    WHERE width > 100 AND width != -9999 AND lakeflag = 0
    {where_clause.replace('r1.', '')}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="A003",
        name="width_trend",
        severity=Severity.INFO,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=f"Reaches where downstream width < {ratio_threshold*100:.0f}% of upstream width",
        threshold=ratio_threshold,
    )


@register_check(
    "A004",
    Category.ATTRIBUTES,
    Severity.INFO,
    "Check attribute completeness for required fields",
)
def check_attribute_completeness(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check completeness of required attributes.

    Reports percentage of null/missing values for key attributes.
    """
    where_clause = f"AND region = '{region}'" if region else ""

    required_attrs = [
        "dist_out", "facc", "wse", "width", "slope",
        "reach_length", "lakeflag", "n_rch_up", "n_rch_down"
    ]

    # Build completeness query
    select_parts = []
    for attr in required_attrs:
        select_parts.append(f"""
            SUM(CASE WHEN {attr} IS NULL OR {attr} = -9999 THEN 1 ELSE 0 END) as {attr}_missing,
            COUNT(*) as {attr}_total
        """)

    query = f"""
    SELECT
        {', '.join(select_parts)}
    FROM reaches
    WHERE 1=1 {where_clause}
    """

    result = conn.execute(query).fetchone()

    # Build summary DataFrame
    rows = []
    idx = 0
    for attr in required_attrs:
        missing = result[idx]
        total = result[idx + 1]
        pct_missing = 100 * missing / total if total > 0 else 0
        rows.append({
            "attribute": attr,
            "missing_count": missing,
            "total_count": total,
            "pct_missing": round(pct_missing, 2),
        })
        idx += 2

    details = pd.DataFrame(rows)

    # Count attributes with >5% missing as "issues"
    high_missing = details[details["pct_missing"] > 5]

    total_query = f"""
    SELECT COUNT(*) FROM reaches WHERE 1=1 {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="A004",
        name="attribute_completeness",
        severity=Severity.INFO,
        passed=len(high_missing) == 0,
        total_checked=len(required_attrs),
        issues_found=len(high_missing),
        issue_pct=100 * len(high_missing) / len(required_attrs),
        details=details,
        description=f"Attribute completeness check ({len(high_missing)} attrs with >5% missing)",
    )


@register_check(
    "A005",
    Category.ATTRIBUTES,
    Severity.WARNING,
    "trib_flag must match actual tributary count",
)
def check_trib_flag_consistency(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check that trib_flag is consistent with actual upstream neighbor count.

    trib_flag should be 1 if n_rch_up > 1 (has tributaries), 0 otherwise.
    """
    where_clause = f"AND r.region = '{region}'" if region else ""

    query = f"""
    SELECT
        r.reach_id, r.region, r.river_name, r.x, r.y,
        r.trib_flag, r.n_rch_up,
        CASE
            WHEN r.trib_flag = 1 AND r.n_rch_up <= 1 THEN 'flag_1_but_no_trib'
            WHEN r.trib_flag = 0 AND r.n_rch_up > 1 THEN 'flag_0_but_has_trib'
        END as issue_type
    FROM reaches r
    WHERE r.trib_flag IS NOT NULL
        AND ((r.trib_flag = 1 AND r.n_rch_up <= 1)
             OR (r.trib_flag = 0 AND r.n_rch_up > 1))
        {where_clause}
    ORDER BY r.reach_id
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r
    WHERE trib_flag IS NOT NULL
    {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="A005",
        name="trib_flag_consistency",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Reaches where trib_flag doesn't match actual tributary count",
    )


@register_check(
    "A006",
    Category.ATTRIBUTES,
    Severity.INFO,
    "Check for extreme or outlier values in key attributes",
)
def check_attribute_outliers(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check for extreme outlier values in key attributes.

    Flags reaches with values outside typical ranges:
    - width > 50km
    - wse > 8000m (higher than any river)
    - facc > 10M km² (larger than Amazon basin)
    """
    where_clause = f"AND r.region = '{region}'" if region else ""

    query = f"""
    SELECT
        r.reach_id, r.region, r.river_name, r.x, r.y,
        r.width, r.wse, r.facc,
        CASE
            WHEN r.width > 50000 THEN 'extreme_width'
            WHEN r.wse > 8000 THEN 'extreme_wse'
            WHEN r.facc > 10000000 THEN 'extreme_facc'
        END as issue_type
    FROM reaches r
    WHERE (r.width > 50000
           OR r.wse > 8000
           OR r.facc > 10000000)
        AND r.width != -9999 AND r.wse != -9999 AND r.facc != -9999
        {where_clause}
    ORDER BY r.reach_id
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r WHERE 1=1 {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="A006",
        name="attribute_outliers",
        severity=Severity.INFO,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Reaches with extreme outlier values (width>50km, wse>8000m, facc>10M km²)",
    )
