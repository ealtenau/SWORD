"""
SWORD Lint - Network Checks (Nxxx)

Validates main_side and stream_order consistency.
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
    "N001",
    Category.NETWORK,
    Severity.ERROR,
    "main_side must be in {0, 1, 2}",
)
def check_main_side_values(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Validate main_side values.

    Valid values:
    - 0: main channel (~95%)
    - 1: side channel (~3%)
    - 2: secondary outlet (~2%)
    """
    where_clause = f"AND r.region = '{region}'" if region else ""

    query = f"""
    SELECT
        r.reach_id, r.region, r.river_name, r.x, r.y,
        r.main_side, r.stream_order, r.lakeflag
    FROM reaches r
    WHERE r.main_side IS NOT NULL
        AND r.main_side NOT IN (0, 1, 2)
        {where_clause}
    ORDER BY r.reach_id
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r
    WHERE main_side IS NOT NULL
    {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="N001",
        name="main_side_values",
        severity=Severity.ERROR,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Reaches with invalid main_side (not in {0, 1, 2})",
    )


@register_check(
    "N002",
    Category.NETWORK,
    Severity.ERROR,
    "main_side=0 (main channel) should have valid stream_order",
)
def check_main_side_stream_order(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check that main channel reaches have a valid stream_order.

    main_side=0 means the reach is on the main channel. These should have
    a valid stream_order (not -9999). Side channels (main_side=1) and
    secondary outlets (main_side=2) are expected to lack stream_order.
    """
    where_clause = f"AND r.region = '{region}'" if region else ""

    # Check if 'type' column exists for filtering unreliable reaches
    try:
        conn.execute("SELECT type FROM reaches LIMIT 0")
        type_filter = "AND r.type NOT IN (5, 6)"
        type_filter_bare = "AND type NOT IN (5, 6)"
    except duckdb.BinderException:
        type_filter = ""
        type_filter_bare = ""

    query = f"""
    SELECT
        r.reach_id, r.region, r.river_name, r.x, r.y,
        r.main_side, r.stream_order, r.path_freq, r.width, r.lakeflag
    FROM reaches r
    WHERE r.main_side = 0
        AND (r.stream_order IS NULL OR r.stream_order = -9999)
        {type_filter}
        {where_clause}
    ORDER BY r.reach_id
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r
    WHERE main_side = 0 {type_filter_bare}
    {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="N002",
        name="main_side_stream_order",
        severity=Severity.ERROR,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Main channel reaches (main_side=0) with invalid stream_order (-9999)",
    )
