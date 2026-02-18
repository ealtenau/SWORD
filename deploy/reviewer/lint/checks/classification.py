"""
SWORD Lint - Classification Checks (C0xx)

Validates lake/river classification and type assignments.
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
    "C001",
    Category.CLASSIFICATION,
    Severity.WARNING,
    "River reaches between lake reaches (lake sandwich)",
)
def check_lake_sandwich(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Find river reaches sandwiched between lake reaches.

    A "lake sandwich" is a river reach (lakeflag=0) that has:
    - At least one upstream lake neighbor (lakeflag=1)
    - At least one downstream lake neighbor (lakeflag=1)

    These may be misclassified lake sections.
    """
    where_clause = f"AND r.region = '{region}'" if region else ""

    query = f"""
    WITH river_reaches AS (
        SELECT reach_id, region, x, y, reach_length, width, river_name
        FROM reaches
        WHERE lakeflag = 0 {where_clause.replace("r.", "")}
    ),
    has_lake_upstream AS (
        SELECT DISTINCT rt.reach_id, rt.region
        FROM reach_topology rt
        JOIN reaches r ON rt.neighbor_reach_id = r.reach_id AND rt.region = r.region
        WHERE rt.direction = 'up' AND r.lakeflag = 1
    ),
    has_lake_downstream AS (
        SELECT DISTINCT rt.reach_id, rt.region
        FROM reach_topology rt
        JOIN reaches r ON rt.neighbor_reach_id = r.reach_id AND rt.region = r.region
        WHERE rt.direction = 'down' AND r.lakeflag = 1
    )
    SELECT
        rr.reach_id, rr.region, rr.river_name, rr.x, rr.y,
        rr.reach_length, rr.width
    FROM river_reaches rr
    JOIN has_lake_upstream hu ON rr.reach_id = hu.reach_id AND rr.region = hu.region
    JOIN has_lake_downstream hd ON rr.reach_id = hd.reach_id AND rr.region = hd.region
    ORDER BY rr.reach_length DESC
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r WHERE lakeflag = 0 {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="C001",
        name="lake_sandwich",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="River reaches sandwiched between lake reaches (potential misclassification)",
    )


@register_check(
    "C002",
    Category.CLASSIFICATION,
    Severity.INFO,
    "Check lakeflag distribution",
)
def check_lakeflag_distribution(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Report lakeflag distribution.

    lakeflag values:
    - 0: river
    - 1: lake
    - 2: canal
    - 3: tidal
    """
    where_clause = f"AND region = '{region}'" if region else ""

    query = f"""
    SELECT
        lakeflag,
        CASE lakeflag
            WHEN 0 THEN 'river'
            WHEN 1 THEN 'lake'
            WHEN 2 THEN 'canal'
            WHEN 3 THEN 'tidal'
            ELSE 'unknown'
        END as type_name,
        COUNT(*) as count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) as pct
    FROM reaches
    WHERE lakeflag IS NOT NULL
    {where_clause}
    GROUP BY lakeflag
    ORDER BY lakeflag
    """

    details = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches WHERE lakeflag IS NOT NULL {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    # Flag unknown lakeflag values
    unknown_count = 0
    if len(details) > 0:
        unknown = details[~details["lakeflag"].isin([0, 1, 2, 3])]
        unknown_count = unknown["count"].sum() if len(unknown) > 0 else 0

    return CheckResult(
        check_id="C002",
        name="lakeflag_distribution",
        severity=Severity.INFO,
        passed=unknown_count == 0,
        total_checked=total,
        issues_found=unknown_count,
        issue_pct=100 * unknown_count / total if total > 0 else 0,
        details=details,
        description=f"Lakeflag distribution ({unknown_count} unknown values)",
    )


@register_check(
    "C003",
    Category.CLASSIFICATION,
    Severity.INFO,
    "Check type field distribution",
)
def check_type_distribution(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Report reach type distribution.

    Type values:
    - 1: river
    - 2: lake
    - 3: tidal river
    - 4: artificial (canal/dam)
    - 5: unassigned
    - 6: unreliable
    """
    where_clause = f"AND region = '{region}'" if region else ""

    # Check if 'type' column exists
    try:
        query = f"""
        SELECT
            type,
            CASE type
                WHEN 1 THEN 'river'
                WHEN 2 THEN 'lake'
                WHEN 3 THEN 'tidal_river'
                WHEN 4 THEN 'artificial'
                WHEN 5 THEN 'unassigned'
                WHEN 6 THEN 'unreliable'
                ELSE 'unknown'
            END as type_name,
            COUNT(*) as count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) as pct
        FROM reaches
        WHERE type IS NOT NULL
        {where_clause}
        GROUP BY type
        ORDER BY type
        """

        details = conn.execute(query).fetchdf()

        total_query = f"""
        SELECT COUNT(*) FROM reaches WHERE type IS NOT NULL {where_clause}
        """
        total = conn.execute(total_query).fetchone()[0]

        # Flag type=6 (unreliable) reaches
        unreliable_count = 0
        if len(details) > 0:
            unreliable = details[details["type"] == 6]
            unreliable_count = unreliable["count"].sum() if len(unreliable) > 0 else 0

        return CheckResult(
            check_id="C003",
            name="type_distribution",
            severity=Severity.INFO,
            passed=True,  # Informational
            total_checked=total,
            issues_found=unreliable_count,
            issue_pct=100 * unreliable_count / total if total > 0 else 0,
            details=details,
            description=f"Type distribution ({unreliable_count} unreliable reaches)",
        )

    except (duckdb.CatalogException, duckdb.BinderException):
        # 'type' column doesn't exist
        return CheckResult(
            check_id="C003",
            name="type_distribution",
            severity=Severity.INFO,
            passed=True,
            total_checked=0,
            issues_found=0,
            issue_pct=0,
            details=pd.DataFrame(),
            description="Type column not present in database",
        )


@register_check(
    "C004",
    Category.CLASSIFICATION,
    Severity.INFO,  # Changed to INFO - needs investigation which is authoritative
    "Lakeflag/type cross-tabulation",
)
def check_lakeflag_type_consistency(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Report lakeflag vs type cross-tabulation for investigation.

    SWORD type values (from reach_id last digit):
    - 1 = river
    - 3 = lake on river
    - 4 = dam or waterfall
    - 5 = unreliable topology
    - 6 = ghost reach

    SWORD lakeflag values:
    - 0 = river
    - 1 = lake
    - 2 = canal
    - 3 = tidal

    Key mismatches to investigate:
    - lakeflag=0 (river) but type=3 (lake_on_river): river section through lake?
    - lakeflag=1 (lake) but type=1 (river): misclassification?

    NOTE: Unclear which field is more authoritative - needs investigation.
    """
    where_clause = f"AND region = '{region}'" if region else ""

    try:
        # Get cross-tabulation
        query = f"""
        SELECT
            lakeflag, type, COUNT(*) as count
        FROM reaches
        WHERE lakeflag IS NOT NULL AND type IS NOT NULL
            {where_clause}
        GROUP BY lakeflag, type
        ORDER BY lakeflag, type
        """

        stats = conn.execute(query).fetchdf()

        # Count potential mismatches (for reporting)
        # Main expected: lakeflag=0/type=1 (river/river), lakeflag=1/type=3 (lake/lake_on_river)
        mismatch_query = f"""
        SELECT COUNT(*) FROM reaches
        WHERE lakeflag IS NOT NULL AND type IS NOT NULL
            AND NOT (
                    (lakeflag = 0 AND type IN (1, 4))   -- river: type=river or dam
                    OR (lakeflag = 1 AND type IN (3, 4)) -- lake: type=lake_on_river or dam
                    OR (lakeflag = 2 AND type IN (1, 4)) -- canal: type=river or artificial
                    OR (lakeflag = 3 AND type IN (3, 4)) -- tidal: lake_on_river or dam
                    OR type IN (5, 6)                    -- unreliable/ghost
                )
            {where_clause}
        """
        mismatch_count = conn.execute(mismatch_query).fetchone()[0]

        total_query = f"""
        SELECT COUNT(*) FROM reaches
        WHERE lakeflag IS NOT NULL AND type IS NOT NULL
        {where_clause}
        """
        total = conn.execute(total_query).fetchone()[0]

        return CheckResult(
            check_id="C004",
            name="lakeflag_type_consistency",
            severity=Severity.INFO,
            passed=True,  # Informational - needs investigation
            total_checked=total,
            issues_found=mismatch_count,
            issue_pct=100 * mismatch_count / total if total > 0 else 0,
            details=stats,
            description=f"Potential lakeflag/type mismatches: {mismatch_count} ({100 * mismatch_count / total:.1f}%) - needs investigation",
        )

    except (duckdb.CatalogException, duckdb.BinderException):
        # 'type' column doesn't exist
        return CheckResult(
            check_id="C004",
            name="lakeflag_type_consistency",
            severity=Severity.INFO,
            passed=True,
            total_checked=0,
            issues_found=0,
            issue_pct=0,
            details=pd.DataFrame(),
            description="Type column not present in database (cannot check consistency)",
        )
