"""
SWORD Lint - Geometry Checks (G0xx)

Validates reach geometry properties like length bounds, validity,
self-intersection, sinuosity, bbox consistency, and endpoint alignment.
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


def _ensure_spatial(conn: duckdb.DuckDBPyConnection) -> bool:
    """Try to load the DuckDB spatial extension. Return True if available."""
    try:
        conn.execute("LOAD spatial")
        return True
    except Exception:
        try:
            conn.execute("INSTALL spatial; LOAD spatial")
            return True
        except Exception:
            return False


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
    too_short = (
        len(issues[issues["issue_type"] == "too_short"]) if len(issues) > 0 else 0
    )
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
        description=f"Reaches where node length sum differs from reach length by >{tolerance * 100:.0f}%",
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


# ---------------------------------------------------------------------------
# G004 – Self-intersecting geometries
# ---------------------------------------------------------------------------


@register_check(
    "G004",
    Category.GEOMETRY,
    Severity.WARNING,
    "Reach geometry should not self-intersect (ST_IsSimple)",
)
def check_self_intersection(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """Flag reaches whose geometry is not simple (self-intersecting)."""
    if not _ensure_spatial(conn):
        return CheckResult(
            check_id="G004",
            name="self_intersection",
            severity=Severity.WARNING,
            passed=True,
            total_checked=0,
            issues_found=0,
            issue_pct=0.0,
            details=pd.DataFrame(),
            description="SKIPPED – spatial extension unavailable",
        )

    where_clause = f"AND r.region = '{region}'" if region else ""

    query = f"""
    SELECT r.reach_id, r.region, r.river_name, r.x, r.y, r.reach_length
    FROM reaches r
    WHERE r.geom IS NOT NULL
        AND ST_IsSimple(r.geom) = FALSE
        {where_clause}
    ORDER BY r.reach_id
    """
    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r
    WHERE r.geom IS NOT NULL {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="G004",
        name="self_intersection",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Reaches with self-intersecting geometry",
    )


# ---------------------------------------------------------------------------
# G005 – reach_length vs geometry length
# ---------------------------------------------------------------------------


@register_check(
    "G005",
    Category.GEOMETRY,
    Severity.WARNING,
    "reach_length should be within 20% of geometry length",
    default_threshold=0.2,
)
def check_reach_length_vs_geom_length(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """Compare reach_length attribute to ST_Length_Spheroid(geom) (geodesic metres)."""
    tolerance = threshold if threshold is not None else 0.2

    if not _ensure_spatial(conn):
        return CheckResult(
            check_id="G005",
            name="reach_length_vs_geom_length",
            severity=Severity.WARNING,
            passed=True,
            total_checked=0,
            issues_found=0,
            issue_pct=0.0,
            details=pd.DataFrame(),
            description="SKIPPED – spatial extension unavailable",
            threshold=tolerance,
        )

    where_clause = f"AND r.region = '{region}'" if region else ""

    # ST_Length_Spheroid computes geodesic length on the WGS-84 ellipsoid
    # and returns metres directly.  DuckDB spatial expects (lat, lon) order
    # for spheroid functions, but our geometries are (lon, lat), so we
    # flip coordinates first.
    query = f"""
    WITH lengths AS (
        SELECT
            r.reach_id, r.region, r.river_name, r.x, r.y,
            r.reach_length,
            ST_Length_Spheroid(ST_FlipCoordinates(r.geom)) AS geom_length_m
        FROM reaches r
        WHERE r.geom IS NOT NULL
            AND r.reach_length IS NOT NULL
            AND r.reach_length > 0
            AND r.reach_length != -9999
            {where_clause}
    )
    SELECT
        reach_id, region, river_name, x, y,
        reach_length,
        ROUND(geom_length_m, 1) AS geom_length_m,
        ROUND(ABS(reach_length - geom_length_m) / reach_length, 3) AS pct_diff
    FROM lengths
    WHERE geom_length_m > 0
        AND ABS(reach_length - geom_length_m) / reach_length > {tolerance}
    ORDER BY pct_diff DESC
    """
    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r
    WHERE r.geom IS NOT NULL
        AND r.reach_length IS NOT NULL
        AND r.reach_length > 0
        AND r.reach_length != -9999
        {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="G005",
        name="reach_length_vs_geom_length",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=f"Reaches where reach_length differs from geom length by >{tolerance * 100:.0f}%",
        threshold=tolerance,
    )


# ---------------------------------------------------------------------------
# G006 – Excessive sinuosity
# ---------------------------------------------------------------------------


@register_check(
    "G006",
    Category.GEOMETRY,
    Severity.INFO,
    "Sinuosity > 10 is unusually high and may indicate geometry error",
    default_threshold=10.0,
)
def check_excessive_sinuosity(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """Flag reaches with extreme sinuosity values."""
    limit = threshold if threshold is not None else 10.0
    where_clause = f"AND r.region = '{region}'" if region else ""

    # sinuosity column may not exist – handle gracefully
    try:
        query = f"""
        SELECT
            r.reach_id, r.region, r.river_name, r.x, r.y,
            r.sinuosity, r.reach_length
        FROM reaches r
        WHERE r.sinuosity > {limit}
            {where_clause}
        ORDER BY r.sinuosity DESC
        """
        issues = conn.execute(query).fetchdf()
    except (duckdb.CatalogException, duckdb.BinderException):
        return CheckResult(
            check_id="G006",
            name="excessive_sinuosity",
            severity=Severity.INFO,
            passed=True,
            total_checked=0,
            issues_found=0,
            issue_pct=0.0,
            details=pd.DataFrame(),
            description="SKIPPED – sinuosity column not present",
            threshold=limit,
        )

    total_query = f"""
    SELECT COUNT(*) FROM reaches r
    WHERE r.sinuosity IS NOT NULL {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="G006",
        name="excessive_sinuosity",
        severity=Severity.INFO,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=f"Reaches with sinuosity > {limit}",
        threshold=limit,
    )


# ---------------------------------------------------------------------------
# G008 – Null geometry
# ---------------------------------------------------------------------------


@register_check(
    "G008",
    Category.GEOMETRY,
    Severity.ERROR,
    "Reach geometry must not be NULL",
)
def check_geom_not_null(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """Flag reaches with NULL geometry."""
    where_clause = f"AND r.region = '{region}'" if region else ""

    query = f"""
    SELECT r.reach_id, r.region, r.river_name, r.x, r.y, r.reach_length
    FROM reaches r
    WHERE r.geom IS NULL
        {where_clause}
    ORDER BY r.reach_id
    """
    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r WHERE 1=1 {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="G008",
        name="geom_not_null",
        severity=Severity.ERROR,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Reaches with NULL geometry",
    )


# ---------------------------------------------------------------------------
# G009 – Invalid geometry
# ---------------------------------------------------------------------------


@register_check(
    "G009",
    Category.GEOMETRY,
    Severity.ERROR,
    "Reach geometry must be valid (ST_IsValid)",
)
def check_geom_is_valid(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """Flag reaches with invalid geometry according to OGC rules."""
    if not _ensure_spatial(conn):
        return CheckResult(
            check_id="G009",
            name="geom_is_valid",
            severity=Severity.ERROR,
            passed=True,
            total_checked=0,
            issues_found=0,
            issue_pct=0.0,
            details=pd.DataFrame(),
            description="SKIPPED – spatial extension unavailable",
        )

    where_clause = f"AND r.region = '{region}'" if region else ""

    query = f"""
    SELECT r.reach_id, r.region, r.river_name, r.x, r.y, r.reach_length
    FROM reaches r
    WHERE r.geom IS NOT NULL
        AND ST_IsValid(r.geom) = FALSE
        {where_clause}
    ORDER BY r.reach_id
    """
    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r
    WHERE r.geom IS NOT NULL {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="G009",
        name="geom_is_valid",
        severity=Severity.ERROR,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Reaches with invalid geometry (ST_IsValid = FALSE)",
    )


# ---------------------------------------------------------------------------
# G010 – Minimum points in geometry
# ---------------------------------------------------------------------------


@register_check(
    "G010",
    Category.GEOMETRY,
    Severity.ERROR,
    "Reach geometry must have at least 2 points",
)
def check_geom_min_points(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """Flag reaches whose geometry has fewer than 2 points (degenerate)."""
    if not _ensure_spatial(conn):
        return CheckResult(
            check_id="G010",
            name="geom_min_points",
            severity=Severity.ERROR,
            passed=True,
            total_checked=0,
            issues_found=0,
            issue_pct=0.0,
            details=pd.DataFrame(),
            description="SKIPPED – spatial extension unavailable",
        )

    where_clause = f"AND r.region = '{region}'" if region else ""

    query = f"""
    SELECT
        r.reach_id, r.region, r.river_name, r.x, r.y,
        r.reach_length,
        ST_NPoints(r.geom) AS n_points
    FROM reaches r
    WHERE r.geom IS NOT NULL
        AND ST_NPoints(r.geom) < 2
        {where_clause}
    ORDER BY r.reach_id
    """
    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r
    WHERE r.geom IS NOT NULL {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="G010",
        name="geom_min_points",
        severity=Severity.ERROR,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Reaches with fewer than 2 geometry points",
    )


# ---------------------------------------------------------------------------
# G011 – Bounding box consistency
# ---------------------------------------------------------------------------


@register_check(
    "G011",
    Category.GEOMETRY,
    Severity.WARNING,
    "Centroid should be inside bbox; bbox min should not exceed max",
)
def check_bbox_consistency(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """Check that x/y centroid is inside x_min/x_max/y_min/y_max and that
    min <= max for both axes.  Uses a 1e-9 degree tolerance (~0.1 mm) to
    ignore floating-point noise on degenerate (single-axis) bboxes."""
    where_clause = f"AND r.region = '{region}'" if region else ""
    eps = 1e-9  # ~0.1 mm tolerance

    # Columns may not all exist
    try:
        query = f"""
        SELECT
            r.reach_id, r.region, r.river_name,
            r.x, r.y, r.x_min, r.x_max, r.y_min, r.y_max,
            CASE
                WHEN r.x_min - r.x_max > {eps} THEN 'inverted_x'
                WHEN r.y_min - r.y_max > {eps} THEN 'inverted_y'
                WHEN r.x < r.x_min - {eps} OR r.x > r.x_max + {eps} THEN 'centroid_outside_x'
                WHEN r.y < r.y_min - {eps} OR r.y > r.y_max + {eps} THEN 'centroid_outside_y'
            END AS issue_type
        FROM reaches r
        WHERE (
                (r.x_min - r.x_max > {eps})
                OR (r.y_min - r.y_max > {eps})
                OR (r.x < r.x_min - {eps} OR r.x > r.x_max + {eps})
                OR (r.y < r.y_min - {eps} OR r.y > r.y_max + {eps})
            )
            AND r.x IS NOT NULL AND r.y IS NOT NULL
            AND r.x_min IS NOT NULL AND r.x_max IS NOT NULL
            AND r.y_min IS NOT NULL AND r.y_max IS NOT NULL
            {where_clause}
        ORDER BY r.reach_id
        """
        issues = conn.execute(query).fetchdf()
    except duckdb.CatalogException:
        return CheckResult(
            check_id="G011",
            name="bbox_consistency",
            severity=Severity.WARNING,
            passed=True,
            total_checked=0,
            issues_found=0,
            issue_pct=0.0,
            details=pd.DataFrame(),
            description="SKIPPED – bbox columns not present",
        )

    total_query = f"""
    SELECT COUNT(*) FROM reaches r
    WHERE r.x IS NOT NULL AND r.y IS NOT NULL
        AND r.x_min IS NOT NULL AND r.x_max IS NOT NULL
        AND r.y_min IS NOT NULL AND r.y_max IS NOT NULL
        {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="G011",
        name="bbox_consistency",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Reaches with bbox issues (inverted min/max or centroid outside bbox)",
    )


# ---------------------------------------------------------------------------
# G012 – Endpoint alignment between connected reaches
# ---------------------------------------------------------------------------


@register_check(
    "G012",
    Category.GEOMETRY,
    Severity.INFO,
    "Connected reach endpoints should be within 500m of each other",
    default_threshold=500.0,
)
def check_endpoint_alignment(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """Check minimum gap between all 4 endpoint pairs of connected reaches.

    For each downstream connection A → B we check start/end of A against
    start/end of B and take the minimum distance.  Gaps > threshold (default
    500 m) are flagged.  Uses ST_Distance_Spheroid for geodesic metres.
    """
    gap_m = threshold if threshold is not None else 500.0

    if not _ensure_spatial(conn):
        return CheckResult(
            check_id="G012",
            name="endpoint_alignment",
            severity=Severity.INFO,
            passed=True,
            total_checked=0,
            issues_found=0,
            issue_pct=0.0,
            details=pd.DataFrame(),
            description="SKIPPED – spatial extension unavailable",
            threshold=gap_m,
        )

    where_clause = f"AND t.region = '{region}'" if region else ""

    query = f"""
    WITH pairs AS (
        SELECT
            t.reach_id AS src_id,
            t.neighbor_reach_id AS dst_id,
            ST_StartPoint(a.geom) AS a_start,
            ST_EndPoint(a.geom)   AS a_end,
            ST_StartPoint(b.geom) AS b_start,
            ST_EndPoint(b.geom)   AS b_end
        FROM reach_topology t
        JOIN reaches a ON a.reach_id = t.reach_id AND a.region = t.region
        JOIN reaches b ON b.reach_id = t.neighbor_reach_id AND b.region = t.region
        WHERE t.direction = 'down'
            AND a.geom IS NOT NULL AND b.geom IS NOT NULL
            {where_clause}
    ),
    dists AS (
        SELECT
            src_id, dst_id,
            LEAST(
                ST_Distance_Spheroid(ST_FlipCoordinates(a_start), ST_FlipCoordinates(b_start)),
                ST_Distance_Spheroid(ST_FlipCoordinates(a_start), ST_FlipCoordinates(b_end)),
                ST_Distance_Spheroid(ST_FlipCoordinates(a_end),   ST_FlipCoordinates(b_start)),
                ST_Distance_Spheroid(ST_FlipCoordinates(a_end),   ST_FlipCoordinates(b_end))
            ) AS min_gap_m
        FROM pairs
    )
    SELECT
        src_id AS reach_id,
        dst_id AS neighbor_reach_id,
        ROUND(min_gap_m, 1) AS min_gap_m
    FROM dists
    WHERE min_gap_m > {gap_m}
    ORDER BY min_gap_m DESC
    LIMIT 10000
    """
    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reach_topology t
    WHERE t.direction = 'down' {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="G012",
        name="endpoint_alignment",
        severity=Severity.INFO,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=f"Connected reach pairs with endpoint gap > {gap_m}m",
        threshold=gap_m,
    )


# ---------------------------------------------------------------------------
# G013 – Width greater than length
# ---------------------------------------------------------------------------


@register_check(
    "G013",
    Category.GEOMETRY,
    Severity.WARNING,
    "River reach width should not exceed reach length (lakes excluded)",
)
def check_width_gt_length(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """Flag non-lake reaches where width > reach_length."""
    where_clause = f"AND r.region = '{region}'" if region else ""

    query = f"""
    SELECT
        r.reach_id, r.region, r.river_name, r.x, r.y,
        r.width, r.reach_length, r.lakeflag
    FROM reaches r
    WHERE r.lakeflag = 0
        AND r.width > 0 AND r.width != -9999
        AND r.reach_length > 0 AND r.reach_length != -9999
        AND r.width > r.reach_length
        {where_clause}
    ORDER BY r.width / r.reach_length DESC
    """
    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r
    WHERE r.lakeflag = 0
        AND r.width > 0 AND r.width != -9999
        AND r.reach_length > 0 AND r.reach_length != -9999
        {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="G013",
        name="width_gt_length",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Non-lake reaches where width exceeds reach length",
    )


# ---------------------------------------------------------------------------
# G014 – Duplicate geometry
# ---------------------------------------------------------------------------


@register_check(
    "G014",
    Category.GEOMETRY,
    Severity.WARNING,
    "No two reaches should share identical geometry",
)
def check_duplicate_geometry(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """Flag pairs of reaches with identical geometry (ST_Equals)."""
    if not _ensure_spatial(conn):
        return CheckResult(
            check_id="G014",
            name="duplicate_geometry",
            severity=Severity.WARNING,
            passed=True,
            total_checked=0,
            issues_found=0,
            issue_pct=0.0,
            details=pd.DataFrame(),
            description="SKIPPED – spatial extension unavailable",
        )

    where_clause_a = f"AND a.region = '{region}'" if region else ""
    where_clause_r = f"AND r.region = '{region}'" if region else ""

    query = f"""
    SELECT
        a.reach_id AS reach_id_a,
        b.reach_id AS reach_id_b,
        a.region, a.river_name, a.x, a.y
    FROM reaches a
    JOIN reaches b
        ON a.region = b.region
        AND a.reach_id < b.reach_id
        AND ABS(a.x - b.x) < 0.01
        AND ABS(a.y - b.y) < 0.01
    WHERE a.geom IS NOT NULL AND b.geom IS NOT NULL
        AND ST_Equals(a.geom, b.geom)
        {where_clause_a}
    ORDER BY a.reach_id
    LIMIT 5000
    """
    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r
    WHERE r.geom IS NOT NULL {where_clause_r}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="G014",
        name="duplicate_geometry",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Reach pairs with identical geometry",
    )


# ---------------------------------------------------------------------------
# G015 – Node-to-reach distance
# ---------------------------------------------------------------------------


@register_check(
    "G015",
    Category.GEOMETRY,
    Severity.WARNING,
    "Nodes should be within 100m of their parent reach geometry",
    default_threshold=100.0,
)
def check_node_reach_distance(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """Flag nodes whose point geometry is >threshold metres from parent reach."""
    max_dist = threshold if threshold is not None else 100.0

    if not _ensure_spatial(conn):
        return CheckResult(
            check_id="G015",
            name="node_reach_distance",
            severity=Severity.WARNING,
            passed=True,
            total_checked=0,
            issues_found=0,
            issue_pct=0.0,
            details=pd.DataFrame(),
            description="SKIPPED – spatial extension unavailable",
            threshold=max_dist,
        )

    where_clause_n = f"AND n.region = '{region}'" if region else ""

    # ST_Distance_Spheroid only works on POINT types, not point-vs-linestring.
    # Use ST_Distance (Cartesian degrees) and convert to approximate metres.
    # 1 degree ≈ 111 km at equator — good enough for a lint threshold.
    deg_thresh = max_dist / 111000.0

    query = f"""
    SELECT
        n.node_id, n.reach_id, n.region, n.x, n.y,
        ROUND(ST_Distance(n.geom, r.geom) * 111000, 1) AS dist_m
    FROM nodes n
    JOIN reaches r ON n.reach_id = r.reach_id AND n.region = r.region
    WHERE n.geom IS NOT NULL AND r.geom IS NOT NULL
        AND ST_Distance(n.geom, r.geom) > {deg_thresh}
        {where_clause_n}
    ORDER BY dist_m DESC
    LIMIT 10000
    """
    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM nodes n
    JOIN reaches r ON n.reach_id = r.reach_id AND n.region = r.region
    WHERE n.geom IS NOT NULL AND r.geom IS NOT NULL
        {where_clause_n}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="G015",
        name="node_reach_distance",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=f"Nodes more than ~{max_dist}m from parent reach",
        threshold=max_dist,
    )


# ---------------------------------------------------------------------------
# G016 – Node spacing uniformity
# ---------------------------------------------------------------------------


@register_check(
    "G016",
    Category.GEOMETRY,
    Severity.INFO,
    "Node spacing should be roughly uniform within a reach",
    default_threshold=2.0,
)
def check_node_spacing(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """Flag nodes whose length is >2× or <0.5× the reach mean node length."""
    ratio = threshold if threshold is not None else 2.0
    where_clause = f"AND n.region = '{region}'" if region else ""

    query = f"""
    WITH reach_avg AS (
        SELECT
            reach_id, region,
            AVG(node_length) AS avg_len
        FROM nodes
        WHERE node_length > 0 AND node_length != -9999
        GROUP BY reach_id, region
        HAVING COUNT(*) >= 3
    )
    SELECT
        n.node_id, n.reach_id, n.region, n.x, n.y,
        n.node_length,
        ROUND(ra.avg_len, 1) AS avg_node_length,
        ROUND(n.node_length / ra.avg_len, 2) AS ratio
    FROM nodes n
    JOIN reach_avg ra ON n.reach_id = ra.reach_id AND n.region = ra.region
    WHERE n.node_length > 0 AND n.node_length != -9999
        AND (n.node_length > ra.avg_len * {ratio}
             OR n.node_length < ra.avg_len / {ratio})
        {where_clause}
    ORDER BY ABS(n.node_length / ra.avg_len - 1) DESC
    LIMIT 10000
    """
    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM nodes n
    WHERE n.node_length > 0 AND n.node_length != -9999
        {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="G016",
        name="node_spacing",
        severity=Severity.INFO,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=f"Nodes with spacing >{ratio}× or <1/{ratio}× reach mean",
        threshold=ratio,
    )


# ---------------------------------------------------------------------------
# G017 – Cross-reach nodes (node closer to another reach)
# ---------------------------------------------------------------------------


@register_check(
    "G017",
    Category.GEOMETRY,
    Severity.WARNING,
    "Nodes should not be closer to a different reach than their own",
    default_threshold=50.0,
)
def check_cross_reach_nodes(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """Two-pass: find nodes far from own reach, then check if closer to another."""
    own_dist = threshold if threshold is not None else 50.0

    if not _ensure_spatial(conn):
        return CheckResult(
            check_id="G017",
            name="cross_reach_nodes",
            severity=Severity.WARNING,
            passed=True,
            total_checked=0,
            issues_found=0,
            issue_pct=0.0,
            details=pd.DataFrame(),
            description="SKIPPED – spatial extension unavailable",
            threshold=own_dist,
        )

    where_clause_n = f"AND n.region = '{region}'" if region else ""

    # ST_Distance_Spheroid only works point-vs-point, so use Cartesian
    # ST_Distance (degrees) and convert to approximate metres.
    deg_thresh = own_dist / 111000.0

    # Two-pass approach:
    # 1. Find nodes > own_dist from their own reach
    # 2. For those, find if any other reach (bbox prefilter) is closer
    query = f"""
    WITH far_nodes AS (
        SELECT
            n.node_id, n.reach_id, n.region, n.x, n.y, n.geom AS node_geom,
            ST_Distance(n.geom, r.geom) AS own_dist_deg
        FROM nodes n
        JOIN reaches r ON n.reach_id = r.reach_id AND n.region = r.region
        WHERE n.geom IS NOT NULL AND r.geom IS NOT NULL
            AND ST_Distance(n.geom, r.geom) > {deg_thresh}
            {where_clause_n}
    ),
    nearest_other AS (
        SELECT
            fn.node_id, fn.reach_id, fn.region, fn.x, fn.y,
            ROUND(fn.own_dist_deg * 111000, 1) AS own_dist_m,
            r2.reach_id AS alt_reach_id,
            ROUND(ST_Distance(fn.node_geom, r2.geom) * 111000, 1) AS alt_dist_m
        FROM far_nodes fn
        JOIN reaches r2
            ON r2.region = fn.region
            AND r2.reach_id != fn.reach_id
            AND r2.geom IS NOT NULL
            AND ABS(r2.x - fn.x) < 0.05
            AND ABS(r2.y - fn.y) < 0.05
        WHERE ST_Distance(fn.node_geom, r2.geom) < fn.own_dist_deg
    )
    SELECT node_id, reach_id, region, x, y,
        own_dist_m, alt_reach_id, alt_dist_m
    FROM nearest_other
    QUALIFY ROW_NUMBER() OVER (PARTITION BY node_id ORDER BY alt_dist_m) = 1
    ORDER BY own_dist_m DESC
    LIMIT 5000
    """
    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM nodes n
    JOIN reaches r ON n.reach_id = r.reach_id AND n.region = r.region
    WHERE n.geom IS NOT NULL AND r.geom IS NOT NULL
        {where_clause_n}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="G017",
        name="cross_reach_nodes",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=f"Nodes >{own_dist}m from own reach and closer to another",
        threshold=own_dist,
    )


# ---------------------------------------------------------------------------
# G018 – dist_out vs reach_length consistency
# ---------------------------------------------------------------------------


@register_check(
    "G018",
    Category.GEOMETRY,
    Severity.WARNING,
    "dist_out difference between connected reaches should approximate reach_length",
    default_threshold=0.2,
)
def check_dist_out_vs_reach_length(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """Single-hop: |r1.dist_out - r2.dist_out - r1.reach_length| / r1.reach_length > tol."""
    tolerance = threshold if threshold is not None else 0.2
    where_clause = f"AND t.region = '{region}'" if region else ""

    query = f"""
    SELECT
        r1.reach_id, r1.region, r1.river_name, r1.x, r1.y,
        r1.dist_out AS dist_out_up,
        r2.dist_out AS dist_out_down,
        r1.reach_length,
        ROUND(ABS(r1.dist_out - r2.dist_out - r1.reach_length), 1) AS diff_m,
        ROUND(ABS(r1.dist_out - r2.dist_out - r1.reach_length)
              / r1.reach_length, 3) AS pct_diff
    FROM reach_topology t
    JOIN reaches r1 ON t.reach_id = r1.reach_id AND t.region = r1.region
    JOIN reaches r2 ON t.neighbor_reach_id = r2.reach_id AND t.region = r2.region
    WHERE t.direction = 'down'
        AND r1.reach_length > 0 AND r1.reach_length != -9999
        AND r1.dist_out IS NOT NULL AND r1.dist_out > 0 AND r1.dist_out != -9999
        AND r2.dist_out IS NOT NULL AND r2.dist_out > 0 AND r2.dist_out != -9999
        AND ABS(r1.dist_out - r2.dist_out - r1.reach_length)
            / r1.reach_length > {tolerance}
        {where_clause}
    ORDER BY pct_diff DESC
    LIMIT 10000
    """
    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reach_topology t
    JOIN reaches r1 ON t.reach_id = r1.reach_id AND t.region = r1.region
    JOIN reaches r2 ON t.neighbor_reach_id = r2.reach_id AND t.region = r2.region
    WHERE t.direction = 'down'
        AND r1.reach_length > 0 AND r1.reach_length != -9999
        AND r1.dist_out IS NOT NULL AND r1.dist_out > 0 AND r1.dist_out != -9999
        AND r2.dist_out IS NOT NULL AND r2.dist_out > 0 AND r2.dist_out != -9999
        {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="G018",
        name="dist_out_vs_reach_length",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=f"Reach pairs where dist_out gap differs from reach_length by >{tolerance * 100:.0f}%",
        threshold=tolerance,
    )


# ---------------------------------------------------------------------------
# G019 – Confluence geometry (upstream endpoints near downstream start)
# ---------------------------------------------------------------------------


@register_check(
    "G019",
    Category.GEOMETRY,
    Severity.INFO,
    "At confluences, upstream reach endpoints should be near downstream start",
    default_threshold=500.0,
)
def check_confluence_geometry(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """At junctions (n_rch_up >= 2): min gap between upstream endpoints and
    downstream startpoint. Uses LEAST-of-4-distances pattern."""
    gap_m = threshold if threshold is not None else 500.0

    if not _ensure_spatial(conn):
        return CheckResult(
            check_id="G019",
            name="confluence_geometry",
            severity=Severity.INFO,
            passed=True,
            total_checked=0,
            issues_found=0,
            issue_pct=0.0,
            details=pd.DataFrame(),
            description="SKIPPED – spatial extension unavailable",
            threshold=gap_m,
        )

    where_clause = f"AND ds.region = '{region}'" if region else ""

    query = f"""
    WITH confluences AS (
        SELECT reach_id, region
        FROM reaches
        WHERE n_rch_up >= 2 AND geom IS NOT NULL {where_clause.replace("ds.", "")}
    ),
    upstream_pairs AS (
        SELECT
            c.reach_id AS ds_id, c.region,
            t.neighbor_reach_id AS us_id
        FROM confluences c
        JOIN reach_topology t
            ON c.reach_id = t.reach_id AND c.region = t.region
        WHERE t.direction = 'up'
    ),
    dists AS (
        SELECT
            up.ds_id, up.us_id, up.region,
            LEAST(
                ST_Distance_Spheroid(
                    ST_FlipCoordinates(ST_StartPoint(us.geom)),
                    ST_FlipCoordinates(ST_StartPoint(ds.geom))),
                ST_Distance_Spheroid(
                    ST_FlipCoordinates(ST_StartPoint(us.geom)),
                    ST_FlipCoordinates(ST_EndPoint(ds.geom))),
                ST_Distance_Spheroid(
                    ST_FlipCoordinates(ST_EndPoint(us.geom)),
                    ST_FlipCoordinates(ST_StartPoint(ds.geom))),
                ST_Distance_Spheroid(
                    ST_FlipCoordinates(ST_EndPoint(us.geom)),
                    ST_FlipCoordinates(ST_EndPoint(ds.geom)))
            ) AS min_gap_m
        FROM upstream_pairs up
        JOIN reaches us ON up.us_id = us.reach_id AND up.region = us.region
        JOIN reaches ds ON up.ds_id = ds.reach_id AND up.region = ds.region
        WHERE us.geom IS NOT NULL
    )
    SELECT
        ds_id AS reach_id,
        us_id AS upstream_reach_id,
        region,
        ROUND(min_gap_m, 1) AS min_gap_m
    FROM dists
    WHERE min_gap_m > {gap_m}
    ORDER BY min_gap_m DESC
    LIMIT 10000
    """
    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*)
    FROM reaches ds
    JOIN reach_topology t ON ds.reach_id = t.reach_id AND ds.region = t.region
    JOIN reaches us ON t.neighbor_reach_id = us.reach_id AND t.region = us.region
    WHERE t.direction = 'up'
        AND ds.n_rch_up >= 2
        AND ds.geom IS NOT NULL AND us.geom IS NOT NULL
        {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="G019",
        name="confluence_geometry",
        severity=Severity.INFO,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=f"Confluence upstream-downstream endpoint gap > {gap_m}m",
        threshold=gap_m,
    )


# ---------------------------------------------------------------------------
# G020 – Bifurcation geometry (downstream endpoints near upstream end)
# ---------------------------------------------------------------------------


@register_check(
    "G020",
    Category.GEOMETRY,
    Severity.INFO,
    "At bifurcations, downstream reach endpoints should be near upstream end",
    default_threshold=500.0,
)
def check_bifurcation_geometry(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """At bifurcations (n_rch_down >= 2): min gap between downstream
    startpoints and upstream endpoint."""
    gap_m = threshold if threshold is not None else 500.0

    if not _ensure_spatial(conn):
        return CheckResult(
            check_id="G020",
            name="bifurcation_geometry",
            severity=Severity.INFO,
            passed=True,
            total_checked=0,
            issues_found=0,
            issue_pct=0.0,
            details=pd.DataFrame(),
            description="SKIPPED – spatial extension unavailable",
            threshold=gap_m,
        )

    where_clause = f"AND us.region = '{region}'" if region else ""

    query = f"""
    WITH bifurcations AS (
        SELECT reach_id, region
        FROM reaches
        WHERE n_rch_down >= 2 AND geom IS NOT NULL {where_clause.replace("us.", "")}
    ),
    downstream_pairs AS (
        SELECT
            b.reach_id AS us_id, b.region,
            t.neighbor_reach_id AS ds_id
        FROM bifurcations b
        JOIN reach_topology t
            ON b.reach_id = t.reach_id AND b.region = t.region
        WHERE t.direction = 'down'
    ),
    dists AS (
        SELECT
            dp.us_id, dp.ds_id, dp.region,
            LEAST(
                ST_Distance_Spheroid(
                    ST_FlipCoordinates(ST_StartPoint(us.geom)),
                    ST_FlipCoordinates(ST_StartPoint(ds.geom))),
                ST_Distance_Spheroid(
                    ST_FlipCoordinates(ST_StartPoint(us.geom)),
                    ST_FlipCoordinates(ST_EndPoint(ds.geom))),
                ST_Distance_Spheroid(
                    ST_FlipCoordinates(ST_EndPoint(us.geom)),
                    ST_FlipCoordinates(ST_StartPoint(ds.geom))),
                ST_Distance_Spheroid(
                    ST_FlipCoordinates(ST_EndPoint(us.geom)),
                    ST_FlipCoordinates(ST_EndPoint(ds.geom)))
            ) AS min_gap_m
        FROM downstream_pairs dp
        JOIN reaches us ON dp.us_id = us.reach_id AND dp.region = us.region
        JOIN reaches ds ON dp.ds_id = ds.reach_id AND dp.region = ds.region
        WHERE ds.geom IS NOT NULL
    )
    SELECT
        us_id AS reach_id,
        ds_id AS downstream_reach_id,
        region,
        ROUND(min_gap_m, 1) AS min_gap_m
    FROM dists
    WHERE min_gap_m > {gap_m}
    ORDER BY min_gap_m DESC
    LIMIT 10000
    """
    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*)
    FROM reaches us
    JOIN reach_topology t ON us.reach_id = t.reach_id AND us.region = t.region
    JOIN reaches ds ON t.neighbor_reach_id = ds.reach_id AND t.region = ds.region
    WHERE t.direction = 'down'
        AND us.n_rch_down >= 2
        AND us.geom IS NOT NULL AND ds.geom IS NOT NULL
        {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="G020",
        name="bifurcation_geometry",
        severity=Severity.INFO,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=f"Bifurcation upstream-downstream endpoint gap > {gap_m}m",
        threshold=gap_m,
    )


# ---------------------------------------------------------------------------
# G021 – Reach overlap (non-connected reaches intersecting)
# ---------------------------------------------------------------------------


@register_check(
    "G021",
    Category.GEOMETRY,
    Severity.INFO,
    "Non-connected reaches should not spatially overlap",
)
def check_reach_overlap(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """Flag non-connected reach pairs whose geometries intersect."""
    if not _ensure_spatial(conn):
        return CheckResult(
            check_id="G021",
            name="reach_overlap",
            severity=Severity.INFO,
            passed=True,
            total_checked=0,
            issues_found=0,
            issue_pct=0.0,
            details=pd.DataFrame(),
            description="SKIPPED – spatial extension unavailable",
        )

    where_clause_a = f"AND a.region = '{region}'" if region else ""
    where_clause_r = f"AND r.region = '{region}'" if region else ""

    query = f"""
    SELECT
        a.reach_id AS reach_id_a,
        b.reach_id AS reach_id_b,
        a.region
    FROM reaches a
    JOIN reaches b
        ON a.region = b.region
        AND a.reach_id < b.reach_id
        AND a.geom IS NOT NULL AND b.geom IS NOT NULL
        AND ABS(a.x - b.x) < 0.15
        AND ABS(a.y - b.y) < 0.15
    WHERE ST_Intersects(a.geom, b.geom)
        AND NOT EXISTS (
            SELECT 1 FROM reach_topology t
            WHERE t.reach_id = a.reach_id AND t.region = a.region
              AND t.neighbor_reach_id = b.reach_id
        )
        AND NOT EXISTS (
            SELECT 1 FROM reach_topology t
            WHERE t.reach_id = b.reach_id AND t.region = b.region
              AND t.neighbor_reach_id = a.reach_id
        )
        {where_clause_a}
    ORDER BY a.reach_id
    LIMIT 5000
    """
    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r
    WHERE r.geom IS NOT NULL {where_clause_r}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="G021",
        name="reach_overlap",
        severity=Severity.INFO,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Non-connected reach pairs with overlapping geometry",
    )
