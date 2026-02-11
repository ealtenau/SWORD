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
        description=f"Reaches where reach_length differs from geom length by >{tolerance*100:.0f}%",
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
# G007 – Width exceeds reach length
# ---------------------------------------------------------------------------


@register_check(
    "G007",
    Category.GEOMETRY,
    Severity.WARNING,
    "Width should not exceed reach length (implausible river geometry)",
)
def check_width_exceeds_length(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """Flag river/canal reaches where width > reach_length.

    Lakes (lakeflag=1) and tidal reaches (lakeflag=3) are excluded because
    width > length is physically normal for wide water bodies.
    """
    where_clause = f"AND r.region = '{region}'" if region else ""

    query = f"""
    SELECT
        r.reach_id, r.region, r.river_name, r.x, r.y,
        r.width, r.reach_length, r.lakeflag
    FROM reaches r
    WHERE r.width IS NOT NULL
        AND r.reach_length IS NOT NULL
        AND r.width > 0 AND r.width != -9999
        AND r.reach_length > 0 AND r.reach_length != -9999
        AND r.width > r.reach_length
        AND r.lakeflag NOT IN (1, 3)
        {where_clause}
    ORDER BY (r.width / r.reach_length) DESC
    """
    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r
    WHERE r.width IS NOT NULL AND r.width > 0 AND r.width != -9999
        AND r.reach_length IS NOT NULL AND r.reach_length > 0
        AND r.reach_length != -9999
        AND r.lakeflag NOT IN (1, 3)
        {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="G007",
        name="width_exceeds_length",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="River/canal reaches where width exceeds reach_length (lakes/tidal excluded)",
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
    min <= max for both axes."""
    where_clause = f"AND r.region = '{region}'" if region else ""

    # Columns may not all exist
    try:
        query = f"""
        SELECT
            r.reach_id, r.region, r.river_name,
            r.x, r.y, r.x_min, r.x_max, r.y_min, r.y_max,
            CASE
                WHEN r.x_min > r.x_max THEN 'inverted_x'
                WHEN r.y_min > r.y_max THEN 'inverted_y'
                WHEN r.x < r.x_min OR r.x > r.x_max THEN 'centroid_outside_x'
                WHEN r.y < r.y_min OR r.y > r.y_max THEN 'centroid_outside_y'
            END AS issue_type
        FROM reaches r
        WHERE (
                (r.x_min > r.x_max)
                OR (r.y_min > r.y_max)
                OR (r.x < r.x_min OR r.x > r.x_max)
                OR (r.y < r.y_min OR r.y > r.y_max)
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
