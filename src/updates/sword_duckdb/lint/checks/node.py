"""
SWORD Lint - Node Checks (N0xx)

Validates node-level data: spacing, dist_out continuity, boundary alignment,
count consistency, and index contiguity.
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
    "N003",
    Category.NETWORK,
    Severity.WARNING,
    "Adjacent nodes within a reach spaced >400m apart",
    default_threshold=400.0,
)
def check_node_spacing_gap(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """Flag adjacent nodes within a reach that are >400m apart using equirectangular distance."""
    max_spacing = threshold if threshold is not None else 400.0
    where_clause = f"AND n.region = '{region}'" if region else ""

    query = f"""
    WITH ordered_nodes AS (
        SELECT
            node_id, reach_id, region, x, y,
            LAG(node_id) OVER (PARTITION BY reach_id, region ORDER BY node_id) as prev_node_id,
            LAG(x) OVER (PARTITION BY reach_id, region ORDER BY node_id) as prev_x,
            LAG(y) OVER (PARTITION BY reach_id, region ORDER BY node_id) as prev_y
        FROM nodes n
        WHERE 1=1 {where_clause}
    )
    SELECT
        node_id, prev_node_id, reach_id, region, x, y,
        111000.0 * SQRT(
            POWER((x - prev_x) * COS(RADIANS((y + prev_y) / 2.0)), 2)
            + POWER(y - prev_y, 2)
        ) as spacing_m
    FROM ordered_nodes
    WHERE prev_node_id IS NOT NULL
        AND 111000.0 * SQRT(
            POWER((x - prev_x) * COS(RADIANS((y + prev_y) / 2.0)), 2)
            + POWER(y - prev_y, 2)
        ) > {max_spacing}
    ORDER BY spacing_m DESC
    LIMIT 10000
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM nodes n WHERE 1=1 {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="N003",
        name="node_spacing_gap",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=f"Adjacent nodes spaced >{max_spacing:.0f}m apart",
        threshold=max_spacing,
    )


@register_check(
    "N004",
    Category.NETWORK,
    Severity.ERROR,
    "Node dist_out must decrease along node_id order within a reach",
)
def check_node_dist_out_monotonicity(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """Check that node dist_out decreases as node_id increases within each reach."""
    where_clause = f"AND n.region = '{region}'" if region else ""

    query = f"""
    WITH ordered_nodes AS (
        SELECT
            node_id, reach_id, region, dist_out,
            LAG(dist_out) OVER (PARTITION BY reach_id, region ORDER BY node_id) as prev_dist_out,
            LAG(node_id) OVER (PARTITION BY reach_id, region ORDER BY node_id) as prev_node_id
        FROM nodes n
        WHERE dist_out IS NOT NULL AND dist_out != -9999
            {where_clause}
    )
    SELECT
        node_id, prev_node_id, reach_id, region,
        prev_dist_out, dist_out,
        (dist_out - prev_dist_out) as dist_out_increase
    FROM ordered_nodes
    WHERE prev_dist_out IS NOT NULL
        AND dist_out > prev_dist_out
    ORDER BY dist_out_increase DESC
    LIMIT 10000
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM nodes n
    WHERE dist_out IS NOT NULL AND dist_out != -9999
    {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="N004",
        name="node_dist_out_monotonicity",
        severity=Severity.ERROR,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Nodes where dist_out increases along node_id order (should decrease)",
    )


@register_check(
    "N005",
    Category.NETWORK,
    Severity.WARNING,
    "Adjacent node dist_out jump >600m within a reach",
    default_threshold=600.0,
)
def check_node_dist_out_jump(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """Flag large dist_out jumps between adjacent nodes."""
    max_jump = threshold if threshold is not None else 600.0
    where_clause = f"AND n.region = '{region}'" if region else ""

    query = f"""
    WITH ordered_nodes AS (
        SELECT
            node_id, reach_id, region, dist_out, x, y,
            LAG(dist_out) OVER (PARTITION BY reach_id, region ORDER BY node_id) as prev_dist_out,
            LAG(node_id) OVER (PARTITION BY reach_id, region ORDER BY node_id) as prev_node_id
        FROM nodes n
        WHERE dist_out IS NOT NULL AND dist_out != -9999
            {where_clause}
    )
    SELECT
        node_id, prev_node_id, reach_id, region, x, y,
        prev_dist_out, dist_out,
        ABS(dist_out - prev_dist_out) as dist_out_jump
    FROM ordered_nodes
    WHERE prev_dist_out IS NOT NULL
        AND ABS(dist_out - prev_dist_out) > {max_jump}
    ORDER BY dist_out_jump DESC
    LIMIT 10000
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM nodes n
    WHERE dist_out IS NOT NULL AND dist_out != -9999
    {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="N005",
        name="node_dist_out_jump",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=f"Adjacent nodes with dist_out jump >{max_jump:.0f}m",
        threshold=max_jump,
    )


@register_check(
    "N006",
    Category.NETWORK,
    Severity.WARNING,
    "Boundary node dist_out mismatch >1000m between connected reaches",
    default_threshold=1000.0,
)
def check_boundary_dist_out(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """Check dist_out continuity at reach boundaries."""
    max_diff = threshold if threshold is not None else 1000.0
    where_clause = f"AND rt.region = '{region}'" if region else ""

    query = f"""
    WITH last_node AS (
        SELECT reach_id, region,
            MAX(node_id) as last_node_id
        FROM nodes
        GROUP BY reach_id, region
    ),
    first_node AS (
        SELECT reach_id, region,
            MIN(node_id) as first_node_id
        FROM nodes
        GROUP BY reach_id, region
    ),
    boundary_pairs AS (
        SELECT
            rt.reach_id as up_reach,
            rt.neighbor_reach_id as dn_reach,
            rt.region,
            n1.dist_out as up_last_dist_out,
            n2.dist_out as dn_first_dist_out,
            n1.node_id as up_last_node,
            n2.node_id as dn_first_node
        FROM reach_topology rt
        JOIN last_node ln ON rt.reach_id = ln.reach_id AND rt.region = ln.region
        JOIN first_node fn ON rt.neighbor_reach_id = fn.reach_id AND rt.region = fn.region
        JOIN nodes n1 ON ln.last_node_id = n1.node_id AND ln.region = n1.region
        JOIN nodes n2 ON fn.first_node_id = n2.node_id AND fn.region = n2.region
        WHERE rt.direction = 'down'
            AND n1.dist_out IS NOT NULL AND n1.dist_out != -9999
            AND n2.dist_out IS NOT NULL AND n2.dist_out != -9999
            {where_clause}
    )
    SELECT
        up_reach, dn_reach, region,
        up_last_node, dn_first_node,
        up_last_dist_out, dn_first_dist_out,
        ABS(up_last_dist_out - dn_first_dist_out) as boundary_gap
    FROM boundary_pairs
    WHERE ABS(up_last_dist_out - dn_first_dist_out) > {max_diff}
    ORDER BY boundary_gap DESC
    LIMIT 10000
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reach_topology rt
    WHERE direction = 'down' {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="N006",
        name="boundary_dist_out",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=f"Reach boundary node dist_out gap >{max_diff:.0f}m",
        threshold=max_diff,
    )


@register_check(
    "N007",
    Category.NETWORK,
    Severity.WARNING,
    "Boundary nodes of connected reaches >400m apart geographically",
    default_threshold=400.0,
)
def check_boundary_geolocation(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """Check geographic proximity of boundary nodes at reach junctions."""
    max_dist = threshold if threshold is not None else 400.0
    where_clause = f"AND rt.region = '{region}'" if region else ""

    query = f"""
    WITH last_node AS (
        SELECT reach_id, region,
            MAX(node_id) as last_node_id
        FROM nodes
        GROUP BY reach_id, region
    ),
    first_node AS (
        SELECT reach_id, region,
            MIN(node_id) as first_node_id
        FROM nodes
        GROUP BY reach_id, region
    ),
    boundary_pairs AS (
        SELECT
            rt.reach_id as up_reach,
            rt.neighbor_reach_id as dn_reach,
            rt.region,
            n1.x as x1, n1.y as y1,
            n2.x as x2, n2.y as y2,
            n1.node_id as up_last_node,
            n2.node_id as dn_first_node
        FROM reach_topology rt
        JOIN last_node ln ON rt.reach_id = ln.reach_id AND rt.region = ln.region
        JOIN first_node fn ON rt.neighbor_reach_id = fn.reach_id AND rt.region = fn.region
        JOIN nodes n1 ON ln.last_node_id = n1.node_id AND ln.region = n1.region
        JOIN nodes n2 ON fn.first_node_id = n2.node_id AND fn.region = n2.region
        WHERE rt.direction = 'down'
            {where_clause}
    )
    SELECT
        up_reach, dn_reach, region,
        up_last_node, dn_first_node,
        x1, y1, x2, y2,
        111000.0 * SQRT(
            POWER((x1 - x2) * COS(RADIANS((y1 + y2) / 2.0)), 2)
            + POWER(y1 - y2, 2)
        ) as boundary_dist_m
    FROM boundary_pairs
    WHERE 111000.0 * SQRT(
            POWER((x1 - x2) * COS(RADIANS((y1 + y2) / 2.0)), 2)
            + POWER(y1 - y2, 2)
        ) > {max_dist}
    ORDER BY boundary_dist_m DESC
    LIMIT 10000
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reach_topology rt
    WHERE direction = 'down' {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="N007",
        name="boundary_geolocation",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=f"Boundary nodes >{max_dist:.0f}m apart at reach junctions",
        threshold=max_dist,
    )


@register_check(
    "N008",
    Category.NETWORK,
    Severity.ERROR,
    "Actual node count must match reaches.n_nodes",
)
def check_node_count_vs_n_nodes(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """Check that actual node count per reach matches the n_nodes column."""
    where_clause = f"AND r.region = '{region}'" if region else ""

    query = f"""
    WITH actual_counts AS (
        SELECT reach_id, region, COUNT(*) as actual_count
        FROM nodes
        GROUP BY reach_id, region
    )
    SELECT
        r.reach_id, r.region, r.river_name, r.x, r.y,
        r.n_nodes as expected_count,
        COALESCE(ac.actual_count, 0) as actual_count,
        ABS(r.n_nodes - COALESCE(ac.actual_count, 0)) as count_diff
    FROM reaches r
    LEFT JOIN actual_counts ac ON r.reach_id = ac.reach_id AND r.region = ac.region
    WHERE r.n_nodes != COALESCE(ac.actual_count, 0)
        {where_clause}
    ORDER BY count_diff DESC
    LIMIT 10000
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r WHERE 1=1 {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="N008",
        name="node_count_vs_n_nodes",
        severity=Severity.ERROR,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Reaches where actual node count doesn't match n_nodes column",
    )


@register_check(
    "N010",
    Category.NETWORK,
    Severity.WARNING,
    "Node indexes within a reach are not contiguous",
)
def check_node_index_contiguity(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """Check that node indexes (last 3 digits of node_id) are contiguous within each reach."""
    where_clause = f"AND n.region = '{region}'" if region else ""

    query = f"""
    WITH node_suffixes AS (
        SELECT
            reach_id, region,
            CAST(node_id AS BIGINT) % 1000 as suffix,
            node_id
        FROM nodes n
        WHERE 1=1 {where_clause}
    ),
    reach_stats AS (
        SELECT
            reach_id, region,
            MIN(suffix) as min_suffix,
            MAX(suffix) as max_suffix,
            COUNT(*) as node_count
        FROM node_suffixes
        GROUP BY reach_id, region
    )
    SELECT
        rs.reach_id, rs.region,
        rs.min_suffix, rs.max_suffix, rs.node_count,
        (rs.max_suffix - rs.min_suffix + 1) as expected_count,
        (rs.max_suffix - rs.min_suffix + 1) - rs.node_count as gap_count
    FROM reach_stats rs
    WHERE (rs.max_suffix - rs.min_suffix + 1) != rs.node_count
    ORDER BY gap_count DESC
    LIMIT 10000
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(DISTINCT reach_id) FROM nodes n WHERE 1=1 {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="N010",
        name="node_index_contiguity",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Reaches with non-contiguous node index suffixes (gaps in numbering)",
    )
