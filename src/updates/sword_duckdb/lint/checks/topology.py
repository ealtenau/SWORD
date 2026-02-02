"""
SWORD Lint - Topology Checks (T0xx)

Validates topology consistency and flow direction properties.
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
    "T001",
    Category.TOPOLOGY,
    Severity.ERROR,
    "dist_out must decrease downstream",
    default_threshold=100.0,  # tolerance in meters
)
def check_dist_out_monotonicity(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check that dist_out (distance to outlet) decreases downstream.

    A reach's downstream neighbor should have a smaller dist_out value.
    Violations indicate topology errors or incorrect flow direction.
    """
    tolerance = threshold if threshold is not None else 100.0
    where_clause = f"AND r1.region = '{region}'" if region else ""

    query = f"""
    WITH reach_pairs AS (
        SELECT
            r1.reach_id,
            r1.region,
            r1.dist_out as dist_out_up,
            r2.dist_out as dist_out_down,
            r1.river_name,
            r1.x, r1.y
        FROM reaches r1
        JOIN reach_topology rt ON r1.reach_id = rt.reach_id AND r1.region = rt.region
        JOIN reaches r2 ON rt.neighbor_reach_id = r2.reach_id AND rt.region = r2.region
        WHERE rt.direction = 'down'
            AND r1.dist_out > 0 AND r1.dist_out != -9999
            AND r2.dist_out > 0 AND r2.dist_out != -9999
            {where_clause}
    )
    SELECT
        reach_id, region, river_name, x, y,
        dist_out_up, dist_out_down,
        (dist_out_down - dist_out_up) as dist_out_increase
    FROM reach_pairs
    WHERE dist_out_down > dist_out_up + {tolerance}
    ORDER BY dist_out_increase DESC
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r1
    WHERE dist_out > 0 AND dist_out != -9999
    {where_clause.replace('r1.', '')}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="T001",
        name="dist_out_monotonicity",
        severity=Severity.ERROR,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Reaches where dist_out increases downstream (topology error)",
        threshold=tolerance,
    )


@register_check(
    "T002",
    Category.TOPOLOGY,
    Severity.WARNING,
    "path_freq should increase toward outlets",
)
def check_path_freq_monotonicity(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check that path_freq increases toward outlets.

    At confluences, downstream path_freq should be >= max(upstream path_freqs).
    """
    where_clause = f"AND r1.region = '{region}'" if region else ""

    query = f"""
    WITH downstream_freqs AS (
        SELECT
            r1.reach_id,
            r1.region,
            r1.path_freq as pf_up,
            r2.path_freq as pf_down,
            r1.river_name,
            r1.x, r1.y
        FROM reaches r1
        JOIN reach_topology rt ON r1.reach_id = rt.reach_id AND r1.region = rt.region
        JOIN reaches r2 ON rt.neighbor_reach_id = r2.reach_id AND rt.region = r2.region
        WHERE rt.direction = 'down'
            AND r1.path_freq > 0 AND r1.path_freq != -9999
            AND r2.path_freq > 0 AND r2.path_freq != -9999
            {where_clause}
    )
    SELECT
        reach_id, region, river_name, x, y,
        pf_up, pf_down,
        (pf_up - pf_down) as pf_decrease
    FROM downstream_freqs
    WHERE pf_down < pf_up
    ORDER BY pf_decrease DESC
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r1
    WHERE path_freq > 0 AND path_freq != -9999
    {where_clause.replace('r1.', '')}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="T002",
        name="path_freq_monotonicity",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Reaches where path_freq decreases downstream",
    )


@register_check(
    "T003",
    Category.TOPOLOGY,
    Severity.WARNING,
    "facc should increase downstream",
)
def check_facc_monotonicity(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check that flow accumulation (facc) increases downstream.

    Downstream reaches should have >= facc values.
    """
    where_clause = f"AND r1.region = '{region}'" if region else ""

    query = f"""
    WITH reach_pairs AS (
        SELECT
            r1.reach_id,
            r1.region,
            r1.facc as facc_up,
            r2.facc as facc_down,
            r1.river_name,
            r1.x, r1.y,
            r1.lakeflag
        FROM reaches r1
        JOIN reach_topology rt ON r1.reach_id = rt.reach_id AND r1.region = rt.region
        JOIN reaches r2 ON rt.neighbor_reach_id = r2.reach_id AND rt.region = r2.region
        WHERE rt.direction = 'down'
            AND r1.facc > 0 AND r1.facc != -9999
            AND r2.facc > 0 AND r2.facc != -9999
            {where_clause}
    )
    SELECT
        reach_id, region, river_name, x, y,
        facc_up, facc_down,
        (facc_up - facc_down) as facc_decrease
    FROM reach_pairs
    WHERE facc_down < facc_up * 0.95  -- 5% tolerance
    ORDER BY facc_decrease DESC
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r1
    WHERE facc > 0 AND facc != -9999
    {where_clause.replace('r1.', '')}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="T003",
        name="facc_monotonicity",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Reaches where facc decreases downstream",
    )


@register_check(
    "T004",
    Category.TOPOLOGY,
    Severity.WARNING,
    "Orphan reaches with no neighbors",
)
def check_orphan_reaches(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Find reaches with no upstream AND no downstream neighbors.

    These are disconnected from the network (unless single-reach networks).
    """
    where_clause = f"AND r.region = '{region}'" if region else ""

    query = f"""
    SELECT
        r.reach_id, r.region, r.river_name, r.x, r.y,
        r.n_rch_up, r.n_rch_down, r.network, r.reach_length, r.width
    FROM reaches r
    WHERE r.n_rch_up = 0 AND r.n_rch_down = 0
        {where_clause}
    ORDER BY r.reach_length DESC
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r WHERE 1=1 {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="T004",
        name="orphan_reaches",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Reaches with no upstream or downstream neighbors (disconnected)",
    )


@register_check(
    "T005",
    Category.TOPOLOGY,
    Severity.ERROR,
    "n_rch_up/down must match actual topology counts",
)
def check_neighbor_count_consistency(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check that n_rch_up and n_rch_down match actual topology table counts.

    Mismatches indicate stale or corrupted topology data.
    """
    where_clause = f"AND r.region = '{region}'" if region else ""
    where_clause_rt = f"AND rt.region = '{region}'" if region else ""

    query = f"""
    WITH actual_counts AS (
        SELECT
            rt.reach_id,
            rt.region,
            SUM(CASE WHEN rt.direction = 'up' THEN 1 ELSE 0 END) as actual_up,
            SUM(CASE WHEN rt.direction = 'down' THEN 1 ELSE 0 END) as actual_down
        FROM reach_topology rt
        WHERE 1=1 {where_clause_rt}
        GROUP BY rt.reach_id, rt.region
    )
    SELECT
        r.reach_id, r.region, r.river_name, r.x, r.y,
        r.n_rch_up, r.n_rch_down,
        COALESCE(ac.actual_up, 0) as actual_up,
        COALESCE(ac.actual_down, 0) as actual_down,
        ABS(r.n_rch_up - COALESCE(ac.actual_up, 0)) as up_diff,
        ABS(r.n_rch_down - COALESCE(ac.actual_down, 0)) as down_diff
    FROM reaches r
    LEFT JOIN actual_counts ac ON r.reach_id = ac.reach_id AND r.region = ac.region
    WHERE (r.n_rch_up != COALESCE(ac.actual_up, 0)
           OR r.n_rch_down != COALESCE(ac.actual_down, 0))
        {where_clause}
    ORDER BY (ABS(r.n_rch_up - COALESCE(ac.actual_up, 0)) +
              ABS(r.n_rch_down - COALESCE(ac.actual_down, 0))) DESC
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r WHERE 1=1 {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="T005",
        name="neighbor_count_consistency",
        severity=Severity.ERROR,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Reaches where n_rch_up/n_rch_down doesn't match topology table",
    )


@register_check(
    "T006",
    Category.TOPOLOGY,
    Severity.INFO,
    "Connected component analysis",
)
def check_connected_components(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Analyze network connectivity and report component statistics.

    Uses the 'network' field to identify distinct connected components.
    Reports single-reach networks and very small networks.
    """
    where_clause = f"AND r.region = '{region}'" if region else ""

    # Get network size distribution
    query = f"""
    WITH network_sizes AS (
        SELECT
            network,
            region,
            COUNT(*) as reach_count,
            MIN(reach_id) as sample_reach_id
        FROM reaches r
        WHERE network IS NOT NULL AND network > 0
            {where_clause}
        GROUP BY network, region
    )
    SELECT
        network, region, reach_count, sample_reach_id
    FROM network_sizes
    WHERE reach_count = 1  -- Single-reach networks (most suspicious)
    ORDER BY network
    """

    issues = conn.execute(query).fetchdf()

    # Get total network count and reach count
    stats_query = f"""
    SELECT
        COUNT(DISTINCT network) as total_networks,
        COUNT(*) as total_reaches
    FROM reaches r
    WHERE network IS NOT NULL AND network > 0
        {where_clause}
    """
    stats = conn.execute(stats_query).fetchone()
    total_networks = stats[0] if stats else 0
    total_reaches = stats[1] if stats else 0

    return CheckResult(
        check_id="T006",
        name="connected_components",
        severity=Severity.INFO,
        passed=True,  # Informational, always "passes"
        total_checked=total_networks,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total_networks if total_networks > 0 else 0,
        details=issues,
        description=f"Single-reach networks ({len(issues)} of {total_networks} total networks)",
    )


@register_check(
    "T007",
    Category.TOPOLOGY,
    Severity.WARNING,
    "Topology reciprocity (A→B implies B→A)",
)
def check_topology_reciprocity(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check that topology relationships are reciprocal.

    If reach A has downstream neighbor B, then B should have upstream neighbor A.
    """
    where_clause = f"AND rt1.region = '{region}'" if region else ""

    query = f"""
    WITH forward_edges AS (
        SELECT
            rt1.reach_id as from_reach,
            rt1.neighbor_reach_id as to_reach,
            rt1.region,
            rt1.direction
        FROM reach_topology rt1
        WHERE 1=1 {where_clause}
    ),
    expected_reverse AS (
        SELECT
            to_reach as reach_id,
            from_reach as expected_neighbor,
            region,
            CASE WHEN direction = 'down' THEN 'up' ELSE 'down' END as expected_direction
        FROM forward_edges
    ),
    missing_reverse AS (
        SELECT
            er.reach_id,
            er.expected_neighbor,
            er.region,
            er.expected_direction
        FROM expected_reverse er
        LEFT JOIN reach_topology rt2
            ON er.reach_id = rt2.reach_id
            AND er.expected_neighbor = rt2.neighbor_reach_id
            AND er.region = rt2.region
            AND er.expected_direction = rt2.direction
        WHERE rt2.reach_id IS NULL
    )
    SELECT
        mr.reach_id, mr.region, mr.expected_neighbor, mr.expected_direction,
        r.river_name, r.x, r.y
    FROM missing_reverse mr
    JOIN reaches r ON mr.reach_id = r.reach_id AND mr.region = r.region
    ORDER BY mr.reach_id
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reach_topology rt1 WHERE 1=1 {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="T007",
        name="topology_reciprocity",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Missing reciprocal topology edges (A→B without B←A)",
    )


@register_check(
    "T008",
    Category.TOPOLOGY,
    Severity.ERROR,
    "dist_out must be non-negative (except fill value -9999)",
)
def check_dist_out_negative(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check that dist_out values are non-negative.

    Negative values (other than -9999 fill) indicate data corruption.
    """
    where_clause = f"AND r.region = '{region}'" if region else ""

    query = f"""
    SELECT
        r.reach_id, r.region, r.river_name, r.x, r.y,
        r.dist_out, r.n_rch_up, r.n_rch_down
    FROM reaches r
    WHERE r.dist_out < 0 AND r.dist_out != -9999
        {where_clause}
    ORDER BY r.dist_out ASC
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r
    WHERE dist_out IS NOT NULL AND dist_out != -9999
    {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="T008",
        name="dist_out_negative",
        severity=Severity.ERROR,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Reaches with negative dist_out (data corruption)",
    )


@register_check(
    "T009",
    Category.TOPOLOGY,
    Severity.ERROR,
    "dist_out=0 only valid for outlets (end_reach=2)",
)
def check_dist_out_zero_at_nonoutlet(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check that dist_out=0 only occurs at outlet reaches.

    Outlets are marked with end_reach=2 or n_rch_down=0.
    """
    where_clause = f"AND r.region = '{region}'" if region else ""

    query = f"""
    SELECT
        r.reach_id, r.region, r.river_name, r.x, r.y,
        r.dist_out, r.end_reach, r.n_rch_down
    FROM reaches r
    WHERE r.dist_out = 0
        AND r.end_reach != 2  -- Not marked as outlet
        AND r.n_rch_down > 0  -- Has downstream neighbors (not actually outlet)
        {where_clause}
    ORDER BY r.reach_id
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r
    WHERE dist_out = 0
    {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="T009",
        name="dist_out_zero_at_nonoutlet",
        severity=Severity.ERROR,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Reaches with dist_out=0 but not outlets (topology error)",
    )


@register_check(
    "T010",
    Category.TOPOLOGY,
    Severity.ERROR,
    "Headwaters must have path_freq >= 1",
)
def check_headwater_path_freq(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check that headwater reaches (n_rch_up=0) have path_freq >= 1.

    Headwaters should be visited at least once in any outlet-to-headwater traversal.
    """
    where_clause = f"AND r.region = '{region}'" if region else ""

    query = f"""
    SELECT
        r.reach_id, r.region, r.river_name, r.x, r.y,
        r.path_freq, r.n_rch_up, r.n_rch_down, r.type
    FROM reaches r
    WHERE r.n_rch_up = 0  -- Headwater
        AND (r.path_freq IS NULL OR r.path_freq < 1 OR r.path_freq = -9999)
        AND r.type NOT IN (5, 6)  -- Exclude unreliable/ghost
        {where_clause}
    ORDER BY r.reach_id
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r
    WHERE n_rch_up = 0 AND type NOT IN (5, 6)
    {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="T010",
        name="headwater_path_freq",
        severity=Severity.ERROR,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Headwater reaches with path_freq < 1 (traversal error)",
    )


@register_check(
    "T011",
    Category.TOPOLOGY,
    Severity.WARNING,
    "Zero path_freq only valid for disconnected reaches",
)
def check_path_freq_zero(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check that path_freq=0 only occurs for disconnected reaches.

    Connected reaches (n_rch_up > 0 OR n_rch_down > 0) should have path_freq > 0.
    """
    where_clause = f"AND r.region = '{region}'" if region else ""

    query = f"""
    SELECT
        r.reach_id, r.region, r.river_name, r.x, r.y,
        r.path_freq, r.n_rch_up, r.n_rch_down, r.type
    FROM reaches r
    WHERE r.path_freq = 0
        AND (r.n_rch_up > 0 OR r.n_rch_down > 0)  -- Connected
        AND r.type NOT IN (5, 6)  -- Exclude unreliable/ghost
        {where_clause}
    ORDER BY r.reach_id
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r
    WHERE (n_rch_up > 0 OR n_rch_down > 0) AND type NOT IN (5, 6)
    {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="T011",
        name="path_freq_zero",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description="Connected reaches with path_freq=0 (traversal bug)",
    )
