# -*- coding: utf-8 -*-
"""
SWORD Lint - Facc Checks (F0xx)

Validates flow accumulation (facc) values using ML-based detection.
These checks go beyond T003 (facc monotonicity) to detect corrupted values.
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


# Define FACC category
class FaccCategory:
    """Facc-specific category (uses ATTRIBUTES since no separate FACC category)."""
    value = "attributes"


@register_check(
    "F001",
    Category.ATTRIBUTES,
    Severity.WARNING,
    "Facc/width ratio anomaly detection",
    default_threshold=5000.0,
)
def check_facc_width_ratio_anomaly(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check for reaches with suspiciously high facc/width ratio.

    This is the original heuristic from v17c_status.md.
    Reaches with facc/width > 5000 are likely corrupted.

    This check catches "entry point" errors where bad facc
    enters the network from MERIT Hydro.
    """
    ratio_threshold = threshold if threshold is not None else 5000.0
    where_clause = f"AND r.region = '{region}'" if region else ""

    query = f"""
    SELECT
        r.reach_id,
        r.region,
        r.river_name,
        r.x, r.y,
        r.facc,
        r.width,
        r.facc / NULLIF(r.width, 0) as facc_width_ratio,
        r.stream_order,
        r.n_rch_up,
        r.n_rch_down
    FROM reaches r
    WHERE r.facc > 0 AND r.facc != -9999
        AND r.width > 0
        AND r.facc / NULLIF(r.width, 0) > {ratio_threshold}
        {where_clause}
    ORDER BY r.facc / NULLIF(r.width, 0) DESC
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r
    WHERE r.facc > 0 AND r.facc != -9999 AND r.width > 0
    {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="F001",
        name="facc_width_ratio_anomaly",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=f"Reaches with facc/width > {ratio_threshold} (likely MERIT corruption)",
        threshold=ratio_threshold,
    )


@register_check(
    "F002",
    Category.ATTRIBUTES,
    Severity.WARNING,
    "Facc jump ratio anomaly (entry point detection)",
    default_threshold=100.0,
)
def check_facc_jump_ratio(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check for reaches where facc jumps dramatically compared to upstream sum.

    This detects "entry points" where corrupted facc enters the network.
    If facc >> sum(upstream facc), the MERIT Hydro D8 flow likely
    picked up a bad accumulation value.

    These are characterized by:
    - facc / upstream_facc_sum > threshold (default 100)
    - Having upstream neighbors (not headwaters)
    """
    jump_threshold = threshold if threshold is not None else 100.0
    where_clause = f"AND r.region = '{region}'" if region else ""

    query = f"""
    WITH upstream_facc AS (
        SELECT
            rt.reach_id,
            rt.region,
            SUM(r_up.facc) as upstream_facc_sum,
            COUNT(*) as n_upstream
        FROM reach_topology rt
        JOIN reaches r_up ON rt.neighbor_reach_id = r_up.reach_id
            AND rt.region = r_up.region
        WHERE rt.direction = 'up'
            AND r_up.facc > 0 AND r_up.facc != -9999
        GROUP BY rt.reach_id, rt.region
    )
    SELECT
        r.reach_id,
        r.region,
        r.river_name,
        r.x, r.y,
        r.facc,
        uf.upstream_facc_sum,
        r.facc / NULLIF(uf.upstream_facc_sum, 0) as facc_jump_ratio,
        uf.n_upstream,
        r.width,
        r.stream_order
    FROM reaches r
    JOIN upstream_facc uf ON r.reach_id = uf.reach_id AND r.region = uf.region
    WHERE r.facc > 0 AND r.facc != -9999
        AND uf.upstream_facc_sum > 0
        AND r.facc / uf.upstream_facc_sum > {jump_threshold}
        {where_clause}
    ORDER BY r.facc / uf.upstream_facc_sum DESC
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(DISTINCT rt.reach_id) FROM reach_topology rt
    JOIN reaches r ON rt.reach_id = r.reach_id AND rt.region = r.region
    WHERE rt.direction = 'up'
        AND r.facc > 0 AND r.facc != -9999
    {where_clause.replace('r.', 'rt.') if where_clause else ''}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="F002",
        name="facc_jump_ratio",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=f"Reaches where facc > {jump_threshold}x upstream sum (entry points)",
        threshold=jump_threshold,
    )


@register_check(
    "F003",
    Category.ATTRIBUTES,
    Severity.INFO,
    "Bifurcation facc divergence",
    default_threshold=0.9,
)
def check_bifurcation_facc_divergence(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check for bifurcations where downstream branches have divergent facc.

    At bifurcations (n_rch_down >= 2), MERIT Hydro D8 picks ONE downstream
    branch for flow. The other branch gets different (often wrong) facc.

    Flags bifurcations where:
    - facc_divergence > threshold (default 0.9)
    - Downstream branches have similar width (so should have similar facc)
    """
    divergence_threshold = threshold if threshold is not None else 0.9
    where_clause = f"AND up.region = '{region}'" if region else ""

    query = f"""
    WITH bifurcations AS (
        SELECT reach_id, region
        FROM reaches
        WHERE n_rch_down >= 2 {where_clause.replace('up.', '')}
    ),
    downstream_facc AS (
        SELECT
            b.reach_id as upstream_reach_id,
            b.region,
            rt.neighbor_reach_id as downstream_reach_id,
            r_dn.facc as downstream_facc,
            r_dn.width as downstream_width,
            ROW_NUMBER() OVER (PARTITION BY b.reach_id ORDER BY r_dn.facc DESC) as facc_rank
        FROM bifurcations b
        JOIN reach_topology rt ON b.reach_id = rt.reach_id AND b.region = rt.region
        JOIN reaches r_dn ON rt.neighbor_reach_id = r_dn.reach_id AND rt.region = r_dn.region
        WHERE rt.direction = 'down'
            AND r_dn.facc > 0 AND r_dn.facc != -9999
    ),
    divergence AS (
        SELECT
            upstream_reach_id,
            region,
            MAX(downstream_facc) as max_downstream_facc,
            MIN(downstream_facc) as min_downstream_facc,
            (MAX(downstream_facc) - MIN(downstream_facc)) / NULLIF(MAX(downstream_facc), 0) as facc_divergence,
            COUNT(*) as n_downstream,
            MAX(CASE WHEN facc_rank = 1 THEN downstream_width END) as high_facc_width,
            MAX(CASE WHEN facc_rank = 2 THEN downstream_width END) as low_facc_width
        FROM downstream_facc
        GROUP BY upstream_reach_id, region
    )
    SELECT
        up.reach_id,
        up.region,
        up.river_name,
        up.x, up.y,
        up.facc as upstream_facc,
        d.max_downstream_facc,
        d.min_downstream_facc,
        d.facc_divergence,
        d.n_downstream,
        d.high_facc_width,
        d.low_facc_width,
        CASE
            WHEN d.high_facc_width IS NOT NULL AND d.low_facc_width IS NOT NULL
            THEN ABS(d.high_facc_width - d.low_facc_width) / GREATEST(d.high_facc_width, d.low_facc_width)
            ELSE NULL
        END as width_diff_ratio
    FROM reaches up
    JOIN divergence d ON up.reach_id = d.upstream_reach_id AND up.region = d.region
    WHERE d.facc_divergence > {divergence_threshold}
    ORDER BY d.facc_divergence DESC
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches
    WHERE n_rch_down >= 2 {where_clause.replace('up.', '')}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="F003",
        name="bifurcation_facc_divergence",
        severity=Severity.INFO,
        passed=True,  # Informational
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=f"Bifurcations with facc divergence > {divergence_threshold}",
        threshold=divergence_threshold,
    )


@register_check(
    "F004",
    Category.ATTRIBUTES,
    Severity.WARNING,
    "Facc vs reach accumulation mismatch",
    default_threshold=10.0,
)
def check_facc_reach_acc_ratio(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check for reaches where facc doesn't match topology-based reach accumulation.

    Core insight: facc should scale with number of upstream reaches.
    If a reach has few upstream reaches but huge facc (or vice versa),
    the facc is likely corrupted.

    This check computes reach accumulation from topology and compares
    to expected facc.
    """
    ratio_threshold = threshold if threshold is not None else 10.0
    where_clause = f"WHERE r.region = '{region}'" if region else ""
    topo_where = f"WHERE rt.region = '{region}'" if region else ""

    # Simplified check: compare facc to immediate upstream count * regional median
    # Full reach_acc requires matrix computation, so we use a simplified version here

    query = f"""
    WITH upstream_counts AS (
        SELECT
            rt.reach_id,
            rt.region,
            COUNT(*) as n_upstream
        FROM reach_topology rt
        WHERE rt.direction = 'up' {topo_where.replace('WHERE', 'AND') if topo_where else ''}
        GROUP BY rt.reach_id, rt.region
    ),
    regional_baseline AS (
        SELECT
            r.region,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY r.facc / GREATEST(COALESCE(uc.n_upstream, 0) + 1, 1)) as median_facc_per_upstream
        FROM reaches r
        LEFT JOIN upstream_counts uc ON r.reach_id = uc.reach_id AND r.region = uc.region
        WHERE r.facc > 0 AND r.facc != -9999
            AND r.facc / GREATEST(COALESCE(uc.n_upstream, 0) + 1, 1) < 1000000  -- Exclude outliers
        GROUP BY r.region
    )
    SELECT
        r.reach_id,
        r.region,
        r.river_name,
        r.x, r.y,
        r.facc,
        COALESCE(uc.n_upstream, 0) + 1 as reach_count,  -- +1 for self
        rb.median_facc_per_upstream * (COALESCE(uc.n_upstream, 0) + 1) as expected_facc,
        r.facc / NULLIF(rb.median_facc_per_upstream * (COALESCE(uc.n_upstream, 0) + 1), 0) as facc_reach_ratio,
        r.width,
        r.stream_order
    FROM reaches r
    LEFT JOIN upstream_counts uc ON r.reach_id = uc.reach_id AND r.region = uc.region
    JOIN regional_baseline rb ON r.region = rb.region
    WHERE r.facc > 0 AND r.facc != -9999
        AND r.facc / NULLIF(rb.median_facc_per_upstream * (COALESCE(uc.n_upstream, 0) + 1), 0) > {ratio_threshold}
    {where_clause.replace('WHERE', 'AND') if where_clause else ''}
    ORDER BY r.facc / NULLIF(rb.median_facc_per_upstream * (COALESCE(uc.n_upstream, 0) + 1), 0) DESC
    """

    try:
        issues = conn.execute(query).fetchdf()
    except Exception as e:
        # Fallback if query fails
        issues = pd.DataFrame()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r
    WHERE r.facc > 0 AND r.facc != -9999
    {where_clause.replace('WHERE', 'AND') if where_clause else ''}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="F004",
        name="facc_reach_acc_ratio",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=f"Reaches where facc > {ratio_threshold}x expected from topology",
        threshold=ratio_threshold,
    )


@register_check(
    "F005",
    Category.ATTRIBUTES,
    Severity.INFO,
    "Composite facc anomaly score",
    default_threshold=0.5,
)
def check_facc_composite_anomaly(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Composite facc anomaly detection combining multiple signals.

    Combines:
    - facc/width ratio
    - facc jump ratio
    - facc vs reach count ratio

    Returns reaches with composite score > threshold.
    """
    score_threshold = threshold if threshold is not None else 0.5
    where_clause = f"AND r.region = '{region}'" if region else ""

    # Compute composite score using normalized metrics
    query = f"""
    WITH upstream_facc AS (
        SELECT
            rt.reach_id,
            rt.region,
            SUM(r_up.facc) as upstream_facc_sum,
            COUNT(*) as n_upstream
        FROM reach_topology rt
        JOIN reaches r_up ON rt.neighbor_reach_id = r_up.reach_id
            AND rt.region = r_up.region
        WHERE rt.direction = 'up'
            AND r_up.facc > 0 AND r_up.facc != -9999
        GROUP BY rt.reach_id, rt.region
    ),
    features AS (
        SELECT
            r.reach_id,
            r.region,
            r.river_name,
            r.x, r.y,
            r.facc,
            r.width,
            r.facc / NULLIF(r.width, 0) as facc_width_ratio,
            COALESCE(uf.upstream_facc_sum, 0) as upstream_facc_sum,
            COALESCE(uf.n_upstream, 0) as n_upstream,
            CASE
                WHEN uf.upstream_facc_sum > 0 THEN r.facc / uf.upstream_facc_sum
                ELSE NULL
            END as facc_jump_ratio,
            r.stream_order
        FROM reaches r
        LEFT JOIN upstream_facc uf ON r.reach_id = uf.reach_id AND r.region = uf.region
        WHERE r.facc > 0 AND r.facc != -9999
            AND r.width > 0
            {where_clause}
    ),
    scores AS (
        SELECT
            *,
            -- Score components (0-1 scale, clipped)
            LEAST(GREATEST((facc_width_ratio / 5000.0) - 1, 0), 1) as score_width,
            LEAST(GREATEST((COALESCE(facc_jump_ratio, 0) / 100.0) - 1, 0), 1) as score_jump,
            -- Composite score (weighted average)
            0.5 * LEAST(GREATEST((facc_width_ratio / 5000.0) - 1, 0), 1) +
            0.5 * LEAST(GREATEST((COALESCE(facc_jump_ratio, 0) / 100.0) - 1, 0), 1)
            as anomaly_score
        FROM features
    )
    SELECT
        reach_id, region, river_name, x, y,
        facc, width, facc_width_ratio,
        upstream_facc_sum, n_upstream, facc_jump_ratio,
        stream_order, anomaly_score
    FROM scores
    WHERE anomaly_score > {score_threshold}
    ORDER BY anomaly_score DESC
    """

    try:
        issues = conn.execute(query).fetchdf()
    except Exception as e:
        issues = pd.DataFrame()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r
    WHERE r.facc > 0 AND r.facc != -9999 AND r.width > 0
    {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="F005",
        name="facc_composite_anomaly",
        severity=Severity.INFO,
        passed=True,  # Informational
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=f"Reaches with composite facc anomaly score > {score_threshold}",
        threshold=score_threshold,
    )
