# -*- coding: utf-8 -*-
"""
SWORD Lint - Facc Checks (F0xx)

Validates flow accumulation (facc) values post-conservation-correction.

Checks:
  F001 - facc/width ratio anomaly — WARNING (threshold 50k post-propagation)
  F002 - facc jump ratio (entry point detection)
  F006 - junction conservation (facc < sum upstream at 2+ input junctions) — ERROR
  F009 - facc_quality tag coverage — INFO
  F010 - junction-raise drop (raised junction → unchanged downstream) — INFO
  F011 - 1:1 link monotonicity (facc drop on single-upstream links) — INFO
  F012 - incremental area non-negativity (junctions only) — ERROR
Removed:
  F013 - incremental area Tukey IQR outlier — removed because Tukey IQR on
         fat-tailed drainage-area distribution flags 6% as "outliers" — just
         the natural tail, not meaningful signal.
  F004 - facc vs reach accumulation mismatch — removed because regional-median
         heuristic is meaningless after topology-driven propagation rewrites
         facc values network-wide.
  F007 - bifurcation balance — removed because all flagged cases are junction
         children that correctly receive additional tributary flow. Pure
         bifurcations are perfectly balanced (0/2816 violated).
  F008 - bifurcation child surplus — removed because all cases are junction
         children (170/171) or FP rounding (1/171). Expected behavior.
  F014 - facc vs node-level consistency — removed because topology-driven
         propagation intentionally overrides raw D8 node samples. Comparing
         reach facc against node MAX is circular after the denoise pipeline.
  F015 - junction surplus — removed because surplus = lateral drainage from
         unmapped tributaries, not a violation.
  F003 - bifurcation divergence — removed because post-correction children
         SHOULD diverge (width-proportional split). Was only useful pre-correction.
  F005 - composite anomaly score — removed, was just a weighted average of
         F001 + F002 with no additional signal.
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
    "F001",
    Category.ATTRIBUTES,
    Severity.WARNING,
    "Facc/width ratio anomaly detection",
    default_threshold=50000.0,
)
def check_facc_width_ratio_anomaly(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check for reaches with suspiciously high facc/width ratio.

    After topology-driven propagation, large downstream rivers normally have
    high facc/width (~p99=21k, p99.5=39k). Threshold set to 50,000 to flag
    only genuine outliers (0.4% of reaches) rather than normal big rivers.
    """
    ratio_threshold = threshold if threshold is not None else 50000.0
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

    Detects entry points where corrupted facc enters the network.
    If facc >> sum(upstream facc), MERIT D8 likely picked up a bad value.
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
    {where_clause.replace("r.", "rt.") if where_clause else ""}
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
    "F006",
    Category.ATTRIBUTES,
    Severity.ERROR,
    "Junction conservation (facc < sum upstream at junctions with 2+ inputs)",
    default_threshold=1.0,
)
def check_facc_conservation(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check conservation at JUNCTIONS (2+ upstream reaches).

    This is the core physics check: at a confluence, total drainage area
    must be >= sum of incoming branches. Deficits indicate D8 cloning or
    topology errors. Threshold defaults to 1 km² to ignore FP rounding.
    """
    min_deficit = threshold if threshold is not None else 1.0
    where_clause = f"AND r.region = '{region}'" if region else ""

    query = f"""
    WITH upstream_facc AS (
        SELECT rt.reach_id, rt.region,
               SUM(r_up.facc) as upstream_facc_sum,
               COUNT(*) as n_upstream
        FROM reach_topology rt
        JOIN reaches r_up ON rt.neighbor_reach_id = r_up.reach_id
            AND rt.region = r_up.region
        WHERE rt.direction = 'up'
          AND r_up.facc > 0 AND r_up.facc != -9999
        GROUP BY rt.reach_id, rt.region
    )
    SELECT r.reach_id, r.region, r.river_name, r.x, r.y,
           r.facc, uf.upstream_facc_sum,
           uf.upstream_facc_sum - r.facc as conservation_deficit,
           100.0 * (uf.upstream_facc_sum - r.facc) / uf.upstream_facc_sum as deficit_pct,
           uf.n_upstream, r.width, r.stream_order
    FROM reaches r
    JOIN upstream_facc uf ON r.reach_id = uf.reach_id AND r.region = uf.region
    WHERE r.facc > 0 AND r.facc != -9999
      AND uf.n_upstream >= 2
      AND r.facc < uf.upstream_facc_sum
      AND (uf.upstream_facc_sum - r.facc) > {min_deficit}
      {where_clause}
    ORDER BY (uf.upstream_facc_sum - r.facc) DESC
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM (
        SELECT rt.reach_id FROM reach_topology rt
        JOIN reaches r ON rt.reach_id = r.reach_id AND rt.region = r.region
        WHERE rt.direction = 'up'
            AND r.facc > 0 AND r.facc != -9999
            {where_clause}
        GROUP BY rt.reach_id, rt.region
        HAVING COUNT(*) >= 2
    )
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="F006",
        name="facc_junction_conservation",
        severity=Severity.ERROR,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=f"Junctions (2+ upstream) where facc < sum(upstream facc) - {min_deficit} km²",
        threshold=min_deficit,
    )


@register_check(
    "F011",
    Category.ATTRIBUTES,
    Severity.INFO,
    "1:1 link facc drop (downstream < single upstream)",
    default_threshold=0.0,
)
def check_facc_link_monotonicity(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check monotonicity on 1:1 links (single upstream → single downstream).

    Drops here are expected post-conservation (junction raises create
    boundaries with unchanged downstream). These are D8 artifacts that
    cannot be fixed without re-running MERIT Hydro flow accumulation.
    """
    min_deficit = threshold if threshold is not None else 0.0
    where_clause = f"AND r.region = '{region}'" if region else ""

    # Exclude bifurcation children: a reach whose single upstream neighbor
    # has out_degree >= 2 is a bifurcation child — facc drop is expected
    # (width-proportional split).
    query = f"""
    WITH upstream_info AS (
        SELECT rt.reach_id, rt.region,
               rt.neighbor_reach_id as up_id,
               r_up.facc as up_facc
        FROM reach_topology rt
        JOIN reaches r_up ON rt.neighbor_reach_id = r_up.reach_id
            AND rt.region = r_up.region
        WHERE rt.direction = 'up'
          AND r_up.facc > 0 AND r_up.facc != -9999
    ),
    single_upstream AS (
        SELECT reach_id, region, MIN(up_id) as up_id, MIN(up_facc) as up_facc
        FROM upstream_info
        GROUP BY reach_id, region
        HAVING COUNT(*) = 1
    ),
    parent_outdegree AS (
        SELECT reach_id, region, COUNT(*) as n_down
        FROM reach_topology
        WHERE direction = 'down'
        GROUP BY reach_id, region
    )
    SELECT r.reach_id, r.region, r.river_name, r.x, r.y,
           r.facc, su.up_facc as upstream_facc,
           su.up_facc - r.facc as deficit,
           r.width, r.stream_order
    FROM reaches r
    JOIN single_upstream su ON r.reach_id = su.reach_id AND r.region = su.region
    LEFT JOIN parent_outdegree po ON su.up_id = po.reach_id AND su.region = po.region
    WHERE r.facc > 0 AND r.facc != -9999
      AND r.facc < su.up_facc
      AND (su.up_facc - r.facc) > {min_deficit}
      AND COALESCE(po.n_down, 1) = 1
      {where_clause}
    ORDER BY (su.up_facc - r.facc) DESC
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    WITH single_up AS (
        SELECT rt.reach_id, rt.region, MIN(rt.neighbor_reach_id) as up_id
        FROM reach_topology rt
        JOIN reaches r ON rt.reach_id = r.reach_id AND rt.region = r.region
        WHERE rt.direction = 'up' AND r.facc > 0 AND r.facc != -9999
            {where_clause}
        GROUP BY rt.reach_id, rt.region
        HAVING COUNT(*) = 1
    ),
    par_out AS (
        SELECT reach_id, region, COUNT(*) as n_down
        FROM reach_topology
        WHERE direction = 'down'
        GROUP BY reach_id, region
    )
    SELECT COUNT(*)
    FROM single_up su
    LEFT JOIN par_out po ON su.up_id = po.reach_id AND su.region = po.region
    WHERE COALESCE(po.n_down, 1) = 1
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="F011",
        name="facc_link_monotonicity",
        severity=Severity.INFO,
        passed=True,  # INFO — always passes
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=f"1:1 links where facc < upstream facc (D8 artifact, {len(issues)} drops)",
        threshold=min_deficit,
    )


# ---------------------------------------------------------------------------
# New checks for post-conservation-correction validation
# ---------------------------------------------------------------------------


@register_check(
    "F009",
    Category.ATTRIBUTES,
    Severity.INFO,
    "Facc quality tag coverage",
)
def check_facc_quality_coverage(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Report distribution of facc_quality tags.

    After conservation correction, modified reaches should be tagged with
    facc_quality values like 'conservation_single_pass' or 'topology_derived'.
    This check reports the tag distribution for auditing.
    """
    where_clause = f"WHERE r.region = '{region}'" if region else ""

    # Check if facc_quality column exists
    cols = {
        row[0].lower()
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'reaches'"
        ).fetchall()
    }

    if "facc_quality" not in cols:
        return CheckResult(
            check_id="F009",
            name="facc_quality_coverage",
            severity=Severity.INFO,
            passed=True,
            total_checked=0,
            issues_found=0,
            issue_pct=0,
            details=pd.DataFrame(),
            description="facc_quality column not present (no corrections applied)",
        )

    query = f"""
    SELECT
        COALESCE(r.facc_quality, 'untagged') as facc_quality,
        COUNT(*) as reach_count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as pct
    FROM reaches r
    {where_clause}
    GROUP BY COALESCE(r.facc_quality, 'untagged')
    ORDER BY reach_count DESC
    """

    details = conn.execute(query).fetchdf()

    total_query = f"SELECT COUNT(*) FROM reaches r {where_clause}"
    total = conn.execute(total_query).fetchone()[0]

    tagged = (
        total - int(details[details["facc_quality"] == "untagged"]["reach_count"].sum())
        if "untagged" in details["facc_quality"].values
        else total
    )

    return CheckResult(
        check_id="F009",
        name="facc_quality_coverage",
        severity=Severity.INFO,
        passed=True,  # Informational
        total_checked=total,
        issues_found=tagged,
        issue_pct=100 * tagged / total if total > 0 else 0,
        details=details,
        description=f"Facc quality tag distribution ({tagged}/{total} tagged)",
    )


@register_check(
    "F010",
    Category.ATTRIBUTES,
    Severity.INFO,
    "Junction-raise facc drop on downstream 1:1 link",
)
def check_junction_raise_drop(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Detect facc drops immediately downstream of multi-input junctions.

    After conservation correction, junctions may be raised to
    sum(upstream facc). If the downstream 1:1 neighbor was unchanged,
    a drop appears: junction_facc > downstream_facc.

    These drops are an expected consequence of conservation correction
    (the alternative — raising downstream — causes inflation cascades).
    Severity is INFO because these are known and accepted.
    """
    # 5% tolerance, same as T003
    drop_threshold = threshold if threshold is not None else 0.95
    where_clause = f"AND r_junc.region = '{region}'" if region else ""

    query = f"""
    SELECT
        r_dn.reach_id,
        r_dn.region,
        r_dn.river_name,
        r_dn.x, r_dn.y,
        r_junc.reach_id as junction_reach_id,
        r_junc.facc as junction_facc,
        r_junc.n_rch_up as junction_n_up,
        r_dn.facc as downstream_facc,
        r_junc.facc - r_dn.facc as drop_km2,
        ROUND(100.0 * (r_junc.facc - r_dn.facc) / r_junc.facc, 1) as drop_pct
    FROM reach_topology rt
    JOIN reaches r_junc ON rt.reach_id = r_junc.reach_id
        AND rt.region = r_junc.region
    JOIN reaches r_dn ON rt.neighbor_reach_id = r_dn.reach_id
        AND rt.region = r_dn.region
    WHERE rt.direction = 'down'
        AND r_junc.n_rch_up >= 2
        AND r_junc.n_rch_down = 1
        AND r_dn.n_rch_up = 1
        AND r_junc.facc > 0 AND r_junc.facc != -9999
        AND r_dn.facc > 0 AND r_dn.facc != -9999
        AND r_dn.facc < r_junc.facc * {drop_threshold}
        {where_clause}
    ORDER BY (r_junc.facc - r_dn.facc) DESC
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*)
    FROM reach_topology rt
    JOIN reaches r_junc ON rt.reach_id = r_junc.reach_id
        AND rt.region = r_junc.region
    JOIN reaches r_dn ON rt.neighbor_reach_id = r_dn.reach_id
        AND rt.region = r_dn.region
    WHERE rt.direction = 'down'
        AND r_junc.n_rch_up >= 2
        AND r_junc.n_rch_down = 1
        AND r_dn.n_rch_up = 1
        AND r_junc.facc > 0 AND r_junc.facc != -9999
        AND r_dn.facc > 0 AND r_dn.facc != -9999
        {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="F010",
        name="junction_raise_drop",
        severity=Severity.INFO,
        passed=True,  # Informational — expected tradeoff
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=(
            "Junction → 1:1 downstream facc drops (expected post-conservation; "
            "raising downstream would cause inflation cascade)"
        ),
    )


# ---------------------------------------------------------------------------
# Integrator-derived checks (F012, F013)
#
# Back-inferred from DrainageAreaFix/ CVXPY integrator constraints:
#   x >= 0  →  F012 (incremental area non-negativity)
#   Tukey IQR outlier downweighting  →  F013 (correction magnitude outlier)
# ---------------------------------------------------------------------------


@register_check(
    "F012",
    Category.ATTRIBUTES,
    Severity.ERROR,
    "Incremental area non-negativity (facc >= sum upstream facc at junctions)",
    default_threshold=1.0,
)
def check_incremental_area_nonneg(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check that incremental drainage area is non-negative at junctions.

    For every reach with 2+ upstream neighbors, compute:
        incr_area = facc - sum(upstream facc)
    Flag if incr_area < -threshold.

    Scope: junctions only (n_upstream >= 2). Excluded by design:
      - Bifurcation children: child facc < parent is correct (width-split).
      - 1:1 links: downstream < upstream is D8 raster noise (flagged by T003).

    Threshold default 1.0 km² (same as F006) absorbs floating-point noise
    from junction floor enforcement.
    """
    epsilon = threshold if threshold is not None else 1.0
    where_clause = f"AND r.region = '{region}'" if region else ""

    query = f"""
    WITH upstream_facc AS (
        SELECT rt.reach_id, rt.region,
               SUM(r_up.facc) as upstream_facc_sum,
               COUNT(*) as n_upstream
        FROM reach_topology rt
        JOIN reaches r_up ON rt.neighbor_reach_id = r_up.reach_id
            AND rt.region = r_up.region
        WHERE rt.direction = 'up'
          AND r_up.facc > 0 AND r_up.facc != -9999
        GROUP BY rt.reach_id, rt.region
    )
    SELECT r.reach_id, r.region, r.river_name, r.x, r.y,
           r.facc, uf.upstream_facc_sum,
           r.facc - uf.upstream_facc_sum as incremental_area,
           uf.n_upstream, r.width, r.stream_order,
           r.n_rch_down
    FROM reaches r
    JOIN upstream_facc uf ON r.reach_id = uf.reach_id AND r.region = uf.region
    WHERE r.facc > 0 AND r.facc != -9999
      AND uf.n_upstream >= 2
      AND (r.facc - uf.upstream_facc_sum) < -{epsilon}
      {where_clause}
    ORDER BY (r.facc - uf.upstream_facc_sum) ASC
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(DISTINCT uf.reach_id)
    FROM (
        SELECT rt.reach_id, rt.region, COUNT(*) as n_upstream
        FROM reach_topology rt
        JOIN reaches r_up ON rt.neighbor_reach_id = r_up.reach_id
            AND rt.region = r_up.region
        WHERE rt.direction = 'up'
          AND r_up.facc > 0 AND r_up.facc != -9999
        GROUP BY rt.reach_id, rt.region
        HAVING COUNT(*) >= 2
    ) uf
    JOIN reaches r ON uf.reach_id = r.reach_id AND uf.region = r.region
    WHERE r.facc > 0 AND r.facc != -9999
        {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="F012",
        name="incremental_area_nonneg",
        severity=Severity.ERROR,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=(
            f"Junctions where facc < sum(upstream facc) - {epsilon} km² "
            f"(negative incremental area). Bifurc children and 1:1 links "
            f"excluded (covered by T003)."
        ),
        threshold=epsilon,
    )
