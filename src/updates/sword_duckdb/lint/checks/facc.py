# -*- coding: utf-8 -*-
"""
SWORD Lint - Facc Checks (F0xx)

Validates flow accumulation (facc) values post-conservation-correction.

Checks:
  F001 - facc/width ratio anomaly (MERIT entry point corruption)
  F002 - facc jump ratio (entry point detection)
  F004 - facc vs reach accumulation mismatch — INFO (structural D8/vector noise)
  F006 - junction conservation (facc < sum upstream at 2+ input junctions) — ERROR
  F011 - 1:1 link monotonicity (facc drop on single-upstream links) — INFO
  F007 - bifurcation balance (children sum / parent ratio) — INFO (expected post-correction)
  F008 - bifurcation surplus (child facc > parent facc) — WARNING
  F009 - facc_quality tag coverage — INFO
  F010 - junction-raise drop (raised junction → unchanged downstream) — INFO
  F012 - incremental area non-negativity (junctions only) — ERROR
  F013 - incremental area Tukey IQR outlier — INFO (statistical, not a violation)
Removed:
  F015 - junction surplus — removed because surplus = lateral drainage from
         unmapped tributaries, not a violation.
  F003 - bifurcation divergence — removed because post-correction children
         SHOULD diverge (width-proportional split). Was only useful pre-correction.
  F005 - composite anomaly score — removed, was just a weighted average of
         F001 + F002 with no additional signal.
"""

from typing import Optional

import duckdb
import numpy as np
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
    default_threshold=5000.0,
)
def check_facc_width_ratio_anomaly(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Check for reaches with suspiciously high facc/width ratio.

    Reaches with facc/width > 5000 are likely corrupted entry points
    where bad facc enters the network from MERIT Hydro.
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
    "F004",
    Category.ATTRIBUTES,
    Severity.INFO,
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

    facc should scale with number of upstream reaches. If a reach has few
    upstream reaches but huge facc (or vice versa), facc is likely corrupted.
    """
    ratio_threshold = threshold if threshold is not None else 10.0
    where_clause = f"WHERE r.region = '{region}'" if region else ""
    topo_where = f"WHERE rt.region = '{region}'" if region else ""

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
            AND r.facc / GREATEST(COALESCE(uc.n_upstream, 0) + 1, 1) < 1000000
        GROUP BY r.region
    )
    SELECT
        r.reach_id,
        r.region,
        r.river_name,
        r.x, r.y,
        r.facc,
        COALESCE(uc.n_upstream, 0) + 1 as reach_count,
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
    except Exception:
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
        severity=Severity.INFO,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=f"Reaches where facc > {ratio_threshold}x expected from topology",
        threshold=ratio_threshold,
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
    "F007",
    Category.ATTRIBUTES,
    Severity.INFO,
    "Bifurcation balance (children sum vs parent)",
    default_threshold=0.10,
)
def check_bifurcation_balance(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    At bifurcations, children's facc should sum to approximately parent's facc.

    After width-proportional splitting, sum(child_facc) / parent_facc should
    be close to 1.0. Deviations indicate incomplete correction.

    Threshold (default 0.10) controls how far from 1.0 is acceptable:
    flags bifurcations where |ratio - 1.0| > threshold AND ratio > 1 + threshold
    (only flags over-splits since under-splits are handled by local drainage).
    """
    tol = threshold if threshold is not None else 0.10
    where_clause = f"AND r_parent.region = '{region}'" if region else ""

    query = f"""
    WITH bifurc_children AS (
        SELECT
            rt.reach_id as parent_id,
            r_parent.region,
            r_parent.facc as parent_facc,
            r_parent.river_name,
            r_parent.x, r_parent.y,
            rt.neighbor_reach_id as child_id,
            r_child.facc as child_facc,
            r_child.width as child_width
        FROM reach_topology rt
        JOIN reaches r_parent ON rt.reach_id = r_parent.reach_id
            AND rt.region = r_parent.region
        JOIN reaches r_child ON rt.neighbor_reach_id = r_child.reach_id
            AND rt.region = r_child.region
        WHERE rt.direction = 'down'
            AND r_parent.n_rch_down >= 2
            AND r_parent.facc > 0 AND r_parent.facc != -9999
            AND r_child.facc > 0 AND r_child.facc != -9999
            {where_clause}
    ),
    bifurc_summary AS (
        SELECT
            parent_id,
            region,
            parent_facc,
            river_name,
            x, y,
            SUM(child_facc) as child_sum_facc,
            COUNT(*) as n_children,
            SUM(child_facc) / parent_facc as balance_ratio
        FROM bifurc_children
        GROUP BY parent_id, region, parent_facc, river_name, x, y
    )
    SELECT
        parent_id as reach_id,
        region,
        river_name,
        x, y,
        parent_facc,
        child_sum_facc,
        n_children,
        ROUND(balance_ratio, 4) as balance_ratio
    FROM bifurc_summary
    WHERE balance_ratio > 1.0 + {tol}
    ORDER BY balance_ratio DESC
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*) FROM reaches r_parent
    WHERE r_parent.n_rch_down >= 2
        AND r_parent.facc > 0 AND r_parent.facc != -9999
        {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="F007",
        name="bifurcation_balance",
        severity=Severity.INFO,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=(
            f"Bifurcations where sum(child facc) / parent facc > {1.0 + tol:.2f} "
            f"(D8 cloning not fully corrected)"
        ),
        threshold=tol,
    )


@register_check(
    "F008",
    Category.ATTRIBUTES,
    Severity.WARNING,
    "Bifurcation child facc exceeds parent",
    default_threshold=1.05,
)
def check_bifurcation_child_surplus(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    No bifurcation child should have facc > parent (D8 cloning artifact).

    After conservation correction, every child at a bifurcation should have
    facc <= parent_facc. A child with facc > parent means D8 cloned the
    full parent value and the correction missed it.

    Threshold (default 1.05) allows 5% tolerance for local drainage.
    """
    ratio_threshold = threshold if threshold is not None else 1.05
    where_clause = f"AND r_parent.region = '{region}'" if region else ""

    query = f"""
    SELECT
        r_child.reach_id,
        r_child.region,
        r_child.river_name,
        r_child.x, r_child.y,
        r_child.facc as child_facc,
        r_parent.reach_id as parent_reach_id,
        r_parent.facc as parent_facc,
        r_child.facc / r_parent.facc as child_parent_ratio,
        r_child.width as child_width,
        r_parent.n_rch_down
    FROM reach_topology rt
    JOIN reaches r_parent ON rt.reach_id = r_parent.reach_id
        AND rt.region = r_parent.region
    JOIN reaches r_child ON rt.neighbor_reach_id = r_child.reach_id
        AND rt.region = r_child.region
    WHERE rt.direction = 'down'
        AND r_parent.n_rch_down >= 2
        AND r_parent.facc > 0 AND r_parent.facc != -9999
        AND r_child.facc > 0 AND r_child.facc != -9999
        AND r_child.facc / r_parent.facc > {ratio_threshold}
        {where_clause}
    ORDER BY r_child.facc / r_parent.facc DESC
    """

    issues = conn.execute(query).fetchdf()

    total_query = f"""
    SELECT COUNT(*)
    FROM reach_topology rt
    JOIN reaches r_parent ON rt.reach_id = r_parent.reach_id
        AND rt.region = r_parent.region
    WHERE rt.direction = 'down'
        AND r_parent.n_rch_down >= 2
        AND r_parent.facc > 0 AND r_parent.facc != -9999
        {where_clause}
    """
    total = conn.execute(total_query).fetchone()[0]

    return CheckResult(
        check_id="F008",
        name="bifurcation_child_surplus",
        severity=Severity.WARNING,
        passed=len(issues) == 0,
        total_checked=total,
        issues_found=len(issues),
        issue_pct=100 * len(issues) / total if total > 0 else 0,
        details=issues,
        description=(
            f"Bifurcation children where child_facc / parent_facc > {ratio_threshold} "
            f"(uncorrected D8 clone)"
        ),
        threshold=ratio_threshold,
    )


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

    tagged = total - int(details[details["facc_quality"] == "untagged"]["reach_count"].sum()) if "untagged" in details["facc_quality"].values else total

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


@register_check(
    "F013",
    Category.ATTRIBUTES,
    Severity.INFO,
    "Incremental area Tukey IQR outlier (correction magnitude)",
    default_threshold=1.5,
)
def check_incremental_area_outlier(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None,
    threshold: Optional[float] = None,
) -> CheckResult:
    """
    Flag reaches whose incremental area is a Tukey IQR outlier.

    Mirrors the iterative Tukey IQR downweighting in the CVXPY integrator's
    ``fix_drainage_area()`` — outliers in ``rel_change`` distribution get
    epsilon weight.

    Computes incremental_area = facc - sum(upstream facc) for all non-headwater
    reaches (those with at least 1 upstream neighbor). Then applies log transform
    and Tukey fences: Q1 - k*IQR .. Q3 + k*IQR where k = threshold (default 1.5).

    Only checks reaches with positive incremental area (headwaters with 0
    upstream are excluded since they have no meaningful incremental value).
    """
    k = threshold if threshold is not None else 1.5
    where_clause = f"AND r.region = '{region}'" if region else ""

    # Step 1: compute incremental areas in SQL
    incr_query = f"""
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
           uf.n_upstream, r.width, r.stream_order
    FROM reaches r
    JOIN upstream_facc uf ON r.reach_id = uf.reach_id AND r.region = uf.region
    WHERE r.facc > 0 AND r.facc != -9999
      AND (r.facc - uf.upstream_facc_sum) > 0
      {where_clause}
    """

    df = conn.execute(incr_query).fetchdf()

    if len(df) == 0:
        return CheckResult(
            check_id="F013",
            name="incremental_area_outlier",
            severity=Severity.INFO,
            passed=True,
            total_checked=0,
            issues_found=0,
            issue_pct=0,
            details=pd.DataFrame(),
            description="No reaches with positive incremental area to check",
            threshold=k,
        )

    # Step 2: Tukey IQR on log(incremental_area)
    log_incr = np.log(df["incremental_area"].values)
    q1 = np.percentile(log_incr, 25)
    q3 = np.percentile(log_incr, 75)
    iqr = q3 - q1
    low_fence = q1 - k * iqr
    high_fence = q3 + k * iqr

    outlier_mask = (log_incr < low_fence) | (log_incr > high_fence)
    issues = df[outlier_mask].copy()
    issues["log_incremental_area"] = log_incr[outlier_mask]
    issues["low_fence"] = low_fence
    issues["high_fence"] = high_fence

    return CheckResult(
        check_id="F013",
        name="incremental_area_outlier",
        severity=Severity.INFO,
        passed=len(issues) == 0,
        total_checked=len(df),
        issues_found=len(issues),
        issue_pct=100 * len(issues) / len(df) if len(df) > 0 else 0,
        details=issues,
        description=(
            f"Reaches with log(incremental_area) outside Tukey fences "
            f"[{low_fence:.2f}, {high_fence:.2f}] (k={k})"
        ),
        threshold=k,
    )


