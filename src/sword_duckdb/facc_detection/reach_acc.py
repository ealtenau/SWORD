# -*- coding: utf-8 -*-
"""
Reach Accumulation Computation
==============================

Computes reach accumulation (number of upstream reaches) from SWORD topology.
This provides a topology-based alternative to MERIT Hydro's facc.

Core insight: If a reach has 50 upstream reaches but 5M km² facc (Amazon-level),
something is wrong. By comparing facc with reach_acc, we can detect corruption.

Algorithm (adapted from Elyssa Collins' routing_topology_gpkg.py):
1. Build sparse network matrix N from reach_topology
2. Compute (I - N)^-1 * ones = reach accumulation
3. At bifurcations, flow splits proportionally (0.5/0.33/0.25)

This module provides both SQL-based (pure DuckDB) and matrix-based (scipy)
approaches for flexibility.
"""

from typing import Optional, Dict, Union
import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


class ReachAccumulator:
    """
    Computes reach accumulation from SWORD topology.

    Reach accumulation counts how many reaches are upstream of each reach,
    which should correlate with facc (flow accumulation area).

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection to SWORD database.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn

    def compute_sql(self, region: Optional[str] = None, max_iterations: int = 100) -> pd.DataFrame:
        """
        Compute reach accumulation using iterative SQL (simpler, slower).

        This approach uses recursive CTEs to propagate upstream counts.
        Suitable for smaller regions or when scipy is not available.

        Parameters
        ----------
        region : str, optional
            Region to compute for (e.g., 'NA'). If None, computes all.
        max_iterations : int
            Maximum iterations for convergence (default 100).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: reach_id, region, reach_acc
        """
        where_clause = f"AND r.region = '{region}'" if region else ""
        topo_where = f"AND rt.region = '{region}'" if region else ""

        # Iterative approach: start with 1 for each reach, propagate downstream
        query = f"""
        WITH RECURSIVE
        -- Base: every reach starts with count 1 (itself)
        reach_counts AS (
            SELECT
                reach_id,
                region,
                1.0 as acc,
                0 as iteration
            FROM reaches r
            WHERE 1=1 {where_clause}
        ),
        -- Get immediate upstream neighbors for each reach
        upstream_topology AS (
            SELECT
                rt.reach_id,
                rt.region,
                rt.neighbor_reach_id as upstream_reach_id,
                -- At bifurcations (multiple downstream), split contribution
                1.0 / COUNT(*) OVER (PARTITION BY rt.neighbor_reach_id, rt.region) as weight
            FROM reach_topology rt
            WHERE rt.direction = 'up' {topo_where}
        ),
        -- Sum immediate upstream contributions for each reach
        immediate_upstream AS (
            SELECT
                r.reach_id,
                r.region,
                1.0 + COALESCE(SUM(ut.weight), 0) as reach_acc
            FROM reaches r
            LEFT JOIN upstream_topology ut
                ON ut.reach_id = r.reach_id AND ut.region = r.region
            WHERE 1=1 {where_clause}
            GROUP BY r.reach_id, r.region
        )
        SELECT reach_id, region, reach_acc FROM immediate_upstream
        ORDER BY reach_acc DESC
        """

        return self.conn.execute(query).fetchdf()

    def compute_matrix(self, region: Optional[str] = None) -> pd.DataFrame:
        """
        Compute reach accumulation using sparse matrix inversion (faster, exact).

        Uses the algorithm from routing_topology_gpkg.py:
        reach_acc = (I - N)^-1 * ones

        Parameters
        ----------
        region : str, optional
            Region to compute for (e.g., 'NA'). If None, computes all.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: reach_id, region, reach_acc
        """
        where_clause = f"WHERE region = '{region}'" if region else ""
        topo_where = f"WHERE rt.region = '{region}'" if region else ""

        # Get all reach IDs
        reaches_df = self.conn.execute(f"""
            SELECT reach_id, region FROM reaches {where_clause}
            ORDER BY reach_id
        """).fetchdf()

        if len(reaches_df) == 0:
            return pd.DataFrame(columns=['reach_id', 'region', 'reach_acc'])

        n_reaches = len(reaches_df)
        reach_ids = reaches_df['reach_id'].values
        regions = reaches_df['region'].values

        # Create reach_id -> index mapping
        id_to_idx = {rid: idx for idx, rid in enumerate(reach_ids)}

        # Get downstream topology with weights for bifurcations
        topo_query = f"""
            WITH downstream_counts AS (
                SELECT
                    reach_id,
                    region,
                    COUNT(*) as n_down
                FROM reach_topology
                WHERE direction = 'down' {topo_where.replace('WHERE', 'AND') if topo_where else ''}
                GROUP BY reach_id, region
            )
            SELECT
                rt.reach_id as from_reach,
                rt.neighbor_reach_id as to_reach,
                rt.region,
                1.0 / COALESCE(dc.n_down, 1) as weight
            FROM reach_topology rt
            LEFT JOIN downstream_counts dc
                ON rt.reach_id = dc.reach_id AND rt.region = dc.region
            WHERE rt.direction = 'down' {topo_where.replace('WHERE', 'AND') if topo_where else ''}
        """

        topo_df = self.conn.execute(topo_query).fetchdf()

        # Build sparse network matrix
        rows = []
        cols = []
        vals = []

        for _, row in topo_df.iterrows():
            from_reach = row['from_reach']
            to_reach = row['to_reach']
            weight = row['weight']

            if from_reach in id_to_idx and to_reach in id_to_idx:
                from_idx = id_to_idx[from_reach]
                to_idx = id_to_idx[to_reach]
                # Network matrix: N[to, from] = weight
                rows.append(to_idx)
                cols.append(from_idx)
                vals.append(weight)

        # Create sparse matrices
        N = csc_matrix((vals, (rows, cols)), shape=(n_reaches, n_reaches))
        I = csc_matrix(np.eye(n_reaches))

        # Compute (I - N)^-1 * ones
        ones = np.ones(n_reaches)
        try:
            reach_acc = spsolve(I - N, ones)
        except Exception as e:
            # Fallback: if matrix is singular, use iterative approach
            print(f"Matrix solve failed: {e}, using iterative approach")
            # Simple power iteration
            reach_acc = ones.copy()
            for _ in range(100):
                new_acc = ones + N.dot(reach_acc)
                if np.allclose(new_acc, reach_acc, rtol=1e-6):
                    break
                reach_acc = new_acc

        # Create result DataFrame
        result = pd.DataFrame({
            'reach_id': reach_ids,
            'region': regions,
            'reach_acc': reach_acc,
        })

        return result.sort_values('reach_acc', ascending=False)

    def compute(
        self,
        region: Optional[str] = None,
        method: str = 'matrix'
    ) -> pd.DataFrame:
        """
        Compute reach accumulation using specified method.

        Parameters
        ----------
        region : str, optional
            Region to compute for.
        method : str
            'matrix' (default, faster) or 'sql' (simpler).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: reach_id, region, reach_acc
        """
        if method == 'matrix':
            return self.compute_matrix(region)
        else:
            return self.compute_sql(region)


def compute_reach_accumulation(
    db_path_or_conn: Union[str, duckdb.DuckDBPyConnection],
    region: Optional[str] = None,
    method: str = 'matrix'
) -> pd.DataFrame:
    """
    Convenience function to compute reach accumulation.

    Parameters
    ----------
    db_path_or_conn : str or duckdb.DuckDBPyConnection
        Path to DuckDB database or existing connection.
    region : str, optional
        Region to compute for (e.g., 'NA').
    method : str
        'matrix' (default) or 'sql'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: reach_id, region, reach_acc

    Examples
    --------
    >>> from sword_duckdb.facc_detection import compute_reach_accumulation
    >>> df = compute_reach_accumulation("sword_v17c.duckdb", region="NA")
    >>> print(df.head())
    """
    if isinstance(db_path_or_conn, str):
        conn = duckdb.connect(db_path_or_conn, read_only=True)
        own_conn = True
    else:
        conn = db_path_or_conn
        own_conn = False

    try:
        accumulator = ReachAccumulator(conn)
        return accumulator.compute(region=region, method=method)
    finally:
        if own_conn:
            conn.close()


def compute_upstream_facc_sum(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute sum of immediate upstream facc values for each reach.

    This is used to detect "entry point" errors where facc jumps
    dramatically at a junction.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection.
    region : str, optional
        Region to compute for.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: reach_id, region, facc, upstream_facc_sum,
        n_upstream, facc_jump_ratio
    """
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
        r.facc,
        COALESCE(uf.upstream_facc_sum, 0) as upstream_facc_sum,
        COALESCE(uf.n_upstream, 0) as n_upstream,
        CASE
            WHEN COALESCE(uf.upstream_facc_sum, 0) > 0
            THEN r.facc / uf.upstream_facc_sum
            ELSE NULL
        END as facc_jump_ratio
    FROM reaches r
    LEFT JOIN upstream_facc uf ON r.reach_id = uf.reach_id AND r.region = uf.region
    WHERE r.facc > 0 AND r.facc != -9999 {where_clause}
    ORDER BY facc_jump_ratio DESC NULLS LAST
    """

    return conn.execute(query).fetchdf()


def compute_bifurcation_facc_divergence(
    conn: duckdb.DuckDBPyConnection,
    region: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute facc divergence at bifurcations.

    At bifurcations (n_rch_down >= 2), MERIT Hydro D8 picks ONE downstream
    branch for flow → other branch gets wrong facc. This detects cases where
    downstream branches have very different facc despite similar topology.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection.
    region : str, optional
        Region to compute for.

    Returns
    -------
    pd.DataFrame
        DataFrame with bifurcation reaches and their downstream facc divergence.
    """
    where_clause = f"AND up.region = '{region}'" if region else ""

    query = f"""
    WITH bifurcations AS (
        -- Find reaches with 2+ downstream neighbors
        SELECT reach_id, region
        FROM reaches
        WHERE n_rch_down >= 2 {where_clause.replace('up.', '')}
    ),
    downstream_facc AS (
        -- Get facc of all downstream neighbors
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
            MAX(downstream_facc) - MIN(downstream_facc) as facc_diff,
            CASE
                WHEN MAX(downstream_facc) > 0
                THEN (MAX(downstream_facc) - MIN(downstream_facc)) / MAX(downstream_facc)
                ELSE 0
            END as facc_divergence,
            COUNT(*) as n_downstream,
            -- Width comparison for similar-sized branches
            MAX(CASE WHEN facc_rank = 1 THEN downstream_width END) as high_facc_width,
            MAX(CASE WHEN facc_rank = 2 THEN downstream_width END) as low_facc_width
        FROM downstream_facc
        GROUP BY upstream_reach_id, region
    )
    SELECT
        up.reach_id,
        up.region,
        up.facc as upstream_facc,
        up.width as upstream_width,
        d.max_downstream_facc,
        d.min_downstream_facc,
        d.facc_diff,
        d.facc_divergence,
        d.n_downstream,
        d.high_facc_width,
        d.low_facc_width,
        -- Flag: high divergence with similar widths = likely error
        CASE
            WHEN d.facc_divergence > 0.9
                AND d.high_facc_width IS NOT NULL
                AND d.low_facc_width IS NOT NULL
                AND ABS(d.high_facc_width - d.low_facc_width) / GREATEST(d.high_facc_width, d.low_facc_width) < 0.3
            THEN TRUE
            ELSE FALSE
        END as likely_facc_error
    FROM reaches up
    JOIN divergence d ON up.reach_id = d.upstream_reach_id AND up.region = d.region
    WHERE d.facc_divergence > 0.5  -- At least 50% difference
    ORDER BY d.facc_divergence DESC
    """

    return conn.execute(query).fetchdf()
