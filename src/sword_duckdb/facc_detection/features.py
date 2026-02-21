# -*- coding: utf-8 -*-
"""
Facc Feature Extraction
=======================

Extracts features for ML-based facc anomaly detection.

Feature categories:
1. **Ratio features**: facc/width, facc/reach_acc, facc_jump_ratio
2. **Topology features**: n_rch_up, n_rch_down, stream_order
3. **Context features**: lakeflag, type, end_reach, main_side
4. **Neighbor features**: upstream_facc_sum, downstream_facc_diff
5. **SWOT features**: n_obs, width_obs_mean (if available)

The key insight is that facc should scale with reach_acc:
    expected_facc ≈ reach_acc × regional_avg_facc_per_reach

If actual_facc >> expected_facc, the reach likely has corrupted facc.
"""

from typing import Optional, List, Dict, Union
import duckdb
import pandas as pd
import numpy as np

from .reach_acc import (
    compute_reach_accumulation,
    compute_upstream_facc_sum,
    compute_bifurcation_facc_divergence,
)


class FaccFeatureExtractor:
    """
    Extracts features for facc anomaly detection.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection to SWORD database.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn
        self._reach_acc_cache: Dict[str, pd.DataFrame] = {}

    def get_reach_accumulation(self, region: Optional[str] = None) -> pd.DataFrame:
        """Get cached reach accumulation or compute it."""
        cache_key = region or "all"
        if cache_key not in self._reach_acc_cache:
            self._reach_acc_cache[cache_key] = compute_reach_accumulation(
                self.conn, region=region
            )
        return self._reach_acc_cache[cache_key]

    def extract_base_features(self, region: Optional[str] = None) -> pd.DataFrame:
        """
        Extract base features from reaches table.

        Parameters
        ----------
        region : str, optional
            Region to extract for.

        Returns
        -------
        pd.DataFrame
            Base features for all reaches.
        """
        where_clause = f"WHERE r.region = '{region}'" if region else ""

        query = f"""
        SELECT
            r.reach_id,
            r.region,
            r.facc,
            r.width,
            r.reach_length,
            r.dist_out,
            r.stream_order,
            r.path_freq,
            r.n_rch_up,
            r.n_rch_down,
            r.lakeflag,
            r.main_side,
            r.end_reach,
            r.network,
            -- Width-based ratios
            CASE
                WHEN r.width > 0 THEN r.facc / r.width
                ELSE NULL
            END as facc_width_ratio,
            -- SWOT observations if available
            r.n_obs,
            r.width_obs_mean
        FROM reaches r
        WHERE r.facc > 0 AND r.facc != -9999
            AND r.width > 0
        {where_clause.replace("WHERE", "AND") if where_clause else ""}
        """

        return self.conn.execute(query).fetchdf()

    def extract_topology_features(self, region: Optional[str] = None) -> pd.DataFrame:
        """
        Extract topology-based features.

        Includes upstream facc sum and facc jump ratios.

        Parameters
        ----------
        region : str, optional
            Region to extract for.

        Returns
        -------
        pd.DataFrame
            Topology features for all reaches.
        """
        return compute_upstream_facc_sum(self.conn, region=region)

    def extract_bifurcation_features(
        self, region: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract bifurcation-specific features.

        At bifurcations, downstream branches may have divergent facc.

        Parameters
        ----------
        region : str, optional
            Region to extract for.

        Returns
        -------
        pd.DataFrame
            Bifurcation features (only for reaches with n_rch_down >= 2).
        """
        return compute_bifurcation_facc_divergence(self.conn, region=region)

    def compute_reach_acc_ratio(self, region: Optional[str] = None) -> pd.DataFrame:
        """
        Compute facc vs reach_acc ratio.

        The key anomaly detection metric: facc / (reach_acc × baseline)

        Parameters
        ----------
        region : str, optional
            Region to compute for.

        Returns
        -------
        pd.DataFrame
            DataFrame with facc_reach_acc_ratio for anomaly scoring.
        """
        # Get reach accumulation
        reach_acc_df = self.get_reach_accumulation(region)

        # Get base features
        base_df = self.extract_base_features(region)

        # Merge
        merged = base_df.merge(
            reach_acc_df[["reach_id", "region", "reach_acc"]],
            on=["reach_id", "region"],
            how="left",
        )

        # Compute regional baseline: median facc per reach_acc unit
        if len(merged) > 0:
            # Exclude extreme values for baseline calculation
            valid = merged[
                (merged["reach_acc"] > 0)
                & (merged["facc"] > 0)
                & (merged["facc_width_ratio"] < 5000)  # Exclude known corrupted
            ]
            if len(valid) > 0:
                # Compute baseline facc per reach_acc
                baseline = valid["facc"].median() / valid["reach_acc"].median()
            else:
                baseline = 1000  # Default fallback
        else:
            baseline = 1000

        # Compute ratio: how many times expected facc
        merged["expected_facc"] = merged["reach_acc"] * baseline
        merged["facc_reach_acc_ratio"] = np.where(
            merged["expected_facc"] > 0,
            merged["facc"] / merged["expected_facc"],
            np.nan,
        )

        return merged

    def extract_all_features(self, region: Optional[str] = None) -> pd.DataFrame:
        """
        Extract all features for ML model.

        Combines base, topology, bifurcation, and reach_acc features.

        Parameters
        ----------
        region : str, optional
            Region to extract for.

        Returns
        -------
        pd.DataFrame
            Complete feature set for all reaches.
        """
        # Start with reach_acc ratio features
        features = self.compute_reach_acc_ratio(region)

        # Add topology features (upstream facc sum)
        topo_features = self.extract_topology_features(region)
        features = features.merge(
            topo_features[
                [
                    "reach_id",
                    "region",
                    "upstream_facc_sum",
                    "n_upstream",
                    "facc_jump_ratio",
                ]
            ],
            on=["reach_id", "region"],
            how="left",
        )

        # Add bifurcation indicators
        bifurc_features = self.extract_bifurcation_features(region)
        if len(bifurc_features) > 0:
            # Mark reaches that are downstream of suspicious bifurcations
            suspicious_bifurc = bifurc_features[bifurc_features["likely_facc_error"]]
            features["downstream_of_suspicious_bifurc"] = features["reach_id"].isin(
                suspicious_bifurc["reach_id"]
            )
        else:
            features["downstream_of_suspicious_bifurc"] = False

        # Compute composite anomaly features
        features = self._compute_composite_features(features)

        return features

    def _compute_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute composite features for anomaly detection.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with additional composite features.
        """
        df = df.copy()

        # Log-scale features for better ML performance
        df["log_facc"] = np.log1p(df["facc"])
        df["log_width"] = np.log1p(df["width"])
        df["log_reach_acc"] = np.log1p(df["reach_acc"].fillna(1))

        # Normalized ratios
        df["facc_width_ratio_log"] = np.log1p(df["facc_width_ratio"].fillna(0))

        # Stream order interaction
        df["facc_per_stream_order"] = np.where(
            df["stream_order"] > 0, df["facc"] / df["stream_order"], df["facc"]
        )

        # Headwater/outlet flags
        df["is_headwater"] = (df["end_reach"] == 1).astype(int)
        df["is_outlet"] = (df["end_reach"] == 2).astype(int)

        # Fill NaN for ML features
        for col in ["facc_jump_ratio", "upstream_facc_sum", "n_upstream", "reach_acc"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        return df


def extract_facc_features(
    db_path_or_conn: Union[str, duckdb.DuckDBPyConnection], region: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to extract facc features.

    Parameters
    ----------
    db_path_or_conn : str or duckdb.DuckDBPyConnection
        Path to DuckDB database or existing connection.
    region : str, optional
        Region to extract for (e.g., 'NA').

    Returns
    -------
    pd.DataFrame
        Complete feature set for all reaches.

    Examples
    --------
    >>> from sword_duckdb.facc_detection import extract_facc_features
    >>> features = extract_facc_features("sword_v17c.duckdb", region="NA")
    >>> print(features[features['facc_reach_acc_ratio'] > 10].head())
    """
    if isinstance(db_path_or_conn, str):
        conn = duckdb.connect(db_path_or_conn, read_only=True)
        own_conn = True
    else:
        conn = db_path_or_conn
        own_conn = False

    try:
        extractor = FaccFeatureExtractor(conn)
        return extractor.extract_all_features(region=region)
    finally:
        if own_conn:
            conn.close()


def get_seed_reach_features(
    conn: duckdb.DuckDBPyConnection, seed_reach_ids: List[int]
) -> pd.DataFrame:
    """
    Extract features for known corrupted seed reaches.

    Used to establish baseline for what "corrupted" looks like.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection.
    seed_reach_ids : list of int
        Known corrupted reach IDs.

    Returns
    -------
    pd.DataFrame
        Features for seed reaches with corruption mode labels.
    """
    # Known seed reaches from v17c_status.md with their modes
    SEED_MODES = {
        64231000301: "entry",  # facc/width = 35,239
        62236100011: "entry",  # facc/width = 22,811
        62238000021: "entry",  # facc/width = 1,559
        64231000291: "propagation",  # facc/width = 982
        62255000451: "propagation",  # facc/width = 528
    }

    if not seed_reach_ids:
        return pd.DataFrame()

    # Get all features
    extractor = FaccFeatureExtractor(conn)

    # We need to find the region for each seed
    reach_ids_str = ", ".join(str(r) for r in seed_reach_ids)
    regions_df = conn.execute(f"""
        SELECT DISTINCT region FROM reaches WHERE reach_id IN ({reach_ids_str})
    """).fetchdf()

    all_features = []
    for _, row in regions_df.iterrows():
        region = row["region"]
        features = extractor.extract_all_features(region=region)
        seed_features = features[features["reach_id"].isin(seed_reach_ids)]
        all_features.append(seed_features)

    if not all_features:
        return pd.DataFrame()

    result = pd.concat(all_features, ignore_index=True)

    # Add corruption mode
    result["corruption_mode"] = result["reach_id"].map(
        lambda x: SEED_MODES.get(x, "unknown")
    )

    return result
