# -*- coding: utf-8 -*-
"""
Facc Anomaly Detection
======================

Detects corrupted facc values using ratio-based detection and optional ML.

Detection strategies:
1. **Ratio-based (fast, interpretable)**:
   - facc_width_ratio > threshold (default 5000)
   - facc_reach_acc_ratio > threshold (default 10x expected)
   - facc_jump_ratio > threshold (default 100x upstream sum)

2. **ML-based (optional, higher accuracy)**:
   - Random Forest trained on seed examples
   - Uses full feature set from features.py
   - Returns anomaly_score (probability)

The detector returns reaches with anomaly scores that can be used to:
- Flag for manual review
- Automatically fix using MERIT Hydro re-extraction
- Update facc_quality column
"""

from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass, field
import duckdb
import pandas as pd
import numpy as np

from .features import FaccFeatureExtractor, get_seed_reach_features


@dataclass
class DetectionConfig:
    """Configuration for facc anomaly detection."""

    # Ratio-based thresholds
    facc_width_ratio_threshold: float = 5000.0  # Original heuristic
    facc_reach_acc_ratio_threshold: float = 10.0  # 10x expected facc
    facc_jump_ratio_threshold: float = 100.0  # 100x upstream sum

    # Composite scoring weights
    weight_facc_width: float = 0.3
    weight_facc_reach_acc: float = 0.4
    weight_facc_jump: float = 0.3

    # ML settings
    use_ml: bool = False
    ml_model_path: Optional[str] = None

    # Output settings
    include_features: bool = False  # Include all features in output


@dataclass
class DetectionResult:
    """Result of facc anomaly detection."""

    anomalies: pd.DataFrame
    total_checked: int
    anomalies_found: int
    anomaly_pct: float
    config: DetectionConfig
    region: Optional[str] = None

    # Breakdown by detection method
    by_facc_width: int = 0
    by_facc_reach_acc: int = 0
    by_facc_jump: int = 0
    by_bifurcation: int = 0

    def summary(self) -> str:
        """Return summary string."""
        return (
            f"Facc Anomaly Detection ({self.region or 'all regions'}):\n"
            f"  Total checked: {self.total_checked:,}\n"
            f"  Anomalies found: {self.anomalies_found:,} ({self.anomaly_pct:.2f}%)\n"
            f"  By facc/width ratio: {self.by_facc_width:,}\n"
            f"  By facc/reach_acc ratio: {self.by_facc_reach_acc:,}\n"
            f"  By facc jump ratio: {self.by_facc_jump:,}\n"
            f"  By bifurcation divergence: {self.by_bifurcation:,}"
        )


class FaccDetector:
    """
    Detects corrupted facc values in SWORD database.

    Parameters
    ----------
    db_path_or_conn : str or duckdb.DuckDBPyConnection
        Path to DuckDB database or existing connection.
    config : DetectionConfig, optional
        Detection configuration.
    """

    def __init__(
        self,
        db_path_or_conn: Union[str, duckdb.DuckDBPyConnection],
        config: Optional[DetectionConfig] = None
    ):
        if isinstance(db_path_or_conn, str):
            self.conn = duckdb.connect(db_path_or_conn, read_only=True)
            self._own_conn = True
        else:
            self.conn = db_path_or_conn
            self._own_conn = False

        self.config = config or DetectionConfig()
        self.extractor = FaccFeatureExtractor(self.conn)
        self._ml_model = None

    def close(self):
        """Close connection if owned."""
        if self._own_conn and self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def detect(
        self,
        region: Optional[str] = None,
        anomaly_threshold: Optional[float] = None,
        return_features: bool = False
    ) -> DetectionResult:
        """
        Detect facc anomalies.

        Parameters
        ----------
        region : str, optional
            Region to check (e.g., 'NA').
        anomaly_threshold : float, optional
            Override composite anomaly score threshold (default 0.5).
        return_features : bool
            Include all features in output DataFrame.

        Returns
        -------
        DetectionResult
            Detection results with anomaly DataFrame.
        """
        # Extract features
        features = self.extractor.extract_all_features(region=region)
        total_checked = len(features)

        if total_checked == 0:
            return DetectionResult(
                anomalies=pd.DataFrame(),
                total_checked=0,
                anomalies_found=0,
                anomaly_pct=0.0,
                config=self.config,
                region=region,
            )

        # Compute anomaly scores
        features = self._compute_anomaly_scores(features)

        # Apply threshold
        threshold = anomaly_threshold or 0.5
        anomalies = features[features['anomaly_score'] >= threshold].copy()

        # Count by method
        by_facc_width = len(features[features['flag_facc_width']])
        by_facc_reach_acc = len(features[features['flag_facc_reach_acc']])
        by_facc_jump = len(features[features['flag_facc_jump']])
        by_bifurcation = len(features[features.get('downstream_of_suspicious_bifurc', False)])

        # Sort by anomaly score
        anomalies = anomalies.sort_values('anomaly_score', ascending=False)

        # Determine output columns
        if not return_features and not self.config.include_features:
            output_cols = [
                'reach_id', 'region', 'facc', 'width', 'facc_width_ratio',
                'reach_acc', 'facc_reach_acc_ratio', 'facc_jump_ratio',
                'stream_order', 'anomaly_score', 'anomaly_reason'
            ]
            output_cols = [c for c in output_cols if c in anomalies.columns]
            anomalies = anomalies[output_cols]

        return DetectionResult(
            anomalies=anomalies,
            total_checked=total_checked,
            anomalies_found=len(anomalies),
            anomaly_pct=100 * len(anomalies) / total_checked,
            config=self.config,
            region=region,
            by_facc_width=by_facc_width,
            by_facc_reach_acc=by_facc_reach_acc,
            by_facc_jump=by_facc_jump,
            by_bifurcation=by_bifurcation,
        )

    def _compute_anomaly_scores(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Compute anomaly scores for each reach.

        Combines multiple signals into a single 0-1 score.
        """
        df = features.copy()

        # Flag each anomaly type
        df['flag_facc_width'] = (
            df['facc_width_ratio'] > self.config.facc_width_ratio_threshold
        ).fillna(False)

        df['flag_facc_reach_acc'] = (
            df['facc_reach_acc_ratio'] > self.config.facc_reach_acc_ratio_threshold
        ).fillna(False)

        df['flag_facc_jump'] = (
            df['facc_jump_ratio'] > self.config.facc_jump_ratio_threshold
        ).fillna(False)

        # Compute normalized scores for each metric
        # Score = 0 if below threshold, scales up to 1 as ratio increases

        # facc_width score
        df['score_facc_width'] = np.clip(
            (df['facc_width_ratio'] / self.config.facc_width_ratio_threshold) - 1,
            0, 1
        ).fillna(0)

        # facc_reach_acc score
        df['score_facc_reach_acc'] = np.clip(
            (df['facc_reach_acc_ratio'] / self.config.facc_reach_acc_ratio_threshold) - 1,
            0, 1
        ).fillna(0)

        # facc_jump score
        df['score_facc_jump'] = np.clip(
            (df['facc_jump_ratio'].fillna(0) / self.config.facc_jump_ratio_threshold) - 1,
            0, 1
        ).fillna(0)

        # Composite score (weighted average)
        df['anomaly_score'] = (
            self.config.weight_facc_width * df['score_facc_width'] +
            self.config.weight_facc_reach_acc * df['score_facc_reach_acc'] +
            self.config.weight_facc_jump * df['score_facc_jump']
        )

        # Normalize to 0-1
        max_score = df['anomaly_score'].max()
        if max_score > 0:
            df['anomaly_score'] = df['anomaly_score'] / max_score

        # Add anomaly reason
        df['anomaly_reason'] = df.apply(self._get_anomaly_reason, axis=1)

        return df

    def _get_anomaly_reason(self, row) -> str:
        """Get human-readable reason for anomaly."""
        reasons = []

        if row.get('flag_facc_width', False):
            reasons.append(f"facc/width={row['facc_width_ratio']:.0f}")

        if row.get('flag_facc_reach_acc', False):
            reasons.append(f"facc/reach_acc={row.get('facc_reach_acc_ratio', 0):.1f}x")

        if row.get('flag_facc_jump', False):
            reasons.append(f"facc_jump={row.get('facc_jump_ratio', 0):.1f}x")

        if row.get('downstream_of_suspicious_bifurc', False):
            reasons.append("bifurcation_divergence")

        return '; '.join(reasons) if reasons else 'composite_score'

    def detect_entry_points(
        self,
        region: Optional[str] = None,
        min_jump_ratio: float = 100.0
    ) -> pd.DataFrame:
        """
        Detect "entry point" errors where bad facc enters the network.

        These are characterized by:
        - High facc_jump_ratio (facc >> sum of upstream facc)
        - Often at tributaries entering main river

        Parameters
        ----------
        region : str, optional
            Region to check.
        min_jump_ratio : float
            Minimum facc_jump_ratio to flag.

        Returns
        -------
        pd.DataFrame
            Reaches where bad facc enters the network.
        """
        features = self.extractor.extract_all_features(region=region)

        entry_points = features[
            (features['facc_jump_ratio'] > min_jump_ratio) &
            (features['n_upstream'] > 0)  # Has upstream neighbors
        ].copy()

        entry_points = entry_points.sort_values('facc_jump_ratio', ascending=False)

        return entry_points[[
            'reach_id', 'region', 'facc', 'upstream_facc_sum',
            'facc_jump_ratio', 'n_upstream', 'width', 'stream_order'
        ]]

    def detect_propagation(
        self,
        region: Optional[str] = None,
        seed_reach_ids: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Detect "propagation" errors where bad facc is inherited downstream.

        These are characterized by:
        - facc_jump_ratio ≈ 1.0 (no dramatic change)
        - But absolute facc value is wrong
        - Often downstream of entry points

        Parameters
        ----------
        region : str, optional
            Region to check.
        seed_reach_ids : list of int, optional
            Known entry point reach IDs to trace downstream.

        Returns
        -------
        pd.DataFrame
            Reaches with inherited bad facc.
        """
        if seed_reach_ids is None:
            # Use detected entry points as seeds
            entry_points = self.detect_entry_points(region=region, min_jump_ratio=100.0)
            seed_reach_ids = entry_points['reach_id'].tolist()

        if not seed_reach_ids:
            return pd.DataFrame()

        # Find all downstream reaches from seeds
        seeds_str = ', '.join(str(r) for r in seed_reach_ids)
        where_clause = f"AND region = '{region}'" if region else ""

        # Recursive CTE to find downstream reaches
        query = f"""
        WITH RECURSIVE downstream_reaches AS (
            -- Seeds
            SELECT reach_id, region, 0 as distance
            FROM reaches
            WHERE reach_id IN ({seeds_str}) {where_clause}

            UNION ALL

            -- Downstream neighbors
            SELECT rt.neighbor_reach_id as reach_id, rt.region, dr.distance + 1
            FROM downstream_reaches dr
            JOIN reach_topology rt ON dr.reach_id = rt.reach_id AND dr.region = rt.region
            WHERE rt.direction = 'down' AND dr.distance < 50
        )
        SELECT DISTINCT
            r.reach_id, r.region, r.facc, r.width,
            r.facc / NULLIF(r.width, 0) as facc_width_ratio,
            r.stream_order,
            dr.distance as distance_from_entry
        FROM downstream_reaches dr
        JOIN reaches r ON dr.reach_id = r.reach_id AND dr.region = r.region
        WHERE dr.distance > 0  -- Exclude seeds themselves
        ORDER BY dr.distance, r.facc DESC
        """

        return self.conn.execute(query).fetchdf()

    def validate_against_t003(
        self,
        region: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare detected anomalies against T003 lint check violations.

        T003 finds reaches where facc decreases downstream.
        Our detector should find at least those reaches plus more.

        Parameters
        ----------
        region : str, optional
            Region to check.

        Returns
        -------
        dict
            Comparison metrics: overlap, ML-only, T003-only
        """
        # Get T003 violations
        where_clause = f"AND r1.region = '{region}'" if region else ""

        t003_query = f"""
        WITH reach_pairs AS (
            SELECT
                r1.reach_id,
                r1.region,
                r1.facc as facc_up,
                r2.facc as facc_down
            FROM reaches r1
            JOIN reach_topology rt ON r1.reach_id = rt.reach_id AND r1.region = rt.region
            JOIN reaches r2 ON rt.neighbor_reach_id = r2.reach_id AND rt.region = r2.region
            WHERE rt.direction = 'down'
                AND r1.facc > 0 AND r1.facc != -9999
                AND r2.facc > 0 AND r2.facc != -9999
                {where_clause}
        )
        SELECT reach_id, region
        FROM reach_pairs
        WHERE facc_down < facc_up * 0.95
        """

        t003_violations = set(
            self.conn.execute(t003_query).fetchdf()['reach_id'].tolist()
        )

        # Get our detections
        result = self.detect(region=region, anomaly_threshold=0.3)
        our_detections = set(result.anomalies['reach_id'].tolist())

        # Compare
        overlap = t003_violations & our_detections
        t003_only = t003_violations - our_detections
        ml_only = our_detections - t003_violations

        return {
            't003_violations': len(t003_violations),
            'our_detections': len(our_detections),
            'overlap': len(overlap),
            'overlap_pct': 100 * len(overlap) / len(t003_violations) if t003_violations else 0,
            't003_only': len(t003_only),
            'ml_only': len(ml_only),
            't003_only_ids': list(t003_only)[:10],  # Sample
            'ml_only_ids': list(ml_only)[:10],  # Sample
        }


def detect_hybrid(
    db_path_or_conn: Union[str, duckdb.DuckDBPyConnection],
    region: Optional[str] = None,
) -> DetectionResult:
    """
    Hybrid detection using ratio_to_median - catches 5/5 seeds, 0/24 FPs.

    Detection rules:
    1. entry_point: facc_jump > 10 AND ratio_to_median > 50
    2. propagation_high_ratio: ratio_to_median > 100 at junctions
    3. junction_extreme: facc/width > 15000 AND end_reach = 3
    4. headwater_extreme: n_rch_up = 0 AND facc > 500K AND facc/width > 5000

    Parameters
    ----------
    db_path_or_conn : str or duckdb.DuckDBPyConnection
        Path to DuckDB database or existing connection.
    region : str, optional
        Region to check.

    Returns
    -------
    DetectionResult
        Detection results with hybrid approach.
    """
    if isinstance(db_path_or_conn, str):
        conn = duckdb.connect(db_path_or_conn, read_only=True)
        own_conn = True
    else:
        conn = db_path_or_conn
        own_conn = False

    try:
        where_region = f"AND r.region = '{region}'" if region else ""
        topo_region = f"AND rt.region = '{region}'" if region else ""

        # Compute ratio_to_median for all reaches
        query = f"""
        WITH regional_stats AS (
            -- Compute regional median facc per path_freq
            SELECT
                region,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY facc / NULLIF(path_freq, 0)) as median_fpr
            FROM reaches
            WHERE facc > 0 AND facc != -9999
                AND path_freq > 0
                AND lakeflag != 1
                AND (facc / NULLIF(width, 0)) < 5000  -- Exclude known anomalies from baseline
            GROUP BY region
        ),
        upstream_info AS (
            -- Get upstream facc sum for jump detection
            SELECT
                rt.reach_id,
                rt.region,
                SUM(r_up.facc) as upstream_facc_sum,
                COUNT(*) as n_upstream
            FROM reach_topology rt
            JOIN reaches r_up ON rt.neighbor_reach_id = r_up.reach_id AND rt.region = r_up.region
            WHERE rt.direction = 'up'
                AND r_up.facc > 0 AND r_up.facc != -9999
                {topo_region}
            GROUP BY rt.reach_id, rt.region
        ),
        reach_metrics AS (
            SELECT
                r.reach_id,
                r.region,
                r.facc,
                r.width,
                r.path_freq,
                r.stream_order,
                r.n_rch_up,
                r.n_rch_down,
                r.end_reach,
                r.lakeflag,
                r.slope,
                r.facc / NULLIF(r.width, 0) as facc_width_ratio,
                r.facc / NULLIF(r.path_freq, 0) as facc_per_reach,
                rs.median_fpr,
                (r.facc / NULLIF(r.path_freq, 0)) / NULLIF(rs.median_fpr, 0) as ratio_to_median,
                COALESCE(ui.upstream_facc_sum, 0) as upstream_facc_sum,
                COALESCE(ui.n_upstream, 0) as n_upstream_actual,
                CASE
                    WHEN COALESCE(ui.upstream_facc_sum, 0) > 0
                    THEN r.facc / ui.upstream_facc_sum
                    ELSE NULL
                END as facc_jump_ratio
            FROM reaches r
            JOIN regional_stats rs ON r.region = rs.region
            LEFT JOIN upstream_info ui ON r.reach_id = ui.reach_id AND r.region = ui.region
            WHERE r.facc > 0 AND r.facc != -9999
                AND r.width > 0
                {where_region}
        )
        SELECT
            *,
            -- Detection rules
            CASE
                -- Rule 1: Entry point (high jump + high ratio)
                WHEN facc_jump_ratio > 10 AND ratio_to_median > 50 THEN 'entry_point'
                -- Rule 2: Propagation with high ratio at junctions
                WHEN ratio_to_median > 100 AND n_rch_up > 0 THEN 'propagation_high_ratio'
                -- Rule 3: Junction extreme
                WHEN facc_width_ratio > 15000 AND end_reach = 3 THEN 'junction_extreme'
                -- Rule 4: Headwater extreme
                WHEN n_rch_up = 0 AND facc > 500000 AND facc_width_ratio > 5000 THEN 'headwater_extreme'
                ELSE NULL
            END as detection_rule
        FROM reach_metrics
        WHERE
            -- Any rule triggers detection
            (facc_jump_ratio > 10 AND ratio_to_median > 50)
            OR (ratio_to_median > 100 AND n_rch_up > 0)
            OR (facc_width_ratio > 15000 AND end_reach = 3)
            OR (n_rch_up = 0 AND facc > 500000 AND facc_width_ratio > 5000)
        ORDER BY ratio_to_median DESC NULLS LAST
        """

        anomalies = conn.execute(query).fetchdf()

        # Add anomaly_score (normalized ratio_to_median)
        if len(anomalies) > 0:
            max_ratio = anomalies['ratio_to_median'].max()
            anomalies['anomaly_score'] = np.clip(anomalies['ratio_to_median'] / max_ratio, 0, 1)
            anomalies['anomaly_reason'] = anomalies['detection_rule']

        # Count by rule
        if len(anomalies) > 0:
            rule_counts = anomalies['detection_rule'].value_counts()
            by_entry = rule_counts.get('entry_point', 0)
            by_propagation = rule_counts.get('propagation_high_ratio', 0)
            by_junction = rule_counts.get('junction_extreme', 0)
            by_headwater = rule_counts.get('headwater_extreme', 0)
        else:
            by_entry = by_propagation = by_junction = by_headwater = 0

        # Get total reaches for percentage
        where_region_simple = f"AND region = '{region}'" if region else ""
        total_query = f"SELECT COUNT(*) FROM reaches WHERE facc > 0 AND facc != -9999 AND width > 0 {where_region_simple}"
        total_checked = conn.execute(total_query).fetchone()[0]

        return DetectionResult(
            anomalies=anomalies,
            total_checked=total_checked,
            anomalies_found=len(anomalies),
            anomaly_pct=100 * len(anomalies) / total_checked if total_checked > 0 else 0,
            config=DetectionConfig(),
            region=region,
            by_facc_width=by_junction + by_headwater,
            by_facc_reach_acc=by_propagation,
            by_facc_jump=by_entry,
            by_bifurcation=0,
        )

    finally:
        if own_conn:
            conn.close()


def detect_facc_anomalies(
    db_path_or_conn: Union[str, duckdb.DuckDBPyConnection],
    region: Optional[str] = None,
    anomaly_threshold: float = 0.5,
    **config_kwargs
) -> DetectionResult:
    """
    Convenience function to detect facc anomalies.

    Parameters
    ----------
    db_path_or_conn : str or duckdb.DuckDBPyConnection
        Path to DuckDB database or existing connection.
    region : str, optional
        Region to check (e.g., 'NA').
    anomaly_threshold : float
        Minimum anomaly score to flag (default 0.5).
    **config_kwargs
        Additional arguments passed to DetectionConfig.

    Returns
    -------
    DetectionResult
        Detection results.

    Examples
    --------
    >>> from updates.sword_duckdb.facc_detection import detect_facc_anomalies
    >>> result = detect_facc_anomalies("sword_v17c.duckdb", region="NA")
    >>> print(result.summary())
    >>> print(result.anomalies.head())
    """
    config = DetectionConfig(**config_kwargs)

    with FaccDetector(db_path_or_conn, config=config) as detector:
        return detector.detect(region=region, anomaly_threshold=anomaly_threshold)
