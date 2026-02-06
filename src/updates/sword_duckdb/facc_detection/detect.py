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
from pathlib import Path
import json
import duckdb
import pandas as pd
import numpy as np

from .features import FaccFeatureExtractor, get_seed_reach_features

# Known false positives - reaches that look anomalous but are legitimate
# These are used to filter detection results and track FP patterns
KNOWN_FALSE_POSITIVES = {
    # Ob River multi-channel (legitimate high facc, consistent FWR through network)
    31239000161: {'region': 'AS', 'reason': 'Ob River multi-channel'},
    31239000251: {'region': 'AS', 'reason': 'Ob River multi-channel'},
    31231000181: {'region': 'AS', 'reason': 'Ob River junction, consistent FWR'},
    # Narrow width inflating FWR (width < 15m)
    28160700191: {'region': 'EU', 'reason': 'width=11m, consistent FWR up/down'},
    45585500221: {'region': 'AS', 'reason': 'width=2m, upstream/downstream FWR=N/A'},
    28106300011: {'region': 'EU', 'reason': 'width=2m, narrow channel'},
    28105000371: {'region': 'EU', 'reason': 'width=8m, narrow channel'},
    # Complex tidal/delta areas
    45630500041: {'region': 'AS', 'reason': 'Indus junction, strange geometry'},
    44570000065: {'region': 'AS', 'reason': 'Irrawaddy tidal, main_side=2'},
    # Nile delta distributaries (legitimate high facc, consistent FWR, main_side=2)
    17211100904: {'region': 'AF', 'reason': 'Nile Rosetta branch, delta distributary, consistent FWR'},
    # AF region - moderate FWR, consistent through network
    17291500221: {'region': 'AF', 'reason': 'path_freq=1 but moderate FWR, consistent'},
    17291500351: {'region': 'AF', 'reason': 'main_side=1 but moderate FWR, consistent'},
    # RF regressor false positives (2026-02-05) - wrongly corrected
    77250000153: {'region': 'OC', 'reason': 'mainstem reach, should not be corrected'},
    74300400575: {'region': 'SA', 'reason': 'incorrectly flipped by RF regressor'},
    74300400565: {'region': 'SA', 'reason': 'incorrectly flipped by RF regressor'},
}


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
    filter_false_positives: bool = True,
    min_width: float = 15.0,
    include_propagation: bool = True,
    propagation_max_hops: int = 3,
) -> DetectionResult:
    """
    Hybrid detection using ratio_to_median with FP filtering.

    Detection rules:
    1. entry_point: facc_jump > 10 AND ratio_to_median > 50
    2. extreme_fwr: facc/width > 15000
    3. headwater_extreme: n_rch_up = 0 AND facc > 500K AND facc/width > 5000
    4. jump_entry: path_freq invalid AND facc_jump > 20 AND FWR > 500
    5. high_ratio: ratio_to_median > 500 AND (fwr_drop > 2 OR no downstream)
    6. fwr_drop: FWR drops >5x downstream AND FWR > 500
    7. impossible_headwater: path_freq <= 2 AND facc > 1M AND (fwr_drop > 2 OR FWR > 5000)
    8. upstream_fwr_spike: upstream FWR > 10x this reach AND facc > 100K
    9. side_channel_mainstem_facc: main_side=1 + dramatic FWR drop downstream
    10. invalid_side_channel: path_freq=-9999 + main_side=1 + facc>200K + fwr_drop>3
    11. facc_sum_inflation: facc > 3x sum(upstream) at confluence (n_rch_up >= 2)

    FP filtering:
    - Excludes reaches with width < min_width (inflated FWR from narrow width)
    - Excludes known FPs from KNOWN_FALSE_POSITIVES
    - Excludes consistent flow-through channels (facc_jump <= 2 AND width_ratio > 0.5)

    Parameters
    ----------
    db_path_or_conn : str or duckdb.DuckDBPyConnection
        Path to DuckDB database or existing connection.
    region : str, optional
        Region to check.
    filter_false_positives : bool, default True
        If True, filter out reaches with no facc_jump + wide (likely FPs).
    min_width : float, default 15.0
        Minimum width in meters. Reaches narrower than this are excluded
        (FWR is artificially inflated for narrow reaches).

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

        # Build known FP exclusion list
        known_fp_ids = list(KNOWN_FALSE_POSITIVES.keys())
        known_fp_str = ', '.join(str(r) for r in known_fp_ids) if known_fp_ids else '0'

        # Build FP filter clause
        # FPs have: facc_jump <= 2 AND (width_ratio_to_dn > 0.5 OR long+wide)
        # We EXCLUDE these from detection UNLESS:
        # - ratio_to_median > 100 (clearly anomalous)
        # - main_side = 1 (side channel - keep for side_channel rules)
        # - path_freq invalid (may have real issues masked by bad metadata)
        # - fwr_drop_ratio > 10 (dramatic FWR change indicates problem)
        fp_filter = ""
        if filter_false_positives:
            fp_filter = f"""
                -- Exclude known false positives
                AND reach_id NOT IN ({known_fp_str})
                -- Exclude FPs: no facc_jump + wide relative to downstream
                AND NOT (
                    (facc_jump_ratio IS NOT NULL AND facc_jump_ratio <= 2)
                    AND (
                        width_ratio_to_dn > 0.5
                        OR (reach_length > 10000 AND width > 200)
                    )
                    AND (ratio_to_median < 100 OR ratio_to_median IS NULL OR ratio_to_median < 0)
                    -- BUT don't filter if:
                    AND (main_side != 1)  -- Keep side channels
                    AND (path_freq > 0 AND path_freq != -9999)  -- Keep invalid path_freq
                    AND (fwr_drop_ratio IS NULL OR fwr_drop_ratio <= 10)  -- Keep dramatic FWR drops
                )
                -- 1:1:1 stable filter DISABLED for v17b baseline
                -- On v17b, propagation is stable (bad facc flows consistently)
                -- These reaches need correction, not filtering
                -- AND NOT (n_rch_up = 1 AND n_rch_down = 1 AND end_reach != 3
                --          AND facc_jump_ratio IS NOT NULL AND facc_jump_ratio BETWEEN 0.9 AND 1.1
                --          AND facc_diff_from_downstream_pct IS NOT NULL AND facc_diff_from_downstream_pct < 0.1)
            """

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
            -- Get MAX upstream facc and FWR for jump/spike detection
            SELECT
                rt.reach_id,
                rt.region,
                MAX(r_up.facc) as max_upstream_facc,
                SUM(r_up.facc) as upstream_facc_sum,
                COUNT(*) as n_upstream,
                MAX(r_up.facc / NULLIF(r_up.width, 0)) as max_upstream_fwr,
                MAX(r_up.type) as max_upstream_type
            FROM reach_topology rt
            JOIN reaches r_up ON rt.neighbor_reach_id = r_up.reach_id AND rt.region = r_up.region
            WHERE rt.direction = 'up'
                AND r_up.facc > 0 AND r_up.facc != -9999
                {topo_region}
            GROUP BY rt.reach_id, rt.region
        ),
        downstream_info AS (
            -- Get max downstream width, FWR, and facc for ratio calculations
            SELECT
                rt.reach_id,
                rt.region,
                MAX(r_dn.width) as max_downstream_width,
                MAX(r_dn.facc / NULLIF(r_dn.width, 0)) as max_downstream_fwr,
                MAX(r_dn.facc) as max_downstream_facc
            FROM reach_topology rt
            JOIN reaches r_dn ON rt.neighbor_reach_id = r_dn.reach_id AND rt.region = r_dn.region
            WHERE rt.direction = 'down'
                {topo_region}
            GROUP BY rt.reach_id, rt.region
        ),
        reach_metrics AS (
            SELECT
                r.reach_id,
                r.region,
                r.facc,
                r.width,
                r.reach_length,
                r.path_freq,
                r.stream_order,
                r.n_rch_up,
                r.n_rch_down,
                r.end_reach,
                r.lakeflag,
                r.main_side,
                r.type,
                r.slope,
                -- Cap width at min_width to avoid inflated FWR from narrow reaches
                r.facc / GREATEST(r.width, {min_width}) as facc_width_ratio,
                r.facc / NULLIF(r.path_freq, 0) as facc_per_reach,
                rs.median_fpr,
                (r.facc / NULLIF(r.path_freq, 0)) / NULLIF(rs.median_fpr, 0) as ratio_to_median,
                COALESCE(ui.max_upstream_facc, 0) as max_upstream_facc,
                COALESCE(ui.upstream_facc_sum, 0) as upstream_facc_sum,
                COALESCE(ui.n_upstream, 0) as n_upstream_actual,
                ui.max_upstream_fwr,
                ui.max_upstream_type,
                -- Upstream FWR spike: how much higher is upstream FWR than this reach?
                ui.max_upstream_fwr / NULLIF(r.facc / GREATEST(r.width, {min_width}), 0) as upstream_fwr_ratio,
                CASE
                    WHEN COALESCE(ui.max_upstream_facc, 0) > 0
                    THEN r.facc / ui.max_upstream_facc
                    ELSE NULL
                END as facc_jump_ratio,
                r.width / NULLIF(di.max_downstream_width, 0) as width_ratio_to_dn,
                di.max_downstream_fwr,
                di.max_downstream_facc,
                -- FWR drop ratio: how much does FWR drop going downstream?
                (r.facc / GREATEST(r.width, {min_width})) / NULLIF(di.max_downstream_fwr, 0) as fwr_drop_ratio,
                -- Facc difference from downstream (for 1:1:1 stable filter)
                ABS(r.facc - COALESCE(di.max_downstream_facc, 0)) / NULLIF(di.max_downstream_facc, 0) as facc_diff_from_downstream_pct
            FROM reaches r
            JOIN regional_stats rs ON r.region = rs.region
            LEFT JOIN upstream_info ui ON r.reach_id = ui.reach_id AND r.region = ui.region
            LEFT JOIN downstream_info di ON r.reach_id = di.reach_id AND r.region = di.region
            WHERE r.facc > 0 AND r.facc != -9999
                AND r.width >= 0  -- Allow zero width (FWR capped to min_width)
                {where_region}
        )
        SELECT
            *,
            -- Detection rules
            CASE
                -- Rule 1: Entry point (high jump + elevated ratio)
                WHEN facc_jump_ratio > 10 AND ratio_to_median > 40 THEN 'entry_point'
                -- Rule 2: Extreme FWR (facc/width > 15000, regardless of jump)
                -- Catches cases where upstream also has bad facc (no jump)
                WHEN facc_width_ratio > 15000 THEN 'extreme_fwr'
                -- Rule 3: Headwater extreme (no upstream = true entry point)
                WHEN n_rch_up = 0 AND facc > 500000 AND facc_width_ratio > 5000 THEN 'headwater_extreme'
                -- Rule 4: Jump entry (path_freq invalid but large facc jump from upstream)
                -- Catches D8 routing errors where bad facc enters mid-network
                WHEN (path_freq <= 0 OR path_freq = -9999) AND facc_jump_ratio > 20 AND facc_width_ratio > 500 THEN 'jump_entry'
                -- Rule 5: Very high ratio_to_median (even without large FWR)
                -- BUT require fwr_drop > 2 to exclude legitimate multi-channel rivers (Ob, etc.)
                WHEN ratio_to_median > 500 AND (fwr_drop_ratio > 2 OR fwr_drop_ratio IS NULL) THEN 'high_ratio'
                -- Rule 6: FWR drops dramatically downstream (facc doesn't belong here)
                -- Key discriminator: bad facc has high FWR that drops downstream
                WHEN fwr_drop_ratio > 5 AND facc_width_ratio > 500 THEN 'fwr_drop'
                -- Rule 7: Impossible facc for path_freq (mainstem facc on tributary)
                -- path_freq=1 means near headwater, shouldn't have >1M facc
                -- Require FWR inconsistency (drop > 2x) to exclude delta distributaries with consistent FWR
                WHEN path_freq <= 2 AND facc > 1000000 AND (fwr_drop_ratio > 2 OR fwr_drop_ratio IS NULL) THEN 'impossible_headwater'
                -- Rule 8: Upstream FWR spike (upstream has much higher FWR than this reach)
                -- Indicates bad facc entered upstream and propagated here
                WHEN upstream_fwr_ratio > 10 AND facc > 100000 THEN 'upstream_fwr_spike'
                -- Rule 9: Side channel with mainstem facc (main_side=1 + extreme FWR drop)
                -- Side channels shouldn't have dramatic FWR that drops to near-zero downstream
                WHEN main_side = 1 AND fwr_drop_ratio > 20 AND facc > 100000 THEN 'side_channel_misroute'
                -- Rule 10: Invalid path_freq side channel (common pattern in seeds)
                -- path_freq=-9999 with main_side=1 and significant facc
                WHEN (path_freq <= 0 OR path_freq = -9999) AND main_side = 1 AND facc > 200000 AND fwr_drop_ratio > 3 THEN 'invalid_side_channel'
                -- Rule 11: facc inflation at confluence (facc >> sum of tributaries)
                -- At confluences, facc should roughly equal sum of upstream. 3x+ indicates D8 error.
                WHEN n_rch_up >= 2 AND upstream_facc_sum > 50000 AND facc > 3 * upstream_facc_sum THEN 'facc_sum_inflation'
                ELSE NULL
            END as detection_rule
        FROM reach_metrics
        WHERE
            -- Any rule triggers detection
            (
                (facc_jump_ratio > 10 AND ratio_to_median > 40)
                OR (facc_width_ratio > 15000)
                OR (n_rch_up = 0 AND facc > 500000 AND facc_width_ratio > 5000)
                OR ((path_freq <= 0 OR path_freq = -9999) AND facc_jump_ratio > 20 AND facc_width_ratio > 500)
                OR (ratio_to_median > 500 AND (fwr_drop_ratio > 2 OR fwr_drop_ratio IS NULL))
                OR (fwr_drop_ratio > 5 AND facc_width_ratio > 500)
                OR (path_freq <= 2 AND facc > 1000000 AND (fwr_drop_ratio > 2 OR fwr_drop_ratio IS NULL))
                OR (upstream_fwr_ratio > 10 AND facc > 100000)
                OR (main_side = 1 AND fwr_drop_ratio > 20 AND facc > 100000)
                OR ((path_freq <= 0 OR path_freq = -9999) AND main_side = 1 AND facc > 200000 AND fwr_drop_ratio > 3)
                OR (n_rch_up >= 2 AND upstream_facc_sum > 50000 AND facc > 3 * upstream_facc_sum)
            )
            {fp_filter}
        ORDER BY ratio_to_median DESC NULLS LAST
        """

        anomalies = conn.execute(query).fetchdf()

        # Add anomaly_score (normalized ratio_to_median)
        if len(anomalies) > 0:
            max_ratio = anomalies['ratio_to_median'].max()
            anomalies['anomaly_score'] = np.clip(anomalies['ratio_to_median'] / max_ratio, 0, 1)
            anomalies['anomaly_reason'] = anomalies['detection_rule']

        # Pass 2: Topology-aware propagation detection
        if include_propagation and len(anomalies) > 0:
            entry_rules = ['entry_point', 'extreme_fwr', 'headwater_extreme', 'jump_entry',
                          'facc_sum_inflation']
            entry_df = anomalies[anomalies['detection_rule'].isin(entry_rules)]

            if len(entry_df) > 0:
                entry_ids = entry_df['reach_id'].tolist()
                entry_facc = dict(zip(entry_df['reach_id'], entry_df['facc']))

                propagation = detect_propagation_topology_aware(
                    conn=conn,
                    entry_point_ids=entry_ids,
                    entry_point_facc=entry_facc,
                    max_hops=propagation_max_hops,
                    min_width=min_width,
                )

                if len(propagation) > 0:
                    already_detected = set(anomalies['reach_id'].tolist())
                    propagation = propagation[~propagation['reach_id'].isin(already_detected)]

                    if len(propagation) > 0:
                        propagation_aligned = propagation.reindex(columns=anomalies.columns)
                        anomalies = pd.concat([anomalies, propagation_aligned], ignore_index=True)

        # Count by rule
        if len(anomalies) > 0:
            rule_counts = anomalies['detection_rule'].value_counts()
            by_entry = rule_counts.get('entry_point', 0)
            by_extreme_fwr = rule_counts.get('extreme_fwr', 0)
            by_headwater = rule_counts.get('headwater_extreme', 0)
            by_jump_entry = rule_counts.get('jump_entry', 0)
            by_high_ratio = rule_counts.get('high_ratio', 0)
            by_fwr_drop = rule_counts.get('fwr_drop', 0)
            by_side_channel = rule_counts.get('side_channel_misroute', 0) + rule_counts.get('invalid_side_channel', 0)
        else:
            by_entry = by_extreme_fwr = by_headwater = by_jump_entry = by_high_ratio = by_fwr_drop = by_side_channel = 0

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
            by_facc_width=by_extreme_fwr + by_headwater,
            by_facc_reach_acc=by_high_ratio + by_fwr_drop,
            by_facc_jump=by_entry + by_jump_entry,
            by_bifurcation=by_side_channel,
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


def export_categorized_geojsons(
    db_path_or_conn: Union[str, duckdb.DuckDBPyConnection],
    anomalies: pd.DataFrame,
    output_dir: Union[str, Path],
    seed_reach_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Export detection results as categorized GeoJSON files for QGIS review.

    Creates separate GeoJSON files for each detection rule, plus:
    - propagation.geojson: downstream reaches from entry points
    - all_anomalies.geojson: combined for quick overview
    - detection_summary.json: counts and seed verification

    Parameters
    ----------
    db_path_or_conn : str or duckdb.DuckDBPyConnection
        Path to DuckDB database or existing connection.
    anomalies : pd.DataFrame
        Detection results from detect_hybrid() with detection_rule column.
    output_dir : str or Path
        Directory to write GeoJSON files.
    seed_reach_ids : list of int, optional
        Known seed reach IDs to verify in summary.

    Returns
    -------
    dict
        Summary with file paths and counts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(db_path_or_conn, str):
        conn = duckdb.connect(db_path_or_conn, read_only=True)
        own_conn = True
    else:
        conn = db_path_or_conn
        own_conn = False

    try:
        # Load spatial extension for ST_AsGeoJSON
        conn.execute("INSTALL spatial; LOAD spatial;")
        summary = {
            'total_anomalies': len(anomalies),
            'by_rule': {},
            'files': {},
            'seeds_detected': [],
            'seeds_missed': [],
        }

        if len(anomalies) == 0:
            # Write empty summary
            summary_path = output_dir / 'detection_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            summary['files']['summary'] = str(summary_path)
            return summary

        # Get geometries for all anomalies
        reach_ids = anomalies['reach_id'].tolist()
        reach_ids_str = ', '.join(str(r) for r in reach_ids)

        geom_query = f"""
        SELECT
            reach_id,
            region,
            ST_AsGeoJSON(geom) as geom_json
        FROM reaches
        WHERE reach_id IN ({reach_ids_str})
        """
        geom_df = conn.execute(geom_query).fetchdf()

        # Merge geometry with anomalies
        anomalies_with_geom = anomalies.merge(
            geom_df[['reach_id', 'geom_json']],
            on='reach_id',
            how='left'
        )

        # Export by detection rule
        rules = anomalies['detection_rule'].dropna().unique()
        for rule in rules:
            rule_df = anomalies_with_geom[anomalies_with_geom['detection_rule'] == rule]
            if len(rule_df) == 0:
                continue

            filename = f"{rule}.geojson"
            filepath = output_dir / filename
            _write_geojson(rule_df, filepath)

            summary['by_rule'][rule] = len(rule_df)
            summary['files'][rule] = str(filepath)

        # Export combined all_anomalies.geojson
        all_filepath = output_dir / 'all_anomalies.geojson'
        _write_geojson(anomalies_with_geom, all_filepath)
        summary['files']['all_anomalies'] = str(all_filepath)

        # Verify seeds
        if seed_reach_ids:
            detected_ids = set(anomalies['reach_id'].tolist())
            seeds_detected = [s for s in seed_reach_ids if s in detected_ids]
            seeds_missed = [s for s in seed_reach_ids if s not in detected_ids]
            summary['seeds_detected'] = seeds_detected
            summary['seeds_missed'] = seeds_missed
            summary['seed_recall'] = len(seeds_detected) / len(seed_reach_ids) if seed_reach_ids else 0

        # Write summary JSON
        summary_path = output_dir / 'detection_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        summary['files']['summary'] = str(summary_path)

        return summary

    finally:
        if own_conn:
            conn.close()


def _write_geojson(df: pd.DataFrame, filepath: Path) -> None:
    """Write DataFrame with geom_json column to GeoJSON file."""
    features = []

    for _, row in df.iterrows():
        if pd.isna(row.get('geom_json')):
            continue

        # Build properties from all non-geometry columns
        properties = {}
        for col in df.columns:
            if col == 'geom_json':
                continue
            val = row[col]
            # Convert numpy types to Python types for JSON serialization
            if pd.isna(val):
                properties[col] = None
            elif isinstance(val, (np.integer, np.int64)):
                properties[col] = int(val)
            elif isinstance(val, (np.floating, np.float64)):
                properties[col] = float(val)
            else:
                properties[col] = val

        try:
            geometry = json.loads(row['geom_json'])
        except (json.JSONDecodeError, TypeError):
            continue

        features.append({
            'type': 'Feature',
            'properties': properties,
            'geometry': geometry,
        })

    geojson = {
        'type': 'FeatureCollection',
        'features': features,
    }

    with open(filepath, 'w') as f:
        json.dump(geojson, f)


def _trace_downstream(
    conn: duckdb.DuckDBPyConnection,
    entry_reach_ids: List[int],
    max_depth: int = 50,
) -> pd.DataFrame:
    """
    Trace downstream from entry points to find propagation reaches.

    Returns reaches downstream of entry points (excluding the entry points themselves).
    """
    if not entry_reach_ids:
        return pd.DataFrame()

    seeds_str = ', '.join(str(r) for r in entry_reach_ids)

    query = f"""
    WITH RECURSIVE downstream_reaches AS (
        -- Seeds (entry points)
        SELECT
            reach_id,
            region,
            reach_id as entry_reach_id,
            0 as distance
        FROM reaches
        WHERE reach_id IN ({seeds_str})

        UNION ALL

        -- Downstream neighbors
        SELECT
            rt.neighbor_reach_id as reach_id,
            rt.region,
            dr.entry_reach_id,
            dr.distance + 1 as distance
        FROM downstream_reaches dr
        JOIN reach_topology rt ON dr.reach_id = rt.reach_id AND dr.region = rt.region
        WHERE rt.direction = 'down'
            AND dr.distance < {max_depth}
    )
    SELECT DISTINCT
        reach_id,
        region,
        entry_reach_id,
        MIN(distance) as distance_from_entry
    FROM downstream_reaches
    WHERE distance > 0  -- Exclude entry points themselves
    GROUP BY reach_id, region, entry_reach_id
    ORDER BY distance_from_entry
    """

    return conn.execute(query).fetchdf()


def detect_propagation_topology_aware(
    conn: duckdb.DuckDBPyConnection,
    entry_point_ids: List[int],
    entry_point_facc: Dict[int, float],
    max_hops: int = 3,
    facc_similarity_threshold: float = 0.9,
    facc_drop_threshold: float = 0.5,
    min_width: float = 15.0,
) -> pd.DataFrame:
    """
    Trace downstream from entry points and flag propagation reaches.

    A reach is flagged as propagation if:
    1. It's within max_hops of an entry point
    2. Its facc is similar to entry point (within similarity threshold)
    3. Downstream facc drops significantly (indicates bad facc doesn't belong)

    This catches "1:1:1 stable propagation" patterns where bad facc flows
    through unchanged.
    """
    if not entry_point_ids:
        return pd.DataFrame()

    seeds_str = ', '.join(str(r) for r in entry_point_ids)

    query = f"""
    WITH RECURSIVE downstream_trace AS (
        SELECT
            r.reach_id,
            r.region,
            r.facc,
            r.width,
            r.reach_id as entry_reach_id,
            0 as hops
        FROM reaches r
        WHERE r.reach_id IN ({seeds_str})

        UNION ALL

        SELECT
            r_dn.reach_id,
            r_dn.region,
            r_dn.facc,
            r_dn.width,
            dt.entry_reach_id,
            dt.hops + 1 as hops
        FROM downstream_trace dt
        JOIN reach_topology rt ON dt.reach_id = rt.reach_id AND dt.region = rt.region
        JOIN reaches r_dn ON rt.neighbor_reach_id = r_dn.reach_id AND rt.region = r_dn.region
        WHERE rt.direction = 'down'
            AND dt.hops < {max_hops}
            AND r_dn.facc > 0 AND r_dn.facc != -9999
    ),
    with_downstream AS (
        SELECT
            dt.reach_id,
            dt.region,
            dt.facc,
            dt.width,
            dt.entry_reach_id,
            dt.hops,
            MAX(r_dn.facc) as downstream_facc
        FROM downstream_trace dt
        LEFT JOIN reach_topology rt ON dt.reach_id = rt.reach_id AND dt.region = rt.region
        LEFT JOIN reaches r_dn ON rt.neighbor_reach_id = r_dn.reach_id AND rt.region = r_dn.region
            AND r_dn.facc > 0 AND r_dn.facc != -9999
        WHERE dt.hops > 0
            AND (rt.direction = 'down' OR rt.direction IS NULL)
        GROUP BY dt.reach_id, dt.region, dt.facc, dt.width, dt.entry_reach_id, dt.hops
    )
    SELECT DISTINCT
        reach_id,
        region,
        facc,
        width,
        facc / GREATEST(width, {min_width}) as facc_width_ratio,
        entry_reach_id,
        hops as hops_from_entry,
        downstream_facc
    FROM with_downstream
    ORDER BY entry_reach_id, hops
    """

    traced = conn.execute(query).fetchdf()

    if len(traced) == 0:
        return pd.DataFrame()

    traced['entry_facc'] = traced['entry_reach_id'].map(entry_point_facc)
    traced['facc_similarity'] = traced['facc'] / traced['entry_facc']
    traced['facc_drop_ratio'] = traced['downstream_facc'] / traced['facc']

    propagation = traced[
        (traced['facc_similarity'] >= facc_similarity_threshold)
        & (
            (traced['facc_drop_ratio'] < facc_drop_threshold)
            | (traced['downstream_facc'].isna())
        )
    ].copy()

    if len(propagation) == 0:
        return pd.DataFrame()

    propagation['detection_rule'] = 'propagation_from_entry'

    def _make_reason(r):
        dn_facc = f"{r['downstream_facc']:.0f}" if pd.notna(r['downstream_facc']) else 'outlet'
        return (f"propagation: {r['hops_from_entry']} hops from {r['entry_reach_id']}, "
                f"facc={r['facc']:.0f} (~{r['facc_similarity']*100:.0f}% of entry), "
                f"drops to {dn_facc}")

    propagation['anomaly_reason'] = propagation.apply(_make_reason, axis=1)
    propagation['ratio_to_median'] = None
    propagation['anomaly_score'] = 0.8

    return propagation[[
        'reach_id', 'region', 'facc', 'width', 'facc_width_ratio',
        'entry_reach_id', 'entry_facc', 'hops_from_entry',
        'downstream_facc', 'facc_drop_ratio', 'facc_similarity',
        'detection_rule', 'anomaly_reason', 'anomaly_score'
    ]]
