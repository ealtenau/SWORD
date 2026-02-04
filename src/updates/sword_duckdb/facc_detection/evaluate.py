# -*- coding: utf-8 -*-
"""
Facc Detection Evaluation
=========================

Evaluates facc anomaly detection against known corrupted reaches.

Metrics:
- Precision: What % of flagged reaches are truly corrupted?
- Recall: What % of known corruptions are flagged?
- F1: Harmonic mean of precision and recall

Workflow:
1. Start with known seed reaches (from v17c_status.md)
2. Detect anomalies and compare
3. Expand training set via manual review
4. Iterate to improve model
"""

from typing import Optional, List, Dict, Any, Union, Set
from dataclasses import dataclass
import duckdb
import pandas as pd
import numpy as np

from .detect import FaccDetector, DetectionResult, DetectionConfig
from .features import get_seed_reach_features


# Known corrupted reaches from v17c_status.md
# All seeds are BAD in v17b (have corrupted facc values)
SEED_REACHES = {
    # Original 5 (SA region - Amazon basin)
    64231000301: {'mode': 'propagation', 'region': 'SA', 'facc_width_ratio': 35239},
    62236100011: {'mode': 'entry', 'region': 'SA', 'facc_width_ratio': 22811},
    62238000021: {'mode': 'entry', 'region': 'SA', 'facc_width_ratio': 1559},
    64231000291: {'mode': 'propagation', 'region': 'SA', 'facc_width_ratio': 982},
    62255000451: {'mode': 'propagation', 'region': 'SA', 'facc_width_ratio': 528},
    # Additional SA seeds
    17211100181: {'mode': 'entry', 'region': 'SA'},
    13261100101: {'mode': 'entry', 'region': 'SA'},
    13214000011: {'mode': 'entry', 'region': 'SA'},
    13212000011: {'mode': 'entry', 'region': 'SA'},
    # Critical: facc should be on parallel channel (62210000055, 62210000045, 62210000035)
    62210000705: {'mode': 'misrouted', 'region': 'SA'},
    62233000095: {'mode': 'entry', 'region': 'SA'},
    # EU seeds
    28315000523: {'mode': 'propagation', 'region': 'EU'},
    28315000751: {'mode': 'propagation', 'region': 'EU', 'note': 'inherited from nearby lake'},
    28315000783: {'mode': 'entry', 'region': 'EU'},
    # AF seeds
    31251000111: {'mode': 'entry', 'region': 'AF'},
    31248100141: {'mode': 'jump_entry', 'region': 'AF', 'note': 'side channel inherited downstream facc'},
    32257000231: {'mode': 'entry', 'region': 'AF'},
}


@dataclass
class EvaluationResult:
    """Result of evaluating facc detection."""

    precision: float  # TP / (TP + FP)
    recall: float     # TP / (TP + FN)
    f1: float         # 2 * precision * recall / (precision + recall)

    true_positives: int
    false_positives: int
    false_negatives: int

    # Detailed breakdown
    detected_seeds: List[int]
    missed_seeds: List[int]
    false_positive_ids: List[int]

    # T003 comparison
    t003_overlap: Dict[str, Any]

    def summary(self) -> str:
        """Return summary string."""
        return (
            f"Facc Detection Evaluation:\n"
            f"  Precision: {self.precision:.2%}\n"
            f"  Recall: {self.recall:.2%}\n"
            f"  F1 Score: {self.f1:.2%}\n"
            f"\n"
            f"  True Positives: {self.true_positives}\n"
            f"  False Positives: {self.false_positives}\n"
            f"  False Negatives: {self.false_negatives}\n"
            f"\n"
            f"  Detected seeds: {self.detected_seeds}\n"
            f"  Missed seeds: {self.missed_seeds}\n"
            f"\n"
            f"  T003 overlap: {self.t003_overlap['overlap_pct']:.1f}% of T003 violations detected"
        )


class FaccEvaluator:
    """
    Evaluates facc anomaly detection.

    Parameters
    ----------
    db_path_or_conn : str or duckdb.DuckDBPyConnection
        Path to DuckDB database or existing connection.
    seed_reaches : dict, optional
        Known corrupted reaches. Default uses SEED_REACHES.
    """

    def __init__(
        self,
        db_path_or_conn: Union[str, duckdb.DuckDBPyConnection],
        seed_reaches: Optional[Dict[int, Dict]] = None
    ):
        if isinstance(db_path_or_conn, str):
            self.conn = duckdb.connect(db_path_or_conn, read_only=True)
            self._own_conn = True
        else:
            self.conn = db_path_or_conn
            self._own_conn = False

        self.seed_reaches = seed_reaches or SEED_REACHES
        self.detector = FaccDetector(self.conn)

    def close(self):
        """Close connections if owned."""
        if self._own_conn and self.conn:
            self.detector.close()
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def evaluate(
        self,
        region: Optional[str] = None,
        anomaly_threshold: float = 0.5,
        expanded_positives: Optional[Set[int]] = None
    ) -> EvaluationResult:
        """
        Evaluate detection against known corrupted reaches.

        Parameters
        ----------
        region : str, optional
            Region to evaluate.
        anomaly_threshold : float
            Anomaly score threshold for detection.
        expanded_positives : set of int, optional
            Additional known corrupted reaches beyond seeds.

        Returns
        -------
        EvaluationResult
            Evaluation metrics and details.
        """
        # Get known positives
        known_positives = set(self.seed_reaches.keys())
        if expanded_positives:
            known_positives |= expanded_positives

        # Run detection
        result = self.detector.detect(
            region=region,
            anomaly_threshold=anomaly_threshold
        )
        detected = set(result.anomalies['reach_id'].tolist())

        # Compute metrics
        true_positives = known_positives & detected
        false_positives = detected - known_positives
        false_negatives = known_positives - detected

        tp = len(true_positives)
        fp = len(false_positives)
        fn = len(false_negatives)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # T003 comparison
        t003_overlap = self.detector.validate_against_t003(region=region)

        return EvaluationResult(
            precision=precision,
            recall=recall,
            f1=f1,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            detected_seeds=list(true_positives),
            missed_seeds=list(false_negatives),
            false_positive_ids=list(false_positives)[:50],  # Limit for display
            t003_overlap=t003_overlap,
        )

    def profile_seeds(self) -> pd.DataFrame:
        """
        Profile seed reaches to understand what makes them anomalous.

        Returns
        -------
        pd.DataFrame
            Features for all seed reaches.
        """
        seed_ids = list(self.seed_reaches.keys())
        return get_seed_reach_features(self.conn, seed_ids)

    def find_similar_reaches(
        self,
        region: Optional[str] = None,
        similarity_threshold: float = 0.8,
        max_candidates: int = 100
    ) -> pd.DataFrame:
        """
        Find reaches with similar feature profiles to seeds.

        Used for expanding the training set via manual review.

        Parameters
        ----------
        region : str, optional
            Region to search.
        similarity_threshold : float
            Minimum similarity to seeds (0-1).
        max_candidates : int
            Maximum candidates to return.

        Returns
        -------
        pd.DataFrame
            Candidate reaches for manual review.
        """
        # Get seed profiles
        seed_profiles = self.profile_seeds()
        if len(seed_profiles) == 0:
            return pd.DataFrame()

        # Define feature ranges from seeds
        feature_cols = ['facc_width_ratio', 'facc_reach_acc_ratio', 'facc_jump_ratio']
        available_cols = [c for c in feature_cols if c in seed_profiles.columns]

        if not available_cols:
            return pd.DataFrame()

        # Get candidates from detector
        result = self.detector.detect(
            region=region,
            anomaly_threshold=0.3,  # Lower threshold to get more candidates
            return_features=True
        )

        if len(result.anomalies) == 0:
            return pd.DataFrame()

        candidates = result.anomalies.copy()

        # Exclude known seeds
        candidates = candidates[
            ~candidates['reach_id'].isin(self.seed_reaches.keys())
        ]

        # Score similarity to seeds
        for col in available_cols:
            if col in candidates.columns and col in seed_profiles.columns:
                seed_range = (
                    seed_profiles[col].min(),
                    seed_profiles[col].max()
                )
                if seed_range[1] > seed_range[0]:
                    # Check if value is in range
                    candidates[f'{col}_in_range'] = (
                        (candidates[col] >= seed_range[0] * 0.5) &
                        (candidates[col] <= seed_range[1] * 2.0)
                    ).astype(int)

        # Compute overall similarity
        range_cols = [c for c in candidates.columns if c.endswith('_in_range')]
        if range_cols:
            candidates['similarity'] = candidates[range_cols].mean(axis=1)
            candidates = candidates[candidates['similarity'] >= similarity_threshold]
            candidates = candidates.sort_values('similarity', ascending=False)

        return candidates.head(max_candidates)

    def suggest_new_labels(
        self,
        region: Optional[str] = None,
        n_candidates: int = 20
    ) -> pd.DataFrame:
        """
        Suggest reaches for manual labeling to expand training set.

        Prioritizes reaches that:
        1. Have high anomaly scores
        2. Are similar to known seeds
        3. Are near (up/downstream of) known seeds

        Parameters
        ----------
        region : str, optional
            Region to search.
        n_candidates : int
            Number of candidates to suggest.

        Returns
        -------
        pd.DataFrame
            Candidates for manual review with reasons.
        """
        suggestions = []

        # High scorers not in seeds
        result = self.detector.detect(
            region=region,
            anomaly_threshold=0.5,
            return_features=True
        )

        if len(result.anomalies) > 0:
            high_scorers = result.anomalies[
                ~result.anomalies['reach_id'].isin(self.seed_reaches.keys())
            ].head(n_candidates // 2)
            high_scorers = high_scorers.copy()
            high_scorers['suggestion_reason'] = 'high_anomaly_score'
            suggestions.append(high_scorers)

        # Downstream of seeds (propagation candidates)
        propagation = self.detector.detect_propagation(
            region=region,
            seed_reach_ids=list(self.seed_reaches.keys())
        )
        if len(propagation) > 0:
            propagation = propagation.head(n_candidates // 2)
            propagation = propagation.copy()
            propagation['suggestion_reason'] = 'downstream_of_seed'
            suggestions.append(propagation)

        if suggestions:
            result = pd.concat(suggestions, ignore_index=True)
            return result.drop_duplicates(subset='reach_id').head(n_candidates)

        return pd.DataFrame()

    def threshold_sweep(
        self,
        region: Optional[str] = None,
        thresholds: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Sweep detection thresholds to find optimal operating point.

        Parameters
        ----------
        region : str, optional
            Region to evaluate.
        thresholds : list of float, optional
            Thresholds to test. Default: [0.1, 0.2, ..., 0.9]

        Returns
        -------
        pd.DataFrame
            Metrics at each threshold.
        """
        if thresholds is None:
            thresholds = [0.1 * i for i in range(1, 10)]

        results = []
        for thresh in thresholds:
            eval_result = self.evaluate(
                region=region,
                anomaly_threshold=thresh
            )
            results.append({
                'threshold': thresh,
                'precision': eval_result.precision,
                'recall': eval_result.recall,
                'f1': eval_result.f1,
                'true_positives': eval_result.true_positives,
                'false_positives': eval_result.false_positives,
                'false_negatives': eval_result.false_negatives,
            })

        return pd.DataFrame(results)


def evaluate_detection(
    db_path_or_conn: Union[str, duckdb.DuckDBPyConnection],
    region: Optional[str] = None,
    anomaly_threshold: float = 0.5,
    seed_reaches: Optional[Dict[int, Dict]] = None
) -> EvaluationResult:
    """
    Convenience function to evaluate facc detection.

    Parameters
    ----------
    db_path_or_conn : str or duckdb.DuckDBPyConnection
        Path to DuckDB database or existing connection.
    region : str, optional
        Region to evaluate.
    anomaly_threshold : float
        Anomaly score threshold.
    seed_reaches : dict, optional
        Known corrupted reaches.

    Returns
    -------
    EvaluationResult
        Evaluation metrics.

    Examples
    --------
    >>> from updates.sword_duckdb.facc_detection import evaluate_detection
    >>> result = evaluate_detection("sword_v17c.duckdb", region="NA")
    >>> print(result.summary())
    """
    with FaccEvaluator(db_path_or_conn, seed_reaches=seed_reaches) as evaluator:
        return evaluator.evaluate(region=region, anomaly_threshold=anomaly_threshold)
