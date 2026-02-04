# -*- coding: utf-8 -*-
"""
SWORD Facc Error Detection Module
=================================

ML/regression-based detection of corrupted flow accumulation (facc) values.

Two failure modes are addressed:
1. **Entry points**: Bad facc ENTERS tributary (ratio 200x-3000x jump)
2. **Propagation**: Inherited bad facc (ratio ~1.0, wrong absolute)

Core approach:
- Compare MERIT-derived facc with topology-based reach accumulation
- At bifurcations, D8 picks ONE downstream branch → other gets wrong facc
- Use feature engineering + optional Random Forest for detection

Usage:
    from updates.sword_duckdb.facc_detection import (
        FaccDetector,
        compute_reach_accumulation,
        extract_facc_features,
    )

    # Basic usage
    detector = FaccDetector("sword_v17c.duckdb")
    anomalies = detector.detect(region="NA")

    # With custom threshold
    anomalies = detector.detect(region="NA", anomaly_threshold=3.0)

    # Evaluate against known corrupted reaches
    metrics = detector.evaluate(seed_reach_ids=[64231000301, 62236100011])
"""

from .reach_acc import compute_reach_accumulation, ReachAccumulator
from .features import extract_facc_features, FaccFeatureExtractor
from .detect import FaccDetector, detect_facc_anomalies
from .evaluate import evaluate_detection, FaccEvaluator

__all__ = [
    # Reach accumulation
    "compute_reach_accumulation",
    "ReachAccumulator",
    # Feature extraction
    "extract_facc_features",
    "FaccFeatureExtractor",
    # Detection
    "FaccDetector",
    "detect_facc_anomalies",
    # Evaluation
    "evaluate_detection",
    "FaccEvaluator",
]
