# -*- coding: utf-8 -*-
"""
SWORD Facc Error Detection and Correction Module
=================================================

ML/regression-based detection and correction of corrupted flow accumulation (facc) values.

Two failure modes are addressed:
1. **Entry points**: Bad facc ENTERS tributary (ratio 200x-3000x jump)
2. **Propagation**: Inherited bad facc (ratio ~1.0, wrong absolute)

Core approach:
- Compare MERIT-derived facc with topology-based reach accumulation
- At bifurcations, D8 picks ONE downstream branch → other gets wrong facc
- Use feature engineering + regression for detection and correction

Detection Usage:
    from updates.sword_duckdb.facc_detection import FaccDetector

    detector = FaccDetector("sword_v17c.duckdb")
    result = detector.detect(region="NA", anomaly_threshold=0.5)
    print(result.summary())

Correction Usage (Phase 2):
    from updates.sword_duckdb.facc_detection import FaccCorrector

    with FaccCorrector("sword_v17c.duckdb") as corrector:
        # Detect anomalies
        detector = FaccDetector(corrector.conn)
        anomalies = detector.detect(region="NA").anomalies

        # Filter, classify, estimate, apply
        fixable, skipped = corrector.filter_fixable(anomalies)
        classified = corrector.classify_anomalies(fixable)
        corrections = corrector.estimate_corrections(classified)
        result = corrector.apply_corrections(corrections, dry_run=True)
        print(result.summary())

CLI Usage:
    # Detect anomalies
    python -m src.updates.sword_duckdb.facc_detection.cli --db sword_v17c.duckdb --region NA

    # Dry-run correction
    python -m src.updates.sword_duckdb.facc_detection.cli --db sword_v17c.duckdb --fix --region NA

    # Apply corrections
    python -m src.updates.sword_duckdb.facc_detection.cli --db sword_v17c.duckdb --fix --apply --region NA

    # Rollback
    python -m src.updates.sword_duckdb.facc_detection.cli --db sword_v17c.duckdb --rollback --batch-id 1
"""

from .reach_acc import compute_reach_accumulation, ReachAccumulator
from .features import extract_facc_features, FaccFeatureExtractor
from .detect import FaccDetector, detect_facc_anomalies, detect_hybrid
from .evaluate import evaluate_detection, FaccEvaluator
from .correct import FaccCorrector, correct_facc_anomalies, CorrectionResult

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
    "detect_hybrid",
    # Evaluation
    "evaluate_detection",
    "FaccEvaluator",
    # Correction (Phase 2)
    "FaccCorrector",
    "correct_facc_anomalies",
    "CorrectionResult",
]
