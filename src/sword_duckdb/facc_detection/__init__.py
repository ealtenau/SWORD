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
    from sword_duckdb.facc_detection import FaccDetector

    detector = FaccDetector("sword_v17c.duckdb")
    result = detector.detect(region="NA", anomaly_threshold=0.5)
    print(result.summary())

Correction Usage (Phase 2):
    from sword_duckdb.facc_detection import FaccCorrector

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

RF Classifier Usage:
    from sword_duckdb.facc_detection import RFFeatureExtractor, RFClassifier

    # Extract features
    extractor = RFFeatureExtractor("sword_v17c.duckdb")
    features = extractor.extract_all()

    # Train classifier
    clf = RFClassifier(use_rfe=True)
    clf.fit(X_train, y_train)
    importance = clf.get_feature_importance()

CLI Usage:
    # Detect anomalies
    python -m src.sword_duckdb.facc_detection.cli --db sword_v17c.duckdb --region NA

    # Dry-run correction
    python -m src.sword_duckdb.facc_detection.cli --db sword_v17c.duckdb --fix --region NA

    # Apply corrections
    python -m src.sword_duckdb.facc_detection.cli --db sword_v17c.duckdb --fix --apply --region NA

    # Rollback
    python -m src.sword_duckdb.facc_detection.cli --db sword_v17c.duckdb --rollback --batch-id 1

    # RF training
    python -m src.sword_duckdb.facc_detection.rf_classifier \\
        --features output/facc_detection/rf_features.parquet \\
        --labels output/facc_detection/all_anomalies.geojson \\
        --output output/facc_detection/

    # RF evaluation
    python -m src.sword_duckdb.facc_detection.rf_evaluate \\
        --predictions output/facc_detection/rf_predictions.parquet \\
        --model output/facc_detection/rf_model.joblib \\
        --output output/facc_detection/
"""

from .reach_acc import compute_reach_accumulation, ReachAccumulator
from .features import extract_facc_features, FaccFeatureExtractor
from .detect import FaccDetector, detect_facc_anomalies, detect_hybrid
from .evaluate import evaluate_detection, FaccEvaluator
from .correct import FaccCorrector, correct_facc_anomalies, CorrectionResult


# Lazy import — merit_search requires GDAL which may not be installed
def __getattr__(name):
    if name in ("MeritGuidedSearch", "create_merit_search"):
        from .merit_search import MeritGuidedSearch, create_merit_search

        globals()["MeritGuidedSearch"] = MeritGuidedSearch
        globals()["create_merit_search"] = create_merit_search
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from .rf_features import RFFeatureExtractor, extract_rf_features, load_anomaly_labels
from .rf_classifier import RFClassifier, train_rf_classifier
from .rf_evaluate import RFEvaluator, evaluate_rf_classifier
from .rf_regressor import FaccRegressor

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
    # MERIT guided search
    "MeritGuidedSearch",
    "create_merit_search",
    # RF Classifier
    "RFFeatureExtractor",
    "extract_rf_features",
    "load_anomaly_labels",
    "RFClassifier",
    "train_rf_classifier",
    "RFEvaluator",
    "evaluate_rf_classifier",
    # RF Regressor
    "FaccRegressor",
]
