# -*- coding: utf-8 -*-
"""
RF Classifier with RFE for FACC Anomaly Detection
==================================================

Trains Random Forest classifier with Recursive Feature Elimination (RFE)
to identify facc anomalies and rank feature importance.

Key design decisions:
- Class imbalance: class_weight='balanced' (simple, no data manipulation)
- Missing SWOT: Impute with 0 + has_swot_obs binary indicator
- CV strategy: Stratified by region for proportional representation
- Feature selection: RFECV for automatic optimal feature count

Usage:
    from facc_detection.rf_classifier import RFClassifier

    clf = RFClassifier()
    clf.fit(X_train, y_train)
    importance = clf.get_feature_importance()
    predictions = clf.predict(X_test)

CLI:
    python -m src.sword_duckdb.facc_detection.rf_classifier \\
        --features output/facc_detection/rf_features.parquet \\
        --labels output/facc_detection/all_anomalies.geojson \\
        --output output/facc_detection/
"""

from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
import joblib


# Known false positives from detect.py - exclude from training
KNOWN_FALSE_POSITIVES = [
    31239000161,
    31239000251,
    31231000181,  # Ob River multi-channel
    28160700191,
    45585500221,
    28106300011,
    28105000371,  # Narrow width
    45630500041,
    44570000065,  # Complex tidal/delta
    17211100904,  # Nile delta
    17291500221,
    17291500351,  # AF moderate FWR
]


class RFClassifier:
    """
    Random Forest classifier for facc anomaly detection with RFE feature selection.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    min_samples_leaf : int
        Minimum samples required at leaf node.
    max_depth : int, optional
        Maximum tree depth. None for unlimited.
    random_state : int
        Random seed for reproducibility.
    use_rfe : bool
        If True, use RFECV for feature selection.
    cv_folds : int
        Number of cross-validation folds for RFECV.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        min_samples_leaf: int = 5,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        use_rfe: bool = True,
        cv_folds: int = 5,
    ):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state
        self.use_rfe = use_rfe
        self.cv_folds = cv_folds

        # Initialize classifier with class_weight='balanced' for imbalance
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

        self.rfe = None
        self.feature_names: List[str] = []
        self.selected_features: List[str] = []
        self.feature_ranking: Dict[str, int] = {}

    def fit(
        self, X: pd.DataFrame, y: pd.Series, feature_names: Optional[List[str]] = None
    ) -> "RFClassifier":
        """
        Fit the classifier.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix.
        y : pd.Series or np.ndarray
            Target labels (0=clean, 1=anomaly).
        feature_names : list of str, optional
            Feature names. If None, uses X.columns.

        Returns
        -------
        self
        """
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y

        if self.use_rfe:
            # Use RFECV for automatic feature selection
            print(f"Running RFECV with {self.cv_folds}-fold CV...")
            self.rfe = RFECV(
                estimator=self.clf,
                step=1,
                cv=StratifiedKFold(
                    self.cv_folds, shuffle=True, random_state=self.random_state
                ),
                scoring="f1",
                min_features_to_select=10,
                n_jobs=-1,
            )
            self.rfe.fit(X_arr, y_arr)

            # Store rankings
            for name, rank in zip(self.feature_names, self.rfe.ranking_):
                self.feature_ranking[name] = int(rank)

            # Get selected features
            self.selected_features = [
                name for name, rank in self.feature_ranking.items() if rank == 1
            ]

            print(f"RFECV selected {len(self.selected_features)} features")
            print(
                f"Optimal CV F1 score: {self.rfe.cv_results_['mean_test_score'].max():.4f}"
            )
        else:
            # Train without RFE
            self.clf.fit(X_arr, y_arr)
            self.selected_features = self.feature_names.copy()

            # Store importance-based ranking
            importances = self.clf.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            for rank, idx in enumerate(sorted_idx, start=1):
                self.feature_ranking[self.feature_names[idx]] = rank

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict labels for X."""
        X_arr = X.values if isinstance(X, pd.DataFrame) else X

        if self.use_rfe and self.rfe is not None:
            return self.rfe.predict(X_arr)
        else:
            return self.clf.predict(X_arr)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities for X."""
        X_arr = X.values if isinstance(X, pd.DataFrame) else X

        if self.use_rfe and self.rfe is not None:
            return self.rfe.predict_proba(X_arr)
        else:
            return self.clf.predict_proba(X_arr)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance rankings.

        Returns
        -------
        pd.DataFrame
            DataFrame with feature names, importance scores, and RFE rankings.
        """
        if self.use_rfe and self.rfe is not None:
            # Get importance from the final fitted estimator
            importances = self.rfe.estimator_.feature_importances_
            # Map back to selected features
            selected_mask = self.rfe.support_
            importance_dict = {}
            imp_idx = 0
            for i, name in enumerate(self.feature_names):
                if selected_mask[i]:
                    importance_dict[name] = importances[imp_idx]
                    imp_idx += 1
                else:
                    importance_dict[name] = 0.0
        else:
            importance_dict = dict(
                zip(self.feature_names, self.clf.feature_importances_)
            )

        df = pd.DataFrame(
            [
                {
                    "feature": name,
                    "importance": importance_dict.get(name, 0.0),
                    "rfe_rank": self.feature_ranking.get(name, 999),
                    "selected": name in self.selected_features,
                }
                for name in self.feature_names
            ]
        )

        return df.sort_values("rfe_rank")

    def evaluate(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Evaluate classifier on test set.

        Parameters
        ----------
        X_test : pd.DataFrame or np.ndarray
            Test features.
        y_test : pd.Series or np.ndarray
            Test labels.

        Returns
        -------
        dict
            Evaluation metrics including precision, recall, F1, confusion matrix.
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]

        y_true = y_test.values if isinstance(y_test, pd.Series) else y_test

        # Compute metrics
        metrics = {
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(
                y_true, y_pred, output_dict=True
            ),
            "n_test": len(y_true),
            "n_positives": int(y_true.sum()),
            "n_predicted_positives": int(y_pred.sum()),
        }

        # Precision-recall curve
        precision_curve, recall_curve, thresholds = precision_recall_curve(
            y_true, y_proba
        )
        metrics["pr_auc"] = auc(recall_curve, precision_curve)

        return metrics

    def save(self, path: str):
        """Save classifier to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "clf": self.clf,
            "rfe": self.rfe,
            "feature_names": self.feature_names,
            "selected_features": self.selected_features,
            "feature_ranking": self.feature_ranking,
            "params": {
                "n_estimators": self.n_estimators,
                "min_samples_leaf": self.min_samples_leaf,
                "max_depth": self.max_depth,
                "random_state": self.random_state,
                "use_rfe": self.use_rfe,
                "cv_folds": self.cv_folds,
            },
        }
        joblib.dump(save_dict, path)
        print(f"Saved model to {path}")

    @classmethod
    def load(cls, path: str) -> "RFClassifier":
        """Load classifier from file."""
        save_dict = joblib.load(path)

        instance = cls(**save_dict["params"])
        instance.clf = save_dict["clf"]
        instance.rfe = save_dict["rfe"]
        instance.feature_names = save_dict["feature_names"]
        instance.selected_features = save_dict["selected_features"]
        instance.feature_ranking = save_dict["feature_ranking"]

        return instance


def prepare_training_data(
    features_df: pd.DataFrame,
    anomaly_reach_ids: List[int],
    exclude_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_by_region: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Prepare training and test data from features DataFrame.

    Parameters
    ----------
    features_df : pd.DataFrame
        Full feature set from rf_features.extract_rf_features().
    anomaly_reach_ids : list of int
        Reach IDs labeled as anomalies.
    exclude_cols : list of str, optional
        Columns to exclude from features (e.g., 'reach_id', 'region').
    test_size : float
        Fraction of data for testing.
    random_state : int
        Random seed.
    stratify_by_region : bool
        If True, stratify split by region.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names
    """
    # Default exclude columns
    if exclude_cols is None:
        exclude_cols = ["reach_id", "region", "geometry", "is_anomaly", "geom"]

    # Create labels
    df = features_df.copy()
    df["is_anomaly"] = df["reach_id"].isin(anomaly_reach_ids).astype(int)

    # Remove false positives from positive class
    fp_mask = df["reach_id"].isin(KNOWN_FALSE_POSITIVES)
    df.loc[fp_mask, "is_anomaly"] = 0

    print(f"Positives (after FP removal): {df['is_anomaly'].sum()}")
    print(f"Negatives: {(~df['is_anomaly'].astype(bool)).sum()}")

    # Select feature columns
    feature_cols = [
        c for c in df.columns if c not in exclude_cols and c not in ["is_anomaly"]
    ]

    # Handle missing values
    X = df[feature_cols].copy()

    # Fill remaining NaN with 0 (after imputation in rf_features)
    X = X.fillna(0)

    # Replace infinities with large finite values
    X = X.replace([np.inf, -np.inf], 0)

    y = df["is_anomaly"]

    # Stratify by region if requested
    if stratify_by_region and "region" in df.columns:
        # Create combined stratification key
        df["strat_key"] = df["region"].astype(str) + "_" + df["is_anomaly"].astype(str)
        stratify = df["strat_key"]
    else:
        stratify = y

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    return X_train, X_test, y_train, y_test, feature_cols


def train_rf_classifier(
    features_path: str,
    labels_path: str,
    output_dir: str,
    test_size: float = 0.2,
    use_rfe: bool = True,
    n_estimators: int = 100,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train RF classifier end-to-end.

    Parameters
    ----------
    features_path : str
        Path to rf_features.parquet.
    labels_path : str
        Path to all_anomalies.geojson.
    output_dir : str
        Directory for outputs.
    test_size : float
        Test set fraction.
    use_rfe : bool
        Use RFECV for feature selection.
    n_estimators : int
        Number of RF trees.
    random_state : int
        Random seed.

    Returns
    -------
    dict
        Training results including metrics and file paths.
    """
    from .rf_features import load_anomaly_labels

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading features...")
    features_df = pd.read_parquet(features_path)
    print(f"Loaded {len(features_df)} reaches with {len(features_df.columns)} features")

    print("Loading labels...")
    labels_df = load_anomaly_labels(labels_path, false_positives=KNOWN_FALSE_POSITIVES)
    anomaly_ids = labels_df["reach_id"].tolist()
    print(f"Loaded {len(anomaly_ids)} anomaly labels (after FP removal)")

    # Prepare data
    print("Preparing training data...")
    X_train, X_test, y_train, y_test, feature_names = prepare_training_data(
        features_df, anomaly_ids, test_size=test_size, random_state=random_state
    )

    print(f"Training set: {len(X_train)} samples ({y_train.sum()} positives)")
    print(f"Test set: {len(X_test)} samples ({y_test.sum()} positives)")

    # Train classifier
    print("\nTraining Random Forest classifier...")
    clf = RFClassifier(
        n_estimators=n_estimators, use_rfe=use_rfe, random_state=random_state
    )
    clf.fit(X_train, y_train, feature_names=feature_names)

    # Evaluate
    print("\nEvaluating on test set...")
    metrics = clf.evaluate(X_test, y_test)

    print("\nTest Set Results:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"  PR AUC: {metrics['pr_auc']:.4f}")
    print("\nConfusion Matrix:")
    cm = np.array(metrics["confusion_matrix"])
    print(f"  TN={cm[0, 0]}, FP={cm[0, 1]}")
    print(f"  FN={cm[1, 0]}, TP={cm[1, 1]}")

    # Get feature importance
    importance_df = clf.get_feature_importance()

    # Save outputs
    model_path = output_dir / "rf_model.joblib"
    clf.save(model_path)

    importance_path = output_dir / "rf_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"\nSaved feature importance to {importance_path}")

    metrics_path = output_dir / "rf_metrics.json"
    with open(metrics_path, "w") as f:
        # Convert numpy types for JSON serialization
        json_metrics = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in metrics.items()
        }
        json.dump(json_metrics, f, indent=2, default=str)
    print(f"Saved metrics to {metrics_path}")

    # Generate predictions on full dataset
    print("\nGenerating predictions on full dataset...")
    X_full = features_df[[c for c in feature_names if c in features_df.columns]].fillna(
        0
    )
    X_full = X_full.replace([np.inf, -np.inf], 0)

    predictions = clf.predict(X_full)
    probabilities = clf.predict_proba(X_full)[:, 1]

    predictions_df = features_df[["reach_id", "region"]].copy()
    predictions_df["rf_prediction"] = predictions
    predictions_df["rf_probability"] = probabilities
    predictions_df["is_rule_anomaly"] = (
        predictions_df["reach_id"].isin(anomaly_ids).astype(int)
    )

    predictions_path = output_dir / "rf_predictions.parquet"
    predictions_df.to_parquet(predictions_path, index=False)
    print(f"Saved predictions to {predictions_path}")

    # Summary
    results = {
        "model_path": str(model_path),
        "importance_path": str(importance_path),
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "metrics": metrics,
        "n_features": len(feature_names),
        "n_selected_features": len(clf.selected_features),
        "selected_features": clf.selected_features,
        "top_10_features": importance_df.head(10)["feature"].tolist(),
        "training_summary": {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_positives_train": int(y_train.sum()),
            "n_positives_test": int(y_test.sum()),
        },
    }

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train RF classifier for facc anomaly detection"
    )
    parser.add_argument("--features", required=True, help="Path to rf_features.parquet")
    parser.add_argument("--labels", required=True, help="Path to all_anomalies.geojson")
    parser.add_argument(
        "--output", default="output/facc_detection/", help="Output directory"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test set fraction"
    )
    parser.add_argument(
        "--no-rfe", action="store_true", help="Disable RFE feature selection"
    )
    parser.add_argument(
        "--n-estimators", type=int, default=100, help="Number of RF trees"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    results = train_rf_classifier(
        features_path=args.features,
        labels_path=args.labels,
        output_dir=args.output,
        test_size=args.test_size,
        use_rfe=not args.no_rfe,
        n_estimators=args.n_estimators,
        random_state=args.seed,
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {results['model_path']}")
    print(
        f"Features selected: {results['n_selected_features']}/{results['n_features']}"
    )
    print(f"Test F1 Score: {results['metrics']['f1']:.4f}")
    print("\nTop 10 Features:")
    for i, feat in enumerate(results["top_10_features"], 1):
        print(f"  {i}. {feat}")
