# -*- coding: utf-8 -*-
"""
RF Regressor for FACC Correction
================================

Train RF to predict what facc SHOULD be based on topology and width features.
Train on clean reaches, apply to anomalies.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib


class FaccRegressor:
    """RF regressor to predict correct facc values."""

    # Features to exclude from training (target or derived from target)
    EXCLUDE_FEATURES = [
        'reach_id', 'region', 'facc', 'log_facc',
        'facc_width_ratio', 'facc_per_reach', 'facc_jump_ratio',
        'max_upstream_facc', 'max_downstream_facc', 'upstream_facc_sum',
        'fwr_drop_ratio', 'fwr_ratio_to_median', 'max_upstream_fwr',
        'max_downstream_fwr', 'upstream_fwr_ratio', 'median_fwr', 'avg_fwr',
        'ratio_to_median', 'fwr_upstream_consistency', 'fwr_network_zscore',
        'network_avg_fwr', 'network_std_fwr', 'side_channel_high_fwr',
        'median_fpr'  # facc per reach median
    ]

    def __init__(self, n_estimators: int = 100, max_depth: int = 20,
                 min_samples_leaf: int = 10, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_names = None
        self.metrics = {}

    def prepare_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target."""
        # Get feature columns
        feature_cols = [c for c in df.columns if c not in self.EXCLUDE_FEATURES]

        X = df[feature_cols].copy()
        y = df['facc'].copy()

        # Handle missing values and infinities
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Store feature names
        self.feature_names = list(X.columns)

        return X, y

    def fit(self, df: pd.DataFrame, test_size: float = 0.2) -> dict:
        """Train the regressor."""
        print(f"Preparing features from {len(df):,} clean reaches...")
        X, y = self.prepare_features(df)

        # Log transform target (facc spans many orders of magnitude)
        y_log = np.log1p(y)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_log, test_size=test_size, random_state=42
        )

        print(f"Training on {len(X_train):,} samples, testing on {len(X_test):,}...")
        print(f"Features: {len(self.feature_names)}")

        # Fit
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred_log = self.model.predict(X_test)

        # Back-transform for interpretable metrics
        y_pred = np.expm1(y_pred_log)
        y_test_orig = np.expm1(y_test)

        self.metrics = {
            'r2_log': r2_score(y_test, y_pred_log),
            'r2': r2_score(y_test_orig, y_pred),
            'mae': mean_absolute_error(y_test_orig, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred)),
            'median_ae': np.median(np.abs(y_test_orig - y_pred)),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': len(self.feature_names)
        }

        # Percent error metrics
        mask = y_test_orig > 0
        pct_errors = np.abs(y_pred[mask] - y_test_orig.values[mask]) / y_test_orig.values[mask] * 100
        self.metrics['median_pct_error'] = np.median(pct_errors)
        self.metrics['p90_pct_error'] = np.percentile(pct_errors, 90)

        print(f"\nResults:")
        print(f"  R² (log space): {self.metrics['r2_log']:.4f}")
        print(f"  R² (original):  {self.metrics['r2']:.4f}")
        print(f"  MAE:           {self.metrics['mae']:,.0f} km²")
        print(f"  Median AE:     {self.metrics['median_ae']:,.0f} km²")
        print(f"  Median % err:  {self.metrics['median_pct_error']:.1f}%")
        print(f"  P90 % err:     {self.metrics['p90_pct_error']:.1f}%")

        return self.metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importances ranked."""
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        return importance

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict facc for given reaches."""
        feature_cols = [c for c in self.feature_names if c in df.columns]
        X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        # Predict in log space, back-transform
        y_pred_log = self.model.predict(X)
        y_pred = np.expm1(y_pred_log)

        result = df[['reach_id', 'facc']].copy()
        result['predicted_facc'] = y_pred
        result['facc_diff'] = result['predicted_facc'] - result['facc']
        result['facc_ratio'] = result['predicted_facc'] / result['facc'].replace(0, np.nan)

        return result

    def save(self, path: Path):
        """Save model and metadata."""
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }, path)
        print(f"Saved to {path}")

    @classmethod
    def load(cls, path: Path) -> 'FaccRegressor':
        """Load saved model."""
        data = joblib.load(path)
        regressor = cls()
        regressor.model = data['model']
        regressor.feature_names = data['feature_names']
        regressor.metrics = data['metrics']
        return regressor


def main():
    """Train facc regressor on clean reaches."""
    output_dir = Path('output/facc_detection')

    # Load features
    print("Loading features...")
    features_path = output_dir / 'rf_features.csv'
    if not features_path.exists():
        features_path = output_dir / 'rf_features_v2.csv'

    df = pd.read_csv(features_path)
    print(f"Loaded {len(df):,} reaches, {len(df.columns)} columns")

    # Load anomaly labels to exclude
    import geopandas as gpd
    anomalies_path = output_dir / 'all_anomalies.geojson'
    anomalies = gpd.read_file(anomalies_path)
    anomaly_ids = set(anomalies['reach_id'].values)
    print(f"Excluding {len(anomaly_ids):,} anomalous reaches")

    # Filter to clean reaches
    clean = df[~df['reach_id'].isin(anomaly_ids)].copy()
    print(f"Training on {len(clean):,} clean reaches")

    # Train
    regressor = FaccRegressor(n_estimators=100, max_depth=20)
    metrics = regressor.fit(clean)

    # Feature importance
    importance = regressor.get_feature_importance()
    print("\nTop 20 features:")
    print(importance.head(20).to_string(index=False))

    # Save
    regressor.save(output_dir / 'rf_regressor.joblib')
    importance.to_csv(output_dir / 'rf_regressor_importance.csv', index=False)

    # Apply to anomalies
    print("\n" + "="*60)
    print("Applying to anomalous reaches...")

    anomaly_features = df[df['reach_id'].isin(anomaly_ids)].copy()
    predictions = regressor.predict(anomaly_features)

    print(f"\nCorrection summary for {len(predictions):,} anomalies:")
    print(f"  Median current facc: {predictions['facc'].median():,.0f} km²")
    print(f"  Median predicted:    {predictions['predicted_facc'].median():,.0f} km²")
    print(f"  Median ratio:        {predictions['facc_ratio'].median():.2f}x")

    # Save predictions
    predictions.to_csv(output_dir / 'rf_regressor_predictions.csv', index=False)
    print(f"\nSaved predictions to {output_dir / 'rf_regressor_predictions.csv'}")

    return regressor, predictions


if __name__ == '__main__':
    main()
