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
from typing import Dict, Optional, Tuple, List, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib


class FaccRegressor:
    """RF regressor to predict correct facc values."""

    # Features to exclude from training (target or derived from target)
    # Default: exclude basic facc-derived features but keep multi-hop for context
    EXCLUDE_FEATURES_DEFAULT = [
        'reach_id', 'region', 'facc', 'log_facc',
        'facc_width_ratio', 'facc_per_reach', 'facc_jump_ratio',
        'max_upstream_facc', 'max_downstream_facc', 'upstream_facc_sum',
        'fwr_drop_ratio', 'fwr_ratio_to_median', 'max_upstream_fwr',
        'max_downstream_fwr', 'upstream_fwr_ratio', 'median_fwr', 'avg_fwr',
        'ratio_to_median', 'fwr_upstream_consistency', 'fwr_network_zscore',
        'network_avg_fwr', 'network_std_fwr', 'side_channel_high_fwr',
        'median_fpr',  # facc per reach median
    ]

    # Extended exclusion: ALL facc-derived features for "no-facc" model
    # Use this to train model that predicts facc from topology/width only
    EXCLUDE_FEATURES_NO_FACC = EXCLUDE_FEATURES_DEFAULT + [
        # 2-hop facc features
        'max_2hop_upstream_facc', 'max_2hop_downstream_facc',
        'facc_2hop_ratio', 'facc_chain_direction',

        # 3-hop and 5-hop facc (if present)
        'max_3hop_upstream_facc', 'max_3hop_downstream_facc',
        'max_5hop_upstream_facc', 'max_5hop_downstream_facc',
        'facc_3hop_ratio', 'facc_5hop_ratio',

        # All FWR features (facc/width derived)
        'fwr_network_ratio', 'fwr_per_path_freq', 'fwr_downstream_consistency',
        'fwr_propagation', 'avg_upstream_fwr', 'avg_downstream_fwr',
        'avg_2hop_upstream_fwr', 'avg_2hop_downstream_fwr',
        'avg_3hop_upstream_fwr', 'avg_3hop_downstream_fwr',
        'avg_5hop_upstream_fwr', 'avg_5hop_downstream_fwr',

        # Binary FWR indicators
        'extreme_fwr_1000', 'extreme_fwr_2000', 'extreme_fwr_5000',
        'low_pf_high_facc', 'invalid_pf_high_fwr', 'side_channel_high_fwr',

        # Cross-hop ratios (facc-derived)
        'hop2_hop5_ratio', 'hop3_hop5_ratio', 'dn_hop5_hop2_ratio',
    ]

    # Default to standard exclusion (for backward compatibility)
    EXCLUDE_FEATURES = EXCLUDE_FEATURES_DEFAULT

    def __init__(self, n_estimators: int = 100, max_depth: int = 20,
                 min_samples_leaf: int = 10, random_state: int = 42,
                 exclude_facc_features: bool = False):
        """
        Initialize FaccRegressor.

        Parameters
        ----------
        n_estimators : int
            Number of trees in the forest.
        max_depth : int
            Maximum depth of trees.
        min_samples_leaf : int
            Minimum samples per leaf.
        random_state : int
            Random seed.
        exclude_facc_features : bool
            If True, exclude ALL facc-derived features (2-hop, FWR, etc).
            Use this to train a "no-facc" model that predicts from topology/width only.
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_names = None
        self.metrics = {}
        self.exclude_facc_features = exclude_facc_features
        self._exclude_list = (
            self.EXCLUDE_FEATURES_NO_FACC if exclude_facc_features
            else self.EXCLUDE_FEATURES_DEFAULT
        )

    def prepare_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target."""
        # Get feature columns using instance exclusion list
        feature_cols = [c for c in df.columns if c not in self._exclude_list]

        X = df[feature_cols].copy()
        y = df['facc'].copy()

        # Handle missing values and infinities
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Store feature names
        self.feature_names = list(X.columns)

        return X, y

    def fit(self, df: pd.DataFrame, test_size: float = 0.2) -> dict:
        """Train the regressor."""
        mode = "NO-FACC (topology/width only)" if self.exclude_facc_features else "standard"
        print(f"Preparing features from {len(df):,} clean reaches... [mode: {mode}]")
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
            'metrics': self.metrics,
            'exclude_facc_features': self.exclude_facc_features,
        }, path)
        print(f"Saved to {path}")

    @classmethod
    def load(cls, path: Path) -> 'FaccRegressor':
        """Load saved model."""
        data = joblib.load(path)
        exclude_facc = data.get('exclude_facc_features', False)
        regressor = cls(exclude_facc_features=exclude_facc)
        regressor.model = data['model']
        regressor.feature_names = data['feature_names']
        regressor.metrics = data['metrics']
        return regressor


class SplitFaccRegressor:
    """
    RF regressor with separate models per main_side × lakeflag group.

    Trains up to 9 models (3 main_side × 3 lakeflag groups).
    Groups with < min_samples fall back to main channel model.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth_main: int = 20,
        max_depth_other: int = 15,
        min_samples_leaf: int = 10,
        min_samples_per_group: int = 100,
        random_state: int = 42,
        exclude_facc_features: bool = False
    ):
        """
        Initialize SplitFaccRegressor.

        Parameters
        ----------
        exclude_facc_features : bool
            If True, exclude ALL facc-derived features (2-hop, FWR, etc).
            Use this to train a "no-facc" model that predicts from topology/width only.
        """
        self.n_estimators = n_estimators
        self.max_depth_main = max_depth_main
        self.max_depth_other = max_depth_other
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_per_group = min_samples_per_group
        self.random_state = random_state
        self.exclude_facc_features = exclude_facc_features
        self._exclude_list = (
            FaccRegressor.EXCLUDE_FEATURES_NO_FACC if exclude_facc_features
            else FaccRegressor.EXCLUDE_FEATURES_DEFAULT
        )

        self.models: Dict[Tuple[int, int], RandomForestRegressor] = {}
        self.feature_names: Optional[List[str]] = None
        self.group_metrics: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.overall_metrics: Dict[str, Any] = {}
        self.fallback_key: Optional[Tuple[int, int]] = None

    def _get_lakeflag_group(self, lakeflag: np.ndarray) -> np.ndarray:
        """Map lakeflag to 3 groups: 0=river, 1=lake, 2=other (canal+tidal)."""
        return np.where(lakeflag <= 1, lakeflag, 2)

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target."""
        feature_cols = [c for c in df.columns if c not in self._exclude_list]
        X = df[feature_cols].copy()
        y = df['facc'].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        self.feature_names = list(X.columns)
        return X, y

    def fit(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train separate models per main_side × lakeflag group.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with features, facc, main_side, lakeflag columns.
        test_size : float
            Fraction for test split.

        Returns
        -------
        dict
            Overall and per-group metrics.
        """
        mode = "NO-FACC (topology/width only)" if self.exclude_facc_features else "standard"
        print(f"Preparing features from {len(df):,} clean reaches... [mode: {mode}]")
        X, y = self._prepare_features(df)
        y_log = np.log1p(y)

        main_side = df['main_side'].values
        lakeflag_group = self._get_lakeflag_group(df['lakeflag'].values)

        # Train/test split (stratified by group if possible)
        X_train, X_test, y_train, y_test, ms_train, ms_test, lf_train, lf_test = train_test_split(
            X, y_log, main_side, lakeflag_group,
            test_size=test_size, random_state=self.random_state
        )

        print(f"Training on {len(X_train):,} samples, testing on {len(X_test):,}...")
        print(f"Features: {len(self.feature_names)}")

        # Train models per group
        group_counts = {}
        for ms in [0, 1, 2]:
            for lf in [0, 1, 2]:
                mask_train = (ms_train == ms) & (lf_train == lf)
                n_samples = mask_train.sum()
                group_counts[(ms, lf)] = n_samples

                if n_samples < self.min_samples_per_group:
                    print(f"  Group ({ms}, {lf}): {n_samples} samples - skipping (< {self.min_samples_per_group})")
                    continue

                # Use shallower trees for smaller groups
                max_depth = self.max_depth_main if ms == 0 else self.max_depth_other

                model = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state,
                    n_jobs=-1
                )

                model.fit(X_train[mask_train], y_train[mask_train])
                self.models[(ms, lf)] = model

                # Evaluate on group test set
                mask_test = (ms_test == ms) & (lf_test == lf)
                if mask_test.sum() > 0:
                    y_pred_log = model.predict(X_test[mask_test])
                    y_pred = np.expm1(y_pred_log)
                    y_true = np.expm1(y_test[mask_test])

                    self.group_metrics[(ms, lf)] = {
                        'n_train': n_samples,
                        'n_test': mask_test.sum(),
                        'r2_log': r2_score(y_test[mask_test], y_pred_log),
                        'r2': r2_score(y_true, y_pred),
                        'mae': mean_absolute_error(y_true, y_pred),
                        'median_ae': np.median(np.abs(y_true - y_pred)),
                    }

                    # Percent errors
                    valid = y_true > 0
                    if valid.sum() > 0:
                        pct_err = np.abs(y_pred[valid] - y_true.values[valid]) / y_true.values[valid] * 100
                        self.group_metrics[(ms, lf)]['median_pct_error'] = np.median(pct_err)
                        self.group_metrics[(ms, lf)]['p90_pct_error'] = np.percentile(pct_err, 90)

                    print(f"  Group ({ms}, {lf}): n={n_samples:,}, R²={self.group_metrics[(ms, lf)]['r2']:.4f}, "
                          f"median_pct_err={self.group_metrics[(ms, lf)].get('median_pct_error', 0):.1f}%")

        # Set fallback key (prefer main river model)
        for fallback in [(0, 0), (0, 1), (0, 2)]:
            if fallback in self.models:
                self.fallback_key = fallback
                break

        if not self.models:
            raise ValueError("No models trained - not enough samples in any group")

        # Overall evaluation
        y_pred_all = self.predict(df.loc[X_test.index])['predicted_facc'].values
        y_test_orig = np.expm1(y_test)

        self.overall_metrics = {
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_models': len(self.models),
            'r2': r2_score(y_test_orig, y_pred_all),
            'mae': mean_absolute_error(y_test_orig, y_pred_all),
            'median_ae': np.median(np.abs(y_test_orig - y_pred_all)),
        }

        valid = y_test_orig > 0
        pct_errors = np.abs(y_pred_all[valid] - y_test_orig.values[valid]) / y_test_orig.values[valid] * 100
        self.overall_metrics['median_pct_error'] = np.median(pct_errors)
        self.overall_metrics['p90_pct_error'] = np.percentile(pct_errors, 90)

        print(f"\n{'='*60}")
        print("Overall Results:")
        print(f"  Models trained: {len(self.models)}")
        print(f"  R² (combined): {self.overall_metrics['r2']:.4f}")
        print(f"  MAE: {self.overall_metrics['mae']:,.0f} km²")
        print(f"  Median % error: {self.overall_metrics['median_pct_error']:.1f}%")
        print(f"  P90 % error: {self.overall_metrics['p90_pct_error']:.1f}%")

        return {
            'overall': self.overall_metrics,
            'by_group': self.group_metrics
        }

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict facc for given reaches using appropriate group model.

        Parameters
        ----------
        df : pd.DataFrame
            Features with main_side and lakeflag columns.

        Returns
        -------
        pd.DataFrame
            Predictions with reach_id, facc, predicted_facc, facc_ratio.
        """
        if not self.models:
            raise ValueError("No models trained - call fit() first")

        feature_cols = [c for c in self.feature_names if c in df.columns]
        X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        main_side = df['main_side'].values if 'main_side' in df.columns else np.zeros(len(df))
        lakeflag_group = self._get_lakeflag_group(
            df['lakeflag'].values if 'lakeflag' in df.columns else np.zeros(len(df))
        )

        y_pred = np.zeros(len(X))
        model_used = np.full(len(X), '', dtype=object)

        for i in range(len(X)):
            ms, lf = int(main_side[i]), int(lakeflag_group[i])
            key = (ms, lf)

            if key in self.models:
                model = self.models[key]
                model_used[i] = f"({ms},{lf})"
            else:
                model = self.models[self.fallback_key]
                model_used[i] = f"fallback_{self.fallback_key}"

            y_pred_log = model.predict(X.iloc[i:i+1])
            y_pred[i] = np.expm1(y_pred_log[0])

        result = df[['reach_id', 'facc']].copy()
        result['predicted_facc'] = y_pred
        result['facc_diff'] = result['predicted_facc'] - result['facc']
        result['facc_ratio'] = result['predicted_facc'] / result['facc'].replace(0, np.nan)
        result['model_used'] = model_used

        return result

    def get_feature_importance(self, group: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
        """Get feature importances for a specific group or combined."""
        if group is not None:
            if group not in self.models:
                raise ValueError(f"No model for group {group}")
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.models[group].feature_importances_
            })
        else:
            # Average across all models
            all_importances = np.zeros(len(self.feature_names))
            for model in self.models.values():
                all_importances += model.feature_importances_
            all_importances /= len(self.models)

            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': all_importances
            })

        return importance.sort_values('importance', ascending=False)

    def save(self, path: Path):
        """Save models and metadata."""
        joblib.dump({
            'models': self.models,
            'feature_names': self.feature_names,
            'overall_metrics': self.overall_metrics,
            'group_metrics': self.group_metrics,
            'fallback_key': self.fallback_key,
            'exclude_facc_features': self.exclude_facc_features,
        }, path)
        print(f"Saved {len(self.models)} models to {path}")

    @classmethod
    def load(cls, path: Path) -> 'SplitFaccRegressor':
        """Load saved models."""
        data = joblib.load(path)
        exclude_facc = data.get('exclude_facc_features', False)
        regressor = cls(exclude_facc_features=exclude_facc)
        regressor.models = data['models']
        regressor.feature_names = data['feature_names']
        regressor.overall_metrics = data['overall_metrics']
        regressor.group_metrics = data['group_metrics']
        regressor.fallback_key = data['fallback_key']
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
