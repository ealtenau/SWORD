# -*- coding: utf-8 -*-
"""
Train Split Facc Regressor
==========================

Training script for the improved SplitFaccRegressor:
1. Loads features (with new multi-hop and regional encoding)
2. Trains separate models per main_side × lakeflag group
3. Compares against baseline single-model regressor
4. Exports comparison metrics and predictions

Usage:
    python -m src.updates.sword_duckdb.facc_detection.train_split_regressor \
        --db data/duckdb/sword_v17c.duckdb \
        --output-dir output/facc_detection
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd

from .rf_features import RFFeatureExtractor
from .rf_regressor import FaccRegressor, SplitFaccRegressor


def extract_features_with_enhancements(db_path: str, output_path: Path) -> pd.DataFrame:
    """
    Extract features with new multi-hop and regional encoding.

    Parameters
    ----------
    db_path : str
        Path to DuckDB database.
    output_path : Path
        Path to save features CSV.

    Returns
    -------
    pd.DataFrame
        Complete feature set.
    """
    print("Extracting features with multi-hop and regional encoding...")

    with RFFeatureExtractor(db_path) as extractor:
        features = extractor.extract_all(region=None)

    # Save for reuse
    features.to_csv(output_path, index=False)
    print(f"Saved {len(features):,} rows with {len(features.columns)} columns to {output_path}")

    return features


def train_and_compare(
    features: pd.DataFrame,
    anomalies_path: Path,
    output_dir: Path,
    skip_baseline: bool = False,
    exclude_facc_features: bool = False
) -> dict:
    """
    Train split regressor and compare against baseline.

    Parameters
    ----------
    features : pd.DataFrame
        Full feature set.
    anomalies_path : Path
        Path to all_anomalies.geojson.
    output_dir : Path
        Output directory for models and results.
    skip_baseline : bool
        If True, skip training baseline for comparison.
    exclude_facc_features : bool
        If True, exclude ALL facc-derived features (2-hop, FWR, etc).
        Trains a "no-facc" model that predicts from topology/width only.

    Returns
    -------
    dict
        Comparison metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if exclude_facc_features:
        print("\n" + "="*60)
        print("MODE: NO-FACC FEATURES")
        print("Training RF to predict facc using ONLY topology/width features")
        print("="*60 + "\n")

    # Load anomaly IDs
    anomalies = gpd.read_file(anomalies_path)
    anomaly_ids = set(anomalies['reach_id'].values)
    print(f"Excluding {len(anomaly_ids):,} anomalous reaches from training")

    # Filter to clean reaches
    clean = features[~features['reach_id'].isin(anomaly_ids)].copy()
    print(f"Training on {len(clean):,} clean reaches")

    results = {}

    # Determine output suffix for no-facc mode
    suffix = "_nofacc" if exclude_facc_features else ""

    # Train baseline (single model)
    if not skip_baseline:
        print("\n" + "="*60)
        print("BASELINE: Single RF Model")
        print("="*60)

        baseline = FaccRegressor(
            n_estimators=100,
            max_depth=20,
            exclude_facc_features=exclude_facc_features
        )
        baseline_metrics = baseline.fit(clean)
        baseline.save(output_dir / f'rf_regressor_baseline{suffix}.joblib')

        # Feature importance
        baseline_importance = baseline.get_feature_importance()
        baseline_importance.to_csv(output_dir / f'rf_regressor_baseline{suffix}_importance.csv', index=False)

        results['baseline'] = baseline_metrics

    # Train split model
    print("\n" + "="*60)
    print("SPLIT MODEL: Per main_side × lakeflag")
    print("="*60)

    split_model = SplitFaccRegressor(
        n_estimators=100,
        max_depth_main=20,
        max_depth_other=15,
        min_samples_per_group=100,
        exclude_facc_features=exclude_facc_features
    )
    split_metrics = split_model.fit(clean)
    split_model.save(output_dir / f'rf_split_regressor{suffix}.joblib')

    # Feature importance (averaged across models)
    split_importance = split_model.get_feature_importance()
    split_importance.to_csv(output_dir / f'rf_split_regressor{suffix}_importance.csv', index=False)

    results['split'] = split_metrics

    # Apply to anomalies
    print("\n" + "="*60)
    print("Applying to anomalous reaches...")
    print("="*60)

    anomaly_features = features[features['reach_id'].isin(anomaly_ids)].copy()

    if not skip_baseline:
        baseline_preds = baseline.predict(anomaly_features)
        baseline_preds.to_csv(output_dir / f'rf_baseline_predictions{suffix}.csv', index=False)

    split_preds = split_model.predict(anomaly_features)
    split_preds.to_csv(output_dir / f'rf_split_predictions{suffix}.csv', index=False)

    # Comparison
    print("\nCorrection summary:")
    print(f"  Anomalies: {len(split_preds):,}")
    print(f"  Median current facc: {split_preds['facc'].median():,.0f} km²")
    print(f"  Median split predicted: {split_preds['predicted_facc'].median():,.0f} km²")
    print(f"  Median ratio: {split_preds['facc_ratio'].median():.2f}x")

    if not skip_baseline:
        print(f"  Median baseline predicted: {baseline_preds['predicted_facc'].median():,.0f} km²")
        print(f"  Median baseline ratio: {baseline_preds['facc_ratio'].median():.2f}x")

    # Per-group analysis for anomalies
    print("\nPredictions by model used:")
    model_counts = split_preds['model_used'].value_counts()
    for model, count in model_counts.items():
        subset = split_preds[split_preds['model_used'] == model]
        print(f"  {model}: n={count}, median_ratio={subset['facc_ratio'].median():.2f}x")

    # Save comparison summary
    comparison = {
        'split_overall_r2': results['split']['overall']['r2'],
        'split_median_pct_error': results['split']['overall']['median_pct_error'],
        'split_n_models': len(split_model.models),
        'exclude_facc_features': exclude_facc_features,
        'n_features_used': len(split_model.feature_names),
    }

    if not skip_baseline:
        comparison.update({
            'baseline_r2': results['baseline']['r2'],
            'baseline_median_pct_error': results['baseline']['median_pct_error'],
            'r2_improvement': results['split']['overall']['r2'] - results['baseline']['r2'],
            'error_reduction': results['baseline']['median_pct_error'] - results['split']['overall']['median_pct_error'],
        })

    import json
    with open(output_dir / f'rf_split_comparison{suffix}.json', 'w') as f:
        json.dump(comparison, f, indent=2)

    results['comparison'] = comparison
    return results


def main():
    parser = argparse.ArgumentParser(description='Train split facc regressor')
    parser.add_argument('--db', required=True, help='Path to DuckDB database')
    parser.add_argument('--output-dir', default='output/facc_detection', help='Output directory')
    parser.add_argument('--features', help='Path to existing features CSV (skip extraction)')
    parser.add_argument('--anomalies', help='Path to anomalies GeoJSON')
    parser.add_argument('--skip-baseline', action='store_true', help='Skip baseline training')
    parser.add_argument(
        '--exclude-facc-features', action='store_true',
        help='Exclude ALL facc-derived features (2-hop, FWR, etc). '
             'Train a "no-facc" model that predicts from topology/width only.'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or extract features
    if args.features and Path(args.features).exists():
        print(f"Loading features from {args.features}")
        features = pd.read_csv(args.features)
    else:
        features_path = output_dir / 'rf_features_v3.csv'
        features = extract_features_with_enhancements(args.db, features_path)

    # Anomalies path
    anomalies_path = Path(args.anomalies) if args.anomalies else output_dir / 'all_anomalies.geojson'
    if not anomalies_path.exists():
        print(f"ERROR: Anomalies file not found: {anomalies_path}")
        print("Run detect_hybrid() first to generate anomalies.")
        return

    # Train and compare
    results = train_and_compare(
        features=features,
        anomalies_path=anomalies_path,
        output_dir=output_dir,
        skip_baseline=args.skip_baseline,
        exclude_facc_features=args.exclude_facc_features
    )

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if args.exclude_facc_features:
        print("MODE: NO-FACC FEATURES (topology/width only)")
        print(f"Features used: {results['comparison']['n_features_used']}")

    if 'baseline' in results:
        print(f"Baseline R²: {results['baseline']['r2']:.4f}")
        print(f"Baseline median % error: {results['baseline']['median_pct_error']:.1f}%")

    print(f"Split model R²: {results['split']['overall']['r2']:.4f}")
    print(f"Split model median % error: {results['split']['overall']['median_pct_error']:.1f}%")
    print(f"Split model trained: {len(results['split']['by_group'])} group models")

    if 'comparison' in results and 'r2_improvement' in results['comparison']:
        r2_imp = results['comparison']['r2_improvement']
        err_red = results['comparison']['error_reduction']
        print(f"\nImprovement: R² +{r2_imp:.4f}, Error -{err_red:.1f}%")


if __name__ == '__main__':
    main()
