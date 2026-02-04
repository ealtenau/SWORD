#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Facc Anomaly Detection CLI
==========================

Command-line interface for facc anomaly detection.

Usage:
    python -m src.updates.sword_duckdb.facc_detection.cli --db sword_v17c.duckdb --region NA
    python -m src.updates.sword_duckdb.facc_detection.cli --db sword_v17c.duckdb --all --threshold 0.3
    python -m src.updates.sword_duckdb.facc_detection.cli --db sword_v17c.duckdb --evaluate
    python -m src.updates.sword_duckdb.facc_detection.cli --db sword_v17c.duckdb --profile-seeds
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from .detect import FaccDetector, DetectionConfig
from .evaluate import FaccEvaluator, SEED_REACHES


def main():
    parser = argparse.ArgumentParser(
        description="Facc Anomaly Detection for SWORD database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect anomalies in North America
  python -m src.updates.sword_duckdb.facc_detection.cli --db sword_v17c.duckdb --region NA

  # Detect with custom threshold
  python -m src.updates.sword_duckdb.facc_detection.cli --db sword_v17c.duckdb --all --threshold 0.3

  # Evaluate against known corrupted reaches
  python -m src.updates.sword_duckdb.facc_detection.cli --db sword_v17c.duckdb --evaluate

  # Profile seed reaches
  python -m src.updates.sword_duckdb.facc_detection.cli --db sword_v17c.duckdb --profile-seeds

  # Compare against T003 lint check
  python -m src.updates.sword_duckdb.facc_detection.cli --db sword_v17c.duckdb --compare-t003 --region NA
        """,
    )

    parser.add_argument(
        "--db", "-d",
        required=True,
        help="Path to SWORD DuckDB database",
    )
    parser.add_argument(
        "--region", "-r",
        help="Region to check (NA, SA, EU, AF, AS, OC)",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Check all regions",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Anomaly score threshold (default: 0.5)",
    )
    parser.add_argument(
        "--facc-width-threshold",
        type=float,
        default=5000.0,
        help="Facc/width ratio threshold (default: 5000)",
    )
    parser.add_argument(
        "--facc-jump-threshold",
        type=float,
        default=100.0,
        help="Facc jump ratio threshold (default: 100)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for results (CSV format)",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "csv", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--evaluate", "-e",
        action="store_true",
        help="Evaluate detection against known corrupted reaches",
    )
    parser.add_argument(
        "--profile-seeds",
        action="store_true",
        help="Profile known seed reaches",
    )
    parser.add_argument(
        "--compare-t003",
        action="store_true",
        help="Compare detection with T003 lint check",
    )
    parser.add_argument(
        "--entry-points",
        action="store_true",
        help="Detect entry point errors only",
    )
    parser.add_argument(
        "--propagation",
        action="store_true",
        help="Detect propagation errors from seeds",
    )
    parser.add_argument(
        "--suggest-labels",
        action="store_true",
        help="Suggest reaches for manual labeling",
    )
    parser.add_argument(
        "--threshold-sweep",
        action="store_true",
        help="Sweep thresholds to find optimal operating point",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.all and not args.region and not args.evaluate and not args.profile_seeds:
        print("Error: Must specify --region, --all, --evaluate, or --profile-seeds")
        sys.exit(1)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        sys.exit(1)

    # Create config
    config = DetectionConfig(
        facc_width_ratio_threshold=args.facc_width_threshold,
        facc_jump_ratio_threshold=args.facc_jump_threshold,
    )

    # Run appropriate mode
    if args.profile_seeds:
        run_profile_seeds(str(db_path), args)
    elif args.evaluate:
        run_evaluate(str(db_path), args)
    elif args.compare_t003:
        run_compare_t003(str(db_path), args)
    elif args.threshold_sweep:
        run_threshold_sweep(str(db_path), args)
    elif args.suggest_labels:
        run_suggest_labels(str(db_path), args)
    elif args.entry_points:
        run_entry_points(str(db_path), args, config)
    elif args.propagation:
        run_propagation(str(db_path), args, config)
    else:
        run_detect(str(db_path), args, config)


def run_detect(db_path: str, args, config: DetectionConfig):
    """Run anomaly detection."""
    regions = ['NA', 'SA', 'EU', 'AF', 'AS', 'OC'] if args.all else [args.region]

    all_anomalies = []

    with FaccDetector(db_path, config=config) as detector:
        for region in regions:
            print(f"\nDetecting anomalies in {region}...")
            result = detector.detect(
                region=region,
                anomaly_threshold=args.threshold,
            )

            print(result.summary())

            if len(result.anomalies) > 0:
                result.anomalies['detection_region'] = region
                all_anomalies.append(result.anomalies)

    if all_anomalies:
        combined = pd.concat(all_anomalies, ignore_index=True)

        if args.output:
            if args.format == "csv":
                combined.to_csv(args.output, index=False)
            elif args.format == "json":
                combined.to_json(args.output, orient="records", indent=2)
            else:
                combined.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")
        elif args.verbose:
            print("\nTop 20 anomalies:")
            print(combined.head(20).to_string())


def run_entry_points(db_path: str, args, config: DetectionConfig):
    """Detect entry point errors."""
    regions = ['NA', 'SA', 'EU', 'AF', 'AS', 'OC'] if args.all else [args.region]

    with FaccDetector(db_path, config=config) as detector:
        for region in regions:
            print(f"\nDetecting entry points in {region}...")
            entry_points = detector.detect_entry_points(
                region=region,
                min_jump_ratio=args.facc_jump_threshold,
            )

            print(f"Found {len(entry_points)} entry points")

            if len(entry_points) > 0 and args.verbose:
                print("\nTop 10 entry points:")
                print(entry_points.head(10).to_string())


def run_propagation(db_path: str, args, config: DetectionConfig):
    """Detect propagation errors."""
    seed_ids = list(SEED_REACHES.keys())

    with FaccDetector(db_path, config=config) as detector:
        region = args.region if not args.all else None
        print(f"\nDetecting propagation from {len(seed_ids)} seeds...")
        propagation = detector.detect_propagation(
            region=region,
            seed_reach_ids=seed_ids,
        )

        print(f"Found {len(propagation)} reaches with inherited bad facc")

        if len(propagation) > 0 and args.verbose:
            print("\nTop 20 propagation candidates:")
            print(propagation.head(20).to_string())


def run_evaluate(db_path: str, args):
    """Evaluate detection against known corrupted reaches."""
    with FaccEvaluator(db_path) as evaluator:
        region = args.region if args.region else None
        result = evaluator.evaluate(
            region=region,
            anomaly_threshold=args.threshold,
        )
        print(result.summary())


def run_profile_seeds(db_path: str, args):
    """Profile known seed reaches."""
    with FaccEvaluator(db_path) as evaluator:
        profile = evaluator.profile_seeds()

        if len(profile) == 0:
            print("No seed reaches found in database")
            return

        print("\nSeed Reach Profiles:")
        print("=" * 80)

        for _, row in profile.iterrows():
            print(f"\nReach {row['reach_id']} ({row.get('corruption_mode', 'unknown')}):")
            print(f"  facc: {row.get('facc', 'N/A'):,.0f}")
            print(f"  width: {row.get('width', 'N/A'):,.1f}")
            print(f"  facc/width: {row.get('facc_width_ratio', 'N/A'):,.1f}")
            print(f"  facc_reach_acc_ratio: {row.get('facc_reach_acc_ratio', 'N/A'):.2f}")
            print(f"  facc_jump_ratio: {row.get('facc_jump_ratio', 'N/A')}")
            print(f"  stream_order: {row.get('stream_order', 'N/A')}")


def run_compare_t003(db_path: str, args):
    """Compare detection with T003 lint check."""
    with FaccDetector(db_path) as detector:
        region = args.region if args.region else None
        comparison = detector.validate_against_t003(region=region)

        print("\nComparison with T003 (facc monotonicity):")
        print("=" * 50)
        print(f"T003 violations: {comparison['t003_violations']:,}")
        print(f"Our detections: {comparison['our_detections']:,}")
        print(f"Overlap: {comparison['overlap']:,} ({comparison['overlap_pct']:.1f}%)")
        print(f"T003-only (we missed): {comparison['t003_only']:,}")
        print(f"ML-only (we found extra): {comparison['ml_only']:,}")

        if args.verbose:
            print(f"\nSample T003-only IDs: {comparison['t003_only_ids']}")
            print(f"Sample ML-only IDs: {comparison['ml_only_ids']}")


def run_threshold_sweep(db_path: str, args):
    """Sweep thresholds to find optimal operating point."""
    with FaccEvaluator(db_path) as evaluator:
        region = args.region if args.region else None
        results = evaluator.threshold_sweep(region=region)

        print("\nThreshold Sweep Results:")
        print("=" * 60)
        print(results.to_string(index=False))

        # Find optimal F1
        best_idx = results['f1'].idxmax()
        best = results.loc[best_idx]
        print(f"\nBest F1 at threshold={best['threshold']:.1f}: {best['f1']:.2%}")


def run_suggest_labels(db_path: str, args):
    """Suggest reaches for manual labeling."""
    with FaccEvaluator(db_path) as evaluator:
        region = args.region if args.region else None
        suggestions = evaluator.suggest_new_labels(region=region, n_candidates=20)

        if len(suggestions) == 0:
            print("No suggestions found")
            return

        print("\nSuggested reaches for manual labeling:")
        print("=" * 60)
        print(suggestions.to_string())


if __name__ == "__main__":
    main()
