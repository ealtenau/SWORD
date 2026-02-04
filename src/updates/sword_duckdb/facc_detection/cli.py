#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Facc Anomaly Detection and Correction CLI
==========================================

Command-line interface for facc anomaly detection and correction.

Usage:
    # Detection
    python -m src.updates.sword_duckdb.facc_detection.cli --db sword_v17c.duckdb --region NA
    python -m src.updates.sword_duckdb.facc_detection.cli --db sword_v17c.duckdb --all --threshold 0.3
    python -m src.updates.sword_duckdb.facc_detection.cli --db sword_v17c.duckdb --evaluate
    python -m src.updates.sword_duckdb.facc_detection.cli --db sword_v17c.duckdb --profile-seeds

    # Correction (Phase 2)
    python -m src.updates.sword_duckdb.facc_detection.cli --db sword_v17c.duckdb --fix --dry-run
    python -m src.updates.sword_duckdb.facc_detection.cli --db sword_v17c.duckdb --fix --region NA
    python -m src.updates.sword_duckdb.facc_detection.cli --db sword_v17c.duckdb --rollback --batch-id 1
"""

import argparse
import sys
import logging
from pathlib import Path

import pandas as pd

from .detect import FaccDetector, DetectionConfig, detect_hybrid, export_categorized_geojsons
from .evaluate import FaccEvaluator, SEED_REACHES
from .correct import FaccCorrector, correct_facc_anomalies

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


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

    # Phase 2: Correction arguments
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Detect and correct facc anomalies",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Don't actually apply corrections (default: True)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply corrections (overrides --dry-run)",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback corrections from a batch",
    )
    parser.add_argument(
        "--batch-id",
        type=int,
        help="Batch ID for rollback",
    )
    parser.add_argument(
        "--show-batches",
        action="store_true",
        help="Show history of correction batches",
    )
    parser.add_argument(
        "--verify-seeds",
        action="store_true",
        help="Verify that seed reaches would be fixed correctly",
    )
    parser.add_argument(
        "--use-basic-detector",
        action="store_true",
        help="Use basic detector instead of hybrid (hybrid is default, catches 5/5 seeds)",
    )
    parser.add_argument(
        "--include-lakes",
        action="store_true",
        help="Include lakes in correction (normally skipped, but some have bad facc)",
    )
    parser.add_argument(
        "--merit-path",
        help="Path to MERIT Hydro base directory (enables guided MERIT search for corrections)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    # GeoJSON export arguments
    parser.add_argument(
        "--export-geojson",
        action="store_true",
        help="Export detection results as categorized GeoJSON files for QGIS review",
    )
    parser.add_argument(
        "--output-dir",
        default="output/facc_detection",
        help="Output directory for GeoJSON files (default: output/facc_detection)",
    )

    args = parser.parse_args()

    # Validate arguments
    needs_region = not (
        args.evaluate or args.profile_seeds or args.rollback or
        args.show_batches or args.fix or args.verify_seeds or args.export_geojson
    )
    if needs_region and not args.all and not args.region:
        print("Error: Must specify --region, --all, or a specific mode")
        sys.exit(1)

    if args.rollback and not args.batch_id:
        print("Error: --rollback requires --batch-id")
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
    if args.rollback:
        run_rollback(str(db_path), args)
    elif args.show_batches:
        run_show_batches(str(db_path), args)
    elif args.fix:
        run_fix(str(db_path), args, config)
    elif args.verify_seeds:
        run_verify_seeds(str(db_path), args, config)
    elif args.export_geojson:
        run_export_geojson(str(db_path), args)
    elif args.profile_seeds:
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


def run_export_geojson(db_path: str, args):
    """Export detection results as categorized GeoJSON files for QGIS review."""
    regions = ['NA', 'SA', 'EU', 'AF', 'AS', 'OC'] if args.all else ([args.region] if args.region else ['NA', 'SA', 'EU', 'AF', 'AS', 'OC'])
    output_dir = Path(args.output_dir)

    print(f"\nRunning facc detection with GeoJSON export...")
    print(f"  Database: {db_path}")
    print(f"  Regions: {regions}")
    print(f"  Output directory: {output_dir}")
    print()

    # Collect all anomalies from all regions
    all_anomalies = []

    for region in regions:
        print(f"Detecting anomalies in {region}...")
        result = detect_hybrid(db_path, region=region)

        print(f"  {region}: {len(result.anomalies)} anomalies detected")

        if len(result.anomalies) > 0:
            all_anomalies.append(result.anomalies)

    if not all_anomalies:
        print("\nNo anomalies detected in any region.")
        return

    combined = pd.concat(all_anomalies, ignore_index=True)
    print(f"\nTotal anomalies: {len(combined)}")

    # Get seed reach IDs for verification
    seed_reach_ids = list(SEED_REACHES.keys())

    # Export to GeoJSON
    print(f"\nExporting categorized GeoJSON files to {output_dir}...")
    summary = export_categorized_geojsons(
        db_path,
        combined,
        output_dir,
        seed_reach_ids=seed_reach_ids,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Export Summary")
    print("=" * 60)
    print(f"Total anomalies: {summary['total_anomalies']}")
    print("\nBy detection rule:")
    for rule, count in summary['by_rule'].items():
        print(f"  {rule}: {count}")

    print("\nFiles created:")
    for name, filepath in summary['files'].items():
        print(f"  {name}: {filepath}")

    # Seed verification
    if seed_reach_ids:
        print(f"\nSeed verification ({len(seed_reach_ids)} seeds):")
        print(f"  Detected: {len(summary['seeds_detected'])}/{len(seed_reach_ids)}")
        if summary['seeds_missed']:
            print(f"  MISSED: {summary['seeds_missed']}")
        else:
            print("  All seeds detected!")

    print("\n" + "=" * 60)
    print("Open the GeoJSON files in QGIS for visual review.")
    print("=" * 60)


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


# ============================================================================
# Phase 2: Correction functions
# ============================================================================

def run_fix(db_path: str, args, config: DetectionConfig):
    """Detect and correct facc anomalies."""
    regions = ['NA', 'SA', 'EU', 'AF', 'AS', 'OC'] if args.all else ([args.region] if args.region else None)

    # Determine if dry run
    dry_run = not args.apply

    # Determine detection method
    use_hybrid = not getattr(args, 'use_basic_detector', False)

    if dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN MODE - No changes will be made")
        print("Use --apply to actually apply corrections")
        print("=" * 60)

    with FaccCorrector(db_path, read_only=dry_run) as corrector:
        # Detect anomalies
        detection_method = "HYBRID (ratio_to_median)" if use_hybrid else "BASIC (threshold)"
        print(f"\nStep 1: Detecting anomalies using {detection_method}...")

        if use_hybrid:
            # Use hybrid detection - catches 5/5 seeds, 0/24 FPs
            if regions:
                all_anomalies = []
                for region in regions:
                    result = corrector.detect_hybrid(region=region)
                    print(f"  {region}: {len(result.anomalies)} anomalies")
                    if len(result.anomalies) > 0:
                        all_anomalies.append(result.anomalies)

                if all_anomalies:
                    anomalies = pd.concat(all_anomalies, ignore_index=True)
                else:
                    anomalies = pd.DataFrame()
            else:
                result = corrector.detect_hybrid()
                anomalies = result.anomalies
        else:
            # Use basic detector with threshold
            detector = FaccDetector(corrector.conn)
            if regions:
                all_anomalies = []
                for region in regions:
                    result = detector.detect(region=region, anomaly_threshold=args.threshold)
                    print(f"  {region}: {len(result.anomalies)} anomalies")
                    if len(result.anomalies) > 0:
                        all_anomalies.append(result.anomalies)

                if all_anomalies:
                    anomalies = pd.concat(all_anomalies, ignore_index=True)
                else:
                    anomalies = pd.DataFrame()
            else:
                result = detector.detect(anomaly_threshold=args.threshold)
                anomalies = result.anomalies

        if len(anomalies) == 0:
            print("No anomalies detected")
            return

        print(f"\nTotal anomalies detected: {len(anomalies)}")

        # Filter fixable
        include_lakes = getattr(args, 'include_lakes', False)
        print(f"\nStep 2: Filtering fixable anomalies (include_lakes={include_lakes})...")
        fixable, skipped = corrector.filter_fixable(anomalies, include_lakes=include_lakes)

        if len(fixable) == 0:
            print("No fixable anomalies found")
            print(f"Skipped: {len(skipped)} (lakes, missing topology)")
            return

        print(f"  Fixable: {len(fixable)}")
        print(f"  Skipped: {len(skipped)}")

        # Classify
        print("\nStep 3: Classifying anomalies...")
        classified = corrector.classify_anomalies(fixable)

        type_counts = classified['fix_type'].value_counts()
        for fix_type, count in type_counts.items():
            print(f"  {fix_type}: {count}")

        # Fit models and estimate corrections
        print("\nStep 4: Fitting regression models...")
        for region in classified['region'].unique():
            models = corrector.fit_regression(region)
            if 'primary' in models:
                print(f"  {region} primary: R²={models['primary'].r_squared:.3f}")
            if 'fallback' in models:
                print(f"  {region} fallback: R²={models['fallback'].r_squared:.3f}")

        # Get MERIT path
        merit_path = getattr(args, 'merit_path', None)
        if merit_path:
            print(f"\nStep 5: Estimating corrections with MERIT guided search...")
            print(f"  MERIT path: {merit_path}")
        else:
            print("\nStep 5: Estimating corrections (regression only)...")

        corrections = corrector.estimate_corrections(classified, merit_path=merit_path)

        if len(corrections) == 0:
            print("No corrections generated")
            return

        # Show sample
        print(f"\nGenerated {len(corrections)} corrections")
        print("\nSample corrections (top 10 by reduction):")
        sample = corrections.nlargest(10, 'reduction_factor')
        for _, row in sample.iterrows():
            print(
                f"  {row['reach_id']}: {row['old_facc']:,.0f} → {row['facc_corrected']:,.0f} "
                f"({row['reduction_factor']:.1f}x reduction, {row['fix_type']})"
            )

        # Validate
        print("\nStep 6: Validating corrections...")
        validation = corrector.validate_corrections(corrections)

        if validation['valid']:
            print("  ✓ All validation checks passed")
        else:
            print("  ⚠ Validation issues:")
            for issue in validation['issues']:
                print(f"    - {issue}")

        # Apply or show summary
        print("\nStep 7: Applying corrections...")
        result = corrector.apply_corrections(corrections, dry_run=dry_run)
        print(result.summary())

        # Save preview to file if requested
        if args.output:
            corrections.to_csv(args.output, index=False)
            print(f"\nCorrections saved to: {args.output}")


def run_rollback(db_path: str, args):
    """Rollback corrections from a batch."""
    batch_id = args.batch_id

    print(f"\nRolling back batch {batch_id}...")

    with FaccCorrector(db_path, read_only=False) as corrector:
        reverted = corrector.rollback(batch_id)

        if reverted > 0:
            print(f"✓ Rolled back {reverted} changes from batch {batch_id}")
        else:
            print(f"No changes found for batch {batch_id}")


def run_show_batches(db_path: str, args):
    """Show history of correction batches."""
    with FaccCorrector(db_path, read_only=True) as corrector:
        history = corrector.get_batch_history()

        if len(history) == 0:
            print("No correction batches found")
            return

        print("\nCorrection Batch History:")
        print("=" * 80)
        print(history.to_string(index=False))


def run_verify_seeds(db_path: str, args, config: DetectionConfig):
    """Verify that seed reaches would be fixed correctly."""
    use_hybrid = not getattr(args, 'use_basic_detector', False)
    merit_path = getattr(args, 'merit_path', None)
    detection_method = "HYBRID" if use_hybrid else "BASIC"

    print(f"\nVerifying seed reach corrections using {detection_method} detection (dry run)...")
    if merit_path:
        print(f"  MERIT guided search enabled: {merit_path}")
    print()

    with FaccCorrector(db_path, read_only=True) as corrector:
        # Get seed reach IDs
        seed_ids = list(SEED_REACHES.keys())

        # Detect using hybrid (default) or basic
        if use_hybrid:
            # Seeds are in SA region
            result = corrector.detect_hybrid(region='SA')
        else:
            detector = FaccDetector(corrector.conn)
            result = detector.detect(anomaly_threshold=args.threshold)

        # Check if seeds are in detected anomalies
        detected_ids = set(result.anomalies['reach_id'].tolist())
        seeds_detected = [s for s in seed_ids if s in detected_ids]
        seeds_missed = [s for s in seed_ids if s not in detected_ids]

        print(f"Seeds detected: {len(seeds_detected)}/{len(seed_ids)}")
        if seeds_missed:
            print(f"  MISSED: {seeds_missed}")

        # Filter to just seeds for correction preview
        seed_anomalies = result.anomalies[result.anomalies['reach_id'].isin(seed_ids)]

        if len(seed_anomalies) == 0:
            print("\nNo seed anomalies to correct")
            return

        # Filter and classify
        fixable, skipped = corrector.filter_fixable(seed_anomalies)
        if len(fixable) == 0:
            print("\nNo fixable seed anomalies")
            return

        classified = corrector.classify_anomalies(fixable)

        # Estimate corrections with optional MERIT search
        corrections = corrector.estimate_corrections(classified, merit_path=merit_path)

        # Show results
        print("\nSeed Reach Corrections Preview:")
        print("=" * 110)
        print(f"{'reach_id':<15} {'mode':<12} {'old_facc':>15} {'new_facc':>15} {'reduction':>10} {'fix_type':<15} {'source':<15}")
        print("-" * 110)

        for _, row in corrections.iterrows():
            seed_info = SEED_REACHES.get(row['reach_id'], {})
            mode = seed_info.get('mode', 'unknown')
            reduction = row['old_facc'] / row['facc_corrected'] if row['facc_corrected'] > 0 else 0
            source = row.get('model_used', 'unknown')

            print(
                f"{row['reach_id']:<15} {mode:<12} {row['old_facc']:>15,.0f} "
                f"{row['facc_corrected']:>15,.0f} {reduction:>10.1f}x {row['fix_type']:<15} {source:<15}"
            )

        # Validate
        validation = corrector.validate_corrections(corrections)
        print("\n" + ("✓ Validation passed" if validation['valid'] else "⚠ Validation issues"))


if __name__ == "__main__":
    main()
