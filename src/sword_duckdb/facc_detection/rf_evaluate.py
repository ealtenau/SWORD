# -*- coding: utf-8 -*-
"""
RF Evaluation and Analysis for FACC Anomaly Detection
======================================================

Evaluates RF classifier predictions against rule-based detection,
analyzes feature importance, and generates comprehensive analysis report.

Key analyses:
1. Compare RF predictions to rule-based detection
2. Analyze 5 missed seeds - can RF catch them?
3. Feature importance ranking and interpretation
4. Identify novel anomalies RF finds that rules miss

Usage:
    from facc_detection.rf_evaluate import RFEvaluator

    evaluator = RFEvaluator(
        predictions_path="output/facc_detection/rf_predictions.parquet",
        features_path="output/facc_detection/rf_features.parquet",
        model_path="output/facc_detection/rf_model.joblib"
    )
    report = evaluator.generate_report()

CLI:
    python -m src.sword_duckdb.facc_detection.rf_evaluate \\
        --predictions output/facc_detection/rf_predictions.parquet \\
        --features output/facc_detection/rf_features.parquet \\
        --model output/facc_detection/rf_model.joblib \\
        --output output/facc_detection/
"""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd

from .evaluate import SEED_REACHES


# The 5 seeds missed by rule-based detection (from detection_summary.json)
MISSED_SEEDS = [
    22513000171,  # EU
    44581100665,  # AS
    44581100675,  # AS
    34211700241,  # AS - propagation from 34211700251
    34211101775,  # AS - side channel with mainstem facc
]


class RFEvaluator:
    """
    Evaluates RF classifier for facc anomaly detection.

    Parameters
    ----------
    predictions_path : str
        Path to rf_predictions.parquet.
    features_path : str, optional
        Path to rf_features.parquet for feature analysis.
    model_path : str, optional
        Path to rf_model.joblib for feature importance.
    """

    def __init__(
        self,
        predictions_path: str,
        features_path: Optional[str] = None,
        model_path: Optional[str] = None
    ):
        self.predictions_df = pd.read_parquet(predictions_path)

        if features_path:
            self.features_df = pd.read_parquet(features_path)
        else:
            self.features_df = None

        self.model = None
        if model_path:
            from .rf_classifier import RFClassifier
            self.model = RFClassifier.load(model_path)

    def compare_to_rules(self) -> Dict[str, Any]:
        """
        Compare RF predictions to rule-based detection.

        Returns
        -------
        dict
            Comparison metrics: overlap, RF-only, rules-only, agreement rate.
        """
        df = self.predictions_df.copy()

        rf_positives = set(df[df['rf_prediction'] == 1]['reach_id'].tolist())
        rule_positives = set(df[df['is_rule_anomaly'] == 1]['reach_id'].tolist())

        overlap = rf_positives & rule_positives
        rf_only = rf_positives - rule_positives
        rules_only = rule_positives - rf_positives

        return {
            'n_rf_positives': len(rf_positives),
            'n_rule_positives': len(rule_positives),
            'n_overlap': len(overlap),
            'n_rf_only': len(rf_only),
            'n_rules_only': len(rules_only),
            'overlap_pct_of_rf': 100 * len(overlap) / len(rf_positives) if rf_positives else 0,
            'overlap_pct_of_rules': 100 * len(overlap) / len(rule_positives) if rule_positives else 0,
            'agreement_rate': 100 * (len(overlap) + len(df) - len(rf_positives.union(rule_positives))) / len(df),
            'rf_only_ids': list(rf_only)[:20],  # Sample
            'rules_only_ids': list(rules_only)[:20],  # Sample
        }

    def analyze_missed_seeds(self) -> pd.DataFrame:
        """
        Analyze the 5 seeds missed by rule-based detection.

        Returns
        -------
        pd.DataFrame
            Details for each missed seed including RF prediction.
        """
        df = self.predictions_df.copy()

        missed_df = df[df['reach_id'].isin(MISSED_SEEDS)].copy()

        if len(missed_df) == 0:
            print("Warning: No missed seeds found in predictions")
            return pd.DataFrame()

        # Add seed metadata
        missed_df['seed_region'] = missed_df['reach_id'].map(
            lambda x: SEED_REACHES.get(x, {}).get('region', 'unknown')
        )
        missed_df['seed_note'] = missed_df['reach_id'].map(
            lambda x: SEED_REACHES.get(x, {}).get('note', '')
        )
        missed_df['rf_caught'] = missed_df['rf_prediction'] == 1

        # Add features if available
        if self.features_df is not None:
            feature_cols = ['facc', 'width', 'facc_width_ratio', 'path_freq',
                           'main_side', 'facc_jump_ratio', 'fwr_drop_ratio']
            available_cols = [c for c in feature_cols if c in self.features_df.columns]

            if available_cols:
                features_subset = self.features_df[['reach_id'] + available_cols]
                missed_df = missed_df.merge(features_subset, on='reach_id', how='left')

        return missed_df

    def analyze_seed_coverage(self) -> Dict[str, Any]:
        """
        Analyze overall seed coverage by RF classifier.

        Returns
        -------
        dict
            Seed coverage metrics.
        """
        df = self.predictions_df.copy()

        all_seed_ids = list(SEED_REACHES.keys())
        seeds_in_data = df[df['reach_id'].isin(all_seed_ids)]

        rf_detected_seeds = seeds_in_data[seeds_in_data['rf_prediction'] == 1]['reach_id'].tolist()
        rf_missed_seeds = seeds_in_data[seeds_in_data['rf_prediction'] == 0]['reach_id'].tolist()

        # Compare to rule detection
        rule_detected_seeds = seeds_in_data[seeds_in_data['is_rule_anomaly'] == 1]['reach_id'].tolist()

        return {
            'total_seeds': len(all_seed_ids),
            'seeds_in_data': len(seeds_in_data),
            'rf_detected': len(rf_detected_seeds),
            'rf_missed': len(rf_missed_seeds),
            'rf_recall': len(rf_detected_seeds) / len(seeds_in_data) if len(seeds_in_data) > 0 else 0,
            'rule_detected': len(rule_detected_seeds),
            'rule_recall': len(rule_detected_seeds) / len(seeds_in_data) if len(seeds_in_data) > 0 else 0,
            'rf_detected_ids': rf_detected_seeds,
            'rf_missed_ids': rf_missed_seeds,
            'rf_catches_missed_rules': [s for s in MISSED_SEEDS if s in rf_detected_seeds],
        }

    def get_novel_anomalies(
        self,
        min_probability: float = 0.7,
        max_results: int = 100
    ) -> pd.DataFrame:
        """
        Get anomalies found by RF but not by rules.

        Parameters
        ----------
        min_probability : float
            Minimum RF probability to include.
        max_results : int
            Maximum number of results.

        Returns
        -------
        pd.DataFrame
            Novel anomalies with features.
        """
        df = self.predictions_df.copy()

        # RF positive but rule negative
        novel = df[
            (df['rf_prediction'] == 1) &
            (df['is_rule_anomaly'] == 0) &
            (df['rf_probability'] >= min_probability)
        ].copy()

        novel = novel.sort_values('rf_probability', ascending=False)

        # Add features if available
        if self.features_df is not None:
            feature_cols = ['facc', 'width', 'facc_width_ratio', 'path_freq',
                           'stream_order', 'main_side', 'end_reach',
                           'facc_jump_ratio', 'fwr_drop_ratio', 'ratio_to_median']
            available_cols = [c for c in feature_cols if c in self.features_df.columns]

            if available_cols:
                features_subset = self.features_df[['reach_id'] + available_cols]
                novel = novel.merge(features_subset, on='reach_id', how='left')

        return novel.head(max_results)

    def get_false_negatives(self, max_results: int = 50) -> pd.DataFrame:
        """
        Get rule-detected anomalies that RF missed.

        Parameters
        ----------
        max_results : int
            Maximum number of results.

        Returns
        -------
        pd.DataFrame
            False negatives with features.
        """
        df = self.predictions_df.copy()

        # Rule positive but RF negative
        fn = df[
            (df['rf_prediction'] == 0) &
            (df['is_rule_anomaly'] == 1)
        ].copy()

        fn = fn.sort_values('rf_probability', ascending=False)

        # Add features if available
        if self.features_df is not None:
            feature_cols = ['facc', 'width', 'facc_width_ratio', 'path_freq',
                           'stream_order', 'main_side', 'end_reach',
                           'facc_jump_ratio', 'fwr_drop_ratio']
            available_cols = [c for c in feature_cols if c in self.features_df.columns]

            if available_cols:
                features_subset = self.features_df[['reach_id'] + available_cols]
                fn = fn.merge(features_subset, on='reach_id', how='left')

        return fn.head(max_results)

    def generate_report(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report.

        Parameters
        ----------
        output_dir : str, optional
            Directory to save report files.

        Returns
        -------
        dict
            Complete analysis results.
        """
        report = {
            'comparison': self.compare_to_rules(),
            'seed_coverage': self.analyze_seed_coverage(),
            'missed_seeds_analysis': self.analyze_missed_seeds().to_dict('records'),
        }

        # Feature importance if model available
        if self.model is not None:
            importance_df = self.model.get_feature_importance()
            report['top_20_features'] = importance_df.head(20).to_dict('records')
            report['n_selected_features'] = len(self.model.selected_features)

        # Novel anomalies
        novel_df = self.get_novel_anomalies()
        report['n_novel_anomalies'] = len(novel_df)
        report['novel_anomalies_sample'] = novel_df.head(10).to_dict('records')

        # False negatives
        fn_df = self.get_false_negatives()
        report['n_false_negatives'] = len(fn_df)
        report['false_negatives_sample'] = fn_df.head(10).to_dict('records')

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save report JSON
            report_path = output_dir / 'rf_analysis_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Saved report to {report_path}")

            # Save markdown report
            md_path = output_dir / 'rf_analysis_report.md'
            self._write_markdown_report(report, md_path)
            print(f"Saved markdown report to {md_path}")

            # Save novel anomalies GeoJSON if features have geometry
            if len(novel_df) > 0 and self.features_df is not None:
                self._export_geojson(
                    novel_df,
                    output_dir / 'rf_novel_anomalies.geojson',
                    'Novel anomalies found by RF but not rules'
                )

        return report

    def _write_markdown_report(self, report: Dict[str, Any], path: Path):
        """Write analysis report as markdown."""
        lines = [
            "# RF Classifier Analysis Report",
            "",
            "## Summary",
            "",
            f"- **RF detections**: {report['comparison']['n_rf_positives']:,}",
            f"- **Rule detections**: {report['comparison']['n_rule_positives']:,}",
            f"- **Overlap**: {report['comparison']['n_overlap']:,} ({report['comparison']['overlap_pct_of_rules']:.1f}% of rules)",
            f"- **RF-only**: {report['comparison']['n_rf_only']:,}",
            f"- **Rules-only**: {report['comparison']['n_rules_only']:,}",
            "",
            "## Seed Coverage",
            "",
            f"- **Total seeds**: {report['seed_coverage']['total_seeds']}",
            f"- **RF recall**: {report['seed_coverage']['rf_recall']:.1%} ({report['seed_coverage']['rf_detected']}/{report['seed_coverage']['seeds_in_data']})",
            f"- **Rule recall**: {report['seed_coverage']['rule_recall']:.1%} ({report['seed_coverage']['rule_detected']}/{report['seed_coverage']['seeds_in_data']})",
            "",
        ]

        # Missed seeds analysis
        lines.extend([
            "## Analysis of 5 Missed Seeds",
            "",
            "Seeds missed by rule-based detection:",
            "",
        ])

        rf_catches = report['seed_coverage'].get('rf_catches_missed_rules', [])
        for seed_info in report['missed_seeds_analysis']:
            reach_id = seed_info.get('reach_id')
            region = seed_info.get('seed_region', seed_info.get('region', '?'))
            rf_caught = seed_info.get('rf_caught', False)
            prob = seed_info.get('rf_probability', 0)
            note = seed_info.get('seed_note', '')

            status = "RF caught" if rf_caught else "RF missed"
            lines.append(f"- **{reach_id}** ({region}): {status} (prob={prob:.3f}) {note}")

        lines.extend([
            "",
            f"**RF catches {len(rf_catches)}/5 missed seeds**",
            "",
        ])

        # Feature importance
        if 'top_20_features' in report:
            lines.extend([
                "## Top 20 Features (by RFE rank)",
                "",
                "| Rank | Feature | Importance | Selected |",
                "|------|---------|------------|----------|",
            ])

            for feat in report['top_20_features']:
                lines.append(
                    f"| {feat['rfe_rank']} | {feat['feature']} | "
                    f"{feat['importance']:.4f} | {'Yes' if feat['selected'] else 'No'} |"
                )

            lines.append("")

        # Novel anomalies
        lines.extend([
            "## Novel Anomalies (RF-only)",
            "",
            f"Found **{report['n_novel_anomalies']}** anomalies detected by RF but not rules.",
            "",
        ])

        if report['novel_anomalies_sample']:
            lines.extend([
                "Sample (top 10 by probability):",
                "",
                "| reach_id | region | probability | facc | width | FWR |",
                "|----------|--------|-------------|------|-------|-----|",
            ])

            for row in report['novel_anomalies_sample']:
                lines.append(
                    f"| {row.get('reach_id')} | {row.get('region')} | "
                    f"{row.get('rf_probability', 0):.3f} | "
                    f"{row.get('facc', 0):,.0f} | {row.get('width', 0):.0f} | "
                    f"{row.get('facc_width_ratio', 0):,.0f} |"
                )

            lines.append("")

        # False negatives
        lines.extend([
            "## False Negatives (Rules-only)",
            "",
            f"Found **{report['n_false_negatives']}** anomalies detected by rules but not RF.",
            "",
        ])

        if report['false_negatives_sample']:
            lines.extend([
                "Sample (top 10 by RF probability):",
                "",
                "| reach_id | region | probability | facc | width | FWR |",
                "|----------|--------|-------------|------|-------|-----|",
            ])

            for row in report['false_negatives_sample']:
                lines.append(
                    f"| {row.get('reach_id')} | {row.get('region')} | "
                    f"{row.get('rf_probability', 0):.3f} | "
                    f"{row.get('facc', 0):,.0f} | {row.get('width', 0):.0f} | "
                    f"{row.get('facc_width_ratio', 0):,.0f} |"
                )

            lines.append("")

        with open(path, 'w') as f:
            f.write('\n'.join(lines))

    def _export_geojson(
        self,
        df: pd.DataFrame,
        path: Path,
        description: str
    ):
        """Export DataFrame as GeoJSON."""
        # Try to get geometry from features
        if self.features_df is not None and 'geometry' in self.features_df.columns:
            geom_df = self.features_df[['reach_id', 'geometry']]
            df = df.merge(geom_df, on='reach_id', how='left')

            # Convert to GeoDataFrame
            try:
                gdf = gpd.GeoDataFrame(df, geometry='geometry')
                gdf.to_file(path, driver='GeoJSON')
                print(f"Exported {len(gdf)} features to {path}")
            except Exception as e:
                print(f"Could not export GeoJSON: {e}")
        else:
            # Export as regular JSON with reach_ids
            features = []
            for _, row in df.iterrows():
                props = {k: (v.item() if hasattr(v, 'item') else v)
                        for k, v in row.items() if k != 'geometry'}
                features.append({
                    'type': 'Feature',
                    'properties': props,
                    'geometry': None
                })

            geojson = {
                'type': 'FeatureCollection',
                'name': description,
                'features': features
            }

            with open(path, 'w') as f:
                json.dump(geojson, f, indent=2, default=str)
            print(f"Exported {len(features)} features to {path} (no geometry)")


def evaluate_rf_classifier(
    predictions_path: str,
    features_path: Optional[str] = None,
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to evaluate RF classifier.

    Parameters
    ----------
    predictions_path : str
        Path to rf_predictions.parquet.
    features_path : str, optional
        Path to rf_features.parquet.
    model_path : str, optional
        Path to rf_model.joblib.
    output_dir : str, optional
        Directory for output files.

    Returns
    -------
    dict
        Analysis report.

    Examples
    --------
    >>> report = evaluate_rf_classifier(
    ...     "output/facc_detection/rf_predictions.parquet",
    ...     features_path="output/facc_detection/rf_features.parquet",
    ...     model_path="output/facc_detection/rf_model.joblib",
    ...     output_dir="output/facc_detection/"
    ... )
    """
    evaluator = RFEvaluator(
        predictions_path=predictions_path,
        features_path=features_path,
        model_path=model_path
    )
    return evaluator.generate_report(output_dir=output_dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate RF classifier for facc anomaly detection')
    parser.add_argument('--predictions', required=True, help='Path to rf_predictions.parquet')
    parser.add_argument('--features', help='Path to rf_features.parquet')
    parser.add_argument('--model', help='Path to rf_model.joblib')
    parser.add_argument('--output', default='output/facc_detection/', help='Output directory')

    args = parser.parse_args()

    report = evaluate_rf_classifier(
        predictions_path=args.predictions,
        features_path=args.features,
        model_path=args.model,
        output_dir=args.output
    )

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"\nComparison Summary:")
    print(f"  RF positives: {report['comparison']['n_rf_positives']:,}")
    print(f"  Rule positives: {report['comparison']['n_rule_positives']:,}")
    print(f"  Overlap: {report['comparison']['n_overlap']:,}")

    print(f"\nSeed Coverage:")
    print(f"  RF recall: {report['seed_coverage']['rf_recall']:.1%}")
    print(f"  Rule recall: {report['seed_coverage']['rule_recall']:.1%}")

    rf_catches = report['seed_coverage'].get('rf_catches_missed_rules', [])
    print(f"\nRF catches {len(rf_catches)}/5 missed seeds: {rf_catches}")

    print(f"\nNovel anomalies (RF-only): {report['n_novel_anomalies']}")
    print(f"False negatives (rules-only): {report['n_false_negatives']}")
