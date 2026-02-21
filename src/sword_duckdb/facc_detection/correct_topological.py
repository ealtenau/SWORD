# -*- coding: utf-8 -*-
"""
Topological Facc Correction (Hybrid Approach)
==============================================

Corrects facc anomalies using hybrid model strategy:

1. **Entry points** (93%): Use NO-FACC model (R²=0.79)
   - No clean upstream to reference
   - Breaks the corruption chain
   - Accept 32.8% error to avoid tautology

2. **Write to DB** (optional): Apply entry point corrections

3. **Re-extract features**: Now 2-hop reads corrected values

4. **Propagation** (7%): Use STANDARD model (R²=0.98)
   - Upstream is now clean
   - Get full accuracy benefit

This avoids the tautology problem where the standard model
predicts corrupted values because 2-hop neighbors are also corrupted.
"""

from typing import List, Dict, Set, Tuple
from pathlib import Path
import duckdb
import pandas as pd
import json
from datetime import datetime

from .rf_regressor import FaccRegressor, SplitFaccRegressor
from .rf_features import RFFeatureExtractor


def identify_entry_points(
    anomalies: pd.DataFrame,
    conn: duckdb.DuckDBPyConnection,
) -> Tuple[Set[int], Set[int]]:
    """
    Separate anomalies into entry points vs propagation.

    Entry points: where bad facc first enters (no corrupted upstream)
    Propagation: downstream of entry points (inherited bad facc)

    Returns
    -------
    tuple of (entry_point_ids, propagation_ids)
    """
    anomaly_ids = set(anomalies["reach_id"].tolist())

    if not anomaly_ids:
        return set(), set()

    # Get upstream neighbors for all anomalies
    ids_str = ", ".join(str(r) for r in anomaly_ids)

    query = f"""
    SELECT
        rt.reach_id,
        rt.neighbor_reach_id as upstream_id
    FROM reach_topology rt
    WHERE rt.reach_id IN ({ids_str})
        AND rt.direction = 'up'
    """

    upstream_df = conn.execute(query).fetchdf()

    # Entry point = no upstream neighbors are also anomalies
    entry_points = set()
    propagation = set()

    for reach_id in anomaly_ids:
        upstream_neighbors = upstream_df[upstream_df["reach_id"] == reach_id][
            "upstream_id"
        ].tolist()
        upstream_anomalies = set(upstream_neighbors) & anomaly_ids

        if len(upstream_anomalies) == 0:
            # No corrupted upstream = entry point
            entry_points.add(reach_id)
        else:
            # Has corrupted upstream = propagation
            propagation.add(reach_id)

    return entry_points, propagation


def get_downstream_order(
    entry_points: Set[int],
    propagation: Set[int],
    conn: duckdb.DuckDBPyConnection,
    max_hops: int = 50,
) -> List[List[int]]:
    """
    Order propagation reaches by distance from entry points.

    Returns list of lists: [[hop1_ids], [hop2_ids], ...]
    """
    if not propagation or not entry_points:
        return []

    entry_str = ", ".join(str(r) for r in entry_points)
    prop_str = ", ".join(str(r) for r in propagation)

    query = f"""
    WITH RECURSIVE downstream_trace AS (
        -- Start at entry points
        SELECT reach_id, 0 as hop
        FROM reaches
        WHERE reach_id IN ({entry_str})

        UNION ALL

        -- Follow downstream
        SELECT rt.neighbor_reach_id as reach_id, dt.hop + 1
        FROM downstream_trace dt
        JOIN reach_topology rt ON dt.reach_id = rt.reach_id
        WHERE rt.direction = 'down'
            AND dt.hop < {max_hops}
            AND rt.neighbor_reach_id IN ({prop_str})
    )
    SELECT reach_id, MIN(hop) as min_hop
    FROM downstream_trace
    WHERE hop > 0
    GROUP BY reach_id
    ORDER BY min_hop
    """

    df = conn.execute(query).fetchdf()

    # Group by hop level
    hop_groups = []
    if len(df) > 0:
        max_hop = int(df["min_hop"].max())
        for h in range(1, max_hop + 1):
            hop_ids = df[df["min_hop"] == h]["reach_id"].tolist()
            if hop_ids:
                hop_groups.append(hop_ids)

    return hop_groups


def apply_corrections_to_db(
    conn: duckdb.DuckDBPyConnection,
    corrections: pd.DataFrame,
    batch_name: str,
) -> int:
    """
    Apply facc corrections to database.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Read-write connection to DuckDB.
    corrections : pd.DataFrame
        DataFrame with reach_id, predicted_facc columns.
    batch_name : str
        Name for this correction batch (for logging).

    Returns
    -------
    int
        Number of rows updated.
    """
    if len(corrections) == 0:
        return 0

    # Load spatial extension for RTREE index support
    conn.execute("INSTALL spatial; LOAD spatial;")

    # Drop RTREE index before update (known DuckDB issue)
    try:
        conn.execute("DROP INDEX IF EXISTS reaches_geom_idx")
    except Exception:
        pass  # Index may not exist

    # Create temp table with corrections
    conn.execute("DROP TABLE IF EXISTS _temp_corrections")
    conn.execute("""
        CREATE TEMP TABLE _temp_corrections (
            reach_id BIGINT PRIMARY KEY,
            new_facc DOUBLE
        )
    """)

    # Insert corrections
    for _, row in corrections.iterrows():
        conn.execute(
            "INSERT INTO _temp_corrections VALUES (?, ?)",
            [int(row["reach_id"]), float(row["predicted_facc"])],
        )

    # Update reaches table
    result = conn.execute("""
        UPDATE reaches
        SET facc = tc.new_facc
        FROM _temp_corrections tc
        WHERE reaches.reach_id = tc.reach_id
    """)

    result.fetchone() if result else None  # noqa: F841 — ensures DuckDB cursor is consumed

    # Clean up
    conn.execute("DROP TABLE IF EXISTS _temp_corrections")

    # Recreate RTREE index
    try:
        conn.execute("CREATE INDEX reaches_geom_idx ON reaches USING RTREE(geom)")
    except Exception:
        pass  # Index may already exist or not be needed

    print(f"  Applied {len(corrections)} corrections to DB [{batch_name}]")

    return len(corrections)


def correct_hybrid(
    db_path: str,
    anomalies: pd.DataFrame,
    nofacc_model_path: Path,
    standard_model_path: Path,
    output_dir: Path,
    dry_run: bool = True,
    use_split_models: bool = False,
) -> pd.DataFrame:
    """
    Correct anomalies using hybrid approach.

    Parameters
    ----------
    db_path : str
        Path to DuckDB database.
    anomalies : pd.DataFrame
        Detected anomalies with reach_id column.
    nofacc_model_path : Path
        Path to no-facc model (.joblib) for entry points.
    standard_model_path : Path
        Path to standard model (.joblib) for propagation.
    output_dir : Path
        Output directory for results.
    dry_run : bool
        If True, don't write to DB. If False, apply corrections.
    use_split_models : bool
        If True, use SplitFaccRegressor variants.

    Returns
    -------
    pd.DataFrame
        Corrections with original and predicted facc.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open connection (read-write if not dry_run)
    conn = duckdb.connect(db_path, read_only=dry_run)

    try:
        # Load models
        if use_split_models:
            nofacc_model = SplitFaccRegressor.load(nofacc_model_path)
            standard_model = SplitFaccRegressor.load(standard_model_path)
        else:
            nofacc_model = FaccRegressor.load(nofacc_model_path)
            standard_model = FaccRegressor.load(standard_model_path)

        print(f"Loaded no-facc model from {nofacc_model_path}")
        print(f"Loaded standard model from {standard_model_path}")
        print(f"Mode: {'DRY RUN' if dry_run else 'APPLYING TO DB'}")

        # Identify entry points vs propagation
        print("\n" + "=" * 60)
        print("STEP 1: Identify entry points vs propagation")
        print("=" * 60)

        entry_points, propagation = identify_entry_points(anomalies, conn)
        print(
            f"  Entry points: {len(entry_points)} ({100 * len(entry_points) / len(anomalies):.1f}%)"
        )
        print(
            f"  Propagation:  {len(propagation)} ({100 * len(propagation) / len(anomalies):.1f}%)"
        )

        # Get downstream order for propagation
        hop_groups = get_downstream_order(entry_points, propagation, conn)
        if hop_groups:
            print(f"  Propagation hops: {len(hop_groups)}")
            for i, group in enumerate(hop_groups):
                print(f"    Hop {i + 1}: {len(group)} reaches")

        # Storage
        all_corrections = []

        # ============================================================
        # STEP 2: Correct entry points with NO-FACC model
        # ============================================================
        print("\n" + "=" * 60)
        print("STEP 2: Correct entry points with NO-FACC model (R²=0.79)")
        print("=" * 60)

        # Extract features (pass conn to avoid read_only mismatch)
        print("  Extracting features...")
        extractor = RFFeatureExtractor(conn)
        all_features = extractor.extract_all(region=None)

        entry_features = all_features[
            all_features["reach_id"].isin(entry_points)
        ].copy()

        if len(entry_features) > 0:
            entry_preds = nofacc_model.predict(entry_features)
            entry_preds["correction_pass"] = 1
            entry_preds["correction_type"] = "entry_point"
            entry_preds["model_type"] = "nofacc"
            all_corrections.append(entry_preds)

            print(f"  Corrected {len(entry_preds)} entry points")
            print(f"  Median original facc: {entry_preds['facc'].median():,.0f} km²")
            print(
                f"  Median predicted facc: {entry_preds['predicted_facc'].median():,.0f} km²"
            )
            print(f"  Median ratio: {entry_preds['facc_ratio'].median():.2f}x")

            # Apply to DB if not dry run
            if not dry_run:
                apply_corrections_to_db(conn, entry_preds, "entry_points")

        # ============================================================
        # STEP 3: Re-extract features (now 2-hop reads corrected values)
        # ============================================================
        if propagation and not dry_run:
            print("\n" + "=" * 60)
            print("STEP 3: Re-extract features from DB")
            print("=" * 60)

            extractor = RFFeatureExtractor(conn)
            all_features = extractor.extract_all(region=None)

            print("  Features re-extracted with corrected upstream values")

        # ============================================================
        # STEP 4: Correct propagation with STANDARD model
        # ============================================================
        if propagation:
            print("\n" + "=" * 60)
            print("STEP 4: Correct propagation with STANDARD model (R²=0.98)")
            print("=" * 60)

            # If dry run, we need to simulate the corrected 2-hop features
            if dry_run:
                corrected_facc = {}
                if all_corrections:
                    for _, row in all_corrections[0].iterrows():
                        corrected_facc[row["reach_id"]] = row["predicted_facc"]

            for hop_idx, hop_ids in enumerate(hop_groups):
                pass_num = hop_idx + 2
                print(
                    f"\n  Pass {pass_num}: Hop {hop_idx + 1} ({len(hop_ids)} reaches)"
                )

                hop_features = all_features[
                    all_features["reach_id"].isin(hop_ids)
                ].copy()

                if len(hop_features) == 0:
                    continue

                # If dry run, update features in-memory
                if dry_run:
                    hop_features = _update_upstream_features(
                        hop_features, corrected_facc, conn
                    )

                # Predict with standard model
                hop_preds = standard_model.predict(hop_features)
                hop_preds["correction_pass"] = pass_num
                hop_preds["correction_type"] = f"propagation_hop{hop_idx + 1}"
                hop_preds["model_type"] = "standard"
                all_corrections.append(hop_preds)

                # Store for next hop (dry run only)
                if dry_run:
                    for _, row in hop_preds.iterrows():
                        corrected_facc[row["reach_id"]] = row["predicted_facc"]

                # Apply to DB if not dry run
                if not dry_run:
                    apply_corrections_to_db(conn, hop_preds, f"hop{hop_idx + 1}")

                print(
                    f"    Median original facc: {hop_preds['facc'].median():,.0f} km²"
                )
                print(
                    f"    Median predicted facc: {hop_preds['predicted_facc'].median():,.0f} km²"
                )
                print(f"    Median ratio: {hop_preds['facc_ratio'].median():.2f}x")

        # Combine all corrections
        if all_corrections:
            corrections = pd.concat(all_corrections, ignore_index=True)
        else:
            corrections = pd.DataFrame()

        # Save results
        output_file = output_dir / "topological_corrections_hybrid.csv"
        corrections.to_csv(output_file, index=False)
        print(f"\nSaved {len(corrections)} corrections to {output_file}")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total corrected: {len(corrections)}")

        if len(corrections) > 0:
            print(f"Median original facc: {corrections['facc'].median():,.0f} km²")
            print(
                f"Median predicted facc: {corrections['predicted_facc'].median():,.0f} km²"
            )
            print(f"Overall median ratio: {corrections['facc_ratio'].median():.2f}x")

            print("\nBy model type:")
            for mtype in corrections["model_type"].unique():
                subset = corrections[corrections["model_type"] == mtype]
                print(
                    f"  {mtype}: n={len(subset)}, median_ratio={subset['facc_ratio'].median():.2f}x"
                )

            print("\nBy correction type:")
            for ctype in corrections["correction_type"].unique():
                subset = corrections[corrections["correction_type"] == ctype]
                print(
                    f"  {ctype}: n={len(subset)}, median_ratio={subset['facc_ratio'].median():.2f}x"
                )

        # Save summary JSON
        summary = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": dry_run,
            "total_anomalies": len(anomalies),
            "entry_points": len(entry_points),
            "propagation": len(propagation),
            "total_corrected": len(corrections),
            "models": {
                "entry_points": str(nofacc_model_path),
                "propagation": str(standard_model_path),
            },
        }

        if len(corrections) > 0:
            summary["metrics"] = {
                "median_original_facc": float(corrections["facc"].median()),
                "median_predicted_facc": float(corrections["predicted_facc"].median()),
                "median_ratio": float(corrections["facc_ratio"].median()),
            }

        with open(output_dir / "topological_corrections_hybrid_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        return corrections

    finally:
        conn.close()


def _update_upstream_features(
    features: pd.DataFrame,
    corrected_facc: Dict[int, float],
    conn: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """
    Update 2-hop upstream features with corrected facc values.

    This is called for propagation reaches after their upstream
    entry points have been corrected (in-memory for dry run).
    """
    if not corrected_facc:
        return features

    features = features.copy()
    reach_ids = features["reach_id"].tolist()

    if not reach_ids:
        return features

    ids_str = ", ".join(str(r) for r in reach_ids)

    # Get 1-hop and 2-hop upstream neighbors
    query = f"""
    WITH hop1 AS (
        SELECT rt.reach_id, rt.neighbor_reach_id as up1
        FROM reach_topology rt
        WHERE rt.reach_id IN ({ids_str}) AND rt.direction = 'up'
    ),
    hop2 AS (
        SELECT h1.reach_id, rt.neighbor_reach_id as up2
        FROM hop1 h1
        JOIN reach_topology rt ON h1.up1 = rt.reach_id AND rt.direction = 'up'
    )
    SELECT reach_id, up2 FROM hop2
    """

    hop2_df = conn.execute(query).fetchdf()

    # For each reach, update max_2hop_upstream_facc if any 2-hop neighbor was corrected
    for reach_id in reach_ids:
        up2_neighbors = hop2_df[hop2_df["reach_id"] == reach_id]["up2"].tolist()

        # Get corrected values for 2-hop neighbors
        corrected_up2 = [
            corrected_facc[n] for n in up2_neighbors if n in corrected_facc
        ]

        if corrected_up2:
            idx = features[features["reach_id"] == reach_id].index

            if len(idx) == 0:
                continue

            # Use the corrected max
            new_max = max(corrected_up2)
            features.loc[idx, "max_2hop_upstream_facc"] = new_max

            # Update facc_2hop_ratio
            facc = features.loc[idx, "facc"].values[0]
            if new_max > 0:
                features.loc[idx, "facc_2hop_ratio"] = facc / new_max

    return features


# Keep the old function for backward compatibility
def correct_in_order(
    db_path: str,
    anomalies: pd.DataFrame,
    model_path: Path,
    output_dir: Path,
    use_split_model: bool = False,
) -> pd.DataFrame:
    """
    [DEPRECATED] Use correct_hybrid() instead.

    This function uses a single model for all corrections,
    which causes tautology issues when 2-hop neighbors are corrupted.
    """
    print("WARNING: correct_in_order() is deprecated. Use correct_hybrid() instead.")

    # Just use the model for everything (old behavior)
    output_dir.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(db_path, read_only=True)

    try:
        if use_split_model:
            model = SplitFaccRegressor.load(model_path)
        else:
            model = FaccRegressor.load(model_path)

        entry_points, propagation = identify_entry_points(anomalies, conn)
        hop_groups = get_downstream_order(entry_points, propagation, conn)

        with RFFeatureExtractor(db_path) as extractor:
            all_features = extractor.extract_all(region=None)

        all_corrections = []
        corrected_facc = {}

        # Entry points
        entry_features = all_features[
            all_features["reach_id"].isin(entry_points)
        ].copy()
        if len(entry_features) > 0:
            entry_preds = model.predict(entry_features)
            entry_preds["correction_pass"] = 1
            entry_preds["correction_type"] = "entry_point"
            all_corrections.append(entry_preds)
            for _, row in entry_preds.iterrows():
                corrected_facc[row["reach_id"]] = row["predicted_facc"]

        # Propagation
        for hop_idx, hop_ids in enumerate(hop_groups):
            hop_features = all_features[all_features["reach_id"].isin(hop_ids)].copy()
            if len(hop_features) == 0:
                continue
            hop_features = _update_upstream_features(hop_features, corrected_facc, conn)
            hop_preds = model.predict(hop_features)
            hop_preds["correction_pass"] = hop_idx + 2
            hop_preds["correction_type"] = f"propagation_hop{hop_idx + 1}"
            all_corrections.append(hop_preds)
            for _, row in hop_preds.iterrows():
                corrected_facc[row["reach_id"]] = row["predicted_facc"]

        corrections = (
            pd.concat(all_corrections, ignore_index=True)
            if all_corrections
            else pd.DataFrame()
        )
        corrections.to_csv(output_dir / "topological_corrections.csv", index=False)

        return corrections

    finally:
        conn.close()


def main():
    import argparse
    import geopandas as gpd

    parser = argparse.ArgumentParser(
        description="Correct facc anomalies using hybrid approach",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # Dry run (no DB changes)
  python -m src.sword_duckdb.facc_detection.correct_topological \\
      --db data/duckdb/sword_v17c.duckdb \\
      --anomalies output/facc_detection/all_anomalies.geojson \\
      --nofacc-model output/facc_detection/rf_regressor_baseline_nofacc.joblib \\
      --standard-model output/facc_detection/rf_regressor_baseline.joblib

  # Apply to DB
  python -m src.sword_duckdb.facc_detection.correct_topological \\
      --db data/duckdb/sword_v17c.duckdb \\
      --anomalies output/facc_detection/all_anomalies.geojson \\
      --nofacc-model output/facc_detection/rf_regressor_baseline_nofacc.joblib \\
      --standard-model output/facc_detection/rf_regressor_baseline.joblib \\
      --apply
        """,
    )
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--anomalies", required=True, help="Path to anomalies GeoJSON")
    parser.add_argument(
        "--nofacc-model", required=True, help="Path to no-facc model for entry points"
    )
    parser.add_argument(
        "--standard-model", required=True, help="Path to standard model for propagation"
    )
    parser.add_argument(
        "--output-dir", default="output/facc_detection", help="Output directory"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply corrections to DB (default: dry run)",
    )
    parser.add_argument(
        "--split-models", action="store_true", help="Use split model variants"
    )

    args = parser.parse_args()

    # Load anomalies
    anomalies = gpd.read_file(args.anomalies)
    print(f"Loaded {len(anomalies)} anomalies from {args.anomalies}")

    # Run hybrid correction
    corrections = correct_hybrid(
        db_path=args.db,
        anomalies=anomalies,
        nofacc_model_path=Path(args.nofacc_model),
        standard_model_path=Path(args.standard_model),
        output_dir=Path(args.output_dir),
        dry_run=not args.apply,
        use_split_models=args.split_models,
    )

    return corrections


if __name__ == "__main__":
    main()
