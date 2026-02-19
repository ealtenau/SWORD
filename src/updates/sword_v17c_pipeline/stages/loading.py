"""Data loading stage for v17c pipeline."""

import duckdb
import pandas as pd

from ._logging import log

# Default model paths for facc correction
DEFAULT_NOFACC_MODEL = "output/facc_detection/rf_regressor_baseline_nofacc.joblib"
DEFAULT_STANDARD_MODEL = "output/facc_detection/rf_regressor_baseline.joblib"


def load_topology(conn: duckdb.DuckDBPyConnection, region: str) -> pd.DataFrame:
    """Load reach_topology from DuckDB."""
    log(f"Loading topology for {region}...")
    df = conn.execute(
        """
        SELECT reach_id, direction, neighbor_rank, neighbor_reach_id
        FROM reach_topology
        WHERE region = ?
    """,
        [region.upper()],
    ).fetchdf()
    log(f"Loaded {len(df):,} topology rows")
    return df


def load_reaches(conn: duckdb.DuckDBPyConnection, region: str) -> pd.DataFrame:
    """Load reaches with attributes."""
    log(f"Loading reaches for {region}...")

    # Get available columns (handles older DBs without v17c columns)
    cols_result = conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'reaches'"
    ).fetchall()
    available_cols = {row[0].lower() for row in cols_result}

    # Core columns (required)
    core_cols = [
        "reach_id",
        "region",
        "reach_length",
        "width",
        "slope",
        "facc",
        "n_rch_up",
        "n_rch_down",
        "dist_out",
        "path_freq",
        "stream_order",
        "lakeflag",
        "trib_flag",
    ]

    # Optional columns (v17c additions)
    optional_cols = [
        "wse_obs_mean",
        "wse_obs_std",
        "width_obs_median",
        "n_obs",
        "main_side",
        "type",
        "end_reach",
        "path_order",
        "path_segs",
    ]

    # Build column list
    select_cols = [c for c in core_cols if c.lower() in available_cols]
    select_cols += [c for c in optional_cols if c.lower() in available_cols]

    df = conn.execute(
        f"""
        SELECT {", ".join(select_cols)}
        FROM reaches
        WHERE region = ?
    """,
        [region.upper()],
    ).fetchdf()
    log(f"Loaded {len(df):,} reaches")
    return df


def run_facc_corrections(
    conn: duckdb.DuckDBPyConnection,
    region: str,
    nofacc_model_path: str,
    standard_model_path: str,
) -> int:
    """
    Detect and correct facc anomalies directly in the DB.

    Uses the hybrid approach: nofacc model for entry points, then standard
    model for propagation reaches after re-extracting features.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Read-write connection to DuckDB.
    region : str
        Region code (e.g. 'NA').
    nofacc_model_path : str
        Path to no-facc RF model (.joblib) for entry points.
    standard_model_path : str
        Path to standard RF model (.joblib) for propagation.

    Returns
    -------
    int
        Number of corrections applied.

    Raises
    ------
    FileNotFoundError
        If model files don't exist.
    RuntimeError
        If detection or correction fails.
    """
    from pathlib import Path as _Path

    # Validate model files exist
    nofacc_path = _Path(nofacc_model_path)
    standard_path = _Path(standard_model_path)
    if not nofacc_path.exists():
        raise FileNotFoundError(f"No-facc model not found: {nofacc_model_path}")
    if not standard_path.exists():
        raise FileNotFoundError(f"Standard model not found: {standard_model_path}")

    # Import facc detection modules
    from sword_duckdb.facc_detection.detect import detect_hybrid
    from sword_duckdb.facc_detection.correct_topological import (
        identify_entry_points,
        get_downstream_order,
        apply_corrections_to_db,
    )
    from sword_duckdb.facc_detection.rf_features import RFFeatureExtractor
    from sword_duckdb.facc_detection.rf_regressor import FaccRegressor

    # Step 1: Detect anomalies
    log(f"Detecting facc anomalies for {region}...")
    result = detect_hybrid(conn, region=region)
    anomalies = result.anomalies

    if len(anomalies) == 0:
        log("No facc anomalies detected")
        return 0

    log(f"Detected {len(anomalies):,} facc anomalies")

    # Step 2: Separate entry points vs propagation
    entry_points, propagation = identify_entry_points(anomalies, conn)
    log(f"  Entry points: {len(entry_points):,}, Propagation: {len(propagation):,}")

    if not entry_points:
        log("No entry points found, skipping correction")
        return 0

    # Step 3: Get downstream ordering for propagation
    hop_groups = get_downstream_order(entry_points, propagation, conn)

    # Load models
    nofacc_model = FaccRegressor.load(nofacc_path)
    standard_model = FaccRegressor.load(standard_path)
    log(f"Loaded models: nofacc={nofacc_path.name}, standard={standard_path.name}")

    total_corrected = 0

    # Step 4: Extract features and correct entry points with nofacc model
    log("Extracting features for entry point correction...")
    extractor = RFFeatureExtractor(conn)
    all_features = extractor.extract_all(region=region)

    entry_features = all_features[all_features["reach_id"].isin(entry_points)].copy()
    if len(entry_features) > 0:
        entry_preds = nofacc_model.predict(entry_features)
        apply_corrections_to_db(conn, entry_preds, "entry_points")
        total_corrected += len(entry_preds)
        log(f"  Corrected {len(entry_preds):,} entry points")

    # Step 5: Re-extract features (2-hop now reads corrected values)
    if propagation and hop_groups:
        log("Re-extracting features after entry point corrections...")
        extractor = RFFeatureExtractor(conn)
        all_features = extractor.extract_all(region=region)

        # Step 6: Correct propagation reaches with standard model
        for hop_idx, hop_ids in enumerate(hop_groups):
            hop_features = all_features[all_features["reach_id"].isin(hop_ids)].copy()
            if len(hop_features) == 0:
                continue
            hop_preds = standard_model.predict(hop_features)
            apply_corrections_to_db(conn, hop_preds, f"propagation_hop{hop_idx + 1}")
            total_corrected += len(hop_preds)
            log(
                f"  Corrected {len(hop_preds):,} propagation reaches (hop {hop_idx + 1})"
            )

    log(f"Total facc corrections applied: {total_corrected:,}")
    return total_corrected
