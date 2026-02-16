"""
Flow direction correction for v17c pipeline.

Detects sections with wrong flow direction (via SWOT WSE slope validation)
and corrects high-confidence cases by flipping topology direction.

Confidence tiers use existing slope quality metrics:
- slope_obs_q bit flags from reach_swot_obs.py
- slope_obs_n_passes, n_obs, wse_obs_mean
"""

import json
from datetime import datetime as dt
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import duckdb
import networkx as nx
import numpy as np
import pandas as pd


def log(msg: str) -> None:
    print(f"[{dt.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def create_flow_corrections_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create v17c_flow_corrections provenance table."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS v17c_flow_corrections (
            run_id VARCHAR, region VARCHAR(2), section_id INTEGER,
            iteration INTEGER, tier VARCHAR(6), action VARCHAR(16),
            slope_from_upstream DOUBLE, slope_from_downstream DOUBLE,
            n_reaches_flipped INTEGER, reach_ids_flipped VARCHAR,
            created_at TIMESTAMP DEFAULT current_timestamp
        )
    """)


def snapshot_topology(conn: duckdb.DuckDBPyConnection, region: str, run_id: str) -> str:
    """Backup reach_topology for a region. Returns backup table name."""
    table_name = f"reach_topology_backup_{region}_{run_id.replace('-', '_')}"
    conn.execute(
        f'CREATE TABLE IF NOT EXISTS "{table_name}" AS '
        "SELECT * FROM reach_topology WHERE region = ?",
        [region.upper()],
    )
    n = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
    log(f"Topology snapshot: {n:,} rows -> {table_name}")
    return table_name


def rollback_flow_corrections(
    conn: duckdb.DuckDBPyConnection, region: str, run_id: str
) -> int:
    """Restore reach_topology from backup. Returns rows restored."""
    table_name = f"reach_topology_backup_{region}_{run_id.replace('-', '_')}"
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
    ]
    if table_name not in tables:
        backups = [t for t in tables if t.startswith("reach_topology_backup_")]
        raise ValueError(f"Backup '{table_name}' not found. Available: {backups}")
    conn.execute("DELETE FROM reach_topology WHERE region = ?", [region.upper()])
    conn.execute(f'INSERT INTO reach_topology SELECT * FROM "{table_name}"')
    n = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
    log(f"Rollback: restored {n:,} rows from {table_name}")
    return n


def _get_reach_quality(reach_ids: List[int], reaches_df: pd.DataFrame) -> Dict:
    """Gather quality metrics for reaches in a section."""
    cols = ["slope_obs_q", "slope_obs_n_passes", "n_obs", "wse_obs_mean", "lakeflag"]
    avail = [c for c in cols if c in reaches_df.columns]
    subset = reaches_df[reaches_df["reach_id"].isin(reach_ids)]

    n_wse = 0
    slope_q_vals = []
    n_passes_vals = []

    for _, row in subset.iterrows():
        if "wse_obs_mean" in avail and pd.notna(row.get("wse_obs_mean")):
            n_wse += 1
        sq = row.get("slope_obs_q")
        if sq is not None and pd.notna(sq):
            slope_q_vals.append(int(sq))
        np_val = row.get("slope_obs_n_passes")
        if np_val is not None and pd.notna(np_val):
            n_passes_vals.append(int(np_val))

    n_good = sum(1 for q in slope_q_vals if q == 0)
    frac_good = n_good / len(slope_q_vals) if slope_q_vals else 0
    med_passes = float(np.median(n_passes_vals)) if n_passes_vals else 0
    has_extreme = any(q & (4 | 8) for q in slope_q_vals)
    has_lake = False
    if "lakeflag" in avail:
        lf_vals = subset["lakeflag"].dropna()
        has_lake = (lf_vals > 0).any()

    return {
        "n_with_wse": n_wse,
        "frac_good_q": frac_good,
        "median_n_passes": med_passes,
        "has_extreme_flags": has_extreme,
        "has_lake": has_lake,
    }


def score_section_confidence(
    validation_row: Dict,
    G: nx.DiGraph,
    reaches_df: pd.DataFrame,
    reach_ids: List[int],
    min_wse_reaches: int = 2,
) -> Tuple[str, Dict]:
    """
    Score a section into HIGH / MEDIUM / LOW / SKIP confidence tier.

    HIGH: both slopes wrong, majority good quality, >=10 passes, >=3 WSE reaches
    MEDIUM: both slopes wrong, >=2 WSE reaches, no extreme flags
    LOW: single slope wrong, quality issues, or insufficient WSE
    SKIP: lake/reservoir, extreme data error, tidal, or already valid
    """
    likely_cause = validation_row.get("likely_cause")
    direction_valid = validation_row.get("direction_valid")
    slope_up = validation_row.get("slope_from_upstream")
    slope_dn = validation_row.get("slope_from_downstream")

    if direction_valid is True or direction_valid is None:
        return "SKIP", {"reason": "valid_or_undetermined"}

    if likely_cause in ("lake_section", "extreme_slope_data_error"):
        return "SKIP", {"reason": f"likely_cause={likely_cause}"}

    # Tidal check
    uj = validation_row.get("upstream_junction")
    if uj is not None and uj in G.nodes:
        if G.nodes[uj].get("lakeflag", 0) == 3:
            return "SKIP", {"reason": "tidal_section"}

    metrics = _get_reach_quality(reach_ids, reaches_df)
    upstream_wrong = pd.notna(slope_up) and slope_up > 0
    downstream_wrong = pd.notna(slope_dn) and slope_dn < 0
    both_wrong = upstream_wrong and downstream_wrong

    meta = {
        "upstream_wrong": upstream_wrong,
        "downstream_wrong": downstream_wrong,
        **metrics,
    }

    if metrics["n_with_wse"] < min_wse_reaches:
        return "LOW", {**meta, "reason": "insufficient_wse"}

    if both_wrong:
        if (
            metrics["frac_good_q"] > 0.5
            and metrics["median_n_passes"] >= 10
            and metrics["n_with_wse"] >= 3
            and not metrics["has_extreme_flags"]
        ):
            return "HIGH", {**meta, "reason": "both_wrong_high_quality"}
        if metrics["n_with_wse"] >= 2 and not metrics["has_extreme_flags"]:
            return "MEDIUM", {**meta, "reason": "both_wrong_moderate_quality"}

    return "LOW", {**meta, "reason": "single_wrong_or_quality_issues"}


def flip_section_topology(
    conn: duckdb.DuckDBPyConnection,
    region: str,
    reach_ids: List[int],
    upstream_junction: int,
    downstream_junction: int,
) -> int:
    """
    Flip direction='up'<->'down' for edges within a section.

    Only flips rows where both reach_id and neighbor_reach_id are in the
    section set (reaches + junctions), preserving external connections.
    """
    section_set = list(set(reach_ids) | {upstream_junction, downstream_junction})
    section_df = pd.DataFrame({"rid": section_set})
    conn.register("_flip_ids", section_df)
    result = conn.execute(
        """
        UPDATE reach_topology
        SET direction = CASE WHEN direction = 'up' THEN 'down' ELSE 'up' END
        WHERE region = ?
          AND reach_id IN (SELECT rid FROM _flip_ids)
          AND neighbor_reach_id IN (SELECT rid FROM _flip_ids)
    """,
        [region.upper()],
    )
    n = result.fetchone()[0]
    conn.unregister("_flip_ids")
    return n


def correct_flow_directions(
    conn: duckdb.DuckDBPyConnection,
    region: str,
    G: nx.DiGraph,
    sections_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    reaches_df: pd.DataFrame,
    max_iterations: int = 5,
    run_id: Optional[str] = None,
    rebuild_fn=None,
) -> Dict:
    """
    Iterative flow direction correction loop.

    Scores invalid sections, flips HIGH+MEDIUM, rebuilds graph, re-validates.
    Oscillation guard: sections flipped >=2 times are demoted to LOW.

    rebuild_fn(conn, region) -> (G, sections_df, validation_df) rebuilds
    after topology changes. If None, single-pass mode (no re-validation).
    """
    if run_id is None:
        run_id = uuid4().hex[:12]
    log(f"Flow direction correction: region={region}, run_id={run_id}")

    create_flow_corrections_table(conn)
    snapshot_topology(conn, region, run_id)

    flip_history: Dict[int, int] = {}
    total_flipped = 0
    manual_review = []
    cur_G, cur_sdf, cur_vdf = G, sections_df, validation_df
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        log(f"  Iteration {iteration}/{max_iterations}")
        if cur_vdf.empty:
            break

        invalid = cur_vdf[cur_vdf["direction_valid"] == False]  # noqa: E712
        if invalid.empty:
            log("  All sections valid, converged!")
            break

        sec_map = {}
        for _, r in cur_sdf.iterrows():
            sec_map[int(r["section_id"])] = {
                "reach_ids": r["reach_ids"],
                "uj": int(r["upstream_junction"]),
                "dj": int(r["downstream_junction"]),
            }

        to_flip = []
        log_rows = []
        for _, vrow in invalid.iterrows():
            sid = int(vrow["section_id"])
            si = sec_map.get(sid)
            if si is None:
                continue

            tier, meta = score_section_confidence(
                vrow.to_dict(), cur_G, reaches_df, si["reach_ids"]
            )
            if flip_history.get(sid, 0) >= 2:
                tier, meta["reason"] = "LOW", "oscillation_guard"

            if tier in ("HIGH", "MEDIUM"):
                to_flip.append((sid, tier, si))
            elif tier == "LOW":
                manual_review.append(
                    {
                        "section_id": sid,
                        "reason": meta.get("reason", ""),
                        "slope_up": vrow.get("slope_from_upstream"),
                        "slope_dn": vrow.get("slope_from_downstream"),
                    }
                )

            action = "flip" if tier in ("HIGH", "MEDIUM") else tier.lower()
            log_rows.append(
                {
                    "run_id": run_id,
                    "region": region.upper(),
                    "section_id": sid,
                    "iteration": iteration,
                    "tier": tier,
                    "action": action,
                    "slope_from_upstream": vrow.get("slope_from_upstream"),
                    "slope_from_downstream": vrow.get("slope_from_downstream"),
                    "n_reaches_flipped": len(si["reach_ids"])
                    if tier in ("HIGH", "MEDIUM")
                    else 0,
                    "reach_ids_flipped": json.dumps(si["reach_ids"])
                    if tier in ("HIGH", "MEDIUM")
                    else "[]",
                }
            )

        _write_log(conn, log_rows)

        if not to_flip:
            log("  No sections to flip, stopping")
            break

        log(f"  Flipping {len(to_flip)} sections")
        for sid, tier, si in to_flip:
            n = flip_section_topology(conn, region, si["reach_ids"], si["uj"], si["dj"])
            flip_history[sid] = flip_history.get(sid, 0) + 1
            total_flipped += 1
            log(f"    Section {sid} ({tier}): {n} rows flipped (#{flip_history[sid]})")

        if rebuild_fn is not None:
            cur_G, cur_sdf, cur_vdf = rebuild_fn(conn, region)
        else:
            break

    log(f"Done: {total_flipped} flipped, {len(manual_review)} manual review")
    return {
        "run_id": run_id,
        "region": region,
        "n_flipped": total_flipped,
        "n_manual_review": len(manual_review),
        "iterations": iteration,
        "manual_review": manual_review,
        "flip_history": flip_history,
    }


def _write_log(conn: duckdb.DuckDBPyConnection, rows: List[Dict]) -> None:
    """Write correction log rows to v17c_flow_corrections."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    conn.register("_corr_log", df)
    conn.execute("""
        INSERT INTO v17c_flow_corrections
            (run_id, region, section_id, iteration, tier, action,
             slope_from_upstream, slope_from_downstream,
             n_reaches_flipped, reach_ids_flipped)
        SELECT run_id, region, section_id, iteration, tier, action,
               slope_from_upstream, slope_from_downstream,
               n_reaches_flipped, reach_ids_flipped
        FROM _corr_log
    """)
    conn.unregister("_corr_log")
