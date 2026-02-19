"""Mainstem computation stage for v17c pipeline."""

from collections import defaultdict
from typing import Dict

import networkx as nx

from ._logging import log


def compute_mainstem(G: nx.DiGraph, hw_out_attrs: Dict[int, Dict]) -> Dict[int, bool]:
    """
    Compute is_mainstem for each reach.

    A reach is on the mainstem if it's on the path from best_headwater to best_outlet.
    """
    log("Computing mainstem classification...")

    is_mainstem = {n: False for n in G.nodes()}

    # Group by (best_headwater, best_outlet) pairs
    paths = defaultdict(list)
    for node, attrs in hw_out_attrs.items():
        key = (attrs["best_headwater"], attrs["best_outlet"])
        paths[key].append(node)

    # For each unique path, mark nodes on it as mainstem
    for (hw, out), nodes in paths.items():
        if hw is None or out is None:
            continue

        try:
            path = nx.shortest_path(G, hw, out)
            for n in path:
                is_mainstem[n] = True
        except nx.NetworkXNoPath:
            continue

    n_mainstem = sum(is_mainstem.values())
    n_total = len(G.nodes())
    pct = 100 * n_mainstem / n_total if n_total > 0 else 0
    log(f"Mainstem reaches: {n_mainstem:,} ({pct:.1f}%)")

    return is_mainstem


def compute_main_neighbors(G: nx.DiGraph) -> Dict[int, Dict]:
    """
    Compute rch_id_up_main and rch_id_dn_main for each reach.

    For each node, selects the main upstream predecessor and main downstream
    successor using the same (effective_width, log_facc) ranking used by
    best_headwater/best_outlet, ensuring consistent routing across all v17c
    columns.

    Returns dict {reach_id: {'rch_id_up_main': int|None, 'rch_id_dn_main': int|None}}.
    """
    log("Computing main neighbors (rch_id_up_main / rch_id_dn_main)...")

    results = {}

    for node in G.nodes():
        # Main upstream neighbor: pick best predecessor
        preds = list(G.predecessors(node))
        if preds:
            best_up = max(
                preds,
                key=lambda n: (
                    G.nodes[n].get("effective_width", 0) or 0,
                    G.nodes[n].get("log_facc", 0) or 0,
                ),
            )
            rch_id_up_main = best_up
        else:
            rch_id_up_main = None

        # Main downstream neighbor: pick best successor
        succs = list(G.successors(node))
        if succs:
            best_dn = max(
                succs,
                key=lambda n: (
                    G.nodes[n].get("effective_width", 0) or 0,
                    G.nodes[n].get("log_facc", 0) or 0,
                ),
            )
            rch_id_dn_main = best_dn
        else:
            rch_id_dn_main = None

        results[node] = {
            "rch_id_up_main": rch_id_up_main,
            "rch_id_dn_main": rch_id_dn_main,
        }

    n_with_up = sum(1 for v in results.values() if v["rch_id_up_main"] is not None)
    n_with_dn = sum(1 for v in results.values() if v["rch_id_dn_main"] is not None)
    log(f"Main neighbors: {n_with_up:,} with up_main, {n_with_dn:,} with dn_main")

    return results
