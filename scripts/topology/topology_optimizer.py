#!/usr/bin/env python3
"""
SWORD Topology Optimizer
------------------------

Experimental tool to optimize SWORD river network topology using phi-based
(distance-to-outlet) MILP optimization. Based on the sword_v17c pipeline.

Pipeline Steps:
1. Build graph from DuckDB (reaches + topology)
2. Classify nodes as Head_water, Outlet, Junction
3. Compute phi (distance to outlets) via Dijkstra on undirected graph
4. Solve MILP to orient ALL edges optimally:
   - Minimize "uphill" flow (going against phi gradient)
   - Hard constraints: Headwaters outgoing only, Outlets incoming only
   - Junction constraints: at least 1 in and 1 out
   - DAG constraints: acyclic (rank variables)
5. Compare new directions with current topology
6. Optionally apply changes

Usage:
    python topology_optimizer.py --region NA --dry-run
    python topology_optimizer.py --region NA --apply
    python topology_optimizer.py --region NA --method milp --output results.csv

Author: Jake Gearon + Claude
Based on: sword_v17c phi optimization by Gearon et al.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Hashable
from collections import defaultdict

import pandas as pd
import networkx as nx
import duckdb

# Optional: MILP solver
try:
    import pulp

    try:
        from pulp import HiGHS_CMD

        HIGHS_AVAILABLE = True
    except ImportError:
        HIGHS_AVAILABLE = False
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    HIGHS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# SWOT DATA LOADING
# -----------------------------------------------------------------------------


def load_swot_data(swot_path: str, region: str = None) -> Optional[pd.DataFrame]:
    """
    Load SWOT WSE data from parquet files.

    Args:
        swot_path: Path to SWOT parquet directory
        region: Optional region filter (NA, SA, EU, AF, AS, OC)

    Returns:
        DataFrame with reach_id and mean wse, or None if not available
    """
    swot_dir = Path(swot_path)
    reaches_dir = swot_dir / "reaches"

    if not reaches_dir.exists():
        logger.warning(f"SWOT reaches directory not found: {reaches_dir}")
        return None

    parquet_files = list(reaches_dir.glob("*.parquet"))
    if not parquet_files:
        logger.warning(f"No parquet files found in {reaches_dir}")
        return None

    logger.info(f"Loading SWOT data from {len(parquet_files)} parquet files...")

    dfs = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf, columns=["reach_id", "wse"])
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Error loading {pf}: {e}")

    if not dfs:
        return None

    swot_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(swot_df):,} SWOT observations")

    # Filter by region
    if region:
        region_prefixes = {
            "NA": "7",
            "SA": "6",
            "EU": "2",
            "AF": "3",
            "AS": "4",
            "OC": "5",
        }
        prefix = region_prefixes.get(region.upper())
        if prefix:
            swot_df = swot_df[swot_df["reach_id"].astype(str).str.startswith(prefix)]
            logger.info(
                f"Filtered to {len(swot_df):,} observations for region {region}"
            )

    # Filter fill values and compute mean
    swot_df = swot_df[(swot_df["wse"] > -1e9) & (swot_df["wse"] < 1e9)]
    if len(swot_df) == 0:
        return None

    swot_mean = (
        swot_df.groupby("reach_id").agg({"wse": ["mean", "std", "count"]}).reset_index()
    )
    swot_mean.columns = ["reach_id", "wse_mean", "wse_std", "wse_count"]
    swot_mean["wse"] = swot_mean["wse_mean"]

    logger.info(f"Computed mean WSE for {len(swot_mean):,} reaches")
    return swot_mean


# -----------------------------------------------------------------------------
# GRAPH BUILDING FROM DUCKDB
# -----------------------------------------------------------------------------


class SWORDGraphBuilder:
    """Build NetworkX graph from SWORD DuckDB database."""

    def __init__(self, db_path: str, region: str):
        self.db_path = db_path
        self.region = region.upper()
        self.conn = None

    def connect(self):
        self.conn = duckdb.connect(self.db_path, read_only=True)
        logger.info(f"Connected to {self.db_path}")

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def load_reaches(self) -> pd.DataFrame:
        """Load reach data from database."""
        query = """
            SELECT
                reach_id, river_name, reach_length, wse, width, facc,
                n_rch_up, n_rch_down, lakeflag, trib_flag, dist_out
            FROM reaches
            WHERE region = ?
        """
        df = self.conn.execute(query, [self.region]).fetchdf()
        logger.info(f"Loaded {len(df):,} reaches for region {self.region}")
        return df

    def load_topology(self) -> pd.DataFrame:
        """Load topology (reach connections) from database."""
        query = """
            SELECT
                reach_id, direction, neighbor_reach_id,
                topology_suspect, topology_approved
            FROM reach_topology
            WHERE region = ?
        """
        df = self.conn.execute(query, [self.region]).fetchdf()
        logger.info(f"Loaded {len(df):,} topology links")
        return df

    def load_reach_centroids(self) -> pd.DataFrame:
        """Load reach centroid coordinates from nodes."""
        query = """
            SELECT
                reach_id,
                AVG(x) as x,
                AVG(y) as y,
                AVG(facc) as avg_facc,
                MIN(dist_out) as min_dist_out
            FROM nodes
            WHERE region = ?
            GROUP BY reach_id
        """
        df = self.conn.execute(query, [self.region]).fetchdf()
        logger.info(f"Loaded centroids for {len(df):,} reaches")
        return df

    def build_undirected_graph(self) -> nx.Graph:
        """
        Build UNDIRECTED graph - the key insight from sword_v17c.

        We build undirected first, then use MILP to determine optimal directions.
        """
        reaches_df = self.load_reaches()
        topology_df = self.load_topology()
        centroids_df = self.load_reach_centroids()

        # Merge centroids
        reaches_df = reaches_df.merge(
            centroids_df[["reach_id", "x", "y", "avg_facc", "min_dist_out"]],
            on="reach_id",
            how="left",
        )

        # Create UNDIRECTED graph
        G = nx.Graph()

        # Add nodes (reaches)
        for _, row in reaches_df.iterrows():
            G.add_node(
                row["reach_id"],
                x=row["x"],
                y=row["y"],
                facc=row["avg_facc"] if pd.notna(row["avg_facc"]) else row["facc"],
                dist_out=row["min_dist_out"],
                reach_length=row["reach_length"],
                river_name=row["river_name"],
                wse=row["wse"],
                width=row["width"],
                lakeflag=row["lakeflag"],
                trib_flag=row["trib_flag"],
                n_rch_up=row["n_rch_up"],
                n_rch_down=row["n_rch_down"],
            )

        # Add undirected edges from topology
        # We use BOTH 'up' and 'down' directions to build connections
        for _, row in topology_df.iterrows():
            src = row["reach_id"]
            dst = row["neighbor_reach_id"]

            if src in G.nodes and dst in G.nodes:
                if not G.has_edge(src, dst):
                    src_len = G.nodes[src].get("reach_length", 1000) or 1000
                    dst_len = G.nodes[dst].get("reach_length", 1000) or 1000
                    distance = (src_len + dst_len) / 2

                    G.add_edge(src, dst, distance=distance)

        logger.info(
            f"Built undirected graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges"
        )
        return G

    def build_directed_graph(self) -> nx.DiGraph:
        """Build directed graph from current SWORD topology (for comparison)."""
        reaches_df = self.load_reaches()
        topology_df = self.load_topology()
        centroids_df = self.load_reach_centroids()

        reaches_df = reaches_df.merge(
            centroids_df[["reach_id", "x", "y", "avg_facc", "min_dist_out"]],
            on="reach_id",
            how="left",
        )

        G = nx.DiGraph()

        for _, row in reaches_df.iterrows():
            G.add_node(
                row["reach_id"],
                x=row["x"],
                y=row["y"],
                facc=row["avg_facc"] if pd.notna(row["avg_facc"]) else row["facc"],
                dist_out=row["min_dist_out"],
                reach_length=row["reach_length"],
            )

        # Only add downstream edges
        down_edges = topology_df[topology_df["direction"] == "down"]
        for _, row in down_edges.iterrows():
            src = row["reach_id"]
            dst = row["neighbor_reach_id"]
            if src in G.nodes and dst in G.nodes:
                src_len = G.nodes[src].get("reach_length", 1000) or 1000
                dst_len = G.nodes[dst].get("reach_length", 1000) or 1000
                G.add_edge(src, dst, distance=(src_len + dst_len) / 2)

        logger.info(
            f"Built directed graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges"
        )
        return G


# -----------------------------------------------------------------------------
# NODE CLASSIFICATION
# -----------------------------------------------------------------------------


def classify_nodes_from_undirected(G: nx.Graph) -> Dict[Hashable, str]:
    """
    Classify nodes based on degree in undirected graph.

    This is a heuristic - true classification depends on edge directions.
    - degree 1 -> potential Head_water or Outlet
    - degree > 2 -> Junction
    - degree 2 -> Connection (pass-through)

    We use additional info (facc, dist_out) to distinguish headwaters from outlets.
    """
    node_types = {}

    # Collect degree-1 nodes
    degree_1_nodes = [n for n in G.nodes() if G.degree(n) == 1]

    # Among degree-1 nodes, classify as headwater or outlet based on facc
    # Headwaters have LOW facc, Outlets have HIGH facc
    if degree_1_nodes:
        facc_values = [(n, G.nodes[n].get("facc", 0) or 0) for n in degree_1_nodes]
        facc_values.sort(key=lambda x: x[1])

        # Use median facc as threshold
        median_idx = len(facc_values) // 2
        median_facc = facc_values[median_idx][1] if facc_values else 0

        for n, facc in facc_values:
            if facc < median_facc:
                node_types[n] = "Head_water"
            else:
                node_types[n] = "Outlet"

    # Classify remaining nodes
    for n in G.nodes():
        if n in node_types:
            continue
        deg = G.degree(n)
        if deg == 0:
            node_types[n] = "Isolated"
        elif deg == 2:
            node_types[n] = "Connection"
        else:  # deg > 2
            node_types[n] = "Junction"

    # Count
    type_counts = defaultdict(int)
    for t in node_types.values():
        type_counts[t] += 1

    logger.info(f"Node classification: {dict(type_counts)}")

    return node_types


def classify_nodes_from_directed(G: nx.DiGraph) -> Dict[Hashable, str]:
    """
    Classify nodes based on in/out degree in directed graph.

    - in=0, out>0 -> Head_water
    - in>0, out=0 -> Outlet
    - in>0, out>0 -> Junction or Connection
    """
    node_types = {}

    for n in G.nodes():
        in_deg = G.in_degree(n)
        out_deg = G.out_degree(n)

        if in_deg == 0 and out_deg > 0:
            node_types[n] = "Head_water"
        elif out_deg == 0 and in_deg > 0:
            node_types[n] = "Outlet"
        elif in_deg > 0 and out_deg > 0:
            if in_deg > 1 or out_deg > 1:
                node_types[n] = "Junction"
            else:
                node_types[n] = "Connection"
        else:
            node_types[n] = "Isolated"

    type_counts = defaultdict(int)
    for t in node_types.values():
        type_counts[t] += 1

    logger.info(f"Node classification: {dict(type_counts)}")
    return node_types


# -----------------------------------------------------------------------------
# PHI COMPUTATION
# -----------------------------------------------------------------------------


def compute_phi(
    G: nx.Graph, outlets: List[Hashable], weight_attr: str = "distance"
) -> Dict[Hashable, float]:
    """
    Compute phi (distance to outlets) using multi-source Dijkstra on undirected graph.

    phi represents "distance to outlet" - higher phi = further from outlet.
    Water flows from high phi to low phi.
    """
    if not outlets:
        raise ValueError("No outlets provided for phi computation")

    # Validate edge weights
    for u, v, d in G.edges(data=True):
        w = d.get(weight_attr)
        if w is None or not isinstance(w, (int, float)) or w <= 0:
            # Set default weight
            G[u][v][weight_attr] = 1000

    try:
        phi = nx.multi_source_dijkstra_path_length(
            G, sources=outlets, weight=weight_attr
        )
    except Exception as e:
        logger.warning(f"Dijkstra failed: {e}. Using dist_out as fallback.")
        phi = {n: G.nodes[n].get("dist_out", 0) or 0 for n in G.nodes()}

    # Handle unreachable nodes
    for n in G.nodes():
        if n not in phi:
            phi[n] = float("inf")

    valid_phi = [v for v in phi.values() if v != float("inf")]
    if valid_phi:
        logger.info(f"Phi range: {min(valid_phi):.0f} - {max(valid_phi):.0f}")

    return phi


# -----------------------------------------------------------------------------
# MILP OPTIMIZATION (from sword_v17c phi_only_global.py)
# -----------------------------------------------------------------------------


def solve_milp_component(
    G_comp: nx.Graph,
    phi: Dict[Hashable, float],
    node_types: Dict[Hashable, str],
    prefer_highs: bool = True,
) -> Tuple[Dict[Tuple[str, str], int], str]:
    """Solve MILP for a single connected component."""
    # Normalize node IDs to strings
    idmap = {str(n): n for n in G_comp.nodes()}
    nodes_s = sorted(idmap.keys())

    if len(nodes_s) == 0:
        return {}, "Optimal"

    # Get undirected edge pairs
    undirected_pairs = []
    for u, v in G_comp.edges():
        a, b = str(u), str(v)
        if a > b:
            a, b = b, a
        if (a, b) not in undirected_pairs:
            undirected_pairs.append((a, b))
    undirected_pairs = sorted(set(undirected_pairs))

    if len(undirected_pairs) == 0:
        return {}, "Optimal"

    # Phi with big value for missing
    finite_phis = [v for v in phi.values() if v != float("inf") and v is not None]
    big_phi = (max(finite_phis) if finite_phis else 0) + 1e9
    phi_s = {
        str(n): (phi.get(n, big_phi) if phi.get(n) != float("inf") else big_phi)
        for n in G_comp.nodes()
    }

    # Compute uphill costs
    cost_uv = {}
    for a, b in undirected_pairs:
        du, dv = phi_s.get(a, big_phi), phi_s.get(b, big_phi)
        cost_a_to_b = max(0.0, dv - du)
        cost_b_to_a = max(0.0, du - dv)
        cost_uv[(a, b)] = (cost_a_to_b, cost_b_to_a)

    # Build neighborhoods
    nbrs = {n: set() for n in nodes_s}
    for a, b in undirected_pairs:
        nbrs[a].add(b)
        nbrs[b].add(a)

    # Node types
    ntype = {str(n): node_types.get(n, "Junction") for n in G_comp.nodes()}

    N = len(nodes_s)
    M = N

    # Build MILP
    m = pulp.LpProblem("phi_orientation", pulp.LpMinimize)
    x = {
        (a, b): pulp.LpVariable(f"x_{a}_{b}", cat="Binary")
        for (a, b) in undirected_pairs
    }
    r = {
        n: pulp.LpVariable(f"r_{n}", lowBound=0, upBound=N - 1, cat="Integer")
        for n in nodes_s
    }

    # Objective
    m += pulp.lpSum(
        x[(a, b)] * cost_uv[(a, b)][0] + (1 - x[(a, b)]) * cost_uv[(a, b)][1]
        for (a, b) in undirected_pairs
    )

    # DAG constraints
    for a, b in undirected_pairs:
        m += r[a] - r[b] >= 1 - M * (1 - x[(a, b)])
        m += r[b] - r[a] >= 1 - M * x[(a, b)]

    # Hydrologic rules
    for n in nodes_s:
        for w in nbrs[n]:
            a, b = (n, w) if n < w else (w, n)
            if (a, b) not in x:
                continue
            var = x[(a, b)]

            if ntype[n] == "Head_water":
                if n < w:
                    m += var == 1
                else:
                    m += var == 0
            elif ntype[n] == "Outlet":
                if n < w:
                    m += var == 0
                else:
                    m += var == 1

    # Junction constraints
    for n in nodes_s:
        if ntype[n] != "Junction":
            continue
        inc_terms, out_terms = [], []
        for w in nbrs[n]:
            a, b = (n, w) if n < w else (w, n)
            if (a, b) not in x:
                continue
            var = x[(a, b)]
            if n < w:
                if ntype[w] != "Outlet":
                    inc_terms.append(1 - var)
                if ntype[w] != "Head_water":
                    out_terms.append(var)
            else:
                if ntype[w] != "Outlet":
                    inc_terms.append(var)
                if ntype[w] != "Head_water":
                    out_terms.append(1 - var)
        if inc_terms:
            m += pulp.lpSum(inc_terms) >= 1
        if out_terms:
            m += pulp.lpSum(out_terms) >= 1

    # Solve
    if prefer_highs and HIGHS_AVAILABLE:
        solver = HiGHS_CMD(msg=False)
    else:
        solver = pulp.PULP_CBC_CMD(msg=False)

    status_code = m.solve(solver)
    status = pulp.LpStatus[status_code]

    edge_dir = {}
    for a, b in undirected_pairs:
        val = pulp.value(x[(a, b)])
        val = 0 if val is None else int(round(val))
        edge_dir[(a, b)] = val

    return edge_dir, status


def solve_milp_orientation(
    G: nx.Graph,
    phi: Dict[Hashable, float],
    node_types: Dict[Hashable, str],
    prefer_highs: bool = True,
    by_component: bool = True,
) -> Tuple[Dict[Tuple[str, str], int], str]:
    """
    Solve MILP to determine optimal edge directions.

    Objective: Minimize uphill flow (edges going from low phi to high phi)

    Hard constraints:
    - Head_water: all edges must point OUT (no incoming)
    - Outlet: all edges must point IN (no outgoing)
    - Junction: at least 1 incoming AND 1 outgoing
    - DAG: no cycles (enforced via rank variables)

    Args:
        by_component: If True, solve each connected component separately (more robust)

    Returns:
        edge_dir: Dict mapping (a, b) -> 1 if a->b, 0 if b->a
        status: Solver status string
    """
    if not PULP_AVAILABLE:
        raise RuntimeError("PuLP not available. Install with: pip install pulp")

    # Component-wise solving for robustness (like sword_v17c)
    if by_component:
        components = list(nx.connected_components(G))
        logger.info(f"Solving {len(components):,} connected components separately")

        all_edge_dir = {}
        all_status = []

        for i, comp_nodes in enumerate(components):
            if len(comp_nodes) < 2:
                continue  # Skip isolated nodes

            G_comp = G.subgraph(comp_nodes).copy()
            edge_dir_comp, status_comp = solve_milp_component(
                G_comp, phi, node_types, prefer_highs
            )
            all_edge_dir.update(edge_dir_comp)
            all_status.append(status_comp)

            if (i + 1) % 100 == 0:
                logger.info(f"  Solved {i + 1}/{len(components)} components")

        # Overall status
        if all(s == "Optimal" for s in all_status):
            status = "Optimal"
        elif any(s == "Optimal" for s in all_status):
            status = "Partial"
        else:
            status = "Infeasible"

        logger.info(f"Component-wise solving complete: {len(all_edge_dir):,} edges")
        return all_edge_dir, status

    # Original monolithic approach
    # Normalize node IDs to strings
    idmap = {str(n): n for n in G.nodes()}
    nodes_s = sorted(idmap.keys())

    # Get undirected edge pairs (sorted by string)
    undirected_pairs = []
    for u, v in G.edges():
        a, b = str(u), str(v)
        if a > b:
            a, b = b, a
        if (a, b) not in undirected_pairs:
            undirected_pairs.append((a, b))
    undirected_pairs = sorted(set(undirected_pairs))

    logger.info(f"Optimizing {len(undirected_pairs):,} edge pairs (monolithic)")

    # Phi with big value for missing
    finite_phis = [v for v in phi.values() if v != float("inf") and v is not None]
    big_phi = (max(finite_phis) if finite_phis else 0) + 1e9
    phi_s = {
        str(n): (phi.get(n, big_phi) if phi.get(n) != float("inf") else big_phi)
        for n in G.nodes()
    }

    # Compute uphill costs
    cost_uv = {}
    for a, b in undirected_pairs:
        du, dv = phi_s.get(a, big_phi), phi_s.get(b, big_phi)
        cost_a_to_b = max(0.0, dv - du)  # Cost if a->b (uphill if phi increases)
        cost_b_to_a = max(0.0, du - dv)
        cost_uv[(a, b)] = (cost_a_to_b, cost_b_to_a)

    # Build neighborhoods
    nbrs = {n: set() for n in nodes_s}
    for a, b in undirected_pairs:
        nbrs[a].add(b)
        nbrs[b].add(a)

    # Node types by string ID
    ntype = {str(n): node_types.get(n, "Junction") for n in G.nodes()}

    N = len(nodes_s)
    M = N  # Big-M for rank constraints

    # Build MILP
    m = pulp.LpProblem("phi_orientation", pulp.LpMinimize)

    # Decision variables: x[(a,b)] = 1 means a->b, 0 means b->a
    x = {
        (a, b): pulp.LpVariable(f"x_{a}_{b}", cat="Binary")
        for (a, b) in undirected_pairs
    }

    # Rank variables for DAG constraints
    r = {
        n: pulp.LpVariable(f"r_{n}", lowBound=0, upBound=N - 1, cat="Integer")
        for n in nodes_s
    }

    # Objective: minimize uphill cost
    m += pulp.lpSum(
        x[(a, b)] * cost_uv[(a, b)][0] + (1 - x[(a, b)]) * cost_uv[(a, b)][1]
        for (a, b) in undirected_pairs
    )

    # DAG constraints: chosen direction must increase rank
    for a, b in undirected_pairs:
        # If x[(a,b)] == 1 (a->b), then r[a] >= r[b] + 1
        m += r[a] - r[b] >= 1 - M * (1 - x[(a, b)])
        # If x[(a,b)] == 0 (b->a), then r[b] >= r[a] + 1
        m += r[b] - r[a] >= 1 - M * x[(a, b)]

    # Hard hydrologic rules
    for n in nodes_s:
        for w in nbrs[n]:
            a, b = (n, w) if n < w else (w, n)
            if (a, b) not in x:
                continue
            var = x[(a, b)]

            if ntype[n] == "Head_water":
                # Force n -> w (all outgoing)
                if n < w:
                    m += var == 1
                else:
                    m += var == 0

            elif ntype[n] == "Outlet":
                # Force w -> n (all incoming)
                if n < w:
                    m += var == 0
                else:
                    m += var == 1

    # Junction constraints: at least 1 incoming AND 1 outgoing
    for n in nodes_s:
        if ntype[n] != "Junction":
            continue

        inc_terms = []
        out_terms = []

        for w in nbrs[n]:
            a, b = (n, w) if n < w else (w, n)
            if (a, b) not in x:
                continue
            var = x[(a, b)]

            if n < w:
                # var=1 means n->w (outgoing from n)
                # var=0 means w->n (incoming to n)
                if ntype[w] != "Outlet":
                    inc_terms.append(1 - var)  # w->n
                if ntype[w] != "Head_water":
                    out_terms.append(var)  # n->w
            else:
                # var=1 means w->n (incoming to n)
                # var=0 means n->w (outgoing from n)
                if ntype[w] != "Outlet":
                    inc_terms.append(var)  # w->n
                if ntype[w] != "Head_water":
                    out_terms.append(1 - var)  # n->w

        if len(inc_terms) > 0:
            m += pulp.lpSum(inc_terms) >= 1
        if len(out_terms) > 0:
            m += pulp.lpSum(out_terms) >= 1

    # Solve
    if prefer_highs and HIGHS_AVAILABLE:
        solver = HiGHS_CMD(msg=False)
    else:
        solver = pulp.PULP_CBC_CMD(msg=False)

    logger.info("Solving MILP...")
    status_code = m.solve(solver)
    status = pulp.LpStatus[status_code]
    logger.info(f"Solver status: {status}")

    # Extract solution
    edge_dir = {}
    for a, b in undirected_pairs:
        val = pulp.value(x[(a, b)])
        val = 0 if val is None else int(round(val))
        edge_dir[(a, b)] = val

    return edge_dir, status


# -----------------------------------------------------------------------------
# COMPARISON WITH CURRENT TOPOLOGY
# -----------------------------------------------------------------------------


def compare_directions(
    G_current: nx.DiGraph,
    edge_dir: Dict[Tuple[str, str], int],
) -> Dict[str, Any]:
    """
    Compare MILP-optimized directions with current SWORD topology.

    Returns:
        Dict with comparison statistics and list of edges to flip
    """
    current_edges = set(G_current.edges())

    confirmed = []
    to_flip = []
    new_edges = []

    for (a, b), val in edge_dir.items():
        src_s, dst_s = (a, b) if val == 1 else (b, a)
        try:
            src, dst = int(src_s), int(dst_s)
        except:
            src, dst = src_s, dst_s

        if (src, dst) in current_edges:
            confirmed.append((src, dst))
        elif (dst, src) in current_edges:
            to_flip.append(
                {
                    "current_upstream": dst,
                    "current_downstream": src,
                    "new_upstream": src,
                    "new_downstream": dst,
                }
            )
        else:
            new_edges.append((src, dst))

    stats = {
        "total_edges": len(edge_dir),
        "confirmed": len(confirmed),
        "to_flip": len(to_flip),
        "new_edges": len(new_edges),
        "pct_confirmed": len(confirmed) / len(edge_dir) * 100 if edge_dir else 0,
        "pct_to_flip": len(to_flip) / len(edge_dir) * 100 if edge_dir else 0,
        "flip_list": to_flip,
    }

    logger.info(
        f"Comparison: {stats['confirmed']:,} confirmed, "
        f"{stats['to_flip']:,} to flip ({stats['pct_to_flip']:.2f}%)"
    )

    return stats


# -----------------------------------------------------------------------------
# VALIDATION SUITE - Automated pass/fail tests
# -----------------------------------------------------------------------------


class TopologyValidator:
    """
    Validate topology optimization results with automated pass/fail tests.

    Tests:
    1. DAG (no cycles)
    2. Headwater constraints (no incoming edges)
    3. Outlet constraints (no outgoing edges)
    4. Facc consistency (increases downstream, except distributaries)
    5. Phi consistency (decreases downstream)
    6. SWOT WSE consistency (decreases downstream where data exists)
    """

    def __init__(
        self,
        G: nx.DiGraph,
        phi: Dict[Hashable, float] = None,
        swot_data: Optional[pd.DataFrame] = None,
    ):
        self.G = G
        self.phi = phi or {}
        self.swot_data = swot_data
        self.results = {}

    def run_all_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run all validation tests and return results."""
        tests = [
            ("dag_test", self.test_dag),
            ("headwater_test", self.test_headwater_constraints),
            ("outlet_test", self.test_outlet_constraints),
            ("facc_test", self.test_facc_consistency),
            ("phi_test", self.test_phi_consistency),
        ]

        if self.swot_data is not None:
            tests.append(("swot_test", self.test_swot_consistency))

        all_passed = True
        for name, test_func in tests:
            result = test_func()
            self.results[name] = result
            if not result["passed"]:
                all_passed = False
            if verbose:
                status = "✅ PASS" if result["passed"] else "❌ FAIL"
                print(f"{status} {name}: {result['message']}")

        self.results["all_passed"] = all_passed
        return self.results

    def test_dag(self) -> Dict[str, Any]:
        """Test that graph is a DAG (no cycles)."""
        try:
            cycles = list(nx.simple_cycles(self.G))
            if len(cycles) == 0:
                return {"passed": True, "message": "No cycles found", "cycles": 0}
            else:
                return {
                    "passed": False,
                    "message": f"Found {len(cycles)} cycles",
                    "cycles": len(cycles),
                    "sample_cycles": cycles[:5],
                }
        except Exception:
            # For large graphs, use is_directed_acyclic_graph
            is_dag = nx.is_directed_acyclic_graph(self.G)
            if is_dag:
                return {"passed": True, "message": "Graph is a DAG", "cycles": 0}
            else:
                return {
                    "passed": False,
                    "message": "Graph contains cycles",
                    "cycles": -1,
                }

    def test_headwater_constraints(self) -> Dict[str, Any]:
        """Test that headwaters have no incoming edges."""
        violations = []
        headwaters = [
            n
            for n in self.G.nodes()
            if self.G.in_degree(n) == 0 and self.G.out_degree(n) > 0
        ]

        # Check nodes marked as headwater but have incoming
        for n, d in self.G.nodes(data=True):
            if d.get("node_type") == "Head_water":
                in_deg = self.G.in_degree(n)
                if in_deg > 0:
                    violations.append({"node": n, "in_degree": in_deg})

        if len(violations) == 0:
            return {
                "passed": True,
                "message": f"{len(headwaters)} headwaters validated",
                "headwater_count": len(headwaters),
            }
        else:
            return {
                "passed": False,
                "message": f"{len(violations)} headwaters have incoming edges",
                "violations": violations[:10],
            }

    def test_outlet_constraints(self) -> Dict[str, Any]:
        """Test that outlets have no outgoing edges."""
        violations = []
        outlets = [
            n
            for n in self.G.nodes()
            if self.G.out_degree(n) == 0 and self.G.in_degree(n) > 0
        ]

        for n, d in self.G.nodes(data=True):
            if d.get("node_type") == "Outlet":
                out_deg = self.G.out_degree(n)
                if out_deg > 0:
                    violations.append({"node": n, "out_degree": out_deg})

        if len(violations) == 0:
            return {
                "passed": True,
                "message": f"{len(outlets)} outlets validated",
                "outlet_count": len(outlets),
            }
        else:
            return {
                "passed": False,
                "message": f"{len(violations)} outlets have outgoing edges",
                "violations": violations[:10],
            }

    def test_facc_consistency(self, threshold: float = 10.0) -> Dict[str, Any]:
        """
        Test facc consistency: should increase downstream (except distributaries).

        Args:
            threshold: Ratio threshold for violation (upstream/downstream > threshold)
        """
        violations = []
        distributary_exceptions = 0
        total_edges = 0

        for u, v in self.G.edges():
            total_edges += 1
            u_data = self.G.nodes[u]
            v_data = self.G.nodes[v]

            u_facc = u_data.get("facc", 0) or 0
            v_facc = v_data.get("facc", 0) or 0

            if u_facc > 0 and v_facc > 0:
                ratio = u_facc / v_facc
                if ratio > threshold:
                    # Check if distributary (valid exception)
                    n_rch_down = u_data.get("n_rch_down", 0) or 0
                    trib_flag = u_data.get("trib_flag", 0) or 0

                    if n_rch_down > 1 or trib_flag >= 2:
                        distributary_exceptions += 1
                    else:
                        violations.append(
                            {
                                "upstream": u,
                                "downstream": v,
                                "upstream_facc": u_facc,
                                "downstream_facc": v_facc,
                                "ratio": ratio,
                            }
                        )

        # Allow up to 1% violations as acceptable
        violation_pct = len(violations) / total_edges * 100 if total_edges > 0 else 0
        passed = violation_pct < 1.0

        return {
            "passed": passed,
            "message": f"{len(violations)} facc violations ({violation_pct:.2f}%), {distributary_exceptions} distributary exceptions",
            "violations": len(violations),
            "violation_pct": violation_pct,
            "distributary_exceptions": distributary_exceptions,
            "sample_violations": sorted(violations, key=lambda x: -x["ratio"])[:10],
        }

    def test_phi_consistency(self) -> Dict[str, Any]:
        """Test phi consistency: should decrease downstream (towards outlets)."""
        if not self.phi:
            return {
                "passed": True,
                "message": "No phi data to validate",
                "skipped": True,
            }

        violations = []
        total_edges = 0

        for u, v in self.G.edges():
            total_edges += 1
            u_phi = self.phi.get(u, float("inf"))
            v_phi = self.phi.get(v, float("inf"))

            if u_phi != float("inf") and v_phi != float("inf"):
                # Phi should decrease downstream (u_phi > v_phi)
                if v_phi > u_phi:
                    violations.append(
                        {
                            "upstream": u,
                            "downstream": v,
                            "upstream_phi": u_phi,
                            "downstream_phi": v_phi,
                            "uphill_amount": v_phi - u_phi,
                        }
                    )

        violation_pct = len(violations) / total_edges * 100 if total_edges > 0 else 0
        # Phi violations are more acceptable since MILP minimizes but doesn't eliminate
        passed = violation_pct < 5.0

        return {
            "passed": passed,
            "message": f"{len(violations)} phi violations ({violation_pct:.2f}%)",
            "violations": len(violations),
            "violation_pct": violation_pct,
            "sample_violations": sorted(violations, key=lambda x: -x["uphill_amount"])[
                :10
            ],
        }

    def test_swot_consistency(self) -> Dict[str, Any]:
        """Test SWOT WSE consistency: should decrease downstream."""
        if self.swot_data is None:
            return {
                "passed": True,
                "message": "No SWOT data to validate",
                "skipped": True,
            }

        violations = []
        validated = 0

        # Convert reach_id to int for consistent lookup
        reach_ids = self.swot_data["reach_id"]
        if reach_ids.dtype == "object":
            reach_ids = reach_ids.astype(int)
        swot_dict = dict(zip(reach_ids, self.swot_data["wse"]))

        for u, v in self.G.edges():
            u_wse = swot_dict.get(u)
            v_wse = swot_dict.get(v)

            if u_wse is not None and v_wse is not None:
                validated += 1
                # WSE should decrease downstream (u_wse > v_wse)
                if v_wse > u_wse + 0.5:  # 0.5m tolerance
                    violations.append(
                        {
                            "upstream": u,
                            "downstream": v,
                            "upstream_wse": u_wse,
                            "downstream_wse": v_wse,
                            "uphill_amount": v_wse - u_wse,
                        }
                    )

        if validated == 0:
            return {
                "passed": True,
                "message": "No edges with SWOT coverage",
                "skipped": True,
            }

        violation_pct = len(violations) / validated * 100
        passed = violation_pct < 10.0  # Allow up to 10% due to SWOT measurement noise

        return {
            "passed": passed,
            "message": f"{len(violations)}/{validated} SWOT violations ({violation_pct:.2f}%)",
            "violations": len(violations),
            "validated": validated,
            "violation_pct": violation_pct,
            "sample_violations": sorted(violations, key=lambda x: -x["uphill_amount"])[
                :10
            ],
        }

    def get_summary(self) -> str:
        """Get human-readable summary of validation results."""
        if not self.results:
            return "No validation results. Run run_all_tests() first."

        lines = ["=" * 50, "VALIDATION SUMMARY", "=" * 50]

        for name, result in self.results.items():
            if name == "all_passed":
                continue
            status = "✅" if result.get("passed", False) else "❌"
            lines.append(f"{status} {name}: {result.get('message', 'Unknown')}")

        lines.append("=" * 50)
        overall = (
            "✅ ALL TESTS PASSED"
            if self.results.get("all_passed")
            else "❌ SOME TESTS FAILED"
        )
        lines.append(overall)

        return "\n".join(lines)


def build_optimized_graph(
    G_undirected: nx.Graph,
    edge_dir: Dict[Tuple[str, str], int],
    phi: Dict[Hashable, float],
) -> nx.DiGraph:
    """Build directed graph from MILP solution."""
    D = nx.DiGraph()

    # Copy nodes with attributes
    for n, d in G_undirected.nodes(data=True):
        D.add_node(n, **d)
        D.nodes[n]["phi"] = phi.get(n, float("inf"))

    # Add directed edges from solution
    for (a, b), val in edge_dir.items():
        src_s, dst_s = (a, b) if val == 1 else (b, a)
        try:
            src, dst = int(src_s), int(dst_s)
        except:
            src, dst = src_s, dst_s

        if src in D.nodes and dst in D.nodes:
            # Copy edge attributes from undirected graph
            edge_data = (
                G_undirected.get_edge_data(src, dst)
                or G_undirected.get_edge_data(dst, src)
                or {}
            )
            D.add_edge(src, dst, **edge_data)

    return D


# -----------------------------------------------------------------------------
# SWOT SLOPE REFINEMENT
# -----------------------------------------------------------------------------


class SWOTRefinement:
    """
    Refine MILP results using SWOT WSE data.

    Uses SWOT observations to:
    1. Validate MILP edge directions
    2. Override MILP where SWOT data is confident
    3. Flag uncertain edges for review

    Confidence levels:
    - 'R' (Reliable): Large WSE diff (>wse_threshold) OR WSE+facc agree
    - 'U' (Uncertain): Small WSE diff, no facc support
    - 'N' (No data): Missing SWOT coverage
    """

    def __init__(
        self,
        G_undirected: nx.Graph,
        edge_dir: Dict[Tuple[str, str], int],
        swot_data: pd.DataFrame,
        wse_threshold: float = 2.0,  # meters - WSE difference threshold for high confidence
        count_threshold: int = 3,  # minimum SWOT observations for confidence
        require_facc_support: bool = True,  # require facc agreement for weak SWOT signals
    ):
        self.G = G_undirected
        self.edge_dir = edge_dir
        self.require_facc_support = require_facc_support
        # Convert reach_id to int for consistent lookup
        reach_ids = (
            swot_data["reach_id"].astype(int)
            if swot_data["reach_id"].dtype == "object"
            else swot_data["reach_id"]
        )
        self.swot_dict = dict(zip(reach_ids, swot_data["wse"]))
        # Handle wse_count column which may or may not exist
        if "wse_count" in swot_data.columns:
            self.swot_counts = dict(zip(reach_ids, swot_data["wse_count"]))
        else:
            # Default to assuming sufficient observations
            self.swot_counts = {r: 10 for r in reach_ids}
        self.wse_threshold = wse_threshold
        self.count_threshold = count_threshold
        logger.info(f"SWOT data loaded: {len(self.swot_dict):,} reaches with WSE")

    def compute_edge_confidence(self) -> Dict[Tuple[str, str], Dict]:
        """
        Compute confidence for each edge based on SWOT WSE and facc.

        Returns dict with:
        - swot_direction: 1 if a->b, 0 if b->a, None if unknown
        - confidence: 'R' (reliable), 'U' (uncertain), 'N' (no data)
        - wse_diff: WSE difference
        - facc_supports: True if facc agrees with SWOT direction
        """
        edge_confidence = {}

        for (a, b), milp_dir in self.edge_dir.items():
            try:
                a_int, b_int = int(a), int(b)
            except:
                a_int, b_int = a, b

            wse_a = self.swot_dict.get(a_int)
            wse_b = self.swot_dict.get(b_int)
            count_a = self.swot_counts.get(a_int, 0)
            count_b = self.swot_counts.get(b_int, 0)

            # Get facc values for cross-validation
            facc_a = self.G.nodes[a_int].get("facc", 0) if a_int in self.G.nodes else 0
            facc_b = self.G.nodes[b_int].get("facc", 0) if b_int in self.G.nodes else 0
            facc_a = facc_a or 0
            facc_b = facc_b or 0

            if wse_a is None or wse_b is None:
                edge_confidence[(a, b)] = {
                    "swot_direction": None,
                    "confidence": "N",  # No data
                    "wse_diff": None,
                    "milp_direction": milp_dir,
                    "facc_supports": None,
                }
                continue

            wse_diff = (
                wse_a - wse_b
            )  # positive means a is higher (a -> b is downstream)

            # SWOT-suggested direction: water flows from high WSE to low WSE
            swot_dir = 1 if wse_diff > 0 else 0  # 1 = a->b, 0 = b->a

            # facc-suggested direction: water flows from low facc to high facc
            # If swot says a->b (a upstream), then facc_a should be < facc_b
            if facc_a > 0 and facc_b > 0:
                facc_suggests_a_to_b = facc_a < facc_b
                facc_supports = (swot_dir == 1 and facc_suggests_a_to_b) or (
                    swot_dir == 0 and not facc_suggests_a_to_b
                )
            else:
                facc_supports = None  # Can't determine

            # Determine confidence level
            has_enough_obs = (
                count_a >= self.count_threshold and count_b >= self.count_threshold
            )
            strong_wse_signal = abs(wse_diff) > self.wse_threshold

            if has_enough_obs:
                if strong_wse_signal:
                    # Strong SWOT signal - reliable on its own
                    confidence = "R"
                elif facc_supports and self.require_facc_support:
                    # Weak SWOT but facc agrees - reliable together
                    confidence = "R"
                elif abs(wse_diff) > 0.5:  # At least some signal
                    confidence = "U"  # Uncertain
                else:
                    confidence = "U"  # Too weak
            else:
                confidence = "U"  # Not enough observations

            edge_confidence[(a, b)] = {
                "swot_direction": swot_dir,
                "confidence": confidence,
                "wse_diff": wse_diff,
                "wse_a": wse_a,
                "wse_b": wse_b,
                "facc_a": facc_a,
                "facc_b": facc_b,
                "facc_supports": facc_supports,
                "milp_direction": milp_dir,
                "agrees_with_milp": swot_dir == milp_dir,
            }

        return edge_confidence

    def refine_directions(
        self,
        override_milp: bool = True,
    ) -> Tuple[Dict[Tuple[str, str], int], Dict[str, Any]]:
        """
        Refine MILP directions using SWOT confidence.

        Args:
            override_milp: If True, override MILP with SWOT where confident

        Returns:
            refined_edge_dir: Updated edge directions
            stats: Refinement statistics
        """
        edge_conf = self.compute_edge_confidence()

        refined = self.edge_dir.copy()
        stats = {
            "total_edges": len(self.edge_dir),
            "swot_coverage": 0,
            "reliable_edges": 0,
            "uncertain_edges": 0,
            "no_data_edges": 0,
            "milp_overridden": 0,
            "milp_confirmed": 0,
            "milp_disagrees": 0,
            "overrides": [],
        }

        for (a, b), conf in edge_conf.items():
            if conf["confidence"] == "R":
                stats["reliable_edges"] += 1
                stats["swot_coverage"] += 1

                if conf["agrees_with_milp"]:
                    stats["milp_confirmed"] += 1
                else:
                    stats["milp_disagrees"] += 1
                    if override_milp and conf["swot_direction"] is not None:
                        refined[(a, b)] = conf["swot_direction"]
                        stats["milp_overridden"] += 1
                        stats["overrides"].append(
                            {
                                "edge": (a, b),
                                "milp": conf["milp_direction"],
                                "swot": conf["swot_direction"],
                                "wse_diff": conf["wse_diff"],
                                "facc_supports": conf.get("facc_supports"),
                            }
                        )

            elif conf["confidence"] == "U":
                stats["uncertain_edges"] += 1
                stats["swot_coverage"] += 1
            else:
                stats["no_data_edges"] += 1

        # Count facc-supported overrides
        facc_supported_overrides = sum(
            1 for o in stats["overrides"] if o.get("facc_supports")
        )
        stats["facc_supported_overrides"] = facc_supported_overrides

        logger.info(
            f"SWOT refinement: {stats['milp_overridden']} overrides "
            f"({facc_supported_overrides} with facc support), "
            f"{stats['milp_confirmed']} confirmed, "
            f"{stats['reliable_edges']} reliable edges"
        )

        return refined, stats


# -----------------------------------------------------------------------------
# APPLY CHANGES TO DATABASE
# -----------------------------------------------------------------------------


class TopologyUpdater:
    """Apply topology changes to the SWORD DuckDB database."""

    def __init__(self, db_path: str, region: str):
        self.db_path = db_path
        self.region = region.upper()
        self.conn = None

    def connect(self, read_only: bool = False):
        self.conn = duckdb.connect(self.db_path, read_only=read_only)
        logger.info(
            f"Connected to {self.db_path} ({'read-only' if read_only else 'read-write'})"
        )

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        self.connect(read_only=False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def apply_flips(
        self,
        flip_list: List[Dict],
        dry_run: bool = True,
        reason: str = "MILP topology optimization",
    ) -> Dict[str, Any]:
        """
        Apply edge direction flips to the database.

        Args:
            flip_list: List of dicts with current_upstream, current_downstream,
                       new_upstream, new_downstream
            dry_run: If True, only show what would be done
            reason: Reason for the changes (for logging)

        Returns:
            Summary dict with counts and any errors
        """
        if not flip_list:
            return {"applied": 0, "errors": 0, "message": "No flips to apply"}

        results = {
            "total": len(flip_list),
            "applied": 0,
            "skipped": 0,
            "errors": 0,
            "error_details": [],
        }

        if dry_run:
            logger.info(f"DRY RUN: Would apply {len(flip_list)} direction flips")
            for flip in flip_list[:5]:
                logger.info(
                    f"  Would flip: {flip['current_upstream']} -> {flip['current_downstream']}"
                )
                logger.info(
                    f"          to: {flip['new_upstream']} -> {flip['new_downstream']}"
                )
            if len(flip_list) > 5:
                logger.info(f"  ... and {len(flip_list) - 5} more")
            results["applied"] = 0
            results["message"] = f"DRY RUN: {len(flip_list)} flips would be applied"
            return results

        # Apply changes in a transaction
        try:
            self.conn.execute("BEGIN TRANSACTION")

            for flip in flip_list:
                old_up = flip["current_upstream"]
                old_down = flip["current_downstream"]
                new_up = flip["new_upstream"]
                new_down = flip["new_downstream"]

                try:
                    # Update the 'down' record: old_up had 'down' to old_down
                    # Change to: old_up has 'up' from old_down (which is new_down)
                    self.conn.execute(
                        """
                        UPDATE reach_topology
                        SET direction = 'up'
                        WHERE region = ?
                          AND reach_id = ?
                          AND direction = 'down'
                          AND neighbor_reach_id = ?
                    """,
                        [self.region, old_up, old_down],
                    )

                    # Update the 'up' record: old_down had 'up' from old_up
                    # Change to: old_down has 'down' to old_up (which is new_down has down to new_up)
                    self.conn.execute(
                        """
                        UPDATE reach_topology
                        SET direction = 'down'
                        WHERE region = ?
                          AND reach_id = ?
                          AND direction = 'up'
                          AND neighbor_reach_id = ?
                    """,
                        [self.region, old_down, old_up],
                    )

                    results["applied"] += 1

                except Exception as e:
                    results["errors"] += 1
                    results["error_details"].append({"flip": flip, "error": str(e)})

            self.conn.execute("COMMIT")
            logger.info(f"Applied {results['applied']} direction flips to database")

        except Exception as e:
            self.conn.execute("ROLLBACK")
            logger.error(f"Transaction failed, rolled back: {e}")
            results["errors"] = len(flip_list)
            results["message"] = f"Transaction failed: {e}"
            return results

        results["message"] = (
            f"Successfully applied {results['applied']}/{results['total']} flips"
        )
        return results

    def verify_flip(self, old_up: int, old_down: int) -> Dict[str, Any]:
        """Verify a flip was applied correctly."""
        # After flip: old_down should now be upstream of old_up
        result = self.conn.execute(
            """
            SELECT direction
            FROM reach_topology
            WHERE region = ?
              AND reach_id = ?
              AND neighbor_reach_id = ?
        """,
            [self.region, old_up, old_down],
        ).fetchdf()

        if len(result) == 0:
            return {"verified": False, "reason": "No topology record found"}

        direction = result["direction"].iloc[0]
        if direction == "up":
            return {"verified": True, "new_direction": "up"}
        else:
            return {
                "verified": False,
                "reason": f"Expected direction=up, got {direction}",
            }


# -----------------------------------------------------------------------------
# SIMPLE VIOLATION DETECTOR (for quick analysis without MILP)
# -----------------------------------------------------------------------------


class SimpleViolationDetector:
    """
    Simple heuristic violation detector - no MILP required.

    Detects edges where physical constraints are violated:
    - facc decreases downstream (except in distributaries)
    - dist_out increases downstream
    - SWOT WSE increases downstream
    """

    def __init__(self, G: nx.DiGraph, swot_data: Optional[pd.DataFrame] = None):
        self.G = G
        self.swot_data = swot_data

    def is_distributary_edge(self, u: int, v: int) -> bool:
        """Check if edge is part of a distributary (facc decrease is valid)."""
        u_data = self.G.nodes.get(u, {})
        n_rch_down = u_data.get("n_rch_down", 0) or 0
        u_trib = u_data.get("trib_flag", 0) or 0
        v_trib = self.G.nodes.get(v, {}).get("trib_flag", 0) or 0

        return n_rch_down > 1 or u_trib >= 2 or v_trib >= 2

    def get_swot_wse(self, reach_id: int) -> Optional[float]:
        """Get mean SWOT WSE for a reach."""
        if self.swot_data is None:
            return None
        reach_data = self.swot_data[self.swot_data["reach_id"] == reach_id]
        if len(reach_data) == 0:
            return None
        return float(reach_data["wse"].iloc[0])

    def find_violations(self, include_distributaries: bool = False) -> List[Dict]:
        """Find edges violating physical constraints."""
        violations = []
        distributary_count = 0

        for u, v in self.G.edges():
            u_data = self.G.nodes[u]
            v_data = self.G.nodes[v]

            is_dist = self.is_distributary_edge(u, v)
            if is_dist:
                distributary_count += 1

            reasons = []
            violation = {
                "upstream": u,
                "downstream": v,
                "is_distributary": is_dist,
            }

            # Check facc
            u_facc = u_data.get("facc", 0) or 0
            v_facc = v_data.get("facc", 0) or 0
            violation["upstream_facc"] = u_facc
            violation["downstream_facc"] = v_facc

            if u_facc > 0 and v_facc > 0:
                ratio = u_facc / v_facc
                if ratio > 10:
                    if not is_dist or include_distributaries:
                        reasons.append(f"facc_ratio={ratio:.1f}")
                        violation["facc_ratio"] = ratio

            # Check SWOT WSE
            u_wse = self.get_swot_wse(u)
            v_wse = self.get_swot_wse(v)
            if u_wse and v_wse:
                violation["upstream_wse"] = u_wse
                violation["downstream_wse"] = v_wse
                if v_wse > u_wse + 0.5:
                    reasons.append(f"wse_uphill={v_wse - u_wse:.2f}m")

            if reasons:
                violation["reasons"] = reasons
                violations.append(violation)

        violations.sort(key=lambda x: x.get("facc_ratio", 0), reverse=True)
        logger.info(
            f"Found {len(violations):,} violations "
            f"(excluded {distributary_count:,} distributary edges)"
        )

        return violations


# -----------------------------------------------------------------------------
# MAIN CLI
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="SWORD Topology Optimizer - phi-based MILP optimization"
    )
    parser.add_argument(
        "--db", default="data/duckdb/sword_v17b.duckdb", help="Path to DuckDB database"
    )
    parser.add_argument(
        "--region",
        default="NA",
        choices=["NA", "SA", "EU", "AF", "AS", "OC"],
        help="Region to process",
    )
    parser.add_argument(
        "--method",
        default="simple",
        choices=["simple", "milp"],
        help="Method: simple (violation detection) or milp (full optimization)",
    )
    parser.add_argument(
        "--swot",
        default="/Volumes/SWORD_DATA/data/swot/parquet_lake_D",
        help="Path to SWOT parquet directory",
    )
    parser.add_argument(
        "--include-distributaries",
        action="store_true",
        help="Include distributary edges in violation detection",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Only analyze, do not apply changes",
    )
    parser.add_argument(
        "--apply", action="store_true", help="Apply suggested fixes to database"
    )
    parser.add_argument("--output", help="Output CSV file for results")
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Run validation tests after optimization",
    )
    parser.add_argument(
        "--validate-current",
        action="store_true",
        help="Validate current topology (before optimization)",
    )
    parser.add_argument(
        "--by-component",
        action="store_true",
        default=True,
        help="Solve MILP by connected component (more robust)",
    )
    parser.add_argument(
        "--monolithic",
        action="store_true",
        help="Solve MILP as single problem (can be slow)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompts when applying changes",
    )
    parser.add_argument(
        "--swot-refine",
        action="store_true",
        help="Apply SWOT WSE refinement to MILP results",
    )
    parser.add_argument(
        "--wse-threshold",
        type=float,
        default=1.0,
        help="WSE difference threshold (meters) for confident direction",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SWORD Topology Optimizer")
    print("=" * 70)
    print(f"Database: {args.db}")
    print(f"Region: {args.region}")
    print(f"Method: {args.method}")
    print(f"Mode: {'DRY RUN' if args.dry_run and not args.apply else 'APPLY'}")
    print("=" * 70)

    with SWORDGraphBuilder(args.db, args.region) as builder:
        if args.method == "simple":
            # Simple violation detection
            G_directed = builder.build_directed_graph()

            # Load SWOT data
            swot_data = None
            if Path(args.swot).exists():
                swot_data = load_swot_data(args.swot, args.region)

            detector = SimpleViolationDetector(G_directed, swot_data)
            violations = detector.find_violations(
                include_distributaries=args.include_distributaries
            )

            print("\n=== SIMPLE VIOLATION DETECTION ===")
            print(f"Total edges: {G_directed.number_of_edges():,}")
            print(f"Violations found: {len(violations):,}")

            if violations:
                print("\n=== TOP 10 VIOLATIONS ===")
                for i, v in enumerate(violations[:10]):
                    print(f"\n{i + 1}. {v['upstream']} -> {v['downstream']}")
                    print(f"   Reasons: {', '.join(v.get('reasons', []))}")
                    print(
                        f"   facc: {v['upstream_facc']:,.0f} -> {v['downstream_facc']:,.0f}"
                    )
                    if v.get("is_distributary"):
                        print("   [DISTRIBUTARY]")

            if args.output and violations:
                df = pd.DataFrame(violations)
                df.to_csv(args.output, index=False)
                print(f"\nResults saved to {args.output}")

        elif args.method == "milp":
            if not PULP_AVAILABLE:
                print("ERROR: MILP requires PuLP. Install with: pip install pulp")
                return

            # Build undirected graph for phi computation
            G_undirected = builder.build_undirected_graph()
            G_directed = builder.build_directed_graph()

            # Classify nodes from directed graph (current topology)
            node_types = classify_nodes_from_directed(G_directed)

            # Set node types on undirected graph
            for n, t in node_types.items():
                if n in G_undirected.nodes:
                    G_undirected.nodes[n]["node_type"] = t

            # Find outlets
            outlets = [n for n, t in node_types.items() if t == "Outlet"]
            if not outlets:
                print("ERROR: No outlets found. Cannot compute phi.")
                return

            print(f"\nFound {len(outlets):,} outlets")

            # Compute phi
            phi = compute_phi(G_undirected, outlets)

            # Solve MILP
            by_component = not args.monolithic
            edge_dir, status = solve_milp_orientation(
                G_undirected,
                phi,
                node_types,
                prefer_highs=True,
                by_component=by_component,
            )

            if status != "Optimal":
                print(f"WARNING: Solver status = {status}")

            # SWOT refinement (if requested)
            swot_data = None
            if Path(args.swot).exists():
                swot_data = load_swot_data(args.swot, args.region)

            if args.swot_refine and swot_data is not None:
                print("\n=== SWOT REFINEMENT ===")
                refiner = SWOTRefinement(
                    G_undirected,
                    edge_dir,
                    swot_data,
                    wse_threshold=args.wse_threshold,
                )
                edge_dir, refine_stats = refiner.refine_directions(override_milp=True)
                print(f"SWOT coverage: {refine_stats['swot_coverage']:,} edges")
                print(f"Reliable (R): {refine_stats['reliable_edges']:,}")
                print(f"MILP confirmed: {refine_stats['milp_confirmed']:,}")
                print(f"MILP overridden: {refine_stats['milp_overridden']:,}")

            # Compare with current topology
            comparison = compare_directions(G_directed, edge_dir)

            print("\n=== MILP OPTIMIZATION RESULTS ===")
            print(f"Solver status: {status}")
            print(f"Total edges: {comparison['total_edges']:,}")
            print(
                f"Confirmed (same direction): {comparison['confirmed']:,} ({comparison['pct_confirmed']:.1f}%)"
            )
            print(
                f"To flip (direction change): {comparison['to_flip']:,} ({comparison['pct_to_flip']:.1f}%)"
            )

            if comparison["flip_list"]:
                print("\n=== SAMPLE EDGES TO FLIP ===")
                for flip in comparison["flip_list"][:10]:
                    print(
                        f"  {flip['current_upstream']} -> {flip['current_downstream']}"
                    )
                    print(
                        f"    BECOMES: {flip['new_upstream']} -> {flip['new_downstream']}"
                    )

            if args.output:
                df = pd.DataFrame(comparison["flip_list"])
                df.to_csv(args.output, index=False)
                print(f"\nFlip list saved to {args.output}")

            # Validation
            if args.validate:
                print("\n=== VALIDATING OPTIMIZED TOPOLOGY ===")

                # Build optimized graph
                G_optimized = build_optimized_graph(G_undirected, edge_dir, phi)

                # Set node types
                for n, t in node_types.items():
                    if n in G_optimized.nodes:
                        G_optimized.nodes[n]["node_type"] = t

                # Use SWOT data already loaded (or load if not done)
                if swot_data is None and Path(args.swot).exists():
                    swot_data = load_swot_data(args.swot, args.region)

                validator = TopologyValidator(G_optimized, phi, swot_data)
                results = validator.run_all_tests(verbose=True)

                print(f"\n{validator.get_summary()}")

                # Return exit code based on validation
                if not results.get("all_passed", False):
                    print("\n⚠️  Some validation tests failed. Review results above.")

            if args.validate_current:
                print("\n=== VALIDATING CURRENT TOPOLOGY ===")
                validator_current = TopologyValidator(G_directed, phi)
                results_current = validator_current.run_all_tests(verbose=True)
                print(f"\n{validator_current.get_summary()}")

            if args.apply:
                print("\n=== APPLYING CHANGES ===")
                if not comparison["flip_list"]:
                    print("No changes to apply.")
                else:
                    proceed = False
                    if args.force:
                        proceed = True
                    else:
                        print(
                            f"About to apply {len(comparison['flip_list'])} direction changes."
                        )
                        print("This will modify the database.")
                        try:
                            response = input("Continue? [y/N]: ").strip().lower()
                            proceed = response in ("y", "yes")
                        except EOFError:
                            print("Non-interactive mode. Use --force to apply.")
                            proceed = False

                    if proceed:
                        with TopologyUpdater(args.db, args.region) as updater:
                            result = updater.apply_flips(
                                comparison["flip_list"],
                                dry_run=False,  # Actually apply since --apply was specified
                                reason="MILP phi-based topology optimization",
                            )
                            print(f"\nResult: {result['message']}")
                            if result["errors"] > 0:
                                print(f"Errors: {result['errors']}")
                                for err in result["error_details"][:5]:
                                    print(f"  - {err['flip']}: {err['error']}")
                    else:
                        print("Aborted. No changes applied.")

    print("\nDone!")


if __name__ == "__main__":
    main()
