#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global φ-optimization to orient an undirected river graph into a DAG.

We assign a direction to every undirected edge by solving a MILP:

Objective (minimize):
  Sum of "uphill" penalties relative to φ (distance-to-outlet).
  For each undirected edge {u,v} with decision x_uv ∈ {0,1} meaning u->v if 1 else v->u:
    cost = x_uv * max(0, φ[v]-φ[u]) + (1 - x_uv) * max(0, φ[u]-φ[v])

Hard constraints:
  - Head_water: all incident edges must point OUT (no incoming)
  - Outlet    : all incident edges must point IN  (no outgoing)
  - Junction  : at least 1 incoming and 1 outgoing IF feasible (guards applied)
  - DAG       : topological ranks (integer) enforce acyclicity with big-M constraints

Outputs (in --outdir):
  - river_directed.pkl        (NetworkX DiGraph)
  - river_nodes.gpkg / .csv   (layer 'nodes')
  - river_edges.gpkg / .csv   (layer 'edges')
  - degree_violations.csv     (if any violations are found)
  - global_solver_report.txt  (status, objective, paths to outputs)

Node attributes required:
  node['x'], node['y']
  node['node_type'] in {"Head_water","Outlet","Junction"}

Edge attribute required:
  edge['distance'] > 0  (used to compute φ via multi-source Dijkstra)

Solver:
  Prefers HiGHS (via PuLP >= 2.7); falls back to CBC if HiGHS unavailable.

Author: (you)
"""

import os
import pickle
import argparse
from typing import Dict, Tuple, List, Hashable
from collections import defaultdict

import networkx as nx
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString
except Exception as e:
    raise SystemExit(
        "This script requires GeoPandas and Shapely.\n"
        "Install with: pip install geopandas shapely\n"
        f"Import error: {e}"
    )

try:
    import pulp
    # Prefer HiGHS if present (fast open-source MILP)
    try:
        from pulp import HiGHS_CMD  # PuLP >= 2.7
        _HIGHS_AVAILABLE = True
    except Exception:
        _HIGHS_AVAILABLE = False
except Exception as e:
    raise SystemExit(
        "This script requires PuLP (MILP solver). Install with: pip install pulp\n"
        f"Import error: {e}"
    )


# -----------------------------
# φ: distances to outlets (undirected)
# -----------------------------

def compute_weighted_outlet_distances(G: nx.Graph, outlets, weight_attr="distance") -> Dict[Hashable, float]:
    for u, v, d in G.edges(data=True):
        w = d.get(weight_attr, None)
        if w is None or not isinstance(w, (int, float)) or w <= 0:
            raise ValueError(f"Edge ({u},{v}) missing positive '{weight_attr}'")
    dlen = nx.multi_source_dijkstra_path_length(G, sources=list(outlets), weight=weight_attr)
    return {n: dlen.get(n, None) for n in G.nodes()}


# -----------------------------
# MILP: Global orientation with DAG + rules
# -----------------------------

def solve_global_phi_orientation(
    G: nx.Graph,
    phi: Dict[Hashable, float],
    node_type_field: str = "node_type",
    prefer_highs: bool = True,
) -> Tuple[Dict[Tuple[str, str], int], Dict[str, int], str, Dict[str, Hashable]]:
    """
    Decide a direction for each undirected edge, minimize uphill cost,
    enforce DAG and hydrologic rules.

    Returns:
      edge_dir: {(a,b): 1 if a->b else 0}, using string-sorted edge keys (a<b)
      rank    : {str(node): integer topological rank}
      status  : PuLP status string
      idmap   : {str_id: original_node_object} for mapping back
    """
    # --- Normalize node ids to strings but remember original objects ---
    idmap: Dict[str, Hashable] = {}
    for n in G.nodes():
        sid = str(n)
        # Ensure uniqueness even if str() collides: append suffix if needed
        if sid in idmap and idmap[sid] is not n:
            # disambiguate by appending a unique counter
            cnt = 1
            base = sid
            while sid in idmap and idmap[sid] is not n:
                sid = f"{base}__dup{cnt}"
                cnt += 1
        idmap[sid] = n

    nodes_s = sorted(idmap.keys())

    # φ mapped to string ids; missing φ -> large number to discourage uphill into them
    finite_phis = [v for v in phi.values() if v is not None]
    big_phi = (max(finite_phis) if finite_phis else 0.0) + 1e9
    phi_s = {str(n): (phi[n] if phi.get(n) is not None else big_phi) for n in G.nodes()}

    # Undirected unique edge pairs keyed by sorted string ids
    undirected_pairs: List[Tuple[str, str]] = []
    for u, v in G.edges():
        a, b = str(u), str(v)
        if a == b:
            continue
        if a > b:
            a, b = b, a
        undirected_pairs.append((a, b))
    undirected_pairs = sorted(list(set(undirected_pairs)))

    # Precompute uphill costs for both directions
    cost_uv = {}
    for a, b in undirected_pairs:
        du, dv = phi_s[a], phi_s[b]
        cost_u_to_v = max(0.0, dv - du)   # uphill if φ increases
        cost_v_to_u = max(0.0, du - dv)
        cost_uv[(a, b)] = (cost_u_to_v, cost_v_to_u)

    # Neighborhood (string ids)
    nbrs = {n: set() for n in nodes_s}
    for a, b in undirected_pairs:
        nbrs[a].add(b)
        nbrs[b].add(a)

    # Node types by string id
    ntype = {str(n): G.nodes[n].get(node_type_field) for n in G.nodes()}

    # Problem size
    N = len(nodes_s)
    M = N  # big-M (sufficient since ranks are in [0, N-1])

    # ----------------- Build MILP (correct ordering) -----------------
    m = pulp.LpProblem("global_phi_orientation", pulp.LpMinimize)

    # Decision variables BEFORE helper usage
    x = { (a,b): pulp.LpVariable(f"x_{a}__{b}", lowBound=0, upBound=1, cat="Binary")
          for (a,b) in undirected_pairs }

    # Rank variables (integers) BEFORE helper usage
    r = { n: pulp.LpVariable(f"r_{n}", lowBound=0, upBound=N-1, cat="Integer")
          for n in nodes_s }

    # Objective: minimize uphill penalties
    m += pulp.lpSum(
        x[(a,b)] * cost_uv[(a,b)][0] + (1 - x[(a,b)]) * cost_uv[(a,b)][1]
        for (a,b) in undirected_pairs
    )

    # DAG constraints: chosen direction must increase rank by ≥1
    for (a, b) in undirected_pairs:
        # If x[a,b] == 1 -> a->b -> r[a] >= r[b] + 1
        m += r[a] - r[b] >= 1 - M * (1 - x[(a, b)])
        # If x[a,b] == 0 -> b->a -> r[b] >= r[a] + 1
        m += r[b] - r[a] >= 1 - M * x[(a, b)]

    # # Helpers to force direction for hard rules
    # def force_dir_out(from_n: str, to_n: str):
    #     # Want from_n -> to_n
    #     a, b = (from_n, to_n) if from_n < to_n else (to_n, from_n)
    #     var = x[(a, b)]
    #     if a == from_n:
    #         m += var == 1  # a->b
    #     else:
    #         m += var == 0  # b->a (i.e., from_n->to_n)

    # def force_dir_in(to_n: str, from_n: str):
    #     # Want from_n -> to_n (i.e., 'to_n' has only incoming)
    #     force_dir_out(from_n, to_n)

    # Hard rules: Head_water (only outgoing), Outlet (only incoming)
    for n in nodes_s:
        for w in nbrs[n]:
            a, b = (n, w) if n < w else (w, n)
            var = x[(a, b)]

            if ntype[n] == "Head_water":
                # Force n -> w
                if n < w:
                    m += var == 1
                else:
                    m += var == 0

            elif ntype[n] == "Outlet":
                # Force w -> n  (i.e. n only receives)
                if n < w:
                    m += var == 0
                else:
                    m += var == 1

    # Junction: >=1 incoming and >=1 outgoing IF feasible
    for n in nodes_s:
        if ntype[n] != "Junction":
            continue
        inc_terms = []
        out_terms = []
        for w in nbrs[n]:
            a, b = (n, w) if n < w else (w, n)
            var = x[(a, b)]
            if n < w:
                # var==1 means n->w; var==0 means w->n
                if ntype[w] != "Outlet":      # Outlet->Junction disallowed => not an eligible incoming source
                    inc_terms.append(1 - var)  # w->n
                if ntype[w] != "Head_water":  # Junction->Head_water disallowed
                    out_terms.append(var)      # n->w
            else:
                # pair is keyed (w,n); var==1 means w->n; var==0 means n->w
                if ntype[w] != "Outlet":
                    inc_terms.append(var)      # w->n
                if ntype[w] != "Head_water":
                    out_terms.append(1 - var)  # n->w

        # Only enforce if feasible on that side
        if len(inc_terms) > 0:
            m += pulp.lpSum(inc_terms) >= 1
        if len(out_terms) > 0:
            m += pulp.lpSum(out_terms) >= 1

    # Solve with HiGHS if available, else CBC
    if prefer_highs and 'HiGHS_CMD' in globals() and _HIGHS_AVAILABLE:
        solver = HiGHS_CMD(msg=False)  # fast, open-source
    else:
        solver = pulp.PULP_CBC_CMD(msg=False)  # fallback

    status_code = m.solve(solver)
    status_str = pulp.LpStatus[status_code]
    if status_str != "Optimal":
        print(f"WARNING: MILP solve status = {status_str}")

    # Extract decisions
    edge_dir = {}
    for (a, b) in undirected_pairs:
        val = pulp.value(x[(a, b)])
        val = 0.0 if val is None else val
        edge_dir[(a, b)] = int(round(val))

    rank = { n: int(round(pulp.value(r[n]) or 0)) for n in nodes_s }

    return edge_dir, rank, status_str, idmap


# -----------------------------
# Build DiGraph from MILP solution
# -----------------------------

# def build_digraph_from_solution(
#     G: nx.Graph,
#     edge_dir: Dict[Tuple[str,str], int],
#     idmap: Dict[str, Hashable],
# ) -> nx.DiGraph:
#     """Map string decisions back to original node objects."""
#     # Reverse map for convenience: str->orig already in idmap
#     D = nx.DiGraph()
#     D.add_nodes_from(G.nodes(data=True))
#     for u, v in G.edges():
#         su, sv = str(u), str(v)
#         if su == sv:
#             continue
#         a, b = (su, sv) if su < sv else (sv, su)
#         val = edge_dir[(a, b)]
#         src_s, dst_s = (a, b) if val == 1 else (b, a)
#         src, dst = idmap[src_s], idmap[dst_s]
#         D.add_edge(src, dst)
#     return D
def build_digraph_from_solution(
    G: nx.Graph,
    edge_dir: Dict[Tuple[str,str], int],
    idmap: Dict[str, Hashable],
) -> nx.DiGraph:
    """Construct directed graph and preserve ALL original node+edge attributes."""
    D = nx.DiGraph()

    # Add original node attributes
    D.add_nodes_from(G.nodes(data=True))

    # Add directed edges with attributes copied from G
    for u, v in G.edges():
        su, sv = str(u), str(v)
        if su == sv:
            continue
        # normalize keying
        a, b = (su, sv) if su < sv else (sv, su)
        val = edge_dir[(a, b)]
        src_s, dst_s = (a, b) if val == 1 else (b, a)
        src, dst = idmap[src_s], idmap[dst_s]

        # Copy original attributes from undirected graph
        attrs = G.get_edge_data(u, v, default={}).copy()
        D.add_edge(src, dst, **attrs)

    return D

# -----------------------------
# Export utilities
# -----------------------------

def add_hydrologic_weights(D: nx.DiGraph, phi: Dict[Hashable, float]):
    nx.set_node_attributes(D, {n: phi.get(n) for n in D.nodes()}, "phi")
    delta = {}
    eq = {}
    for u, v in D.edges():
        pu, pv = phi.get(u), phi.get(v)
        dp = None if (pu is None or pv is None) else (pu - pv)
        delta[(u, v)] = dp
        eq[(u, v)] = (dp == 0) if dp is not None else False
    nx.set_edge_attributes(D, delta, "delta_phi")
    nx.set_edge_attributes(D, eq, "is_equal_level")


def export_all(D: nx.DiGraph, phi: Dict[Hashable, float], outdir: str, crs: str = "EPSG:4326"):
    # Nodes
    nrecs = []
    for n, d in D.nodes(data=True):
        x = d.get("x"); y = d.get("y")
        geom = Point(x, y) if (x is not None and y is not None) else None
        nrecs.append({
            "node_id": n,
            "node_type": d.get("node_type"),
            "phi": phi.get(n),
            "in_deg": D.in_degree(n),
            "out_deg": D.out_degree(n),
            "geometry": geom
        })
    gdf_nodes = gpd.GeoDataFrame(nrecs, geometry="geometry", crs=crs)

    # Edges
    erecs = []
    for u, v, ed in D.edges(data=True):
        nu, nv = D.nodes[u], D.nodes[v]
        x1, y1 = nu.get("x"), nu.get("y")
        x2, y2 = nv.get("x"), nv.get("y")
        geom = LineString([(x1, y1), (x2, y2)]) if None not in (x1, y1, x2, y2) else None
        erecs.append({
            "u": u, "v": v,
            'section_id':ed.get("section_id"),
            "confidence":ed.get("confidence"),
            "delta_phi": ed.get("delta_phi"),
            "UD_slope": ed.get("UD_slope"),
            "DU_slope": ed.get("UD_slope"),
            "slopeF": ed.get("slopeF"),
            "slopeP": ed.get("slopeP"),
            "SE": ed.get("SE"),
            "convergence": ed.get("convergence"),
            "is_equal_level": ed.get("is_equal_level"),
            "geometry": geom
        })
    gdf_edges = gpd.GeoDataFrame(erecs, geometry="geometry", crs=crs)

    # Write GPKG (overwrite if exists)
    nodes_gpkg = os.path.join(outdir, "river_nodes.gpkg")
    edges_gpkg = os.path.join(outdir, "river_edges.gpkg")
    for p in (nodes_gpkg, edges_gpkg):
        if os.path.exists(p): os.remove(p)
    gdf_nodes.to_file(nodes_gpkg, layer="nodes", driver="GPKG")
    gdf_edges.to_file(edges_gpkg, layer="edges", driver="GPKG")

    # CSVs
    gdf_nodes.drop(columns=["geometry"]).to_csv(os.path.join(outdir, "river_nodes.csv"), index=False)
    gdf_edges.drop(columns=["geometry"]).to_csv(os.path.join(outdir, "river_edges.csv"), index=False)

    # Pickle
    pkl = os.path.join(outdir, "river_directed.pkl")
    with open(pkl, "wb") as f:
        pickle.dump(D, f, protocol=pickle.HIGHEST_PROTOCOL)

    return nodes_gpkg, edges_gpkg, pkl


def verify_degrees(D: nx.DiGraph):
    """Return dict of violations in the current directed graph D."""
    bad = {"junctions": [], "headwaters": [], "outlets": []}
    for n, d in D.nodes(data=True):
        t = d.get("node_type")
        indeg, outdeg = D.in_degree(n), D.out_degree(n)
        if t == "Junction" and (indeg == 0 or outdeg == 0):
            bad["junctions"].append((n, indeg, outdeg))
        elif t == "Head_water" and indeg > 0:
            bad["headwaters"].append((n, indeg, outdeg))
        elif t == "Outlet" and outdeg > 0:
            bad["outlets"].append((n, indeg, outdeg))
    return bad


def get_first_u(edges):
    target = None
    for edge in edges:
        if edge[3]['path_seg_pos'] == 0:
            target = edge
    if target is None:
        raise TypeError('u not found')
    return target

def phi_mapping(directory, continent):
    with open(directory + f"/output/{continent}/river_directed.pkl", "rb") as f:
        G_ref = pickle.load(f)
    with open(directory + f"/output/{continent}_MultiDirected.pkl", "rb") as f:
        DG = pickle.load(f)
    for u, v, k in DG.edges(keys=True):
        DG[u][v][k]['phi_direction_change'] = False
    # --- 1. Build mapping from refined graph direction ---
    ref_dir = {}
    for u, v, data in G_ref.edges(data=True):
        seg = int(data["section_id"])
        ref_dir[seg] = (u, v)  # direction in the refined graph itself
    # --- 2. Collect all edges of each path_seg in na_MultiDirected ---
    edges_by_seg = defaultdict(list)
    for u, v, k, data in DG.edges(keys=True, data=True):
        seg = data.get("path_seg")
        if seg is not None:
            edges_by_seg[seg].append((u, v, k, data))
    # --- 3. Compare and flip where needed ---
    for seg, (ref_u, ref_v) in ref_dir.items():
        if seg not in edges_by_seg:
            continue  # segment not found in NA graph
        # Determine the current direction in na_MultiDirected
        # Just look at the first edge (they should all go same way)
        u0, v0, k0, d0 = get_first_u(edges_by_seg[seg])
        # Compare with the refined direction
        if u0 == ref_u:
            continue
        elif u0 == ref_v:
            ####################################
            # add multidirect check
            ####################################
            for u, v, k, d in list(edges_by_seg[seg]):
                DG[u][v][k]['phi_direction_change'] = True
        else:
            print('ERROR Edge missing')
    # write PKL
    with open(directory + f'/output/{continent}_MultiDirected.pkl',"wb") as f:
        pickle.dump(DG,f)

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Global φ-optimization orientation with hydrologic hard rules (HiGHS preferred).")
    parser.add_argument("--input", required=True, help="Path to undirected NetworkX Graph pickle.")
    parser.add_argument("--outdir", required=True, help="Directory to write outputs.")
    parser.add_argument("--crs", default="EPSG:4326", help="CRS for GeoPackages.")
    parser.add_argument("--weight_attr", default="distance", help="Edge weight attribute used for φ.")
    parser.add_argument("--continent", required=True, help="Continent code (e.g., 'na', 'as', 'eu').")
    parser.add_argument("--workdir", required=True, help="Project root directory (WORKDIR).")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)


    with open(args.input, "rb") as f:
        G = pickle.load(f)
    if not isinstance(G, (nx.Graph, nx.MultiGraph)) or G.is_directed():
        raise TypeError("Input must be an undirected NetworkX Graph or MultiGraph.")

    # Collapse MultiGraph to simple Graph (keep min distance per pair)
    if isinstance(G, nx.MultiGraph):
        H = nx.Graph()
        for u, v, d in G.edges(data=True):
            w = d.get(args.weight_attr, None)
            if w is None:
                continue
            if H.has_edge(u, v):
                if w < H[u][v].get(args.weight_attr, w):
                    H[u][v][args.weight_attr] = w
            else:
                H.add_edge(u, v, **{args.weight_attr: w})
        for n, nd in G.nodes(data=True):
            H.add_node(n, **nd)
        G = H

    outlets = {n for n, d in G.nodes(data=True) if d.get("node_type") == "Outlet"}
    if not outlets:
        raise ValueError("No nodes labeled 'Outlet' found — φ requires at least one outlet.")

    # φ from undirected G
    phi = compute_weighted_outlet_distances(G, outlets, weight_attr=args.weight_attr)

    # Solve global MILP (HiGHS preferred)
    edge_dir, rank, status, idmap = solve_global_phi_orientation(G, phi, prefer_highs=True)

    # Build DiGraph from solution
    D = build_digraph_from_solution(G, edge_dir, idmap)

    # Attach φ-based hydrologic weights
    add_hydrologic_weights(D, phi)

    # Verify constraints on the final exported D
    bad = verify_degrees(D)
    viol_rows = []
    for n, i, o in bad["junctions"]:
        viol_rows.append({"node_id": n, "node_type": "Junction", "in_deg": i, "out_deg": o})
    for n, i, o in bad["headwaters"]:
        viol_rows.append({"node_id": n, "node_type": "Head_water", "in_deg": i, "out_deg": o})
    for n, i, o in bad["outlets"]:
        viol_rows.append({"node_id": n, "node_type": "Outlet", "in_deg": i, "out_deg": o})

    if viol_rows:
        pd.DataFrame(viol_rows).to_csv(os.path.join(args.outdir, "degree_violations.csv"), index=False)

    # Export files
    nodes_gpkg, edges_gpkg, pkl = export_all(D, phi, args.outdir, args.crs)

    # Report
    total_uphill_cost = 0.0
    for u, v in D.edges():
        pu, pv = phi.get(u), phi.get(v)
        if pu is not None and pv is not None and pv > pu:
            total_uphill_cost += (pv - pu)

    report_path = os.path.join(args.outdir, "global_solver_report.txt")
    with open(report_path, "w") as f:
        f.write("Global φ-optimization orientation\n")
        f.write(f"Solver: {'HiGHS' if 'HiGHS_CMD' in globals() and _HIGHS_AVAILABLE else 'CBC'}\n")
        f.write(f"PuLP status: {status}\n")
        f.write(f"Total uphill cost (sum max(0, φ[v]-φ[u])): {total_uphill_cost}\n")
        f.write(f"Nodes GPKG: {nodes_gpkg}\n")
        f.write(f"Edges GPKG: {edges_gpkg}\n")
        f.write(f"Pickle: {pkl}\n")
        if viol_rows:
            f.write("Degree violations written to degree_violations.csv\n")

    print("\n=== Global φ-optimization complete ===")
    print("Solve status:", status)
    print("Outputs:")
    print(" -", pkl)
    print(" -", nodes_gpkg)
    print(" -", edges_gpkg)
    print(" - river_nodes.csv, river_edges.csv")
    if viol_rows:
        print("⚠️ degree_violations.csv written (check junction feasibility).")
    print("Report:", report_path)

    phi_mapping(args.workdir, args.continent)

if __name__ == "__main__":
    main()
