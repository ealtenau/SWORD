#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Component-wise refinement of φ-only directed river networks using reliable upstream/downstream metadata.

Hard rules (cannot be violated):
  - Head_water: only outgoing edges
  - Outlet: only incoming edges
  - Junctions: >=1 incoming AND >=1 outgoing, IF feasible
  - DAG: enforced via topological rank constraints

Soft objective (weights adjustable):
  1) Minimize disagreement with R-metadata directions (strong penalty)
  2) Minimize flips of U-metadata vs φ-only direction (weaker penalty)
  3) Minimize φ-uphill flow (tiny tie-breaker)

Outputs:
  - river_directed_refined.pkl
  - river_nodes_refined.gpkg, river_edges_refined.gpkg
  - river_nodes_refined.csv, river_edges_refined.csv
  - r_direction_conflicts.csv (only edges where R requested a flip but hydrology prevented agreement)
  - degree_violations_refined.csv (only if any exist)
  - refine_report.txt

Solver:
  Prefers HiGHS (fast) if installed. Otherwise uses CBC automatically.
"""

import os
import math
import pickle
import argparse
from typing import Dict, Tuple, List, Hashable

import pandas as pd
import networkx as nx

from collections import defaultdict


try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString
except ImportError:
    raise SystemExit("Requires: pip install geopandas shapely")

try:
    import pulp
    try:
        from pulp import HiGHS_CMD  # PuLP >= 2.7
        _HIGHS_AVAILABLE = True
    except Exception:
        _HIGHS_AVAILABLE = False
except ImportError:
    raise SystemExit("Requires: pip install pulp")


# ------------------------------------------------
# Helper utilities
# ------------------------------------------------

def build_undirected_skeleton(D: nx.DiGraph, weight_attr="distance") -> nx.Graph:
    """Simple G from D — min weight per pair, preserving node attributes."""
    G = nx.Graph()
    for n, nd in D.nodes(data=True):
        G.add_node(n, **nd)
    for u, v, ed in D.edges(data=True):
        w = ed.get(weight_attr)
        if not G.has_edge(u, v):
            G.add_edge(u, v, **({weight_attr: w} if w is not None else {}))
        else:
            old = G[u][v].get(weight_attr)
            if w is not None and (old is None or w < old):
                G[u][v][weight_attr] = w
    return G


def compute_phi_undirected(G: nx.Graph, weight_attr="distance") -> Dict[Hashable, float]:
    outlets = [n for n, d in G.nodes(data=True) if d.get("node_type") == "Outlet"]
    if not outlets:
        raise RuntimeError("No Outlet found → cannot compute φ")
    for u, v, d in G.edges(data=True):
        w = d.get(weight_attr)
        if w is None or not isinstance(w, (int, float)) or w <= 0:
            raise ValueError(f"Edge ({u},{v}) missing positive '{weight_attr}'")
    return nx.multi_source_dijkstra_path_length(G, outlets, weight=weight_attr)


# ------------------------------------------------
# MILP solve for a single component
# ------------------------------------------------

def solve_component(
    D_sub: nx.DiGraph,
    phi: Dict[Hashable, float],
    weight_attr="distance",
    wR=1000.0,
    wU=1.0,
    wUp=0.001,
    prefer_highs=True,
):
    """Solve global orientation for a single connected component."""

    nodes = list(D_sub.nodes())
    N = D_sub.number_of_nodes()
    M = N  # Big-M rank bound

    # Undirected edge pairs with sorted repr
    pairs = sorted({
        tuple(sorted((u, v), key=str)) for u, v in D_sub.edges()
    })

    # Neighborhood index
    nbrs = {n: set() for n in nodes}
    for a, b in pairs:
        nbrs[a].add(b)
        nbrs[b].add(a)

    # Node types
    ntype = nx.get_node_attributes(D_sub, "node_type")

    # Gather current φ-only directions and R metadata
    edge_info = {}
    for a, b in pairs:
        ed = D_sub.get_edge_data(a, b) or D_sub.get_edge_data(b, a)
        curr = 1 if D_sub.has_edge(a, b) else 0

        up = ed.get("upstream_node")
        dn = ed.get("downstream_node")
        conf = ed.get("confidence")

        desired = None
        has_prior = False
        if conf == "R" and up is not None and dn is not None \
           and isinstance(up, (str,int)) and isinstance(dn,(str,int)):
            su, sv = str(a), str(b)
            sup, sdn = str(up), str(dn)
            if sup == su and sdn == sv: desired = 1; has_prior = True
            elif sup == sv and sdn == su: desired = 0; has_prior = True

        pa, pb = phi.get(a, math.inf), phi.get(b, math.inf)
        edge_info[(a, b)] = dict(
            curr_u_to_v=curr,
            has_prior=has_prior,
            desired=desired,
            uphill_a2b=max(0, pb-pa),
            uphill_b2a=max(0, pa-pb),
        )

    # MILP
    m = pulp.LpProblem("refine_component", pulp.LpMinimize)

    x = { (a,b): pulp.LpVariable(f"x_{a}__{b}", 0, 1, cat="Binary")
          for a,b in pairs }
    r = { n: pulp.LpVariable(f"r_{n}", 0, N-1, cat="Integer")
          for n in nodes }

    terms = []
    for (a, b), info in edge_info.items():
        if info["has_prior"]:
            if info["desired"] == 1:
                terms.append(wR*(1 - x[(a,b)]))
            else:
                terms.append(wR*(x[(a,b)]))
        else:
            if info["curr_u_to_v"] == 1:
                terms.append(wU*(1 - x[(a,b)]))
            else:
                terms.append(wU*(x[(a,b)]))
        terms.append(wUp*( x[(a,b)]*info["uphill_a2b"]
                         + (1-x[(a,b)])*info["uphill_b2a"]))
    m += pulp.lpSum(terms)

    # DAG rank constraints
    for a,b in pairs:
        m += r[a] - r[b] >= 1 - M*(1 - x[(a,b)])
        m += r[b] - r[a] >= 1 - M*(x[(a,b)])

    # Hard hydrology rules
    for n,t in ntype.items():
        if t == "Head_water":
            for w in nbrs[n]:
                a,b = sorted((n,w), key=str)
                var = x[(a,b)]
                if n == a: m += var == 1
                else:      m += var == 0
        elif t == "Outlet":
            for w in nbrs[n]:
                a,b = sorted((n,w), key=str)
                var = x[(a,b)]
                if n == a: m += var == 0
                else:      m += var == 1
        elif t == "Junction":
            inc, out = [], []
            for w in nbrs[n]:
                a,b = sorted((n,w), key=str)
                var = x[(a,b)]
                if ntype.get(w) != "Outlet":
                    inc.append((var if n!=a else (1-var)))
                if ntype.get(w) != "Head_water":
                    out.append((var if n==a else (1-var)))
            if inc: m += pulp.lpSum(inc) >= 1
            if out: m += pulp.lpSum(out) >= 1

    solver = HiGHS_CMD(msg=False) if (prefer_highs and _HIGHS_AVAILABLE) \
             else pulp.PULP_CBC_CMD(msg=False)

    status_code = m.solve(solver)
    status = pulp.LpStatus[status_code]

    chosen = { (a,b): int(round(pulp.value(x[(a,b)]) or 0)) for a,b in pairs }

    return chosen, status, edge_info


# ------------------------------------------------
# Final build + export
# ------------------------------------------------

def build_refined_graph(Din, components, comp_results, phi):
    D = nx.DiGraph()
    D.add_nodes_from(Din.nodes(data=True))
    edge_flags = {}

    for comp_nodes, (chosen, _, _) in zip(components, comp_results):
        pairs = chosen.keys()
        for a,b in pairs:
            val = chosen[(a,b)]
            u,v = (a,b) if val==1 else (b,a)
            ed = Din.get_edge_data(a,b) or Din.get_edge_data(b,a) or {}
            D.add_edge(u, v, **ed.copy())
            edge_flags[(u,v)] = (a,b,val)

    # add phi and flags
    for (u,v) in D.edges():
        pu,pv = phi.get(u,math.inf), phi.get(v,math.inf)
        dp = None if math.isinf(pu) or math.isinf(pv) else pu-pv
        D.edges[u,v]["delta_phi"] = dp
        D.edges[u,v]["is_equal_level"] = (dp==0)

    return D


def export_outputs(D, outdir, phi, comp_results):
    # write nodes
    nrows=[]
    for n,nd in D.nodes(data=True):
        nrows.append({
            "node_id":n, "node_type":nd.get("node_type"),
            "phi":phi.get(n),
            "in_deg":D.in_degree(n),
            "out_deg":D.out_degree(n),
            "geometry":Point(nd.get("x"),nd.get("y"))
        })
    gdfn=gpd.GeoDataFrame(nrows,geometry="geometry",crs="EPSG:4326")
    gdfn.to_file(os.path.join(outdir,"river_nodes_refined.gpkg"),
                 layer="nodes",driver="GPKG")
    gdfn.drop(columns=["geometry"]).to_csv(
        os.path.join(outdir,"river_nodes_refined.csv"),index=False)

    # write edges
    erows=[]
    for u,v,ed in D.edges(data=True):
        nu,nv=D.nodes[u],D.nodes[v]
        geom=LineString([(nu.get("x"),nu.get("y")),(nv.get("x"),nv.get("y"))]) \
             if None not in (nu.get("x"),nu.get("y"),nv.get("x"),nv.get("y")) else None
        
        
        if u == ed.get("upstream_node"):
            slope = abs(ed.get("UD_slope", 0))
            if ed.get("UD_slope", 0) < 0:
                slopeF = ed.get("slopeF", 0) * -1
            else:
                slopeF = ed.get("slopeF", 0)
        else:
            slope = -abs(ed.get("UD_slope", 0))
            if ed.get("UD_slope", 0) > 0:
                slopeF = ed.get("slopeF", 0) * -1
            else:
                slopeF = ed.get("slopeF", 0)

        D[u][v]['slope'] = slope
        erows.append({
            "u":u,"v":v,
            "phi":phi.get(u),
            "delta_phi":ed.get("delta_phi"),
            "is_equal_level":ed.get("is_equal_level"),
            "upstream_node":ed.get("upstream_node"),
            "downstream_node":ed.get("downstream_node"),
            "confidence":ed.get("confidence"),
            "distance":ed.get("distance"),
            "geometry":geom,
            'section_id':ed.get("section_id"),
            'slope':slope,
            'slopeP':ed.get("slopeP"),
            'slopeF':slopeF,
            'SE':ed.get("SE"),
            'convergence':ed.get("convergence")
        })
    gdfe=gpd.GeoDataFrame(erows,geometry="geometry",crs="EPSG:4326")
    gdfe.to_file(os.path.join(outdir,"river_edges_refined.gpkg"),
                 layer="edges",driver="GPKG")
    gdfe.drop(columns=["geometry"]).to_csv(
        os.path.join(outdir,"river_edges_refined.csv"),index=False)

    # write PKL
    with open(os.path.join(outdir,"river_directed_refined.pkl"),"wb") as f:
        pickle.dump(D,f)

    # report comp stats
    with open(os.path.join(outdir,"refine_report.txt"),"w") as f:
        for i,(chosen,status,_) in enumerate(comp_results):
            f.write(f"Component {i}: status={status}, edges={len(chosen)}\n")

    print("✅ Export complete")
    return gdfe

# ------------------------------------------------
# Map new reach directions to previous SWORD graph
# ------------------------------------------------
def get_first_u(edges):
    target = None
    for edge in edges:
        if edge[3]['path_seg_pos'] == 0:
            target = edge
    if target is None:
        raise TypeError('u not found')    
    return target

def update_node_labels_from_reach_ids(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Update node labels in a MultiDiGraph so each node label becomes
    a combination of the 'reach_id' values of all directly connected edges.

    Rules:
      - Graph must be a MultiDiGraph (raises ValueError otherwise)
      - For each node:
          * Collect 'reach_id' from both incoming and outgoing edges
          * Sort the reach_ids and join with underscores, e.g., 1_2_5
          * If no edges, label becomes 'isolated_<old_label>'
      - Node labels are updated (relabelled), not stored as attributes.

    Returns the same graph (modified in place).
    """
    if not isinstance(G, nx.MultiDiGraph):
        raise ValueError("Graph must be a networkx.MultiDiGraph.")

    mapping = {}
    num_connected_reaches, up_reach, dn_reach = {}, {}, {}


    for node in list(G.nodes()):
        # Collect reach_ids from both incoming and outgoing edges
        uprid, dnrid = set(), set()

        for _, _, data in G.in_edges(node, data=True, keys=False):
            rid = data.get("reach_id")
            if isinstance(rid, int):
                uprid.add(rid)
             

        for _, _, data in G.out_edges(node, data=True, keys=False):
            rid = data.get("reach_id")
            if isinstance(rid, int):
                dnrid.add(rid)
        reach_ids = uprid | dnrid
        # Define new node label
        if not reach_ids:
            new_label = f"isolated_{node}"
        else:
            new_label = "-".join(str(rid) for rid in sorted(reach_ids))

        mapping[node] = new_label
        num_connected_reaches[new_label] = len(reach_ids)
        up_reach[new_label] = list(uprid)
        dn_reach[new_label] = list(dnrid)
        
    # Check for potential collisions and make labels unique
    seen = {}
    for old, new in list(mapping.items()):
        if new in seen:
            seen[new] += 1
            mapping[old] = f"{new}__{seen[new]}"
        else:
            seen[new] = 0

    # Relabel nodes in place
    nx.relabel_nodes(G, mapping, copy=False)
    
    updateDict = {
        'num_connected_reaches': num_connected_reaches,
        'upstream_reach':up_reach,
        'downstream_reach':dn_reach,
    }
    for attr_name, values in updateDict.items():
        nx.set_node_attributes(G, values, attr_name)

def update_edge_attributes_fast(G):
    # --- Step 1. Precompute node→edge mappings ---
    node_to_in_edges = {}
    node_to_out_edges = {}

    for u, v, k, data in G.edges(keys=True, data=True):
        reach_id = data.get('reach_id')
        if reach_id is None:
            continue
        # Incoming to v
        node_to_in_edges.setdefault(v, []).append(reach_id)
        # Outgoing from u
        node_to_out_edges.setdefault(u, []).append(reach_id)

    # --- Step 2. Write attributes explicitly via G[u][v][k] ---
    for u, v, k, data in G.edges(keys=True, data=True):
        r_id = data.get('reach_id')

        rch_id_up = node_to_in_edges.get(u, [])
        rch_id_dn = node_to_out_edges.get(v, [])
        n_rch_up = len(rch_id_up)
        n_rch_dn = len(rch_id_dn)

        # ✅ Guaranteed in-place update
        G[u][v][k].update({
            'n_rch_up': n_rch_up,
            'n_rch_dn': n_rch_dn,
            'rch_id_up': rch_id_up,
            'rch_id_dn': rch_id_dn,
            'up_network_node': u,
            'dn_network_node': v
        })

    # return G

def map_new_directions(directory):
    continent = directory[-2:]
    short_dir = directory[:-2]
    # Load graphs
    with open(directory + f"/river_directed_refined.pkl", "rb") as f:
        G_ref = pickle.load(f)

    with open(short_dir + f"/{continent}_MultiDirected.pkl", "rb") as f:
        DG = pickle.load(f)

    for u, v, k in DG.edges(keys=True):
        DG[u][v][k]['swot_direction_change'] = False

    # --- 1. Build mapping from refined graph direction ---
    lookup = defaultdict(list)
    for u, v, k, data in DG.edges(keys=True, data=True):
        key = (data['path_start_node'], data['path_end_node'])
        seg = data['path_seg']
        if seg not in lookup[key]:   # only append if not already present
            lookup[key].append(seg)
    ref_dir, slope_dir, reliable_dir = {}, {}, {}
    for u, v, data in G_ref.edges(data=True):
        seg = int(data["section_id"])
        ref_dir[seg] = (u, v)  # direction in the refined graph itself
        slope_dir[seg] = data['slope']
        reliable_dir[seg] = data['confidence']
        segs = lookup.get((u, v), -9999)
        if segs == -9999:
            segs = lookup.get((v, u), -9999)
        if len(segs) == 0:
            print('Error no reference path segments found in for matching')
        elif len(segs) > 1:
            add_seg = list(set(segs) - set([seg]))
            for s in add_seg:
                ref_dir[s] = (u, v)  # direction in the refined graph itself
                slope_dir[s] = data['slope']
                reliable_dir[s] = data['confidence']
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
            max_pos = max(e[3].get("path_seg_pos", 0) for e in edges_by_seg[seg])
            for u, v, k, d in list(edges_by_seg[seg]):
                DG.remove_edge(u, v, key=k)
                new_d = d.copy()
                new_d['swot_direction_change'] = True



                new_d["path_seg_pos"] = max_pos - d.get("path_seg_pos", 0)
                DG.add_edge(v, u, **new_d)
        else:
            print('ERROR Edge missing')
    for u,v,k,d in DG.edges(data =True,keys =True):
        s = slope_dir.get(d['path_seg'], 0)
        DG[u][v][k]['path_seg_slope'] = s
        DG[u][v][k]['path_seg_reliable'] = reliable_dir.get(d['path_seg'], 'U')
    # update network attributes
    update_node_labels_from_reach_ids(DG)
    update_edge_attributes_fast(DG)

    # write PKL
    with open(directory + f'/{continent}_MultiDirected_refined.pkl',"wb") as f:
        pickle.dump(DG,f)

    

# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_pkl",required=True)
    ap.add_argument("--outdir",required=True)
    ap.add_argument("--prefer_highs",action="store_true")
    ap.add_argument("--wR",type=float,default=1000.0)
    ap.add_argument("--wU",type=float,default=1.0)
    ap.add_argument("--wUp",type=float,default=0.001)
    args = ap.parse_args()
    os.makedirs(args.outdir,exist_ok=True)

    with open(args.input_pkl,"rb") as f:
        Din: nx.DiGraph = pickle.load(f)

    # skeleton + φ once for whole network
    G = build_undirected_skeleton(Din)
    phi = compute_phi_undirected(G)

    # components for solving
    components = list(nx.weakly_connected_components(Din))
    comp_results=[]

    for comp in components:
        D_sub = Din.subgraph(comp).copy()
        chosen,status,ein = solve_component(
            D_sub,phi,
            wR=args.wR,wU=args.wU,wUp=args.wUp,
            prefer_highs=args.prefer_highs)
        comp_results.append((chosen,status,ein))

    # assemble final D
    Dref = build_refined_graph(Din,components,comp_results,phi)

    # verify & export
    gdfe = export_outputs(Dref,args.outdir,phi,comp_results)

    # R-disagreement csv
    rows=[]
    for u,v,ed in Dref.edges(data=True):
        conf=ed.get("confidence")
        up,dn=ed.get("upstream_node"),ed.get("downstream_node")
        if conf=="R" and up is not None and dn is not None:
            if str(u)!=str(up) and str(v)!=str(up):
                # disagreement
                rows.append({
                    "u":u,"v":v,
                    "upstream_node":up,
                    "downstream_node":dn,
                    "confidence":conf,
                    "phi":ed.get("phi"),
                    "delta_phi":ed.get("delta_phi"),
                })
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir,
                                           "r_direction_conflicts.csv"),
                              index=False)

    map_new_directions(args.outdir)

    print("✅ φ+R refinement complete!")
    print("Outputs in",args.outdir)



if __name__=="__main__":
    main()
