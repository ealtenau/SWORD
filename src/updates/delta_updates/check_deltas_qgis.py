import os, numpy as np, pandas as pd, geopandas as gp
from shapely.geometry import LineString, Point
from src.updates.sword import SWORD
import src.updates.delta_updates.delta_utils as dlt

# inputs
region='SA'; version='v18'; name='Amazon'
p=os.getcwd()
delta_nc=f"{p}/data/inputs/Deltas/delta_updates/netcdf/added_sword_v18/{name}_delta_sword.nc"

# load SWORD (post-write) + delta
s=SWORD(p, region, version)
d=dlt.read_delta(delta_nc)

# start set: prefer SWORD additions; fallback to delta sword_rch_id_up
ef = np.array([str(v) for v in s.reaches.edit_flag])
has7 = np.array(['7' in v for v in ef])
xmin,xmax=float(np.min(d.lon)),float(np.max(d.lon))
ymin,ymax=float(np.min(d.lat)),float(np.max(d.lat))
overlap = (s.reaches.x_max >= xmin) & (s.reaches.x_min <= xmax) & \
          (s.reaches.y_max >= ymin) & (s.reaches.y_min <= ymax)
add_ids = s.reaches.id[has7 & overlap].astype(int)

sword_up = np.unique(getattr(d, 'sword_rch_id_up', np.array([0]))); 
sword_up = sword_up[sword_up > 0].astype(int)

start = set(np.concatenate([add_ids, sword_up]))
# spatial gate (expanded bbox to contain context)
xmin,xmax=float(np.min(d.lon)),float(np.max(d.lon))
ymin,ymax=float(np.min(d.lat)),float(np.max(d.lat))
pad=0.8
gate=lambda rid: (
    (s.reaches.x_max[np.where(s.reaches.id==rid)[0]][0] >= xmin-pad) and
    (s.reaches.x_min[np.where(s.reaches.id==rid)[0]][0] <= xmax+pad) and
    (s.reaches.y_max[np.where(s.reaches.id==rid)[0]][0] >= ymin-pad) and
    (s.reaches.y_min[np.where(s.reaches.id==rid)[0]][0] <= ymax+pad)
)

# id -> index map
idx={int(r):i for i,r in enumerate(s.reaches.id)}

def neigh(ids):
    us, ds = [], []
    for rid in ids:
        if rid not in idx: continue
        i=idx[rid]
        us.extend(s.reaches.rch_id_up[:,i].tolist())
        ds.extend(s.reaches.rch_id_down[:,i].tolist())
    us=[int(x) for x in us if x>0]
    ds=[int(x) for x in ds if x>0]
    return set(us)|set(ds)

# BFS expand with caps
visited=set([int(x) for x in start if x in idx and gate(int(x))])
frontier=set(visited)
max_hops=60
for _ in range(max_hops):
    nxt=neigh(frontier)
    nxt={rid for rid in nxt if (rid in idx and gate(rid))}
    nxt=nxt-visited
    if not nxt: break
    visited|=nxt
    frontier=nxt

keep=sorted(list(visited))

# reaches geometry
geoms=[]; keep_valid=[]
for rid in keep:
    cidx=np.where(s.centerlines.reach_id[0,:]==rid)[0]
    if cidx.size<2: continue
    order=np.argsort(s.centerlines.cl_id[cidx])
    x=s.centerlines.x[cidx][order]; y=s.centerlines.y[cidx][order]
    geoms.append(LineString(np.c_[x,y])); keep_valid.append(rid)

rgdf=gp.GeoDataFrame(pd.DataFrame({'reach_id':keep_valid}), geometry=geoms, crs='EPSG:4326')

# nodes geometry
nid=np.isin(s.nodes.reach_id, keep_valid)
ngdf=gp.GeoDataFrame(
    pd.DataFrame({'node_id':s.nodes.id[nid], 'reach_id':s.nodes.reach_id[nid]}),
    geometry=gp.points_from_xy(s.nodes.x[nid], s.nodes.y[nid]),
    crs='EPSG:4326'
)

# optional: delta→SWORD junction points
jx=[]; jy=[]; jto=[]
for rid in np.unique(d.reach_id[0,:]):
    idx_d=np.where(d.reach_id[0,:]==rid)[0]
    if idx_d.size==0: continue
    mx_idx=idx_d[np.argmax(d.cl_id[idx_d])]
    ups=np.unique(d.sword_rch_id_up[:,mx_idx]); ups=ups[ups>0]
    if ups.size>0:
        jx.append(d.lon[mx_idx]); jy.append(d.lat[mx_idx]); jto.append(int(ups[0]))
jgdf=gp.GeoDataFrame(pd.DataFrame({'delta_reach':np.unique(d.reach_id[0,:])[:len(jx)],'sword_up':jto}),
                     geometry=gp.points_from_xy(jx,jy), crs='EPSG:4326')

# write
out_dir=f"{p}/data/outputs/Reaches_Nodes/{version}/gpkg"
os.makedirs(out_dir, exist_ok=True)
out=f"{out_dir}/as_delta_{name}_connected_{version}.gpkg"
rgdf.to_file(out, driver='GPKG', layer='reaches_connected', overwrite=True)
ngdf.to_file(out, driver='GPKG', layer='nodes_connected')
jgdf.to_file(out, driver='GPKG', layer='junctions')
print("Wrote:", out)