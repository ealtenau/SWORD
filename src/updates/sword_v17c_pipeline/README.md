# SWORD v17c Pipeline

Pipeline for generating SWORD v17c from v17b with topology enhancements.

## Pipeline Steps

1. **SWORD_graph.py** - Convert SWORD GPKG to NetworkX graph
2. **SWOT_slopes.py** - Calculate slopes from SWOT data
3. **phi_only_global.py** - Run phi algorithm for flow direction
4. **phi_r_global_refine.py** - Refine phi with additional constraints
5. **assign_attribute.py** - Assign new attributes

## New Columns Added

Core topology:
- `best_headwater`: Chosen headwater reach ID upstream
- `best_outlet`: Chosen outlet reach ID downstream
- `is_mainstem_edge`: Boolean flag for mainstem edges
- `rch_id_up_main`: Main upstream reach ID
- `rch_id_dn_main`: Main downstream reach ID

Distance metrics:
- `pathlen_hw`: Path length to headwater (m)
- `pathlen_out`: Path length to outlet (m)
- `dist_out_short`: Shorter distance to outlet
- `hydro_dist_out`: Hydrological distance to outlet
- `hydro_dist_hw`: Hydrological distance to headwater

Network structure:
- `subnetwork_id`: Subnetwork identifier
- `outlet_count`: Number of outlets downstream
- `main_path_id`: Main path identifier

Path segments:
- `path_seg`: Path segment ID
- `path_seg_pos`: Position within path segment
- `path_start_node`: Start node of path
- `path_end_node`: End node of path

Quality flags:
- `only_lake_section`: Section is lake-only
- `phi_direction_change`: Direction changed by phi
- `swot_direction_change`: Direction changed by SWOT
- `path_seg_slope`: Slope of path segment
- `path_seg_reliable`: Slope reliability flag

## Data Requirements

Input GPKG files (v17b) must be in `data/` directory:
```
data/
  {cont}_sword_nodes_v17b.gpkg
  {cont}_sword_reaches_v17b.gpkg
```

Where `{cont}` is: af, as, eu, na, oc, sa

Source location: `/Users/jakegearon/projects/sword_v17c/data/`

## Usage

### All continents:
```bash
cd src/updates/sword_v17c_pipeline
WORKDIR=/path/to/workdir ./run_all_continents.sh
```

### Single continent:
```bash
CONT=na WORKDIR=/path/to/workdir ./run_pipeline.sh
```

### Pipeline Parameters (in run_pipeline.sh):

```bash
# Slope fractions
FRACTION_LOW=-0.8
FRACTION_HIGH=0.8

# Phi refinement weights
WRCONSTANT=1000
WUCONSTANT=1
WUPCONSTANT=0.001

# Main head/outlet weights
WIDTHWEIGHT=0.6
FREQHWWEIGHT=0.2
FREQOUTWEIGHT=0.4
DISTHWWEIGHT=0.2
DISTOUTWEIGHT=0.0
```

## Output

Results are written to `output/` directory:
- `{cont}_slope_single.pkl` - Slope graph
- `{cont}/river_directed.pkl` - Directed river graph
- `{cont}_MultiDirected_refined.pkl` - Refined multi-directed graph
- Final attributed GPKG files

## Supporting Scripts

- `validate_edges.py` - Validate edge orientations
- `check_dag.py` - Check for DAG properties
- `combine_global_networks.py` - Combine continental networks
- `orient_global_edges.py` - Orient edges globally
- `ocn.py` - Optimal Channel Network algorithms
- `visualize_sections.py` - Visualization utilities
