# SWORD Topology Updates (Version 17 ONLY)
## _NOTE:** These scripts were retired for versions after SWORD v17_. 

This directory houses the orginial scripts written by Elyssa Collins (modified by Elizabeth Altenau), to help correct and update topology in the SWORD v17 database. Script summaries are as follows:

- **1_init_geom.py**: Identifies geometric intersections in the SWORD geopackage files. 
- **2_SWORD_Prep_Topo_Clean.py**: Allows for automatic-manual iteration for fixing SWORD geometric intersections. 
- **3_SWORD_Geom_fast_v2.py**: Determining river connectivity using SWORD polyline geometry.
- **4_SWORD_Topo_Geom_auto.py**: Defining topology attributes in SWORD using polyline geometry. 
- **5_CheckTopology_Route_MultiDown.py**: Routing topology in SWORD for visual validation. 
- **check_junction_connections.py**: Helps automatically identify poor junction geometry in the SWORD polylines. 
- **rmv_hw_ol_from_gaps.py**: Removes end reaches from identified gap list produced in 2_SWORD_Prep_Topo_Clean.py. 
- **update_routing_files_manual.py**: Reproduces the routing input files needed in 5_CheckTopology_Route_MultiDown.py after any manual edits to the SWORD geopackage files. 

