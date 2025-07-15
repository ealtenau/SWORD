# SWORD Network Scripts

This directory houses scripts related to SWORD network/topology updates. Scripts in the **main directory** are intended for ongoing SWORD updates, while scripts in the **/topology_v17** and **/path_building_v17** subdirectories are archived and only relevant to SWORD v17. 

Script summaries are as follows:

- **attach_topology_to_nc.py**: Updates topology in the SWORD netCDF file based on manual edits to the SWORD geopackage file.
- **cl_node_dimensions.py**: Fills in the node ID neighbor dimensions in the SWORD netCDF file. 
- **dist_out_from_topo.py**: Calculates distance from outlet based on SWORD topology. 
- **node_cl_id_reversals.py**: Reverses centerline and node IDs based on topology. 
- **update_routing_files_from_gpkg.py**: Generates routing input files. 
- **routing_topology_gpkg.py**: Routes/accumulates SWORD number of reaches based on topology. 
