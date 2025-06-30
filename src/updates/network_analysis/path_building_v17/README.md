# SWORD Path Building (Version 17 ONLY)
## _NOTE:** These scripts were retired for versions after SWORD v17_. 

This directory houses the scripts used to ingest and format shortest path files generated in QGIS. Shortest path files contain individual paths from every outlet to every headwater in the SWORD database. These shortest paths were used to calculate distance from outlet to help with topology definition in the SWORD v17 database. Script summaries are as follows:

- **path_variables_nc.py**: Ingests and formats shortest path geopackage files calculated along the SWORD v17 network in QGIS. Shortest paths are derived from every outlet to every connected headwater point in SWORD. 
- **path_vars_to_main_nc.py**: Attaches path variables to the SWORD netCDF file.
- **post_path_updates.py**: Filters path variables and calculates a network attribute.
- **stream_order.py**: Calculates stream order based on SWORD path frequency values.
- **post_manual_updates.py**: Updates the SWORD netCDF file based on any manual edits done to the SWORD geopackage files.

