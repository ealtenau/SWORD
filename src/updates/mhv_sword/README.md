# MERIT Hydro Vector (MHV) - SWORD Database
This directory houses scripts for creating a MHV database that contains a SWORD flag indicating which MHV reaches have a coincident SWORD reach. This MHV-SWORD database is used as input for extending SWORD (/channel_additions) as well as shifting offests SWORD centerlines (/centerline_shifting).  

There are five primary scripts for creating the MHV-SWORD database and they should be run in numerical order.

Script summaries:
- **1_mhv_sword_flag_db.py**: Converts MHV to points and creates a SWORD-flag indicating if MHV has a nearby SWORD reach.   
- **2_attach_global_attributes.py**: Attaches attributes from global auxillary datasets. 
- **3_attach_mhv_attributes.py**: Attaches elevation, width, and flow accumulation attributes from MERIT Hydro. 
- **4_divide_mhv_into_reaches.py**: Divides MHV into SWORD-structured reaches and nodes. 
- **5_add_coastal_flag.py**: Adds a flag indicating if a MHV reach is within 10 km of the coastline. 
- **mh_download.py**: Downloads MERIT Hydro raster files. 
- **mhv_reach_def_tools.py**: Utilities for adding auxillary data. 



