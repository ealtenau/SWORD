# GLObal river Width from Sentinel-2 (GLOW-S) attachment to SWORD v16. 
This directory contains scripts used for attaching [GLOW-S](https://zenodo.org/records/12627519) width information to SWORD. Scripts were primarily used for width validation of SWOT widths using the SWORD v16 database.  

Script summaries:
- **calval_sword_glow-s_matchups.py**: Finds GLOW-S COMID IDs associated with SWORD calibration/validation (calval) reach IDs.      
- **glow-s_to_sword_matching.py**: Finds GLOW-S COMID IDs associated with all SWORD reach IDs.
- **glow-s_to_sword_widths.py**: Calculates and attaches GLOW-S width values to SWORD. 
- **Write_Node_Shp_Paths_GLOWS.py**: Writes SWORD node vectors with attached GLOW-S width data. 
- **Write_Reach_Shp_Paths_GLOWS.py**: Writes SWORD reach vectors with attached GLOW-S width data. 




