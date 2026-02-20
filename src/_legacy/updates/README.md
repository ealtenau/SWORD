# SWORD Updates
SWORD is a continuosly evolving database that requires updates based on feedback from the SWOT Alorithm Team and end users. There is no need to rederive SWORD's entire structure (as detailed in the **/development** directory) for improvements, therefore, the **/updates** directory contains scripts and modules necessary for repetitive and/or bespoke updates to SWORD. 

Scripts located in the **main directory** directory contain the SWORD class and important utilities for reading, writing, manipulating the database, while the **subdirectories** contain scripts related to specific types of SWORD updates. 

Script summaries:
- **auxillary_utils.py**: Contains important utilities for processing auxillary datasets. 
- **geo_utils.py**: Contains general geoprocessing utilities. 
- **sword_utils.py**: Contains important utilities for the SWORD class. 
- **sword_vectors.py**: Script for writing SWORD vector files. 
- **sword.py**: SWORD class object. 

Directory summaries:
- **/centerline_sifting**: Scripts for shifting offset centerlines in SWORD. 
- **/channel_additions**: Scripts for extending SWORD using MERIT Hydro Vector. 
- **/delta_updates**: Scripts for adding updated delta centerlines to SWORD. 
- **/formatting_scripts**: Scripts for general SWORD formatting updates or changes. 
- **/mhv_sword**: Scripts for building a MERIT Hydro Vector - SWORD translation database. 
- **/network_analysis**: Scripts related to SWORD network/topology updates. 
- **/quality_checking**: Scripts to quality check SWORD's structure and formatting. 
