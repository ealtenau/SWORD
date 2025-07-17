# SWORD Centerline Additions/Extensions
This directory contains scripts for extending SWORD centerlines based on MERIT Hydro Vector (MHV). The custom **MHV-SWORD database** built in the **src/updates/mhv_sword directory** is used to identify and add MHV rivers to SWORD. Additions are completed in two rounds: One for interior extensions to existing SWORD rivers (/interior) and one for adding coastal rivers that do not exist in the current SWORD version (/coastal). Scripts for both types of additions are similar but have slight differences based on how the river additions relate to SWORD. 

Directory summaries:
- **/coastal**: Scripts for adding rivers to existing SWORD centerlines. 
- **/interior**: Scripts for adding coastal rivers not in SWORD.
- **/tools**: Scripts for post-formatting and quality checking the MHV additions to SWORD.

SWORD extensions for v18:
![Fig1](https://github.com/ealtenau/SWORD/blob/main/docs/figures%20/global_extensions.png)