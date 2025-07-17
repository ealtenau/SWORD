# SWORD Centerline Interior Additions
Scripts for adding rivers to existing SWORD centerlines.

There are five primary scripts for adding interior MHV rivers to SWORD and they should be run in numerical order.

Script summaries:
- **1_identify_mhv_rivers_to_add.py**: Finds and exports interior MHV rivers to add to SWORD.
- **2_format_mhv_additions.py**: Formats interior MHV additions based on manual edits. 
- **3_find_mhv_tributary_breaks.py**: Finds reaches to break in SWORD based on MHV addition locations. 
- **4_attach_mhv_to_sword.py**: Adds interior MHV rivers to SWORD.
- **5_filter_zero_widths.py**: Filters out interior MHV additions with zero width values. 
- **mhv_to_sword_tools.py**: Tools for adding interior MHV rivers to SWORD. 
