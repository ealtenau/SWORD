"""
SWORD v17c Pipeline
===================

This module contains the complete pipeline for generating SWORD v17c from v17b.

Pipeline Steps:
1. SWORD_graph.py - Convert SWORD to NetworkX graph
2. SWOT_slopes.py - Calculate slopes from SWOT data
3. phi_only_global.py - Run phi algorithm for flow direction
4. phi_r_global_refine.py - Refine phi with additional constraints
5. assign_attribute.py - Assign new attributes (best_headwater, best_outlet, etc.)

New Columns Added by Pipeline:
- best_headwater: Chosen headwater reach ID upstream
- best_outlet: Chosen outlet reach ID downstream  
- is_mainstem: Boolean flag for mainstem reaches
- is_mainstem_edge: Boolean flag for mainstem edges

Usage:
    cd src/updates/sword_v17c_pipeline
    ./run_all_continents.sh
    
Or for a single continent:
    CONT=na ./run_pipeline.sh
"""

__version__ = "1.0.0"
