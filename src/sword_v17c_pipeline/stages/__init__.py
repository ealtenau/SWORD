"""v17c pipeline stage modules."""

from .loading import load_topology, load_reaches, run_facc_corrections
from .graph import (
    get_effective_width,
    build_reach_graph,
    identify_junctions,
    build_section_graph,
)
from .path_variables import compute_path_variables
from .distances import compute_hydro_distances, compute_best_headwater_outlet
from .mainstem import compute_mainstem, compute_main_neighbors
from .output import save_to_duckdb, save_sections_to_duckdb, apply_swot_slopes

__all__ = [
    "load_topology",
    "load_reaches",
    "run_facc_corrections",
    "get_effective_width",
    "build_reach_graph",
    "identify_junctions",
    "build_section_graph",
    "compute_path_variables",
    "compute_hydro_distances",
    "compute_best_headwater_outlet",
    "compute_mainstem",
    "compute_main_neighbors",
    "save_to_duckdb",
    "save_sections_to_duckdb",
    "apply_swot_slopes",
]
