"""OSM river name enrichment for SWORD reaches."""

from .acquire import acquire_region, acquire_all
from .match import match_osm_names, save_osm_names

__all__ = ["acquire_region", "acquire_all", "match_osm_names", "save_osm_names"]
