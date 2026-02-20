# Legacy Code Archive

Archived pre-DuckDB code from SWORD v15-v17a era. Preserved for reference — not maintained, not linted, imports may be broken.

**Do not extend this code.** Use `src/sword_duckdb/` and `src/sword_v17c_pipeline/` for all active work.

## Porting Status

| Directory | Status | Modern Equivalent | Unique Value |
|-----------|--------|-------------------|--------------|
| development/ | Archived | N/A (v15 DB creation) | Reference only |
| updates/sword.py | Superseded | sword_duckdb.SWORDWorkflow | None |
| updates/sword_utils.py | Superseded | sword_duckdb.sword_class.SWORD | None |
| updates/quality_checking/ | Superseded | sword_duckdb.lint (61 checks) | None |
| updates/network_analysis/ | Reference | reactive.py (partial) | path_freq/stream_order algorithms |
| updates/formatting_scripts/ | Reference | reactive.py (partial) | Bulk edit patterns |
| updates/channel_additions/ | Dormant | None | MHV reach expansion |
| updates/mhv_sword/ | Dormant | None | MERIT Hydro linking |
| updates/centerline_shifting/ | Dormant | imagery.RiverTracer (partial) | Legacy shifting approach |
| updates/glows_sword_attachment/ | Dormant | None | GLOW-S integration |
| updates/delta_updates/ | Dormant | None | Delta centerlines |
| updates/geo_utils.py | Partial | DuckDB spatial | Utility functions |
| updates/auxillary_utils.py | Dormant | None | Aux dataset attachment |
