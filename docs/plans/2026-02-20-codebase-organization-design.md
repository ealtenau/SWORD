# Codebase Organization Design

**Date:** 2026-02-20
**Branch:** codebase-organization
**Approach:** Big-bang (single PR, all moves + import fixes)

## Motivation

Navigability — for both humans and LLMs. The repo has 17 root-level scripts, legacy code mixed with active code, and several directories that should be consolidated.

## Constraints

- Nothing deleted — legacy code archived, not removed
- No logic changes — purely structural (file moves + import path updates)
- Backward compatibility preserved for legacy code (archived intact)

## Current → Target Layout

### Before

```
SWORD/
├── *.py (17 root scripts, mixed purpose)
├── src/
│   ├── development/              # v15-era DB creation
│   └── updates/
│       ├── sword.py, sword_utils.py, geo_utils.py, auxillary_utils.py
│       ├── sword_duckdb/         # ACTIVE core module
│       ├── sword_v17c_pipeline/  # ACTIVE pipeline module
│       ├── centerline_shifting/  # legacy
│       ├── channel_additions/    # legacy
│       ├── delta_updates/        # legacy
│       ├── formatting_scripts/   # legacy
│       ├── glows_sword_attachment/ # legacy
│       ├── mhv_sword/            # legacy
│       ├── network_analysis/     # legacy
│       └── quality_checking/     # legacy (superseded by lint)
├── scripts/                      # load_from_duckdb.py + SQL files
├── deepwatermap/                 # ML model weights
├── ml4floods_models/             # ML model weights
├── maps/                         # generated maps
├── docs/                         # loose files + subdirs
└── ...
```

### After

```
SWORD/
├── src/
│   ├── sword_duckdb/             # core module (flattened from updates/)
│   ├── sword_v17c_pipeline/      # pipeline module (flattened from updates/)
│   └── _legacy/
│       ├── README.md             # what's here, why, porting status
│       ├── development/
│       │   ├── merging_databases/
│       │   ├── preprocessing_scripts/
│       │   └── reach_definition/
│       └── updates/
│           ├── sword.py
│           ├── sword_utils.py
│           ├── geo_utils.py
│           ├── auxillary_utils.py
│           ├── sword_vectors.py
│           ├── centerline_shifting/
│           ├── channel_additions/
│           ├── delta_updates/
│           ├── formatting_scripts/
│           ├── glows_sword_attachment/
│           ├── mhv_sword/
│           ├── network_analysis/
│           └── quality_checking/
├── scripts/
│   ├── README.md                 # index of all scripts by category
│   ├── topology/
│   │   ├── run_v17c_topology.py
│   │   ├── topology_investigator.py
│   │   └── topology_optimizer.py
│   ├── visualization/
│   │   ├── visualize_all_pipelines.py
│   │   ├── visualize_comparison_3panel.py
│   │   ├── visualize_reach_maps.py
│   │   ├── visualize_samples.py
│   │   ├── presentation_hires.py
│   │   ├── presentation_lake_context.py
│   │   └── presentation_materials.py
│   ├── analysis/
│   │   └── compare_v17b_v17c.py
│   ├── maintenance/
│   │   ├── rebuild_v17b.py
│   │   ├── reimport_fixes.py
│   │   ├── check_reviewer_setup.py
│   │   └── load_from_duckdb.py   # moved from scripts/
│   └── sql/                      # existing SQL files from scripts/
├── tests/                        # unchanged
├── docs/
│   ├── README.md
│   ├── technical/
│   │   ├── SWORD_v17b_Technical_Documentation.md
│   │   ├── facc_conservation_algorithm.md
│   │   ├── facc_correction_methodology.md
│   │   └── facc_correction_summary.md
│   ├── roadmaps/
│   │   ├── v17c_status.md
│   │   ├── v17c_v18_roadmap.md
│   │   └── v17c_plan_to_april.md
│   ├── guides/
│   │   └── reviewer_quickstart.md
│   ├── plans/                    # existing (design docs)
│   │   └── *.md
│   ├── figures/                  # existing
│   └── validation_specs/         # existing (28 specs)
├── deploy/                       # unchanged (canonical Streamlit app)
├── data/                         # unchanged (DuckDB + NetCDF)
├── models/
│   ├── deepwatermap/             # merged from root deepwatermap/
│   └── ml4floods/                # merged from root ml4floods_models/
├── outputs/                      # existing + merged maps/
├── notebooks/                    # unchanged
└── (config: pytest.ini, requirements*.txt, .gitignore, .env.example, etc.)
```

## Import Path Changes

All imports referencing `sword_duckdb` or `sword_v17c_pipeline` need updating:

| Old | New |
|-----|-----|
| `from sword_duckdb import ...` | `from sword_duckdb import ...` |
| `from sword_duckdb.lint import ...` | `from sword_duckdb.lint import ...` |
| `from sword_v17c_pipeline import ...` | `from sword_v17c_pipeline import ...` |
| `import sword_duckdb` | `import sword_duckdb` |

Also update any hardcoded paths to:
- `deepwatermap/` → `models/deepwatermap/`
- `ml4floods_models/` → `models/ml4floods/`
- Root-level scripts (if referenced by path anywhere)

## Legacy README Content

`src/_legacy/README.md` should contain:
- Purpose: archived pre-DuckDB code, preserved for reference
- Porting status table (which dirs are superseded vs have unique value)
- Warning: not maintained, not linted, may have broken imports
- Pointer to modernized equivalents in `sword_duckdb/` and `sword_v17c_pipeline/`

## Scripts README Content

`scripts/README.md` should contain:
- One-line description per script
- Grouped by subdirectory category
- Usage examples for key scripts (run_v17c_topology, rebuild_v17b, load_from_duckdb)

## Root-Level Streamlit Apps

`topology_reviewer.py` and `lake_reviewer.py` at root are legacy copies.
Canonical app is in `deploy/reviewer/`. Root copies move to `src/_legacy/updates/`.

## VALIDATION_PLAN.md Placement

Content determines destination:
- If it's a plan/checklist → `docs/plans/`
- If it's a validation spec → `docs/validation_specs/`

## Files That Stay at Root

- README.md, CLAUDE.md, DEVELOPMENT_STATUS.md
- pytest.ini
- requirements.txt, requirements-reviewer.txt, environment.yml
- .env.example, .gitignore
- pyproject.toml (if exists)

## Risk Mitigation

- Run `ruff check` after all moves to catch broken imports
- Run `pytest tests/sword_duckdb/ -q` to verify nothing broke
- Git tracks renames cleanly (move, not delete+create)
- One commit per logical group within the single PR for clean history
