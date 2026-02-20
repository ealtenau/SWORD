# Codebase Organization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorganize SWORD repo for navigability — flatten src/, archive legacy, organize scripts, consolidate dirs.

**Architecture:** Purely structural. `git mv` for moves, bulk sed for imports, no logic changes. One PR.

**Tech Stack:** git, ruff, pytest

**Design doc:** `docs/plans/2026-02-20-codebase-organization-design.md`

---

### Task 1: Move legacy code to src/_legacy/

Create `src/_legacy/{updates,development}` dirs, then move all non-active code.

**Step 1: Move everything**

```bash
mkdir -p src/_legacy/updates src/_legacy/development
git mv src/development/* src/_legacy/development/
# Legacy files from src/updates/
for f in sword.py sword_utils.py sword_vectors.py geo_utils.py auxillary_utils.py README.md; do
  git mv "src/updates/$f" src/_legacy/updates/ 2>/dev/null
done
# Legacy subdirs
for d in centerline_shifting channel_additions delta_updates formatting_scripts glows_sword_attachment mhv_sword network_analysis quality_checking; do
  git mv "src/updates/$d" src/_legacy/updates/
done
# Legacy root Streamlit apps
git mv topology_reviewer.py src/_legacy/updates/
git mv lake_reviewer.py src/_legacy/updates/
```

**Step 2: Commit**

`git commit -m "chore: move legacy code to src/_legacy/"`

---

### Task 2: Flatten src/ — move active modules up

After Task 1, `src/updates/` has only `sword_duckdb/` and `sword_v17c_pipeline/`.

**Step 1: Move and clean**

```bash
git mv src/sword_duckdb src/sword_duckdb
git mv src/sword_v17c_pipeline src/sword_v17c_pipeline
rm -rf src/updates/__pycache__
git rm -f src/updates/__init__.py 2>/dev/null || true
rmdir src/updates 2>/dev/null || rm -rf src/updates
```

**Step 2: Verify** — `ls src/` shows `_legacy/ sword_duckdb/ sword_v17c_pipeline/`

**Step 3: Commit**

`git commit -m "chore: flatten src/ — remove updates/ layer"`

---

### Task 3: Fix all imports (CRITICAL)

Bulk replace all import paths. Order matters — longer patterns first.

**Step 1: Replace imports in all .py files**

```bash
# src.sword_v17c_pipeline → src.sword_v17c_pipeline
find . -name '*.py' -not -path './_legacy/*' -not -path './src/_legacy/*' | xargs sed -i '' 's/src\.updates\.sword_v17c_pipeline/src.sword_v17c_pipeline/g'
# src.sword_duckdb → src.sword_duckdb
find . -name '*.py' -not -path './_legacy/*' -not -path './src/_legacy/*' | xargs sed -i '' 's/src\.updates\.sword_duckdb/src.sword_duckdb/g'
# sword_v17c_pipeline → sword_v17c_pipeline (sys.path=src/)
find . -name '*.py' -not -path './_legacy/*' -not -path './src/_legacy/*' | xargs sed -i '' 's/updates\.sword_v17c_pipeline/sword_v17c_pipeline/g'
# sword_duckdb → sword_duckdb (sys.path=src/)
find . -name '*.py' -not -path './_legacy/*' -not -path './src/_legacy/*' | xargs sed -i '' 's/updates\.sword_duckdb/sword_duckdb/g'
```

**Step 2: Also fix path references in .md, .yml, .toml files**

```bash
find . -name '*.md' -not -path './src/_legacy/*' | xargs sed -i '' 's/src\/updates\/sword_duckdb/src\/sword_duckdb/g; s/src\/updates\/sword_v17c_pipeline/src\/sword_v17c_pipeline/g; s/src\.updates\.sword_duckdb/src.sword_duckdb/g; s/src\.updates\.sword_v17c_pipeline/src.sword_v17c_pipeline/g; s/updates\.sword_duckdb/sword_duckdb/g; s/updates\/sword_duckdb/sword_duckdb/g'
```

**Step 3: Verify no stale references outside _legacy/**

```bash
rg "updates\.sword_duckdb|updates\.sword_v17c_pipeline|sword_duckdb|sword_v17c_pipeline" --type py --type md | grep -v "_legacy/"
```

Expected: zero matches.

**Step 4: Run ruff + tests**

```bash
ruff check src/sword_duckdb/ src/sword_v17c_pipeline/ tests/ --select E,F
python -m pytest tests/sword_duckdb/ -q -x -m "not slow and not postgres"
```

Both must pass.

**Step 5: Commit**

`git commit -m "fix: update all import paths after src/ flattening"`

---

### Task 4: Move root scripts to scripts/ subdirs

**Step 1: Create subdirs and move**

```bash
mkdir -p scripts/{topology,visualization,analysis,maintenance,sql}
# Topology
git mv run_v17c_topology.py scripts/topology/
git mv topology_investigator.py scripts/topology/
git mv topology_optimizer.py scripts/topology/
# Visualization
for f in visualize_all_pipelines.py visualize_comparison_3panel.py visualize_reach_maps.py visualize_samples.py presentation_hires.py presentation_lake_context.py presentation_materials.py; do
  git mv "$f" scripts/visualization/
done
# Analysis
git mv compare_v17b_v17c.py scripts/analysis/
# Maintenance
git mv rebuild_v17b.py scripts/maintenance/
git mv reimport_fixes.py scripts/maintenance/
git mv check_reviewer_setup.py scripts/maintenance/
# Existing scripts/ content
git mv scripts/load_from_duckdb.py scripts/maintenance/ 2>/dev/null || true
ls scripts/*.sql 2>/dev/null && git mv scripts/*.sql scripts/sql/ || true
```

**Step 2: Commit**

`git commit -m "chore: organize root scripts into scripts/ subdirectories"`

---

### Task 5: Merge model dirs and maps/

**Step 1: Move**

```bash
mkdir -p models
git mv deepwatermap models/deepwatermap
git mv ml4floods_models models/ml4floods
# maps/ into outputs/
mkdir -p outputs/maps
git mv maps/* outputs/maps/ 2>/dev/null || mv maps/* outputs/maps/ 2>/dev/null
rmdir maps 2>/dev/null || rm -rf maps
```

**Step 2: Update .gitignore** — replace `deepwatermap/`, `ml4floods_models/`, `maps/` with new paths if listed.

**Step 3: Update hardcoded paths**

```bash
rg "deepwatermap/|ml4floods_models/" --type py -l | grep -v _legacy
```

Fix matches: `deepwatermap/` → `models/deepwatermap/`, `ml4floods_models/` → `models/ml4floods/`.

**Step 4: Commit**

`git commit -m "chore: consolidate model dirs into models/, maps into outputs/"`

---

### Task 6: Reorganize docs/

**Step 1: Move loose files into subdirs**

```bash
mkdir -p docs/{technical,roadmaps,guides}
git mv docs/SWORD_v17b_Technical_Documentation.md docs/technical/
git mv docs/facc_conservation_algorithm.md docs/technical/
git mv docs/facc_correction_methodology.md docs/technical/
git mv docs/facc_correction_summary.md docs/technical/
git mv docs/v17c_status.md docs/roadmaps/
git mv docs/v17c_v18_roadmap.md docs/roadmaps/
git mv docs/v17c_plan_to_april.md docs/roadmaps/
git mv docs/reviewer_quickstart.md docs/guides/
# VALIDATION_PLAN.md — read it, move to plans/ or validation_specs/
git mv docs/VALIDATION_PLAN.md docs/plans/ 2>/dev/null || true
```

**Step 2: Commit**

`git commit -m "chore: organize docs/ into subdirectories"`

---

### Task 7: Write README files

**Files:** Create `src/_legacy/README.md` and `scripts/README.md`

See design doc for content. `_legacy/README.md` has porting status table. `scripts/README.md` has per-script descriptions grouped by subdir.

**Commit:** `git commit -m "docs: add README files for _legacy/ and scripts/"`

---

### Task 8: Update CLAUDE.md and config files

**Step 1: Update CLAUDE.md** — replace all old paths per design doc mappings. Key changes:
- `src/sword_duckdb/` → `src/sword_duckdb/`
- `src/sword_v17c_pipeline/` → `src/sword_v17c_pipeline/`
- `src/development/` → `src/_legacy/development/`
- Root script paths → `scripts/{subdir}/` paths
- Import examples: `from sword_duckdb` → `from sword_duckdb`

**Step 2: Update deploy/, .github/ if they reference old paths**

```bash
rg "sword_duckdb|sword_duckdb" deploy/ .github/ 2>/dev/null | grep -v _legacy
```

**Step 3: Final test run**

```bash
ruff check src/sword_duckdb/ src/sword_v17c_pipeline/ tests/ --select E,F
python -m pytest tests/sword_duckdb/ -q -x -m "not slow and not postgres"
```

**Step 4: Commit**

`git commit -m "docs: update CLAUDE.md and configs for new directory structure"`

---

### Task 9: Final verification

**Step 1: Check no stale refs**

```bash
rg "src/updates/|src\.updates\." --type py --type md --type yaml | grep -v _legacy
```

**Step 2: Verify structure matches design**

```bash
ls src/ && ls scripts/*/ && ls docs/{technical,roadmaps,guides}/ && ls models/
```

**Step 3: Clean up**

```bash
find . -type d -name __pycache__ -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null
test -d src/updates && echo "WARNING: src/updates still exists" || echo "OK"
```

**Step 4: Final commit if needed**

`git commit -m "chore: final cleanup after codebase reorganization"`

---

## Commit Log

| # | Message |
|---|---------|
| 1 | `chore: move legacy code to src/_legacy/` |
| 2 | `chore: flatten src/ — remove updates/ layer` |
| 3 | `fix: update all import paths after src/ flattening` |
| 4 | `chore: organize root scripts into scripts/ subdirectories` |
| 5 | `chore: consolidate model dirs into models/, maps into outputs/` |
| 6 | `chore: organize docs/ into subdirectories` |
| 7 | `docs: add README files for _legacy/ and scripts/` |
| 8 | `docs: update CLAUDE.md and configs for new directory structure` |
| 9 | `chore: final cleanup after codebase reorganization` |

## Risks

- **Import breakage** (Task 3) is the critical path — run tests immediately after
- **Legacy imports** inside `src/_legacy/` will be broken — expected, documented in README
- **deploy/reviewer/** may have its own imports — check in Task 8
- Use `git mv` for history preservation
