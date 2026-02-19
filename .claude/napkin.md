# Napkin

## Corrections
| Date | Source | What Went Wrong | What To Do Instead |
|------|--------|----------------|-------------------|
| 2026-02-15 | self | ogr2ogr GPKG doesn't have `name:en` as column — it's in `other_tags` hstore | Always check GPKG column schema before assuming column existence; use `regexp_extract(other_tags, ...)` for extended OSM tags |
| 2026-02-15 | self | `river_name = 'NODATA'` not NULL — sentinel string | SWORD uses `'NODATA'` string, not NULL, for missing `river_name` values |
| 2026-02-15 | self | DuckDB `ALTER TABLE RENAME COLUMN` fails when views depend on table | Drop dependent views/indexes first, rename, then recreate |
| 2026-02-15 | self | Deployed DuckDB on GCS only has `reaches` table, no `nodes` | Use `reaches.geom` LINESTRING as fallback for geometry; parse WKT with `ST_AsText` |
| 2026-02-15 | self | SWORD LINESTRING geom runs downstream-to-upstream, not upstream-to-downstream | Reverse coords after parsing WKT; nodes `ORDER BY dist_out DESC` gives upstream-to-downstream |
| 2026-02-15 | self | C004 valid combos had lakeflag=1+type=3 as valid (lake+tidal_river) | **WRONG** — this "correction" was itself wrong and caused 22k false positives. See 2026-02-16 entry. |
| 2026-02-16 | self | Previous session changed lakeflag=1 valid types from (3,4) to (2,4), thinking type=3 was "tidal_river". type=3 is actually "lake_on_river" per PDD. Caused 22,580 false mismatches. type=2 doesn't even exist in SWORD data. | **SWORD type field**: 1=river, 3=lake_on_river, 4=dam, 5=unreliable, 6=ghost. NO type=2. Tidal is lakeflag=3, NOT type=3. Always check validation spec before changing valid combos. |
| 2026-02-15 | self | Substring match `"lake" in issue_type` caught "river_labeled_as_lake_type" | Use exact `==` match on issue_type strings, never substring match for dispatch |

## User Preferences
- Prefers concise plans, sacrifice grammar for brevity
- No speculative features
- `uv` not pip, `ruff` not black/pylint, `pytest -q`
- Never push to main — feature branches + PRs to gearon_dev3
- Use SCRATCHPAD.md for working memory, HANDOVER.md for session summaries

| 2026-02-16 | self (audit) | C004 fixes were logged via `log_skip()` — recorded as action='skip' with NULL old/new values, making them untrackable and un-undoable | Use `apply_column_fix()` which reads old value, does UPDATE, and logs with action='fix', column_changed, old_value, new_value |
| 2026-02-16 | self (audit) | Dead code after `st.rerun()` in C004 tab — nested button block could never execute | Code after `st.rerun()` is unreachable; delete it |
| 2026-02-16 | self (audit) | `undo_last_fix` hardcoded `lakeflag` column regardless of what `column_changed` was | Read `column_changed` from fix log and use it dynamically in the undo UPDATE |
| 2026-02-16 | user report | Refreshing deployed app lost all review progress — Cloud Run copies fresh DuckDB from GCS on each instance start, lint_fix_log and data fixes in /tmp are ephemeral | JSON session files on GCS persist; added `replay_persisted_fixes()` at startup to re-apply fixes and restore lint_fix_log from JSON |
| 2026-02-19 | self | DuckDB `DEFAULT FALSE` on Cloud Run produces NULL, not FALSE, when column omitted from INSERT column list. `NOT undone` filters out NULL rows (since `NOT NULL` = NULL = falsy). Buttons appeared to do nothing — write succeeded but row was invisible to reviewed-count query. | **Always include explicit `undone = FALSE`** (or any boolean default) in INSERT column lists. Never rely on `DEFAULT` for boolean columns in DuckDB. |

## Patterns That Work
- RTREE drop/recreate pattern for DuckDB UPDATEs on spatial tables
- Parallel background agents for independent region processing
- Parallel agents editing same file works if they touch non-overlapping string regions (Edit tool uses string match, not line numbers)
- path_freq topological summation: headwater=1, confluence=sum(upstream pf), side channels=-9999
- Check `ST_Read(...) LIMIT 0` description to detect column availability
- Skills need `SKILL.md` inside a directory (`.claude/skills/name/SKILL.md`), not flat `.md` files, for proper frontmatter discovery
- `workflow.export()` is broken — `_do_export` passes `self._sword.db` (connection) not `self._sword` (SWORD instance). Call export functions directly.
- Playwright on deployed URL is the only reliable way to reproduce Cloud Run bugs — local DuckDB behavior differs from containerized Linux DuckDB
- Debug cycle for deployed Streamlit: add logging → `bash deploy.sh` (~2 min) → test deployed URL → `gcloud logging read` — don't waste time trying to reproduce locally
- For map zoom in folium with `fit_bounds`: the zoom is driven by the extent of coords passed to `fit_bounds`, NOT `zoom_start`. Reduce `hops` to control how many reaches are fetched, which controls the bounding box.
- Single reviewer app: `reviewer.py` (local) / `app.py` (deploy). `APP_FILE` defaults to `app.py` in `start.sh`.

## Patterns That Don't Work
- Relying on DuckDB `DEFAULT FALSE` in INSERT statements — on Cloud Run (DuckDB 1.4.x) omitting a boolean column from INSERT produces NULL, not the DEFAULT value. This silently breaks `NOT column` WHERE clauses.
- Inline `if st.button():` for state mutations — use `on_click` callbacks instead. Inline handlers with `st.rerun()` have timing issues where state changes can get lost during Streamlit's execution model.
- The old topology_reviewer.py was removed — there's only one reviewer app now (`reviewer.py` / `deploy/reviewer/app.py`)

## Streamlit + Google Cloud Run Lessons

These are hard-won lessons from debugging the SWORD Lake QA Reviewer deployed on Cloud Run gen2 with GCS FUSE. Future agents: read this before touching the deployed app.

### Architecture
- **DB is ephemeral**: `start.sh` copies DuckDB from GCS FUSE (`/mnt/gcs/`) → `/tmp/sword/` on every container start. Writes to `/tmp` are lost when the container scales to zero or redeploys.
- **JSON is persistent**: Fix/skip data is backed up as JSON to `/mnt/gcs/sword/lint_fixes/` (GCS FUSE mount). On startup, `replay_persisted_fixes()` (decorated with `@st.cache_resource`) re-inserts all JSON data into the fresh DuckDB.
- **Scale-to-zero**: `min-instances=0, max-instances=1`. After ~15 min idle, container dies. Next request cold-starts a new one. All `/tmp` state is gone, only GCS JSON survives.

### Debugging Deployed Bugs
- **Local testing is insufficient**: DuckDB behavior can differ between macOS (local) and Linux (Cloud Run container). The `DEFAULT FALSE` bug only manifested on Cloud Run — local testing showed everything working.
- **Use Playwright on the deployed URL**: Navigate to the Cloud Run URL, interact with the app, check if state actually changes. This is the only reliable way to reproduce deployed-only bugs.
- **Use `gcloud logging read`**: Cloud Run logs go to GCP Logging. Filter by service name and severity. JSON writes to GCS FUSE show up as separate log entries from the GCS FUSE sidecar.
- **Add diagnostic logging, deploy, test, read logs**: The fastest debug cycle is: add `logging.info()` calls → `bash deploy.sh` → test on deployed URL → `gcloud logging read` to see what happened. Each deploy takes ~2 min.
- **Deploy script**: `cd deploy/reviewer && bash deploy.sh` — builds Docker image via Cloud Build, pushes to GCR, deploys to Cloud Run. Output shows revision number (e.g., `sword-reviewer-00022-v8f`).

### Streamlit Gotchas on Cloud Run
- **`@st.cache_resource` runs once per process**: `init_database()` creates tables and replays JSON fixes. If it errors, the whole app is broken until the container restarts.
- **`st.session_state` is per-browser-tab**: Each tab gets its own DuckDB connection in `st.session_state.conn`. But the underlying DuckDB file is shared, so writes from one tab are visible to others.
- **`on_click` callbacks vs inline `if st.button()`**: Callbacks execute before the script re-runs and are more reliable for state mutations. Inline button handlers can have timing issues with `st.rerun()`.
- **`st.cache_data.clear()` is essential after DB writes**: Without it, queries return stale cached results. Every callback that writes to DB must call `st.cache_data.clear()`.
- **Module-level `conn = get_connection()`**: The module-level variable captures the connection on first import. Callbacks reference this module-level `conn`. If the session_state connection gets replaced, the module-level `conn` can go stale.

### DuckDB on Cloud Run
- **Never rely on DEFAULT values in INSERT**: Always specify every column explicitly, especially booleans. `DEFAULT FALSE` can produce NULL instead of FALSE.
- **`NOT column` where column is NULL = NULL (falsy)**: This is standard SQL three-valued logic, but it's a nasty silent bug when you expect DEFAULT to populate the value.
- **Single-writer lock**: Only one connection can write at a time. With `max-instances=1` this is fine, but if you ever scale up, DuckDB will deadlock.
- **INSTALL spatial at build time**: The Dockerfile runs `python -c "import duckdb; duckdb.connect().execute('INSTALL spatial')"` during build so extensions are cached.

## Domain Notes
- **SWORD `type` field**: 1=river, 3=lake_on_river, 4=dam, 5=unreliable, 6=ghost. **Type=2 does NOT exist.** Type=3 is NOT tidal — tidal is lakeflag=3.
- **lakeflag=1 + type=3** is the PRIMARY expected lake combo (21k+ reaches). Do NOT flag it as mismatch.
- **type=3 "lake_on_river" is misleading** — it doesn't mean "a lake sitting on a river." It means "a lake-classified reach within the SWORD river network." All lake reaches (lakeflag=1) should have type=3. This includes lake islands, reservoir segments, wide lake sections — anything with lakeflag=1 that isn't a dam (type=4).
- **Lake sandwich = river reach (lakeflag=0) between two lake reaches (lakeflag=1)** — the archetypal case is an island in a lake being miscoded as a river. Fix is to set lakeflag=1 (making it a lake). The type mismatch (lakeflag=1 + type=1) then gets fixed separately via C004 tab (set type=3).
- SWORD name columns: `river_name` (GRWL), `river_name_en` (standardized English), `river_name_local` (OSM local name)
- 248,674 total reaches; 55,671 still completely unnamed after OSM enrichment
- OSM `name:en` coverage is sparse — only 69K reaches have real English translations
- OC is worst region for OSM coverage (Papua/Pacific Islands barely mapped)
- v17b is READ-ONLY reference; all edits go to v17c
- Region views (`af_reaches`, `na_reaches`, etc.) use `SELECT *` — must drop/recreate when renaming columns
- **42215900111** (labeled "Amur", w=1016): NOT mainstem — it's a tributary with wrong facc (11.8M, Amur-scale). Candidate for facc fix (#14).
- **main_side=0 + stream_order=-9999**: remaining 27 are correct main_side=0 but path_freq was never computed → #16 scope, not main_side bug
- Tributary mainstem → larger river confluence (e.g. Munneru→Krishna): already handled by rch_id_up/dn_main routing. Leave main_side=0.
