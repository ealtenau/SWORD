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
- Always create PRs — never merge locally, never ask which option
- After PR, ask: self-review or agent review (sonnet, not haiku)
- Prefers concise plans, sacrifice grammar for brevity
- No speculative features
- `uv` not pip, `ruff` not black/pylint, `pytest -q`
- Never push to main — feature branches + PRs to v17c-updates
- Use SCRATCHPAD.md for working memory, HANDOVER.md for session summaries

| 2026-02-16 | self (audit) | C004 fixes were logged via `log_skip()` — recorded as action='skip' with NULL old/new values, making them untrackable and un-undoable | Use `apply_column_fix()` which reads old value, does UPDATE, and logs with action='fix', column_changed, old_value, new_value |
| 2026-02-16 | self (audit) | Dead code after `st.rerun()` in C004 tab — nested button block could never execute | Code after `st.rerun()` is unreachable; delete it |
| 2026-02-16 | self (audit) | `undo_last_fix` hardcoded `lakeflag` column regardless of what `column_changed` was | Read `column_changed` from fix log and use it dynamically in the undo UPDATE |
| 2026-02-16 | user report | Refreshing deployed app lost all review progress — Cloud Run copies fresh DuckDB from GCS on each instance start, lint_fix_log and data fixes in /tmp are ephemeral | JSON session files on GCS persist; added `replay_persisted_fixes()` at startup to re-apply fixes and restore lint_fix_log from JSON |

| 2026-02-19 | self | Linter strips unused imports between individual Edit calls | When adding imports that won't be used until later edits, use a single Python script via Bash to apply all edits atomically before the linter runs |
| 2026-02-19 | self | SWORDWorkflow stores user_id as `_user_id` (private), not `user_id` | Use `getattr(workflow, "_user_id", default)` when accessing user_id from outside the class |
| 2026-02-19 | self | Reassigning `workflow`/`conn` inside an inner function doesn't update the outer caller's `finally` block — leaks the new connection | Don't close/reopen connections in inner functions; restructure so the caller owns the lifecycle (e.g. run facc BEFORE opening workflow) |
| 2026-02-19 | self | DuckDB RTREE index blocks UPDATE even in test fixtures | Always call `conn.execute("INSTALL spatial; LOAD spatial;")` before UPDATE on RTREE-indexed tables (reaches) |
| 2026-02-19 | self | `git stash` only saves tracked changes — untracked file deletions and unstaged changes to already-modified-but-untracked files are lost | Stage changes before stashing, or avoid stash when verifying pre-existing lint errors |
| 2026-02-19 | self | _phase4d_node_validation spiked reaches to node_max when corrected < 10% of node_max, then B5 propagated spikes via bifurc children → junction floors → 5-10x inflation on large rivers | Removed B4 (node validation) from both Stage A and Stage B; replaced with empty dicts to preserve return signature |
| 2026-02-19 | self (review) | `save_to_duckdb` in output.py only loaded spatial extension but didn't drop/recreate RTREE indexes — same segfault risk documented in CLAUDE.md | Always apply the full RTREE drop/recreate pattern (drop → UPDATE → recreate), not just LOAD spatial |
| 2026-02-19 | self (review) | `conn.register("name", df)` without try/finally leaks on exception — subsequent calls get "Table already exists" | Always wrap `conn.register` / `conn.unregister` in try/finally |
| 2026-02-19 | self (review) | DuckDB gate opening read-only LintRunner while write connection is active causes stale reads | Call `conn.execute("CHECKPOINT")` before opening a second connection to the same DB |

| 2026-02-20 | code review | N006 boundary_dist_out used MAX(node_id) for upstream reach's downstream boundary and MIN(node_id) for downstream reach's upstream boundary — backwards given SWORD convention (node_id increases upstream) | Upstream reach's downstream boundary = MIN(node_id); downstream reach's upstream boundary = MAX(node_id). Keep threshold at 1000m. |
| 2026-02-20 | code review | N010 node_index_contiguity used `max - min + 1` formula assuming step-1 suffixes, but SWORD uses step-10 (001, 011, 021, ..., 991) | Use `(max - min) / 10 + 1` for expected count |
| 2026-02-20 | user decision | N006 threshold discussed — 1000m kept as-is per user preference | Do not change N006 threshold from 1000m |

| 2026-02-20 | self | Initially designed dn_node_id/up_node_id using MIN/MAX(node_id), but test fixture had opposite convention from production | Always use dist_out for semantic ordering, not node_id — flow direction changes can reorder node IDs |

| 2026-02-20 | self | Conflict marker hook (`^={7}`) triggers on RST docstring underlines (e.g. `===============`) | Change RST underlines to dashes, or ensure docstring headers don't start at column 0 with 7+ `=` chars |

## Patterns That Work
- `SWORDWorkflow.__new__(SWORDWorkflow)` creates uninitialized workflow for testing aggregation methods in isolation with raw DuckDB connections
- RTREE drop/recreate pattern for DuckDB UPDATEs on spatial tables
- Parallel background agents for independent region processing
- Parallel agents editing same file works if they touch non-overlapping string regions (Edit tool uses string match, not line numbers)
- path_freq topological summation: headwater=1, confluence=sum(upstream pf), side channels=-9999
- Check `ST_Read(...) LIMIT 0` description to detect column availability
- Skills need `SKILL.md` inside a directory (`.claude/skills/name/SKILL.md`), not flat `.md` files, for proper frontmatter discovery
- `workflow.export()` is broken — `_do_export` passes `self._sword.db` (connection) not `self._sword` (SWORD instance). Call export functions directly.

## Domain Notes
- **SWORD `type` field**: 1=river, 3=lake_on_river, 4=dam, 5=unreliable, 6=ghost. **Type=2 does NOT exist.** Type=3 is NOT tidal — tidal is lakeflag=3.
- **lakeflag=1 + type=3** is the PRIMARY expected lake combo (21k+ reaches). Do NOT flag it as mismatch.
- SWORD name columns: `river_name` (GRWL), `river_name_en` (standardized English), `river_name_local` (OSM local name)
- 248,674 total reaches; 55,671 still completely unnamed after OSM enrichment
- OSM `name:en` coverage is sparse — only 69K reaches have real English translations
- OC is worst region for OSM coverage (Papua/Pacific Islands barely mapped)
- v17b is READ-ONLY reference; all edits go to v17c
- Region views (`af_reaches`, `na_reaches`, etc.) use `SELECT *` — must drop/recreate when renaming columns
- **42215900111** (labeled "Amur", w=1016): NOT mainstem — it's a tributary with wrong facc (11.8M, Amur-scale). Candidate for facc fix (#14).
- **main_side=0 + stream_order=-9999**: remaining 27 are correct main_side=0 but path_freq was never computed → #16 scope, not main_side bug
- Tributary mainstem → larger river confluence (e.g. Munneru→Krishna): already handled by rch_id_up/dn_main routing. Leave main_side=0.
