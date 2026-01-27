# v17c Pipeline WITHOUT phi Optimization

This alternative pipeline computes v17c attributes using the **original v17b topology** instead of phi/MILP optimization.

## Rationale

The original v17b topology was validated and found to be:
- ✅ A valid DAG (no cycles)
- ✅ 100% dist_out monotonicity (decreases downstream on all edges)
- ✅ 91.6% of sections have VALID flow direction (SWOT slopes match expected signs)
- ⚠️ 8.4% have potential issues, but most are explainable:
  - Lake/reservoir sections (207 sections)
  - Potential topology errors (124 sections = ~2%)
  - Extreme slope data errors (6 sections)

**Conclusion**: The original topology is correct for ~98% of the network.

## SWOT Slope Validation (Junction-Level)

**IMPORTANT**: SWOT slopes are computed along **sections** (junction-to-junction paths), not individual reaches. The validation uses this structure:

For each section (chain of reaches between junctions):
- **Upstream junction slope** should be NEGATIVE (WSE decreases as you move away)
- **Downstream junction slope** should be POSITIVE (WSE increases as you move away)

If both slopes have correct signs → topology is valid for this section.

### Why This Matters

The old approach (checking per-reach slope signs) was **incorrect** because:
1. SWOT slopes measure local gradients affected by wind, waves, timing
2. The reduced graph structure defines how slopes are computed
3. A "negative slope" at one point doesn't mean wrong direction

## Files

| File | Description |
|------|-------------|
| `v17c_nophi.py` | Main script - computes all attributes without phi |
| `create_v17b_graph.py` | Creates directed graph from v17b topology |
| `validate_topology.py` | Validates any topology graph |
| `run_pipeline_nophi.sh` | Shell script for full pipeline |

## Usage

```bash
# Full computation with junction-level slope validation
python v17c_nophi.py \
    --continent NA \
    --db-path /path/to/sword_v17b.duckdb \
    --v17c-db-path /path/to/sword_v17c.duckdb \
    --output-dir output

# Skip slope validation (faster)
python v17c_nophi.py \
    --continent NA \
    --db-path /path/to/sword_v17b.duckdb \
    --output-dir output \
    --skip-slopes

# Validate topology only (no attributes)
python create_v17b_graph.py \
    --continent NA \
    --workdir . \
    --db-path /path/to/sword_v17b.duckdb \
    --validate-only
```

## Outputs

| File | Description |
|------|-------------|
| `{cont}_v17c_nophi_attrs.parquet` | New attributes per reach |
| `{cont}_sections.parquet` | Section definitions (junction-to-junction) |
| `{cont}_section_slope_validation.parquet` | Junction-level slope validation per section |
| `{cont}_reach_slope_validation.parquet` | Reach-to-section mapping with validation status |
| `{cont}_suspect_sections.parquet` | Sections requiring manual review |
| `{cont}_reach_graph.pkl` | NetworkX DiGraph (reach-level) |
| `{cont}_section_graph.pkl` | NetworkX DiGraph (junction-level) |

## Computed Attributes

| Attribute | Description |
|-----------|-------------|
| `hydro_dist_out` | Distance to outlet (recomputed via Dijkstra) |
| `hydro_dist_hw` | Distance from headwater (max across all HWs) |
| `best_headwater` | Optimal headwater reach ID (by width + path length) |
| `best_outlet` | Optimal outlet reach ID |
| `pathlen_hw` | Path length from best headwater |
| `pathlen_out` | Path length to best outlet |
| `path_freq` | Number of headwaters upstream (convergence count) |
| `is_mainstem` | True if on path from best_headwater to best_outlet |

## Slope Validation Results (NA example)

| Metric | Value |
|--------|-------|
| Total sections with SWOT data | 3,993 |
| Direction VALID (slopes match expected) | 3,659 (91.6%) |
| Direction INVALID | 334 (8.4%) |

### Breakdown of Invalid Sections

| Cause | Count | Action |
|-------|-------|--------|
| Lake sections | 207 | Expected - lakes have complex hydraulics |
| **Potential topology errors** | **124** | **Requires manual review** |
| Extreme slope data errors | 6 | Data quality issue |

## Comparison with phi Pipeline

| Aspect | phi Pipeline | nophi Pipeline |
|--------|--------------|----------------|
| Flow direction | MILP optimization | Original v17b |
| Edge changes | ~5% modified | 0% modified |
| Validation | Requires SWOT slopes | Uses existing topology + validation |
| Speed | Slower (MILP solver) | Faster |
| Attributes | Same | Same |
| Suspect sections identified | N/A | 124 potential errors |

## When to Use

**Use nophi when:**
- You trust the original v17b topology
- You want faster processing
- You're comparing against v17b
- You want junction-level slope validation

**Keep phi available when:**
- New regions with known topology errors
- Fixing the 124 identified suspect sections
- Research comparing optimization approaches
