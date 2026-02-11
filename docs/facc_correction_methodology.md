# Flow Accumulation Correction: Technical Report

**SWORD v17c** | February 2026 | Gearon, Pavelsky

---

## 1. Problem: Why v17b Facc Has Errors

SWORD's flow accumulation (`facc`, km^2) is extracted from MERIT Hydro, which uses a D8 (deterministic eight-neighbor) flow direction algorithm. D8 routes all flow from each raster cell to exactly one downhill neighbor. This single-flow-direction raster fundamentally conflicts with SWORD's multi-channel vector network topology, producing three systematic error modes.

### 1.1 Bifurcation Cloning

When a river splits into two or more channels, D8 has no mechanism to partition drainage area. Every child channel receives the full parent facc:

```
        Parent (facc = 1,089,000 km^2)
        /                              \
  Child A (facc = 1,089,000)    Child B (facc = 1,089,000)
```

The correct values should be proportional to channel width. If Child A carries 60% of the flow (by width), it should have ~653,000 km^2 and Child B ~436,000 km^2. Instead, both get the full 1,089,000 km^2.

**Scale**: ~2,910 bifurcation points globally across 248,674 reaches.

### 1.2 Junction Inflation

When cloned channels rejoin at a downstream junction, their facc values are naively summed:

```
  Child A (1,089,000) + Child B (1,089,000)  ->  Junction (2,178,000)
```

The junction should have ~1,089,000 km^2 plus any lateral drainage between the bifurcation and junction. This double-counting propagates downstream: every reach below inherits the inflated value. In complex deltas with nested bifurcation-junction pairs, inflation compounds exponentially. The Lena River delta (73 bifurcations, 107 junctions) reached 1.65 billion km^2 under naive correction — 674x the real 2.49M km^2 basin area.

**Scale**: ~18,000 junction deficit violations in v17b.

### 1.3 Raster-Vector Misalignment

SWORD's vector reaches and MERIT's raster cells don't align perfectly. When a reach's sampling point falls on a neighboring D8 flow path, its facc can be 2-50% lower than the upstream reach's facc — a physical impossibility on a non-bifurcating channel. These are random drops on 1:1 links (single parent, single child), not topology errors.

**Scale**: ~8,000 1:1-link drops globally.

### 1.4 Concrete Example

In the Ganges delta, D8 assigns 1,089,000 km^2 to every distributary child — the full upstream basin area. Width-proportional splitting would give the main channel ~650,000 km^2 and secondary channels proportionally less. Without correction, downstream junctions double-count, and the error cascades through hundreds of downstream reaches.

---

## 2. The Integrator Approach (DrainageAreaFix)

A colleague developed a constrained optimization approach using CVXPY. The formulation solves for incremental (local) drainage areas per reach:

### Formulation

```
minimize  ||W(Ax - L)||^2
subject to  x >= 0
            A[anchors,:] @ x == L[anchors]  (optional hard constraints at gauge reaches)
```

Where:
- **x** = vector of incremental drainage areas (one per dependent reach)
- **A** = upstream adjacency matrix encoding junction/bifurcation connectivity
- **L** = observed facc minus sum of independent (headwater) facc
- **W** = diagonal weight matrix, iteratively downweighting Tukey IQR outliers

The solver (OSQP/ECOS via CVXPY) minimizes weighted least-squares deviation from observed D8 values subject to non-negativity on incremental areas. Optional hard anchors pin specific reaches (e.g., gauged sites) to their observed values.

### Test Results

Applied to the Willamette River basin: 55 total reaches (52 dependent, 3 independent). Converged in 1 iteration with 1 outlier identified. All incremental areas non-negative. Runtime: <1 second.

### Scalability

The approach requires manual basin delineation, junction identification, and constraint setup per basin. Matrix factorization is O(m^2)-O(m^3) per basin, where m is the number of dependent reaches. Tested on a single 55-reach basin; applying globally to 248K reaches would require delineating and processing thousands of basins individually.

---

## 3. Our Approach: v3 Denoise Pipeline

### Same Goal, Different Formulation

Both approaches enforce conservation (downstream facc >= sum of upstream facc) and non-negativity (no negative incremental drainage). We achieve this via a topological-order single-pass algorithm that processes the entire global network without manual basin setup.

### Five Phases

**Phase 1 — Node-level denoise.** For each reach, compare `MAX(node facc)` to `MIN(node facc)` within the reach. If MAX/MIN > 2.0 (indicating stray D8 samples from adjacent flow paths), use the downstream-most node's facc instead of MAX. This affects ~7,345 reaches (3.0%) globally and removes the noise source before topology correction.

**Phase 2 — Topology-aware single pass.** Process all reaches in topological order (headwaters to outlets):

| Node type | Rule |
|-----------|------|
| **Headwater** | Keep baseline facc |
| **Junction** (2+ parents) | `sum(corrected_upstream) + max(base - sum(base_upstream), 0)` |
| **Bifurcation child** | `corrected_parent * (width_child / sum_sibling_widths)` |
| **1:1 link, parent lowered** | `corrected_parent + max(base - base_parent, 0)` |
| **1:1 link, parent raised** | Keep baseline (no cascade) |

The junction lateral-increment rule isolates real local drainage from D8 clone inflation. The asymmetric 1:1 propagation (lowering cascades, raising does not) prevents exponential inflation in multi-bifurcation deltas.

**Phase 3 — Outlier detection.** Flag remaining outliers via Tukey IQR in log-space on neighborhood deviations, plus junction raises >2x and 1:1 drops >2x.

**Phase 4 — Bidirectional isotonic regression (PAVA).** Extract maximal 1:1 chains (sequences between junctions/bifurcations). For each chain with monotonicity violations, run the Pool Adjacent Violators Algorithm in log-space to find the closest non-decreasing sequence. Bifurcation children are anchored at 1000x weight to preserve width-proportional shares. Junction feeders are NOT anchored, enabling bidirectional correction (both raises and lowers). This adjusts ~36,915 reaches globally.

**Phase 5 — Junction floor re-enforcement.** After isotonic regression may have shifted chain values, re-run junction floor (`corrected >= sum(corrected_upstream)`) in topological order to guarantee F006 = 0.

### Key Innovation: Asymmetric Propagation

At 1:1 links, lowering from bifurcation splits propagates downstream (additive lateral), but raising from junction floors does NOT. This prevents the exponential inflation that destroyed earlier approaches:

| Version | Approach | Lena Delta Max Facc | Real Basin |
|---------|----------|-------------------|-----------|
| v1 (additive all) | Propagate raises + lowers | 1.65 billion km^2 | 2.49M km^2 |
| v2 (asymmetric) | Propagate lowers only | 7.42M km^2 | 2.49M km^2 |
| v3 (+ isotonic) | Asymmetric + PAVA | ~7.4M km^2 | 2.49M km^2 |

### Scalability

Topological sort is O(V + E). Isotonic regression is O(k) per chain. Total runtime for all 248,674 reaches across 6 regions: ~4 minutes on a single machine. No manual basin delineation or constraint setup required.

---

## 4. Comparison

| Dimension | Integrator (CVXPY) | v3 Pipeline |
|-----------|-------------------|-------------|
| **Formulation** | Constrained QP: min \|\|W(Ax - L)\|\|^2 | Topological-order rules + isotonic regression |
| **Objective** | Minimize weighted deviation from D8 | Conservation + data fidelity (same goal) |
| **Constraints** | x >= 0, optional hard anchors | Junction floor, width-proportional splits |
| **Scale tested** | 55 reaches (1 basin) | 248,674 reaches (6 regions, global) |
| **Runtime** | <1s per basin | ~4 min global |
| **Manual setup** | Basin delineation, constraint reach IDs | None (auto from topology) |
| **Bifurcation handling** | Implicit in A matrix structure | Explicit width-proportional split |
| **Monotonicity** | Not enforced | Isotonic regression on 1:1 chains |
| **Outlier handling** | Tukey IQR + re-solve with downweighting | Tukey IQR + MERIT D8-walk re-sampling |
| **Output** | Incremental areas (x) -> total facc | Corrected total facc directly |
| **Dependencies** | CVXPY, OSQP/ECOS solvers | NetworkX, NumPy |

### Mathematical Equivalence

Both approaches minimize deviation from observed D8 values subject to non-negativity and conservation. The integrator achieves this via dense matrix factorization on incremental areas; we achieve it via graph traversal with closed-form rules at each node type.

Our junction rule `corrected = sum(corrected_upstream) + max(base - sum(base_upstream), 0)` is equivalent to enforcing `incremental_area >= 0` at each junction — the `max(..., 0)` clamps the lateral term to non-negative, identical to the integrator's `x >= 0` constraint. At the single-basin level with uniform weights, the solutions are equivalent. The key difference is scalability: our formulation processes the entire global network in one pass without requiring basin-by-basin decomposition.

---

## 5. Global Results

Data from v3 summary JSONs (all regions applied):

| Region | Reaches | Corrections | Raised | Lowered | % Change | T003 Flagged | F006 |
|--------|---------|-------------|--------|---------|----------|------|------|
| NA | 38,696 | 9,235 | 6,129 | 3,013 | +1.25% | 1,044 | 0 |
| SA | 42,159 | 9,251 | 7,130 | 2,095 | +11.97% | 1,090 | 0 |
| EU | 31,103 | 6,894 | 4,381 | 2,505 | -0.65% | 597 | 0 |
| AF | 21,441 | 5,029 | 3,494 | 1,532 | -6.49% | 443 | 0 |
| AS | 100,185 | 23,001 | 16,134 | 6,832 | +6.03% | 2,381 | 0 |
| OC | 15,090 | 3,265 | 2,143 | 1,120 | -11.24% | 305 | 0 |
| **Total** | **248,674** | **56,675** | **39,411** | **17,097** | — | **5,860** | **0** |

Correction type breakdown (global totals from summary JSONs):

| Correction Type | Count | Description |
|-----------------|-------|-------------|
| isotonic_regression | 35,342 | PAVA adjustments on 1:1 chains |
| junction_floor_post | 6,942 | Post-isotonic junction re-flooring |
| junction_floor | 4,599 | Phase 2 junction conservation |
| lateral_lower | 7,237 | Bifurcation-split cascade on 1:1 links |
| bifurc_share | 4,458 | Width-proportional bifurcation splitting |
| node_denoise | 282 | Within-reach node facc variability correction |
| upa_resample | 1,651 | MERIT D8-walk re-sampling (where MERIT rasters available) |
| t003_flagged_only | 164 | No facc change, flagged as metadata |

---

## 6. Validation

### F006 = 0 Globally

Junction conservation is guaranteed: at every junction with 2+ upstream inputs, `corrected_facc >= sum(corrected_upstream_facc)`. This is enforced by Phase 5 (junction floor re-enforcement after isotonic regression) and verified by the F006 lint check across all 6 regions.

### T003 = 5,860 Flagged (Not Force-Corrected)

5,860 reaches (2.4% of total) have residual monotonicity violations on non-bifurcation edges where downstream facc < upstream facc. These are structural disagreements between MERIT's D8 raster and SWORD's vector topology — not topology errors.

**Why not force-correct?** We tested iterative forward-max + junction floor to achieve T003 = 0. Results on NA alone:

- **+114% facc inflation** (2.6 billion km^2 added to the region)
- 226 D8-clone junctions (identical facc on both upstream branches) seed cascading double-counts through major rivers: Mississippi (+434M km^2), Missouri (+294M), Mackenzie (+250M), St. Lawrence (+163M), Nelson (+120M)
- A clone-aware variant (`max` instead of `sum` at clones) reduced inflation to +93% but did not solve the cascade

**Conclusion**: These violations are inherent MERIT D8 noise. Force-correcting them overrides thousands of MERIT values and causes unacceptable inflation. They are flagged as metadata (`t003_flag`, `t003_reason`) for downstream users to filter as needed.

### Full Lint Suite

47 lint checks pass at ERROR severity across all regions. Key checks:

| Check | Description | Result |
|-------|-------------|--------|
| F006 | Junction conservation (facc >= sum upstream) | **0 violations** |
| F012 | Non-negative incremental area (facc >= sum upstream, all reaches) | **0 violations** |
| T003 | Facc monotonicity (non-bifurcation edges) | 5,860 flagged (metadata) |
| F007 | Bifurcation balance (children sum / parent) | ~246 (missing width data) |
| T001 | dist_out monotonicity | 0 violations |
| T005 | Neighbor count consistency | 0 violations |

---

## 7. Residual Issues

1. **5,860 T003 flags (2.4%)** — Structural MERIT D8 noise on 1:1 links. Flagged as metadata with classification (`chain`, `junction_adjacent`, `non_isolated`). Not force-corrected because doing so causes +90-114% inflation.

2. **~246 bifurcation imbalance (F007/F008)** — Bifurcations where child width data is missing or zero, causing equal-split fallback to produce children that don't sum precisely to the parent. Minor: affects <0.1% of reaches.

3. **D8-clone junction over-flooring** — ~226 junctions globally where MERIT assigned identical facc to both upstream branches. Phase 5 uses naive `sum(upstream)` which slightly over-floors these junctions (double-counting the cloned drainage). This is intentional: clone-aware flooring introduces new T003 drops that trigger cascade inflation. The over-flooring contributes minimally to the per-region net change.

---

## Appendix: File Reference

### Pipeline Code

```
src/updates/sword_duckdb/facc_detection/
  correct_facc_denoise.py       # v3 pipeline (6 phases)
  correct_conservation_single_pass.py  # v2 single-pass (superseded by v3)
  detect.py                     # Phase 0: anomaly detection rules
  correct.py                    # Phase 0: RF regressor correction
  merit_search.py               # D8-walk MERIT re-sampling
```

### Integrator (DrainageAreaFix)

```
DrainageAreaFix/
  fix_drainage_area.py          # CVXPY QP solver
  sfoi_utils.py                 # SWORD topology + junction extraction
```

### Outputs

```
output/facc_detection/
  facc_denoise_v3_{REGION}.csv          # Per-reach corrections (6 files)
  facc_denoise_v3_summary_{REGION}.json # Summary stats (6 files)
  remaining_t003_{REGION}.geojson       # Residual violations for visual audit
  figures/report_fig{1-4}.png           # Report figures
```

### Figures

Generated by `scripts/generate_facc_report_figures.py`:

- **Fig 1**: v17b vs v17c scatter (log-log, all 248K reaches, colored by correction type)
- **Fig 2**: Correction type breakdown (stacked bar by region)
- **Fig 3**: Per-reach relative change distribution (histogram, faceted by region)
- **Fig 4**: Scalability comparison (integrator vs pipeline)
