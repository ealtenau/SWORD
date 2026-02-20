# Facc Conservation: Single-Pass Correction Algorithm

## Problem

SWORD's `facc` (flow accumulation, km^2) comes from MERIT Hydro's D8 flow direction grid. D8 routing is single-path: at every cell, water flows to exactly one downhill neighbor. This creates two systematic errors in a multi-channel river network:

### 1. Bifurcation Cloning

When a river splits into two (or more) channels, D8 assigns the **full parent facc to every child**. There is no flow partitioning.

```
        Parent (facc = 1000)
        /                  \
  Child A (facc = 1000)   Child B (facc = 1000)   <-- both get 1000, should sum to 1000
```

The correct values depend on the relative channel size. If Child A is 3x wider than Child B, the split should be ~750 / ~250.

### 2. Junction Inflation

When the split channels rejoin at a downstream junction, their facc values are summed. Since both were cloned (not split), the junction's facc is double-counted:

```
  Child A (1000) + Child B (1000)  →  Junction (2000)
```

The junction should have ~1000 (plus any lateral drainage between the bifurcation and junction), not 2000. This error propagates downstream — every reach below the junction inherits the inflated value.

### 3. Nested Bifurcation Compounding

In complex deltas (e.g. the Lena with 73 bifurcation points and 107 junctions), bifurcations are nested — channels from one bifurcation split again before rejoining. If junction corrections propagate downstream on 1:1 links, each nested bifurcation-junction pair re-inflates values. The Lena delta inflated **674x** under the naive `max(base, sum_upstream)` junction rule with additive lateral propagation.

The root cause: at inner junctions of nested bifurcations, the D8 base value is the full cloned parent (e.g. 2.4M km^2 for the entire Lena basin). When `max(base, sum_corrected_upstream)` is used, the max recovers this inflated clone whenever the corrected upstream sum (from width splits) is less than the clone. This undoes the bifurcation correction, and the inflated value propagates downstream through 1:1 links to the next junction, where it compounds further.

### Scale of the Problem

SWORD v17b has **2,910 bifurcation points** across 248,674 reaches globally. Without correction, the inflation compounds: a river with multiple bifurcation/junction pairs can have facc inflated 2x, 4x, or more by the outlet.

## Algorithm

One pass, headwater to outlet, in **topological order** (every node is processed after all its upstream predecessors).

### Design Principles

1. **Bifurcations are the only D8 error to fix.** D8 clones the full parent facc to every child. We replace this with width-proportional splitting.
2. **Lowering propagates, raising does not.** When a bifurcation split lowers a child's facc, that reduction must cascade downstream (ratio-based). When a junction raises facc above the D8 base (because corrected upstream values sum to more), that raise must NOT cascade — it would compound through nested bifurcation-junction pairs.
3. **Lateral drainage is preserved at junctions.** The D8-measured difference between a junction's basin area and the sum of its upstream basins represents real local drainage. This increment is added on top of the corrected upstream sum.
4. **D8 values are trusted on 1:1 links** unless the parent was lowered by a bifurcation split.

### Input

- **v17b facc**: Original MERIT Hydro D8 values (read-only reference)
- **v17c topology**: `reach_topology` table (upstream/downstream neighbor links)
- **v17c width**: Per-reach channel width (meters)

### Definitions

```
baseline[n]   = v17b facc for reach n (original D8 value, never modified)
corrected[n]  = output facc for reach n (computed by this algorithm)
preds(n)      = set of upstream predecessors of n in the topology graph
succs(n)      = set of downstream successors of n
out_degree(n) = |succs(n)|, number of downstream neighbors
```

### Initialization

```
for each reach n:
    corrected[n] = max(baseline[n], 0)
```

### Width Shares (precomputed)

For every bifurcation point (a reach with 2+ downstream successors), compute the width-proportional share for each child:

```
for each bifurcation parent p where out_degree(p) >= 2:
    children = succs(p)
    total_width = sum(width[c] for c in children)

    if total_width > 0:
        for each child c:
            share[p, c] = width[c] / total_width
    else:
        # Fallback: equal split when width data is missing
        for each child c:
            share[p, c] = 1 / |children|
```

### Main Loop

Process every reach in topological order (headwaters first, outlets last):

```
for node in topological_order:
    base = max(baseline[node], 0)
    preds = predecessors(node)

    CASE 1 — Headwater (no predecessors):
        corrected[node] = base
        # Nothing upstream to correct against. Keep the D8 value.

    CASE 2 — Junction (2+ predecessors):
        floor = sum(corrected[p] for p in preds)
        sum_base_up = sum(max(baseline[p], 0) for p in preds)
        lateral = max(base - sum_base_up, 0)
        corrected[node] = floor + lateral

        # floor: the sum of corrected upstream values (from width-split
        # bifurcation children and/or other corrected reaches).
        #
        # lateral: the D8-measured local drainage at this junction — the
        # difference between the junction's own D8 basin area and the sum
        # of its upstream D8 basin areas. This captures real watershed
        # area that drains directly into the junction between it and its
        # predecessors.
        #
        # If the upstream D8 values are clones (sum_base_up > base), then
        # lateral = 0 and the junction gets exactly the corrected upstream
        # sum. This prevents recovering the inflated D8 clone value.
        #
        # If the upstream D8 values are from real tributaries (sum_base_up
        # < base), then lateral > 0 and the junction gets the corrected
        # sum plus the local drainage — preserving real watershed area.

    CASE 3 — Bifurcation child (1 predecessor, parent out_degree >= 2):
        parent = preds[0]
        corrected[node] = corrected[parent] * share[parent, node]
        # Replace the cloned D8 value with the parent's corrected facc
        # multiplied by this child's width share. This is the core fix
        # for the D8 cloning problem.

    CASE 4 — 1:1 link (1 predecessor, parent out_degree == 1):
        parent = preds[0]
        parent_base = max(baseline[parent], 0)

        if parent_base == 0 and corrected[parent] == 0:
            corrected[node] = 0
            # Zero chain: parent has no drainage, neither does this node.

        elif corrected[parent] < parent_base:
            # Parent was LOWERED (by a bifurcation split upstream).
            # Propagate: take the parent's corrected value and add
            # only the local lateral drainage difference.
            lateral = max(base - parent_base, 0)
            corrected[node] = corrected[parent] + lateral

            # The lateral increment (base - parent_base) is the D8-
            # measured local watershed area that drains into this reach
            # between it and its parent. This is real drainage that
            # enters regardless of how the upstream was split, so it
            # is added in full (not scaled by any ratio).
            #
            # The absolute reduction from the bifurcation split stays
            # constant along the chain: if the split removed 400 km^2,
            # every downstream 1:1 link also has ~400 km^2 less than
            # its D8 value.  The junction rule resets corrected values
            # to sum(corrected upstream) + lateral.

        else:
            # Parent was RAISED (by a junction floor) or unchanged.
            # DO NOT propagate the raise. Keep the original D8 value.
            corrected[node] = base

            # This is the key anti-inflation rule. Junction raises
            # represent structural corrections (ensuring downstream
            # facc >= sum upstream), not real increases in drainage
            # area. Propagating them would cause exponential inflation
            # in multi-bifurcation deltas — each nested junction would
            # sum already-raised values, producing 100x-700x inflation.
            #
            # Keeping the D8 value introduces a facc drop at 1:1 links
            # immediately downstream of raised junctions. This is an
            # acceptable tradeoff: the D8 value at these reaches is
            # correct (it reflects the actual upstream drainage area),
            # and the drop is limited to the junction's raise amount.
```

### Correction Types (output tags)

Each modified reach is tagged with a `correction_type`:

| Type | Meaning |
|------|---------|
| `junction_floor` | Junction set to sum(corrected upstream) + lateral drainage |
| `bifurc_share` | Bifurcation child set to width-proportional share of parent |
| `lateral_lower` | 1:1 link lowered by ratio cascade from upstream bifurcation split |
| `cascade_zero` | 1:1 link zeroed because parent chain is zero |

Note: there is no `lateral_raise` type. 1:1 links downstream of raised junctions keep their D8 value (no correction needed).

## Why This Works

### The D8 Cloning Fix

At bifurcations, D8 gives each child the full parent value. We replace this with `parent * (width_child / sum_widths)`. The children now sum to exactly the parent value — conservation holds at the bifurcation point.

### Junction Lateral-Increment Rule

At junctions, we use `corrected = sum(corrected_upstream) + lateral` where `lateral = max(base - sum(baseline_upstream), 0)`.

This generalizes the 1:1 lateral-increment concept to multiple predecessors. The key insight: the D8 base value at a junction includes both **real lateral drainage** and **clone inflation**. By subtracting the sum of upstream D8 baselines, we isolate the real lateral drainage. If the upstream D8 values are clones (their sum exceeds the junction's base), lateral = 0 — the clone inflation is excluded.

**Example — nested bifurcation (clone filtering):**
```
P (base=1000, corrected=1000) → A (base=1000, corrected=600), B (base=1000, corrected=400)
    A → A1 (base=1000, corrected=300), A2 (base=1000, corrected=300)
        J_inner: floor=300+300=600, sum_base_up=1000+1000=2000
                 lateral=max(1000-2000, 0)=0
                 corrected=600  ← CORRECT (not 1000)
    J_inner + B → J_outer: floor=600+400=1000, sum_base_up=1000+1000=2000
                           lateral=0, corrected=1000  ← CORRECT (recovers parent)
```

**Example — real tributary junction (lateral preserved):**
```
River A (base=500), River B (base=300)
Junction (base=850):
    floor=500+300=800, sum_base_up=500+300=800
    lateral=max(850-800, 0)=50
    corrected=800+50=850  ← CORRECT (50 km^2 lateral drainage preserved)
```

### Asymmetric 1:1 Propagation

On 1:1 links, corrections propagate **asymmetrically**:

- **Lowering propagates** (additive lateral): If a parent was lowered by a bifurcation split, the child gets the parent's corrected value plus the local lateral drainage: `corrected = corrected_parent + max(base - base_parent, 0)`. The lateral drainage is real watershed area entering between parent and child — it is added in full, not scaled by the split ratio. The absolute reduction from the split stays constant along the chain.

- **Raising does NOT propagate**: If a parent was raised by a junction floor, the child keeps its original D8 value. This is equivalent to adding the lateral drainage on top of the parent's *baseline* (not corrected) value: `base_parent + (base - base_parent) = base`. The junction raise is a structural correction, not a physical increase in drainage.

This asymmetry is what prevents inflation in complex deltas. The invariant: **corrected values on 1:1 links are <= their D8 baselines** (unless the D8 baseline itself was zero). Junction raises are confined to the junction reach — they don't cascade.

### Why Not Propagate Raises via Additive Lateral?

An earlier version (v1) used `corrected = corrected_parent + max(base - base_parent, 0)` on ALL 1:1 links — both raised and lowered parents. This propagates junction raises downstream: the raised parent's corrected value carries through, and at the next junction it gets summed with other already-raised values, compounding.

Results on the Lena River (73 bifurcations in the delta):
- v1 (propagate all): max facc = **1.65 billion** (674x the real 2.49M km^2 basin area)
- v2 (propagate lowering only): max facc = **7.42 million** (3.0x)

The v1 approach is correct for simple networks with few bifurcations but breaks catastrophically in complex deltas.

### Why Not Multiplicative Ratio Cascade?

We tested ratio-based propagation for lowered parents (`corrected = base * corrected_parent / base_parent`). This scales the lateral drainage by the split ratio, which is physically wrong — if a bifurcation split sends 60% of flow to one channel, the local watershed drainage entering that channel is still 100% of the local drainage, not 60%. Additive lateral preserves the full local drainage increment.

## Properties

### Guaranteed

1. **Junction conservation**: At every junction with 2+ inputs, `corrected[node] >= sum(corrected[predecessors])`. The junction equals the sum plus any real lateral drainage. Verified by lint check F006 = 0.
2. **Bifurcation partitioning**: At every bifurcation, children's facc values sum to exactly the parent's corrected value (proportional to width).
3. **Non-negative**: All corrected values >= 0.
4. **Bounded inflation**: Max facc values are within ~1.5-4.3x of v17b baselines (vs 28-674x under additive propagation).
5. **Single pass**: O(V + E) time, no iteration needed.

### Known Tradeoffs

1. **Junction-raise drops (F010 = ~5,070)**: At 1:1 links immediately downstream of raised junctions, facc drops from the junction's corrected value back to the D8 value. These drops are intentional — they prevent inflation cascading. The drop magnitude equals the junction's raise amount.

2. **1:1 link drops (F011 = ~11,950)**: Includes the junction-raise drops plus D8 artifacts where `baseline[node] < baseline[parent]`. Under additive lateral, if the parent was lowered, the child is also lowered (parent_corrected + lateral < base when parent_corrected < parent_base).

3. **Monotonicity violations (T003 = ~14,000)**: Includes all of the above plus bifurcation children (who are always lower than their parent by design). This is a superset of F010 + F011 + bifurcation children.

4. **Net facc change of -0.8% to -12% per region** (SA is +6.7% due to large tributary junctions): The algorithm removes more facc (via bifurcation splits and lowering cascade) than it adds (via junction floors). This net decrease means corrected facc values are, on average, slightly below D8 values — appropriate since D8 values are inflated by cloning.

5. **F007/F008 residuals (94/152)**: A small number of bifurcations where width data is missing or zero, causing the equal-split fallback to produce children with facc > parent (when parent has additional successors not in the topology).

## Results

### Per-Region Summary

| Region | Reaches | Corrections | Raised | Lowered | Net Change | Max v17c/v17b |
|--------|---------|-------------|--------|---------|------------|---------------|
| NA | 38,696 | 3,302 | 1,732 | 1,570 | -0.83% | 1.92x |
| SA | 42,159 | 3,950 | 2,143 | 1,807 | +6.73% | 4.33x |
| EU | 31,103 | 3,200 | 1,188 | 2,012 | -4.05% | 1.89x |
| AF | 21,441 | 2,242 | 981 | 1,261 | -6.79% | 1.99x |
| AS | 100,185 | 11,301 | 5,427 | 5,874 | +0.35% | 3.16x |
| OC | 15,090 | 1,608 | 688 | 920 | -11.79% | 1.51x |
| **Total** | **248,674** | **25,603** | **12,159** | **13,444** | | |

### River-Level Comparison

| River | Real Basin (km^2) | v17b Max | v17c Max | v17c/v17b | v17c/Real |
|-------|-------------------|----------|----------|-----------|-----------|
| Lena | 2,490,000 | 2,456,228 | 7,415,028 | 3.02x | 2.98x |
| Ob | 2,970,000 | 2,514,491 | 6,339,436 | 2.52x | 2.13x |
| Mississippi | 2,980,000 | 2,955,666 | 5,661,534 | 1.92x | 1.90x |
| Amazon | 6,300,000 | 872,327 | 2,179,810 | 2.50x | 0.35x |

### Lint Check Results

| Check | What it tests | Result |
|-------|---------------|--------|
| F006 | Junction conservation (facc >= sum upstream, threshold 1 km^2) | **0 violations** |
| F007 | Bifurcation balance (children sum / parent ratio) | 94 (residual width issues) |
| F008 | Bifurcation child surplus (child > parent) | 152 (residual width issues) |
| F010 | Junction-raise drops (junction raised but downstream keeps D8) | 5,070 (expected, by design) |
| F011 | 1:1 link drops (excluding bifurcation children) | 11,953 (includes lowering cascade + D8 artifacts) |
| T003 | General facc monotonicity | 13,977 (includes bifurc children + all drops) |

## Algorithm Evolution

### v1: Additive Lateral Propagation (rejected)

Junction rule: `max(base, sum_upstream)`. 1:1 rule: `corrected_parent + max(base - base_parent, 0)`.

Propagated both raises and lowers on 1:1 links. Inflation cascaded through nested bifurcation-junction pairs: each junction summed inflated values, each 1:1 link carried the inflation forward, and the next junction compounded it. Produced 88,632 corrections globally with net changes of +88% (OC) to +3,525% (AS). Lena River reached 1.65 billion km^2 (674x real basin area).

### v2: Asymmetric Propagation (current)

Junction rule: `sum_upstream + max(base - sum(base_upstream), 0)`. 1:1 rule: ratio cascade for lowered parents, keep D8 for raised parents.

Only propagates lowering (from bifurcation splits) using additive lateral increment. Junction raises are confined to the junction reach. Produces 25,603 corrections globally with net changes of -12% (OC) to +7% (SA). Lena River at 7.42 million km^2 (3.0x real basin area). F006 = 0 (conservation satisfied).

## DB Application

1. Restore v17b facc as baseline (overwrite any previous corrections)
2. Clear old tags (`facc_quality`, `edit_flag`) from previous passes
3. Write corrected values for all modified reaches
4. Tag with `edit_flag='facc_conservation_single'`, `facc_quality='conservation_single_pass'`

All DB writes use the RTREE-safe pattern (drop spatial indexes, update, recreate).

## Usage

```bash
# Dry run (no DB changes)
python -m src.sword_duckdb.facc_detection.correct_conservation_single_pass \
    --db data/duckdb/sword_v17c.duckdb \
    --v17b data/duckdb/sword_v17b.duckdb \
    --region NA

# Apply to all regions
python -m src.sword_duckdb.facc_detection.correct_conservation_single_pass \
    --db data/duckdb/sword_v17c.duckdb \
    --v17b data/duckdb/sword_v17b.duckdb \
    --all --apply
```

## Validation (Lint Checks)

| Check | What it tests | Expected result |
|-------|---------------|-----------------|
| F006 | Junction conservation (facc >= sum upstream, threshold 1 km^2) | 0 violations |
| F007 | Bifurcation balance (children sum / parent ratio) | < 100 (residual width issues) |
| F008 | Bifurcation child surplus (child > parent) | < 200 (residual width issues) |
| F010 | Junction-raise drops (junction raised but downstream keeps D8) | ~5,000 (by design) |
| F011 | 1:1 link drops (excluding bifurcation children) | ~12,000 (lowering cascade + D8 artifacts) |
| T003 | General facc monotonicity | ~14,000 (includes bifurc children + all drops) |

---

## v3: Topology-Aware Denoising

### Motivation

v2 (single-pass conservation) operates on v17b facc values which are derived from `MAX(node facc)` per reach. This works well for 95% of reaches, but ~11,876 reaches (4.8%) have within-reach node facc variability >1.5x — the MAX grabs stray values from adjacent D8 flow paths. 1,732 reaches have >10x jumps on 1:1 links; another 3,695 have 2-10x jumps.

v2 also produces 9,064 junction raises (some 48,000x) due to upstream sampling mismatches, and leaves 13,977 T003 monotonicity violations.

### v3 Algorithm

v3 replaces v2 with a unified pipeline: **Phase 1** (topology-aware baseline correction), **Phase 2** (monotonicity enforcement), plus diagnostics and validation.

#### Phase 1: Topology-aware baseline correction

##### Phase 1a: Node-based initialization

For each reach, use the downstream-most node's facc (minimum `dist_out`) instead of `MAX(node facc)`:

```
dn_node_facc[reach] = facc of node with min dist_out in reach
```

For 99% of reaches this is identical to v17b. For the ~1% with high within-reach variability, it avoids grabbing stray values from wrong D8 flow paths. Falls back to v17b facc when node data is unavailable.

##### Phase 1b: Topology-aware correction

Same algorithm as v2 (single-pass, topological order) but using `dn_node_facc` as baseline instead of v17b:

```
for node in topological_order:
    base = dn_node_facc[node]
    preds = predecessors(node)

    Headwater:     corrected[node] = base
    Junction:      floor + max(base - sum_base_up, 0)
    Bifurc child:  corrected[parent] * width_share
    1:1 lowered:   corrected[parent] + max(base - base_parent, 0)
    1:1 raised:    base  (no propagation)
```

##### Phase 1c: T003-targeted MERIT UPA re-sampling

For each 1:1 T003 violation (downstream facc < upstream on non-bifurcation edge):

1. **Dual-endpoint UPA walk** (primary): Walk downstream along MERIT's D8 flow direction from two starting points — the downstream reach's upstream endpoint (walk A) and the upstream reach's downstream endpoint (walk B). Up to 150 steps (~13.5km at 90m). Picks the first UPA value >= target, or the max found.
2. **Radial buffer** (fallback): If neither UPA walk fixes the violation, sample UPA in a buffer around the junction point.

This "snaps" to MERIT's actual thalweg, solving structural offset between SWORD junctions and MERIT confluences. Walk A fixes ~770 violations; walk B adds ~3 more fixes and 145 gap reductions.

Selection rule: pick the **minimum UPA >= corrected upstream** (least distortion fix). If none exceeds the target, pick the maximum candidate (reduces the gap).

After resampling, re-runs Phase 1b on all reaches downstream of resampled ones to propagate improved values. Globally, ~1,651 reaches are resampled.

#### Diagnostics: Outlier detection (log-space, Tukey IQR)

After Phase 1, detect remaining outliers using three methods. **These flags are metadata for downstream users. They do not modify facc values.**

1. **Neighborhood log-deviation**: For each reach, compute `|log(corrected) - log(median_neighbors)|`. Flag if above Tukey upper fence (Q3 + 1.5*IQR) or fixed threshold (default: 1.0 ~ 2.7x).
2. **Junction raises >2x**: Junctions where `corrected > base * 2`.
3. **1:1 link drops >2x**: Links where `upstream / downstream > 2`.

Expected: 5K-15K reaches flagged.

#### Phase 2: Monotonicity enforcement

##### Phase 2a: Chain-wise isotonic regression (PAVA) — bidirectional

For remaining T003 violations on 1:1 chains, runs isotonic regression (pool adjacent violators algorithm) in log-space. This finds the closest monotonically non-decreasing sequence to MERIT's actual values, adjusting values both **up and down** — no inflation bias.

1. Extract maximal 1:1 chains (sequences between junctions/bifurcations)
2. For each chain with violations, run PAVA (non-decreasing) in log-space
3. **Anchor policy — bifurcation children only** (relaxed from earlier versions):
   - **Bifurcation children**: Pinned at 1000x PAVA weight. Preserves width-proportional shares from Phase 1b.
   - **Junction feeders**: NOT anchored. This enables bidirectional correction — isotonic can lower upstream values that feed into junctions, not just raise downstream values. Phase 2b re-enforces junction conservation afterward.
4. Back-transform from log-space and apply

This adjusts ~6,800 reaches: ~4,400 raised and ~2,500 lowered (bidirectional).

**Why relax junction-feeder anchors?** The original policy hard-anchored junction feeders to prevent F006 violations. But this blocked bidirectional correction: isotonic could only raise downstream values, never lower upstream ones. Removing junction-feeder anchors lets isotonic freely adjust both directions, with Phase 2b's junction floor guaranteeing F006=0 afterward.

**Why anchor bifurcation children?** Without anchoring, isotonic sees a chain like `[639K, 14.3]` and pools them in log-space to ~3,022. This undoes the width-proportional split from Phase 1b — a Mississippi delta branch that should get ~632K (proportional to its width) instead gets 3K. The anchor prevents this by making the bifurc share a high-confidence fixed point that isotonic must respect.

##### Phase 2b: Junction floor (re-enforce conservation)

After isotonic regression, re-run junction floor in topological order:
```
for each junction v (in_degree >= 2):
    corrected[v] = max(corrected[v], sum(corrected[upstream]))
```

This restores F006 = 0 after isotonic may have shifted chain values.

**Note on D8-clone junctions**: ~226 junctions in NA (8.3%) have two upstream branches with identical facc — D8 clones from MERIT island/braid re-merges. Phase 2b and F006 both use naive `sum(upstream)`, which slightly over-floors these junctions (double-counting the cloned drainage area). This is intentional: clone-aware flooring (`max` instead of `sum`) produces correct junction values but introduces new T003 drops on downstream 1:1 links, triggering the same cascade inflation rejected in the strict compliance investigation. The over-flooring contributes minimally to the +1.2% net inflation and keeps Phase 2b and F006 consistent.

#### T003 Flag Collection (after Phase 2b)

Remaining T003 violations (non-bifurcation edges where downstream facc < upstream) are **not force-corrected**. Instead, they are classified and flagged as metadata:

- `t003_flag` (boolean) — True if this reach has a T003 violation
- `t003_reason` — structural cause:
  - `chain` — fixing would cascade to downstream reaches
  - `junction_adjacent` — immediately feeds a junction (clamping could break F006)
  - `non_isolated` — other structural disagreement between MERIT and SWORD topology

These flags are included in the output CSV and written to the database alongside corrected facc values.

#### Why not strict compliance? (tested and rejected)

We tested iterative forward-max + junction floor to achieve T003=0 (zero remaining monotonicity violations). The algorithm repeats until convergence: (1) propagate `max(parent, child)` downstream on 1:1 links, (2) enforce junction floor `corrected >= sum(upstream)`.

**Results on NA**: T003=0 achieved, but **+114% inflation** (2.6 billion km² added).

**Root causes**:

1. **D8-clone junctions** (226 in NA, 8.3% of all junctions): MERIT's D8 assigns identical facc to both upstream branches at island/braid re-merges. Summing these double-counts the drainage area. The top clone junctions are on Slave River (2.7M km²), Mackenzie (2.7M), Missouri (2.7M), Nelson (2.5M), St. Lawrence (2.1M).

2. **1:1 D8 drops** (thousands): Forward-max raises every downstream value to match its upstream, but these raises cascade through the network. At non-clone junctions, inflated tributary values sum together, compounding the inflation. The cascade propagates through Mississippi (+434M km²), Missouri (+294M), Mackenzie (+250M), St. Lawrence (+163M), Nelson (+120M), Columbia (+107M), Colorado (+105M).

**Clone-aware variant**: Using `max(clone_group)` instead of `sum` at D8-clone junctions reduced inflation to +93%, but the 1:1 cascade remains the dominant driver. 95.7% of inflation traces back to the 226 clone junctions and their downstream cascades.

**Conclusion**: Remaining T003 violations are inherent MERIT D8 noise — not topology errors. Force-correcting them requires overriding thousands of MERIT values, causing unacceptable inflation. The defensible product is v3 (Phase 2a+2b): data-faithful, low inflation (+1.2%), conservation preserved (F006=0), with T003 flags as metadata.

#### Validation

Inline checks (no DB needed):
- F006 (junction conservation) — must be 0
- T003 (monotonicity) — report remaining count (bifurcation-excluded)
- T003 flagged count and breakdown by reason
- Junction raise count
- 1:1 link drop count
- Remaining T003 exported as GeoJSON for visual audit

### Relationship to Yushan's Integrator

Yushan's approach solves for incremental local areas per reach via constrained convex optimization (CVXPY):

```
min ||W(Ax - L)||²  subject to x >= 0
```

Where A = upstream connectivity matrix, L = adjusted target, W = outlier weights, x = incremental (local) area per reach.

| Aspect | Yushan (Integrator) | v3 (Ours) |
|--------|-------------------|-----------|
| Formulation | Quadratic optimization (CVXPY) | Topological-order pass |
| Variables | Incremental (local) area | Total facc |
| Conservation | Matrix structure (implicit) | Junction floor rule (explicit) |
| Bifurcation | Solver distributes implicitly | Width-share split (explicit) |
| Outlier handling | Tukey IQR + re-solve | Tukey IQR + re-sample from UPA |
| Complexity | O(n³) per basin | O(n) global |
| Scaling | Basin-by-basin (matrix size) | Full region in one pass |
| Dependencies | CVXPY, SCS/ECOS solver | NetworkX, GDAL |

**Equivalence**: Both enforce conservation (sum of upstream ≤ downstream) and non-negativity (no negative drainage). Our junction rule `corrected = sum(corrected_upstream) + max(base - sum(base_upstream), 0)` is equivalent to enforcing `local_area >= 0` at each junction — the `max(..., 0)` clamps the lateral term to non-negative, same as Yushan's `x >= 0` constraint on incremental areas.

### v3 Usage

```bash
# Dry run on NA (no DB changes, no UPA re-sampling)
python -m src.sword_duckdb.facc_detection.correct_facc_denoise \
    --db data/duckdb/sword_v17c.duckdb --v17b data/duckdb/sword_v17b.duckdb --region NA

# Full run with UPA re-sampling
python -m src.sword_duckdb.facc_detection.correct_facc_denoise \
    --db data/duckdb/sword_v17c.duckdb --v17b data/duckdb/sword_v17b.duckdb \
    --region NA --merit /Volumes/SWORD_DATA/data/MERIT_Hydro

# Apply to all regions
python -m src.sword_duckdb.facc_detection.correct_facc_denoise \
    --db data/duckdb/sword_v17c.duckdb --v17b data/duckdb/sword_v17b.duckdb \
    --all --apply --merit /Volumes/SWORD_DATA/data/MERIT_Hydro
```

### v3 Outputs

- `facc_denoise_v3_{REGION}.csv` — per-reach corrections with columns: reach_id, region, original_facc, dn_node_facc, corrected_facc, delta, delta_pct, correction_type, was_resampled, resample_method, resample_d8_steps
- `facc_denoise_v3_summary_{REGION}.json` — summary stats including validation results, resample counts, correction type breakdown, imputed/floored counts
- `remaining_t003_{REGION}.geojson` — spatial layer of remaining T003 violations for visual audit. Properties: reach_id, upstream_id, downstream_id, facc_base, facc_corrected, facc_upstream, delta, ratio_up, ratio_dn, reason (chain/junction_adjacent/non_isolated), is_junction_child, is_junction_parent

### v3 Correction Types

| Type | Phase | Meaning |
|------|-------|---------|
| `junction_floor` | 1b | Junction set to sum(corrected upstream) + lateral |
| `bifurc_share` | 1b | Bifurcation child set to width-proportional share of parent |
| `lateral_lower` | 1b | 1:1 link lowered by cascade from upstream bifurcation split |
| `upa_resample` | 1c | Re-sampled from MERIT UPA via D8 walk or radial buffer |
| `isotonic_regression` | 2a | Adjusted by chain-wise PAVA to enforce monotonicity |
| `junction_floor_post` | 2b | Junction re-floored after isotonic regression |
| `node_denoise` | 1a | Switched from MAX(node facc) to downstream-node facc |
| `t003_flagged_only` | — | No facc change — reach included solely because it has a T003 flag |

### v3 Results (NA, bidirectional isotonic)

Results with relaxed anchors (bifurc-children only) and F012 bifurc-child exclusion:

| Metric | v3 (original anchors) | v3 (bidirectional) |
|--------|----------------------|-------------------|
| F006 (junction conservation) | 0 | **0** |
| T003 (non-bifurc monotonicity) | 885 | **1,044** (flagged, not force-corrected) |
| Net facc change | -1.2% | **+1.249%** |
| Corrections | 8,803 | **9,235** |
| Raised / Lowered | 6,064 / 2,729 | **6,129 / 3,013** |
| Isotonic adjustments | ~5,300 | **5,927** |
| Junction re-floors | — | **1,280** |

T003 flag breakdown (NA):
| Reason | Count |
|--------|-------|
| chain | 824 |
| junction_adjacent | 132 |
| non_isolated | 88 |

### v3 NA Detail (vs v2)

| Metric | v2 (single-pass) | v3 (bidirectional) |
|--------|-------------------|--------------------|
| F006 (junction conservation) | 0 | **0** |
| T003 (non-bifurc monotonicity) | ~9,700 | **1,044** (flagged, not force-corrected) |
| Net facc change | -0.83% | **+1.249%** |
| Corrections | 3,302 | **9,235** |
| Raised / Lowered | 1,732 / 1,570 | **6,129 / 3,013** |
| D8 walk fixes | — | 862 (with MERIT) |
| Isotonic adjustments | — | ~5,927 |

### Algorithm Evolution Summary

| Version | Approach | F006 | T003 (non-bifurc) | Net change | Key issue |
|---------|----------|------|-------------------|------------|-----------|
| v1 | Additive propagation (all) | 0 | Unbounded | +88% to +3525% | 674x inflation in deltas |
| v2 | Asymmetric (lower only) | 0 | ~9,700 | -0.8% to +6.7% | D8 sampling noise, no monotonicity fix |
| v3 | Node init + D8 walk + bidirectional isotonic | 0 | **~1,044** (NA, flagged) | +1.2% (NA) | Remaining are MERIT D8 noise (not force-correctable without +90-114% inflation) |

### Key commits

| Commit | Description |
|--------|-------------|
| `67d8c02` | Replace monotonic clamp with t003 flags for residual violations |
| `2de8ed7` | Anchor bifurcation children in isotonic regression |

### DB Application

Applied to both:
- **DuckDB** (`sword_v17c.duckdb`) — all 6 regions via `--apply` flag
- **PostgreSQL** (`sword_v17c`) — 248,674 rows loaded via `load_from_duckdb.py`
