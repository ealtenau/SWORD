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

### Scale of the Problem

SWORD v17b has **2,910 bifurcation points** across 248,674 reaches globally. The inflation compounds: a river with multiple bifurcation/junction pairs can have facc inflated 2x, 4x, or more by the outlet.

## Algorithm

One pass, headwater to outlet, in **topological order** (every node is processed after all its upstream predecessors).

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
        corrected[node] = max(base, floor)
        # The junction must have at least as much facc as the sum of its
        # corrected upstream inputs. If the D8 value is already higher
        # (due to additional lateral drainage), keep it.

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
        else:
            lateral = max(base - parent_base, 0)
            corrected[node] = corrected[parent] + lateral

        # The lateral increment (base - parent_base) represents the local
        # watershed area that drains into this reach between it and its
        # parent. We add this increment on top of whatever the parent's
        # corrected value is.
        #
        # If base < parent_base (D8 artifact where facc decreases
        # downstream), lateral = 0 and the node gets exactly the
        # parent's corrected value (monotonicity preserved).
```

### Correction Types (output tags)

Each modified reach is tagged with a `correction_type`:

| Type | Meaning |
|------|---------|
| `junction_floor` | Junction raised to sum of corrected upstream |
| `bifurc_share` | Bifurcation child set to width-proportional share of parent |
| `lateral_raise` | 1:1 link raised because parent was raised upstream |
| `lateral_lower` | 1:1 link lowered because parent was lowered (bifurc share < clone) |
| `cascade_zero` | 1:1 link zeroed because parent chain is zero |

## Why This Works

### The D8 Cloning Fix

At bifurcations, D8 gives each child the full parent value. We replace this with `parent * (width_child / sum_widths)`. The children now sum to exactly the parent value — conservation holds.

### Junction Conservation

At junctions, we take `max(original, sum_upstream_corrected)`. Since the corrected upstream values come from width-split bifurcation children (smaller than the original clones), the sum is often smaller than the inflated D8 value. The junction gets the larger of the two, ensuring it never drops below what flows into it.

### Lateral Increment on 1:1 Links

On 1:1 links (one upstream, one downstream), we don't just copy the parent — we add the **local lateral drainage**. This is the difference between the D8 values at this reach and its parent (`base[node] - base[parent]`), representing the local watershed contribution. This preserves the D8 information about how much area drains into each individual reach segment.

If a parent was **raised** (e.g., by a junction floor), the raise propagates downstream through the lateral addition. If a parent was **lowered** (e.g., by bifurcation splitting), the reduction also propagates. The lateral increment is always >= 0 (clamped), so facc never decreases along a 1:1 chain — strict monotonicity.

### Why Not Multiplicative Cascade?

We tested ratio-based propagation (`corrected = base * corrected_parent / base_parent`). This compounds exponentially: a 2x raise at a junction becomes 4x two junctions downstream, 8x three downstream, etc. The lateral-increment approach is additive — raises accumulate linearly, not exponentially. A +1000 km^2 raise at a junction adds +1000 at every downstream 1:1 link, not +1000, +2000, +4000...

## Properties

### Guaranteed

1. **Junction conservation**: At every junction with 2+ inputs, `corrected[node] >= sum(corrected[predecessors])`. Verified by lint check F006 = 0.
2. **Bifurcation partitioning**: At every bifurcation, children's facc values sum to exactly the parent's corrected value (proportional to width).
3. **1:1 monotonicity**: On 1:1 links, `corrected[node] >= corrected[parent]` (lateral increment >= 0).
4. **Non-negative**: All corrected values >= 0.
5. **Single pass**: O(V + E) time, no iteration needed.

### Known Tradeoffs

1. **Absolute inflation**: Net facc increases ~88-3525% per region because junction raises propagate downstream on 1:1 links. This is acceptable for discharge algorithms that use facc for **relative partitioning** (what fraction of flow goes to each child), not absolute values.

2. **720 residual 1:1 drops globally**: These are D8 artifacts where `baseline[node] < baseline[parent]` on a true 1:1 link (not a bifurcation child). The algorithm sets lateral = 0 and copies the parent's value, so the drop is eliminated — but the node's facc no longer matches the D8 grid.

3. **F007/F008 residuals (94/152)**: A small number of bifurcations where width data is missing or zero, causing the equal-split fallback to produce children with facc > parent (when parent has additional successors not in the topology).

## DB Application

1. Restore v17b facc as baseline (overwrite any previous corrections)
2. Clear old tags (`facc_quality`, `edit_flag`) from previous passes
3. Write corrected values for all modified reaches
4. Tag with `edit_flag='facc_conservation_single'`, `facc_quality='conservation_single_pass'`

All DB writes use the RTREE-safe pattern (drop spatial indexes, update, recreate).

## Usage

```bash
# Dry run (no DB changes)
python -m src.updates.sword_duckdb.facc_detection.correct_conservation_single_pass \
    --db data/duckdb/sword_v17c.duckdb \
    --v17b data/duckdb/sword_v17b.duckdb \
    --region NA

# Apply to all regions
python -m src.updates.sword_duckdb.facc_detection.correct_conservation_single_pass \
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
| F010 | Junction-raise drops (junction raised but downstream unchanged) | 0 violations |
| F011 | 1:1 link drops (excluding bifurcation children) | ~720 (D8 artifacts) |
| T003 | General facc monotonicity | ~5,500 (includes bifurc children + D8 artifacts) |
