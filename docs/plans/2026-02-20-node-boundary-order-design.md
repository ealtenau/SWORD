# Design: Add node boundary IDs and node_order (Issue #149)

## Problem

POM needs node ordering when flow direction changes have reordered node IDs.
Two new capabilities requested:

1. **Reach boundary nodes** — which node_id is the downstream/upstream end of each reach
2. **Node position** — integer 1..n giving each node's position within its reach

## Schema Changes

### `reaches` table — 2 new columns

| Column | Type | Description |
|--------|------|-------------|
| `dn_node_id` | BIGINT | Downstream boundary node (MIN node_id for reach) |
| `up_node_id` | BIGINT | Upstream boundary node (MAX node_id for reach) |

### `nodes` table — 1 new column

| Column | Type | Description |
|--------|------|-------------|
| `node_order` | INTEGER | 1-based position within reach (1=downstream, n=upstream) |

### No centerline changes

Centerlines already link to nodes via `node_id` and to reaches via `reach_id`.

## Computation

Uses **dist_out ordering**, not node_id. Flow direction changes in v17c can
reorder node IDs, but dist_out is always semantically correct.

Verified on v17c production: 222,944 reaches have MIN(node_id)=downstream,
0 have MIN(node_id)=upstream, 25,732 are single-node (boundary=same node).
Using dist_out is both correct today and future-proof.

```sql
-- Reach boundary nodes (from dist_out extremes)
UPDATE reaches
SET dn_node_id = boundary.dn_nid,
    up_node_id = boundary.up_nid
FROM (
    SELECT reach_id,
        FIRST(node_id ORDER BY dist_out ASC) AS dn_nid,
        FIRST(node_id ORDER BY dist_out DESC) AS up_nid
    FROM nodes GROUP BY reach_id
) boundary
WHERE reaches.reach_id = boundary.reach_id;

-- Node order (1=downstream, n=upstream, by dist_out)
UPDATE nodes SET node_order = ordered.rn
FROM (
    SELECT node_id, region,
        ROW_NUMBER() OVER (PARTITION BY reach_id ORDER BY dist_out ASC) AS rn
    FROM nodes
) ordered
WHERE nodes.node_id = ordered.node_id AND nodes.region = ordered.region;
```

## Implementation

1. Script: `scripts/maintenance/add_node_columns.py`
   - ALTER TABLE to add columns
   - RTREE drop/recreate for reaches UPDATE
   - Run UPDATE queries
   - Validate: no NULLs, node_order range matches n_nodes, boundary nodes exist in nodes table
2. Schema: update `schema.py` with new columns
3. Tests: add test for new columns using test fixture DB

## Not in scope

- v17c pipeline changes
- Reactive system changes
- Export format changes
- NetCDF import
