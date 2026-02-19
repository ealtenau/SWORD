# Validation Spec: Geometry Metadata Variables

## Summary

| Variable | Type | Description | Source |
|----------|------|-------------|--------|
| `x` | DOUBLE | Centroid longitude | Computed from geom |
| `y` | DOUBLE | Centroid latitude | Computed from geom |
| `x_min` | DOUBLE | Bounding box min longitude | Computed from geom |
| `x_max` | DOUBLE | Bounding box max longitude | Computed from geom |
| `y_min` | DOUBLE | Bounding box min latitude | Computed from geom |
| `y_max` | DOUBLE | Bounding box max latitude | Computed from geom |
| `cl_id_min` | BIGINT | First centerline ID in reach | From centerlines table |
| `cl_id_max` | BIGINT | Last centerline ID in reach | From centerlines table |

## Purpose

These variables provide spatial indexing and geometry metadata for reaches. They are derived automatically from the underlying geometry and centerline data.

## Valid Ranges

| Variable | Min | Max | Notes |
|----------|-----|-----|-------|
| x, x_min, x_max | -180 | 180 | Longitude (WGS84) |
| y, y_min, y_max | -90 | 90 | Latitude (WGS84) |
| cl_id_min | 1 | MAX(cl_id) | Must exist in centerlines |
| cl_id_max | cl_id_min | MAX(cl_id) | Must be >= cl_id_min |

## Invariants

1. **Bounding box consistency:**
   - `x_min <= x <= x_max`
   - `y_min <= y <= y_max`

2. **Centerline range consistency:**
   - `cl_id_min <= cl_id_max`
   - All cl_ids in range should belong to this reach

3. **Centroid within bounds:**
   - `x_min <= x <= x_max` (centroid longitude within bbox)
   - `y_min <= y <= y_max` (centroid latitude within bbox)

## Failure Modes

| Failure | Cause | Impact |
|---------|-------|--------|
| x/y outside valid range | Projection error | Spatial queries fail |
| Inverted bbox (min > max) | Geometry corruption | Spatial index broken |
| cl_id range doesn't match centerlines | Data sync issue | Centerline queries wrong |

## Proposed Lint Checks

| ID | Severity | Rule |
|----|----------|------|
| G004 | ERROR | x_min <= x <= x_max |
| G005 | ERROR | y_min <= y <= y_max |
| G006 | ERROR | cl_id_min <= cl_id_max |
| G007 | WARNING | cl_id range matches actual centerlines |

## Notes

These are low-priority metadata columns that are automatically derived. Issues are rare but indicate upstream geometry problems.

---

*Created: 2026-02-02*
