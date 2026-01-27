#!/bin/bash
set -e  # stop if any command fails

# ==============================================================================
# Pipeline WITHOUT phi optimization
# ==============================================================================
# This script uses the ORIGINAL v17b topology (from reach_topology table)
# instead of running the phi algorithm to determine flow directions.
#
# Use this to:
# 1. Validate original v17b topology
# 2. Compare phi-optimized vs original topology
# 3. Generate v17c attributes using original flow directions
#
# Usage:
#   CONT=na WORKDIR=/path ./run_pipeline_nophi.sh
# ==============================================================================

########## USER CONSTANTS (edit these) ##########
WORKDIR="${WORKDIR:-$(pwd)}"
CONT="${CONT:-as}"
DB_PATH="${DB_PATH:-/Users/jakegearon/projects/SWORD/data/duckdb/sword_v17b.duckdb}"

########## FOLDER PATHS ##########
OUTDIR="$WORKDIR/output"
LOGDIR="$OUTDIR/logs"
mkdir -p "$LOGDIR"
mkdir -p "$OUTDIR/${CONT}"

# Logfile path
LOGFILE="$LOGDIR/pipeline_nophi_$(date +'%Y-%m-%d_%H-%M-%S').log"
exec &> >(tee -a "$LOGFILE")
echo "Logging to: $LOGFILE"

########## CONSTANTS ##########
# These are for SWOT slopes (still needed for assign_attribute)
FRACTION_LOW=-0.8
FRACTION_HIGH=0.8

# Main head/outlet weights (used by assign_attribute)
WIDTHWEIGHT=0.6
FREQHWWEIGHT=0.2
FREQOUTWEIGHT=0.4
DISTHWWEIGHT=0.2
DISTOUTWEIGHT=0.0

#################################################
echo "=== Pipeline WITHOUT phi optimization ==="
echo "=== Using original v17b topology ==="
echo "=== Continent: $CONT ==="
echo "=== Database: $DB_PATH ==="

######################################
# 1. Create directed graph from v17b topology
######################################
echo "=== Step 1: Creating directed graph from v17b topology ==="
poetry run python create_v17b_graph.py \
  --continent "$CONT" \
  --workdir "$WORKDIR" \
  --db-path "$DB_PATH" \
  --output "$OUTDIR/${CONT}/${CONT}_v17b_directed.pkl"

######################################
# 2. SWOT_slopes (still needed for width data in assign_attribute)
######################################
echo "=== Step 2: Running SWOT_slopes ==="
poetry run python SWOT_slopes.py \
  --dir "$WORKDIR" \
  --continent "$CONT" \
  --fraction_low $FRACTION_LOW \
  --fraction_high $FRACTION_HIGH

######################################
# 3. SKIP phi_only_global
######################################
echo "=== Step 3: SKIPPING phi_only_global (using original v17b topology) ==="

######################################
# 4. SKIP phi_r_global_refine
######################################
echo "=== Step 4: SKIPPING phi_r_global_refine (using original v17b topology) ==="

######################################
# 5. assign_attribute (using v17b directed graph)
######################################
echo "=== Step 5: Running assign_attribute on v17b topology ==="
# Note: We use the v17b directed graph instead of phi refined graph
poetry run python assign_attribute.py \
  --graph "${CONT}_v17b_directed.pkl" \
  --continent "$CONT" \
  --inputdir "$WORKDIR" \
  --outdir "$WORKDIR" \
  --width-weight $WIDTHWEIGHT \
  --freq-hw-weight $FREQHWWEIGHT \
  --dist-hw-weight $DISTHWWEIGHT \
  --freq-out-weight $FREQOUTWEIGHT \
  --dist-out-weight $DISTOUTWEIGHT

echo "=== ALL SCRIPTS COMPLETED SUCCESSFULLY ==="
echo "=== Output uses ORIGINAL v17b topology ==="
echo "=== Compare with phi-optimized output to evaluate phi algorithm ==="
