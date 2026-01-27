#!/bin/bash
set -e  # stop if any command fails

# ==============================================================================
# v17c Pipeline - Simplified SWORD v17c attribute computation
# ==============================================================================
# This script runs the v17c pipeline which:
# 1. Uses original v17b topology (no MILP optimization)
# 2. Computes hydrologic attributes (hydro_dist_out, best_headwater, is_mainstem)
# 3. Optionally integrates SWOT-derived slopes
# 4. Writes results directly to sword_v17c.duckdb
#
# Usage:
#   # Process all regions
#   ./run_pipeline.sh
#
#   # Process single region
#   REGION=NA ./run_pipeline.sh
#
#   # Skip SWOT integration (faster)
#   SKIP_SWOT=1 ./run_pipeline.sh
#
#   # Custom database path
#   DB=/path/to/sword_v17c.duckdb ./run_pipeline.sh
# ==============================================================================

########## USER CONSTANTS (edit these or set via environment) ##########
DB_PATH="${DB:-/Users/jakegearon/projects/SWORD/data/duckdb/sword_v17c.duckdb}"
SWOT_PATH="${SWOT_PATH:-/Volumes/SWORD_DATA/data/swot/RiverSP_D_parq/node}"
REGION="${REGION:-}"  # Empty = all regions
SKIP_SWOT="${SKIP_SWOT:-0}"

########## SCRIPT DIRECTORY ##########
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

########## LOGGING ##########
LOGDIR="$PROJECT_ROOT/output/logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/v17c_pipeline_$(date +'%Y-%m-%d_%H-%M-%S').log"
exec &> >(tee -a "$LOGFILE")
echo "Logging to: $LOGFILE"

########## BANNER ##########
echo "=============================================================="
echo "v17c Pipeline"
echo "=============================================================="
echo "Database: $DB_PATH"
echo "SWOT path: $SWOT_PATH"
echo "Region: ${REGION:-ALL}"
echo "Skip SWOT: $SKIP_SWOT"
echo "=============================================================="

########## BUILD COMMAND ##########
CMD="python -m src.updates.sword_v17c_pipeline.v17c_pipeline --db $DB_PATH"

# Add region or --all flag
if [ -n "$REGION" ]; then
    CMD="$CMD --region $REGION"
else
    CMD="$CMD --all"
fi

# Add SWOT options
if [ "$SKIP_SWOT" = "1" ]; then
    CMD="$CMD --skip-swot"
else
    CMD="$CMD --swot-path $SWOT_PATH"
fi

########## RUN ##########
echo "Running: $CMD"
cd "$PROJECT_ROOT"

# Use poetry if available, otherwise direct python
if command -v poetry &> /dev/null; then
    poetry run $CMD
else
    $CMD
fi

echo "=============================================================="
echo "Pipeline completed successfully"
echo "=============================================================="
