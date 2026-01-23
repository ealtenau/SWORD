#!/bin/bash
set -e  # stop if any command fails

# Required foldering structure:
# WORKDIR 



########## USER CONSTANTS (edit these) ##########
# WORKDIR: base directory for processing (defaults to current directory)
WORKDIR="${WORKDIR:-$(pwd)}"
# CONT can be set via environment variable or defaults to "as"
CONT="${CONT:-as}"

########## FOLDER PATHS ##########
OUTDIR="$WORKDIR/output"

########## LOG FILES ##########
# Create a timestamped log file
# Create log directory
LOGDIR="$OUTDIR/logs"
mkdir -p "$LOGDIR"

# Logfile path
LOGFILE="$LOGDIR/pipeline_$(date +'%Y-%m-%d_%H-%M-%S').log"
exec &> >(tee -a "$LOGFILE")

echo "Logging to: $LOGFILE"


########## CONSTANTS ##########
FRACTION_LOW=-0.8
FRACTION_HIGH=0.8

# phi_refined weights
WRCONSTANT=1000
WUCONSTANT=1
WUPCONSTANT=0.001

#main head/outlet weights
WIDTHWEIGHT=0.6
FREQHWWEIGHT=0.2
FREQOUTWEIGHT=0.4
DISTHWWEIGHT=0.2
DISTOUTWEIGHT=0.0



#################################################

echo "=== Using poetry environment ==="

######################################
# 1. SWORD_graph
######################################
echo "=== Running SWORD_graph ==="
poetry run python SWORD_graph.py \
  --directory "$WORKDIR" \
  --continent "$CONT"


######################################
# 2. SWOT_slopes
######################################
echo "=== Running SWOT_slopes ==="
poetry run python SWOT_slopes.py \
  --dir "$WORKDIR" \
  --continent "$CONT" \
  --fraction_low $FRACTION_LOW \
  --fraction_high $FRACTION_HIGH

######################################
# 3. phi_only_global
######################################
echo "=== Running phi_only_global ==="
poetry run python phi_only_global.py \
  --input "$OUTDIR/${CONT}_slope_single.pkl" \
  --outdir "$OUTDIR/${CONT}" \
  --continent "$CONT" \
  --workdir "$WORKDIR"

######################################
# 4. phi_r_global_refine
######################################
echo "=== Running phi_r_global_refine ==="
poetry run python phi_r_global_refine.py \
  --input_pkl "$OUTDIR/${CONT}/river_directed.pkl" \
  --outdir    "$OUTDIR/${CONT}" \
  --prefer_highs \
  --wR $WRCONSTANT \
  --wU $WUCONSTANT \
  --wUp $WUPCONSTANT

######################################
# 5. assign_attribute
######################################
echo "=== Running assign_attribute ==="
poetry run python assign_attribute.py \
  --graph ${CONT}_MultiDirected_refined.pkl \
  --continent "$CONT" \
  --inputdir "$WORKDIR" \
  --outdir "$WORKDIR" \
  --width-weight $WIDTHWEIGHT \
  --freq-hw-weight $FREQHWWEIGHT \
  --dist-hw-weight $DISTHWWEIGHT \
  --freq-out-weight $FREQOUTWEIGHT \
  --dist-out-weight $DISTOUTWEIGHT

echo "=== ALL SCRIPTS COMPLETED SUCCESSFULLY ==="
