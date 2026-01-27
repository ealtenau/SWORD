#!/bin/bash
set -e  # stop if any command fails

# Script to run the pipeline for all continents
# This loops through each continent and runs run_pipeline.sh with the appropriate continent code

########## USER CONSTANTS (edit these) ##########
# WORKDIR: base directory for processing (defaults to current directory)
WORKDIR="${WORKDIR:-$(pwd)}"

# List of all continents to process
CONTINENTS=("af" "as" "eu" "na" "oc" "sa")

########## FOLDER PATHS ##########
OUTDIR="$WORKDIR/output"

########## LOG FILES ##########
# Create log directory
LOGDIR="$OUTDIR/logs"
mkdir -p "$LOGDIR"

# Master logfile path
MASTER_LOGFILE="$LOGDIR/all_continents_$(date +'%Y-%m-%d_%H-%M-%S').log"
exec &> >(tee -a "$MASTER_LOGFILE")

echo "=========================================="
echo "Running pipeline for all continents"
echo "Logging to: $MASTER_LOGFILE"
echo "=========================================="
echo ""

# Track success/failure for each continent
declare -a SUCCESSFUL_CONTINENTS=()
declare -a FAILED_CONTINENTS=()

# Loop through each continent
for CONT in "${CONTINENTS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing continent: $CONT"
    echo "=========================================="
    echo ""
    
    # Set the CONT environment variable and run the pipeline
    # Export CONT so run_pipeline.sh can use it
    export CONT="$CONT"
    if bash "$WORKDIR/run_pipeline.sh"; then
        echo ""
        echo "✅ Successfully completed pipeline for $CONT"
        SUCCESSFUL_CONTINENTS+=("$CONT")
    else
        echo ""
        echo "❌ Pipeline failed for $CONT"
        FAILED_CONTINENTS+=("$CONT")
        # Continue with next continent instead of stopping
        echo "Continuing with next continent..."
    fi
    
    echo ""
done

# Print summary
echo ""
echo "=========================================="
echo "PIPELINE SUMMARY"
echo "=========================================="
echo ""
echo "Successfully processed continents:"
if [ ${#SUCCESSFUL_CONTINENTS[@]} -eq 0 ]; then
    echo "  (none)"
else
    for cont in "${SUCCESSFUL_CONTINENTS[@]}"; do
        echo "  ✅ $cont"
    done
fi

echo ""
echo "Failed continents:"
if [ ${#FAILED_CONTINENTS[@]} -eq 0 ]; then
    echo "  (none) - All continents processed successfully!"
else
    for cont in "${FAILED_CONTINENTS[@]}"; do
        echo "  ❌ $cont"
    done
fi

echo ""
if [ ${#FAILED_CONTINENTS[@]} -eq 0 ]; then
    echo "🎉 ALL CONTINENTS PROCESSED SUCCESSFULLY!"
    exit 0
else
    echo "⚠️  Some continents failed. Check logs for details."
    exit 1
fi

