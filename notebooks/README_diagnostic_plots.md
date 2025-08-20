# Diagnostic Plotting for WSE Drop Analysis

This document explains how to use the diagnostic visualization features added to `master_wse_analysis_crossreach_diag.py`.

## Overview

The script now includes comprehensive diagnostic plotting capabilities to visualize the spatial distribution of data swarms around obstructions. This helps verify that the swarm logic is working correctly before running the full analysis.

## Features

### 1. Individual Swarm Visualization
- **Function**: `plot_swarm_diagnostic()`
- **Purpose**: Creates a single plot showing one obstruction and its associated swarm nodes
- **Shows**:
  - Obstruction point (black star)
  - Upstream swarm nodes (blue dots)
  - Downstream swarm nodes (red dots)
  - Distance circles (2km upstream, 1km downstream)
  - Reach boundaries (if available)

### 2. Multi-Sample Diagnostic Plots
- **Function**: `create_swarm_diagnostic_plots()`
- **Purpose**: Creates a grid of plots showing multiple obstructions simultaneously
- **Default**: Shows 10 random obstructions in a 3x4 grid
- **Configurable**: Can specify any number of samples

### 3. Quick Test Function
- **Function**: `quick_test_100_samples()`
- **Purpose**: Runs a complete test with 100 obstructions to verify swarm logic
- **Includes**: Swarm definition, visualization, and statistics
- **Output**: Diagnostic plots + detailed logging

## Usage

### Option 1: Run Full Analysis with Diagnostic Plots
```bash
python master_wse_analysis_crossreach_diag.py
```
This will:
1. Run the full WSE drop analysis
2. Show diagnostic plots for the first 10 obstructions
3. Run a quick test with 100 samples at the end

### Option 2: Run Only the Quick Test
```bash
python master_wse_analysis_crossreach_diag.py --quick-test
```
This will:
1. Skip the full analysis
2. Run only the 100-sample diagnostic test
3. Show diagnostic plots and statistics

### Option 3: Test Diagnostic Functions Directly
```bash
python test_diagnostic_plots.py
```
This will:
1. Check and verify all file paths
2. Import and test the diagnostic functions
3. Show any errors or issues
4. Verify the plotting functionality works

### Option 4: Check File Paths Only
```bash
python master_wse_analysis_crossreach_diag.py --check-paths
```
This will:
1. Display all important file paths
2. Verify database and output directories exist
3. Help debug path-related issues

## What the Plots Show

### Color Coding
- **Black Star (*)**: The obstruction point
- **Blue Dots**: Upstream swarm nodes (within 2km + parent reaches)
- **Red Dots**: Downstream swarm nodes (within 1km + child reaches)
- **Gray Rectangle**: Approximate reach boundary (if available)
- **Dashed Circles**: Distance thresholds (1.5km upstream, 1.5km downstream)

### Information Displayed
- Obstruction ID and reach ID
- Number of upstream and downstream nodes
- Spatial distribution of swarm nodes
- Distance relationships between nodes

## Interpreting Results

### Good Swarm Distribution
- Upstream nodes should be mostly upstream of the obstruction
- Downstream nodes should be mostly downstream of the obstruction
- Nodes should be within reasonable distance thresholds
- Cross-reach connections should show logical patterns

### Potential Issues to Watch For
- Empty swarms (no nodes found)
- Swarms with very few nodes
- Nodes appearing on the wrong side of the obstruction
- Unrealistic spatial distributions

## Configuration

### Swarm Parameters
- **Upstream distance**: 1.5km from obstruction
- **Downstream distance**: 1.5km from obstruction
- **Cross-reach levels**: Up to 4 parent/child reaches

### Quality Control Features
- **Filtering**: Removes obstructions with 0 nodes in either upstream or downstream swarm
- **Balancing**: Clips larger swarms to match smaller swarm size for statistical balance
- **Node capping**: Limits individual swarms to maximum 10 nodes for manageable analysis
- **Balancing flags**: Indicates when swarms were artificially balanced
- **Data integrity**: Ensures only high-quality, balanced data is used for WSE drop calculations

### Data Source Configuration
- **SWOT collection**: SWOT_L2_HR_RiverSP_D (Version D)
- **API endpoint**: Hydrocron v1 timeseries
- **Feature type**: Node (individual measurement points)
- **Time range**: Version D start (2025-05-05) to present

### Plotting Parameters
- **Figure size**: 5x5 inches per subplot
- **Grid layout**: Maximum 3 columns
- **Random seed**: 42 (for reproducible sampling)

## Troubleshooting

### Common Issues
1. **Database not found**: Use `--check-paths` to verify file locations
2. **No plots displayed**: Check if matplotlib backend is working
3. **Empty swarms**: Verify database connectivity and data availability
4. **Coordinate errors**: Check if x,y coordinates are in the expected format
5. **Memory issues**: Reduce the number of samples or nodes per swarm

### Debug Information
The script provides extensive logging:
- Swarm definition progress
- Node counts and statistics
- Spatial distribution summaries
- Error details for failed operations

## Performance Notes

- **Diagnostic plots**: Fast, only queries metadata
- **Full analysis**: Includes heavy API calls to Hydrocron
- **Caching**: Uses pickle files to avoid repeated downloads
- **Parallel processing**: Uses Dask for concurrent API requests

## Future Enhancements

Potential improvements to consider:
- Interactive plots with hover information
- Export plots to high-resolution formats
- Additional statistical visualizations
- Reach network graph overlays
- Time-series data visualization
