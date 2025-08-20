"""
A definitive, unified, end-to-end diagnostic script to calculate the WSE drop 
across all global obstructions, incorporating cross-reach graph traversal.

This script's purpose is to find the ABSOLUTE MAXIMUM number of obstructions
for which any data can be retrieved, establishing a true upper bound.

This script will:
1. Query DuckDB for ALL obstructions.
2. Define swarms by traversing reach boundaries (upstream and downstream).
3. Fetch all available time-series data with NO quality filtering.
4. Calculate WSE drop based on any available data.
5. Save the diagnostic results to new CSV and GeoJSON files.
"""
import hashlib
import logging
import os
import pickle
from datetime import datetime
import time
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import List, Optional

import dask.dataframe as dd
import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.axes import Axes
import pandas as pd
import requests
from dask.delayed import delayed
from dask.diagnostics.progress import ProgressBar
from dask.dataframe.dispatch import make_meta
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
OBSTRUCTION_SAMPLE = int(os.getenv("OBSTRUCTION_SAMPLE", "10000"))
DO_PLOTS = int(os.getenv("DO_PLOTS", "0")) == 1
DASK_NUM_WORKERS = int(os.getenv("DASK_NUM_WORKERS", "64"))
HYDROCRON_POOLSIZE = int(os.getenv("HYDROCRON_POOLSIZE", "256"))
TIME_BIN = os.getenv("TIME_BIN", "1D")  # Bin for temporal dynamics (e.g., '1D', '1H')
MIN_N_UP = int(os.getenv("MIN_N_UP", "3"))
MIN_N_DOWN = int(os.getenv("MIN_N_DOWN", "3"))
EXPORT_TIMESERIES = int(os.getenv("EXPORT_TIMESERIES", "1")) == 1
EXPORT_SEASONALITY = int(os.getenv("EXPORT_SEASONALITY", "1")) == 1

# --- Hydrocron & Data Configuration ---
HYDROCRON_URL = "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries"
# Get project root directory (parent of notebooks directory)
PROJECT_ROOT = Path(__file__).parent.parent
DUCKDB_FILE = PROJECT_ROOT / 'data/duckdb/sword_global.duckdb'
OUTPUT_DIR = PROJECT_ROOT / "data/analysis"
OUTPUT_CSV = OUTPUT_DIR / "master_wse_drop_crossreach_10k.csv"
OUTPUT_GEOJSON = OUTPUT_DIR / "master_wse_drop_crossreach_10k.geojson"
CACHE_DIR = PROJECT_ROOT / "cache_wse_drop_diag" # Use a separate cache for this unfiltered run

# API Parameters for MAXIMAL data retrieval
START_TIME = "2025-05-05T00:00:00Z" # Version D collection start date - no data exists before this
END_TIME = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
FIELDS = ["node_id", "time_str", "wse"]
COLLECTION_NAME = "SWOT_L2_HR_RiverSP_D"  # Version D collection - latest stable release
SENTINEL = -999_999_999_999

# Dask Metadata
META_COLS = {
    "node_id": "object", "time_str": "object", "wse": "float64"
}
META = make_meta(META_COLS)

# Requests session with pooling, retries, and gzip
SESSION = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=frozenset(['GET']))
adapter = HTTPAdapter(pool_connections=HYDROCRON_POOLSIZE, pool_maxsize=HYDROCRON_POOLSIZE, max_retries=retries)
SESSION.mount('https://', adapter)
SESSION.headers.update({'Accept-Encoding': 'gzip', 'Connection': 'keep-alive'})

# --- Timing Utilities ---
TIMINGS: dict[str, float] = {}

@contextmanager
def time_block(name: str):
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        TIMINGS[name] = TIMINGS.get(name, 0.0) + elapsed
        logger.info(f"[timing] {name}: {elapsed:.3f}s")

def log_timings_summary(header: str = "Execution Timings") -> None:
    if not TIMINGS:
        return
    logger.info(f"=== {header} ===")
    for k, v in sorted(TIMINGS.items(), key=lambda kv: kv[1], reverse=True):
        logger.info(f"  {k}: {v:.3f}s")
    logger.info("=====================")

# --- Caching Functions ---
def get_cache_key(node_ids: List[str], start_time: str, end_time: str, fields: List[str]) -> str:
    param_str = f"{sorted(node_ids)}_{start_time}_{end_time}_{sorted(fields)}_{COLLECTION_NAME}"
    return hashlib.md5(param_str.encode()).hexdigest()

def load_from_cache(cache_key: str) -> Optional[pd.DataFrame]:
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"{cache_key}.pkl"
    if not cache_path.exists(): return None
    try:
        with open(cache_path, 'rb') as f: data = pickle.load(f)
        logger.info(f"Loaded {len(data)} records from cache for key {cache_key[:8]}...")
        return data
    except Exception as e:
        logger.warning(f"Could not load from cache: {e}")
        return None

def save_to_cache(cache_key: str, data: pd.DataFrame):
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"{cache_key}.pkl"
    try:
        with open(cache_path, 'wb') as f: pickle.dump(data, f)
        logger.info(f"Saved {len(data)} records to cache with key {cache_key[:8]}...")
    except Exception as e: logger.warning(f"Could not save to cache: {e}")

# Per-node cache to avoid repeated downloads across runs
NODE_CACHE_DIR = CACHE_DIR / 'nodes' / f"{COLLECTION_NAME}_{START_TIME[:10]}"

def _typed_empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=list(META_COLS.keys())).astype(META_COLS)

def load_node_from_cache(node_id: str) -> Optional[pd.DataFrame]:
    NODE_CACHE_DIR.mkdir(exist_ok=True, parents=True)
    p = NODE_CACHE_DIR / f"{node_id}.pkl"
    if not p.exists():
        return None
    try:
        with open(p, 'rb') as f:
            df = pickle.load(f)
        # Keep only expected columns, add any missing, order and type
        df = df[[c for c in df.columns if c in META_COLS]]
        for col in META_COLS.keys():
            if col not in df.columns:
                df[col] = pd.NA
        df = df[list(META_COLS.keys())].astype(META_COLS)
        return df
    except Exception:
        return None

def save_node_to_cache(node_id: str, df: pd.DataFrame) -> None:
    NODE_CACHE_DIR.mkdir(exist_ok=True, parents=True)
    p = NODE_CACHE_DIR / f"{node_id}.pkl"
    try:
        df_to_save = df[[c for c in META_COLS.keys() if c in df.columns]]
        for col in META_COLS.keys():
            if col not in df_to_save.columns:
                df_to_save[col] = pd.NA
        df_to_save = df_to_save[list(META_COLS.keys())].astype(META_COLS)
        with open(p, 'wb') as f:
            pickle.dump(df_to_save, f)
    except Exception:
        pass

# --- API Fetching Functions (Unchanged interface) ---
@delayed
def query_hydrocron_node(node_id: str, start_time: str, end_time: str, fields: List[str]) -> pd.DataFrame:
    # Per-node cache first
    cached = load_node_from_cache(node_id)
    if cached is not None:
        return cached

    params = {"feature": "Node", "feature_id": node_id, "output": "csv",
              "start_time": start_time, "end_time": end_time, "fields": ",".join(fields),
              "collection_name": COLLECTION_NAME}
    try:
        response = SESSION.get(HYDROCRON_URL, params=params, timeout=45)
        if response.status_code == 200:
            response_json = response.json()
            results_csv = response_json.get("results", {}).get("csv", "")
            if results_csv:
                df = pd.read_csv(StringIO(results_csv), dtype={"node_id": str})
                if df.empty:
                    empty_df = _typed_empty_df()
                    save_node_to_cache(node_id, empty_df)
                    return empty_df
                # Keep only expected columns, add missing, order and type
                df = df[[c for c in df.columns if c in META_COLS]]
                for col in META_COLS.keys():
                    if col not in df.columns:
                        df[col] = pd.NA
                # Enforce time filter defensively to avoid pre-collection data
                try:
                    _t = pd.to_datetime(df["time_str"], errors='coerce', utc=True)
                    df = df[_t >= pd.Timestamp(START_TIME)]
                except Exception:
                    pass
                typed = df[list(META_COLS.keys())].astype(META_COLS)
                save_node_to_cache(node_id, typed)
                return typed
        elif response.status_code == 404:
            empty_df = _typed_empty_df()
            save_node_to_cache(node_id, empty_df)
            return empty_df
        else:
            logger.warning(f"API request for node {node_id} failed with status {response.status_code}: {response.text}")
    except (requests.exceptions.RequestException, ValueError) as e:
        logger.error(f"Network or parsing error querying node {node_id}: {e}")
    empty_df = _typed_empty_df()
    save_node_to_cache(node_id, empty_df)
    return empty_df


def fetch_all_nodes_parallel(node_ids: List[str]) -> pd.DataFrame:
    logger.info(f"Creating delayed Dask queries for {len(node_ids)} nodes...")
    with time_block("queue_api_calls"):
        delayed_results = [query_hydrocron_node(nid, START_TIME, END_TIME, FIELDS) for nid in tqdm(node_ids, desc="Queueing API Calls")]
        ddf = dd.from_delayed(delayed_results, meta=META)
    logger.info("Computing Dask DataFrame... This will fetch data in parallel.")
    with time_block("fetch_api_parallel"), ProgressBar():
        df = ddf.compute(scheduler='threads', num_workers=DASK_NUM_WORKERS)
    logger.info(f"Completed fetching. Retrieved {len(df)} total observations.")
    return df

# --- Path Verification Functions ---
def check_paths():
    """Check and display all important paths for debugging."""
    logger.info("=== Path Verification ===")
    logger.info(f"Current working directory: {Path.cwd()}")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Database file: {DUCKDB_FILE}")
    logger.info(f"Database exists: {DUCKDB_FILE.exists()}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Cache directory: {CACHE_DIR}")
    logger.info(f"SWOT collection: {COLLECTION_NAME}")
    logger.info("========================")

# --- Swarm Quality Control Functions ---
def filter_and_balance_swarms(swarm_map_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and balance swarms for quality control.
    
    This function:
    1. Filters out obstructions with 0 nodes in either upstream or downstream swarm
    2. Balances swarms by clipping the larger one to match the smaller one's size
    3. Adds flags to indicate when balancing occurred
    
    Args:
        swarm_map_df: DataFrame with swarm node mappings
        
    Returns:
        Filtered and balanced swarm DataFrame with quality flags
    """
    logger.info("Starting swarm quality control...")
    
    # Get swarm sizes for each obstruction
    swarm_sizes = swarm_map_df.groupby(['obstruction_node_id', 'swarm_type']).size().unstack(fill_value=0)
    
    # Filter out obstructions with 0 nodes in either swarm
    valid_obstructions = swarm_sizes[(swarm_sizes['upstream'] > 0) & (swarm_sizes['downstream'] > 0)].index
    logger.info(f"Filtering: {len(swarm_sizes)} total obstructions -> {len(valid_obstructions)} with both swarms")
    
    # Keep only valid obstructions
    filtered_swarms = swarm_map_df[swarm_map_df['obstruction_node_id'].isin(valid_obstructions)].copy()
    
    # Balance swarms by clipping larger ones to match smaller ones
    balanced_swarms = []
    
    for obs_id in valid_obstructions:
        obs_swarms = filtered_swarms[filtered_swarms['obstruction_node_id'] == obs_id]
        
        upstream_size = len(obs_swarms[obs_swarms['swarm_type'] == 'upstream'])
        downstream_size = len(obs_swarms[obs_swarms['swarm_type'] == 'downstream'])
        
        # Determine target size (minimum of the two)
        target_size = min(upstream_size, downstream_size)
        
        # Balance upstream swarm
        upstream_swarm = obs_swarms[obs_swarms['swarm_type'] == 'upstream']
        if len(upstream_swarm) > target_size:
            upstream_swarm = upstream_swarm.sample(n=target_size, random_state=42).copy()
            upstream_swarm['swarm_balanced'] = True
        else:
            upstream_swarm = upstream_swarm.copy()
            upstream_swarm['swarm_balanced'] = False
            
        # Balance downstream swarm
        downstream_swarm = obs_swarms[obs_swarms['swarm_type'] == 'downstream']
        if len(downstream_swarm) > target_size:
            downstream_swarm = downstream_swarm.sample(n=target_size, random_state=42).copy()
            downstream_swarm['swarm_balanced'] = True
        else:
            downstream_swarm = downstream_swarm.copy()
            downstream_swarm['swarm_balanced'] = False
        
        # Combine balanced swarms
        balanced_swarms.append(upstream_swarm)
        balanced_swarms.append(downstream_swarm)
    
    # Combine all balanced swarms
    result_df = pd.concat(balanced_swarms, ignore_index=True)
    
    # Add summary statistics
    total_balanced = result_df['swarm_balanced'].sum()
    total_swarms = len(result_df)
    logger.info(f"Balancing complete: {total_balanced}/{total_swarms} swarms were balanced")
    
    return result_df

# --- Diagnostic Plotting Functions ---
def plot_swarm_diagnostic(obstruction_id: str, obstruction_data: pd.Series, 
                         swarm_data: pd.DataFrame, reach_data: pd.DataFrame,
                         ax: Optional[Axes] = None, show_reach_boundaries: bool = True):
    """
    Plot diagnostic visualization of a single obstruction's swarm distribution.
    
    Args:
        obstruction_id: ID of the obstruction to visualize
        obstruction_data: Series containing obstruction coordinates and metadata
        swarm_data: DataFrame with swarm node information
        reach_data: DataFrame with reach boundary information
        ax: Matplotlib axis to plot on (creates new one if None)
        show_reach_boundaries: Whether to show reach boundary polygons
    """
    if ax is None:
        _fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Filter swarm data for this obstruction
    obs_swarm = swarm_data[swarm_data['obstruction_node_id'] == obstruction_id].copy()
    
    if obs_swarm.empty:
        ax.text(0.5, 0.5, f"No swarm data for obstruction {obstruction_id}", 
                transform=ax.transAxes, ha='center', va='center')
        return ax
    
    # Get obstruction coordinates
    obs_x, obs_y = obstruction_data['x'], obstruction_data['y']
    
    # Plot reach boundaries if requested
    if show_reach_boundaries:
        reach_id = obstruction_data['reach_id']
        reach_boundary = reach_data[reach_data['reach_id'] == reach_id]
        if not reach_boundary.empty:
            # This is a simplified representation - in practice you'd need actual geometry
            # For now, create a bounding box around the reach
            reach_nodes = reach_data[reach_data['reach_id'] == reach_id]
            if not reach_nodes.empty:
                x_min, x_max = reach_nodes['x'].min(), reach_nodes['x'].max()
                y_min, y_max = reach_nodes['y'].min(), reach_nodes['y'].max()
                rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, 
                                       linewidth=1, edgecolor='gray', facecolor='none', alpha=0.3)
                ax.add_patch(rect)
    
    # Plot upstream swarm nodes
    upstream_nodes = obs_swarm[obs_swarm['swarm_type'] == 'upstream']
    if not upstream_nodes.empty:
        ax.scatter(upstream_nodes['x'], upstream_nodes['y'], 
                  c='blue', s=30, alpha=0.6, label='Upstream Swarm', zorder=3)
    
    # Plot downstream swarm nodes
    downstream_nodes = obs_swarm[obs_swarm['swarm_type'] == 'downstream']
    if not downstream_nodes.empty:
        ax.scatter(downstream_nodes['x'], downstream_nodes['y'], 
                  c='red', s=30, alpha=0.6, label='Downstream Swarm', zorder=3)
    
    # Plot obstruction point
    ax.scatter(obs_x, obs_y, c='black', s=100, marker='*', 
              label='Obstruction', zorder=5, edgecolors='white', linewidth=2)
    
    # Add distance circles for context
    circle_up = patches.Circle((obs_x, obs_y), 1500, fill=False, color='blue', 
                          linestyle='--', alpha=0.5, linewidth=1)
    circle_down = patches.Circle((obs_x, obs_y), 1500, fill=False, color='red', 
                            linestyle='--', alpha=0.5, linewidth=1)
    ax.add_patch(circle_up)
    ax.add_patch(circle_down)
    
    # Customize plot
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Add balancing information to title
    upstream_balanced = upstream_nodes['swarm_balanced'].iloc[0] if not upstream_nodes.empty and 'swarm_balanced' in upstream_nodes.columns else False
    downstream_balanced = downstream_nodes['swarm_balanced'].iloc[0] if not downstream_nodes.empty and 'swarm_balanced' in downstream_nodes.columns else False
    
    balance_info = ""
    if upstream_balanced or downstream_balanced:
        balance_info = "\n⚠️ Swarms balanced"
    
    ax.set_title(f'Swarm Diagnostic: Obstruction {obstruction_id}\nReach {obstruction_data["reach_id"]}\nUp: {len(upstream_nodes)}, Down: {len(downstream_nodes)}{balance_info}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set reasonable axis limits
    all_x = [obs_x] + obs_swarm['x'].tolist()
    all_y = [obs_y] + obs_swarm['y'].tolist()
    if all_x and all_y:
        x_range = max(all_x) - min(all_x)
        y_range = max(all_y) - min(all_y)
        margin = max(x_range, y_range) * 0.1
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    return ax

def create_swarm_diagnostic_plots(swarm_map_df: pd.DataFrame, master_obstructions_df: pd.DataFrame,
                                 reach_data: pd.DataFrame, num_samples: int = 10):
    """
    Create diagnostic plots for random sample of obstructions to visualize swarm distributions.
    
    Args:
        swarm_map_df: DataFrame with swarm node mappings
        master_obstructions_df: DataFrame with obstruction information
        reach_data: DataFrame with reach boundary information
        num_samples: Number of random obstructions to visualize
    """
    logger.info(f"Creating diagnostic plots for {num_samples} random obstructions...")
    
    # Sample random obstructions
    sample_obstructions = master_obstructions_df.sample(n=min(num_samples, len(master_obstructions_df)), 
                                                      random_state=42)
    
    # Create subplot grid
    cols = min(3, num_samples)
    rows = (num_samples + cols - 1) // cols
    _fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    
    if num_samples == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.reshape(rows, cols)
    
    # Plot each sample
    for i, (_, obstruction) in enumerate(sample_obstructions.iterrows()):
        row, col = i // cols, i % cols
        ax = axes[row][col]
        
        plot_swarm_diagnostic(
            obstruction_id=obstruction['obstruction_node_id'],
            obstruction_data=obstruction,
            swarm_data=swarm_map_df,
            reach_data=reach_data,
            ax=ax,
            show_reach_boundaries=True
        )
    
    # Hide unused subplots
    for i in range(num_samples, rows * cols):
        row, col = i // cols, i % cols
        axes[row][col].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    logger.info(f"Created diagnostic plots for {num_samples} obstructions")
    logger.info("Swarm summary statistics:")
    for _, obstruction in sample_obstructions.iterrows():
        obs_id = obstruction['obstruction_node_id']
        obs_swarm = swarm_map_df[swarm_map_df['obstruction_node_id'] == obs_id]
        upstream_count = len(obs_swarm[obs_swarm['swarm_type'] == 'upstream'])
        downstream_count = len(obs_swarm[obs_swarm['swarm_type'] == 'downstream'])
        logger.info(f"  Obstruction {obs_id}: {upstream_count} upstream, {downstream_count} downstream nodes")

def quick_test_100_samples():
    """
    Run a quick test with 100 obstructions to verify the swarm logic and data flow.
    """
    logger.info("--- Starting Quick Test with 100 Samples ---")
    
    # Verify database file exists
    if not DUCKDB_FILE.exists():
        logger.error(f"Database file not found: {DUCKDB_FILE}")
        logger.error(f"Current working directory: {Path.cwd()}")
        logger.error(f"Project root: {PROJECT_ROOT}")
        logger.error(f"Database path: {DUCKDB_FILE.absolute()}")
        raise FileNotFoundError(f"Database file not found: {DUCKDB_FILE}")
    
    logger.info(f"Using database: {DUCKDB_FILE}")
    
    # Temporarily override the sample size
    original_sample = OBSTRUCTION_SAMPLE
    test_sample_size = 100
    
    try:
        with duckdb.connect(database=DUCKDB_FILE, read_only=True) as con:
            # Get 100 obstructions
            logger.info("Step 1: Querying for 100 obstructions...")
            with time_block("qt_query_obstructions"):
                master_obstructions_df = con.execute("""
                SELECT node_id AS obstruction_node_id, reach_id, dist_out, x, y 
                FROM nodes 
                WHERE obstr_type IN (1, 2, 3)
                LIMIT 100
            """).df()
            logger.info(f"Found {len(master_obstructions_df)} obstructions for quick test.")
            
            # Get reach data for visualization
            with time_block("qt_query_reach_metadata"):
                reach_data = con.execute("""
                SELECT reach_id, rch_id_up_1, rch_id_up_2, rch_id_up_3, rch_id_up_4,
                       rch_id_dn_1, rch_id_dn_2, rch_id_dn_3, rch_id_dn_4
                FROM reaches
                WHERE reach_id IN (SELECT DISTINCT reach_id FROM nodes WHERE obstr_type IN (1, 2, 3) LIMIT 100)
            """).df()
            
            # Get node coordinates for reaches
            with time_block("qt_query_reach_nodes"):
                reach_nodes = con.execute("""
                SELECT reach_id, x, y, dist_out
                FROM nodes
                WHERE reach_id IN (SELECT DISTINCT reach_id FROM nodes WHERE obstr_type IN (1, 2, 3) LIMIT 100)
            """).df()
            
            # Define symmetrical swarms: 1.5km upstream and 1.5km downstream
            logger.info("Step 2: Defining cross-reach upstream and downstream swarms...")
            con.register('master_obstructions_df', master_obstructions_df)
            
            swarm_query = """
            WITH obstruction_reaches AS (
                SELECT
                    obs.obstruction_node_id,
                    obs.reach_id,
                    obs.dist_out,
                    r.rch_id_up_1, r.rch_id_up_2, r.rch_id_up_3, r.rch_id_up_4,
                    r.rch_id_dn_1, r.rch_id_dn_2, r.rch_id_dn_3, r.rch_id_dn_4
                FROM master_obstructions_df obs
                JOIN reaches r ON obs.reach_id = r.reach_id
            ),
            upstream_candidates AS (
                SELECT obs.obstruction_node_id, CAST(n.node_id AS VARCHAR) AS swarm_node_id, 'upstream' AS swarm_type,
                       n.x, n.y, n.dist_out, n.reach_id,
                       ABS(n.dist_out - obs.dist_out) AS dist_metric
                FROM master_obstructions_df obs JOIN nodes n ON obs.reach_id = n.reach_id
                WHERE n.dist_out > obs.dist_out AND n.dist_out <= (obs.dist_out + 1500) AND n.lakeflag = 0
                UNION ALL
                SELECT obs_r.obstruction_node_id, CAST(n.node_id AS VARCHAR), 'upstream', n.x, n.y, n.dist_out, n.reach_id,
                       ABS(n.dist_out - obs_r.dist_out) AS dist_metric
                FROM obstruction_reaches obs_r JOIN nodes n 
                    ON n.reach_id IN (obs_r.rch_id_up_1, obs_r.rch_id_up_2, obs_r.rch_id_up_3, obs_r.rch_id_up_4)
                WHERE n.lakeflag = 0
            ),
            upstream_ranked AS (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY obstruction_node_id ORDER BY dist_metric ASC) AS rn
                FROM upstream_candidates
            ),
            top_upstream AS (
                SELECT obstruction_node_id, swarm_node_id, 'upstream' AS swarm_type, x, y, dist_out, reach_id
                FROM upstream_ranked WHERE rn <= 10
            ),
            downstream_candidates AS (
                SELECT obs.obstruction_node_id, CAST(n.node_id AS VARCHAR) AS swarm_node_id, 'downstream' AS swarm_type,
                       n.x, n.y, n.dist_out, n.reach_id,
                       ABS(n.dist_out - obs.dist_out) AS dist_metric
                FROM master_obstructions_df obs JOIN nodes n ON obs.reach_id = n.reach_id
                WHERE n.dist_out < obs.dist_out AND n.dist_out >= (obs.dist_out - 1500) AND n.lakeflag = 0
                UNION ALL
                SELECT obs_r.obstruction_node_id, CAST(n.node_id AS VARCHAR), 'downstream', n.x, n.y, n.dist_out, n.reach_id,
                       ABS(n.dist_out - obs_r.dist_out) AS dist_metric
                FROM obstruction_reaches obs_r JOIN nodes n 
                    ON n.reach_id IN (obs_r.rch_id_dn_1, obs_r.rch_id_dn_2, obs_r.rch_id_dn_3, obs_r.rch_id_dn_4)
                WHERE n.lakeflag = 0
            ),
            downstream_ranked AS (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY obstruction_node_id ORDER BY dist_metric ASC) AS rn
                FROM downstream_candidates
            ),
            top_downstream AS (
                SELECT obstruction_node_id, swarm_node_id, 'downstream' AS swarm_type, x, y, dist_out, reach_id
                FROM downstream_ranked WHERE rn <= 10
            )
            SELECT * FROM top_upstream
            UNION ALL
            SELECT * FROM top_downstream
            """
            with time_block("qt_build_swarms"):
                swarm_map_df = con.execute(swarm_query).df()
            
            # Filter and balance swarms for quality control
            logger.info("Filtering and balancing swarms for quality control...")
            with time_block("qt_filter_balance_swarms"):
                swarm_map_df = filter_and_balance_swarms(swarm_map_df)
            
            # Swarms already limited to at most 10 nodes per side in SQL
            
            logger.info(f"Defined swarms for {swarm_map_df['obstruction_node_id'].nunique()} obstructions.")
            
            # Create diagnostic plots
            logger.info("Step 3: Creating diagnostic visualizations...")
            if DO_PLOTS:
                with time_block("qt_plot_diagnostics"):
                    create_swarm_diagnostic_plots(swarm_map_df, master_obstructions_df, reach_nodes, num_samples=10)
            
            # Show swarm statistics
            logger.info("Step 4: Analyzing swarm statistics...")
            with time_block("qt_swarm_stats"):
                swarm_stats = swarm_map_df.groupby(['obstruction_node_id', 'swarm_type']).size().unstack(fill_value=0)
            logger.info(f"Swarm size statistics:")
            logger.info(f"  Upstream swarms: mean={swarm_stats['upstream'].mean():.1f}, std={swarm_stats['upstream'].std():.1f}")
            logger.info(f"  Downstream swarms: mean={swarm_stats['downstream'].mean():.1f}, std={swarm_stats['downstream'].std():.1f}")
            
            # Show balancing statistics
            if 'swarm_balanced' in swarm_map_df.columns:
                balanced_count = swarm_map_df['swarm_balanced'].sum()
                total_swarms = len(swarm_map_df)
                logger.info(f"Balancing statistics: {balanced_count}/{total_swarms} swarms were balanced")
            
            # Show spatial distribution
            logger.info(f"Total unique nodes in swarms: {swarm_map_df['swarm_node_id'].nunique()}")
            logger.info(f"Total swarm assignments: {len(swarm_map_df)}")
            
    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        raise
    finally:
        # Restore original sample size
        pass  # No need to restore since we didn't modify the global
    
    logger.info("--- Quick Test Complete ---")
    log_timings_summary("Quick Test Timings")

# --- Main Analysis Function ---
def main():
    """Main execution function to perform the WSE drop analysis."""
    logger.info("--- Starting Definitive Cross-Reach WSE Drop Diagnostic ---")
    
    # Verify database file exists
    if not DUCKDB_FILE.exists():
        logger.error(f"Database file not found: {DUCKDB_FILE}")
        logger.error(f"Current working directory: {Path.cwd()}")
        logger.error(f"Project root: {PROJECT_ROOT}")
        logger.error(f"Database path: {DUCKDB_FILE.absolute()}")
        raise FileNotFoundError(f"Database file not found: {DUCKDB_FILE}")
    
    logger.info(f"Using database: {DUCKDB_FILE}")
    logger.info(f"Using SWOT collection: {COLLECTION_NAME}")
    OUTPUT_DIR.mkdir(exist_ok=True)

    with duckdb.connect(database=DUCKDB_FILE, read_only=True) as con:
        # 1. Get All Obstructions and their Coordinates
        logger.info("Step 1: Querying for all obstructions...")
        with time_block("query_all_obstructions"):
            master_obstructions_df = con.execute("""
            SELECT node_id AS obstruction_node_id, reach_id, dist_out, x, y 
            FROM nodes 
            WHERE obstr_type IN (1, 2, 3)
        """).df()
        logger.info(f"Found {len(master_obstructions_df)} total obstructions.")
        if OBSTRUCTION_SAMPLE and len(master_obstructions_df) > OBSTRUCTION_SAMPLE:
            master_obstructions_df = master_obstructions_df.sample(n=OBSTRUCTION_SAMPLE, random_state=42).reset_index(drop=True)
            logger.info(f"Sampled {OBSTRUCTION_SAMPLE} obstructions for a fast test run.")

        # 2. Define Swarms with Cross-Reach Logic
        logger.info("Step 2: Defining cross-reach upstream and downstream swarms...")
        con.register('master_obstructions_df', master_obstructions_df)
        
        # Define symmetrical swarms: 1.5km upstream and 1.5km downstream
        swarm_query = """
        WITH obstruction_reaches AS (
            SELECT
                obs.obstruction_node_id,
                obs.reach_id,
                obs.dist_out,
                r.rch_id_up_1, r.rch_id_up_2, r.rch_id_up_3, r.rch_id_up_4,
                r.rch_id_dn_1, r.rch_id_dn_2, r.rch_id_dn_3, r.rch_id_dn_4
            FROM master_obstructions_df obs
            JOIN reaches r ON obs.reach_id = r.reach_id
        ),
        upstream_candidates AS (
            SELECT obs.obstruction_node_id, CAST(n.node_id AS VARCHAR) AS swarm_node_id, 'upstream' AS swarm_type,
                   n.x, n.y, n.dist_out, n.reach_id,
                   ABS(n.dist_out - obs.dist_out) AS dist_metric
            FROM master_obstructions_df obs JOIN nodes n ON obs.reach_id = n.reach_id
            WHERE n.dist_out > obs.dist_out AND n.dist_out <= (obs.dist_out + 1500) AND n.lakeflag = 0
            UNION ALL
            SELECT obs_r.obstruction_node_id, CAST(n.node_id AS VARCHAR), 'upstream', n.x, n.y, n.dist_out, n.reach_id,
                   ABS(n.dist_out - obs_r.dist_out) AS dist_metric
            FROM obstruction_reaches obs_r JOIN nodes n 
                ON n.reach_id IN (obs_r.rch_id_up_1, obs_r.rch_id_up_2, obs_r.rch_id_up_3, obs_r.rch_id_up_4)
            WHERE n.lakeflag = 0
        ),
        upstream_ranked AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY obstruction_node_id ORDER BY dist_metric ASC) AS rn
            FROM upstream_candidates
        ),
        top_upstream AS (
            SELECT obstruction_node_id, swarm_node_id, 'upstream' AS swarm_type, x, y, dist_out, reach_id
            FROM upstream_ranked WHERE rn <= 10
        ),
        downstream_candidates AS (
            SELECT obs.obstruction_node_id, CAST(n.node_id AS VARCHAR) AS swarm_node_id, 'downstream' AS swarm_type,
                   n.x, n.y, n.dist_out, n.reach_id,
                   ABS(n.dist_out - obs.dist_out) AS dist_metric
            FROM master_obstructions_df obs JOIN nodes n ON obs.reach_id = n.reach_id
            WHERE n.dist_out < obs.dist_out AND n.dist_out >= (obs.dist_out - 1500) AND n.lakeflag = 0
            UNION ALL
            SELECT obs_r.obstruction_node_id, CAST(n.node_id AS VARCHAR), 'downstream', n.x, n.y, n.dist_out, n.reach_id,
                   ABS(n.dist_out - obs_r.dist_out) AS dist_metric
            FROM obstruction_reaches obs_r JOIN nodes n 
                ON n.reach_id IN (obs_r.rch_id_dn_1, obs_r.rch_id_dn_2, obs_r.rch_id_dn_3, obs_r.rch_id_dn_4)
            WHERE n.lakeflag = 0
        ),
        downstream_ranked AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY obstruction_node_id ORDER BY dist_metric ASC) AS rn
            FROM downstream_candidates
        ),
        top_downstream AS (
            SELECT obstruction_node_id, swarm_node_id, 'downstream' AS swarm_type, x, y, dist_out, reach_id
            FROM downstream_ranked WHERE rn <= 10
        )
        SELECT * FROM top_upstream
        UNION ALL
        SELECT * FROM top_downstream
        """
        with time_block("build_swarms"):
            swarm_map_df = con.execute(swarm_query).df()
        
        # Filter and balance swarms for quality control
        logger.info("Filtering and balancing swarms for quality control...")
        swarm_map_df = filter_and_balance_swarms(swarm_map_df)
        
        # Swarms already limited to at most 10 nodes per side in SQL
        
        # Create diagnostic plots for first 10 obstructions
        if DO_PLOTS:
            logger.info("Creating diagnostic plots for first 10 obstructions...")
            sample_obstructions = master_obstructions_df.head(10)
            
            # Get reach data for visualization
            with time_block("query_reach_nodes_for_plots"):
                reach_nodes = con.execute("""
                    SELECT reach_id, x, y, dist_out
                    FROM nodes
                    WHERE reach_id IN (SELECT DISTINCT reach_id FROM master_obstructions_df)
                """).df()
            with time_block("plot_diagnostics"):
                create_swarm_diagnostic_plots(swarm_map_df, sample_obstructions, 
                                            reach_nodes, num_samples=10)
        
        # Show balancing statistics
        if 'swarm_balanced' in swarm_map_df.columns:
            balanced_count = swarm_map_df['swarm_balanced'].sum()
            total_swarms = len(swarm_map_df)
            logger.info(f"Balancing statistics: {balanced_count}/{total_swarms} swarms were balanced")
    
    # 3. Parallel Data Fetching (Unfiltered)
    with time_block("extract_unique_nodes"):
        all_unique_nodes = swarm_map_df['swarm_node_id'].dropna().astype(str).unique().tolist()
    logger.info(f"Step 3: Found {len(all_unique_nodes)} unique swarm nodes to query from Hydrocron.")
    
    cache_key = get_cache_key(all_unique_nodes, START_TIME, END_TIME, FIELDS)
    with time_block("load_cache"):
        time_series_df = load_from_cache(cache_key)
    if time_series_df is None:
        time_series_df = fetch_all_nodes_parallel(all_unique_nodes)
        if not time_series_df.empty:
            with time_block("save_cache"):
                save_to_cache(cache_key, time_series_df)
    
    if time_series_df.empty:
        logger.error("No time-series data could be fetched. Exiting.")
        return

    # 4. Analyze and Calculate WSE Drop (Unfiltered)
    logger.info("Step 4: Analyzing data and calculating WSE drop (NO quality filter)...")
    # Enforce collection start date globally (defensive in case of stale cache)
    with time_block("filter_start_date"):
        try:
            _ts = pd.to_datetime(time_series_df['time_str'], errors='coerce', utc=True)
            mask = _ts >= pd.Timestamp(START_TIME)
            time_series_df = time_series_df[mask].copy()
            if mask.any():
                logger.info(f"Earliest kept timestamp: {_ts[mask].min()} (START_TIME={START_TIME})")
        except Exception:
            pass
    # ONLY filter for the sentinel value, keeping all other data regardless of quality flags.
    with time_block("filter_sentinel"):
        filtered_df = time_series_df[time_series_df['wse'] != SENTINEL].copy()
    logger.info(f"Filtered for sentinel values. Kept {len(filtered_df)} total observations.")

    with time_block("normalize_id_types"):
        filtered_df['node_id'] = filtered_df['node_id'].astype(str)
        swarm_map_df['swarm_node_id'] = swarm_map_df['swarm_node_id'].astype(str)
    
    with time_block("merge_swarm_info"):
        wse_with_swarm_info = pd.merge(filtered_df, swarm_map_df, left_on='node_id', right_on='swarm_node_id', how='inner')
    
    # --- Targeted temporal dynamics export (daily-binned time series) ---
    if EXPORT_TIMESERIES:
        try:
            with time_block("build_time_series"):
                wsi = wse_with_swarm_info.copy()
                wsi['time'] = pd.to_datetime(wsi['time_str'], errors='coerce', utc=True)
                wsi['time_bin'] = wsi['time'].dt.floor(TIME_BIN)

                agg = (wsi.groupby(['obstruction_node_id', 'swarm_type', 'time_bin'])
                       .agg(median_wse=('wse', 'median'), n_obs=('wse', 'size'))
                       .reset_index())

                ts = agg.pivot(index=['obstruction_node_id', 'time_bin'],
                               columns='swarm_type',
                               values=['median_wse', 'n_obs']).reset_index()

                # Flatten columns
                flat_cols = ['obstruction_node_id', 'time_bin']
                for top, side in ts.columns[2:]:
                    flat_cols.append(f"{top}_{side}")
                ts.columns = flat_cols

                # Normalize expected names and filter by counts
                # Expected sides: 'upstream', 'downstream'
                if 'n_obs_upstream' in ts.columns and 'n_obs_downstream' in ts.columns:
                    ts = ts[(ts['n_obs_upstream'] >= MIN_N_UP) & (ts['n_obs_downstream'] >= MIN_N_DOWN)]
                # Compute drop if medians exist
                if 'median_wse_upstream' in ts.columns and 'median_wse_downstream' in ts.columns:
                    ts['wse_drop_m_time'] = ts['median_wse_upstream'] - ts['median_wse_downstream']

            # Save time series
            sample_count = len(master_obstructions_df)
            timeseries_path = OUTPUT_DIR / f"master_wse_drop_timeseries_{sample_count}.parquet"
            with time_block("save_timeseries"):
                ts.to_parquet(timeseries_path, index=False)
            logger.info(f"Saved time-series drops to {timeseries_path}")

            # Build monthly seasonality summary (long format: one row per obstruction-month)
            if EXPORT_SEASONALITY and 'wse_drop_m_time' in ts.columns:
                with time_block("build_seasonality"):
                    ts_seas = ts.dropna(subset=['wse_drop_m_time']).copy()
                    ts_seas['month'] = ts_seas['time_bin'].dt.month
                    overall = ts_seas.groupby('obstruction_node_id')['wse_drop_m_time'].median().rename('overall_median')
                    monthly = (ts_seas.groupby(['obstruction_node_id', 'month'])['wse_drop_m_time']
                               .median().rename('median_by_month').reset_index())
                    coverage = (ts_seas.groupby(['obstruction_node_id', 'month'])
                                .size().rename('n_days').reset_index())
                    monthly = monthly.merge(coverage, on=['obstruction_node_id', 'month'], how='left')
                    monthly = monthly.merge(overall, on='obstruction_node_id', how='left')
                    monthly['seasonal_index'] = monthly['median_by_month'] - monthly['overall_median']
                    # Amplitude per obstruction
                    amp = (monthly.groupby('obstruction_node_id')['median_by_month']
                           .quantile([0.25, 0.75]).unstack().rename(columns={0.25: 'q25', 0.75: 'q75'}))
                    amp['amplitude'] = amp['q75'] - amp['q25']
                    seasonality = monthly.merge(amp[['amplitude']], left_on='obstruction_node_id', right_index=True, how='left')

                seasonality_path = OUTPUT_DIR / f"master_wse_drop_seasonality_{sample_count}.parquet"
                with time_block("save_seasonality"):
                    seasonality.to_parquet(seasonality_path, index=False)
                logger.info(f"Saved seasonality summary to {seasonality_path}")
        except Exception as e:
            logger.warning(f"Failed to build/save temporal dynamics outputs: {e}")
    
    with time_block("groupby_median"):
        median_wse_df = wse_with_swarm_info.groupby(['obstruction_node_id', 'swarm_type'])['wse'].median().reset_index()
    
    with time_block("pivot_medians"):
        pivoted_wse_df = median_wse_df.pivot(index='obstruction_node_id', columns='swarm_type', values='wse'
        ).rename(columns={'upstream': 'median_wse_upstream', 'downstream': 'median_wse_downstream'}).reset_index()
    logger.info(f"Calculated median WSE for {len(pivoted_wse_df)} obstructions with ANY valid data.")

    # 5. Final Merge and Output
    logger.info("Step 5: Merging results and saving diagnostic outputs...")
    with time_block("final_merge"):
        analysis_df = pd.merge(master_obstructions_df, pivoted_wse_df, on='obstruction_node_id', how='left')
        analysis_df['wse_drop_m'] = analysis_df['median_wse_upstream'] - analysis_df['median_wse_downstream']
    
    # Add balancing information to the analysis
    if 'swarm_balanced' in swarm_map_df.columns:
        with time_block("merge_balancing_flags"):
            balancing_info = swarm_map_df.groupby('obstruction_node_id')['swarm_balanced'].any().reset_index()
            balancing_info = balancing_info.rename(columns={'swarm_balanced': 'swarms_were_balanced'})
            analysis_df = pd.merge(analysis_df, balancing_info, on='obstruction_node_id', how='left')
            analysis_df['swarms_were_balanced'] = analysis_df['swarms_were_balanced'].fillna(False).astype(bool)
        logger.info(f"Added balancing flags: {analysis_df['swarms_were_balanced'].sum()} obstructions had balanced swarms")

    # Save CSV
    sample_count = len(master_obstructions_df)
    output_csv_path = OUTPUT_DIR / f"master_wse_drop_crossreach_{sample_count}.csv"
    with time_block("save_csv"):
        analysis_df.to_csv(output_csv_path, index=False)
        # Retain legacy 10k-named output if sample is exactly 10000
        if sample_count == 10000:
            try:
                analysis_df.to_csv(OUTPUT_CSV, index=False)
            except Exception:
                pass
    logger.info(f"Saved CSV analysis to {output_csv_path}")

    # Save GeoJSON
    output_geojson_path = OUTPUT_DIR / f"master_wse_drop_crossreach_{sample_count}.geojson"
    with time_block("save_geojson"):
        geo_df = analysis_df.dropna(subset=['x', 'y']).copy()
        gdf = gpd.GeoDataFrame(geo_df)
        gdf.set_geometry(gpd.points_from_xy(gdf.x, gdf.y), crs="EPSG:4326", inplace=True)
        gdf.to_file(output_geojson_path, driver='GeoJSON')
        # Retain legacy 10k-named output if sample is exactly 10000
        if sample_count == 10000:
            try:
                gdf.to_file(OUTPUT_GEOJSON, driver='GeoJSON')
            except Exception:
                pass
    logger.info(f"Saved GeoJSON analysis to {output_geojson_path}")
    
    logger.info("--- Diagnostic Complete ---")
    valid_drops = analysis_df['wse_drop_m'].dropna()
    logger.info(f"Successfully calculated WSE drop for {len(valid_drops)} of {len(master_obstructions_df)} total obstructions.")
    if not valid_drops.empty:
        logger.info(f"Mean global WSE drop: {valid_drops.mean():.2f} meters.")
        logger.info(f"Median global WSE drop: {valid_drops.median():.2f} meters.")
        logger.info(f"Max WSE drop observed: {valid_drops.max():.2f} meters.")
    log_timings_summary()
    
    # Quick test is available via --quick-test flag; not run automatically after main

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick-test":
            # Run only the quick test
            quick_test_100_samples()
        elif sys.argv[1] == "--check-paths":
            # Check and display all paths
            check_paths()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Available options:")
            print("  --quick-test    : Run only the 100-sample diagnostic test")
            print("  --check-paths   : Check and display all file paths")
            print("  (no args)       : Run the full analysis")
    else:
        # Run the full analysis
        main() 