#!/usr/bin/env python3
"""
Comprehensive EDA script for WSE drop analysis across global obstructions.
Loads WSE drop data and merges in all relevant columns from SWORD database.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import duckdb
from pathlib import Path
import re
import geopandas as gpd
try:
    from scipy import stats as _scistats  # optional for QQ plot
except Exception:
    _scistats = None

# Configuration
DUCKDB_FILE = 'data/duckdb/sword_global.duckdb'
WSE_DROP_CSV = 'data/analysis/master_wse_drop_crossreach_20734.csv'  # Default; auto-detect if missing
OUTPUT_DIR = Path("data/analysis")

def _parse_sample_count_from_name(path: Path) -> int:
    """Extract numeric sample count from filename; handles legacy '_10k' as 10000."""
    name = path.stem  # e.g., master_wse_drop_crossreach_10000
    m = re.search(r"master_wse_drop_crossreach_(\d+)$", name)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return 0
    if name.endswith("_10k"):
        return 10000
    return 0

def _auto_find_wse_csv(default_path: Path) -> Path | None:
    """Find the best WSE CSV: prefer highest sample count; fallback to most recent."""
    if default_path.exists():
        return default_path
    candidates = list(OUTPUT_DIR.glob("master_wse_drop_crossreach_*.csv"))
    if not candidates:
        return None
    # Rank by parsed sample count, then by mtime
    candidates.sort(key=lambda p: (_parse_sample_count_from_name(p), p.stat().st_mtime), reverse=True)
    return candidates[0]

def load_and_merge_data():
    """Load WSE drop data and merge with SWORD database columns."""
    print("Loading WSE drop data...")
    
    # Resolve WSE drop analysis results path
    configured_path = Path(WSE_DROP_CSV)
    resolved_path = _auto_find_wse_csv(configured_path)
    if resolved_path is None:
        print(f"No WSE drop CSV found in {OUTPUT_DIR}. Please run the master analysis first.")
        return None
    print(f"Using WSE CSV: {resolved_path}")
    wse_df = pd.read_csv(resolved_path)
    print(f"Loaded {len(wse_df)} WSE drop records")
    
    # Connect to DuckDB and merge additional columns
    print("Merging additional columns from SWORD database...")
    with duckdb.connect(database=DUCKDB_FILE, read_only=True) as con:
        # Get the lists of IDs for filtering
        obstruction_node_ids = wse_df['obstruction_node_id'].dropna().unique().tolist()
        reach_ids = wse_df['reach_id'].dropna().unique().tolist()
        
        # Convert to strings for SQL IN clause
        node_ids_str = ','.join([str(int(x)) for x in obstruction_node_ids if pd.notna(x)])
        reach_ids_str = ','.join([str(int(x)) for x in reach_ids if pd.notna(x)])
        
        # Get node-level attributes
        if node_ids_str:
            node_attrs = con.execute(f"""
                SELECT 
                    node_id,
                    facc,
                    stream_order,
                    main_side,
                    sinuosity,
                    max_width,
                    meander_length,
                    width,
                    lakeflag,
                    trib_flag,
                    end_reach,
                    network
                FROM nodes 
                WHERE node_id IN ({node_ids_str})
            """).df()
        else:
            node_attrs = pd.DataFrame()
        
        # Get reach-level attributes  
        if reach_ids_str:
            reach_attrs = con.execute(f"""
                SELECT 
                    reach_id,
                    facc AS reach_facc,
                    reach_length,
                    n_nodes,
                    width AS reach_width
                FROM reaches 
                WHERE reach_id IN ({reach_ids_str})
            """).df()
        else:
            reach_attrs = pd.DataFrame()
    
    # Merge all the data
    print("Merging node attributes...")
    if not node_attrs.empty:
        clean = wse_df.merge(
            node_attrs, 
            left_on='obstruction_node_id', 
            right_on='node_id', 
            how='left'
        )
    else:
        clean = wse_df.copy()
        # Add empty columns for missing attributes
        for col in ['facc', 'stream_order', 'main_side', 'sinuosity', 'max_width', 'meander_length', 'width', 'lakeflag', 'trib_flag', 'end_reach', 'network']:
            clean[col] = pd.NA
    
    print("Merging reach attributes...")
    if not reach_attrs.empty:
        clean = clean.merge(
            reach_attrs, 
            on='reach_id', 
            how='left'
        )
    else:
        # Add empty columns for missing reach attributes
        for col in ['reach_facc', 'reach_length', 'n_nodes', 'reach_width']:
            clean[col] = pd.NA
    
    # Create derived columns
    print("Creating derived columns...")
    clean['slope'] = clean['reach_length'] / clean['dist_out']  # Approximate slope
    clean['facc_col'] = clean['facc'].fillna(clean['reach_facc'])  # Use node facc, fallback to reach facc
    clean['ord'] = clean['stream_order']  # Alias for compatibility
    clean['ms_col'] = clean['main_side']  # Alias for compatibility
    
    print(f"Final dataset: {len(clean)} rows, {len(clean.columns)} columns")
    print(f"Columns: {list(clean.columns)}")
    
    return clean

def print_distribution(df: pd.DataFrame, label: str = "") -> None:
    """Print distribution diagnostics for wse_drop_m."""
    if 'wse_drop_m' not in df.columns:
        print("wse_drop_m not found; skipping distribution report.")
        return
    s = pd.to_numeric(df['wse_drop_m'], errors='coerce').dropna()
    if s.empty:
        print("wse_drop_m is empty after coercion; skipping distribution report.")
        return
    qs = s.quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    print(f"\n=== Distribution report for wse_drop_m ({label}) ===")
    print(f"count={len(s)}, mean={s.mean():.3f}, std={s.std():.3f}, skew={s.skew():.3f}, kurtosis={s.kurtosis():.3f}")
    print("quantiles:")
    for q, v in qs.items():
        print(f"  q={q:>4}: {float(v):.3f}")

def clean_data(clean):
    """Clean the data by removing outliers and handling extreme values."""
    print("\n=== Data Cleaning ===")
    print(f"Initial dataset: {len(clean)} rows")
    
    # Convert numeric columns
    numeric_cols = ['wse_drop_m', 'dist_out', 'facc_col', 'slope', 'sinuosity']
    for col in numeric_cols:
        if col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors='coerce')
    
    # Remove rows with missing WSE drop (our target variable)
    initial_count = len(clean)
    clean = clean.dropna(subset=['wse_drop_m'])
    print(f"Removed {initial_count - len(clean)} rows with missing WSE drop")
    
    # Outlier removal using IQR method for key variables
    outlier_counts = {}
    
    # 1. WSE drop outliers (main target variable)
    if 'wse_drop_m' in clean.columns:
        Q1 = clean['wse_drop_m'].quantile(0.25)
        Q3 = clean['wse_drop_m'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = clean[(clean['wse_drop_m'] < lower_bound) | (clean['wse_drop_m'] > upper_bound)]
        outlier_counts['wse_drop_m'] = len(outliers)
        print(f"WSE drop outliers: {len(outliers)} ({len(outliers)/len(clean)*100:.1f}%)")
        print(f"  Range: {clean['wse_drop_m'].min():.2f} to {clean['wse_drop_m'].max():.2f}")
        print(f"  IQR bounds: {lower_bound:.2f} to {upper_bound:.2f}")
    
    # 2. Distance outliers
    if 'dist_out' in clean.columns:
        Q1 = clean['dist_out'].quantile(0.25)
        Q3 = clean['dist_out'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = clean[(clean['dist_out'] < lower_bound) | (clean['dist_out'] > upper_bound)]
        outlier_counts['dist_out'] = len(outliers)
        print(f"Distance outliers: {len(outliers)} ({len(outliers)/len(clean)*100:.1f}%)")
    
    # 3. Flow accumulation outliers
    if 'facc_col' in clean.columns:
        # Log transform for facc to handle extreme ranges
        clean['facc_log10'] = np.log10(np.clip(clean['facc_col'].astype(float), 1.0, None))
        
        Q1 = clean['facc_log10'].quantile(0.25)
        Q3 = clean['facc_log10'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = clean[(clean['facc_log10'] < lower_bound) | (clean['facc_log10'] > upper_bound)]
        outlier_counts['facc_log10'] = len(outliers)
        print(f"Flow accumulation (log) outliers: {len(outliers)} ({len(outliers)/len(clean)*100:.1f}%)")
    
    # 4. Slope ratio outliers
    if 'slope' in clean.columns:
        # Calculate dist_out/slope ratio
        clean['dist_out_slope_ratio'] = np.where(
            clean['slope'] > 0, 
            clean['dist_out'] / clean['slope'], 
            np.nan
        )
        
        # Log transform for the ratio
        clean['dist_out_slope_ratio_log'] = np.log10(np.clip(clean['dist_out_slope_ratio'], 1.0, None))
        
        Q1 = clean['dist_out_slope_ratio_log'].quantile(0.25)
        Q3 = clean['dist_out_slope_ratio_log'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = clean[(clean['dist_out_slope_ratio_log'] < lower_bound) | (clean['dist_out_slope_ratio_log'] > upper_bound)]
        outlier_counts['dist_out_slope_ratio_log'] = len(outliers)
        print(f"Distance/slope ratio (log) outliers: {len(outliers)} ({len(outliers)/len(clean)*100:.1f}%)")
    
    # Create cleaned dataset (remove outliers from all variables)
    print("\nCreating cleaned dataset...")
    mask = pd.Series([True] * len(clean), index=clean.index)
    
    for col, outlier_count in outlier_counts.items():
        if outlier_count > 0:
            if col == 'wse_drop_m':
                Q1 = clean[col].quantile(0.25)
                Q3 = clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                col_mask = (clean[col] >= lower_bound) & (clean[col] <= upper_bound)
            elif col == 'dist_out':
                Q1 = clean[col].quantile(0.25)
                Q3 = clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                col_mask = (clean[col] >= lower_bound) & (clean[col] <= upper_bound)
            elif col == 'facc_log10':
                Q1 = clean[col].quantile(0.25)
                Q3 = clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                col_mask = (clean[col] >= lower_bound) & (clean[col] <= upper_bound)
            elif col == 'dist_out_slope_ratio_log':
                Q1 = clean[col].quantile(0.25)
                Q3 = clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                col_mask = (clean[col] >= lower_bound) & (clean[col] <= upper_bound)
            else:
                col_mask = pd.Series([True] * len(clean), index=clean.index)
            
            mask = mask & col_mask
    
    clean_filtered = clean[mask].copy()
    print(f"Final cleaned dataset: {len(clean_filtered)} rows")
    print(f"Removed {len(clean) - len(clean_filtered)} outlier rows ({((len(clean) - len(clean_filtered))/len(clean)*100):.1f}%)")
    
    return clean_filtered

def create_plots(clean):
    """Create all the EDA plots with RANSAC fits and log transformations."""
    print("\nCreating plots...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1) Distribution of cleaned wse_drop_m
    if 'wse_drop_m' in clean.columns:
        print("Plot 1: WSE drop distribution")
        qlo, qhi = clean['wse_drop_m'].quantile([0.01, 0.99])
        bins = np.linspace(float(qlo), float(qhi), 50)
        plt.figure(figsize=(8, 6))
        sns.histplot(data=clean, x='wse_drop_m', bins=bins, color='steelblue', alpha=0.7)
        plt.title('Distribution of WSE Drop (cleaned)', fontsize=14)
        plt.xlabel('WSE Drop (m)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'wse_drop_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        # Optional QQ plot against Normal
        if _scistats is not None:
            try:
                plt.figure(figsize=(6, 6))
                _scistats.probplot(pd.to_numeric(clean['wse_drop_m'], errors='coerce').dropna(), dist="norm", plot=plt)
                plt.title('QQ Plot of WSE Drop (cleaned)')
                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / 'wse_drop_qq.png', dpi=300, bbox_inches='tight')
                plt.show()
            except Exception as _:
                pass

    # 2) dist_out vs wse_drop_m
    if {'dist_out','wse_drop_m'}.issubset(clean.columns):
        print("Plot 2: WSE drop vs distance from outlet")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=clean, x='dist_out', y='wse_drop_m', s=8, alpha=0.25, color='darkgreen')
        
        # Add RANSAC fit
        try:
            from sklearn.linear_model import RANSACRegressor
            from sklearn.preprocessing import StandardScaler
            
            # Prepare data for RANSAC
            X = clean[['dist_out']].dropna()
            y = clean.loc[X.index, 'wse_drop_m']
            
            if len(X) > 10:  # Need enough data for RANSAC
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                ransac = RANSACRegressor(random_state=42, max_trials=100)
                ransac.fit(X_scaled, y)
                
                # Plot RANSAC line
                X_plot = np.linspace(X.min().iloc[0], X.max().iloc[0], 100).reshape(-1, 1)
                X_plot_scaled = scaler.transform(X_plot)
                y_plot = ransac.predict(X_plot_scaled)
                
                plt.plot(X_plot, y_plot, color='red', linewidth=2, label=f'RANSAC (R²={ransac.score(X_scaled, y):.3f})')
                plt.legend()
                print(f"RANSAC fit: R² = {ransac.score(X_scaled, y):.3f}")
        except Exception as e:
            print(f"Could not add RANSAC fit: {e}")
        
        plt.title('WSE Drop vs Distance from Outlet (cleaned)', fontsize=14)
        plt.xlabel('Distance from Outlet (m)')
        plt.ylabel('WSE Drop (m)')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'wse_drop_vs_dist_out.png', dpi=300, bbox_inches='tight')
        plt.show()

    # 3) wse_drop_m vs facc (log10) with RANSAC
    if 'facc_log10' in clean.columns and clean['facc_log10'].notna().any():
        print("Plot 3: WSE drop vs log10(flow accumulation)")
        tmp = clean[['facc_log10', 'wse_drop_m']].dropna()
        
        if len(tmp) > 0:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=tmp, x='facc_log10', y='wse_drop_m', s=8, alpha=0.25, color='purple')
            
            # Add RANSAC fit
            try:
                from sklearn.linear_model import RANSACRegressor
                from sklearn.preprocessing import StandardScaler
                
                X = tmp[['facc_log10']]
                y = tmp['wse_drop_m']
                
                if len(X) > 10:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    ransac = RANSACRegressor(random_state=42, max_trials=100)
                    ransac.fit(X_scaled, y)
                    
                    # Plot RANSAC line
                    X_plot = np.linspace(X.min().iloc[0], X.max().iloc[0], 100).reshape(-1, 1)
                    X_plot_scaled = scaler.transform(X_plot)
                    y_plot = ransac.predict(X_plot_scaled)
                    
                    plt.plot(X_plot, y_plot, color='red', linewidth=2, label=f'RANSAC (R²={ransac.score(X_scaled, y):.3f})')
                    plt.legend()
                    print(f"RANSAC fit: R² = {ransac.score(X_scaled, y):.3f}")
            except Exception as e:
                print(f"Could not add RANSAC fit: {e}")
            
            plt.title('WSE Drop vs Log10(Flow Accumulation) (cleaned)', fontsize=14)
            plt.xlabel('Log10(Flow Accumulation)')
            plt.ylabel('WSE Drop (m)')
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'wse_drop_vs_facc.png', dpi=300, bbox_inches='tight')
            plt.show()

    # 4) wse_drop_m vs dist_out/slope (log) with RANSAC
    if 'dist_out_slope_ratio_log' in clean.columns and clean['dist_out_slope_ratio_log'].notna().any():
        print("Plot 4: WSE drop vs log10(dist_out/slope ratio)")
        tmp = clean[['dist_out_slope_ratio_log', 'wse_drop_m']].dropna()
        
        if len(tmp) > 0:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=tmp, x='dist_out_slope_ratio_log', y='wse_drop_m', s=8, alpha=0.25, color='orange')
            
            # Add RANSAC fit
            try:
                from sklearn.linear_model import RANSACRegressor
                from sklearn.preprocessing import StandardScaler
                
                X = tmp[['dist_out_slope_ratio_log']]
                y = tmp['wse_drop_m']
                
                if len(X) > 10:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    ransac = RANSACRegressor(random_state=42, max_trials=100)
                    ransac.fit(X_scaled, y)
                    
                    # Plot RANSAC line
                    X_plot = np.linspace(X.min().iloc[0], X.max().iloc[0], 100).reshape(-1, 1)
                    X_plot_scaled = scaler.transform(X_plot)
                    y_plot = ransac.predict(X_plot_scaled)
                    
                    plt.plot(X_plot, y_plot, color='red', linewidth=2, label=f'RANSAC (R²={ransac.score(X_scaled, y):.3f})')
                    plt.legend()
                    print(f"RANSAC fit: R² = {ransac.score(X_scaled, y):.3f}")
            except Exception as e:
                print(f"Could not add RANSAC fit: {e}")
            
            plt.title('WSE Drop vs Log10(Distance/Slope Ratio) (cleaned)', fontsize=14)
            plt.xlabel('Log10(Distance from Outlet / Slope)')
            plt.ylabel('WSE Drop (m)')
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'wse_drop_vs_dist_slope_ratio.png', dpi=300, bbox_inches='tight')
            plt.show()

    # 5) Group summaries by stream order
    if 'ord' in clean.columns and 'wse_drop_m' in clean.columns:
        print("Plot 5: WSE drop by stream order")
        g = clean[['ord','wse_drop_m']].copy()
        g['ord'] = pd.to_numeric(g['ord'], errors='coerce')
        g['wse_drop_m'] = pd.to_numeric(g['wse_drop_m'], errors='coerce')
        g = g.dropna()
        
        if len(g) > 0:
            # Create box plot
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=g, x='ord', y='wse_drop_m', color='lightblue')
            plt.title('WSE Drop by Stream Order (cleaned)', fontsize=14)
            plt.xlabel('Stream Order')
            plt.ylabel('WSE Drop (m)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'wse_drop_by_stream_order.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Print summary statistics
            agg = g.groupby('ord')['wse_drop_m'].agg(size='size', median='median', mean='mean').reset_index().sort_values('median', ascending=False)
            print('\nBy stream order (cleaned):')
            print(agg)

    # 6) Group summaries by main/side
    if 'ms_col' in clean.columns and 'wse_drop_m' in clean.columns:
        print("Plot 6: WSE drop by main/side")
        g2 = clean[['ms_col','wse_drop_m']].copy()
        g2['wse_drop_m'] = pd.to_numeric(g2['wse_drop_m'], errors='coerce')
        g2 = g2.dropna()
        
        if len(g2) > 0:
            # Create box plot
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=g2, x='ms_col', y='wse_drop_m', color='lightgreen')
            plt.title('WSE Drop by Main/Side (cleaned)', fontsize=14)
            plt.xlabel('Main/Side (1=Main, 2=Side)')
            plt.ylabel('WSE Drop (m)')
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'wse_drop_by_main_side.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Print summary statistics
            agg2 = g2.groupby('ms_col')['wse_drop_m'].agg(size='size', median='median', mean='mean').reset_index().sort_values('median', ascending=False)
            print('\nBy main/side (cleaned):')
            print(agg2)

    # 7) Additional useful plots
    if 'sinuosity' in clean.columns and 'wse_drop_m' in clean.columns:
        print("Plot 7: WSE drop vs sinuosity")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=clean, x='sinuosity', y='wse_drop_m', s=8, alpha=0.25, color='red')
        plt.title('WSE Drop vs Sinuosity (cleaned)', fontsize=14)
        plt.xlabel('Sinuosity')
        plt.ylabel('WSE Drop (m)')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'wse_drop_vs_sinuosity.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main execution function."""
    print("=== WSE Drop EDA Analysis with Robust Cleaning ===")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load and merge data
    clean = load_and_merge_data()
    if clean is None:
        return
    
    # Clean the data
    clean_filtered = clean_data(clean)
    # Distribution diagnostics
    print_distribution(clean, label="raw")
    print_distribution(clean_filtered, label="cleaned")
    
    # Create all plots with cleaned data
    create_plots(clean_filtered)
    
    # Save both datasets
    output_csv = OUTPUT_DIR / 'wse_drop_enriched.csv'
    output_cleaned_csv = OUTPUT_DIR / 'wse_drop_enriched_cleaned.csv'
    
    clean.to_csv(output_csv, index=False)
    clean_filtered.to_csv(output_cleaned_csv, index=False)
    # Save cleaned GeoJSON for visualization (requires x/y)
    if {'x', 'y'}.issubset(clean_filtered.columns):
        try:
            gf = clean_filtered.dropna(subset=['x', 'y']).copy()
            gdf = gpd.GeoDataFrame(gf, geometry=gpd.points_from_xy(gf['x'], gf['y']), crs="EPSG:4326")
            out_geo = OUTPUT_DIR / 'wse_drop_enriched_cleaned.geojson'
            gdf.to_file(out_geo, driver='GeoJSON')
            print(f"Cleaned GeoJSON saved to: {out_geo}")
        except Exception as e:
            print(f"Could not save cleaned GeoJSON: {e}")
    
    print(f"\nOriginal enriched dataset saved to: {output_csv}")
    print(f"Cleaned enriched dataset saved to: {output_cleaned_csv}")
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()
