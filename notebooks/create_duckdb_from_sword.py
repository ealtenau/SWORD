import netCDF4 as nc
import pandas as pd
import duckdb
import numpy as np
import os
import glob

def create_global_duckdb_from_sword(nc_dir, db_path):
    """
    Scans a directory for SWORD NetCDF files, processes each one,
    and loads the data into a single, consolidated DuckDB database.

    Args:
        nc_dir (str): The directory containing the SWORD NetCDF files.
        db_path (str): The file path for the output global DuckDB database.
    """
    print(f"Initializing Global DuckDB at: {db_path}")
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    
    # Delete the old database file if it exists to start fresh
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database file at {db_path}")

    con = duckdb.connect(database=db_path, read_only=False)

    # Find all continental SWORD files
    nc_files = glob.glob(os.path.join(nc_dir, '*_sword_v17b.nc'))
    print(f"Found {len(nc_files)} continental files to process.\n")

    is_first_file = True

    for nc_path in sorted(nc_files):
        continent_code = os.path.basename(nc_path).split('_')[0]
        print(f"--- Processing: {os.path.basename(nc_path)} (Continent: {continent_code}) ---")

        with nc.Dataset(nc_path, 'r') as ds:
            # 1. Process 'centerlines'
            centerlines_group = ds.groups['centerlines']
            centerlines_data = {'continent': continent_code}
            for var_name, var in centerlines_group.variables.items():
                if var_name in ['reach_id', 'node_id'] and len(var.dimensions) == 2:
                    for i in range(var.shape[0]):
                        centerlines_data[f'{var_name}_{i+1}'] = var[i, :]
                else:
                    centerlines_data[var_name] = var[:]
            centerlines_df = pd.DataFrame(centerlines_data)
            
            if is_first_file:
                con.execute("CREATE TABLE centerlines AS SELECT * FROM centerlines_df")
            else:
                con.execute("INSERT INTO centerlines SELECT * FROM centerlines_df")
            print(f"  -> Appended {len(centerlines_df)} records to 'centerlines'")

            # 2. Process 'nodes'
            nodes_group = ds.groups['nodes']
            nodes_data = {'continent': continent_code}
            for var_name, var in nodes_group.variables.items():
                if var_name == 'cl_ids' and len(var.dimensions) == 2:
                    nodes_data['cl_ids_min'] = var[0, :]
                    nodes_data['cl_ids_max'] = var[1, :]
                else:
                    if hasattr(var[:], 'filled'):
                        if np.issubdtype(var.dtype, np.integer):
                            fill_value = getattr(var, '_FillValue', -9999)
                            nodes_data[var_name] = var[:].filled(fill_value)
                        elif np.issubdtype(var.dtype, np.floating):
                            nodes_data[var_name] = var[:].filled(np.nan)
                        else:
                            nodes_data[var_name] = var[:].filled(None)
                    else:
                        nodes_data[var_name] = var[:]
            nodes_df = pd.DataFrame(nodes_data)

            if is_first_file:
                con.execute("CREATE TABLE nodes AS SELECT * FROM nodes_df")
            else:
                con.execute("INSERT INTO nodes SELECT * FROM nodes_df")
            print(f"  -> Appended {len(nodes_df)} records to 'nodes'")

            # 3. Process 'reaches'
            def extract_reaches_data(group, prefix=''):
                """Recursively extract data from nested groups, flattening into a single dict."""
                data = {}
                for var_name, var in group.variables.items():
                    col_name = f"{prefix}{var_name}"
                    
                    var_data = var[:]
                    # Use .filled() to handle masked arrays, with type awareness
                    if hasattr(var_data, 'filled'):
                        if np.issubdtype(var.dtype, np.integer):
                            fill_value = getattr(var, '_FillValue', -9999)
                            var_data = var_data.filled(fill_value)
                        elif np.issubdtype(var.dtype, np.floating):
                            var_data = var_data.filled(np.nan)
                        else:
                            var_data = var_data.filled(None) # For other types like string

                    # Handle specific multi-dimensional variables based on documentation
                    if var_name == 'cl_ids':
                        data[f'{col_name}_min'] = var_data[0, :]
                        data[f'{col_name}_max'] = var_data[1, :]
                    elif var_name in ['rch_id_up', 'rch_id_dn', 'h_break', 'w_break']:
                        for i in range(var.shape[0]):
                            data[f'{col_name}_{i+1}'] = var_data[i, :]
                    elif len(var.dimensions) > 1:
                        # For complex arrays (e.g., time series, coefficients), transpose
                        # and store as a list of lists/arrays in a single column.
                        # This is suitable for DuckDB's LIST type.
                        transposed_data = var_data.T
                        data[col_name] = [list(row) for row in transposed_data]
                    else:
                        data[col_name] = var_data
                
                for grp_name, sub_group in group.groups.items():
                    # Recursively call for nested groups, extending the prefix
                    new_prefix = f"{prefix}{grp_name}_"
                    data.update(extract_reaches_data(sub_group, prefix=new_prefix))
                
                return data

            reaches_data = extract_reaches_data(ds.groups['reaches'])
            reaches_df = pd.DataFrame(reaches_data)
            reaches_df['continent'] = continent_code
            
            # Reorder columns to have 'continent' first
            cols = ['continent'] + [col for col in reaches_df.columns if col != 'continent']
            reaches_df = reaches_df[cols]

            if is_first_file:
                con.execute("CREATE TABLE reaches AS SELECT * FROM reaches_df")
            else:
                con.execute("INSERT INTO reaches SELECT * FROM reaches_df")
            print(f"  -> Appended {len(reaches_df)} records to 'reaches'\n")
        
        is_first_file = False

    print("--- Global Database Creation Complete ---")
    print("Final table counts:")
    for table in ['centerlines', 'nodes', 'reaches']:
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  - {table}: {count:,} records")

    con.close()
    print(f"\nConnection to {db_path} closed.")

if __name__ == '__main__':
    NC_DIR = 'data/netcdf'
    DUCKDB_FILE = 'data/duckdb/sword_global.duckdb'
    
    create_global_duckdb_from_sword(nc_dir=NC_DIR, db_path=DUCKDB_FILE) 