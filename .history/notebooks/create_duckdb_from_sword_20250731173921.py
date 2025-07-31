import netCDF4 as nc
import pandas as pd
import duckdb
import numpy as np
import os

def create_duckdb_from_sword(nc_path, db_path):
    """
    Reads a SWORD NetCDF file, processes its groups (centerlines, nodes, reaches),
    and loads the data into a DuckDB database with three corresponding tables.

    This function handles the nested and multi-dimensional nature of the NetCDF data
    by flattening structures into columns and using list types for complex arrays.

    Args:
        nc_path (str): The file path for the input SWORD NetCDF file.
        db_path (str): The file path for the output DuckDB database file.
    """
    print(f"Connecting to DuckDB at: {db_path}")
    # Ensure the directory for the database exists
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    con = duckdb.connect(database=db_path, read_only=False)

    with nc.Dataset(nc_path, 'r') as ds:
        # 1. Process 'centerlines' group
        print("Processing 'centerlines' group...")
        centerlines_group = ds.groups['centerlines']
        centerlines_data = {}
        for var_name, var in centerlines_group.variables.items():
            # Handle multi-dimensional ID fields by splitting them into separate columns
            if var_name in ['reach_id', 'node_id'] and len(var.dimensions) == 2:
                for i in range(var.shape[0]):
                    centerlines_data[f'{var_name}_{i+1}'] = var[i, :]
            else:
                centerlines_data[var_name] = var[:]
        centerlines_df = pd.DataFrame(centerlines_data)
        con.execute("CREATE OR REPLACE TABLE centerlines AS SELECT * FROM centerlines_df")
        print("'centerlines' table created successfully.")

        # 2. Process 'nodes' group
        print("\nProcessing 'nodes' group...")
        nodes_group = ds.groups['nodes']
        nodes_data = {}
        for var_name, var in nodes_group.variables.items():
            # Handle min/max ID fields
            if var_name == 'cl_ids' and len(var.dimensions) == 2:
                nodes_data['cl_id_min'] = var[0, :]
                nodes_data['cl_id_max'] = var[1, :]
            else:
                # Use .filled() to handle masked arrays from NetCDF, with type awareness
                if hasattr(var[:], 'filled'):
                    if np.issubdtype(var.dtype, np.integer):
                        # For integers, use a specific integer fill value, like the one from the file
                        fill_value = getattr(var, '_FillValue', -9999)
                        nodes_data[var_name] = var[:].filled(fill_value)
                    elif np.issubdtype(var.dtype, np.floating):
                        nodes_data[var_name] = var[:].filled(np.nan)
                    else:
                        nodes_data[var_name] = var[:].filled(None) # For other types like string
                else:
                    nodes_data[var_name] = var[:]
        nodes_df = pd.DataFrame(nodes_data)
        con.execute("CREATE OR REPLACE TABLE nodes AS SELECT * FROM nodes_df")
        print("'nodes' table created successfully.")

        # 3. Process 'reaches' group (the most complex one)
        print("\nProcessing 'reaches' group...")

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
        con.execute("CREATE OR REPLACE TABLE reaches AS SELECT * FROM reaches_df")
        print("'reaches' table created successfully.")

    print("\nDatabase creation complete.")
    print("Tables created in the database:")
    print(con.execute("SHOW TABLES").fetchdf())

    con.close()
    print(f"\nConnection to {db_path} closed.")

if __name__ == '__main__':
    # Define file paths
    # Assumes the script is run from the root of the SWORD project directory
    NC_FILE = 'data/netcdf/eu_sword_v17b.nc'
    DUCKDB_FILE = 'data/duckdb/sword_eu.duckdb'
    
    # Check if the NetCDF file exists before running
    if not os.path.exists(NC_FILE):
        print(f"Error: NetCDF file not found at '{NC_FILE}'")
        print("Please ensure the file is downloaded and placed in the correct directory.")
    else:
        create_duckdb_from_sword(nc_path=NC_FILE, db_path=DUCKDB_FILE) 