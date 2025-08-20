import duckdb
import netCDF4 as nc
import numpy as np
import pandas as pd
import os
import argparse
import dataclasses
from typing import get_origin, get_args

from src.updates.sword_models import ReachData, NodeData, CenterlineData

def verify_db_integrity(db_path, nc_path, continent_code):
    """
    Verifies that data from a SWORD NetCDF file has been faithfully
    replicated in a DuckDB database. It checks record counts and summary
    statistics for key variables for a specific continent.

    Args:
        db_path (str): The file path for the target DuckDB database file.
        nc_path (str): The file path for the source SWORD NetCDF file.
        continent_code (str): The two-letter code for the continent to verify.
    
    Returns:
        bool: True if all checks pass, False otherwise.
    """
    print(f"--- Starting Verification for Continent: {continent_code.upper()} ---")
    print(f"NetCDF Source: {nc_path}")
    print(f"DuckDB Target: {db_path}\\n")

    all_checks_passed = True

    try:
        con = duckdb.connect(database=db_path, read_only=True)
        ds = nc.Dataset(nc_path, 'r')

        # --- Verification Checks ---
        checks = {
            'nodes': {
                'dim': 'num_nodes',
                'vars': {'wse': 'DOUBLE', 'width': 'DOUBLE', 'node_id': 'BIGINT'}
            },
            'reaches': {
                'dim': 'num_reaches',
                'vars': {'slope': 'DOUBLE', 'reach_length': 'DOUBLE', 'reach_id': 'BIGINT'}
            },
            'centerlines': {
                'dim': 'num_points',
                'vars': {'x': 'DOUBLE', 'y': 'DOUBLE', 'cl_id': 'BIGINT'}
            }
        }

        for table_name, check_details in checks.items():
            print(f"--- Verifying table: {table_name} ---")
            nc_group = ds.groups[table_name]
            
            # 1. Record Count Check
            nc_count = len(nc_group.dimensions[check_details['dim']])
            db_count_query = f"SELECT COUNT(*) FROM {table_name} WHERE continent = '{continent_code}'"
            db_count = con.execute(db_count_query).fetchone()[0]
            
            print(f"Record Count Check: NC={nc_count}, DB={db_count}")
            if nc_count == db_count:
                print("  -> PASSED\\n")
            else:
                print(f"  -> FAILED: Counts do not match for {table_name}.\\n")
                all_checks_passed = False
                continue # Skip to next table if counts don't match

            # 2. Summary Statistics Check for key variables
            for var_name, duckdb_type in check_details['vars'].items():
                print(f"  Variable '{var_name}' Statistics Check:")
                nc_var = nc_group.variables[var_name]
                
                # Use masked array methods for stats, which correctly handles fill values
                nc_data = nc_var[:]
                nc_mean = nc_data.mean()
                nc_sum = nc_data.sum()
                nc_std = nc_data.std()

                # DuckDB stats
                # Need to handle potential strings in river_name etc.
                # The check is designed for numeric types primarily
                try:
                    # Cast to the correct numeric type in DuckDB for calculation
                    db_stats_query = f"""
                    SELECT 
                        AVG(CAST({var_name} AS {duckdb_type})), 
                        SUM(CAST({var_name} AS {duckdb_type})), 
                        STDDEV_POP(CAST({var_name} AS {duckdb_type})) 
                    FROM {table_name}
                    WHERE continent = '{continent_code}'
                    """
                    db_stats = con.execute(db_stats_query).fetchone()
                    db_mean, db_sum, db_std = db_stats
                    
                    print(f"    Mean Check: NC={nc_mean:.6f}, DB={db_mean:.6f}")
                    assert np.isclose(nc_mean, db_mean, rtol=1e-5), "Mean values do not match"
                    
                    # For large integer IDs, the sum can overflow in numpy but not in DuckDB.
                    # We will skip the sum check for these specific fields.
                    if var_name in ['node_id', 'reach_id', 'cl_id']:
                        print(f"    Sum Check:  SKIPPED for large integer ID.")
                    else:
                        print(f"    Sum Check:  NC={nc_sum:.6f}, DB={db_sum:.6f}")
                        # For integer types, require exact match. For floats, allow tolerance.
                        if np.issubdtype(nc_data.dtype, np.integer):
                            assert nc_sum == db_sum, "Sum values do not match for integer type"
                        else:
                            assert np.isclose(nc_sum, db_sum, rtol=1e-5), "Sum values do not match"

                    print(f"    StdDev Check: NC={nc_std:.6f}, DB={db_std:.6f}")
                    assert np.isclose(nc_std, db_std, rtol=1e-5), "Standard deviation values do not match"
                    
                    print("    -> ALL STATS PASSED")

                except Exception as e:
                    print(f"    -> FAILED: Could not perform statistics check for {var_name}.")
                    print(f"       Error: {e}")
                    all_checks_passed = False
                finally:
                    print("")
            
            # 3. Coordinate Sampling Check (for centerlines)
            if table_name == 'centerlines':
                print(f"  Coordinate Sampling Check:")
                try:
                    num_samples = 50
                    # Generate random indices to sample
                    sample_indices = np.random.choice(nc_count, num_samples, replace=False)
                    sample_indices.sort() # Sorting is important for reproducibility if needed

                    nc_x = nc_group.variables['x'][sample_indices]
                    nc_y = nc_group.variables['y'][sample_indices]
                    # Instead of relying on order, we'll check some samples from DB by cl_id
                    nc_cl_ids = nc_group.variables['cl_id'][sample_indices]
                    
                    db_coords_query = f"""
                    SELECT x, y 
                    FROM {table_name} 
                    WHERE continent = '{continent_code}' AND cl_id IN {tuple(nc_cl_ids.tolist())} 
                    ORDER BY cl_id
                    """
                    db_coords_df = con.execute(db_coords_query).fetchdf()
                    db_x = db_coords_df['x'].to_numpy()
                    db_y = db_coords_df['y'].to_numpy()
                    
                    # Sort nc data by cl_id to match db order
                    sorted_indices = np.argsort(nc_cl_ids)
                    nc_x_sorted = nc_x[sorted_indices]
                    nc_y_sorted = nc_y[sorted_indices]
                    
                    assert np.allclose(nc_x_sorted, db_x), "X coordinates do not match"
                    assert np.allclose(nc_y_sorted, db_y), "Y coordinates do not match"
                    
                    print(f"    -> PASSED: Verified {num_samples} random coordinate pairs.\\n")
                except Exception as e:
                    print(f"    -> FAILED: Coordinate check failed for {table_name}.")
                    print(f"       Error: {e}\\n")
                    all_checks_passed = False

    except Exception as e:
        print(f"An unexpected error occurred during verification: {e}")
        all_checks_passed = False
    finally:
        if 'con' in locals(): con.close()
        if 'ds' in locals(): ds.close()

    print(f"--- Verification Complete ---")
    if all_checks_passed:
        print("Result: All checks passed. The data appears to be faithfully replicated. ✅")
    else:
        print("Result: One or more checks failed. Data replication may be incomplete or incorrect. ❌")
        
    return all_checks_passed

def get_expected_nc_columns(group, prefix=''):
    """
    Recursively scans a NetCDF group and returns a set of expected column names
    in the DuckDB table, mimicking the name-mangling from the creation script.
    """
    expected_cols = set()
    for var_name, var in group.variables.items():
        col_name = f"{prefix}{var_name}"
        
        # Mimic the logic from create_duckdb_from_sword.py
        if var_name == 'cl_ids':
            expected_cols.add(f'{col_name}_min')
            expected_cols.add(f'{col_name}_max')
        elif var_name in ['reach_id', 'node_id', 'rch_id_up', 'rch_id_dn', 'h_break', 'w_break']:
             # These are multi-dimensional arrays that are split into numbered columns
             if len(var.shape) > 1:
                for i in range(var.shape[0]):
                    expected_cols.add(f'{col_name}_{i+1}')
             else:
                expected_cols.add(col_name)
        else:
            expected_cols.add(col_name)
    
    for grp_name, sub_group in group.groups.items():
        new_prefix = f"{prefix}{grp_name}_"
        expected_cols.update(get_expected_nc_columns(sub_group, prefix=new_prefix))
        
    return expected_cols

def verify_schema_completeness(con, ds):
    """
    Compares the NetCDF schema against the DuckDB schema to ensure all expected
    columns have been created.
    """
    print("\\n" + "="*50)
    print("--- Starting Schema Completeness Verification ---")
    all_ok = True
    
    table_map = {
        'reaches': ds.groups['reaches'],
        'nodes': ds.groups['nodes'],
        'centerlines': ds.groups['centerlines']
    }
    
    for table_name, nc_group in table_map.items():
        print(f"\\n--- Verifying schema for table: {table_name} ---")
        
        # Get expected columns from NetCDF structure
        expected_cols = get_expected_nc_columns(nc_group)
        # Add the 'continent' column which is added during ingestion
        expected_cols.add('continent')
        
        # Get actual columns from DuckDB
        db_cols_df = con.execute(f"PRAGMA table_info('{table_name}')").fetchdf()
        actual_cols = set(db_cols_df['name'].tolist())
        
        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols
        
        if not missing_cols and not extra_cols:
            print(f"  -> PASSED: Schema is an exact match. ({len(actual_cols)} columns verified) ✅")
        else:
            all_ok = False
            if missing_cols:
                print(f"  -> FAILED: {len(missing_cols)} columns are missing from the DuckDB table:")
                for col in sorted(list(missing_cols)):
                    print(f"     - {col}")
            if extra_cols:
                print(f"  -> WARNING: {len(extra_cols)} columns exist in DuckDB but were not expected:")
                for col in sorted(list(extra_cols)):
                    print(f"     - {col}")
    
    print("\\n--- Schema Verification Complete ---")
    if all_ok:
        print("Result: All tables have a complete and matching schema. ✅")
    else:
        print("Result: Schema mismatches found. See details above. ❌")
        
    return all_ok

def verify_class_typing_against_netcdf(ds):
    """
    Verifies that the Python data models are type-compatible with the source NetCDF file.
    """
    print("\\n--- Verifying Class Types against NetCDF Schema ---")
    all_ok = True
    model_map = {'reaches': ReachData, 'nodes': NodeData, 'centerlines': CenterlineData}
    name_map = {'len': 'node_length', 'rch_id': 'reach_id'} # Model name -> NC name

    for group_name, model_class in model_map.items():
        print(f"\\nVerifying model '{model_class.__name__}' against NC group '{group_name}'...")
        nc_group = ds.groups[group_name]
        
        for model_field in dataclasses.fields(model_class):
            field_name = model_field.name
            field_type = model_field.type
            
            nc_var_name = name_map.get(field_name, field_name)
            
            if nc_var_name not in nc_group.variables:
                # This field is for compatibility and not in the source, which is acceptable.
                print(f"  - Field '{field_name}' not in NetCDF group. Assumed for compatibility. OK.")
                continue

            nc_var = nc_group.variables[nc_var_name]
            nc_dtype = nc_var.dtype
            nc_dims = len(nc_var.dimensions)
            
            type_ok = False
            origin_type = get_origin(field_type)
            if origin_type is list: # e.g., List[int]
                args = get_args(field_type)
                if args and args[0] is int and np.issubdtype(nc_dtype, np.integer) and nc_dims > 1:
                    type_ok = True
            elif field_type is int:
                if np.issubdtype(nc_dtype, np.integer): type_ok = True
            elif field_type is float:
                if np.issubdtype(nc_dtype, np.floating): type_ok = True
            elif field_type is str:
                if np.issubdtype(nc_dtype, np.character): type_ok = True
            elif field_type is np.ndarray:
                 if nc_dims >= 1: type_ok = True

            if type_ok:
                print(f"  - Field '{field_name}': Types compatible. OK.")
            else:
                all_ok = False
                print(f"  - Field '{field_name}': [MISMATCH] Model type '{field_type}' vs NC dtype '{nc_dtype}' with {nc_dims} dims.")

    if all_ok:
        print("\\nResult: All class models are type-compatible with the NetCDF source. ✅")
    else:
        print("\\nResult: Type mismatches found between class models and NetCDF source. ❌")
        
    return all_ok

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Verify SWORD data integrity between NetCDF and DuckDB.")
    parser.add_argument("continent", help="Two-letter continent code to verify (e.g., 'eu').")
    args = parser.parse_args()

    continent = args.continent.lower()
    NC_FILE = f'data/netcdf/{continent}_sword_v17b.nc'
    DUCKDB_FILE = 'data/duckdb/sword_global.duckdb'

    if not all([os.path.exists(NC_FILE), os.path.exists(DUCKDB_FILE)]):
        print("Error: Ensure both the NetCDF source file and the global DuckDB database file exist.")
    else:
        db_ok = verify_db_integrity(db_path=DUCKDB_FILE, nc_path=NC_FILE, continent_code=continent)
        
        # Also verify the class types
        print("\\n" + "="*50)
        ds = nc.Dataset(NC_FILE, 'r')
        con = duckdb.connect(database=DUCKDB_FILE, read_only=True)

        schema_ok = verify_schema_completeness(con, ds)
        class_ok = verify_class_typing_against_netcdf(ds)
        
        ds.close()
        con.close()
        
        print("\\n" + "="*50)
        if db_ok and class_ok and schema_ok:
            print("OVERALL RESULT: SUCCESS ✅")
        else:
            print("OVERALL RESULT: FAILED ❌") 