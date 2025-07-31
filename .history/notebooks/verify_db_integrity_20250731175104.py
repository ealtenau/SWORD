import duckdb
import netCDF4 as nc
import numpy as np
import pandas as pd
import os

def verify_db_integrity(nc_path, db_path):
    """
    Verifies that data from a SWORD NetCDF file has been faithfully
    replicated in a DuckDB database. It checks record counts and summary
    statistics for key variables.

    Args:
        nc_path (str): The file path for the source SWORD NetCDF file.
        db_path (str): The file path for the target DuckDB database file.
    
    Returns:
        bool: True if all checks pass, False otherwise.
    """
    print(f"--- Starting Verification ---")
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
            db_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            
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
                    db_stats = con.execute(f"SELECT AVG(CAST({var_name} AS {duckdb_type})), SUM(CAST({var_name} AS {duckdb_type})), STDDEV_POP(CAST({var_name} AS {duckdb_type})) FROM {table_name}").fetchone()
                    db_mean, db_sum, db_std = db_stats
                    
                    print(f"    Mean Check: NC={nc_mean:.6f}, DB={db_mean:.6f}")
                    assert np.isclose(nc_mean, db_mean, rtol=1e-5), "Mean values do not match"
                    
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

if __name__ == '__main__':
    NC_FILE = 'data/netcdf/eu_sword_v17b.nc'
    DUCKDB_FILE = 'data/duckdb/sword_eu.duckdb'

    if not all([os.path.exists(NC_FILE), os.path.exists(DUCKDB_FILE)]):
        print("Error: Ensure both the NetCDF source file and the DuckDB database file exist.")
    else:
        verify_db_integrity(nc_path=NC_FILE, db_path=DUCKDB_FILE) 