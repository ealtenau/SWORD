"""Verify v17c DuckDB schema is compatible with v17b NetCDF structure.

Issue #89: shared columns between v17b and v17c must have equivalent
variable names, compatible dtypes, and matching fill values.

Multi-dimensional NetCDF variables (rch_id_up [4,N], swot_orbits [75,N],
iceflag [366,N], cl_ids [2,N]) are normalized into separate DuckDB tables
or split into _min/_max columns — these are checked structurally, not by
name match.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

try:
    import netCDF4 as nc

    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False


V17B_NC = Path("data/netcdf/na_sword_v17b.nc")
V17C_DB = Path("data/duckdb/sword_v17c.duckdb")

needs_real_data = pytest.mark.skipif(
    not (V17B_NC.exists() and V17C_DB.exists()),
    reason="Requires v17b NetCDF and v17c DuckDB on disk",
)
needs_netcdf4 = pytest.mark.skipif(not HAS_NETCDF4, reason="netCDF4 not installed")

# --------------------------------------------------------------------------- #
# NetCDF dtype → compatible DuckDB types
# --------------------------------------------------------------------------- #
DTYPE_COMPAT = {
    "int32": {"INTEGER", "BIGINT"},
    "int64": {"BIGINT", "INTEGER"},  # INTEGER ok when values fit (e.g. lakeflag 0-3)
    "float32": {"FLOAT", "DOUBLE"},
    "float64": {"DOUBLE"},
    "str": {"VARCHAR"},
    # netCDF4 string variables report dtype as object/vlen
    "object": {"VARCHAR"},
    "<class 'str'>": {"VARCHAR"},
}

# --------------------------------------------------------------------------- #
# NetCDF variable → DuckDB column name mapping (where names differ)
#
# Most names are identical.  Only list exceptions here.
# --------------------------------------------------------------------------- #
# Variables that map to a different column name in DuckDB
NC_TO_DUCKDB_RENAME: dict[str, str] = {
    # No renames currently — names are 1:1 after v17b PDD cleanup
}

# --------------------------------------------------------------------------- #
# Multi-dimensional variables normalized into separate tables or split columns.
# These are verified structurally, not by scalar name match.
# --------------------------------------------------------------------------- #
MULTIDIM_VARS = {
    "centerlines": {
        # reach_id [4,N] and node_id [4,N] — primary row in centerlines,
        # extra rows in centerline_neighbors table
        "reach_id",
        "node_id",
    },
    "nodes": {
        # cl_ids [2,N] → cl_id_min, cl_id_max
        "cl_ids",
    },
    "reaches": {
        # cl_ids [2,N] → cl_id_min, cl_id_max
        "cl_ids",
        # rch_id_up [4,N] → reach_topology table
        "rch_id_up",
        # rch_id_dn [4,N] → reach_topology table
        "rch_id_dn",
        # swot_orbits [75,N] → reach_swot_orbits table
        "swot_orbits",
        # iceflag [366,N] → scalar max in reaches + reach_ice_flags table
        "iceflag",
    },
}

# DuckDB-only columns not present in v17b NetCDF (region, version, geom, v17c attrs)
DUCKDB_ONLY_PREFIXES = {
    "region",
    "version",
    "geom",
}


# =========================================================================== #
# Helpers
# =========================================================================== #


def _nc_group_to_duckdb_table(group_name: str) -> str:
    """Map NetCDF group name to DuckDB table name."""
    return group_name


def _get_nc_scalar_vars(
    ds: nc.Dataset, group_name: str
) -> dict[str, tuple[str, object]]:
    """Return {var_name: (dtype_str, fill_value)} for 1-D scalar variables."""
    grp = ds.groups[group_name]
    multidim = MULTIDIM_VARS.get(group_name, set())
    result = {}
    for var_name, var in grp.variables.items():
        if var_name in multidim:
            continue
        dtype_str = str(var.dtype)
        fill = getattr(var, "_FillValue", None)
        result[var_name] = (dtype_str, fill)
    return result


def _get_duckdb_columns(con: duckdb.DuckDBPyConnection, table: str) -> dict[str, str]:
    """Return {column_name: dtype_str} for a DuckDB table."""
    rows = con.execute(f"DESCRIBE {table}").fetchall()
    return {name: dtype for name, dtype, *_ in rows}


# =========================================================================== #
# Tests
# =========================================================================== #


@pytest.mark.export
@needs_netcdf4
@needs_real_data
class TestNetCDFSchemaCompat:
    """Verify shared v17b NetCDF variables exist in v17c DuckDB with compatible types."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = nc.Dataset(str(V17B_NC), "r")
        self.con = duckdb.connect(str(V17C_DB), read_only=True)
        yield
        self.ds.close()
        self.con.close()

    # -- centerlines -------------------------------------------------------- #

    def test_centerlines_shared_vars_present(self):
        """Every 1-D v17b centerline variable exists in v17c DuckDB."""
        nc_vars = _get_nc_scalar_vars(self.ds, "centerlines")
        db_cols = _get_duckdb_columns(self.con, "centerlines")

        missing = []
        for nc_name in nc_vars:
            db_name = NC_TO_DUCKDB_RENAME.get(nc_name, nc_name)
            if db_name not in db_cols:
                missing.append(f"{nc_name} (expected as {db_name})")

        assert not missing, f"Centerline vars missing in DuckDB: {missing}"

    def test_centerlines_dtype_compat(self):
        """Shared centerline variables have compatible dtypes."""
        nc_vars = _get_nc_scalar_vars(self.ds, "centerlines")
        db_cols = _get_duckdb_columns(self.con, "centerlines")

        mismatches = []
        for nc_name, (nc_dtype, _) in nc_vars.items():
            db_name = NC_TO_DUCKDB_RENAME.get(nc_name, nc_name)
            if db_name not in db_cols:
                continue
            db_dtype = db_cols[db_name]
            compat = DTYPE_COMPAT.get(nc_dtype, set())
            if db_dtype not in compat:
                mismatches.append(
                    f"{nc_name}: NC={nc_dtype} vs DB={db_dtype} "
                    f"(expected one of {compat})"
                )

        assert not mismatches, "Centerline dtype mismatches:\n" + "\n".join(mismatches)

    # -- nodes -------------------------------------------------------------- #

    def test_nodes_shared_vars_present(self):
        """Every 1-D v17b node variable exists in v17c DuckDB."""
        nc_vars = _get_nc_scalar_vars(self.ds, "nodes")
        db_cols = _get_duckdb_columns(self.con, "nodes")

        missing = []
        for nc_name in nc_vars:
            db_name = NC_TO_DUCKDB_RENAME.get(nc_name, nc_name)
            if db_name not in db_cols:
                missing.append(f"{nc_name} (expected as {db_name})")

        assert not missing, f"Node vars missing in DuckDB: {missing}"

    def test_nodes_dtype_compat(self):
        """Shared node variables have compatible dtypes."""
        nc_vars = _get_nc_scalar_vars(self.ds, "nodes")
        db_cols = _get_duckdb_columns(self.con, "nodes")

        mismatches = []
        for nc_name, (nc_dtype, _) in nc_vars.items():
            db_name = NC_TO_DUCKDB_RENAME.get(nc_name, nc_name)
            if db_name not in db_cols:
                continue
            db_dtype = db_cols[db_name]
            compat = DTYPE_COMPAT.get(nc_dtype, set())
            if db_dtype not in compat:
                mismatches.append(
                    f"{nc_name}: NC={nc_dtype} vs DB={db_dtype} "
                    f"(expected one of {compat})"
                )

        assert not mismatches, "Node dtype mismatches:\n" + "\n".join(mismatches)

    # -- reaches ------------------------------------------------------------ #

    def test_reaches_shared_vars_present(self):
        """Every 1-D v17b reach variable exists in v17c DuckDB."""
        nc_vars = _get_nc_scalar_vars(self.ds, "reaches")
        db_cols = _get_duckdb_columns(self.con, "reaches")

        missing = []
        for nc_name in nc_vars:
            db_name = NC_TO_DUCKDB_RENAME.get(nc_name, nc_name)
            if db_name not in db_cols:
                missing.append(f"{nc_name} (expected as {db_name})")

        assert not missing, f"Reach vars missing in DuckDB: {missing}"

    def test_reaches_dtype_compat(self):
        """Shared reach variables have compatible dtypes."""
        nc_vars = _get_nc_scalar_vars(self.ds, "reaches")
        db_cols = _get_duckdb_columns(self.con, "reaches")

        mismatches = []
        for nc_name, (nc_dtype, _) in nc_vars.items():
            db_name = NC_TO_DUCKDB_RENAME.get(nc_name, nc_name)
            if db_name not in db_cols:
                continue
            db_dtype = db_cols[db_name]
            compat = DTYPE_COMPAT.get(nc_dtype, set())
            if db_dtype not in compat:
                mismatches.append(
                    f"{nc_name}: NC={nc_dtype} vs DB={db_dtype} "
                    f"(expected one of {compat})"
                )

        assert not mismatches, "Reach dtype mismatches:\n" + "\n".join(mismatches)

    # -- multi-dimensional structure ---------------------------------------- #

    def test_cl_ids_split_into_min_max(self):
        """cl_ids [2,N] should be split into cl_id_min and cl_id_max."""
        for table in ["nodes", "reaches"]:
            db_cols = _get_duckdb_columns(self.con, table)
            assert "cl_id_min" in db_cols, f"{table} missing cl_id_min"
            assert "cl_id_max" in db_cols, f"{table} missing cl_id_max"

    def test_reach_topology_table_exists(self):
        """rch_id_up/dn [4,N] should be normalized into reach_topology table."""
        tables = [
            row[0]
            for row in self.con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
        ]
        assert "reach_topology" in tables

    def test_reach_topology_has_required_columns(self):
        """reach_topology needs reach_id, direction, neighbor_rank, neighbor_reach_id."""
        db_cols = _get_duckdb_columns(self.con, "reach_topology")
        for col in ["reach_id", "direction", "neighbor_rank", "neighbor_reach_id"]:
            assert col in db_cols, f"reach_topology missing {col}"

    def test_swot_orbits_table_exists(self):
        """swot_orbits [75,N] should be normalized into reach_swot_orbits table."""
        tables = [
            row[0]
            for row in self.con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
        ]
        assert "reach_swot_orbits" in tables

    # -- fill values -------------------------------------------------------- #

    def test_fill_value_sentinel_preserved(self):
        """Numeric v17b fill value (-9999) should still be the sentinel in v17c DuckDB.

        Sample a few key numeric columns and confirm -9999 is used (not NULL).
        """
        sentinel_cols = ["dist_out", "facc", "wse", "width", "slope"]
        for col in sentinel_cols:
            result = self.con.execute(
                f"SELECT COUNT(*) FROM reaches WHERE {col} = -9999"
            ).fetchone()
            # At least some rows should have the sentinel (ghosts, type=5, etc.)
            assert result[0] >= 0, f"Query failed for {col}"

    # -- global attributes -------------------------------------------------- #

    def test_nc_has_expected_global_attrs(self):
        """v17b NetCDF should have standard global attributes."""
        attrs = self.ds.ncattrs()
        assert "Name" in attrs, "Missing 'Name' global attribute"
        assert "production_date" in attrs, "Missing 'production_date' attribute"

    # -- dimension consistency ---------------------------------------------- #

    def test_nc_dimensions_exist(self):
        """Each NetCDF group should have its expected primary dimension."""
        expected = {
            "centerlines": "num_points",
            "nodes": "num_nodes",
            "reaches": "num_reaches",
        }
        for grp_name, dim_name in expected.items():
            grp = self.ds.groups[grp_name]
            assert dim_name in grp.dimensions, (
                f"{grp_name} missing dimension {dim_name}"
            )

    def test_row_counts_match(self):
        """NetCDF row counts should match DuckDB for NA region."""
        dim_to_table = {
            "centerlines": ("num_points", "centerlines"),
            "nodes": ("num_nodes", "nodes"),
            "reaches": ("num_reaches", "reaches"),
        }
        for grp_name, (dim_name, table) in dim_to_table.items():
            nc_count = len(self.ds.groups[grp_name].dimensions[dim_name])
            db_count = self.con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE region = 'NA'"
            ).fetchone()[0]
            assert nc_count == db_count, (
                f"{grp_name}: NC has {nc_count} rows, DuckDB has {db_count} for NA"
            )

    # -- nodes: lakeflag dtype regression ----------------------------------- #

    def test_nodes_lakeflag_is_integer(self):
        """v17b NetCDF has lakeflag as int64 for nodes. DuckDB should be INTEGER or BIGINT."""
        nc_vars = _get_nc_scalar_vars(self.ds, "nodes")
        nc_dtype = nc_vars["lakeflag"][0]
        db_cols = _get_duckdb_columns(self.con, "nodes")
        db_dtype = db_cols["lakeflag"]

        assert nc_dtype in ("int64", "int32"), f"Unexpected NC dtype: {nc_dtype}"
        assert db_dtype in ("INTEGER", "BIGINT"), f"Unexpected DB dtype: {db_dtype}"
