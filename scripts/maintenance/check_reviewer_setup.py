#!/usr/bin/env python3
"""
SWORD Reviewer Setup Checker
-----------------------------
Run this script to verify all dependencies and data are ready
for the SWORD QA Reviewer.

Usage: python check_reviewer_setup.py
"""

import sys
from pathlib import Path

REQUIRED_PACKAGES = [
    ("streamlit", "streamlit"),
    ("streamlit_folium", "streamlit-folium"),
    ("folium", "folium"),
    ("pydeck", "pydeck"),
    ("duckdb", "duckdb"),
    ("pandas", "pandas"),
]

DB_PATH = Path("data/duckdb/sword_v17c.duckdb")

passed = 0
failed = 0
total = 0


def check(name, ok, fix_msg=""):
    global passed, failed, total
    total += 1
    if ok:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")
        if fix_msg:
            print(f"         -> {fix_msg}")


# 1. Python version
print("=" * 50)
print("SWORD Reviewer Setup Check")
print("=" * 50)
print()
print("[1/5] Python version")
py_ok = sys.version_info >= (3, 9)
check(
    f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    py_ok,
    "Install Python 3.9+ from https://www.python.org/downloads/",
)

# 2. Required packages
print()
print("[2/5] Required packages")
missing = []
for module_name, pip_name in REQUIRED_PACKAGES:
    try:
        __import__(module_name)
        check(pip_name, True)
    except ImportError:
        check(pip_name, False)
        missing.append(pip_name)

if missing:
    print()
    print(f"  Fix: pip install {' '.join(missing)}")
    print("  Or:  pip install -r requirements-reviewer.txt")

# 3. Database file
print()
print("[3/5] Database file")
db_exists = DB_PATH.exists()
check(
    str(DB_PATH),
    db_exists,
    "Copy sword_v17c.duckdb to data/duckdb/ folder",
)

# 4. Database query
print()
print("[4/5] Database connectivity")
if db_exists:
    try:
        import duckdb

        conn = duckdb.connect(str(DB_PATH), read_only=True)
        count = conn.execute("SELECT COUNT(*) FROM reaches").fetchone()[0]
        conn.close()
        check(f"Can query reaches table ({count:,} rows)", True)
    except Exception as e:
        check("Query reaches table", False, str(e))
else:
    check("Query reaches table", False, "Database not found (skipped)")

# 5. Lint check imports
print()
print("[5/5] Lint check imports")
try:
    from src.sword_duckdb.lint.checks.classification import (
        check_lake_sandwich,
        check_lakeflag_type_consistency,
    )
    from src.sword_duckdb.lint.checks.topology import (
        check_facc_monotonicity,
        check_orphan_reaches,
    )
    from src.sword_duckdb.lint.checks.attributes import (
        check_slope_reasonableness,
        check_end_reach_consistency,
    )

    check("All lint checks importable", True)
except ImportError as e:
    check("Lint check imports", False, str(e))

# Summary
print()
print("=" * 50)
if failed == 0:
    print(f"ALL {total} CHECKS PASSED")
    print()
    print("Ready! Launch the reviewer with:")
    print("  streamlit run topology_reviewer.py")
else:
    print(f"{failed} of {total} checks FAILED")
    print()
    print("Fix the issues above and run this script again.")
    sys.exit(1)
