#!/bin/bash
set -e

echo "=== SWORD Reviewer startup ==="
echo "Checking /mnt/gcs..."
ls -la /mnt/gcs/ 2>&1 || echo "/mnt/gcs not found or empty"
ls -la /mnt/gcs/sword/ 2>&1 || echo "/mnt/gcs/sword not found"

# Copy DB from GCS FUSE to local /tmp (DuckDB needs local random access)
mkdir -p /tmp/sword
if [ -f /mnt/gcs/sword/sword_v17c.duckdb ]; then
    cp /mnt/gcs/sword/sword_v17c.duckdb /tmp/sword/sword_v17c.duckdb
    echo "Copied DB: $(du -h /tmp/sword/sword_v17c.duckdb | cut -f1)"
else
    echo "WARNING: /mnt/gcs/sword/sword_v17c.duckdb NOT FOUND"
    echo "SWORD_DB_PATH=$SWORD_DB_PATH"
fi

# Ensure lint_fixes dir exists on GCS mount
if [ -d /mnt/gcs ]; then
    mkdir -p /mnt/gcs/sword/lint_fixes
fi

echo "=== Starting Streamlit ==="
exec streamlit run app.py \
    --server.port=8080 \
    --server.address=0.0.0.0 \
    --server.headless=true
