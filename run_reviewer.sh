#!/bin/bash
# SWORD QA Reviewer - One-click launch
# Run this script to start the reviewer UI.

set -e

echo "Checking setup..."
python check_reviewer_setup.py

echo ""
echo "Launching SWORD Reviewer..."
streamlit run topology_reviewer.py
