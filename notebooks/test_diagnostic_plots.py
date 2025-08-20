#!/usr/bin/env python3
"""
Simple test script to verify the diagnostic plotting functions work correctly.
This script will test the swarm visualization without running the full analysis.
"""

import sys
import os
from pathlib import Path

# Add the notebooks directory to the path
notebooks_dir = Path(__file__).parent
sys.path.insert(0, str(notebooks_dir))

from master_wse_analysis_crossreach_diag import quick_test_100_samples, check_paths

if __name__ == "__main__":
    print("Testing diagnostic plotting functions...")
    
    # First check paths
    print("\n1. Checking paths...")
    try:
        check_paths()
        print("✅ Path verification completed")
    except Exception as e:
        print(f"❌ Path verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Then run the quick test
    print("\n2. Running quick test...")
    try:
        quick_test_100_samples()
        print("✅ Diagnostic plotting test completed successfully!")
    except Exception as e:
        print(f"❌ Diagnostic plotting test failed: {e}")
        import traceback
        traceback.print_exc()
