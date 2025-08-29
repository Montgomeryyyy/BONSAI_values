#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'corebehrt'))

from corebehrt.azure.util.config import map_azure_path

def test_path_mapping():
    """Test the path mapping function with the input from the command."""
    
    # Test the exact path from the command
    test_path = "researcher_data:BC_recreate/data_20240910/MEDS_old_v2/data"
    print(f"Original path: {test_path}")
    
    try:
        mapped_path = map_azure_path(test_path)
        print(f"Mapped path: {mapped_path}")
    except Exception as e:
        print(f"Error mapping path: {e}")
    
    # Test some variations
    test_cases = [
        "researcher_data:test/path",
        "sp_data:test/path", 
        "azureml:test/path",
        "local/path",
        "asset_name:version"
    ]
    
    print("\nTesting other path formats:")
    for path in test_cases:
        try:
            mapped = map_azure_path(path)
            print(f"  {path} -> {mapped}")
        except Exception as e:
            print(f"  {path} -> ERROR: {e}")

if __name__ == "__main__":
    test_path_mapping() 