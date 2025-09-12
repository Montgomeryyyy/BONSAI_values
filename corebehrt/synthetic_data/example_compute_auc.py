#!/usr/bin/env python3
"""
Example script showing how to compute theoretical AUC from actual data.
"""

import pandas as pd
import sys
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from compute_theoretical_auc_from_data import (
    compute_theoretical_auc_from_data,
    print_theoretical_auc_results
)


def main():
    """Example usage of the theoretical AUC computation functions."""
    
    # Example 1: Using a data file
    print("Example 1: Computing theoretical AUC from a data file")
    print("-" * 50)
    
    # You can use this with your actual data file
    # data_file = "path/to/your/data.csv"
    # data = pd.read_csv(data_file)
    # print_theoretical_auc_results(data)
    
    # Example 2: Using the function directly
    print("\nExample 2: Using the function directly")
    print("-" * 50)
    
    # Create some example data for demonstration
    example_data = create_example_data()
    
    # Compute theoretical AUCs
    auc_results = compute_theoretical_auc_from_data(example_data)
    
    print("Theoretical AUC Results:")
    for method, auc in auc_results.items():
        print(f"  {method}: {auc:.4f}")


def create_example_data():
    """Create example synthetic data for demonstration."""
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create example data
    data = []
    
    # High-risk patients (switch between distributions)
    for i in range(10):
        patient_id = f"high_risk_{i}"
        
        # First few labs from high distribution
        for j in range(3):
            data.append({
                'subject_id': patient_id,
                'code': 'S/LAB1',
                'numeric_value': np.random.normal(0.7, 0.05),
                'time': datetime(2020, 1, 1) + timedelta(days=j*30)
            })
        
        # Switch to low distribution
        for j in range(3, 6):
            data.append({
                'subject_id': patient_id,
                'code': 'S/LAB1',
                'numeric_value': np.random.normal(0.3, 0.05),
                'time': datetime(2020, 1, 1) + timedelta(days=j*30)
            })
        
        # Add positive diagnosis
        data.append({
            'subject_id': patient_id,
            'code': 'S/DIAG_POSITIVE',
            'numeric_value': 1.0,
            'time': datetime(2020, 6, 1)
        })
    
    # Low-risk patients (consistent distribution)
    for i in range(10):
        patient_id = f"low_risk_{i}"
        
        # All labs from one distribution (randomly chosen)
        distribution_mean = np.random.choice([0.3, 0.7])
        
        for j in range(6):
            data.append({
                'subject_id': patient_id,
                'code': 'S/LAB1',
                'numeric_value': np.random.normal(distribution_mean, 0.05),
                'time': datetime(2020, 1, 1) + timedelta(days=j*30)
            })
        
        # Add negative diagnosis
        data.append({
            'subject_id': patient_id,
            'code': 'S/DIAG_NEGATIVE',
            'numeric_value': 1.0,
            'time': datetime(2020, 6, 1)
        })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    main()
