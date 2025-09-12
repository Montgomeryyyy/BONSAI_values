"""
Compute theoretical AUC from actual data with single switch detection.

This module provides functions to calculate theoretical ROC AUC values
based on actual synthetic data, specifically for scenarios where
high-risk patients have exactly one switch between distributions.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Tuple, Optional
from sklearn.metrics import roc_auc_score


def detect_switches_in_data(data: pd.DataFrame, lab_code: str = "S/LAB1") -> pd.DataFrame:
    """
    Detect switches in lab values for each patient.
    
    Args:
        data: DataFrame with columns 'subject_id', 'code', 'numeric_value', 'time'
        lab_code: The lab code to analyze (default: "S/LAB1")
        
    Returns:
        pd.DataFrame: DataFrame with switch information for each patient
    """
    # Filter for lab data only
    lab_data = data[data['code'] == lab_code].copy()
    
    if len(lab_data) == 0:
        raise ValueError(f"No data found for lab code: {lab_code}")
    
    # Sort by patient and time
    lab_data = lab_data.sort_values(['subject_id', 'time'])
    
    # Calculate threshold (midpoint between min and max values)
    min_val = lab_data['numeric_value'].min()
    max_val = lab_data['numeric_value'].max()
    threshold = (min_val + max_val) / 2
    
    # Group by patient and detect switches
    patient_switches = []
    
    for patient_id, patient_labs in lab_data.groupby('subject_id'):
        values = patient_labs['numeric_value'].values
        
        if len(values) < 2:
            # Need at least 2 values to detect a switch
            patient_switches.append({
                'subject_id': patient_id,
                'n_labs': len(values),
                'n_switches': 0,
                'has_switch': False,
                'switch_ratio': 0.0
            })
            continue
        
        # Count switches (values crossing the threshold)
        switches = 0
        for i in range(1, len(values)):
            if (values[i-1] < threshold) != (values[i] < threshold):
                switches += 1
        
        patient_switches.append({
            'subject_id': patient_id,
            'n_labs': len(values),
            'n_switches': switches,
            'has_switch': switches > 0,
            'switch_ratio': switches / (len(values) - 1) if len(values) > 1 else 0.0
        })
    
    return pd.DataFrame(patient_switches)


def compute_switch_based_auc(data: pd.DataFrame, positive_diags: list = ["S/DIAG_POSITIVE"]) -> float:
    """
    Compute AUC based on switch detection in the actual data.
    
    Args:
        data: DataFrame with synthetic data
        positive_diags: List of diagnosis codes indicating positive cases
        
    Returns:
        float: AUC based on switch detection
    """
    # Get positive patients
    positive_patients = set()
    for diag in positive_diags:
        positive_patients.update(data[data["code"] == diag]["subject_id"].unique())
    
    # Detect switches for each patient
    switch_data = detect_switches_in_data(data)
    
    # Add positive/negative labels
    switch_data['is_positive'] = switch_data['subject_id'].isin(positive_patients)
    
    # Calculate AUC using number of switches as the score
    if len(switch_data) == 0:
        return 0.5
    
    try:
        auc = roc_auc_score(switch_data['is_positive'], switch_data['n_switches'])
        return auc
    except ValueError:
        # Handle case where all labels are the same
        return 0.5


def compute_variance_based_auc(data: pd.DataFrame, positive_diags: list = ["S/DIAG_POSITIVE"]) -> float:
    """
    Compute AUC based on variance differences in lab sequences.
    
    Args:
        data: DataFrame with synthetic data
        positive_diags: List of diagnosis codes indicating positive cases
        
    Returns:
        float: AUC based on variance differences
    """
    # Get positive patients
    positive_patients = set()
    for diag in positive_diags:
        positive_patients.update(data[data["code"] == diag]["subject_id"].unique())
    
    # Calculate variance for each patient's lab sequence
    lab_data = data[data['code'] == 'S/LAB1'].copy()
    patient_variances = []
    
    for patient_id, patient_labs in lab_data.groupby('subject_id'):
        values = patient_labs['numeric_value'].values
        if len(values) > 1:
            variance = np.var(values)
        else:
            variance = 0.0
        
        patient_variances.append({
            'subject_id': patient_id,
            'variance': variance,
            'is_positive': patient_id in positive_patients
        })
    
    variance_df = pd.DataFrame(patient_variances)
    
    if len(variance_df) == 0:
        return 0.5
    
    try:
        auc = roc_auc_score(variance_df['is_positive'], variance_df['variance'])
        return auc
    except ValueError:
        return 0.5


def compute_sequence_complexity_auc(data: pd.DataFrame, positive_diags: list = ["S/DIAG_POSITIVE"]) -> float:
    """
    Compute AUC based on sequence complexity (combination of switches and variance).
    
    Args:
        data: DataFrame with synthetic data
        positive_diags: List of diagnosis codes indicating positive cases
        
    Returns:
        float: AUC based on sequence complexity
    """
    # Get positive patients
    positive_patients = set()
    for diag in positive_diags:
        positive_patients.update(data[data["code"] == diag]["subject_id"].unique())
    
    # Get switch data
    switch_data = detect_switches_in_data(data)
    
    # Calculate variance for each patient
    lab_data = data[data['code'] == 'S/LAB1'].copy()
    patient_complexity = []
    
    for patient_id, patient_labs in lab_data.groupby('subject_id'):
        values = patient_labs['numeric_value'].values
        
        # Get switch information
        switch_info = switch_data[switch_data['subject_id'] == patient_id]
        n_switches = switch_info['n_switches'].iloc[0] if len(switch_info) > 0 else 0
        
        # Calculate variance
        variance = np.var(values) if len(values) > 1 else 0.0
        
        # Calculate range
        value_range = np.max(values) - np.min(values) if len(values) > 1 else 0.0
        
        # Combine metrics into complexity score
        complexity_score = n_switches * 0.5 + variance * 10 + value_range * 0.1
        
        patient_complexity.append({
            'subject_id': patient_id,
            'complexity': complexity_score,
            'is_positive': patient_id in positive_patients
        })
    
    complexity_df = pd.DataFrame(patient_complexity)
    
    if len(complexity_df) == 0:
        return 0.5
    
    try:
        auc = roc_auc_score(complexity_df['is_positive'], complexity_df['complexity'])
        return auc
    except ValueError:
        return 0.5


def compute_theoretical_auc_from_data(data: pd.DataFrame, 
                                    positive_diags: list = ["S/DIAG_POSITIVE"],
                                    lab_code: str = "S/LAB1") -> Dict[str, float]:
    """
    Compute comprehensive theoretical AUC values from actual data.
    
    This function analyzes the actual data to compute theoretical AUC values
    based on different approaches for detecting high-risk patients who switch
    between distributions.
    
    Args:
        data: DataFrame with synthetic data containing 'subject_id', 'code', 
              'numeric_value', and 'time' columns
        positive_diags: List of diagnosis codes indicating positive cases
        lab_code: The lab code to analyze
        
    Returns:
        Dict[str, float]: Dictionary containing different theoretical AUC values
    """
    results = {}
    
    try:
        # 1. Switch-based AUC
        results['switch_based_auc'] = compute_switch_based_auc(data, positive_diags)
        
        # 2. Variance-based AUC
        results['variance_based_auc'] = compute_variance_based_auc(data, positive_diags)
        
        # 3. Sequence complexity AUC
        results['sequence_complexity_auc'] = compute_sequence_complexity_auc(data, positive_diags)
        
        # 4. Basic distribution AUC (using all lab values)
        lab_data = data[data['code'] == lab_code]
        if len(lab_data) > 0:
            positive_patients = set()
            for diag in positive_diags:
                positive_patients.update(data[data["code"] == diag]["subject_id"].unique())
            
            lab_data['is_positive'] = lab_data['subject_id'].isin(positive_patients)
            
            try:
                results['basic_distribution_auc'] = roc_auc_score(
                    lab_data['is_positive'], lab_data['numeric_value']
                )
            except ValueError:
                results['basic_distribution_auc'] = 0.5
        else:
            results['basic_distribution_auc'] = 0.5
        
        # 5. Calculate summary statistics
        results['average_auc'] = np.mean(list(results.values()))
        results['max_auc'] = np.max(list(results.values()))
        results['min_auc'] = np.min(list(results.values()))
        
    except Exception as e:
        print(f"Error computing theoretical AUCs: {e}")
        # Return default values
        results = {
            'switch_based_auc': 0.5,
            'variance_based_auc': 0.5,
            'sequence_complexity_auc': 0.5,
            'basic_distribution_auc': 0.5,
            'average_auc': 0.5,
            'max_auc': 0.5,
            'min_auc': 0.5
        }
    
    return results


def print_theoretical_auc_results(data: pd.DataFrame, 
                                positive_diags: list = ["S/DIAG_POSITIVE"],
                                lab_code: str = "S/LAB1") -> None:
    """
    Print comprehensive theoretical AUC results from actual data.
    
    Args:
        data: DataFrame with synthetic data
        positive_diags: List of diagnosis codes indicating positive cases
        lab_code: The lab code to analyze
    """
    # Get basic data statistics
    lab_data = data[data['code'] == lab_code]
    positive_patients = set()
    for diag in positive_diags:
        positive_patients.update(data[data["code"] == diag]["subject_id"].unique())
    
    total_patients = data['subject_id'].nunique()
    positive_count = len(positive_patients)
    negative_count = total_patients - positive_count
    
    print("\n" + "="*60)
    print("THEORETICAL AUC FROM ACTUAL DATA")
    print("="*60)
    print(f"Data Statistics:")
    print(f"  Total patients: {total_patients}")
    print(f"  Positive patients: {positive_count}")
    print(f"  Negative patients: {negative_count}")
    print(f"  Lab records: {len(lab_data)}")
    
    if len(lab_data) > 0:
        print(f"  Lab value range: {lab_data['numeric_value'].min():.3f} - {lab_data['numeric_value'].max():.3f}")
        print(f"  Lab value mean: {lab_data['numeric_value'].mean():.3f}")
        print(f"  Lab value std: {lab_data['numeric_value'].std():.3f}")
    
    # Compute theoretical AUCs
    auc_results = compute_theoretical_auc_from_data(data, positive_diags, lab_code)
    
    print(f"\nTheoretical AUC Values:")
    print(f"  Switch-based AUC: {auc_results['switch_based_auc']:.4f}")
    print(f"  Variance-based AUC: {auc_results['variance_based_auc']:.4f}")
    print(f"  Sequence complexity AUC: {auc_results['sequence_complexity_auc']:.4f}")
    print(f"  Basic distribution AUC: {auc_results['basic_distribution_auc']:.4f}")
    
    print(f"\nSummary:")
    print(f"  Average AUC: {auc_results['average_auc']:.4f}")
    print(f"  Best AUC: {auc_results['max_auc']:.4f}")
    print(f"  Worst AUC: {auc_results['min_auc']:.4f}")
    print(f"  AUC Range: {auc_results['max_auc'] - auc_results['min_auc']:.4f}")
    
    print("="*60)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute theoretical AUC from actual data")
    parser.add_argument("--data_file", type=str, required=True, 
                       help="Path to CSV file containing synthetic data")
    parser.add_argument("--positive_diags", nargs="+", default=["S/DIAG_POSITIVE"],
                       help="List of positive diagnosis codes")
    parser.add_argument("--lab_code", type=str, default="S/LAB1",
                       help="Lab code to analyze")
    
    args = parser.parse_args()
    
    # Load data
    data = pd.read_csv(args.data_file)
    
    # Print results
    print_theoretical_auc_results(data, args.positive_diags, args.lab_code)
