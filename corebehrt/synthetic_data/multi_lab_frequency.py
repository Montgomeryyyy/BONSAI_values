"""
Generate synthetic data with multiple lab values where all patients have the same distribution
of lab values, but positive patients have more lab tests on average.
Based on the multi_lab_sharp_edge.py structure with concept relationships.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import os
from theoretical_separation import (
    cohens_d,
    sweep_threshold_auc,
    manual_mann_whitney_u,
    scipy_mann_whitney_u,
)

# Default parameters
N = 100000
DEFAULT_INPUT_FILE = f"../../../data/vals/synthetic_data/{N}n/bn_labs_n{N}_50p_1unq.csv"
PATIENTS_INFO_PATH = f"../../../data/vals/patient_infos/patient_info_{N}n.parquet"
# Gaussian parameters for number of labs per patient
LOW_RISK_LABS_MEAN = 14.3
LOW_RISK_LABS_STD = 2
HIGH_RISK_LABS_MEAN = 14.7
HIGH_RISK_LABS_STD = 2
MIN_LABS_PER_PATIENT = 1
LAB_MEAN = 0.5  # Same mean for all patients
LAB_STD = 0.1   # Same std for all patients
DEFAULT_WRITE_DIR = f"../../../data/vals/synthetic_data/{N}n/"
DEFAULT_PLOT_DIR = f"../../../data/vals/synthetic_data_plots/{N}n/"
SAVE_NAME = f"multi_lab_frequency_gaussian_low{int(LOW_RISK_LABS_MEAN)}p{int(LOW_RISK_LABS_STD*10)}_high{int(HIGH_RISK_LABS_MEAN)}p{int(HIGH_RISK_LABS_STD*10)}_n{N}_mean{int(LAB_MEAN*100)}_std{int(LAB_STD*100)}"
POSITIVE_DIAGS = ["S/DIAG_POSITIVE"]

# Define lab value distributions - same for all patients
LAB_VALUE_INFO = {
    "S/LAB1": {
        "distribution": {
            "dist": "normal",
            "mean": LAB_MEAN,
            "std": LAB_STD,
        },
    },
    "S/LAB2": {
        "distribution": {
            "dist": "normal",
            "mean": LAB_MEAN,
            "std": LAB_STD,
        },
    },
}

# Define concept relationships similar to multi_lab_sharp_edge.py
CONCEPT_RELATIONSHIPS = {
    "S/LAB1": {
        "base_probability": 1.0,  # 100% of patients get LAB1
        "condition_probabilities": {
            "high_risk": 0.5,  # 50% chance of being high-risk (more LAB1)
            "low_risk": 0.5,  # 50% chance of being low-risk (fewer LAB1)
        },
        "add_base_concept": ["high_risk", "low_risk"],  # Add LAB1 for all conditions
        "related_concepts": {
            "S/DIAG_POSITIVE": {
                "prob": 1,  # 100% chance of getting diagnosis if high-risk
                "conditions": ["high_risk"],  # Only high-risk patients get positive diagnosis
                "time_relationship": {
                    "type": "after",  # Diagnosis comes after labs
                    "min_days": 10,
                    "max_days": 180,
                },
            },
            "S/DIAG_NEGATIVE": {
                "prob": 1,  # 100% chance of getting diagnosis if low-risk
                "conditions": ["low_risk"],  # Only low-risk patients get negative diagnosis
                "time_relationship": {
                    "type": "after",  # Diagnosis comes after labs
                    "min_days": 10,
                    "max_days": 180,
                },
            }
        },
    },
    "S/LAB2": {
        "base_probability": 1.0,  # 100% of patients get LAB2 (filler)
        "condition_probabilities": {
            "high_risk": 0.5,  # 50% chance of being high-risk
            "low_risk": 0.5,  # 50% chance of being low-risk
        },
        "add_base_concept": ["high_risk", "low_risk"],  # Add LAB2 for all conditions
    },
}


def get_positive_patients(data: pd.DataFrame, positive_diags: list) -> pd.DataFrame:
    """
    Get positive patients from the data and add is_positive column.

    Args:
        data: DataFrame containing the synthetic data
        positive_diags: List of diagnosis codes that indicate positive cases

    Returns:
        pd.DataFrame: DataFrame with added is_positive column
    """
    positive_patients = set()
    for diag in positive_diags:
        positive_patients.update(data[data["code"] == diag]["subject_id"].unique())

    data["is_positive"] = data["subject_id"].isin(positive_patients)
    return data


def generate_lab_value(lab_name: str) -> Optional[float]:
    """
    Generate a lab value based on the lab name.
    All patients use the same distribution.

    Args:
        lab_name: Name of the lab test

    Returns:
        Optional[float]: Generated lab value or None if invalid input
    """
    if lab_name not in LAB_VALUE_INFO:
        return None

    range_info = LAB_VALUE_INFO[lab_name]["distribution"]
    if range_info["dist"] == "uniform":
        return np.random.choice(range_info["range"])
    elif range_info["dist"] == "normal":
        return np.random.normal(range_info["mean"], range_info["std"])
    return None


def generate_multi_lab_concepts(pids_list: List[str], low_risk_mean: float, low_risk_std: float, 
                               high_risk_mean: float, high_risk_std: float, min_labs: int, patient_risk_map: dict) -> pd.DataFrame:
    """
    Generate multiple lab concepts and values for a list of patient IDs.
    High-risk patients get more LAB1 tests, low-risk patients get fewer LAB1 tests.
    The remaining slots are filled with LAB2 tests. No maximum constraint.

    Args:
        pids_list: List of patient IDs
        low_risk_mean: Mean number of LAB1 tests for low-risk patients
        low_risk_std: Standard deviation for LAB1 tests for low-risk patients
        high_risk_mean: Mean number of LAB1 tests for high-risk patients
        high_risk_std: Standard deviation for LAB1 tests for high-risk patients
        min_labs: Minimum number of labs per patient
        patient_risk_map: Dictionary mapping patient_id to risk status (True=high_risk, False=low_risk)

    Returns:
        pd.DataFrame: DataFrame containing PID, CONCEPT, and RESULT columns
    """
    records = []

    for pid in pids_list:
        # Use existing patient risk assignment
        is_positive = patient_risk_map.get(pid, False)
        condition = "high_risk" if is_positive else "low_risk"

        # Determine number of LAB1 tests based on risk status
        if condition == "high_risk":
            n_lab1 = int(np.random.normal(high_risk_mean, high_risk_std))
        else:
            n_lab1 = int(np.random.normal(low_risk_mean, low_risk_std))
        
        # Ensure LAB1 count is not negative
        n_lab1 = max(0, n_lab1)
        
        # Determine total number of labs based on LAB1 count
        # Total labs = LAB1 + LAB2, where LAB2 fills the remaining slots
        # We want total labs to be at least min_labs, but can be more if needed
        total_labs = max(min_labs, n_lab1 + 1)  # At least min_labs, at least 1 more than LAB1
        n_lab2 = total_labs - n_lab1  # Fill remaining with LAB2
        
        # Generate LAB1 tests
        for i in range(n_lab1):
            value = generate_lab_value("S/LAB1")
            if value is not None:
                records.append({
                    "PID": pid, 
                    "CONCEPT": "S/LAB1", 
                    "RESULT": value,
                    "LAB_INDEX": i,
                    "CONDITION": condition
                })
        
        # Generate LAB2 tests (filler)
        for i in range(n_lab2):
            value = generate_lab_value("S/LAB2")
            if value is not None:
                records.append({
                    "PID": pid, 
                    "CONCEPT": "S/LAB2", 
                    "RESULT": value,
                    "LAB_INDEX": i,
                    "CONDITION": condition
                })

        # Add diagnosis based on risk status
        if condition == "high_risk":
            # High-risk patients get positive diagnosis
            records.append({
                "PID": pid, 
                "CONCEPT": "S/DIAG_POSITIVE", 
                "RESULT": 1.0,
                "LAB_INDEX": -1,
                "CONDITION": condition
            })
        else:
            # Low-risk patients get negative diagnosis
            records.append({
                "PID": pid, 
                "CONCEPT": "S/DIAG_NEGATIVE", 
                "RESULT": 1.0,
                "LAB_INDEX": -1,
                "CONDITION": condition
            })

    return pd.DataFrame(records)


def generate_timestamps(
    pids_list: List[str], concepts: List[str], lab_indices: List[int], 
    patient_df: pd.DataFrame = None
) -> List[pd.Timestamp]:
    """
    Generate timestamps for a list of patient IDs based on time relationships.
    Similar to multi_lab_sharp_edge.py but adapted for multiple labs per patient.

    Args:
        pids_list: List of patient IDs to generate timestamps for
        concepts: List of concepts corresponding to each PID
        lab_indices: List of lab indices corresponding to each record
        patient_df: DataFrame containing patient information with birthdate and deathdate columns

    Returns:
        List[pd.Timestamp]: List of generated timestamps
    """
    timestamps = []
    concept_timestamps = {}  # Store timestamps for each concept per patient

    for i, (pid, concept, lab_index) in enumerate(zip(pids_list, concepts, lab_indices)):
        # Initialize patient's concept timestamps if not exists
        if pid not in concept_timestamps:
            concept_timestamps[pid] = {}
            
            # Get patient birth and death dates from patient_df
            if patient_df is not None:
                patient_info = patient_df[patient_df["subject_id"] == pid].iloc[0]
                birthdate = pd.to_datetime(patient_info["birthdate"])
                
                # Handle deathdate - if NaT, use a default future date
                deathdate = pd.to_datetime(patient_info["deathdate"])
                if pd.isna(deathdate):
                    deathdate = pd.Timestamp(year=2025, month=1, day=1)
                    
                # Ensure deathdate is after birthdate
                if deathdate <= birthdate:
                    deathdate = birthdate + pd.Timedelta(days=1)
                    
                # Generate a random start time for this patient between birth and death
                time_diff = (deathdate - birthdate).total_seconds()
                random_seconds = np.random.randint(0, int(time_diff))
                concept_timestamps[pid]["start_time"] = birthdate + pd.Timedelta(seconds=random_seconds)
            else:
                # Fallback to default time range if no patient_df provided
                start_time = pd.Timestamp(year=2016, month=1, day=1)
                end_time = pd.Timestamp(year=2025, month=1, day=1)
                time_diff = (end_time - start_time).total_seconds()
                random_seconds = np.random.randint(0, int(time_diff))
                concept_timestamps[pid]["start_time"] = start_time + pd.Timedelta(seconds=random_seconds)

        # Find the base concept and its time relationship for this concept
        time_relationship = None
        base_concept = None

        for bc, info in CONCEPT_RELATIONSHIPS.items():
            if concept in info.get("related_concepts", {}):
                time_relationship = info["related_concepts"][concept].get("time_relationship")
                base_concept = bc
                break

        if time_relationship and base_concept:
            # If we have a time relationship and the base concept exists for this patient
            if base_concept in concept_timestamps[pid]:
                base_timestamp = concept_timestamps[pid][base_concept]
                if time_relationship["type"] == "after":
                    # Generate timestamp after the base concept
                    max_days = time_relationship["max_days"]
                    min_days = time_relationship["min_days"]
                    days_after = np.random.randint(min_days, max_days + 1)
                    timestamp = base_timestamp + pd.Timedelta(days=days_after)
            else:
                # If base concept doesn't exist yet, generate a random timestamp
                start_time = concept_timestamps[pid]["start_time"]
                timestamp = start_time + pd.Timedelta(days=np.random.randint(0, 365))
        else:
            # For base concepts (labs), generate timestamp based on lab index
            start_time = concept_timestamps[pid]["start_time"]
            if lab_index == 0:
                timestamp = start_time
            else:
                # Each subsequent lab is 1-30 days after the previous
                days_offset = sum(np.random.randint(1, 31) for _ in range(lab_index))
                timestamp = start_time + pd.Timedelta(days=days_offset)

        # Store the timestamp for this concept
        concept_timestamps[pid][concept] = timestamp
        timestamps.append(timestamp)

    return timestamps


def generate_synthetic_data(
    input_data: pd.DataFrame,
    low_risk_mean: float,
    low_risk_std: float,
    high_risk_mean: float,
    high_risk_std: float,
    min_labs: int,
    patient_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Generate synthetic data with multiple lab values per patient.
    High-risk patients get more LAB1 tests, low-risk patients get fewer LAB1 tests.
    No maximum constraint on total labs.
    
    Args:
        input_data: DataFrame containing existing synthetic data with patient assignments
        low_risk_mean: Mean number of LAB1 tests for low-risk patients
        low_risk_std: Standard deviation for LAB1 tests for low-risk patients
        high_risk_mean: Mean number of LAB1 tests for high-risk patients
        high_risk_std: Standard deviation for LAB1 tests for high-risk patients
        min_labs: Minimum number of labs per patient
        patient_df: DataFrame containing patient information with birthdate and deathdate columns
        
    Returns:
        pd.DataFrame: Generated synthetic data
    """
    # Get positive patients from input data
    input_data_with_risk = get_positive_patients(input_data, POSITIVE_DIAGS)
    
    # Create patient risk mapping
    patient_risk_map = {}
    for patient_id in input_data_with_risk["subject_id"].unique():
        patient_data = input_data_with_risk[input_data_with_risk["subject_id"] == patient_id]
        is_positive = patient_data["is_positive"].iloc[0]
        patient_risk_map[patient_id] = is_positive
    
    # Get patient IDs
    pids_list = list(patient_risk_map.keys())
    
    # Generate concepts and lab values
    concepts_data = generate_multi_lab_concepts(
        pids_list, low_risk_mean, low_risk_std, 
        high_risk_mean, high_risk_std, min_labs, patient_risk_map
    )

    # Create final DataFrame - match multi_lab_sharp_edge.py structure exactly
    data = pd.DataFrame({
        "subject_id": concepts_data["PID"],
        "code": concepts_data["CONCEPT"],
        "numeric_value": concepts_data["RESULT"].astype(float),
    })

    # Generate timestamps for each record
    data["time"] = generate_timestamps(
        data["subject_id"].tolist(), 
        data["code"].tolist(),
        concepts_data["LAB_INDEX"].tolist(),
        patient_df
    )

    return data


def print_statistics(data: pd.DataFrame) -> None:
    """
    Print statistics about the lab values and frequency.

    Args:
        data: DataFrame containing the synthetic data
    """
    # Recreate is_positive column for analysis
    positive_patients = set(data[data["code"] == "S/DIAG_POSITIVE"]["subject_id"].unique())
    data["is_positive"] = data["subject_id"].isin(positive_patients)
    positive_mask = data["is_positive"]

    # Get lab masks for both LAB1 and LAB2
    lab1_mask = data["code"] == "S/LAB1"
    lab2_mask = data["code"] == "S/LAB2"
    all_labs_mask = data["code"].str.startswith("S/LAB")

    print("\nLab value statistics (all patients use same distribution):")
    print(f"LAB1 - Count: {len(data[lab1_mask])}")
    print(f"LAB1 - Mean: {data[lab1_mask]['numeric_value'].mean():.3f}")
    print(f"LAB1 - Std: {data[lab1_mask]['numeric_value'].std():.3f}")
    print(f"LAB1 - Min: {data[lab1_mask]['numeric_value'].min():.3f}")
    print(f"LAB1 - Max: {data[lab1_mask]['numeric_value'].max():.3f}")
    
    print(f"LAB2 - Count: {len(data[lab2_mask])}")
    print(f"LAB2 - Mean: {data[lab2_mask]['numeric_value'].mean():.3f}")
    print(f"LAB2 - Std: {data[lab2_mask]['numeric_value'].std():.3f}")
    print(f"LAB2 - Min: {data[lab2_mask]['numeric_value'].min():.3f}")
    print(f"LAB2 - Max: {data[lab2_mask]['numeric_value'].max():.3f}")

    # Count labs per patient
    positive_lab1_per_patient = data[lab1_mask & positive_mask].groupby("subject_id").size()
    negative_lab1_per_patient = data[lab1_mask & ~positive_mask].groupby("subject_id").size()
    positive_total_per_patient = data[all_labs_mask & positive_mask].groupby("subject_id").size()
    negative_total_per_patient = data[all_labs_mask & ~positive_mask].groupby("subject_id").size()

    print(f"\nLAB1 frequency statistics (determines risk):")
    print(f"High-risk patients - Avg LAB1 per patient: {positive_lab1_per_patient.mean():.1f}")
    print(f"High-risk patients - Min LAB1 per patient: {positive_lab1_per_patient.min()}")
    print(f"High-risk patients - Max LAB1 per patient: {positive_lab1_per_patient.max()}")
    print(f"Low-risk patients - Avg LAB1 per patient: {negative_lab1_per_patient.mean():.1f}")
    print(f"Low-risk patients - Min LAB1 per patient: {negative_lab1_per_patient.min()}")
    print(f"Low-risk patients - Max LAB1 per patient: {negative_lab1_per_patient.max()}")
    
    print(f"\nTotal lab frequency statistics (LAB1 + LAB2):")
    print(f"High-risk patients - Avg total labs per patient: {positive_total_per_patient.mean():.1f}")
    print(f"High-risk patients - Min total labs per patient: {positive_total_per_patient.min()}")
    print(f"High-risk patients - Max total labs per patient: {positive_total_per_patient.max()}")
    print(f"Low-risk patients - Avg total labs per patient: {negative_total_per_patient.mean():.1f}")
    print(f"Low-risk patients - Min total labs per patient: {negative_total_per_patient.min()}")
    print(f"Low-risk patients - Max total labs per patient: {negative_total_per_patient.max()}")


def create_distribution_plot(
    data: pd.DataFrame, save_path: Path
) -> None:
    """
    Create a figure showing the distribution of lab values and frequency.

    Args:
        data: DataFrame containing the synthetic data
        save_path: Path to save the plot
    """
    # Get lab values for positive and negative patients
    lab_mask = data["code"] == "S/LAB1"
    lab_data = data[lab_mask].copy()
    
    # Recreate is_positive column for analysis
    positive_patients = set(data[data["code"] == "S/DIAG_POSITIVE"]["subject_id"].unique())
    lab_data["is_positive"] = lab_data["subject_id"].isin(positive_patients)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Histogram of lab values (should be similar for both groups)
    positive_values = lab_data[lab_data["is_positive"]]["numeric_value"]
    negative_values = lab_data[~lab_data["is_positive"]]["numeric_value"]

    ax1.hist(positive_values, bins=30, alpha=0.7, label="High-Risk Patients", color="red")
    ax1.hist(negative_values, bins=30, alpha=0.7, label="Low-Risk Patients", color="blue")
    ax1.set_xlabel("Lab Value")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Lab Values (Same for All Patients)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Histogram of labs per patient
    positive_labs_per_patient = lab_data[lab_data["is_positive"]].groupby("subject_id").size()
    negative_labs_per_patient = lab_data[~lab_data["is_positive"]].groupby("subject_id").size()

    ax2.hist(positive_labs_per_patient, bins=range(1, max(positive_labs_per_patient.max(), negative_labs_per_patient.max()) + 2), 
             alpha=0.7, label="High-Risk Patients", color="red")
    ax2.hist(negative_labs_per_patient, bins=range(1, max(positive_labs_per_patient.max(), negative_labs_per_patient.max()) + 2), 
             alpha=0.7, label="Low-Risk Patients", color="blue")
    ax2.set_xlabel("Number of Labs per Patient")
    ax2.set_ylabel("Number of Patients")
    ax2.set_title("Distribution of Lab Frequency per Patient")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_path.parent, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Distribution plot saved to {save_path}")



def calculate_theoretical_performance(data: pd.DataFrame, low_risk_mean: float, low_risk_std: float, 
                                     high_risk_mean: float, high_risk_std: float) -> dict:
    """
    Calculate the theoretical performance of the model based on frequency differences.

    Args:
        data: DataFrame containing the synthetic data
        low_risk_mean: Mean LAB1 frequency for low-risk patients
        low_risk_std: Standard deviation for LAB1 frequency for low-risk patients
        high_risk_mean: Mean LAB1 frequency for high-risk patients
        high_risk_std: Standard deviation for LAB1 frequency for high-risk patients
        
    Returns:
        dict: Dictionary containing performance metrics
    """
    # Calculate theoretical frequency-based AUC
    frequency_auc = calculate_frequency_auc(data, low_risk_mean, low_risk_std, high_risk_mean, high_risk_std)
    
    # Calculate other metrics (these may still have some leakage but are less critical)
    sweep_auc = sweep_threshold_auc(data)
    scipy_mann_whitney_u_auc = scipy_mann_whitney_u(data)
    cohens_d_metric = cohens_d(data)
    
    print("\nTheoretical performance:")
    print(f"Theoretical frequency-based AUC (based on distribution separation): {frequency_auc:.4f}")
    print(f"Sweep AUC (LAB1 values only): {sweep_auc:.4f}")
    print(f"Scipy Mann-Whitney U (LAB1 values only): {scipy_mann_whitney_u_auc:.4f}")
    print(f"Cohen's d (LAB1 values only): {cohens_d_metric:.4f}")
    
    return {
        "frequency_auc": frequency_auc,
        "sweep_auc": sweep_auc,
        "scipy_mann_whitney_u_auc": scipy_mann_whitney_u_auc,
        "cohens_d_metric": cohens_d_metric,
    }


def calculate_frequency_auc(data: pd.DataFrame, low_risk_mean: float, low_risk_std: float, 
                          high_risk_mean: float, high_risk_std: float) -> float:
    """
    Calculate theoretical AUC for detecting high-risk patients based on LAB1 frequency distributions.
    
    This function:
    1. Calculates theoretical AUC based on design parameters
    2. Validates the actual generated data against design parameters
    3. Reports both to identify any mismatches
    
    Args:
        data: DataFrame containing the synthetic data
        low_risk_mean: Mean LAB1 frequency for low-risk patients (design parameter)
        low_risk_std: Standard deviation for LAB1 frequency for low-risk patients (design parameter)
        high_risk_mean: Mean LAB1 frequency for high-risk patients (design parameter)
        high_risk_std: Standard deviation for LAB1 frequency for high-risk patients (design parameter)
        
    Returns:
        float: Theoretical AUC for LAB1 frequency-based detection
    """
    from scipy.stats import norm
    
    # Get LAB1 frequency data from the actual generated data
    lab1_data = data[data['code'] == 'S/LAB1']
    
    if len(lab1_data) == 0:
        print("Warning: No LAB1 data found in the dataset")
        return 0.5
    
    # Count LAB1 tests per patient
    lab1_per_patient = lab1_data.groupby("subject_id").size()
    
    if len(lab1_per_patient) < 2:
        print("Warning: Not enough patients with LAB1 data")
        return 0.5
    
    # Calculate observed statistics for validation
    observed_mean = lab1_per_patient.mean()
    observed_std = lab1_per_patient.std()
    
    print(f"Observed LAB1 frequency - Mean: {observed_mean:.2f}, Std: {observed_std:.2f}")
    print(f"Expected - Low-risk: {low_risk_mean:.2f}±{low_risk_std:.2f}, High-risk: {high_risk_mean:.2f}±{high_risk_std:.2f}")
    
    # Calculate theoretical AUC based on the DESIGN parameters
    mean_diff = high_risk_mean - low_risk_mean
    combined_std = np.sqrt(low_risk_std**2 + high_risk_std**2)
    
    if combined_std == 0:
        theoretical_auc = 1.0 if mean_diff > 0 else 0.0
    else:
        z_score = mean_diff / combined_std
        theoretical_auc = norm.cdf(z_score)
    
    print(f"Theoretical AUC (based on design parameters): {theoretical_auc:.4f}")
    
    # Now validate the actual generated data
    # Check if the observed data matches the design parameters
    if abs(observed_mean - (low_risk_mean + high_risk_mean) / 2) > 0.5:
        print(f"WARNING: Observed mean ({observed_mean:.2f}) doesn't match expected mean ({(low_risk_mean + high_risk_mean) / 2:.2f})")
    
    if abs(observed_std - (low_risk_std + high_risk_std) / 2) > 0.5:
        print(f"WARNING: Observed std ({observed_std:.2f}) doesn't match expected std ({(low_risk_std + high_risk_std) / 2:.2f})")
    
    # If design parameters are identical (no separation), the theoretical AUC should be 0.5
    if abs(mean_diff) < 0.01:
        print("INFO: Design parameters are identical - no theoretical separation expected")
        if theoretical_auc != 0.5:
            print(f"WARNING: Theoretical AUC should be 0.5 for identical parameters, but got {theoretical_auc:.4f}")
    
    return theoretical_auc


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data where all patients have the same total number of labs, but high-risk patients get more LAB1 tests (filler with LAB2)"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=DEFAULT_INPUT_FILE,
        help="Path to input synthetic data CSV file",
    )
    parser.add_argument(
        "--low_risk_mean",
        type=float,
        default=LOW_RISK_LABS_MEAN,
        help="Mean number of lab values per low-risk patient",
    )
    parser.add_argument(
        "--low_risk_std",
        type=float,
        default=LOW_RISK_LABS_STD,
        help="Standard deviation for number of lab values per low-risk patient",
    )
    parser.add_argument(
        "--high_risk_mean",
        type=float,
        default=HIGH_RISK_LABS_MEAN,
        help="Mean number of lab values per high-risk patient",
    )
    parser.add_argument(
        "--high_risk_std",
        type=float,
        default=HIGH_RISK_LABS_STD,
        help="Standard deviation for number of lab values per high-risk patient",
    )
    parser.add_argument(
        "--min_labs",
        type=int,
        default=MIN_LABS_PER_PATIENT,
        help="Minimum number of lab values per patient (applies to all patients)",
    )
    parser.add_argument(
        "--write_dir",
        type=str,
        default=DEFAULT_WRITE_DIR,
        help="Directory to write output files",
    )
    parser.add_argument(
        "--patient_info_path",
        type=str,
        default=PATIENTS_INFO_PATH,
        help="Path to patient information parquet file (optional, for realistic birth/death dates)",
    )

    args = parser.parse_args()

    # Read input data
    try:
        input_data = pd.read_csv(args.input_file)
    except FileNotFoundError:
        print(f"Error: Could not find input file at {args.input_file}")
        return

    # Read patient info data if provided
    patient_df = None
    if args.patient_info_path:
        try:
            patient_df = pd.read_parquet(args.patient_info_path)
            print(f"Loaded patient info from {args.patient_info_path}")
        except FileNotFoundError:
            print(f"Warning: Could not find patient info file at {args.patient_info_path}")
            print("Using default timestamp generation")

    print("Initial data:")
    print(input_data.head())

    # Get positive patients and add is_positive column
    input_data = get_positive_patients(input_data, POSITIVE_DIAGS)

    # Print initial statistics
    print("\nInitial data statistics:")
    print(f"Total records: {len(input_data)}")
    print(f"Total patients: {input_data['subject_id'].nunique()}")

    # Count unique positive and negative patients
    positive_patients = input_data[input_data["is_positive"]]["subject_id"].nunique()
    negative_patients = input_data[~input_data["is_positive"]]["subject_id"].nunique()

    print(f"Positive patients: {positive_patients}")
    print(f"Negative patients: {negative_patients}")

    print(f"\nGenerating synthetic data with:")
    print(f"  - {input_data['subject_id'].nunique()} patients")
    print(f"  - All patients get at least {args.min_labs} labs (no maximum constraint)")
    print(f"  - Low-risk patients: ~{args.low_risk_mean:.1f} ± {args.low_risk_std:.1f} LAB1 tests per patient")
    print(f"  - High-risk patients: ~{args.high_risk_mean:.1f} ± {args.high_risk_std:.1f} LAB1 tests per patient")
    print(f"  - Remaining slots filled with LAB2 (filler)")
    print(f"  - All patients use the same lab value distribution (mean={LAB_MEAN}, std={LAB_STD})")
    print(f"  - {positive_patients} high-risk patients (more LAB1)")
    print(f"  - {negative_patients} low-risk patients (fewer LAB1)")

    # Generate synthetic data
    data = generate_synthetic_data(
        input_data,
        args.low_risk_mean,
        args.low_risk_std,
        args.high_risk_mean,
        args.high_risk_std,
        args.min_labs,
        patient_df
    )

    print("\nGenerated data:")
    print(data.head())

    # Print statistics
    print_statistics(data)

    # Write to CSV
    write_dir = Path(args.write_dir)
    write_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(write_dir / f"{SAVE_NAME}.csv", index=False)
    print(f"\nSaved synthetic data to {write_dir / f'{SAVE_NAME}.csv'}")

    # Min-max normalize numeric_value for both S/LAB1 and S/LAB2 separately and save as a separate file
    normalized_data = data.copy()
    
    # Normalize LAB1 separately
    lab1_mask = normalized_data["code"] == "S/LAB1"
    if lab1_mask.any():
        min_val = normalized_data.loc[lab1_mask, "numeric_value"].min()
        max_val = normalized_data.loc[lab1_mask, "numeric_value"].max()
        if max_val > min_val:
            normalized_data.loc[lab1_mask, "numeric_value"] = (
                normalized_data.loc[lab1_mask, "numeric_value"] - min_val
            ) / (max_val - min_val)
        else:
            normalized_data.loc[lab1_mask, "numeric_value"] = 0.0
    
    # Normalize LAB2 separately
    lab2_mask = normalized_data["code"] == "S/LAB2"
    if lab2_mask.any():
        min_val = normalized_data.loc[lab2_mask, "numeric_value"].min()
        max_val = normalized_data.loc[lab2_mask, "numeric_value"].max()
        if max_val > min_val:
            normalized_data.loc[lab2_mask, "numeric_value"] = (
                normalized_data.loc[lab2_mask, "numeric_value"] - min_val
            ) / (max_val - min_val)
        else:
            normalized_data.loc[lab2_mask, "numeric_value"] = 0.0
    
    normalized_filename = write_dir / f"{SAVE_NAME}_minmaxnorm.csv"
    normalized_data.to_csv(normalized_filename, index=False)

    # Calculate theoretical performance
    performance_metrics = calculate_theoretical_performance(
        data, args.low_risk_mean, args.low_risk_std, args.high_risk_mean, args.high_risk_std
    )

    # Create plots
    plot_dir = Path(DEFAULT_PLOT_DIR)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    create_distribution_plot(data, plot_dir / f"{SAVE_NAME}_distribution.png")


if __name__ == "__main__":
    main()
