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
# Gaussian parameters for number of labs per patient
LOW_RISK_LABS_MEAN = 5.0
LOW_RISK_LABS_STD = 2
HIGH_RISK_LABS_MEAN = 10.0
HIGH_RISK_LABS_STD = 2
MIN_LABS_PER_PATIENT = 2
MAX_LABS_PER_PATIENT = 13
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
}

# Define concept relationships similar to multi_lab_sharp_edge.py
CONCEPT_RELATIONSHIPS = {
    "S/LAB1": {
        "base_probability": 1.0,  # 100% of patients get labs
        "condition_probabilities": {
            "high_risk": 0.5,  # 50% chance of being high-risk (more labs)
            "low_risk": 0.5,  # 50% chance of being low-risk (fewer labs)
        },
        "add_base_concept": ["high_risk", "low_risk"],  # Add lab for all conditions
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
                               high_risk_mean: float, high_risk_std: float, min_labs: int, max_labs: int, patient_risk_map: dict) -> pd.DataFrame:
    """
    Generate multiple lab concepts and values for a list of patient IDs.
    High-risk patients get more labs on average using Gaussian distributions.

    Args:
        pids_list: List of patient IDs
        low_risk_mean: Mean number of labs for low-risk patients
        low_risk_std: Standard deviation for low-risk patients
        high_risk_mean: Mean number of labs for high-risk patients
        high_risk_std: Standard deviation for high-risk patients
        min_labs: Minimum number of labs per patient (applies to all patients)
        max_labs: Maximum number of labs per patient (applies to all patients)
        patient_risk_map: Dictionary mapping patient_id to risk status (True=high_risk, False=low_risk)

    Returns:
        pd.DataFrame: DataFrame containing PID, CONCEPT, and RESULT columns
    """
    records = []

    for pid in pids_list:
        # For each base concept in CONCEPT_RELATIONSHIPS
        for base_concept, info in CONCEPT_RELATIONSHIPS.items():
            # Determine if this patient gets this base concept
            if np.random.random() < info["base_probability"]:
                # Use existing patient risk assignment
                is_positive = patient_risk_map.get(pid, False)
                condition = "high_risk" if is_positive else "low_risk"

                # Add multiple lab values for this patient
                if "add_base_concept" in info and condition in info["add_base_concept"]:
                    if base_concept in LAB_VALUE_INFO:
                        # Generate different number of labs based on risk status using Gaussian distributions
                        if condition == "high_risk":
                            n_labs = int(np.random.normal(high_risk_mean, high_risk_std))
                        else:
                            n_labs = int(np.random.normal(low_risk_mean, low_risk_std))
                        
                        # Apply min/max constraints
                        n_labs = max(min_labs, min(max_labs, n_labs))
                        
                        # All patients use the same distribution
                        for i in range(n_labs):
                            value = generate_lab_value(base_concept)
                            if value is not None:
                                records.append({
                                    "PID": pid, 
                                    "CONCEPT": base_concept, 
                                    "RESULT": value,
                                    "LAB_INDEX": i,
                                    "CONDITION": condition
                                })

                # Add related concepts based on their probabilities
                for related_concept, related_info in info["related_concepts"].items():
                    # Check if we should generate this related concept based on condition
                    should_generate = False
                    if "conditions" in related_info:
                        # Only generate if the current condition is in the allowed conditions
                        should_generate = condition in related_info["conditions"]
                    else:
                        # If no conditions specified, use probability
                        should_generate = np.random.random() < related_info["prob"]

                    if should_generate:
                        # This is a diagnosis concept, add without value
                        records.append({
                            "PID": pid, 
                            "CONCEPT": related_concept, 
                            "RESULT": 1.0,
                            "LAB_INDEX": -1,
                            "CONDITION": condition
                        })

    return pd.DataFrame(records)


def generate_timestamps(
    pids_list: List[str], concepts: List[str], lab_indices: List[int]
) -> List[pd.Timestamp]:
    """
    Generate timestamps for a list of patient IDs based on time relationships.
    Similar to multi_lab_sharp_edge.py but adapted for multiple labs per patient.

    Args:
        pids_list: List of patient IDs to generate timestamps for
        concepts: List of concepts corresponding to each PID
        lab_indices: List of lab indices corresponding to each record

    Returns:
        List[pd.Timestamp]: List of generated timestamps
    """
    timestamps = []
    concept_timestamps = {}  # Store timestamps for each concept per patient

    for i, (pid, concept, lab_index) in enumerate(zip(pids_list, concepts, lab_indices)):
        # Initialize patient's concept timestamps if not exists
        if pid not in concept_timestamps:
            concept_timestamps[pid] = {}
            # Generate a random start time for this patient (within the last 2 years)
            # Use seconds precision to match multi_lab_sharp_edge.py format
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
    max_labs: int
) -> pd.DataFrame:
    """
    Generate synthetic data with multiple lab values per patient.
    High-risk patients get more labs on average using Gaussian distributions.
    
    Args:
        input_data: DataFrame containing existing synthetic data with patient assignments
        low_risk_mean: Mean number of labs for low-risk patients
        low_risk_std: Standard deviation for low-risk patients
        high_risk_mean: Mean number of labs for high-risk patients
        high_risk_std: Standard deviation for high-risk patients
        min_labs: Minimum number of labs per patient (applies to all patients)
        max_labs: Maximum number of labs per patient (applies to all patients)
        
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
        high_risk_mean, high_risk_std, min_labs, max_labs, patient_risk_map
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
        concepts_data["LAB_INDEX"].tolist()
    )

    return data


def print_statistics(data: pd.DataFrame) -> None:
    """
    Print statistics about the lab values and frequency.

    Args:
        data: DataFrame containing the synthetic data
    """
    # Get lab values for positive and negative patients
    lab_mask = data["code"] == "S/LAB1"
    
    # Recreate is_positive column for analysis
    positive_patients = set(data[data["code"] == "S/DIAG_POSITIVE"]["subject_id"].unique())
    data["is_positive"] = data["subject_id"].isin(positive_patients)
    positive_mask = data["is_positive"]

    positive_lab_values = data[lab_mask & positive_mask]["numeric_value"]
    negative_lab_values = data[lab_mask & ~positive_mask]["numeric_value"]

    # Count labs per patient
    positive_labs_per_patient = data[lab_mask & positive_mask].groupby("subject_id").size()
    negative_labs_per_patient = data[lab_mask & ~positive_mask].groupby("subject_id").size()

    print("\nLab value statistics (all patients use same distribution):")
    print(f"Overall lab values - Count: {len(data[lab_mask])}")
    print(f"Overall lab values - Mean: {data[lab_mask]['numeric_value'].mean():.3f}")
    print(f"Overall lab values - Std: {data[lab_mask]['numeric_value'].std():.3f}")
    print(f"Overall lab values - Min: {data[lab_mask]['numeric_value'].min():.3f}")
    print(f"Overall lab values - Max: {data[lab_mask]['numeric_value'].max():.3f}")

    print(f"\nLab frequency statistics:")
    print(f"High-risk patients - Avg labs per patient: {positive_labs_per_patient.mean():.1f}")
    print(f"High-risk patients - Min labs per patient: {positive_labs_per_patient.min()}")
    print(f"High-risk patients - Max labs per patient: {positive_labs_per_patient.max()}")
    print(f"Low-risk patients - Avg labs per patient: {negative_labs_per_patient.mean():.1f}")
    print(f"Low-risk patients - Min labs per patient: {negative_labs_per_patient.min()}")
    print(f"Low-risk patients - Max labs per patient: {negative_labs_per_patient.max()}")


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



def calculate_theoretical_performance(data: pd.DataFrame) -> dict:
    """
    Calculate the theoretical performance of the model based on frequency differences.

    Args:
        data: DataFrame containing the synthetic data
        
    Returns:
        dict: Dictionary containing performance metrics
    """
    # Calculate frequency-based AUC
    frequency_auc = calculate_frequency_auc(data)
    
    # Calculate other metrics
    sweep_auc = sweep_threshold_auc(data)
    scipy_mann_whitney_u_auc = scipy_mann_whitney_u(data)
    cohens_d_metric = cohens_d(data)
    
    print("\nTheoretical performance:")
    print(f"Frequency-based AUC: {frequency_auc}")
    print(f"Sweep AUC: {sweep_auc}")
    print(f"Scipy Mann-Whitney U: {scipy_mann_whitney_u_auc}")
    print(f"Cohen's d: {cohens_d_metric}")
    
    return {
        "frequency_auc": frequency_auc,
        "sweep_auc": sweep_auc,
        "scipy_mann_whitney_u_auc": scipy_mann_whitney_u_auc,
        "cohens_d_metric": cohens_d_metric,
    }


def calculate_frequency_auc(data: pd.DataFrame) -> float:
    """
    Calculate AUC for detecting high-risk patients based on lab frequency.
    
    Args:
        data: DataFrame containing the synthetic data
        
    Returns:
        float: AUC for frequency-based detection
    """
    lab_data = data[data['code'] == 'S/LAB1']
    
    # Recreate is_positive column for analysis
    positive_patients = set(data[data["code"] == "S/DIAG_POSITIVE"]["subject_id"].unique())
    lab_data["is_positive"] = lab_data["subject_id"].isin(positive_patients)
    
    # Count labs per patient
    labs_per_patient = lab_data.groupby("subject_id").size()
    patient_risks = lab_data.groupby("subject_id")["is_positive"].first()
    
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(patient_risks, labs_per_patient)
    return auc


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data with multiple lab values where all patients have the same distribution but positive patients have more labs on average"
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
        "--max_labs",
        type=int,
        default=MAX_LABS_PER_PATIENT,
        help="Maximum number of lab values per patient (applies to all patients)",
    )
    parser.add_argument(
        "--write_dir",
        type=str,
        default=DEFAULT_WRITE_DIR,
        help="Directory to write output files",
    )

    args = parser.parse_args()

    # Read input data
    try:
        input_data = pd.read_csv(args.input_file)
    except FileNotFoundError:
        print(f"Error: Could not find input file at {args.input_file}")
        return

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
    print(f"  - Low-risk patients: ~{args.low_risk_mean:.1f} ± {args.low_risk_std:.1f} lab values per patient")
    print(f"  - High-risk patients: ~{args.high_risk_mean:.1f} ± {args.high_risk_std:.1f} lab values per patient")
    print(f"  - All patients constrained to {args.min_labs}-{args.max_labs} lab values")
    print(f"  - All patients use the same lab value distribution (mean={LAB_MEAN}, std={LAB_STD})")
    print(f"  - {positive_patients} high-risk patients (more labs)")
    print(f"  - {negative_patients} low-risk patients (fewer labs)")

    # Generate synthetic data
    data = generate_synthetic_data(
        input_data,
        args.low_risk_mean,
        args.low_risk_std,
        args.high_risk_mean,
        args.high_risk_std,
        args.min_labs,
        args.max_labs
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

    # Min-max normalize numeric_value for S/LAB1 and save as a separate file
    normalized_data = data.copy()
    lab_mask = normalized_data["code"] == "S/LAB1"
    if lab_mask.any():
        min_val = normalized_data.loc[lab_mask, "numeric_value"].min()
        max_val = normalized_data.loc[lab_mask, "numeric_value"].max()
        if max_val > min_val:
            normalized_data.loc[lab_mask, "numeric_value"] = (
                normalized_data.loc[lab_mask, "numeric_value"] - min_val
            ) / (max_val - min_val)
        else:
            normalized_data.loc[lab_mask, "numeric_value"] = 0.0
    
    normalized_filename = write_dir / f"{SAVE_NAME}_minmaxnorm.csv"
    normalized_data.to_csv(normalized_filename, index=False)

    # Calculate theoretical performance
    performance_metrics = calculate_theoretical_performance(data)

    # Create plots
    plot_dir = Path(DEFAULT_PLOT_DIR)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    create_distribution_plot(data, plot_dir / f"{SAVE_NAME}_distribution.png")


if __name__ == "__main__":
    main()
