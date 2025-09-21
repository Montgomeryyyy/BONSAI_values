"""
Generate synthetic data with multiple lab values where positive patients are determined 
by a multiplication equation: LAB1 * LAB2 * ... * LABN > threshold.
Based on the multi_lab_addition.py structure.
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
PATIENTS_INFO_PATH = "../../../data/vals/patient_infos/patient_info_10000n.parquet"

# Lab value distributions - same distribution for all labs by default
LAB_MEAN = 0.3  # Mean for all labs
LAB_STD = 0.1   # Std for all labs

# Alternative: Different means for different labs (creates more realistic separation)
LAB_MEANS = [0.2, 0.3, 0.4]  # Different means for LAB1, LAB2, LAB3
USE_DIFFERENT_MEANS = True  # Set to True to use different means

# Number of labs per patient
NUM_LABS = 3  # Default to 3 labs, can be changed via command line
MULTIPLICATION_THRESHOLD = 0.024  # 0.2 × 0.3 × 0.4 (product of different lab means)
NOISE_LEVEL = 0.1  # 10% noise for realistic AUC

# Diagnosis timing parameters
DIAG_MIN_DAYS = 10  # Minimum days after last lab for diagnosis
DIAG_MAX_DAYS = 180  # Maximum days after last lab for diagnosis

DEFAULT_WRITE_DIR = f"../../../data/vals/synthetic_data/{N}n/"
DEFAULT_PLOT_DIR = f"../../../data/vals/synthetic_data_plots/{N}n/"
POSITIVE_DIAGS = ["S/DIAG_POSITIVE"]

# Define lab value distributions - will be generated dynamically based on num_labs

# Concept relationships are now handled directly in the generation logic


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


def generate_lab_value(lab_name: str, lab_value_info: dict, noise_level: float = 0.0) -> Optional[float]:
    """
    Generate a lab value based on the lab name and distribution info.

    Args:
        lab_name: Name of the lab test
        lab_value_info: Dictionary containing lab distribution information
        noise_level: Amount of noise to add (0.0 = no noise, 0.1 = 10% noise)

    Returns:
        Optional[float]: Generated lab value or None if invalid input
    """
    if lab_name not in lab_value_info:
        return None

    range_info = lab_value_info[lab_name]["distribution"]
    if range_info["dist"] == "uniform":
        base_value = np.random.choice(range_info["range"])
    elif range_info["dist"] == "normal":
        base_value = np.random.normal(range_info["mean"], range_info["std"])
    else:
        return None
    
    # Add noise if specified
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * abs(base_value))
        base_value += noise
    
    return base_value


def generate_n_lab_concepts(pids_list: List[str], threshold: float, num_labs: int, lab_value_info: dict, noise_level: float = 0.0) -> pd.DataFrame:
    """
    Generate exactly N lab concepts (LAB1, LAB2, ..., LABN) for each patient.
    Patient risk is determined by product of CLEAN lab values > threshold.
    Observed lab values have noise added.

    Args:
        pids_list: List of patient IDs
        threshold: Threshold for product of all labs > threshold equation
        num_labs: Number of labs per patient
        lab_value_info: Dictionary containing lab distribution information
        noise_level: Amount of noise to add to observed values

    Returns:
        pd.DataFrame: DataFrame containing PID, CONCEPT, and RESULT columns
    """
    records = []
    patient_risk_map = {}

    for pid in pids_list:
        # Generate exactly N CLEAN lab values for each patient (no noise)
        clean_lab_values = []
        lab_concepts = []
        
        for i in range(1, num_labs + 1):
            lab_concept = f"S/LAB{i}"
            clean_lab_value = generate_lab_value(lab_concept, lab_value_info, noise_level=0.0)  # No noise for true risk
            if clean_lab_value is not None:
                clean_lab_values.append(clean_lab_value)
                lab_concepts.append(lab_concept)
        
        if len(clean_lab_values) == num_labs:  # Ensure we have all labs
            # Determine patient risk based on CLEAN lab values (true underlying risk)
            clean_lab_product = np.prod(clean_lab_values)
            is_high_risk = clean_lab_product > threshold
                            
            patient_risk_map[pid] = is_high_risk
            condition = "high_risk" if is_high_risk else "low_risk"

            # Generate NOISY lab values for the observed data
            noisy_lab_values = []
            for clean_value in clean_lab_values:
                if noise_level > 0:
                    noise = np.random.normal(0, noise_level * abs(clean_value))
                    noisy_value = clean_value + noise
                else:
                    noisy_value = clean_value
                noisy_lab_values.append(noisy_value)

            # Add all lab records using NOISY values (what the model will see)
            for i, (lab_concept, noisy_lab_value) in enumerate(zip(lab_concepts, noisy_lab_values)):
                records.append({
                    "PID": pid, 
                    "CONCEPT": lab_concept, 
                    "RESULT": noisy_lab_value,
                    "LAB_INDEX": i,
                    "CONDITION": condition
                })

            # Add diagnosis based on risk status - every patient gets a diagnosis
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
    diag_min_days: int = DIAG_MIN_DAYS, diag_max_days: int = DIAG_MAX_DAYS, patient_df: pd.DataFrame = None
) -> List[pd.Timestamp]:
    """
    Generate timestamps for a list of patient IDs based on time relationships.
    Similar to multi_lab_addition.py but adapted for multiple labs per patient.

    Args:
        pids_list: List of patient IDs to generate timestamps for
        concepts: List of concepts corresponding to each PID
        lab_indices: List of lab indices corresponding to each record
        diag_min_days: Minimum days after last lab for diagnosis
        diag_max_days: Maximum days after last lab for diagnosis
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

        # Handle timestamps based on concept type
        if concept in ["S/DIAG_POSITIVE", "S/DIAG_NEGATIVE"]:
            # Diagnosis concepts come after the last lab (configurable days after last lab)
            # Find the latest lab timestamp for this patient
            lab_timestamps = [ts for lab_concept, ts in concept_timestamps[pid].items() 
                            if lab_concept.startswith("S/LAB") and lab_concept != "start_time"]
            if lab_timestamps:
                # Use the latest lab timestamp as base
                base_timestamp = max(lab_timestamps)
                days_after = np.random.randint(diag_min_days, diag_max_days + 1)
                timestamp = base_timestamp + pd.Timedelta(days=days_after)
            else:
                # If no lab exists yet, generate a random timestamp
                start_time = concept_timestamps[pid]["start_time"]
                timestamp = start_time + pd.Timedelta(days=np.random.randint(0, 365))
        else:
            # For lab concepts, generate timestamp based on lab index
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
    threshold: float,
    num_labs: int,
    lab_value_info: dict,
    diag_min_days: int = DIAG_MIN_DAYS,
    diag_max_days: int = DIAG_MAX_DAYS,
    noise_level: float = NOISE_LEVEL,
    patient_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Generate synthetic data with exactly N lab values per patient (LAB1, LAB2, ..., LABN).
    Patient risk is determined by product of all labs > threshold.
    
    Args:
        input_data: DataFrame containing existing synthetic data with patient assignments
        threshold: Threshold for product of all labs > threshold equation
        num_labs: Number of labs per patient
        lab_value_info: Dictionary containing lab distribution information
        diag_min_days: Minimum days after last lab for diagnosis
        diag_max_days: Maximum days after last lab for diagnosis
        
    Returns:
        pd.DataFrame: Generated synthetic data
    """
    # Get patient IDs from input data
    pids_list = list(input_data["subject_id"].unique())
    
    # Generate concepts and lab values
    concepts_data = generate_n_lab_concepts(pids_list, threshold, num_labs, lab_value_info, noise_level)

    # Create final DataFrame - match multi_lab_addition.py structure exactly
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
        diag_min_days,
        diag_max_days,
        patient_df
    )

    return data


def print_statistics(data: pd.DataFrame, num_labs: int) -> None:
    """
    Print statistics about the lab values.

    Args:
        data: DataFrame containing the synthetic data
        num_labs: Number of labs per patient
    """
    # Recreate is_positive column for analysis
    positive_patients = set(data[data["code"] == "S/DIAG_POSITIVE"]["subject_id"].unique())
    data["is_positive"] = data["subject_id"].isin(positive_patients)
    positive_mask = data["is_positive"]

    # Get all lab masks
    lab_masks = {}
    for i in range(1, num_labs + 1):
        lab_masks[f"LAB{i}"] = data["code"] == f"S/LAB{i}"

    print("\nLab value statistics:")
    for i in range(1, num_labs + 1):
        lab_mask = lab_masks[f"LAB{i}"]
        print(f"LAB{i} - Count: {len(data[lab_mask])}")
        print(f"LAB{i} - Mean: {data[lab_mask]['numeric_value'].mean():.3f}")
        print(f"LAB{i} - Std: {data[lab_mask]['numeric_value'].std():.3f}")
        print(f"LAB{i} - Min: {data[lab_mask]['numeric_value'].min():.3f}")
        print(f"LAB{i} - Max: {data[lab_mask]['numeric_value'].max():.3f}")

    # Count labs per patient (should be exactly num_labs for all patients)
    all_lab_mask = data["code"].str.startswith("S/LAB")
    positive_labs_per_patient = data[all_lab_mask & positive_mask].groupby("subject_id").size()
    negative_labs_per_patient = data[all_lab_mask & ~positive_mask].groupby("subject_id").size()

    print(f"\nLab frequency statistics (each patient has exactly {num_labs} labs):")
    print(f"High-risk patients - Labs per patient: {positive_labs_per_patient.mean():.1f} (should be {num_labs}.0)")
    print(f"Low-risk patients - Labs per patient: {negative_labs_per_patient.mean():.1f} (should be {num_labs}.0)")
    
    # Calculate multiplication statistics (product of all labs)
    lab_data_dict = {}
    for i in range(1, num_labs + 1):
        lab_data_dict[f"LAB{i}"] = data[data["code"] == f"S/LAB{i}"].groupby("subject_id")["numeric_value"].first()
    
    # Calculate product of all labs for each patient
    # Convert to pandas Series to maintain index information
    multiplication_scores = pd.Series(np.prod(list(lab_data_dict.values()), axis=0), 
                                    index=list(lab_data_dict.values())[0].index)
    
    positive_multiplication = multiplication_scores[multiplication_scores.index.isin(positive_patients)]
    negative_multiplication = multiplication_scores[~multiplication_scores.index.isin(positive_patients)]
    
    print(f"\nMultiplication equation statistics (product of all {num_labs} labs):")
    print(f"High-risk patients - Mean: {positive_multiplication.mean():.6f}")
    print(f"High-risk patients - Std: {positive_multiplication.std():.6f}")
    print(f"Low-risk patients - Mean: {negative_multiplication.mean():.6f}")
    print(f"Low-risk patients - Std: {negative_multiplication.std():.6f}")
    print(f"Threshold: {MULTIPLICATION_THRESHOLD}")
    
    # Verify that every patient has a diagnosis
    total_patients = data["subject_id"].nunique()
    patients_with_positive_diag = data[data["code"] == "S/DIAG_POSITIVE"]["subject_id"].nunique()
    patients_with_negative_diag = data[data["code"] == "S/DIAG_NEGATIVE"]["subject_id"].nunique()
    
    print(f"\nDiagnosis coverage:")
    print(f"Total patients: {total_patients}")
    print(f"Patients with S/DIAG_POSITIVE: {patients_with_positive_diag}")
    print(f"Patients with S/DIAG_NEGATIVE: {patients_with_negative_diag}")
    print(f"Total with diagnosis: {patients_with_positive_diag + patients_with_negative_diag}")
    print(f"Coverage: {(patients_with_positive_diag + patients_with_negative_diag) / total_patients * 100:.1f}%")


def create_distribution_plot(
    data: pd.DataFrame, save_path: Path, num_labs: int
) -> None:
    """
    Create a figure showing the distribution of lab values and multiplication scores.

    Args:
        data: DataFrame containing the synthetic data
        save_path: Path to save the plot
        num_labs: Number of labs per patient
    """
    # Recreate is_positive column for analysis
    positive_patients = set(data[data["code"] == "S/DIAG_POSITIVE"]["subject_id"].unique())
    
    # Create subplots - show first 3 labs and multiplication scores
    n_cols = min(4, num_labs + 1)  # Show up to 3 labs + multiplication scores
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    # Plot individual lab distributions (up to 3 labs)
    for i in range(min(3, num_labs)):
        lab_mask = data["code"] == f"S/LAB{i+1}"
        lab_data = data[lab_mask].copy()
        lab_data["is_positive"] = lab_data["subject_id"].isin(positive_patients)
        
        positive_values = lab_data[lab_data["is_positive"]]["numeric_value"]
        negative_values = lab_data[~lab_data["is_positive"]]["numeric_value"]

        axes[i].hist(positive_values, bins=30, alpha=0.7, label="High-Risk Patients", color="red")
        axes[i].hist(negative_values, bins=30, alpha=0.7, label="Low-Risk Patients", color="blue")
        axes[i].set_xlabel(f"LAB{i+1} Value")
        axes[i].set_ylabel("Count")
        axes[i].set_title(f"Distribution of LAB{i+1} Values")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    # Plot multiplication scores (product of all labs)
    lab_data_dict = {}
    for i in range(1, num_labs + 1):
        lab_data_dict[f"LAB{i}"] = data[data["code"] == f"S/LAB{i}"].groupby("subject_id")["numeric_value"].first()
    
    # Calculate product of all labs for each patient
    # Convert to pandas Series to maintain index information
    multiplication_scores = pd.Series(np.prod(list(lab_data_dict.values()), axis=0), 
                                    index=list(lab_data_dict.values())[0].index)
    
    positive_multiplication = multiplication_scores[multiplication_scores.index.isin(positive_patients)]
    negative_multiplication = multiplication_scores[~multiplication_scores.index.isin(positive_patients)]

    ax_idx = min(3, num_labs)  # Index for multiplication scores plot
    axes[ax_idx].hist(positive_multiplication, bins=30, alpha=0.7, label="High-Risk Patients", color="red")
    axes[ax_idx].hist(negative_multiplication, bins=30, alpha=0.7, label="Low-Risk Patients", color="blue")
    axes[ax_idx].axvline(MULTIPLICATION_THRESHOLD, color='black', linestyle='--', linewidth=2, label=f'Threshold: {MULTIPLICATION_THRESHOLD}')
    axes[ax_idx].set_xlabel(f"Product of All {num_labs} Labs")
    axes[ax_idx].set_ylabel("Number of Patients")
    axes[ax_idx].set_title(f"Distribution of Multiplication Scores (Product of {num_labs} Labs)")
    axes[ax_idx].legend()
    axes[ax_idx].grid(True, alpha=0.3)
    axes[ax_idx].set_xscale('log')  # Use log scale for better visualization of small products

    plt.tight_layout()
    os.makedirs(save_path.parent, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Distribution plot saved to {save_path}")


def calculate_theoretical_performance(data: pd.DataFrame, num_labs: int) -> dict:
    """
    Calculate the theoretical performance of the model based on multiplication equation.

    Args:
        data: DataFrame containing the synthetic data
        num_labs: Number of labs per patient
        
    Returns:
        dict: Dictionary containing performance metrics
    """
    # Calculate multiplication-based AUC
    multiplication_auc = calculate_multiplication_auc(data, num_labs)
    
    # Calculate other metrics
    sweep_auc = sweep_threshold_auc(data)
    scipy_mann_whitney_u_auc = scipy_mann_whitney_u(data)
    cohens_d_metric = cohens_d(data)
    
    print("\nTheoretical performance:")
    print(f"Multiplication-based AUC (product of {num_labs} labs): {multiplication_auc}")
    print(f"Sweep AUC: {sweep_auc}")
    print(f"Scipy Mann-Whitney U: {scipy_mann_whitney_u_auc}")
    print(f"Cohen's d: {cohens_d_metric}")
    
    return {
        "multiplication_auc": multiplication_auc,
        "sweep_auc": sweep_auc,
        "scipy_mann_whitney_u_auc": scipy_mann_whitney_u_auc,
        "cohens_d_metric": cohens_d_metric,
    }


def calculate_multiplication_auc(data: pd.DataFrame, num_labs: int) -> float:
    """
    Calculate AUC for detecting high-risk patients based on product of all labs > threshold.
    
    Args:
        data: DataFrame containing the synthetic data
        num_labs: Number of labs per patient
        
    Returns:
        float: AUC for multiplication-based detection
    """
    # Get lab values for each patient
    lab_data_dict = {}
    for i in range(1, num_labs + 1):
        lab_data_dict[f"LAB{i}"] = data[data['code'] == f'S/LAB{i}'].groupby("subject_id")["numeric_value"].first()
    
    # Get patient risks
    positive_patients = set(data[data["code"] == "S/DIAG_POSITIVE"]["subject_id"].unique())
    patient_risks = list(lab_data_dict.values())[0].index.isin(positive_patients)
    
    # Calculate product of all labs for each patient
    # Convert to pandas Series to maintain index information
    multiplication_scores = pd.Series(np.prod(list(lab_data_dict.values()), axis=0), 
                                    index=list(lab_data_dict.values())[0].index)
    
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(patient_risks, multiplication_scores)
    return auc


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data with exactly N lab values per patient where positive patients are determined by product of all labs > threshold"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=DEFAULT_INPUT_FILE,
        help="Path to input synthetic data CSV file",
    )
    parser.add_argument(
        "--num_labs",
        type=int,
        default=NUM_LABS,
        help="Number of labs per patient",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=MULTIPLICATION_THRESHOLD,
        help="Threshold for product of all labs > threshold equation",
    )
    parser.add_argument(
        "--diag_min_days",
        type=int,
        default=DIAG_MIN_DAYS,
        help="Minimum days after last lab for diagnosis",
    )
    parser.add_argument(
        "--diag_max_days",
        type=int,
        default=DIAG_MAX_DAYS,
        help="Maximum days after last lab for diagnosis",
    )
    parser.add_argument(
        "--write_dir",
        type=str,
        default=DEFAULT_WRITE_DIR,
        help="Directory to write output files",
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=NOISE_LEVEL,
        help="Noise level to add to lab values (0.0 = no noise, 0.1 = 10% noise)",
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

    # Print initial statistics
    print("\nInitial data statistics:")
    print(f"Total records: {len(input_data)}")
    print(f"Total patients: {input_data['subject_id'].nunique()}")

    print(f"\nGenerating synthetic data with:")
    print(f"  - {input_data['subject_id'].nunique()} patients")
    print(f"  - Each patient gets exactly {args.num_labs} lab values (LAB1, LAB2, ..., LAB{args.num_labs})")
    print(f"  - All labs use same distribution: mean={LAB_MEAN}, std={LAB_STD}")
    print(f"  - Multiplication threshold: product of all {args.num_labs} labs > {args.threshold}")
    print(f"  - True risk determined by CLEAN lab values (no noise)")
    print(f"  - Observed lab values have {args.noise_level*100:.0f}% noise added")
    print(f"  - Diagnosis timing: {args.diag_min_days}-{args.diag_max_days} days after last lab")

    # Generate save name dynamically
    noise_suffix = f"_noise{int(args.noise_level*100)}" if args.noise_level > 0 else ""
    save_name = f"n_lab_multiplication_{args.num_labs}labs_mean{int(LAB_MEAN*100)}p{int(LAB_STD*100)}{noise_suffix}_n{N}"

    # Generate lab value info dynamically
    lab_value_info = {}
    for i in range(1, args.num_labs + 1):
        # Use different means if specified, otherwise use same mean for all labs
        if USE_DIFFERENT_MEANS and i <= len(LAB_MEANS):
            lab_mean = LAB_MEANS[i-1]
        else:
            lab_mean = LAB_MEAN
            
        lab_value_info[f"S/LAB{i}"] = {
            "distribution": {
                "dist": "normal",
                "mean": lab_mean,
                "std": LAB_STD,
            },
        }

    # Generate synthetic data
    data = generate_synthetic_data(
        input_data,
        args.threshold,
        args.num_labs,
        lab_value_info,
        args.diag_min_days,
        args.diag_max_days,
        args.noise_level,
        patient_df
    )

    print("\nGenerated data:")
    print(data.head())

    # Print statistics
    print_statistics(data, args.num_labs)

    # Write to CSV
    write_dir = Path(args.write_dir)
    write_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(write_dir / f"{save_name}.csv", index=False)
    print(f"\nSaved synthetic data to {write_dir / f'{save_name}.csv'}")

    # Min-max normalize numeric_value for labs separately for each lab and save as a separate file
    normalized_data = data.copy()
    
    # Normalize each lab separately
    for i in range(1, args.num_labs + 1):
        lab_code = f"S/LAB{i}"
        lab_mask = normalized_data["code"] == lab_code
        
        if lab_mask.any():
            lab_values = normalized_data.loc[lab_mask, "numeric_value"]
            min_val = lab_values.min()
            max_val = lab_values.max()
            
            if max_val > min_val:
                # Apply min-max normalization: (value - min) / (max - min)
                normalized_data.loc[lab_mask, "numeric_value"] = (
                    lab_values - min_val
                ) / (max_val - min_val)
            else:
                # If all values are the same, set to 0.0
                normalized_data.loc[lab_mask, "numeric_value"] = 0.0
    
    normalized_filename = write_dir / f"{save_name}_minmaxnorm.csv"
    normalized_data.to_csv(normalized_filename, index=False)
    print(f"Saved min-max normalized data to {normalized_filename}")

    # Calculate theoretical performance
    performance_metrics = calculate_theoretical_performance(data, args.num_labs)

    # Create plots
    plot_dir = Path(DEFAULT_PLOT_DIR)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    create_distribution_plot(data, plot_dir / f"{save_name}_distribution.png", args.num_labs)


if __name__ == "__main__":
    main()
