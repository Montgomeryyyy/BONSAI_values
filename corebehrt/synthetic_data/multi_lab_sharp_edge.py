"""
Generate synthetic data with multiple lab values where high-risk patients switch between 
distributions, while low-risk patients only have labs from one distribution.
Based on the simulate_synthetic_labs.py structure with concept relationships.
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
from sklearn.metrics import roc_auc_score

# Default parameters
N = 100000
DEFAULT_INPUT_FILE = f"../../../data/vals/synthetic_data/{N}n/bn_labs_n{N}_50p_1unq.csv"
PATIENTS_INFO_PATH = f"../../../data/vals/patient_infos/patient_info_{N}n.parquet"

MIN_LABS_PER_PATIENT = 3
MAX_LABS_PER_PATIENT = 10
SWITCHING_PROBABILITY = 1.0  # 100% probability of switching for high-risk patients
LOW_MEAN = 0.45
HIGH_MEAN = 0.55
STD = 0.10

# Diagnosis timing parameters
DIAG_MIN_DAYS = 10  # Minimum days after last lab for diagnosis
DIAG_MAX_DAYS = 180  # Maximum days after last lab for diagnosis

DEFAULT_WRITE_DIR = f"../../../data/vals/synthetic_data/{N}n/"
DEFAULT_PLOT_DIR = f"../../../data/vals/synthetic_data_plots/{N}n/"
POSITIVE_DIAGS = ["S/DIAG_POSITIVE"]

# Define lab value distributions
LAB_VALUE_INFO = {
    "S/LAB1": {
        "high_distribution": {
            "dist": "normal",
            "mean": HIGH_MEAN,
            "std": STD,
        },
        "low_distribution": {
            "dist": "normal",
            "mean": LOW_MEAN,
            "std": STD,
        },
    },
}

# Define concept relationships similar to simulate_synthetic_labs.py
CONCEPT_RELATIONSHIPS = {
    "S/LAB1": {
        "base_probability": 1.0,  # 100% of patients get labs
        "condition_probabilities": {
            "high_risk": 0.5,  # 50% chance of being high-risk (switching)
            "low_risk": 0.5,  # 50% chance of being low-risk (consistent)
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


def generate_lab_value(lab_name: str, condition: str) -> Optional[float]:
    """
    Generate a lab value based on the lab name and condition.

    Args:
        lab_name: Name of the lab test
        condition: The condition affecting the lab values

    Returns:
        Optional[float]: Generated lab value or None if invalid input
    """
    if lab_name not in LAB_VALUE_INFO or condition not in LAB_VALUE_INFO[lab_name]:
        return None

    range_info = LAB_VALUE_INFO[lab_name][condition]
    if range_info["dist"] == "uniform":
        return np.random.choice(range_info["range"])
    elif range_info["dist"] == "normal":
        return np.random.normal(range_info["mean"], range_info["std"])
    return None


def generate_multi_lab_concepts(pids_list: List[str], min_labs: int, max_labs: int, switching_prob: float, patient_risk_map: dict) -> pd.DataFrame:
    """
    Generate multiple lab concepts and values for a list of patient IDs.
    Based on the simulate_synthetic_labs.py structure.

    Args:
        pids_list: List of patient IDs
        min_labs: Minimum number of labs per patient
        max_labs: Maximum number of labs per patient
        switching_prob: Probability of switching for high-risk patients
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
                # Use existing patient risk assignment instead of random assignment
                is_positive = patient_risk_map.get(pid, False)
                condition = "high_risk" if is_positive else "low_risk"

                # Add multiple lab values for this patient
                if "add_base_concept" in info and condition in info["add_base_concept"]:
                    if base_concept in LAB_VALUE_INFO:
                        # Generate multiple lab values
                        n_labs = np.random.randint(min_labs, max_labs + 1)
                        
                        if condition == "high_risk":
                            # High-risk patients: switch distributions once
                            # Randomly choose which distribution to start with
                            current_distribution = np.random.choice(["high_distribution", "low_distribution"])
                            
                            # Randomly choose the switch point (after first lab, before last lab)
                            if n_labs > 2:
                                switch_point = np.random.randint(2, n_labs)
                            else:
                                switch_point = n_labs  # No switch if only one lab or two labs
                            
                            for i in range(n_labs):
                                # Switch distribution at the switch point
                                if i == switch_point:
                                    current_distribution = "low_distribution" if current_distribution == "high_distribution" else "high_distribution"
                                
                                value = generate_lab_value(base_concept, current_distribution)
                                if value is not None:
                                    records.append({
                                        "PID": pid, 
                                        "CONCEPT": base_concept, 
                                        "RESULT": value,
                                        "LAB_INDEX": i,
                                        "CONDITION": condition
                                    })
                        else:
                            # Low-risk patients: stick to one distribution
                            distribution = np.random.choice(["high_distribution", "low_distribution"])
                            
                            for i in range(n_labs):
                                value = generate_lab_value(base_concept, distribution)
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
                # Check if patient exists in patient_df
                patient_matches = patient_df[patient_df["subject_id"] == pid]
                if len(patient_matches) > 0:
                    patient_info = patient_matches.iloc[0]
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
                    concept_timestamps[pid]["birthdate"] = birthdate
                    concept_timestamps[pid]["deathdate"] = deathdate
                else:
                    # Patient not found in patient_df, use default time range
                    start_time = pd.Timestamp(year=2016, month=1, day=1)
                    end_time = pd.Timestamp(year=2025, month=1, day=1)
                    time_diff = (end_time - start_time).total_seconds()
                    random_seconds = np.random.randint(0, int(time_diff))
                    concept_timestamps[pid]["start_time"] = start_time + pd.Timedelta(seconds=random_seconds)
                    concept_timestamps[pid]["birthdate"] = start_time
                    concept_timestamps[pid]["deathdate"] = end_time
            else:
                # Fallback to default time range if no patient_df provided
                start_time = pd.Timestamp(year=2016, month=1, day=1)
                end_time = pd.Timestamp(year=2025, month=1, day=1)
                time_diff = (end_time - start_time).total_seconds()
                random_seconds = np.random.randint(0, int(time_diff))
                concept_timestamps[pid]["start_time"] = start_time + pd.Timedelta(seconds=random_seconds)
                concept_timestamps[pid]["birthdate"] = start_time
                concept_timestamps[pid]["deathdate"] = end_time

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
                
                # Ensure timestamp is not after death date
                if patient_df is not None:
                    patient_matches = patient_df[patient_df["subject_id"] == pid]
                    if len(patient_matches) > 0:
                        patient_info = patient_matches.iloc[0]
                        deathdate = pd.to_datetime(patient_info["deathdate"])
                        if pd.isna(deathdate):
                            deathdate = pd.Timestamp(year=2025, month=1, day=1)
                        if timestamp > deathdate:
                            timestamp = deathdate - pd.Timedelta(days=1)
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
                
                # Ensure timestamp is not after death date
                if patient_df is not None:
                    patient_matches = patient_df[patient_df["subject_id"] == pid]
                    if len(patient_matches) > 0:
                        patient_info = patient_matches.iloc[0]
                        deathdate = pd.to_datetime(patient_info["deathdate"])
                        if pd.isna(deathdate):
                            deathdate = pd.Timestamp(year=2025, month=1, day=1)
                        if timestamp > deathdate:
                            timestamp = deathdate - pd.Timedelta(days=1)

        # Store the timestamp for this concept
        concept_timestamps[pid][concept] = timestamp
        timestamps.append(timestamp)

    return timestamps


def generate_synthetic_data(
    input_data: pd.DataFrame,
    min_labs_per_patient: int,
    max_labs_per_patient: int,
    switching_probability: float = 1.0,
    patient_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Generate synthetic data with multiple lab values per patient.
    Preserves existing positive/negative patient assignments from input data.
    
    Args:
        input_data: DataFrame containing existing synthetic data with patient assignments
        min_labs_per_patient: Minimum number of lab values per patient
        max_labs_per_patient: Maximum number of lab values per patient
        switching_probability: Probability of switching distributions for high-risk patients
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
    concepts_data = generate_multi_lab_concepts(pids_list, min_labs_per_patient, max_labs_per_patient, switching_probability, patient_risk_map)

    # Create final DataFrame - match simulate_synthetic_labs.py structure exactly
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
        DIAG_MIN_DAYS,
        DIAG_MAX_DAYS,
        patient_df
    )

    return data


def validate_timestamps(data: pd.DataFrame, patient_df: pd.DataFrame = None) -> None:
    """
    Validate that all timestamps fall between birth and death dates.
    Optimized version using vectorized operations.
    
    Args:
        data: DataFrame containing the synthetic data with timestamps
        patient_df: DataFrame containing patient information with birthdate and deathdate columns
    """
    if patient_df is None:
        print("No patient data provided - skipping timestamp validation")
        return
        
    print("\nValidating timestamps against birth/death dates...")
    
    # Merge data with patient info for vectorized validation
    merged_data = data.merge(patient_df[["subject_id", "birthdate", "deathdate"]], on="subject_id", how="left")
    
    # Handle missing death dates
    merged_data["deathdate"] = pd.to_datetime(merged_data["deathdate"])
    merged_data.loc[merged_data["deathdate"].isna(), "deathdate"] = pd.Timestamp(year=2025, month=1, day=1)
    
    # Convert birthdate to datetime
    merged_data["birthdate"] = pd.to_datetime(merged_data["birthdate"])
    
    # Vectorized validation
    before_birth = merged_data["time"] < merged_data["birthdate"]
    after_death = merged_data["time"] > merged_data["deathdate"]
    violations = before_birth | after_death
    
    total_checks = len(merged_data)
    violation_count = violations.sum()
    
    if violation_count > 0:
        # Show first few violations for debugging
        violation_data = merged_data[violations][["subject_id", "time", "birthdate", "deathdate"]].head(5)
        print(f"First few violations:")
        for _, row in violation_data.iterrows():
            print(f"  Patient {row['subject_id']}: timestamp {row['time']} outside birth ({row['birthdate']}) - death ({row['deathdate']}) range")
        if violation_count > 5:
            print(f"  ... and {violation_count - 5} more violations")
    
    print(f"Timestamp validation complete:")
    print(f"  Total timestamp checks: {total_checks}")
    print(f"  Violations found: {violation_count}")
    if violation_count == 0:
        print("  ✓ All timestamps are within birth/death date constraints")
    else:
        print(f"  ✗ {violation_count} timestamps violate birth/death date constraints")


def print_statistics(data: pd.DataFrame) -> None:
    """
    Print statistics about the lab values.

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

    print("\nLab value statistics (positive patients):")
    print(f"Count: {len(positive_lab_values)}")
    print(f"Mean: {positive_lab_values.mean():.3f}")
    print(f"Std: {positive_lab_values.std():.3f}")
    print(f"Min: {positive_lab_values.min():.3f}")
    print(f"Max: {positive_lab_values.max():.3f}")

    print("\nLab value statistics (negative patients):")
    print(f"Count: {len(negative_lab_values)}")
    print(f"Mean: {negative_lab_values.mean():.3f}")
    print(f"Std: {negative_lab_values.std():.3f}")
    print(f"Min: {negative_lab_values.min():.3f}")
    print(f"Max: {negative_lab_values.max():.3f}")


def calculate_theoretical_switch_auc(
    df,
    midpoint,
    lab_code="S/LAB1",
    subject_col="subject_id",
    value_col="numeric_value",
    positive_diag_code="S/DIAG_POSITIVE",
    target_col=None,             # if None, derive from diag code
    use_distance=True
):
    # 1) Work only on the lab rows
    labs = df[df["code"] == lab_code].copy()
    if labs.empty:
        raise ValueError(f"No rows found with code == {lab_code}")

    # 2) Get labels per subject (derive if not provided)
    if target_col is None:
        pos_ids = set(df[df["code"] == positive_diag_code][subject_col].unique())
        labs["_is_positive"] = labs[subject_col].isin(pos_ids).astype(int)
        target_col = "_is_positive"
    elif target_col not in labs.columns:
        raise ValueError(f"Target column '{target_col}' not present in lab rows.")

    # 3) Compute Bayes-optimal “switch” score using midpoint
    x = labs[value_col].to_numpy()
    if use_distance:
        d = x - midpoint
        labs["_pos"] = np.maximum(d, 0.0)
        labs["_neg"] = np.maximum(-d, 0.0)
        g = labs.groupby(subject_col)
        score = 2.0 * np.minimum(g["_pos"].sum(), g["_neg"].sum())
    else:
        left = (labs[value_col] < midpoint).groupby(labs[subject_col]).sum()
        right = (labs[value_col] >= midpoint).groupby(labs[subject_col]).sum()
        score = np.minimum(left, right).astype(float)

    y = labs.groupby(subject_col)[target_col].first().reindex(score.index)
    auc = roc_auc_score(y.values, score.values)
    return auc

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data with multiple lab values where high-risk patients switch between distributions"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=DEFAULT_INPUT_FILE,
        help="Path to input synthetic data CSV file",
    )
    parser.add_argument(
        "--min_labs_per_patient",
        type=int,
        default=MIN_LABS_PER_PATIENT,
        help="Minimum number of lab values per patient",
    )
    parser.add_argument(
        "--max_labs_per_patient",
        type=int,
        default=MAX_LABS_PER_PATIENT,
        help="Maximum number of lab values per patient",
    )
    parser.add_argument(
        "--switching_probability",
        type=float,
        default=SWITCHING_PROBABILITY,
        help="Probability of switching distributions for high-risk patients (0.0-1.0)",
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
        help="Path to patient information parquet file with birthdate and deathdate columns",
    )

    args = parser.parse_args()

    # Read input data
    try:
        input_data = pd.read_csv(args.input_file)
    except FileNotFoundError:
        print(f"Error: Could not find input file at {args.input_file}")
        return

    # Read patient data
    patient_df = None
    try:
        patient_df = pd.read_parquet(args.patient_info_path)
        print(f"Loaded patient data from {args.patient_info_path}")
        print(f"Patient data shape: {patient_df.shape}")
        print(f"Patient data columns: {list(patient_df.columns)}")
    except FileNotFoundError:
        print(f"Warning: Could not find patient info file at {args.patient_info_path}")
        print("Proceeding without birth/death date constraints")
    except Exception as e:
        print(f"Warning: Error loading patient data: {e}")
        print("Proceeding without birth/death date constraints")

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
    print(f"  - {args.min_labs_per_patient}-{args.max_labs_per_patient} lab values per patient")
    print(f"  - {positive_patients} high-risk patients (switching distributions)")
    print(f"  - {negative_patients} low-risk patients (consistent distribution)")
    print(f"  - {args.switching_probability*100:.1f}% switching probability for high-risk patients")

    # Generate synthetic data
    data = generate_synthetic_data(
        input_data,
        args.min_labs_per_patient,
        args.max_labs_per_patient,
        args.switching_probability,
        patient_df
    )

    print("\nGenerated data:")
    print(data.head())

    # Print statistics
    print_statistics(data)
    
    # Validate timestamps against birth/death dates
    validate_timestamps(data, patient_df)

    # Generate save name dynamically
    save_name = f"multi_lab_switching_risk_labs{args.min_labs_per_patient}_{args.max_labs_per_patient}_switch{int(args.switching_probability*100)}_n{N}_mean{int(LOW_MEAN*100)}_{int(HIGH_MEAN*100)}_std{int(STD*100)}"

    # Write to CSV
    write_dir = Path(args.write_dir)
    write_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(write_dir / f"{save_name}.csv", index=False)
    print(f"\nSaved synthetic data to {write_dir / f'{save_name}.csv'}")

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
    
    normalized_filename = write_dir / f"{save_name}_minmaxnorm.csv"
    normalized_data.to_csv(normalized_filename, index=False)

    # Calculate theoretical AUC
    midpoint = (HIGH_MEAN + LOW_MEAN) / 2
    theoretical_auc = calculate_theoretical_switch_auc(normalized_data, midpoint)
    print(f"Theoretical AUC: {theoretical_auc}")

if __name__ == "__main__":
    main()
