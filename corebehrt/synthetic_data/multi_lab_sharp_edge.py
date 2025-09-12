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
MIN_LABS_PER_PATIENT = 3
MAX_LABS_PER_PATIENT = 10
SWITCHING_PROBABILITY = 1.0  # 100% probability of switching for high-risk patients
LOW_MEAN = 0.45
HIGH_MEAN = 0.55
STD = 0.10
DEFAULT_WRITE_DIR = f"../../../data/vals/synthetic_data/{N}n/"
DEFAULT_PLOT_DIR = f"../../../data/vals/synthetic_data_plots/{N}n/"
SAVE_NAME = f"multi_lab_switching_risk_labs{MIN_LABS_PER_PATIENT}_{MAX_LABS_PER_PATIENT}_switch{int(SWITCHING_PROBABILITY*100)}_n{N}_mean{int(LOW_MEAN*100)}_{int(HIGH_MEAN*100)}_std{int(STD*100)}"
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
    pids_list: List[str], concepts: List[str], lab_indices: List[int]
) -> List[pd.Timestamp]:
    """
    Generate timestamps for a list of patient IDs based on time relationships.
    Similar to simulate_synthetic_labs.py but adapted for multiple labs per patient.

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
            # Use seconds precision to match simulate_synthetic_labs.py format
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
    min_labs_per_patient: int,
    max_labs_per_patient: int,
    switching_probability: float = 1.0
) -> pd.DataFrame:
    """
    Generate synthetic data with multiple lab values per patient.
    Preserves existing positive/negative patient assignments from input data.
    
    Args:
        input_data: DataFrame containing existing synthetic data with patient assignments
        min_labs_per_patient: Minimum number of lab values per patient
        max_labs_per_patient: Maximum number of lab values per patient
        switching_probability: Probability of switching distributions for high-risk patients
        
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
        concepts_data["LAB_INDEX"].tolist()
    )

    return data


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
    print(f"  - {args.min_labs_per_patient}-{args.max_labs_per_patient} lab values per patient")
    print(f"  - {positive_patients} high-risk patients (switching distributions)")
    print(f"  - {negative_patients} low-risk patients (consistent distribution)")
    print(f"  - {args.switching_probability*100:.1f}% switching probability for high-risk patients")

    # Generate synthetic data
    data = generate_synthetic_data(
        input_data,
        args.min_labs_per_patient,
        args.max_labs_per_patient,
        args.switching_probability
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

    # Calculate theoretical AUC
    midpoint = (HIGH_MEAN + LOW_MEAN) / 2
    theoretical_auc = calculate_theoretical_switch_auc(normalized_data, midpoint)
    print(f"Theoretical AUC: {theoretical_auc}")

if __name__ == "__main__":
    main()
