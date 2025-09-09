"""
Simulate synthetic medical data with correlated lab values.
This version allows specifying the exact number of patients with high lab values.
"""

import os
import uuid
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

# Default parameters
PATIENTS_INFO_PATH = "../../../data/vals/patient_infos/patient_info_10000n.parquet"
DEFAULT_WRITE_DIR = "../../../data/vals/synthetic_data/10000n/"
SAVE_NAME = "bn_labs_n10000_fixed_n"
DEFAULT_N_HIGH_PATIENTS = 5000  # Default number of high patients
DEFAULT_N_LOW_PATIENTS = 5000  # Default number of low patients

# Define lab value ranges and their probabilities for different conditions
LAB_VALUE_INFO = {
    "S/LAB1": {
        "high": {
            # "dist": "normal",
            # "mean": 0.6,
            # "std": 0.05,
            "dist": "uniform",
            "range": [1],  # High cases get value 1
            "n_samples": [1],  # Only one sample per patient
        },
        "low": {
            # "dist": "normal",
            # "mean": 0.4,
            # "std": 0.05,
            "dist": "uniform",
            "range": [0],  # Low cases get value 0
            "n_samples": [1],  # Only one sample per patient
        },
    },
}

CONCEPT_RELATIONSHIPS = {
    "S/LAB1": {
        "base_probability": 1.0,  # 100% of patients get labs
        "add_base_concept": ["high", "low"],  # Add lab for all conditions
        "related_concepts": {
            "S/DIAG_POSITIVE": {
                "prob": 1,  # 100% chance of getting diagnosis if lab is high
                "conditions": ["high"],  # Only high lab patients get diagnosis
                "time_relationship": {
                    "type": "after",  # Diagnosis comes after lab
                    "min_days": 10,
                    "max_days": 180,
                },
            },
            "S/DIAG_NEGATIVE": {
                "prob": 1,  # 100% chance of getting diagnosis if lab is low
                "conditions": ["low"],  # Only low lab patients get diagnosis
                "time_relationship": {
                    "type": "after",  # Diagnosis comes after lab
                    "min_days": 10,
                    "max_days": 180,
                },
            },
        },
    },
}


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


def get_related_concepts(
    base_concept: str, condition: str
) -> Tuple[List[str], List[float]]:
    """
    Get all related concepts and their values for a given base concept and condition.

    Args:
        base_concept: The base concept to get related concepts for
        condition: The condition (high/low) for this patient

    Returns:
        Tuple[List[str], List[float]]: Lists of related concepts and their values
    """
    related_concepts = []
    values = []

    if base_concept not in CONCEPT_RELATIONSHIPS:
        return related_concepts, values

    for lab_name, info in CONCEPT_RELATIONSHIPS[base_concept][
        "related_concepts"
    ].items():
        # Check if we should generate a lab value based on condition
        should_generate_lab = False
        if "conditions" in info:
            # Only generate if the current condition is in the allowed conditions
            should_generate_lab = condition in info["conditions"]
        else:
            # If no conditions specified, use probability
            should_generate_lab = np.random.random() < info["prob"]

        if should_generate_lab:
            # Generate value based on the selected condition
            value = generate_lab_value(lab_name, condition)
            if value is not None:
                related_concepts.append(lab_name)
                values.append(value)

    return related_concepts, values


def generate_lab_concepts(
    pids_list: List[str], n_high_patients: int, n_low_patients: int
) -> pd.DataFrame:
    """
    Generate lab concepts and values for a list of patient IDs.

    Args:
        pids_list: List of patient IDs
        n_high_patients: Number of patients that should have high lab values
        n_low_patients: Number of patients that should have low lab values

    Returns:
        pd.DataFrame: DataFrame containing PID, CONCEPT, and RESULT columns
    """
    records = []

    # Create a deterministic assignment of conditions
    # First n_high_patients get "high", next n_low_patients get "low"
    n_patients = len(pids_list)
    total_assigned = n_high_patients + n_low_patients

    if total_assigned != n_patients:
        raise ValueError(
            f"Total assigned patients ({total_assigned}) must equal total patients ({n_patients})"
        )

    # Shuffle the patient IDs to randomize which patients get high vs low
    shuffled_pids = pids_list.copy()
    np.random.shuffle(shuffled_pids)

    # Create a mapping from PID to condition
    pid_to_condition = {}
    for i, pid in enumerate(shuffled_pids):
        if i < n_high_patients:
            pid_to_condition[pid] = "high"
        else:
            pid_to_condition[pid] = "low"

    for pid in pids_list:
        # Get the pre-assigned condition for this patient
        condition = pid_to_condition[pid]

        # For each base concept in CONCEPT_RELATIONSHIPS
        for base_concept, info in CONCEPT_RELATIONSHIPS.items():
            # Determine if this patient gets this base concept
            if np.random.random() < info["base_probability"]:
                # Add the base concept if specified in the conditions
                if "add_base_concept" in info and condition in info["add_base_concept"]:
                    # Check if this is a lab concept or a diagnosis concept
                    if base_concept in LAB_VALUE_INFO:
                        # This is a lab concept, generate a value
                        value = generate_lab_value(base_concept, condition)
                        if value is not None:
                            records.append(
                                {"PID": pid, "CONCEPT": base_concept, "RESULT": value}
                            )
                    else:
                        # This is a diagnosis concept, add without value
                        records.append(
                            {"PID": pid, "CONCEPT": base_concept, "RESULT": None}
                        )

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
                        # Check if this is a lab concept or a diagnosis concept
                        if related_concept in LAB_VALUE_INFO:
                            # This is a lab concept, generate a value
                            value = generate_lab_value(related_concept, condition)
                            if value is not None:
                                records.append(
                                    {
                                        "PID": pid,
                                        "CONCEPT": related_concept,
                                        "RESULT": value,
                                    }
                                )
                        else:
                            # This is a diagnosis concept, add without value
                            records.append(
                                {"PID": pid, "CONCEPT": related_concept, "RESULT": None}
                            )

    return pd.DataFrame(records)


def generate_timestamps(
    pids_list: List[str], patient_df: pd.DataFrame, concepts: List[str]
) -> List[pd.Timestamp]:
    """
    Generate timestamps for a list of patient IDs based on time relationships specified in CONCEPT_RELATIONSHIPS.

    Args:
        pids_list: List of patient IDs to generate timestamps for
        patient_df: DataFrame containing patient information with birthdate and deathdate columns
        concepts: List of concepts corresponding to each PID

    Returns:
        List[pd.Timestamp]: List of generated timestamps
    """
    timestamps = []
    concept_timestamps = {}  # Store timestamps for each concept per patient

    for i, (pid, concept) in enumerate(zip(pids_list, concepts)):
        patient_info = patient_df[patient_df["PID"] == pid].iloc[0]
        birthdate = pd.Timestamp(
            year=2016, month=1, day=1
        )  # pd.to_datetime(patient_info["birthdate"])

        # Handle deathdate - if NaT, use a default future date
        deathdate = pd.to_datetime(patient_info["deathdate"])
        if pd.isna(deathdate):
            deathdate = pd.Timestamp(year=2025, month=1, day=1)

        # Ensure deathdate is after birthdate
        if deathdate <= birthdate:
            deathdate = birthdate + pd.Timedelta(days=1)

        # Initialize patient's concept timestamps if not exists
        if pid not in concept_timestamps:
            concept_timestamps[pid] = {}

        # Find the base concept and its time relationship for this concept
        time_relationship = None
        base_concept = None

        for bc, info in CONCEPT_RELATIONSHIPS.items():
            if concept in info.get("related_concepts", {}):
                time_relationship = info["related_concepts"][concept].get(
                    "time_relationship"
                )
                base_concept = bc
                break

        if time_relationship and base_concept:
            # If we have a time relationship and the base concept exists for this patient
            if base_concept in concept_timestamps[pid]:
                base_timestamp = concept_timestamps[pid][base_concept]
                if time_relationship["type"] == "before":
                    # Generate timestamp before the base concept
                    max_days = time_relationship["max_days"]
                    min_days = time_relationship["min_days"]
                    days_before = np.random.randint(min_days, max_days + 1)
                    timestamp = base_timestamp - pd.Timedelta(days=days_before)
                elif time_relationship["type"] == "after":
                    # Generate timestamp after the base concept
                    max_days = time_relationship["max_days"]
                    min_days = time_relationship["min_days"]
                    days_after = np.random.randint(min_days, max_days + 1)
                    timestamp = base_timestamp + pd.Timedelta(days=days_after)
            else:
                # If base concept doesn't exist yet, generate a random timestamp
                time_diff = (deathdate - birthdate).total_seconds()
                random_seconds = np.random.randint(0, int(time_diff))
                timestamp = birthdate + pd.Timedelta(seconds=random_seconds)
        else:
            # For base concepts or concepts without time relationships, generate random timestamp
            time_diff = (deathdate - birthdate).total_seconds()
            random_seconds = np.random.randint(0, int(time_diff))
            timestamp = birthdate + pd.Timedelta(seconds=random_seconds)

        # Store the timestamp for this concept
        concept_timestamps[pid][concept] = timestamp
        timestamps.append(timestamp)

    return timestamps


def generate_data(
    patient_df: pd.DataFrame, write_dir: str, n_high_patients: int, n_low_patients: int
) -> None:
    """
    Generate synthetic data for patients and save to parquet.

    Args:
        patient_df: DataFrame containing patient information
        write_dir: Directory to write output files
        n_high_patients: Number of patients that should have high lab values
        n_low_patients: Number of patients that should have low lab values
    """
    # Filter patient dataset to only include patients that will get lab values
    total_assigned = n_high_patients + n_low_patients
    if total_assigned > len(patient_df):
        raise ValueError(
            f"Total assigned patients ({total_assigned}) cannot be greater than available patients ({len(patient_df)})"
        )

    # Take only the first total_assigned patients
    filtered_patient_df = patient_df.head(total_assigned)

    # Generate concepts and lab values
    concepts_data = generate_lab_concepts(
        filtered_patient_df["PID"].tolist(), n_high_patients, n_low_patients
    )

    # Create final DataFrame
    data = pd.DataFrame(
        {
            "subject_id": concepts_data["PID"],
            "code": concepts_data["CONCEPT"],
            "numeric_value": concepts_data["RESULT"].astype(float),
        }
    )

    # Print statistics
    total_patients = len(
        filtered_patient_df
    )  # This should equal n_high_patients + n_low_patients
    patients_with_labs = len(data[data["code"] == "S/LAB1"])
    patients_with_positive_diagnosis = len(data[data["code"] == "S/DIAG_POSITIVE"])
    patients_with_negative_diagnosis = len(data[data["code"] == "S/DIAG_NEGATIVE"])
    print(f"Total patients: {total_patients}")
    print(f"Patients with labs: {patients_with_labs}")
    print(f"Patients with positive diagnosis: {patients_with_positive_diagnosis}")
    print(f"Patients with negative diagnosis: {patients_with_negative_diagnosis}")
    print(f"Percentage with labs: {patients_with_labs / total_patients * 100:.2f}%")
    print(
        f"Percentage with positive diagnosis: {patients_with_positive_diagnosis / total_patients * 100:.2f}%"
    )
    print(
        f"Percentage with negative diagnosis: {patients_with_negative_diagnosis / total_patients * 100:.2f}%"
    )
    print(f"Requested high patients: {n_high_patients}")
    print(f"Requested low patients: {n_low_patients}")
    print(
        f"Actual high patients: {len(data[(data['code'] == 'S/LAB1') & (data['numeric_value'] == 1)])}"
    )
    print(
        f"Actual low patients: {len(data[(data['code'] == 'S/LAB1') & (data['numeric_value'] == 0)])}"
    )
    print(f"Total patients in dataset: {len(patient_df)} (original)")
    print(f"Patients used: {total_patients} (filtered)")
    print("\nLab value distribution:")
    print(data["numeric_value"].value_counts())
    print("\nLab value percentages:")
    value_counts = data["numeric_value"].value_counts()
    total_labs = value_counts.sum()
    for value, count in value_counts.items():
        print(f"Value {value}: {count / total_labs * 100:.2f}%")

    # Generate timestamps for each record, ensuring labs come before diagnoses
    data["time"] = generate_timestamps(
        data["subject_id"].tolist(), filtered_patient_df, data["code"].tolist()
    )

    print(data.head())

    # Write to csv
    write_dir = Path(write_dir)
    write_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(write_dir / f"{SAVE_NAME}.csv", index=False)

    print(
        f"\nGenerated data for {len(filtered_patient_df)} patients, a total of {len(data)} records"
    )
    print(f"Saved to {write_dir / f'{SAVE_NAME}.csv'}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic medical data with correlated lab values (fixed number of high patients)"
    )
    parser.add_argument(
        "--write_dir",
        type=str,
        default=DEFAULT_WRITE_DIR,
        help="Directory to write output files",
    )
    parser.add_argument(
        "--patients_info_path",
        type=str,
        default=PATIENTS_INFO_PATH,
        help="Path to patient information parquet file",
    )
    parser.add_argument(
        "--n_high_patients",
        type=int,
        default=DEFAULT_N_HIGH_PATIENTS,
        help="Number of patients that should have high lab values",
    )
    parser.add_argument(
        "--n_low_patients",
        type=int,
        default=DEFAULT_N_LOW_PATIENTS,
        help="Number of patients that should have low lab values",
    )

    args = parser.parse_args()

    # Read patient data
    try:
        patient_df = pd.read_parquet(args.patients_info_path)
        patient_df = patient_df.rename(columns={"subject_id": "PID"})
    except FileNotFoundError:
        print(f"Error: Could not find patient info file at {args.patients_info_path}")
        return

    # Generate and save data
    generate_data(patient_df, args.write_dir, args.n_high_patients, args.n_low_patients)


if __name__ == "__main__":
    main()
