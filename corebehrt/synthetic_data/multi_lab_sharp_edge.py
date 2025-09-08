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

# Default parameters
DEFAULT_INPUT_FILE = "../../../data/vals/synthetic_data/100000n/bn_labs_n100000_50p_1unq.csv"
MIN_LABS_PER_PATIENT = 3
MAX_LABS_PER_PATIENT = 10
SWITCHING_PROBABILITY = 1.0  # 100% probability of switching for high-risk patients
LOW_MEAN = 0.35
HIGH_MEAN = 0.65
STD = 0.05
DEFAULT_WRITE_DIR = "../../../data/vals/synthetic_data/100000n/"
DEFAULT_PLOT_DIR = "../../../data/vals/synthetic_data_plots/100000n/"
SAVE_NAME = f"multi_lab_switching_risk_labs{MIN_LABS_PER_PATIENT}_{MAX_LABS_PER_PATIENT}_switch{int(SWITCHING_PROBABILITY*100)}p_mean{int(LOW_MEAN*100)}_{int(HIGH_MEAN*100)}_std{int(STD*100)}"
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


def create_distribution_plot(
    data: pd.DataFrame, save_path: Path, perfect_roc: float
) -> None:
    """
    Create a figure showing the distribution of lab values for positive vs negative patients.

    Args:
        data: DataFrame containing the synthetic data
        save_path: Path to save the plot
        perfect_roc: The theoretical perfect ROC AUC value
    """
    # Get lab values for positive and negative patients
    lab_mask = data["code"] == "S/LAB1"
    lab_data = data[lab_mask].copy()
    
    # Recreate is_positive column for analysis
    positive_patients = set(data[data["code"] == "S/DIAG_POSITIVE"]["subject_id"].unique())
    lab_data["is_positive"] = lab_data["subject_id"].isin(positive_patients)

    # Create a single subplot for the histogram
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    # Histogram
    positive_values = lab_data[lab_data["is_positive"]]["numeric_value"]
    negative_values = lab_data[~lab_data["is_positive"]]["numeric_value"]

    ax1.hist(positive_values, bins=30, alpha=0.7, label="High-Risk Patients", color="red")
    ax1.hist(negative_values, bins=30, alpha=0.7, label="Low-Risk Patients", color="blue")
    ax1.set_xlabel("Lab Value")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Lab Values by Risk Status")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add perfect ROC AUC text box
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # reserve top space for text
    fig.text(
        0.5,
        0.98,  # x=center, y=near top
        f"Theoretical Perfect ROC AUC: {perfect_roc:.4f}",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
    )

    os.makedirs(save_path.parent, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Distribution plot saved to {save_path}")


def create_sequence_plot(
    data: pd.DataFrame, save_path: Path
) -> None:
    """
    Create a plot showing lab value sequences for individual patients.

    Args:
        data: DataFrame containing the synthetic data
        save_path: Path to save the plot
    """
    lab_mask = data["code"] == "S/LAB1"
    lab_data = data[lab_mask].copy()
    
    # Recreate is_positive column for analysis
    positive_patients_set = set(data[data["code"] == "S/DIAG_POSITIVE"]["subject_id"].unique())
    lab_data["is_positive"] = lab_data["subject_id"].isin(positive_patients_set)
    
    # Sample a few patients for visualization
    positive_patients = lab_data[lab_data["is_positive"]]["subject_id"].unique()
    negative_patients = lab_data[~lab_data["is_positive"]]["subject_id"].unique()
    
    # Sample 5 patients from each group
    sample_positive = np.random.choice(positive_patients, min(5, len(positive_patients)), replace=False)
    sample_negative = np.random.choice(negative_patients, min(5, len(negative_patients)), replace=False)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot positive patients (high-risk: switching distributions)
    for patient_id in sample_positive:
        patient_data = lab_data[lab_data["subject_id"] == patient_id].sort_values("time")
        ax1.plot(patient_data["time"], patient_data["numeric_value"], 
                alpha=0.7, linewidth=2, label=f"Patient {patient_id}")
    
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Lab Value")
    ax1.set_title("High-Risk Patients (switching between distributions)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot negative patients (low-risk: consistent distribution)
    for patient_id in sample_negative:
        patient_data = lab_data[lab_data["subject_id"] == patient_id].sort_values("time")
        ax2.plot(patient_data["time"], patient_data["numeric_value"], 
                alpha=0.7, linewidth=2, label=f"Patient {patient_id}")
    
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Lab Value")
    ax2.set_title("Low-Risk Patients (consistent distribution)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(save_path.parent, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Sequence plot saved to {save_path}")


def calculate_theoretical_performance(data: pd.DataFrame) -> dict:
    """
    Calculate the theoretical performance of the model.

    Args:
        data: DataFrame containing the synthetic data
        
    Returns:
        dict: Dictionary containing performance metrics
    """
    sweep_auc = sweep_threshold_auc(data)
    mann_whitney_u = None  # manual_mann_whitney_u(data)
    scipy_mann_whitney_u_auc = scipy_mann_whitney_u(data)
    cohens_d_metric = cohens_d(data)
    
    # Calculate theoretical AUC based on distribution parameters
    # For normal distributions with means HIGH_MEAN and LOW_MEAN, and same std
    theoretical_auc = calculate_theoretical_auc(HIGH_MEAN, LOW_MEAN, STD)
    
    # Calculate switch detection AUCs
    switch_detection_auc = calculate_switch_detection_auc(data)
    theoretical_switch_auc = calculate_theoretical_switch_auc(HIGH_MEAN, LOW_MEAN, STD)
    
    print("\nTheoretical performance:")
    print(f"Sweep AUC: {sweep_auc}")
    print(f"Theoretical AUC (based on distributions): {theoretical_auc}")
    print(f"Switch Detection AUC (actual): {switch_detection_auc}")
    print(f"Theoretical Switch Detection AUC: {theoretical_switch_auc}")
    print(f"Mann-Whitney U: {mann_whitney_u}")
    print(f"Scipy Mann-Whitney U: {scipy_mann_whitney_u_auc}")
    print(f"Cohen's d: {cohens_d_metric}")
    return {
        "sweep_auc": sweep_auc,
        "theoretical_auc": theoretical_auc,
        "switch_detection_auc": switch_detection_auc,
        "theoretical_switch_auc": theoretical_switch_auc,
        "mann_whitney_u": mann_whitney_u,
        "scipy_mann_whitney_u_auc": scipy_mann_whitney_u_auc,
        "cohens_d_metric": cohens_d_metric,
    }


def calculate_theoretical_auc(mean1: float, mean2: float, std: float) -> float:
    """
    Calculate theoretical AUC for two normal distributions.
    
    Args:
        mean1: Mean of first distribution
        mean2: Mean of second distribution  
        std: Standard deviation (assumed same for both)
        
    Returns:
        float: Theoretical AUC
    """
    from scipy.stats import norm
    
    # For two normal distributions with same variance, AUC = Φ((μ1 - μ2) / (σ√2))
    # where Φ is the standard normal CDF
    z_score = (mean1 - mean2) / (std * np.sqrt(2))
    theoretical_auc = norm.cdf(z_score)
    
    return theoretical_auc


def calculate_switch_detection_auc(data: pd.DataFrame) -> float:
    """
    Calculate AUC for detecting high-risk patients based on switch detection.
    
    Args:
        data: DataFrame containing the synthetic data
        
    Returns:
        float: AUC for switch detection
    """
    lab_data = data[data['code'] == 'S/LAB1'].sort_values(['subject_id', 'time'])
    
    # Recreate is_positive column for analysis
    positive_patients = set(data[data["code"] == "S/DIAG_POSITIVE"]["subject_id"].unique())
    lab_data["is_positive"] = lab_data["subject_id"].isin(positive_patients)
    
    patient_switches = []
    patient_risks = []
    
    for patient_id in lab_data['subject_id'].unique():
        patient_labs = lab_data[lab_data['subject_id'] == patient_id]['numeric_value'].values
        is_positive = lab_data[lab_data['subject_id'] == patient_id]['is_positive'].iloc[0]
        
        # Count switches (values crossing the threshold)
        threshold = (HIGH_MEAN + LOW_MEAN) / 2  # Midpoint between distributions
        switches = sum(1 for i in range(1, len(patient_labs)) 
                      if (patient_labs[i-1] < threshold) != (patient_labs[i] < threshold))
        
        patient_switches.append(switches)
        patient_risks.append(is_positive)
    
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(patient_risks, patient_switches)
    return auc


def calculate_theoretical_switch_auc(mean1: float, mean2: float, std: float) -> float:
    """
    Calculate theoretical AUC for switch detection between two normal distributions.
    
    This calculates the probability that we can correctly identify high-risk patients
    (who switch once) vs low-risk patients (who don't switch) based on distribution overlap.
    
    Args:
        mean1: Mean of first distribution
        mean2: Mean of second distribution  
        std: Standard deviation (assumed same for both)
        
    Returns:
        float: Theoretical AUC for switch detection
    """
    from scipy.stats import norm
    
    # Calculate the probability of correctly detecting a switch
    # A switch is detected when a value crosses the midpoint threshold
    
    threshold = (mean1 + mean2) / 2
    
    # Probability that a value from distribution 1 is below threshold
    p1_below = norm.cdf((threshold - mean1) / std)
    
    # Probability that a value from distribution 2 is above threshold  
    p2_above = 1 - norm.cdf((threshold - mean2) / std)
    
    # Probability of detecting a switch from dist1 to dist2
    p_switch_detected = p1_below * p2_above
    
    # Probability of false switch detection (dist1 to dist1, but appears to cross)
    # This happens when both values are near the threshold due to noise
    p_false_switch = 2 * norm.pdf((threshold - mean1) / std) * norm.pdf((threshold - mean2) / std) * std
    
    # For perfect separation (no overlap), AUC = 1.0
    # For overlapping distributions, AUC depends on switch detection probability
    if abs(mean1 - mean2) > 6 * std:  # Essentially no overlap
        theoretical_auc = 1.0
    else:
        # More sophisticated calculation for overlapping case
        # This is an approximation - the exact calculation is complex
        separation = abs(mean1 - mean2) / std
        theoretical_auc = norm.cdf(separation / 2)  # Approximate based on separation
    
    return theoretical_auc


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
    theoretical_performance = calculate_theoretical_performance(data)

    # Create plots
    plot_dir = Path(DEFAULT_PLOT_DIR)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Distribution plot
    distribution_plot_path = plot_dir / f"{SAVE_NAME}_distribution_plot.png"
    create_distribution_plot(
        data, distribution_plot_path, perfect_roc=theoretical_performance["theoretical_switch_auc"]
    )
    
    # Sequence plot
    sequence_plot_path = plot_dir / f"{SAVE_NAME}_sequence_plot.png"
    create_sequence_plot(data, sequence_plot_path)

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
    
    # Create distribution plot for normalized data
    normalized_plot_path = plot_dir / f"{SAVE_NAME}_minmaxnorm_distribution_plot.png"
    normalized_theoretical_performance = calculate_theoretical_performance(normalized_data)
    create_distribution_plot(
        normalized_data,
        normalized_plot_path,
        perfect_roc=normalized_theoretical_performance["theoretical_switch_auc"],
    )
    print(f"\nSaved min-max normalized data to {normalized_filename}")


if __name__ == "__main__":
    main()
