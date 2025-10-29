"""
Generate synthetic lab data for patients based on positive status and percentage selection.
Labs are inserted into a percentage of patients, with values tied to positive status.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import List
from sklearn.metrics import roc_auc_score

# Default parameters
POS_COL = "bc"
PATIENTS_INFO_PATH = f"../../../data/vals/patient_infos/patient_info_real_data.parquet"
BASELINE_AUC = 0.669

# Number of labs per patient
NUM_LABS = 1
LOW_MEAN = 0.48
HIGH_MEAN = 0.52
STD = 0.10
PERCENT_LABS = 0.5

# Timing parameters - days before 2022
MIN_DAYS = 30
MAX_DAYS = 180
CENSOR_DATE = "2022-01-01"

DEFAULT_WRITE_DIR = f"../../../data/vals/synthetic_data/{POS_COL}/"
SAVE_NAME = f"labs_{POS_COL}_{int(PERCENT_LABS*100)}pct_{NUM_LABS}labs_low{int(LOW_MEAN*100)}_high{int(HIGH_MEAN*100)}_std{int(STD*100)}"


def get_positive_patients_from_timestamps(patient_info: pd.DataFrame, pos_col: str) -> pd.DataFrame:
    """
    Get positive patients based on timestamps in the specified column.
    
    Args:
        patient_info: DataFrame containing patient information with birthdate, deathdate, and timestamp columns
        pos_col: Column name that contains timestamps for positive patients
        
    Returns:
        pd.DataFrame: DataFrame with added is_positive column
    """
    # Check if the pos_col exists and has non-null timestamps
    if pos_col not in patient_info.columns:
        print(f"Warning: Column '{pos_col}' not found in patient_info. No patients will be marked as positive.")
        patient_info["is_positive"] = False
        return patient_info
    
    # Convert to datetime and check for non-null values
    patient_info[pos_col] = pd.to_datetime(patient_info[pos_col], errors='coerce')
    
    # Mark patients as positive if they have non-null timestamps in pos_col
    patient_info["is_positive"] = patient_info[pos_col].notna()
    
    positive_count = patient_info["is_positive"].sum()
    total_count = len(patient_info)
    print(f"Found {positive_count} positive patients out of {total_count} total patients ({positive_count/total_count*100:.1f}%)")
    
    return patient_info


def select_patients_for_labs(patient_info: pd.DataFrame, percent_labs: float) -> pd.DataFrame:
    """
    Select a percentage of patients to receive lab values.
    
    Args:
        patient_info: DataFrame containing patient information with is_positive column
        percent_labs: Percentage of patients to select for lab insertion (0.0 to 1.0)
        
    Returns:
        pd.DataFrame: DataFrame with added has_labs column
    """
    total_patients = len(patient_info)
    num_patients_with_labs = int(total_patients * percent_labs)
    
    # Randomly select patients for lab insertion
    selected_indices = np.random.choice(
        patient_info.index, 
        size=num_patients_with_labs, 
        replace=False
    )
    
    patient_info["has_labs"] = False
    patient_info.loc[selected_indices, "has_labs"] = True
    
    print(f"Selected {num_patients_with_labs} patients ({percent_labs*100:.1f}%) for lab insertion")
    
    return patient_info


def generate_lab_value_for_patient(is_positive: bool, low_mean: float, high_mean: float, std: float) -> float:
    """
    Generate a lab value based on patient positive status.
    
    Args:
        is_positive: Whether the patient is positive
        low_mean: Mean value for negative patients (Gaussian distribution)
        high_mean: Mean value for positive patients (Gaussian distribution)
        std: Standard deviation for the Gaussian distribution
        
    Returns:
        float: Generated lab value
    """
    if is_positive:
        # For positive patients, use Gaussian distribution with high_mean and std
        return np.random.normal(high_mean, std)
    else:
        # For negative patients, use Gaussian distribution with low_mean and std
        return np.random.normal(low_mean, std)


def generate_lab_concepts_for_selected_patients(
    patient_info: pd.DataFrame, 
    num_labs: int, 
    low_mean: float, 
    high_mean: float,
    std: float
) -> pd.DataFrame:
    """
    Generate lab concepts for selected patients based on their positive status.
    
    Args:
        patient_info: DataFrame containing patient information with is_positive and has_labs columns
        num_labs: Number of labs per patient
        low_mean: Mean value for negative patients (Gaussian distribution)
        high_mean: Mean value for positive patients (Gaussian distribution)
        std: Standard deviation for the Gaussian distribution
        
    Returns:
        pd.DataFrame: DataFrame containing PID, CONCEPT, and RESULT columns
    """
    records = []
    
    # Get patients who have labs
    patients_with_labs = patient_info[patient_info["has_labs"]]
    
    print(f"Generating labs for {len(patients_with_labs)} patients")
    
    for _, patient in patients_with_labs.iterrows():
        pid = patient["subject_id"]
        is_positive = patient["is_positive"]
        
        # Generate lab values for this patient
        for i in range(num_labs):
            lab_value = generate_lab_value_for_patient(is_positive, low_mean, high_mean, std)
            
            records.append({
                "PID": pid,
                "CONCEPT": f"S/LAB{i+1}",
                "RESULT": lab_value,
                "LAB_INDEX": i,
                "IS_POSITIVE": is_positive
            })
    
    return pd.DataFrame(records)


def generate_timestamps_for_labs(
    patient_info: pd.DataFrame, 
    concepts_data: pd.DataFrame
) -> List[pd.Timestamp]:
    """
    Generate timestamps for lab concepts within patient birth-death or 2022 timeframe.
    Uses MIN_DAYS and MAX_DAYS to determine days before 2022.
    
    Args:
        patient_info: DataFrame containing patient information with birthdate, deathdate
        concepts_data: DataFrame containing lab concepts with PID, CONCEPT, LAB_INDEX
        
    Returns:
        List[pd.Timestamp]: List of generated timestamps
    """
    timestamps = []
    censor_date = pd.Timestamp(CENSOR_DATE)
    
    for _, row in concepts_data.iterrows():
        pid = row["PID"]
        lab_index = row["LAB_INDEX"]
        
        # Get patient information
        patient_match = patient_info[patient_info["subject_id"] == pid]
        if len(patient_match) == 0:
            raise ValueError(f"Patient {pid} not found in patient_info. This should not happen.")
            
        patient = patient_match.iloc[0]
        
        # Handle birthdate - skip if invalid
        raw_birthdate = patient["birthdate"]
        birthdate = pd.to_datetime(raw_birthdate, errors='coerce')
        if pd.isna(birthdate) or birthdate is None:
            print(f"Warning: Patient {pid} has no birthdate or NaT birthdate. Skipping patient.")
            # Use a fallback timestamp for this patient
            timestamp = pd.Timestamp("2020-01-01") + pd.Timedelta(days=lab_index * 30)
            timestamps.append(timestamp)
            continue
        
        # Handle deathdate - if NaT or None, use 2022 as cutoff
        raw_deathdate = patient["deathdate"]
        deathdate = pd.to_datetime(raw_deathdate, errors='coerce')
        
        # Debug: print problematic cases
        if pd.isna(deathdate) or deathdate is None:
            deathdate = censor_date
        else:
            # Use the earlier of deathdate or 2022
            deathdate = min(deathdate, censor_date)
        
        # Ensure deathdate is after birthdate
        if deathdate <= birthdate:
            deathdate = birthdate + pd.Timedelta(days=1)
        
        # Generate timestamp within the valid timeframe
        # Use MIN_DAYS and MAX_DAYS before 2022 (or deathdate)
        end_date = deathdate - pd.Timedelta(days=MIN_DAYS)
        start_date = end_date - pd.Timedelta(days=MAX_DAYS)
        
        # Generate random timestamp within the valid range
        time_diff = (end_date - start_date).total_seconds()
        if time_diff <= 0:
            # Fallback if invalid timeframe
            timestamp = start_date + pd.Timedelta(days=lab_index * 30)
        else:
            # Random time within the valid range, with some spacing for multiple labs
            max_offset = max(0, time_diff - (lab_index * 30 * 24 * 3600))  # Leave room for lab spacing
            random_seconds = np.random.randint(0, int(max_offset))
            timestamp = start_date + pd.Timedelta(seconds=random_seconds) + pd.Timedelta(days=lab_index * 30)
            
            # Ensure timestamp doesn't exceed end_date
            if timestamp > end_date:
                timestamp = end_date - pd.Timedelta(days=1)
        
        timestamps.append(timestamp)
    
    return timestamps


def generate_labs_for_patients(
    patient_info: pd.DataFrame,
    pos_col: str,
    percent_labs: float,
    num_labs: int,
    low_mean: float,
    high_mean: float,
    std: float
) -> pd.DataFrame:
    """
    Main function to generate labs for patients based on positive status and percentage selection.
    
    Args:
        patient_info: DataFrame containing patient information with birthdate, deathdate, and timestamp columns
        pos_col: Column name that contains timestamps for positive patients
        percent_labs: Percentage of patients to select for lab insertion
        num_labs: Number of labs per patient
        low_mean: Mean value for negative patients (Gaussian distribution)
        high_mean: Mean value for positive patients (Gaussian distribution)
        std: Standard deviation for the Gaussian distribution
        
    Returns:
        pd.DataFrame: Generated lab data with timestamps
    """
    print(f"Processing {len(patient_info)} patients...")
    
    # Step 1: Identify positive patients based on timestamps in pos_col
    patient_info = get_positive_patients_from_timestamps(patient_info, pos_col)
    
    # Step 2: Select percentage of patients for lab insertion
    patient_info = select_patients_for_labs(patient_info, percent_labs)
    
    # Step 3: Generate lab concepts for selected patients
    concepts_data = generate_lab_concepts_for_selected_patients(
        patient_info, num_labs, low_mean, high_mean, std
    )
    
    # Step 4: Generate timestamps within patient birth-death or 2022 timeframe
    timestamps = generate_timestamps_for_labs(patient_info, concepts_data)
    
    # Step 5: Create final DataFrame
    data = pd.DataFrame({
        "subject_id": concepts_data["PID"],
        "code": concepts_data["CONCEPT"],
        "numeric_value": concepts_data["RESULT"].astype(float),
        "time": timestamps
    })
    
    return data


def calculate_theoretical_auc_improvement(
    patient_info: pd.DataFrame,
    lab_data: pd.DataFrame,
    baseline_auc: float,
    low_mean: float,
    high_mean: float,
    std: float
) -> dict:
    """
    Calculate the theoretical AUC improvement from adding lab values.
    
    Args:
        patient_info: DataFrame containing patient information with is_positive column
        lab_data: DataFrame containing lab data with subject_id, code, numeric_value
        baseline_auc: Baseline AUC before adding lab values (e.g., 0.75)
        low_mean: Mean value for negative patients
        high_mean: Mean value for positive patients
        std: Standard deviation for lab values
        
    Returns:
        dict: Dictionary containing AUC calculations and improvements
    """
    # Get patients with labs
    patients_with_labs = set(lab_data['subject_id'].unique())
    
    # Separate positive and negative patients
    positive_patients = set(patient_info[patient_info['is_positive']]['subject_id'])
    negative_patients = set(patient_info[~patient_info['is_positive']]['subject_id'])
    
    # Calculate lab AUC for patients with labs
    lab_patients_with_labs = patients_with_labs.intersection(positive_patients.union(negative_patients))
    
    if len(lab_patients_with_labs) == 0:
        return {"error": "No patients with labs found"}
    
    # Get lab values and labels for patients with labs
    lab_values = []
    labels = []
    
    for patient_id in lab_patients_with_labs:
        patient_labs = lab_data[lab_data['subject_id'] == patient_id]
        if len(patient_labs) > 0:
            # Use the first lab value (or average if multiple labs)
            lab_value = patient_labs['numeric_value'].mean()
            lab_values.append(lab_value)
            
            # Get label (1 for positive, 0 for negative)
            is_positive = patient_id in positive_patients
            labels.append(1 if is_positive else 0)
    
    if len(lab_values) == 0:
        return {"error": "No valid lab values found"}
    
    # Calculate lab-only AUC
    lab_auc = roc_auc_score(labels, lab_values)
    
    # Calculate theoretical combined AUC using the formula for combining two predictors
    # Assuming independence, the combined AUC can be approximated using:
    # AUC_combined ≈ 1 - (1 - AUC1) * (1 - AUC2) for independent predictors
    # But this is an approximation. A more accurate method is needed.
    
    # For a more accurate calculation, we can use the fact that:
    # If we have two independent predictors with AUCs A1 and A2,
    # the combined AUC is approximately: AUC_combined ≈ 1 - (1 - A1) * (1 - A2)
    # However, this assumes the predictors are independent and equally weighted.
    
    # A better approximation for combining predictors is:
    # AUC_combined ≈ 1 - (1 - AUC1) * (1 - AUC2) / (1 - AUC1 * AUC2)
    # But this is still an approximation.
    
    # For a more realistic calculation, we can use the fact that:
    # If the lab values are perfectly correlated with the outcome (AUC = 1.0),
    # then the combined AUC would be 1.0.
    # If the lab values are uncorrelated (AUC = 0.5), then the combined AUC would be the baseline.
    
    # A more accurate approach is to use the formula:
    # AUC_combined = 1 - (1 - AUC1) * (1 - AUC2) / (1 - AUC1 * AUC2)
    # This is the formula for combining two independent predictors.
    
    if lab_auc > 0.5:
        # Calculate combined AUC using the formula for combining independent predictors
        combined_auc = 1 - (1 - baseline_auc) * (1 - lab_auc) / (1 - baseline_auc * lab_auc)
    else:
        # If lab AUC is <= 0.5, it's not helpful
        combined_auc = baseline_auc
    
    # Calculate improvement
    auc_improvement = combined_auc - baseline_auc
    
    # Calculate theoretical maximum improvement
    max_improvement = 1.0 - baseline_auc
    
    return {
        "baseline_auc": baseline_auc,
        "lab_auc": lab_auc,
        "combined_auc": combined_auc,
        "auc_improvement": auc_improvement,
        "max_possible_improvement": max_improvement,
        "improvement_percentage": (auc_improvement / max_improvement) * 100 if max_improvement > 0 else 0,
        "patients_with_labs": len(lab_patients_with_labs),
        "total_patients": len(positive_patients) + len(negative_patients)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic lab data for patients based on positive status and percentage selection"
    )
    parser.add_argument(
        "--patient_info_path",
        type=str,
        default=PATIENTS_INFO_PATH,
        help="Path to patient information parquet file",
    )
    parser.add_argument(
        "--pos_col",
        type=str,
        default=POS_COL,
        help="Column name that contains timestamps for positive patients",
    )
    parser.add_argument(
        "--percent_labs",
        type=float,
        default=PERCENT_LABS,
        help="Percentage of patients to select for lab insertion (0.0 to 1.0)",
    )
    parser.add_argument(
        "--num_labs",
        type=int,
        default=NUM_LABS,
        help="Number of labs per patient",
    )
    parser.add_argument(
        "--low_mean",
        type=float,
        default=LOW_MEAN,
        help="Mean value for negative patients (Gaussian distribution)",
    )
    parser.add_argument(
        "--high_mean",
        type=float,
        default=HIGH_MEAN,
        help="Mean value for positive patients (Gaussian distribution)",
    )
    parser.add_argument(
        "--std",
        type=float,
        default=STD,
        help="Standard deviation for the Gaussian distribution",
    )
    parser.add_argument(
        "--min_days",
        type=int,
        default=MIN_DAYS,
        help="Minimum days before 2022 for lab timestamps",
    )
    parser.add_argument(
        "--max_days",
        type=int,
        default=MAX_DAYS,
        help="Maximum days before 2022 for lab timestamps",
    )
    parser.add_argument(
        "--write_dir",
        type=str,
        default=DEFAULT_WRITE_DIR,
        help="Directory to write output files",
    )
    parser.add_argument(
        "--baseline_auc",
        type=float,
        default=BASELINE_AUC,
        help="Baseline AUC before adding lab values (e.g., 0.75). If provided, will calculate theoretical AUC improvement.",
    )

    parser.add_argument(
        "--save_name",
        type=str,
        default=SAVE_NAME,
        help="Name of the save file",
    )

    args = parser.parse_args()

    # Read patient info data
    try:
        patient_info = pd.read_parquet(args.patient_info_path)
        print(f"Loaded patient info from {args.patient_info_path}")
        print(f"Patient info contains {len(patient_info)} patients")
        print("\nPatient info structure:")
        print(patient_info.head())
        print(f"\nColumns: {list(patient_info.columns)}")
    except FileNotFoundError:
        print(f"Error: Could not find patient info file at {args.patient_info_path}")
        return

    print(f"\nGenerating lab data with:")
    print(f"  - {len(patient_info)} patients")
    print(f"  - Positive column: {args.pos_col}")
    print(f"  - Percentage of patients with labs: {args.percent_labs*100:.1f}%")
    print(f"  - Number of labs per patient: {args.num_labs}")
    print(f"  - Gaussian distribution for negative patients: mean={args.low_mean}, std={args.std}")
    print(f"  - Gaussian distribution for positive patients: mean={args.high_mean}, std={args.std}")
    print(f"  - Lab timestamps: {args.min_days}-{args.max_days} days before 2022")

    # Generate lab data
    data = generate_labs_for_patients(
        patient_info,
        args.pos_col,
        args.percent_labs,
        args.num_labs,
        args.low_mean,
        args.high_mean,
        args.std
    )

    print("\nGenerated lab data:")
    print(data.head())

    # Print statistics
    print(f"\nLab data statistics:")
    print(f"Total lab records: {len(data)}")
    print(f"Total patients with labs: {data['subject_id'].nunique()}")
    print(f"Lab codes: {data['code'].unique()}")

    # Calculate percentage of positive and negative patients with labs
    if len(data) > 0:
        # Get patients with labs
        patients_with_labs = set(data['subject_id'].unique())
        
        # Get positive and negative patients from original patient_info
        positive_patients = set(patient_info[patient_info['is_positive']]['subject_id'])
        negative_patients = set(patient_info[~patient_info['is_positive']]['subject_id'])
        
        # Calculate percentages
        positive_with_labs = len(positive_patients.intersection(patients_with_labs))
        negative_with_labs = len(negative_patients.intersection(patients_with_labs))
        
        total_positive = len(positive_patients)
        total_negative = len(negative_patients)
        
        if total_positive > 0:
            positive_percentage = (positive_with_labs / total_positive) * 100
            print(f"\nPositive patients with labs: {positive_with_labs}/{total_positive} ({positive_percentage:.1f}%)")
        else:
            print(f"\nPositive patients with labs: 0/0 (0.0%)")
            
        if total_negative > 0:
            negative_percentage = (negative_with_labs / total_negative) * 100
            print(f"Negative patients with labs: {negative_with_labs}/{total_negative} ({negative_percentage:.1f}%)")
        else:
            print(f"Negative patients with labs: 0/0 (0.0%)")

    # Write to CSV
    write_dir = Path(args.write_dir)
    write_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dynamic save name based on arguments
    save_name = args.save_name
    output_file = write_dir / f"{save_name}.csv"
    data.to_csv(output_file, index=False)
    print(f"\nSaved lab data to {output_file}")

    # Print summary statistics
    if len(data) > 0:
        print(f"\nLab value statistics:")
        for lab_code in data['code'].unique():
            lab_data = data[data['code'] == lab_code]
            print(f"{lab_code}:")
            print(f"  Count: {len(lab_data)}")
            print(f"  Mean: {lab_data['numeric_value'].mean():.3f}")
            print(f"  Std: {lab_data['numeric_value'].std():.3f}")
            print(f"  Min: {lab_data['numeric_value'].min():.3f}")
            print(f"  Max: {lab_data['numeric_value'].max():.3f}")
    
    # Calculate theoretical AUC improvement if baseline AUC is provided
    if args.baseline_auc is not None and len(data) > 0:
        print(f"\n{'='*60}")
        print("THEORETICAL AUC IMPROVEMENT CALCULATION")
        print(f"{'='*60}")
        
        auc_results = calculate_theoretical_auc_improvement(
            patient_info, data, args.baseline_auc, args.low_mean, args.high_mean, args.std
        )
        
        if "error" in auc_results:
            print(f"Error calculating AUC improvement: {auc_results['error']}")
        else:
            print(f"Baseline AUC (before labs): {auc_results['baseline_auc']:.3f}")
            print(f"Lab-only AUC: {auc_results['lab_auc']:.3f}")
            print(f"Theoretical combined AUC: {auc_results['combined_auc']:.3f}")
            print(f"AUC improvement: +{auc_results['auc_improvement']:.3f}")
            print(f"Maximum possible improvement: {auc_results['max_possible_improvement']:.3f}")
            print(f"Improvement percentage: {auc_results['improvement_percentage']:.1f}% of maximum possible")
            print(f"Patients with labs: {auc_results['patients_with_labs']}/{auc_results['total_patients']}")
            
            # Calculate effect size (Cohen's d equivalent for AUC)
            effect_size = (args.high_mean - args.low_mean) / args.std
            print(f"Lab effect size (Cohen's d): {effect_size:.3f}")
            

if __name__ == "__main__":
    main()