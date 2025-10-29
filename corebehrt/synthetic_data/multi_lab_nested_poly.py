"""
Generate synthetic data with multiple lab values where positive patients are determined 
by nested polynomial functions: poly1(poly2(LAB1, LAB2, ...), poly3(LAB1, LAB2, ...), ...) > threshold.
Coefficients are drawn uniformly from -1 to 1.
Based on the multi_lab_logistic.py structure.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import Optional, List, Dict
from theoretical_separation import (
    cohens_d,
    sweep_threshold_auc,
    scipy_mann_whitney_u,
)

# Default parameters
N = 100000
DEFAULT_INPUT_FILE = f"../../../data/vals/synthetic_data/{N}n/bn_labs_n{N}_50p_1unq.csv"
PATIENTS_INFO_PATH = f"../../../data/vals/patient_infos/patient_info_{N}n.parquet"

# Lab value distributions - same distribution for all labs
LAB_MEAN = 0.3  # Mean for all labs
LAB_STD = 0.1   # Std for all labs

# Nested polynomial parameters
NUM_LABS = 2  # Default to 2 labs, can be changed via command line
NOISE_LEVEL = 0.0  # Multiplicative noise applied to the nested polynomial result
POSITIVE_RATE = 0.5  # Default to 50% positive, 50% negative
NUM_NESTED_POLYS = 3  # Number of nested polynomial functions
POLY_DEGREE = 2  # Degree of each polynomial (1=linear, 2=quadratic, etc.)

# Diagnosis timing parameters
DIAG_MIN_DAYS = 10  # Minimum days after last lab for diagnosis
DIAG_MAX_DAYS = 180  # Maximum days after last lab for diagnosis

DEFAULT_WRITE_DIR = f"../../../data/vals/synthetic_data/{N}n/"
DEFAULT_PLOT_DIR = f"../../../data/vals/synthetic_data_plots/{N}n/"
POSITIVE_DIAGS = ["S/DIAG_POSITIVE"]


def generate_nested_polynomial_coefficients(num_labs: int, num_nested_polys: int, poly_degree: int, seed: int = None) -> Dict[str, Dict[str, float]]:
    """
    Generate coefficients for nested polynomial functions.
    Each nested polynomial operates on the lab values, and the final polynomial operates on the results.
    
    Args:
        num_labs: Number of lab variables
        num_nested_polys: Number of nested polynomial functions
        poly_degree: Degree of each polynomial
        seed: Random seed for reproducibility
        
    Returns:
        Dict mapping polynomial names to their coefficient dictionaries
    """
    if seed is not None:
        np.random.seed(seed)
    
    coefficients = {}
    
    # Generate coefficients for each nested polynomial
    for i in range(1, num_nested_polys + 1):
        poly_name = f"poly{i}"
        coefficients[poly_name] = {}
        
        # Constant term
        coefficients[poly_name]['c0'] = np.random.uniform(-1, 1)
        
        # Generate all possible combinations for each degree
        for d in range(1, poly_degree + 1):
            for combo in generate_combinations(num_labs, d):
                coef_name = 'c' + ''.join(map(str, combo))
                coefficients[poly_name][coef_name] = np.random.uniform(-1, 1)
    
    # Generate coefficients for the final polynomial that operates on nested results
    coefficients['final_poly'] = {}
    coefficients['final_poly']['c0'] = np.random.uniform(-1, 1)
    
    # Linear terms for each nested polynomial result
    for i in range(1, num_nested_polys + 1):
        coefficients['final_poly'][f'c{i}'] = np.random.uniform(-1, 1)
    
    return coefficients


def generate_combinations(num_labs: int, degree: int) -> List[List[int]]:
    """
    Generate all combinations of lab indices for a given degree.
    Returns combinations where indices are non-decreasing (i <= j <= k <= ...).
    
    Args:
        num_labs: Number of lab variables
        degree: Degree of the polynomial term
        
    Returns:
        List of lists, where each inner list contains lab indices
    """
    if degree == 0:
        return [[]]
    
    combinations = []
    
    def _generate_combinations(current_combo, start_idx):
        if len(current_combo) == degree:
            combinations.append(current_combo[:])
            return
        
        for i in range(start_idx, num_labs + 1):
            current_combo.append(i)
            _generate_combinations(current_combo, i)  # Allow repetition by starting from i
            current_combo.pop()
    
    _generate_combinations([], 1)
    return combinations


def evaluate_polynomial(lab_values: List[float], coefficients: Dict[str, float], num_labs: int, degree: int) -> float:
    """
    Evaluate a polynomial function for given lab values.
    
    Args:
        lab_values: List of lab values [LAB1, LAB2, ..., LABn]
        coefficients: Dictionary of polynomial coefficients
        num_labs: Number of lab variables
        degree: Polynomial degree
        
    Returns:
        float: Polynomial evaluation result
    """
    result = 0.0
    
    # Constant term
    result += coefficients.get('c0', 0.0)
    
    # Evaluate all polynomial terms
    for d in range(1, degree + 1):
        for combo in generate_combinations(num_labs, d):
            # Check if all indices are within bounds
            if all(idx <= len(lab_values) for idx in combo):
                coef_name = 'c' + ''.join(map(str, combo))
                coef_value = coefficients.get(coef_name, 0.0)
                
                # Calculate the product of lab values for this combination
                term_value = 1.0
                for idx in combo:
                    term_value *= lab_values[idx - 1]  # Convert to 0-based indexing
                
                result += coef_value * term_value
    
    return result


def evaluate_nested_polynomials(lab_values: List[float], coefficients: Dict[str, Dict[str, float]], 
                               num_labs: int, num_nested_polys: int, poly_degree: int) -> float:
    """
    Evaluate nested polynomial functions for given lab values.
    
    Args:
        lab_values: List of lab values [LAB1, LAB2, ..., LABn]
        coefficients: Dictionary of polynomial coefficients for all nested polynomials
        num_labs: Number of lab variables
        num_nested_polys: Number of nested polynomial functions
        poly_degree: Degree of each polynomial
        
    Returns:
        float: Final nested polynomial evaluation result
    """
    # Evaluate each nested polynomial on the lab values
    nested_results = []
    for i in range(1, num_nested_polys + 1):
        poly_name = f"poly{i}"
        nested_result = evaluate_polynomial(lab_values, coefficients[poly_name], num_labs, poly_degree)
        nested_results.append(nested_result)
    
    # Evaluate the final polynomial on the nested results
    final_result = evaluate_polynomial(nested_results, coefficients['final_poly'], num_nested_polys, 1)  # Linear combination of nested results
    
    return final_result


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


def print_nested_polynomial_equation(coefficients: Dict[str, Dict[str, float]], num_labs: int, 
                                   num_nested_polys: int, poly_degree: int) -> None:
    """
    Print the nested polynomial equation in a readable format.
    
    Args:
        coefficients: Dictionary of polynomial coefficients for all nested polynomials
        num_labs: Number of lab variables
        num_nested_polys: Number of nested polynomial functions
        poly_degree: Degree of each polynomial
    """
    print(f"\nNested polynomial equation:")
    print("result = final_poly(", end="")
    
    # Print each nested polynomial
    nested_terms = []
    for i in range(1, num_nested_polys + 1):
        poly_name = f"poly{i}"
        poly_terms = []
        
        # Constant term
        if 'c0' in coefficients[poly_name] and abs(coefficients[poly_name]['c0']) > 1e-6:
            poly_terms.append(f"{coefficients[poly_name]['c0']:.3f}")
        
        # All polynomial terms
        for d in range(1, poly_degree + 1):
            for combo in generate_combinations(num_labs, d):
                coef_name = 'c' + ''.join(map(str, combo))
                if coef_name in coefficients[poly_name] and abs(coefficients[poly_name][coef_name]) > 1e-6:
                    # Format the term
                    term_str = format_polynomial_term(coefficients[poly_name][coef_name], combo)
                    poly_terms.append(term_str)
        
        if not poly_terms:
            nested_terms.append("0")
        else:
            nested_terms.append(" + ".join(poly_terms).replace("+ -", "- "))
    
    print(", ".join(nested_terms), end="")
    print(")")
    
    # Print the final polynomial
    print("where:")
    for i, nested_term in enumerate(nested_terms, 1):
        print(f"  poly{i} = {nested_term}")
    
    # Print final polynomial
    final_terms = []
    if 'c0' in coefficients['final_poly'] and abs(coefficients['final_poly']['c0']) > 1e-6:
        final_terms.append(f"{coefficients['final_poly']['c0']:.3f}")
    
    for i in range(1, num_nested_polys + 1):
        coef_name = f'c{i}'
        if coef_name in coefficients['final_poly'] and abs(coefficients['final_poly'][coef_name]) > 1e-6:
            final_terms.append(f"{coefficients['final_poly'][coef_name]:.3f}*poly{i}")
    
    if not final_terms:
        final_expr = "0"
    else:
        final_expr = " + ".join(final_terms).replace("+ -", "- ")
    
    print(f"  final_poly = {final_expr}")


def format_polynomial_term(coef_value: float, combo: List[int]) -> str:
    """
    Format a polynomial term for display.
    
    Args:
        coef_value: Coefficient value
        combo: List of lab indices (e.g., [1, 1, 2] for LAB1²*LAB2)
        
    Returns:
        Formatted string for the term
    """
    # Count occurrences of each lab index
    lab_counts = {}
    for idx in combo:
        lab_counts[idx] = lab_counts.get(idx, 0) + 1
    
    # Build the term string
    coef_str = f"{coef_value:.3f}"
    
    # Add lab variables with their powers
    lab_parts = []
    for lab_idx in sorted(lab_counts.keys()):
        count = lab_counts[lab_idx]
        if count == 1:
            lab_parts.append(f"LAB{lab_idx}")
        elif count == 2:
            lab_parts.append(f"LAB{lab_idx}²")
        elif count == 3:
            lab_parts.append(f"LAB{lab_idx}³")
        else:
            lab_parts.append(f"LAB{lab_idx}^{count}")
    
    if lab_parts:
        return f"{coef_str}*{'*'.join(lab_parts)}"
    else:
        return coef_str


def find_optimal_threshold(pids_list: List[str], num_labs: int, 
                          lab_value_info: dict, coefficients: Dict[str, Dict[str, float]], 
                          num_nested_polys: int, poly_degree: int, target_positive_rate: float = 0.5) -> float:
    """
    Find the threshold that results in the target positive rate.
    
    Args:
        pids_list: List of patient IDs
        num_labs: Number of labs per patient
        lab_value_info: Dictionary containing lab distribution information
        coefficients: Dictionary of polynomial coefficients for all nested polynomials
        num_nested_polys: Number of nested polynomial functions
        poly_degree: Degree of each polynomial
        target_positive_rate: Target fraction of positive cases (e.g., 0.5 for 50%)
        
    Returns:
        float: Optimal threshold
    """
    # Generate a sample of nested polynomial results to find the threshold
    sample_size = min(10000, len(pids_list))  # Use up to 10k samples for efficiency
    sample_pids = np.random.choice(pids_list, size=sample_size, replace=False)
    
    nested_poly_results = []
    for pid in sample_pids:
        # Generate clean lab values
        clean_lab_values = []
        for i in range(1, num_labs + 1):
            lab_concept = f"S/LAB{i}"
            clean_lab_value = generate_lab_value(lab_concept, lab_value_info, noise_level=0.0)
            if clean_lab_value is not None:
                clean_lab_values.append(clean_lab_value)
        
        if len(clean_lab_values) == num_labs:
            # Calculate clean nested polynomial evaluation
            clean_nested_poly_result = evaluate_nested_polynomials(clean_lab_values, coefficients, num_labs, num_nested_polys, poly_degree)
            nested_poly_results.append(clean_nested_poly_result)
    
    # Find threshold that gives target positive rate
    nested_poly_results = np.array(nested_poly_results)
    threshold = np.percentile(nested_poly_results, (1 - target_positive_rate) * 100)
    
    return threshold


def generate_n_lab_concepts(pids_list: List[str], threshold: float, num_labs: int, 
                          lab_value_info: dict, coefficients: Dict[str, Dict[str, float]], 
                          num_nested_polys: int, poly_degree: int, noise_level: float = 0.0) -> pd.DataFrame:
    """
    Generate exactly N lab concepts (LAB1, LAB2, ..., LABN) for each patient.
    Patient risk is determined by nested polynomial evaluation with multiplicative noise > threshold.
    Individual lab values are clean (no noise), but noise is applied to the nested polynomial result.

    Args:
        pids_list: List of patient IDs
        threshold: Threshold for nested polynomial evaluation > threshold equation
        num_labs: Number of labs per patient
        lab_value_info: Dictionary containing lab distribution information
        coefficients: Dictionary of polynomial coefficients for all nested polynomials
        num_nested_polys: Number of nested polynomial functions
        poly_degree: Degree of each polynomial
        noise_level: Amount of multiplicative noise to add to the nested polynomial result

    Returns:
        pd.DataFrame: DataFrame containing PID, CONCEPT, and RESULT columns
    """
    records = []
    patient_risk_map = {}

    for pid in pids_list:
        # Generate exactly N CLEAN lab values for each patient (no noise on individual labs)
        clean_lab_values = []
        lab_concepts = []
        
        for i in range(1, num_labs + 1):
            lab_concept = f"S/LAB{i}"
            clean_lab_value = generate_lab_value(lab_concept, lab_value_info, noise_level=0.0)  # No noise on individual labs
            if clean_lab_value is not None:
                clean_lab_values.append(clean_lab_value)
                lab_concepts.append(lab_concept)
        
        if len(clean_lab_values) == num_labs:  # Ensure we have all labs
            # Calculate clean nested polynomial evaluation
            clean_nested_poly_result = evaluate_nested_polynomials(clean_lab_values, coefficients, num_labs, num_nested_polys, poly_degree)
            
            # Add multiplicative noise to the nested polynomial result, then determine risk
            if noise_level > 0:
                # Multiplicative noise
                noise_factor = np.random.normal(1.0, noise_level)
                noisy_nested_poly_result = clean_nested_poly_result * noise_factor
            else:
                noisy_nested_poly_result = clean_nested_poly_result
                
            is_high_risk = noisy_nested_poly_result > threshold
                            
            patient_risk_map[pid] = is_high_risk
            condition = "high_risk" if is_high_risk else "low_risk"

            # Add all lab records using CLEAN values (what the model will see)
            for i, (lab_concept, clean_lab_value) in enumerate(zip(lab_concepts, clean_lab_values)):
                records.append({
                    "PID": pid, 
                    "CONCEPT": lab_concept, 
                    "RESULT": clean_lab_value,
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
    coefficients: Dict[str, Dict[str, float]],
    num_nested_polys: int,
    poly_degree: int,
    diag_min_days: int = DIAG_MIN_DAYS,
    diag_max_days: int = DIAG_MAX_DAYS,
    noise_level: float = NOISE_LEVEL,
    patient_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Generate synthetic data with exactly N lab values per patient (LAB1, LAB2, ..., LABN).
    Patient risk is determined by nested polynomial evaluation with multiplicative noise > threshold.
    Noise is applied to the nested polynomial result, not to individual lab values.
    
    Args:
        input_data: DataFrame containing existing synthetic data with patient assignments
        threshold: Threshold for nested polynomial evaluation > threshold equation
        num_labs: Number of labs per patient
        lab_value_info: Dictionary containing lab distribution information
        coefficients: Dictionary of polynomial coefficients for all nested polynomials
        num_nested_polys: Number of nested polynomial functions
        poly_degree: Degree of each polynomial
        diag_min_days: Minimum days after last lab for diagnosis
        diag_max_days: Maximum days after last lab for diagnosis
        noise_level: Amount of multiplicative noise to add to the nested polynomial result
        
    Returns:
        pd.DataFrame: Generated synthetic data
    """
    # Get patient IDs from input data
    pids_list = list(input_data["subject_id"].unique())
    
    # Generate concepts and lab values
    concepts_data = generate_n_lab_concepts(pids_list, threshold, num_labs, lab_value_info, coefficients, num_nested_polys, poly_degree, noise_level)

    # Create final DataFrame - match multi_lab_logistic.py structure exactly
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


def print_statistics(data: pd.DataFrame, num_labs: int, coefficients: Dict[str, Dict[str, float]], 
                    num_nested_polys: int, poly_degree: int) -> None:
    """
    Print statistics about the lab values and nested polynomial coefficients.

    Args:
        data: DataFrame containing the synthetic data
        num_labs: Number of labs per patient
        coefficients: Dictionary of polynomial coefficients for all nested polynomials
        num_nested_polys: Number of nested polynomial functions
        poly_degree: Degree of each polynomial
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
    
    # Print nested polynomial coefficients
    print(f"\nNested polynomial coefficients:")
    for poly_name, poly_coeffs in coefficients.items():
        print(f"  {poly_name}:")
        for coef_name, coef_value in sorted(poly_coeffs.items()):
            print(f"    {coef_name}: {coef_value:.3f}")
    
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


def calculate_theoretical_performance(data: pd.DataFrame, num_labs: int, coefficients: Dict[str, Dict[str, float]], 
                                    num_nested_polys: int, poly_degree: int) -> dict:
    """
    Calculate the theoretical performance of the model based on nested polynomial equation.

    Args:
        data: DataFrame containing the synthetic data
        num_labs: Number of labs per patient
        coefficients: Dictionary of polynomial coefficients for all nested polynomials
        num_nested_polys: Number of nested polynomial functions
        poly_degree: Degree of each polynomial
        
    Returns:
        dict: Dictionary containing performance metrics
    """
    # Calculate nested polynomial-based AUC
    nested_poly_auc = calculate_nested_polynomial_auc(data, num_labs, coefficients, num_nested_polys, poly_degree)
    
    # Calculate other metrics
    sweep_auc = sweep_threshold_auc(data)
    scipy_mann_whitney_u_auc = scipy_mann_whitney_u(data)
    cohens_d_metric = cohens_d(data)
    
    print("\nTheoretical performance:")
    print(f"Nested polynomial-based AUC (clean nested poly vs noisy ground truth): {nested_poly_auc}")
    print(f"Sweep AUC (LAB1 only): {sweep_auc}")
    print(f"Scipy Mann-Whitney U (LAB1 only): {scipy_mann_whitney_u_auc}")
    print(f"Cohen's d (LAB1 only): {cohens_d_metric}")
    
    return {
        "nested_poly_auc": nested_poly_auc,
        "sweep_auc": sweep_auc,
        "scipy_mann_whitney_u_auc": scipy_mann_whitney_u_auc,
        "cohens_d_metric": cohens_d_metric,
    }


def calculate_nested_polynomial_auc(data: pd.DataFrame, num_labs: int, coefficients: Dict[str, Dict[str, float]], 
                                  num_nested_polys: int, poly_degree: int) -> float:
    """
    Calculate AUC for detecting high-risk patients based on nested polynomial evaluation.
    
    Note: This calculates the theoretical maximum AUC that a perfect model could achieve
    using the clean lab values to predict the noisy ground truth labels.
    
    Args:
        data: DataFrame containing the synthetic data
        num_labs: Number of labs per patient
        coefficients: Dictionary of polynomial coefficients for all nested polynomials
        num_nested_polys: Number of nested polynomial functions
        poly_degree: Degree of each polynomial
        
    Returns:
        float: AUC for nested polynomial-based detection
    """
    # Get lab values for each patient (these are clean, no noise)
    lab_data_dict = {}
    for i in range(1, num_labs + 1):
        lab_data_dict[f"LAB{i}"] = data[data['code'] == f'S/LAB{i}'].groupby("subject_id")["numeric_value"].first()
    
    # Get patient risks (these are based on noisy nested polynomial + threshold)
    positive_patients = set(data[data["code"] == "S/DIAG_POSITIVE"]["subject_id"].unique())
    patient_risks = list(lab_data_dict.values())[0].index.isin(positive_patients)
    
    # Calculate actual nested polynomial scores for each patient
    nested_poly_scores = []
    for patient_id in list(lab_data_dict.values())[0].index:
        lab_values = [lab_data_dict[f"LAB{i}"][patient_id] for i in range(1, num_labs + 1)]
        nested_poly_score = evaluate_nested_polynomials(lab_values, coefficients, num_labs, num_nested_polys, poly_degree)
        nested_poly_scores.append(nested_poly_score)
    
    nested_poly_scores = np.array(nested_poly_scores)
    
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(patient_risks, nested_poly_scores)
    return auc


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data with exactly N lab values per patient where positive patients are determined by nested polynomial evaluation with multiplicative noise > threshold"
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
        "--num_nested_polys",
        type=int,
        default=NUM_NESTED_POLYS,
        help="Number of nested polynomial functions",
    )
    parser.add_argument(
        "--poly_degree",
        type=int,
        default=POLY_DEGREE,
        help="Degree of each polynomial (1=linear, 2=quadratic, etc.)",
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
        help="Multiplicative noise level for the nested polynomial result (0.0 = no noise, 0.1 = 10% noise)",
    )
    parser.add_argument(
        "--patient_info_path",
        type=str,
        default=PATIENTS_INFO_PATH,
        help="Path to patient information parquet file (optional, for realistic birth/death dates)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for coefficient generation (for reproducibility)",
    )
    parser.add_argument(
        "--positive_rate",
        type=float,
        default=POSITIVE_RATE,
        help="Target positive rate (fraction of patients that should be positive, e.g., 0.5 for 50%)",
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

    # Generate nested polynomial coefficients
    coefficients = generate_nested_polynomial_coefficients(args.num_labs, args.num_nested_polys, args.poly_degree, args.seed)
    
    # Print the nested polynomial equation
    print_nested_polynomial_equation(coefficients, args.num_labs, args.num_nested_polys, args.poly_degree)
    
    # Generate lab value info for threshold calculation
    lab_value_info = {}
    for i in range(1, args.num_labs + 1):
        lab_value_info[f"S/LAB{i}"] = {
            "distribution": {
                "dist": "normal",
                "mean": LAB_MEAN,
                "std": LAB_STD,
            },
        }
    
    # Find optimal threshold for target positive rate
    pids_list = list(input_data["subject_id"].unique())
    optimal_threshold = find_optimal_threshold(pids_list, args.num_labs, lab_value_info, coefficients, args.num_nested_polys, args.poly_degree, args.positive_rate)
    
    print(f"\nGenerating synthetic data with:")
    print(f"  - {input_data['subject_id'].nunique()} patients")
    print(f"  - Each patient gets exactly {args.num_labs} lab values (LAB1, LAB2, ..., LAB{args.num_labs})")
    print(f"  - All labs use same distribution: mean={LAB_MEAN}, std={LAB_STD}")
    print(f"  - {args.num_nested_polys} nested polynomial functions of degree {args.poly_degree}")
    print(f"  - Optimal threshold for {args.positive_rate*100:.0f}% positive rate: {optimal_threshold:.6f}")
    print(f"  - Individual lab values are clean (no noise)")
    print(f"  - Multiplicative noise ({args.noise_level*100:.0f}%) is applied to the nested polynomial result")
    print(f"  - Diagnosis timing: {args.diag_min_days}-{args.diag_max_days} days after last lab")

    # Generate save name dynamically
    noise_suffix = f"_noise{int(args.noise_level*100)}" if args.noise_level > 0 else ""
    seed_suffix = f"_seed{args.seed}" if args.seed is not None else ""
    save_name = f"n_lab_nested_poly_{args.num_labs}labs_{args.num_nested_polys}polys_deg{args.poly_degree}_mean{int(LAB_MEAN*100)}p{int(LAB_STD*100)}{noise_suffix}{seed_suffix}_n{N}"

    # Generate synthetic data using optimal threshold
    data = generate_synthetic_data(
        input_data,
        optimal_threshold,  # Use optimal threshold instead of args.threshold
        args.num_labs,
        lab_value_info,
        coefficients,
        args.num_nested_polys,
        args.poly_degree,
        args.diag_min_days,
        args.diag_max_days,
        args.noise_level,
        patient_df
    )

    print("\nGenerated data:")
    print(data.head())

    # Print statistics
    print_statistics(data, args.num_labs, coefficients, args.num_nested_polys, args.poly_degree)

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
    performance_metrics = calculate_theoretical_performance(data, args.num_labs, coefficients, args.num_nested_polys, args.poly_degree)

if __name__ == "__main__":
    main()
