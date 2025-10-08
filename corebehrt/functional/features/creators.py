import uuid
import warnings

import numpy as np
import pandas as pd

from corebehrt.constants.data import (
    ABSPOS_COL,
    ADMISSION_CODE,
    BIRTH_CODE,
    BIRTHDATE_COL,
    CONCEPT_COL,
    DEATH_CODE,
    DEATHDATE_COL,
    DISCHARGE_CODE,
    PID_COL,
    SEGMENT_COL,
    TIMESTAMP_COL,
)
from corebehrt.functional.utils.time import get_hours_since_epoch


def create_abspos(concepts: pd.DataFrame) -> pd.DataFrame:
    """
    Assign absolute position in hours since origin point to each row in concepts.
    Parameters:
        concepts: concepts with 'TIMESTAMP' column.
    Returns:
        concepts with a new 'abspos' column
    """
    concepts[ABSPOS_COL] = get_hours_since_epoch(concepts[TIMESTAMP_COL])
    return concepts


def create_age_in_years(concepts: pd.DataFrame) -> pd.DataFrame:
    """
    Compute age in years for each row in concepts
    Parameters:
        concepts: concepts with 'time' and 'birthdate' columns.
    Returns:
        pd.DataFrame: concepts with a new 'age' column
    """
    # Try to convert columns to datetime if they aren't already
    if not pd.api.types.is_datetime64_any_dtype(concepts[TIMESTAMP_COL]):
        print(f"\nConverting {TIMESTAMP_COL} to datetime...")
        concepts[TIMESTAMP_COL] = pd.to_datetime(
            concepts[TIMESTAMP_COL], errors="coerce"
        )

    if not pd.api.types.is_datetime64_any_dtype(concepts[BIRTHDATE_COL]):
        print(f"\nConverting {BIRTHDATE_COL} to datetime...")
        concepts[BIRTHDATE_COL] = pd.to_datetime(
            concepts[BIRTHDATE_COL], errors="coerce"
        )

    # Calculate age
    concepts["age"] = (
        concepts[TIMESTAMP_COL] - concepts[BIRTHDATE_COL]
    ).dt.days // 365.25

    return concepts


def _create_patient_info(concepts: pd.DataFrame) -> pd.DataFrame:
    """
    Create patient information DataFrame from concepts.

    Args:
        concepts: DataFrame with patient concepts

    Returns:
        DataFrame with patient information including birthdate, deathdate, and background variables
    """
    # Get unique patients
    patients = concepts[PID_COL].unique()

    # Initialize patient info - handle empty case
    patient_info = pd.DataFrame({PID_COL: patients})

    # If no patients, return empty DataFrame with proper structure
    if len(patients) == 0:
        warnings.warn("No patients found in concepts")
        patient_info[BIRTHDATE_COL] = pd.Series([], dtype="datetime64[ns]")
        patient_info[DEATHDATE_COL] = pd.Series([], dtype="datetime64[ns]")
        return patient_info

    # Fallback: extract from DOB codes
    dob_data = concepts[concepts[CONCEPT_COL] == BIRTH_CODE]
    birthdate_map = dict(zip(dob_data[PID_COL], dob_data[TIMESTAMP_COL]))
    patient_info[BIRTHDATE_COL] = patient_info[PID_COL].map(birthdate_map)

    # Extract death dates (DOD)
    dod_data = concepts[concepts[CONCEPT_COL] == DEATH_CODE]
    deathdate_map = dict(zip(dod_data[PID_COL], dod_data[TIMESTAMP_COL]))
    patient_info[DEATHDATE_COL] = patient_info[PID_COL].map(deathdate_map)

    # Extract background variables (those that start with BG_)
    bg_concepts = concepts[concepts[CONCEPT_COL].str.startswith("BG_", na=False)]

    # Process background concepts if they exist
    if not bg_concepts.empty:
        bg_info = bg_concepts[[PID_COL, CONCEPT_COL]].copy()

        # Split BG_ concepts into column_name and value, handling cases without "//"
        split_result = bg_info[CONCEPT_COL].str.split("//", expand=True)

        # Ensure we always have at least 2 columns
        if split_result.shape[1] == 1:
            # No "//" separator found, add empty value column
            split_result[1] = None

        bg_info["column_name"] = split_result[0]
        bg_info["value"] = split_result[1]

        # Remove BG_ prefix from column names
        bg_info["column_name"] = bg_info["column_name"].str.replace("BG_", "")

        # Filter out rows without proper column names or with empty column names after cleaning
        bg_info = bg_info[
            bg_info["column_name"].notna() & (bg_info["column_name"] != "")
        ]

        if not bg_info.empty:
            # Create pivot table for background variables
            bg_info_pivot = bg_info.pivot_table(
                index=PID_COL, columns="column_name", values="value", aggfunc="first"
            ).reset_index()

            # Merge with patient_info
            patient_info = pd.merge(patient_info, bg_info_pivot, on=PID_COL, how="left")

    return patient_info


def create_background(concepts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create background concepts for each patient based on the static background variables in the dataframe.
    Sets the time of the background concepts to the birthdate of the patient.
    Expects 'DOB' concept to be present in the patients_info DataFrame.

    Args:
        concepts: DataFrame with columns 'subject_id', 'time', 'code'

    Returns:
        tuple: (updated_concepts_df, patient_info_df)
            - updated_concepts_df: concepts with background concepts updated and birthdate column added
            - patient_info_df: patient information with birthdate, deathdate, and background variables
    """
    # Create a copy to avoid modifying the original DataFrame
    concepts = concepts.copy()

    # Extract birthdates from DOB rows
    dob_rows = concepts[concepts[CONCEPT_COL] == BIRTH_CODE]
    birthdates = dict(zip(dob_rows[PID_COL], dob_rows[TIMESTAMP_COL]))
    concepts[BIRTHDATE_COL] = concepts[PID_COL].map(birthdates)
    
    # Exclude patients without birthdate instead of raising an error
    patients_with_dob = concepts[BIRTHDATE_COL].notna()
    if not patients_with_dob.all():
        excluded_patients = concepts[~patients_with_dob][PID_COL].unique()
        print(f"Warning: Excluding {len(excluded_patients)} patients without birthdate: {excluded_patients}")
        concepts = concepts[patients_with_dob].copy()

    # Use boolean masking instead of index-based selection for background rows
    bg_mask = concepts[TIMESTAMP_COL].isna()
    concepts.loc[bg_mask, TIMESTAMP_COL] = concepts.loc[bg_mask, BIRTHDATE_COL]
    concepts.loc[bg_mask, CONCEPT_COL] = "BG_" + concepts.loc[bg_mask, CONCEPT_COL]
    concepts.loc[bg_mask, SEGMENT_COL] = 0

    # Use boolean masking for admission/discharge rows
    adm_mask = concepts[CONCEPT_COL].str.contains(ADMISSION_CODE, na=False) | concepts[
        CONCEPT_COL
    ].str.contains(DISCHARGE_CODE, na=False)
    concepts.loc[adm_mask, CONCEPT_COL] = "ADM_" + concepts.loc[adm_mask, CONCEPT_COL]

    # Get the patient info
    patient_info = _create_patient_info(concepts)
    return concepts, patient_info


def assign_index_and_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign 'index' and 'order' columns to ensure correct ordering.
    - The 'index' column represents the position of each row within its partition.
    - The 'order' column can be used for additional custom ordering if needed.
    - Both columns are initialized with 0 to ensure consistent behavior across partitions.
    Parameters:
        df: pd.DataFrame with 'PID' column.
    Returns:
        df with 'index' and 'order' columns.
    """
    if "index" in df.columns and "order" in df.columns:
        df.loc[:, "index"] = df["index"].fillna(0)
        df.loc[:, "order"] = df["order"].fillna(0)
    return df


def sort_features(concepts: pd.DataFrame) -> pd.DataFrame:
    """
    Sorting all concepts by 'subject_id' and 'abspos' (and 'index' and 'order' if they exist).
    """
    if "index" in concepts.columns and "order" in concepts.columns:
        concepts = concepts.sort_values(
            [PID_COL, ABSPOS_COL, "index", "order"]
        )  # could maybe be done more optimally, is a bit slow
        concepts = concepts.drop(columns=["index", "order"])
    else:
        concepts = concepts.sort_values([PID_COL, ABSPOS_COL])

    return concepts