import pandas as pd
<<<<<<< HEAD
import numpy as np
from corebehrt.constants.data import CONCEPT_COL, VALUE_COL, VAL_TOKEN

import logging

logger = logging.getLogger(__name__)


def _safe_convert_to_numeric(val):
    """
    Safely convert a value to numeric (float).

    Args:
        val: Value to convert (can be int, float, str, or NaN)

    Returns:
        float value, pd.NA if conversion fails, or original value if already NaN
    """
    if pd.isna(val):
        return val
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except (ValueError, TypeError):
            return pd.NA
    return pd.NA
=======
from corebehrt.constants.data import CONCEPT_COL, VALUE_COL
>>>>>>> c73ff1e5 (added numeric embedding layer)


class ValueCreator:
    """
    A class to load normalise values in data frames.
    Expects a 'result' column and 'concept' column to be present.
    """

    @staticmethod
<<<<<<< HEAD
    def add_values(
=======
    def add_null_token(concepts: pd.DataFrame, null_token: int) -> pd.DataFrame:
        concepts[VALUE_COL] = pd.to_numeric(concepts[VALUE_COL], errors="coerce")
        concepts[VALUE_COL] = concepts[VALUE_COL].fillna(null_token)
        return concepts

    @staticmethod
    def bin_results(
>>>>>>> c73ff1e5 (added numeric embedding layer)
        concepts: pd.DataFrame,
        bin_values: bool = False,
        bin_mapping: dict = None,
        num_bins: int = 10,
        value_type: str = "discrete",
        add_prefix: bool = False,
        separator_regex: str = None,
    ) -> pd.DataFrame:
        """
        Add values to concepts DataFrame. Routes to discrete or continuous implementation based on value_type.

        Args:
            concepts: DataFrame with VALUE_COL to process
            bin_values: Whether to bin values
            bin_mapping: Optional dict mapping concepts to their specific number of bins
            num_bins: Default number of bins to use
            value_type: "discrete" or "continuous"
            add_prefix: Whether to add prefix to codes (only for discrete)
            separator_regex: Regex pattern to extract prefix from concept column (only for discrete)

        Returns:
            DataFrame with processed values
        """
        if value_type == "discrete":
            return ValueCreator.add_values_discrete(
                concepts, bin_values, bin_mapping, num_bins, add_prefix, separator_regex
            )
<<<<<<< HEAD
        elif value_type == "combined":
            return ValueCreator.add_values_continuous(
                concepts, bin_values, bin_mapping, num_bins
            )
        else:
            raise ValueError(f"Unsupported value type: {value_type}")

    @staticmethod
    def add_values_discrete(
        concepts: pd.DataFrame,
        bin_values: bool = False,
        bin_mapping: dict = None,
        num_bins: int = 10,
        add_prefix: bool = False,
        separator_regex: str = None,
    ) -> pd.DataFrame:
        """
        Add values for discrete mode (returns strings with "VAL_" prefix).

        Args:
            concepts: DataFrame with VALUE_COL to process
            bin_values: Whether to bin values
            bin_mapping: Optional dict mapping concepts to their specific number of bins
            num_bins: Default number of bins to use
            add_prefix: Whether to add prefix to codes
            separator_regex: Regex pattern to extract prefix from concept column

        Returns:
            DataFrame with processed values in CONCEPT_COL. VALUE_COL dropped.
        """
        concepts[VALUE_COL] = concepts[VALUE_COL].astype(float)

        # Bin values if bin_values is True (always returns integers)
        concepts = ValueCreator._apply_binning(
            concepts, bin_values, bin_mapping, num_bins
=======
        concepts["binned_value"] = ValueCreator.bin(
            concepts[VALUE_COL], num_bins=num_bins
>>>>>>> c73ff1e5 (added numeric embedding layer)
        )

        # Discretise values (converts integers to "VAL_X" strings)
        concepts[VALUE_COL] = ValueCreator._discretise(concepts[VALUE_COL])

        # Create values dataframe structure
        concepts, values = ValueCreator._create_values_dataframe(concepts)

        if not values.empty:
            # Store original concept for prefix extraction if needed
            original_concepts = values[CONCEPT_COL].copy()
            # Add code column with optional prefix extraction
            values = ValueCreator._add_code_column_discrete(
                values, original_concepts, add_prefix, separator_regex
            )
            concatted = pd.concat([concepts, values], ignore_index=True)
        else:
            concatted = concepts

        return concatted.drop(columns=[VALUE_COL])

    @staticmethod
    def add_values_continuous(
        concepts: pd.DataFrame,
        bin_values: bool = False,
        bin_mapping: dict = None,
        num_bins: int = 10,
    ) -> pd.DataFrame:
        """
        Add values for continuous mode (returns integers).

        Args:
            concepts: DataFrame with VALUE_COL to process
            bin_values: Whether to bin values
            bin_mapping: Optional dict mapping concepts to their specific number of bins
            num_bins: Default number of bins to use

        Returns:
            DataFrame with processed values
        """
        concepts[VALUE_COL] = concepts[VALUE_COL].astype(float)

        # Bin values if bin_values is True (always returns integers)
        concepts = ValueCreator._apply_binning(
            concepts, bin_values, bin_mapping, num_bins
        )

        # Create values dataframe structure
        concepts, values = ValueCreator._create_values_dataframe(
            concepts, add_val_token=True
        )

        if not values.empty:
            concatted = pd.concat([concepts, values], ignore_index=True)
        else:
            concatted = concepts

        return concatted

    @staticmethod
    def bin(values: pd.Series, num_bins=100) -> pd.Series:
        """
        Bins the values in a series into num_bins bins. Expects the values to be normalised.
        Always returns integers (discretisation happens separately).

        Args:
            values: Series of normalized values to bin
            num_bins: Either an integer specifying the number of bins, or a function that takes
                     the number of unique values and returns the number of bins to use.
                     Default is 100.

        Returns:
            Series with binned values as integers
        """
        # Make a copy to avoid modifying the original
        result = values.copy()

        # Convert to numeric - strings will become NaN (ignored)
        # For object dtype, explicitly convert each value to handle strings properly
        if result.dtype == "object":
            numeric_values = result.apply(_safe_convert_to_numeric)
            numeric_values = pd.to_numeric(numeric_values, errors="coerce")
        else:
            # For numeric types, use pd.to_numeric directly
            numeric_values = pd.to_numeric(result, errors="coerce", downcast=None)

        val_mask = numeric_values.notna()

        # Ensure float64 dtype for numeric values
        if val_mask.any():
            numeric_values = numeric_values.astype("float64")

        # Calculate actual number of bins
        if callable(num_bins):
            # Count unique non-null values
            unique_count = numeric_values[val_mask].nunique()
            actual_num_bins = num_bins(unique_count)
        else:
            actual_num_bins = num_bins

        # Clamp values to [0, 1) to ensure we get exactly num_bins bins (0 to num_bins-1)
        if val_mask.any():
            numeric_values[val_mask] = numeric_values[val_mask].clip(0.0, 1.0 - 1e-10)
            # Multiply by number of bins to get bin indices
            numeric_values[val_mask] = numeric_values[val_mask].mul(actual_num_bins)

        # Convert to integers
        numeric_values = numeric_values.astype(object)
        numeric_values[val_mask] = numeric_values[val_mask].astype(int)

        return numeric_values

    @staticmethod
    def _discretise(result: pd.Series) -> pd.Series:
        """
        Converts numeric values to discrete string format with "VAL_" prefix.
        """
        val_mask = result.notna()
        result = result.astype(object)
        # Handle both numeric and string inputs
        if val_mask.any():
            if result[val_mask].dtype == "object":
                # Try to convert strings to int first
                try:
                    result[val_mask] = result[val_mask].astype(int).astype(str)
                except (ValueError, TypeError):
                    # If conversion fails, assume already strings
                    pass
            else:
                result[val_mask] = result[val_mask].astype(int).astype(str)
            result[val_mask] = "VAL_" + result[val_mask]
        return result

    @staticmethod
    def _apply_binning(
        concepts: pd.DataFrame,
        bin_values: bool,
        bin_mapping: dict = None,
        num_bins: int = 10,
    ) -> pd.DataFrame:
        """
        Helper function to apply binning to values in concepts DataFrame.

        Args:
            concepts: DataFrame with VALUE_COL to bin
            bin_values: Whether to bin values
            bin_mapping: Optional dict mapping concepts to their specific number of bins
            num_bins: Default number of bins to use

        Returns:
            DataFrame with binned values (as integers) in VALUE_COL
        """
        if bin_values:
            if bin_mapping is not None:
                concepts[VALUE_COL] = (
                    concepts.groupby(CONCEPT_COL)
                    .apply(
                        lambda group: ValueCreator.bin(
                            group[VALUE_COL],
                            num_bins=bin_mapping.get(
                                group[CONCEPT_COL].iloc[0], num_bins
                            ),
                        )
                        if group[VALUE_COL].notna().any()
                        else pd.Series([None] * len(group), index=group.index)
                    )
                    .reset_index(level=0, drop=True)
                )
            else:
                concepts[VALUE_COL] = ValueCreator.bin(
                    concepts[VALUE_COL], num_bins=num_bins
                )
        return concepts

    @staticmethod
    def _create_values_dataframe(
        concepts: pd.DataFrame, add_val_token: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Helper function to create values dataframe structure from concepts.

        Args:
            concepts: DataFrame with VALUE_COL to extract values from

        Returns:
            Tuple of (concepts_without_values, values_dataframe)
        """
        # Add index and order
        concepts["index"] = concepts.index
        concepts.loc[:, "order"] = 0

        val_mask = concepts[VALUE_COL].notna()
        if val_mask.any():
            values = concepts[val_mask].copy()
            if add_val_token:
                values.loc[:, CONCEPT_COL] = VAL_TOKEN
            values.loc[:, "order"] = 1
            concepts.loc[val_mask, VALUE_COL] = np.nan
            return concepts, values
        else:
            return concepts, pd.DataFrame()

    @staticmethod
    def _add_code_column_discrete(
        values: pd.DataFrame,
        original_concepts: pd.Series,
        add_prefix: bool,
        separator_regex: str,
    ) -> pd.DataFrame:
        """
        Helper function to add code column to values DataFrame for discrete values with optional prefix extraction.

        Args:
            values: DataFrame with VALUE_COL containing the discretised values
            original_concepts: Series with original concept names before conversion to VAL_TOKEN
            add_prefix: Whether to add prefix to codes
            separator_regex: Regex pattern to extract prefix from concept column

        Returns:
            DataFrame with "code" column added.
        """
        if add_prefix and separator_regex is not None:
            values["prefix"] = original_concepts.str.extract(separator_regex)
            # Handle cases where regex doesn't match
            prefix_na_mask = values["prefix"].isna()
            if prefix_na_mask.any():
                values.loc[prefix_na_mask, "prefix"] = "UNK"
            # VALUE_COL contains the discrete values (e.g., "VAL_5")
            values.loc[:, CONCEPT_COL] = (
                values["prefix"] + "/" + values[VALUE_COL].astype(str)
            )
        else:
<<<<<<< HEAD
            # For non-prefixed discrete, use VALUE_COL as code
            values.loc[:, CONCEPT_COL] = values[VALUE_COL].astype(str)
        return values
=======
            values.loc[:, "code"] = values["binned_value"]

        values.loc[:, "order"] = 1
        concatted = pd.concat([concepts, values])

        # Drop columns that are not needed
        columns_to_drop = [VALUE_COL, "binned_value"]
        if add_prefix:
            columns_to_drop.append("prefix")

        return concatted.drop(columns=columns_to_drop, axis=1)

    @staticmethod
    def bin(normalized_values: pd.Series, num_bins=100) -> pd.Series:
        """
        Bins the values in a series into num_bins bins. Expects the values to be normalised.
        """
        normalized_values = pd.to_numeric(normalized_values, errors="coerce")
        val_mask = normalized_values.notna()
        normalized_values[val_mask] = normalized_values[val_mask].mul(num_bins)
        normalized_values = normalized_values.astype(object)
        normalized_values[val_mask] = (
            normalized_values[val_mask].astype(int).astype(str)
        )
        normalized_values[val_mask] = "VAL_" + normalized_values[val_mask]
        return normalized_values
>>>>>>> c73ff1e5 (added numeric embedding layer)
