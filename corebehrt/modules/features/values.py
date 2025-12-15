import pandas as pd
from corebehrt.constants.data import CONCEPT_COL


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


class ValueCreatorDiscrete:
    """
    A class to load normalise values in data frames.
    Expects a 'result' column and 'concept' column to be present.
    """

    @staticmethod
    def bin_results(
        concepts: pd.DataFrame,
        num_bins=100,
        bin_mapping: dict = None,
        add_prefix=False,
        separator_regex=None,
    ) -> pd.DataFrame:
        """
        Bins numeric values in a concepts DataFrame.

        Args:
            concepts: DataFrame containing 'numeric_value' and concept columns to bin
            num_bins: Integer specifying the default number of bins for concepts. Default is 100.
            bin_mapping: Dictionary mapping concept names to their specific number of bins.
                        If a concept is not in the mapping, uses the default num_bins.
            add_prefix: Whether to add prefix to the binned value codes
            separator_regex: Regex pattern to extract prefix from concept column

        Returns:
            DataFrame with binned values and additional metadata columns
        """
        if concepts.empty:
            # Return empty DataFrame with same columns plus the expected new ones
            return concepts.assign(
                index=pd.Series(dtype="int64"),
                order=pd.Series(dtype="int64"),
                code=pd.Series(dtype="object"),
            )

        # Apply binning per concept if bin_mapping is provided
        if bin_mapping is not None:
            concepts["binned_value"] = (
                concepts.groupby(CONCEPT_COL)
                .apply(
                    lambda group: ValueCreatorDiscrete.bin(
                        group["numeric_value"],
                        num_bins=bin_mapping.get(group[CONCEPT_COL].iloc[0], num_bins),
                    )
                    if group["numeric_value"].notna().any()
                    else pd.Series([None] * len(group), index=group.index)
                )
                .reset_index(level=0, drop=True)
            )
        else:
            concepts["binned_value"] = ValueCreatorDiscrete.bin(
                concepts["numeric_value"], num_bins=num_bins
            )

        # Add index + order
        concepts["index"] = concepts.index
        concepts.loc[:, "order"] = 0
        values = concepts.dropna(subset=["binned_value"]).copy()

        # Extract prefix from concept and use it for values codes
        if add_prefix and separator_regex is not None:
            values["prefix"] = values[CONCEPT_COL].str.extract(separator_regex)
            # Handle cases where regex doesn't match
            prefix_na_mask = values["prefix"].isna()
            if prefix_na_mask.any():
                values.loc[prefix_na_mask, "prefix"] = "UNK"
            values.loc[:, "code"] = values["prefix"] + "/" + values["binned_value"]
        else:
            values.loc[:, "code"] = values["binned_value"]

        values.loc[:, "order"] = 1
        concatted = pd.concat([concepts, values])

        # Drop columns that are not needed
        columns_to_drop = ["numeric_value", "binned_value"]
        if add_prefix:
            columns_to_drop.append("prefix")

        return concatted.drop(columns=columns_to_drop, axis=1)

    @staticmethod
    def bin(normalized_values: pd.Series, num_bins=100) -> pd.Series:
        """
        Bins the values in a series into num_bins bins. Expects the values to be normalised.

        Args:
            normalized_values: Series of normalized values to bin
            num_bins: Either an integer specifying the number of bins, or a function that takes
                     the number of unique values and returns the number of bins to use.
                     Default is 100.

        Returns:
            Series with binned values as strings with "VAL_" prefix
        """
        # Make a copy to avoid modifying the original
        result = normalized_values.copy()

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

        # Validate that numeric values are between 0 and 1 (optional check)
        if val_mask.any():
            out_of_range = (numeric_values[val_mask] < 0) | (
                numeric_values[val_mask] > 1
            )
            if out_of_range.any():
                # Values outside [0, 1] are still processed, but this is a warning
                pass

        # Update result with numeric values
        result = numeric_values

        # Calculate actual number of bins
        if callable(num_bins):
            # Count unique non-null values
            unique_count = numeric_values[val_mask].nunique()
            actual_num_bins = num_bins(unique_count)
        else:
            actual_num_bins = num_bins

        # Clamp values to [0, 1) to ensure we get exactly num_bins bins (0 to num_bins-1)
        result[val_mask] = result[val_mask].clip(0.0, 1.0 - 1e-10)

        # Multiply by number of bins to get bin indices
        result[val_mask] = result[val_mask].mul(actual_num_bins)
        result = result.astype(object)
        result[val_mask] = result[val_mask].astype(int).astype(str)
        result[val_mask] = "VAL_" + result[val_mask]
        return result
