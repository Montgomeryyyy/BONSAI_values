import pandas as pd
from corebehrt.constants.data import CONCEPT_COL


class ValueCreator:
    """
    A class to load normalise values in data frames.
    Expects a 'result' column and 'concept' column to be present.
    """

    @staticmethod
    def bin_results(
        concepts: pd.DataFrame,
        num_bins=100,
        bin_mapping:dict=None,
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
            concepts["binned_value"] = concepts.groupby(CONCEPT_COL).apply(
                lambda group: ValueCreator.bin(
                    group["numeric_value"], 
                    num_bins=bin_mapping.get(group[CONCEPT_COL].iloc[0], num_bins)
                ) if group["numeric_value"].notna().any() 
                else pd.Series([None] * len(group), index=group.index)
            ).reset_index(level=0, drop=True)
        else:
            concepts["binned_value"] = ValueCreator.bin(
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
        normalized_values = pd.to_numeric(normalized_values, errors="coerce")
        val_mask = normalized_values.notna()
        
        # Calculate actual number of bins
        if callable(num_bins):
            # Count unique non-null values
            unique_count = normalized_values[val_mask].nunique()
            actual_num_bins = num_bins(unique_count)
        else:
            actual_num_bins = num_bins
        
        normalized_values[val_mask] = normalized_values[val_mask].mul(actual_num_bins)
        normalized_values = normalized_values.astype(object)
        normalized_values[val_mask] = (
            normalized_values[val_mask].astype(int).astype(str)
        )
        normalized_values[val_mask] = "VAL_" + normalized_values[val_mask]
        return normalized_values
