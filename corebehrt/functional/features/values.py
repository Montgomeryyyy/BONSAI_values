import os
import pandas as pd
from corebehrt.constants.data import CONCEPT_COL, VALUE_COL


def power_bin_d_sig_mean(n_unique: int) -> int:
    """
    Bins the values in a series into num_bins bins. Expects the values to be normalised.
    """
    bins = int(3.25*n_unique**0.195)
    return bins

def power_bin_distance_mean(n_unique: int) -> int:
    """
    Bins the values in a series into num_bins bins. Expects the values to be normalised.
    """
    bins = int(4.85*n_unique**0.163)
    return bins


def get_unique_value_counts(features_path: str, splits: list) -> dict:
    """
    Gets the number of unique numeric values for each concept across all shards.
    
    Args:
        features_path: Path to the features directory
        splits: List of split names (e.g., ['train', 'tuning', 'held_out'])
        
    Returns:
        Dictionary mapping concept names to number of unique numeric values
    """
    concept_unique_counts = {}
    
    for split in splits:
        path_name = os.path.join(features_path, split)
                
        for shard in os.listdir(path_name):
            if shard.endswith('.parquet'):
                shard_path = os.path.join(path_name, shard)
                df = pd.read_parquet(shard_path)
                
                # Get rows with non-NaN numeric values
                value_df = df[df[VALUE_COL].notna()]
                
                # Group by concept and count unique values
                for concept, group in value_df.groupby(CONCEPT_COL):
                    unique_values = group[VALUE_COL].nunique()
                    
                    if concept in concept_unique_counts:
                        # Sum across shards
                        concept_unique_counts[concept] += unique_values
                    else:
                        concept_unique_counts[concept] = unique_values
                    
    # Print summary of top 20 concepts by unique value count
    print(f"\nTop 20 concepts by unique value count:")
    for concept, count in sorted(concept_unique_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {concept}: {count} unique values")
    
    return concept_unique_counts