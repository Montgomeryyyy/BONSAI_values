import pandas as pd

from corebehrt.constants.data import (
    CLS_TOKEN,
    SEP_TOKEN,
    SPECIAL_TOKEN_ABSPOS_ADJUSTMENT,
    UNKNOWN_TOKEN,
    PID_COL,
    CONCEPT_COL,
    SEGMENT_COL,
    ABSPOS_COL,
    VAL_TOKEN,
)


def add_special_tokens_partition(
    df: pd.DataFrame, add_sep=True, add_cls=True
) -> pd.DataFrame:
    """
    Efficiently add special tokens to a partition without full sorting.
    PID is assumed to be the index.

    cls token will be added before earliest abspos for each PID
    sep token will be added at segment changes, adjacent to the last event of the previous segment
    """
    special_rows = []

    if add_cls:
        # Find indices of the earliest event for each PID
        cls_rows = df.groupby(PID_COL).first()
        # Create [CLS] rows
        cls_rows[CONCEPT_COL] = CLS_TOKEN
        cls_rows[ABSPOS_COL] -= (
            SPECIAL_TOKEN_ABSPOS_ADJUSTMENT  # Adjust position to come before earliest event
        )
        cls_rows[SEGMENT_COL] = 0
        special_rows.append(cls_rows)

    if add_sep:
        # Find segment changes within same PID
        df = df.sort_values([PID_COL, ABSPOS_COL, SEGMENT_COL])
        pid_series = df.index.to_series()
        segment_changes = (df[SEGMENT_COL] != df[SEGMENT_COL].shift(-1)) & (
            pid_series == pid_series.shift(-1)
        )
        sep_rows = df[segment_changes].copy()
        sep_rows[CONCEPT_COL] = SEP_TOKEN
        sep_rows[ABSPOS_COL] += (
            SPECIAL_TOKEN_ABSPOS_ADJUSTMENT  # Adjust position slightly
        )
        special_rows.append(sep_rows)

    # Combine all rows and sort by 'PID' and 'abspos'
    if special_rows:
        df = pd.concat([df] + special_rows, ignore_index=False)
        df = df.sort_values([PID_COL, ABSPOS_COL])

    return df


def tokenize_partition(series: pd.Series, vocabulary: dict) -> tuple[pd.Series, pd.Series]:
    """Optimized in-partition tokenization using direct dictionary mapping.
    
    Returns:
        tuple: (tokenized_series, values_series)
        - tokenized_series: Series with tokens (VALUE_TOKEN for floats, regular tokens for strings)
        - values_series: Series with actual float values (NaN for non-floats)
    """
    # Create mask for float values
    value_mask = pd.to_numeric(series, errors='coerce').notna()
    unk_token = vocabulary[UNKNOWN_TOKEN]
    
    # Get the VALUE token ID
    value_token = vocabulary[VAL_TOKEN]
    
    # Initialize result with original values
    result = series.copy()
    values_result = pd.Series([float('nan')] * len(series), index=series.index)
    
    # Handle float values - replace with VALUE token and store actual values
    if value_mask.any():
        result[value_mask] = value_token
        values_result[value_mask] = series[value_mask].astype(float)
    
    # Tokenize non-float values
    non_float_mask = ~value_mask
    if non_float_mask.any():
        # Direct mapping with fillna for unknown tokens (only for non-float values)
        tokenized_values = series[non_float_mask].map(vocabulary).fillna(unk_token).astype(int)
        result[non_float_mask] = tokenized_values
    
    return result, values_result


def limit_concept_length_partition(series: pd.Series, cutoffs: dict) -> pd.Series:
    """Efficiently limit concept lengths within a partition.

    Args:
        series: Pandas Series containing concepts
        cutoffs: Dict mapping prefixes to max lengths, e.g. {'D': 6, 'M': 4}
            Will limit concepts starting with 'D' to 6 chars, 'M' to 4 chars.

    Example:
        With cutoffs={'D': 4}, 'D123456' becomes 'D1234'
    """
    # Create a copy to avoid modifying original
    result = series.copy()

    # Vectorized operations for each prefix
    for prefix, length in cutoffs.items():
        # Create mask for matching prefix
        mask = result.str.startswith(prefix)

        # Apply length limit only where mask is True
        if mask.any():
            result.loc[mask] = result.loc[mask].str[:length]

    return result
