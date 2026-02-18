# Synthetic Lab Data Generation Scripts

This directory contains two scripts for generating synthetic medical data with correlated lab values:

## 1. `simulate_synthetic_labs.py` (Original - Probability-based)

**Purpose**: Generates synthetic data where the ratio of high/low lab values is controlled by probabilities.

**Key Features**:
- Uses `condition_probabilities` in `CONCEPT_RELATIONSHIPS` to specify the ratio
- Default: 50% high, 50% low (configurable via probabilities)
- Random assignment of conditions to patients

**Usage**:
```bash
python simulate_synthetic_labs.py --write_dir /path/to/output --patients_info_path /path/to/patient_info.parquet
```

**Configuration**:
```python
CONCEPT_RELATIONSHIPS = {
    "S/LAB1": {
        "condition_probabilities": {
            "high": 0.5,  # 50% chance of being high
            "low": 0.5,   # 50% chance of being low
        },
        # ... rest of configuration
    }
}
```

## 2. `simulate_synthetic_labs_fixed_n.py` (New - Fixed Number)

**Purpose**: Generates synthetic data where you specify the exact number of patients that should have high lab values and low lab values. The patient dataset is filtered to only include these patients.

**Key Features**:
- Uses `--n_high_patients` and `--n_low_patients` parameters to specify exact counts
- Deterministic assignment: exactly N patients get high values, M patients get low values
- Random selection of which specific patients get high vs low values
- Patient dataset is filtered to only include patients that will get lab values (n_high + n_low total patients)

**Usage**:
```bash
python simulate_synthetic_labs_fixed_n.py --n_high_patients 3000 --n_low_patients 4000 --write_dir /path/to/output --patients_info_path /path/to/patient_info.parquet
```

**Parameters**:
- `--n_high_patients`: Number of patients that should have high lab values (default: 5000)
- `--n_low_patients`: Number of patients that should have low lab values (default: 5000)
- `--write_dir`: Output directory (default: `../../data/vals/synthetic_data/10000n/`)
- `--patients_info_path`: Path to patient information file (default: `../../data/vals/patient_infos/patient_info_10000n.parquet`)

## Key Differences

| Aspect | Original Script | Fixed N Script |
|--------|----------------|----------------|
| Control Method | Probability-based ratio | Exact counts for both high and low |
| Predictability | Variable results due to randomness | Exact number of high and low patients |
| Use Case | When you want a specific ratio | When you need exactly N high and M low patients |
| Configuration | Modify probabilities in code | Use command line parameters |
| Patient Exclusion | All patients get labs | Dataset filtered to only include patients with labs |

## Example Scenarios

**Use Original Script When**:
- You want approximately 30% high, 70% low patients
- The exact count doesn't matter, only the ratio
- You're doing exploratory analysis

**Use Fixed N Script When**:
- You need exactly 3000 high patients and 4000 low patients (7000 total)
- You want to work with a filtered dataset containing only patients with lab values
- You're doing controlled experiments with specific counts
- You need reproducible results with exact numbers

## Output

Both scripts generate the same output format:
- CSV file with columns: `subject_id`, `code`, `numeric_value`, `time`
- Lab values: 1 for high, 0 for low
- Diagnoses: `S/DIAG_POSITIVE` for high lab patients, `S/DIAG_NEGATIVE` for low lab patients
- Timestamps ensuring labs come before diagnoses
