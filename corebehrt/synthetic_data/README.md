# Synthetic Data Scripts

This directory contains scripts for creating a base synthetic cohort, building variant datasets from that cohort, and analyzing the resulting datasets.

The intended workflow is:

1. Create a base synthetic cohort with `base_cohort/create_base_synthetic_cohort.py` or `base_cohort/create_base_synthetic_cohort_fixed_counts.py`.
2. Build variant datasets from that base cohort.
3. Use the analysis utilities to inspect separability or compute theoretical performance bounds.

## Recommended Workflow

### 1. Create the base synthetic cohort

Use one of these scripts first:

- `base_cohort/create_base_synthetic_cohort.py`
  - Baseline cohort generator.
  - Creates a simple binary synthetic dataset with `S/LAB1` and a matching positive or negative diagnosis label.
  - Best starting point when you want a canonical patient set and then plan to build other versions later.

- `base_cohort/create_base_synthetic_cohort_fixed_counts.py`
  - Same general baseline setup, but with fixed counts for high-risk and low-risk patients.
  - Useful when class counts must be controlled exactly.

#### Patient info prerequisites

Both base-cohort scripts expect a patient-info parquet file passed via `--patients_info_path`.

Recommended structure:

- One row per patient.
- A `subject_id` column. The scripts rename this internally to `PID`.
- A `deathdate` column. Missing death dates are allowed and are treated as no recorded death date.
- A `birthdate` column is part of the intended patient-info schema, even though the current base-cohort scripts use a fixed start date when generating timestamps.

Additional requirement for `create_base_synthetic_cohort_fixed_counts.py`:

- The parquet file must contain at least `n_high_patients + n_low_patients` patients.

Example:

```bash
python corebehrt/synthetic_data/base_cohort/create_base_synthetic_cohort.py \
  --patients_info_path /path/to/patient_info.parquet \
  --write_dir /path/to/output
```

### 2. Build dataset variants from the base cohort

These scripts should typically be run after a base dataset has already been created.

The key distinction is:

- `modify_*` scripts usually keep the original cohort and update values, labels, or noise properties.
- `multi_lab_*` scripts also belong in this variant-dataset stage, because they still reuse an existing base cohort as the starting patient set even when they redefine the lab-generation rule.

#### Baseline modifiers and perturbations

- `variant_datasets/derive_synthetic_classification_dataset.py`
  - Baseline post-processing script.
  - Keeps the patient/label structure and rewrites lab values from new distributions.
  - This is the main "baseline approach" modifier.

- `variant_datasets/derive_noisy_synthetic_dataset.py`
  - Injects label switching and optional random data removal.
  - Useful for robustness experiments after a clean synthetic dataset has been created.

#### Multi-lab task variants

- `variant_datasets/multi_lab_addition.py`
  - Rebuilds the synthetic lab task so labels depend on the sum of multiple labs.
  - Uses an existing base cohort as the starting patient set.

- `variant_datasets/multi_lab_multiplication.py`
  - Rebuilds the task so labels depend on the product of multiple labs.
  - Uses an existing base cohort as the starting patient set.

- `variant_datasets/multi_lab_logistic.py`
  - Rebuilds the task using a logistic rule over multiple labs.
  - Uses an existing base cohort as the starting patient set.

- `variant_datasets/multi_lab_polynomial.py`
  - Rebuilds the task using a polynomial rule over multiple labs.
  - Uses an existing base cohort as the starting patient set.

- `variant_datasets/multi_lab_nested_poly.py`
  - Rebuilds the task using a nested polynomial rule over multiple labs.
  - Uses an existing base cohort as the starting patient set.

- `variant_datasets/multi_lab_frequency.py`
  - Rebuilds the task using lab-frequency patterns rather than only values.
  - Uses an existing base cohort as the starting patient set.

- `variant_datasets/multi_lab_sharp_edge.py`
  - Rebuilds the task using sharper distribution boundaries or switches.
  - Uses an existing base cohort as the starting patient set.

Examples:

```bash
python corebehrt/synthetic_data/variant_datasets/derive_synthetic_classification_dataset.py \
  --input_file /path/to/base_dataset.csv \
  --write_dir /path/to/output
```

```bash
python corebehrt/synthetic_data/variant_datasets/derive_noisy_synthetic_dataset.py \
  --input_file /path/to/modified_dataset.csv \
  --write_dir /path/to/output \
  --switch_percentage 0.05 \
  --remove_percentage 0.25
```

```bash
python corebehrt/synthetic_data/variant_datasets/multi_lab_addition.py \
  --input_file /path/to/base_dataset.csv \
  --write_dir /path/to/output
```

Use the `multi_lab_*` scripts when the variant dataset should encode a more structured relationship between multiple measurements and the final label while keeping the same underlying patient set.

### 3. Analysis and helper modules

- `analysis/synthetic_separation_metrics.py`
  - Shared utility functions for separation metrics.

- `analysis/calculate_theoretical_auc.py`
  - Computes theoretical AUC quantities for synthetic setups.
