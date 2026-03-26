import os
from dataclasses import dataclass
from os.path import join
from typing import List

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from tqdm import tqdm

from corebehrt.constants.data import (
    ABSPOS_FEAT,
    AGE_FEAT,
    ATTENTION_MASK,
    CONCEPT_FEAT,
    SEGMENT_FEAT,
    VALUE_FEAT,
    TARGET,
    TARGET_VALUE,
    VALUE_MASK_TOKEN,
)
from corebehrt.modules.preparation.mask import ConceptMasker


@dataclass
class PatientData:
    pid: str
    concepts: List[int]  # or List[str], depending on your use
    abspos: List[float]  # or int, depends on your data
    segments: List[int]
    ages: List[float]  # e.g. age at each concept
    values: List[float]
    outcome: int = None


class PatientDataset:
    """A dataset class for managing patient data and vocabulary.

    This class provides functionality to store and process patient data along with their
    associated vocabulary. It supports parallel processing of patient data and saving/loading
    functionality.

    Attributes:
        patients (List[PatientData]): List of patient data objects containing medical concepts,
            positions, segments and ages.
    """

    def __init__(self, patients: List[PatientData]):
        """Initialize the PatientDataset.

        Args:
            patients (List[PatientData]): List of patient data objects.
        """
        self.patients = patients

    def __len__(self):
        """Get the number of patients in the dataset."""
        return len(self.patients)

    def __getitem__(self, idx: int):
        """Get a patient by index.

        Args:
            idx (int): Index of the patient to retrieve.

        Returns:
            PatientData: The patient data at the given index.
        """
        return self.patients[idx]

    def process_in_parallel(self, func, n_jobs=-1, chunk_size=1000, **kwargs):
        """Process all patients in parallel using the given function with chunking support.

        Args:
            func: Function to apply to each patient
            n_jobs (int): Number of parallel jobs. -1 means using all processors
            chunk_size (int): Size of patient chunks to process together
            **kwargs: Additional keyword arguments passed to the function

        Returns:
            list: Results of applying the function to each patient
        """
        # Get the chunk size
        n_jobs = 1 if len(self.patients) < 1000 else n_jobs
        loop = tqdm(
            self.patients,
            total=len(self.patients),
            desc=f"{func.__name__}",
            mininterval=10,
        )
        results = Parallel(n_jobs=n_jobs, batch_size=chunk_size, backend="threading")(
            delayed(func)(patient, **kwargs) for patient in loop
        )

        return results

    def save(self, save_dir: str, suffix: str = ""):
        """Save patient data and vocabulary to disk.

        Args:
            save_dir (str): Directory path to save the files.
        """
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.patients, join(save_dir, f"patients{suffix}.pt"))

    def filter_by_pids(self, pids: List[str]) -> "PatientDataset":
        pids_set = set(pids)
        return PatientDataset([p for p in self.patients if p.pid in pids_set])

    def get_pids(self) -> List[str]:
        return [p.pid for p in self.patients]

    def get_outcomes(self) -> List[int]:
        return [p.outcome for p in self.patients]

    def assign_outcomes(self, outcomes: pd.Series):
        """Assigns binary outcomes to each patient in the dataset.

        Takes a pandas Series mapping patient IDs to outcomes absolute positions and assigns a binary outcome
        to each patient in the dataset.

        Args:
            outcomes (pd.Series): Series with patient IDs as index and outcomes as values.
                The actual outcome values are not used, only whether they are null or not.

        Returns:
            PatientDataset: Returns self for method chaining.
        """
        for p in self.patients:
            p.outcome = outcomes[p.pid]

        return self

    @staticmethod
    def combine_datasets(datasets: List["PatientDataset"]) -> "PatientDataset":
        """Combine multiple PatientDataset objects into one.

        Args:
            datasets (List[PatientDataset]): List of PatientDataset objects to combine.

        Returns:
            PatientDataset: A new PatientDataset object with combined patients.
        """
        combined_patients = []
        for dataset in datasets:
            combined_patients.extend(dataset.patients)
        return PatientDataset(combined_patients)

    def match_datasets(self, reference_dataset: "PatientDataset") -> "PatientDataset":
        """Match current dataset to reference dataset by id and full event tuples."""
        nonidentical_start_patients = 0

        def _values_equal(left, right):
            # Treat missing values as equal so NaN fields can be matched.
            if pd.isna(left) and pd.isna(right):
                return True
            if (
                isinstance(left, (float, int, np.floating, np.integer))
                and isinstance(right, (float, int, np.floating, np.integer))
            ):
                return bool(np.isclose(left, right, rtol=1e-7, atol=1e-7))
            return left == right

        def _events_equal(left_event, right_event):
            # Segment index is sequence-position dependent and can differ after filtering.
            # Match on concept, value, abspos, age only.
            left_concept, left_value, left_abspos, _left_segment, left_age = left_event
            right_concept, right_value, right_abspos, _right_segment, right_age = (
                right_event
            )
            return (
                _values_equal(left_concept, right_concept)
                and _values_equal(left_value, right_value)
                and _values_equal(left_abspos, right_abspos)
                and _values_equal(left_age, right_age)
            )

        def _find_reference_in_source_indices(source_events, reference_events, pid):
            """
            Find indices in source_events that match reference_events in order.
            Events are matched in-order by:
            (concept, value, abspos, age)
            Segment is intentionally ignored.
            """
            matched_indices = []
            source_idx = 0
            for ref_idx, ref_event in enumerate(reference_events):
                while (
                    source_idx < len(source_events)
                    and not _events_equal(source_events[source_idx], ref_event)
                ):
                    source_idx += 1
                if source_idx == len(source_events):
                    source_window_start = max(0, len(source_events) - 5)
                    source_tail = source_events[source_window_start:]
                    reference_window_start = max(0, ref_idx - 2)
                    reference_window_end = min(len(reference_events), ref_idx + 3)
                    reference_window = reference_events[
                        reference_window_start:reference_window_end
                    ]
                    same_concept_candidates = [
                        (idx, event)
                        for idx, event in enumerate(source_events)
                        if event[0] == ref_event[0]
                    ][:10]
                    raise ValueError(
                        "Could not align source to reference. "
                        f"PID={pid}, missing reference event at index {ref_idx}: {ref_event}. "
                        f"Source events={len(source_events)}, reference events={len(reference_events)}. "
                        f"Reference window ({reference_window_start}:{reference_window_end})={reference_window}. "
                        f"Source tail ({source_window_start}:{len(source_events)})={source_tail}. "
                        f"Source candidates with same concept={same_concept_candidates}"
                    )
                matched_indices.append(source_idx)
                source_idx += 1
            return matched_indices

        source_pids = [patient.pid for patient in self.patients]
        reference_pids = [patient.pid for patient in reference_dataset.patients]
        if len(source_pids) != len(reference_pids):
            raise ValueError(
                "Source and reference have different patient counts: "
                f"{len(source_pids)} != {len(reference_pids)}"
            )
        if len(set(source_pids)) != len(source_pids):
            raise ValueError("Source dataset contains duplicate patient IDs")
        if len(set(reference_pids)) != len(reference_pids):
            raise ValueError("Reference dataset contains duplicate patient IDs")
        if set(source_pids) != set(reference_pids):
            missing_in_source = set(reference_pids) - set(source_pids)
            missing_in_reference = set(source_pids) - set(reference_pids)
            raise ValueError(
                "Source and reference patient ID sets differ. "
                f"Missing in source: {sorted(missing_in_source)[:5]}, "
                f"missing in reference: {sorted(missing_in_reference)[:5]}"
            )

        reference_by_pid = {patient.pid: patient for patient in reference_dataset.patients}

        for source_patient in self.patients:
            reference_patient = reference_by_pid[source_patient.pid]

            target_events = list(
                zip(
                    reference_patient.concepts,
                    reference_patient.values,
                    reference_patient.abspos,
                    reference_patient.segments,
                    reference_patient.ages,
                )
            )
            target_outcome = reference_patient.outcome

            source_concepts = source_patient.concepts
            source_values = source_patient.values
            source_abspos = source_patient.abspos
            source_segments = source_patient.segments
            source_ages = source_patient.ages
            source_outcome = source_patient.outcome

            source_events = list(
                zip(
                    source_concepts,
                    source_values,
                    source_abspos,
                    source_segments,
                    source_ages,
                )
            )

            # Count patients whose sequences are not already identical at the start.
            # "Identical" uses the same matching key (concept, value, abspos, age),
            # with segment ignored.
            if len(source_events) != len(target_events):
                nonidentical_start_patients += 1
            else:
                is_identical_at_start = all(
                    _events_equal(se, te) for se, te in zip(source_events, target_events)
                )
                if not is_identical_at_start:
                    nonidentical_start_patients += 1

            matched_indices = _find_reference_in_source_indices(
                source_events, target_events, source_patient.pid
            )
            source_patient.concepts = [source_concepts[i] for i in matched_indices]
            source_patient.values = [source_values[i] for i in matched_indices]
            source_patient.abspos = [source_abspos[i] for i in matched_indices]
            source_patient.segments = [source_segments[i] for i in matched_indices]
            source_patient.ages = [source_ages[i] for i in matched_indices]
            source_patient.outcome = (
                source_outcome[: min(len(source_outcome), len(target_outcome))]
                if isinstance(source_outcome, list) and isinstance(target_outcome, list)
                else source_outcome
            )
        self.match_stats = {
            "nonidentical_start_patients": nonidentical_start_patients,
            "total_patients": len(self.patients),
        }
        print(
            "match_datasets(): non-identical at start "
            f"{nonidentical_start_patients}/{len(self.patients)} patients"
        )
        return self


class MLMDataset(Dataset):
    def __init__(
        self,
        patients: List[PatientData],
        vocabulary: dict,
        select_ratio: float,
        masking_ratio: float = 0.8,
        replace_ratio: float = 0.1,
        ignore_special_tokens: bool = True,
    ):
        self.patients = patients
        self.vocabulary = vocabulary
        if select_ratio > 0:
            self.masker = ConceptMasker(
                vocabulary,
                select_ratio,
                masking_ratio,
                replace_ratio,
                ignore_special_tokens,
            )
        else:
            self.masker = None

    def __getitem__(self, index: int) -> dict:
        """
        1. Retrieve the PatientData.
        2. Mask the 'concepts'.
        3. Convert everything to torch.Tensor.
        4. Return a dict that PyTorch can collate into a batch.
        """
        patient = self.patients[index]
        concepts = torch.tensor(patient.concepts)
        values = torch.tensor(patient.values)
        masked_concepts, target, indices_mask = self.masker.mask_patient_concepts(
            concepts
        )

        # concepts = torch.tensor(patient.concepts, dtype=torch.long)
        # values = torch.tensor(patient.values, dtype=torch.float)
        # masked_concepts, target, selected_indices = self.masker.mask_patient_concepts(concepts)
        masked_values = values.clone()
        masked_values[indices_mask] = VALUE_MASK_TOKEN
        attention_mask = torch.ones_like(masked_concepts)
        sample = {
            CONCEPT_FEAT: masked_concepts,
            TARGET: target,
            VALUE_FEAT: masked_values,
            ABSPOS_FEAT: torch.tensor(patient.abspos, dtype=torch.float),
            SEGMENT_FEAT: torch.tensor(patient.segments, dtype=torch.long),
            AGE_FEAT: torch.tensor(patient.ages, dtype=torch.float),
            ATTENTION_MASK: attention_mask,
            TARGET_VALUE: values,
        }

        return sample

    def __len__(self):
        return len(self.patients)


class BinaryOutcomeDataset(Dataset):
    """
    outcomes: absolute position when outcome occured for each patient
    outcomes is a list of the outcome timestamps to predict
    """

    def __init__(self, patients: List[PatientData]):
        self.patients = patients

    def __getitem__(self, index: int) -> dict:
        patient = self.patients[index]
        attention_mask = torch.ones(
            len(patient.concepts), dtype=torch.long
        )  # Require attention mask for bi-gru head
        sample = {
            CONCEPT_FEAT: torch.tensor(patient.concepts, dtype=torch.long),
            VALUE_FEAT: torch.tensor(patient.values, dtype=torch.float),
            ABSPOS_FEAT: torch.tensor(patient.abspos, dtype=torch.float),
            SEGMENT_FEAT: torch.tensor(patient.segments, dtype=torch.long),
            AGE_FEAT: torch.tensor(patient.ages, dtype=torch.float),
            ATTENTION_MASK: attention_mask,
            TARGET: torch.tensor(patient.outcome, dtype=torch.float),
        }
        return sample

    def __len__(self):
        return len(self.patients)
