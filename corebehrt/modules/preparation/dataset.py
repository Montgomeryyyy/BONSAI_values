import os
from dataclasses import dataclass
from os.path import join
from typing import Dict, List, Optional

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
    VAL_TOKEN,
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


def _invert_vocab(vocab: Dict) -> Dict:
    return {v: k for k, v in vocab.items()}


def concept_id_to_code(concept_id: int, id_to_token: Dict[int, str]) -> str:
    """Map a concept id to its code string using an inverted vocab (id → token)."""
    return id_to_token[concept_id]


def apply_optional_code_mapping(
    token: str, code_mapping: Optional[Dict[str, str]]
) -> str:
    """
    Optional merge on top of the vocab string (same dict as EHRTokenizer when needed).
    """
    if not code_mapping:
        return token
    return code_mapping.get(token, token)


def concept_match_key(
    concept_id: int,
    id_to_token: Dict[int, str],
    code_mapping: Optional[Dict[str, str]] = None,
) -> str:
    """
    Match key: invert the given vocab to ``id_to_token``, map ``concept_id`` to its code
    string, optionally apply ``code_mapping``, then compare to the other side the same way.
    """
    code = concept_id_to_code(concept_id, id_to_token)
    return apply_optional_code_mapping(code, code_mapping)


def translate_concept_token(
    token: str, code_mapping: Optional[Dict[str, str]]
) -> str:
    """Backward-compatible alias for apply_optional_code_mapping."""
    return apply_optional_code_mapping(token, code_mapping)


def _match_values_equal(left, right) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if (
        isinstance(left, (float, int, np.floating, np.integer))
        and isinstance(right, (float, int, np.floating, np.integer))
    ):
        return bool(np.isclose(left, right, rtol=1e-7, atol=1e-7))
    return left == right


def match_events_equal(
    left_event: tuple,
    right_event: tuple,
    id_to_token_source: Dict,
    id_to_token_reference: Dict,
    code_mapping: Optional[Dict[str, str]] = None,
) -> bool:
    left_concept_id, _lv, left_abspos, _ls, left_age = left_event
    right_concept_id, _rv, right_abspos, _rs, right_age = right_event
    if left_concept_id not in id_to_token_source or right_concept_id not in id_to_token_reference:
        return False
    left_key = concept_match_key(left_concept_id, id_to_token_source, code_mapping)
    right_key = concept_match_key(right_concept_id, id_to_token_reference, code_mapping)
    return (
        left_key == right_key
        and _match_values_equal(left_abspos, right_abspos)
        and _match_values_equal(left_age, right_age)
    )


def compute_match_source_indices(
    source_patient: PatientData,
    reference_patient: PatientData,
    vocab_source: Dict,
    vocab_reference: Dict,
    code_mapping: Optional[Dict[str, str]] = None,
) -> List[int]:
    """
    For each reference event in order, find the matching index in the source sequence
    (subsequence alignment). Invert each vocab, map concept ids to code strings, optional
    code_mapping, then match with abspos and age as before.
    """
    id_to_token_source = _invert_vocab(vocab_source)
    id_to_token_reference = _invert_vocab(vocab_reference)

    source_events_all = list(
        zip(
            source_patient.concepts,
            source_patient.values,
            source_patient.abspos,
            source_patient.segments,
            source_patient.ages,
        )
    )
    reference_events_all = list(
        zip(
            reference_patient.concepts,
            reference_patient.values,
            reference_patient.abspos,
            reference_patient.segments,
            reference_patient.ages,
        )
    )
    source_filtered = [
        (idx, event)
        for idx, event in enumerate(source_events_all)
        if id_to_token_source.get(event[0]) != VAL_TOKEN
    ]
    reference_events = [
        event for event in reference_events_all if id_to_token_reference.get(event[0]) != VAL_TOKEN
    ]
    source_events = [event for _, event in source_filtered]
    source_original_indices = [idx for idx, _ in source_filtered]
    pid = source_patient.pid
    if len(source_events) < len(reference_events):
        raise ValueError(
            "Cannot align: source has fewer events than reference, so the full reference "
            "sequence cannot be embedded as an order-preserving subsequence of the source. "
            f"PID={pid}, source_events={len(source_events)}, reference_events={len(reference_events)}. "
            "Use a reference sequence that is not longer than the source, or rebuild the "
            "source so it contains at least as many events as the reference."
        )
    matched_indices: List[int] = []
    source_idx = 0
    for ref_idx, ref_event in enumerate(reference_events):
        while source_idx < len(source_events) and not match_events_equal(
            source_events[source_idx],
            ref_event,
            id_to_token_source,
            id_to_token_reference,
            code_mapping,
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
            ref_concept_token = id_to_token_reference[ref_event[0]]
            ref_match_key = concept_match_key(
                ref_event[0], id_to_token_reference, code_mapping
            )
            same_concept_candidates = [
                (idx, event)
                for idx, event in enumerate(source_events)
                if event[0] in id_to_token_source
                and concept_match_key(event[0], id_to_token_source, code_mapping)
                == ref_match_key
            ][:10]
            extra = ""
            if not same_concept_candidates:
                extra = (
                    f" No source row in this patient matches key {ref_match_key!r} "
                    f"(reference vocab token={ref_concept_token!r}, id={ref_event[0]}); "
                    f"check vocab overlap / pipeline."
                )
            elif len(source_events) - source_idx < len(reference_events) - ref_idx:
                extra = (
                    f" Not enough source positions left ({len(source_events) - source_idx}) "
                    f"to match remaining reference events ({len(reference_events) - ref_idx})."
                )
            raise ValueError(
                "Could not align source to reference. "
                f"PID={pid}, missing reference event at index {ref_idx}: {ref_event}. "
                f"reference_token={ref_concept_token!r}, translated_key={ref_match_key!r}. "
                f"Source events={len(source_events)}, reference events={len(reference_events)}. "
                f"Reference window ({reference_window_start}:{reference_window_end})={reference_window}. "
                f"Source tail ({source_window_start}:{len(source_events)})={source_tail}. "
                f"Source candidates with same concept token={same_concept_candidates}.{extra}"
            )
        matched_indices.append(source_original_indices[source_idx])
        source_idx += 1
    return matched_indices


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

    def match_datasets(
        self,
        reference_dataset: "PatientDataset",
        vocab_source: Dict,
        vocab_reference: Dict,
        code_mapping: Optional[Dict[str, str]] = None,
    ) -> "PatientDataset":
        """Align source patients to reference by PID.

        Invert ``vocab_source`` / ``vocab_reference`` to map concept ids to code strings,
        optionally apply ``code_mapping``, then match those keys with abspos and age.
        Values and segments are ignored; matched rows keep source-side values and segments.
        """
        nonidentical_start_patients = 0
        skipped_unalignable_patients = 0
        skipped_source_shorter_than_reference = 0
        unalignable_examples = []

        id_to_token_source = _invert_vocab(vocab_source)
        id_to_token_reference = _invert_vocab(vocab_reference)

        source_pids = [patient.pid for patient in self.patients]
        reference_pids = [patient.pid for patient in reference_dataset.patients]
        if len(set(source_pids)) != len(source_pids):
            raise ValueError("Source dataset contains duplicate patient IDs")
        if len(set(reference_pids)) != len(reference_pids):
            raise ValueError("Reference dataset contains duplicate patient IDs")
        if len(source_pids) != len(reference_pids):
            print(
                "Warning: Source and reference have different patient counts. "
                f"Proceeding with source-driven matching: {len(source_pids)} vs {len(reference_pids)}"
            )
        source_pid_set = set(source_pids)
        reference_pid_set = set(reference_pids)
        missing_in_source = reference_pid_set - source_pid_set
        missing_in_reference = source_pid_set - reference_pid_set
        if missing_in_source:
            print(
                "Warning: Reference contains patients not present in source; these will be ignored. "
                f"Count={len(missing_in_source)}, examples={sorted(missing_in_source)[:5]}"
            )
        if missing_in_reference:
            print(
                "Warning: Source contains patients not present in reference; these will be skipped. "
                f"Count={len(missing_in_reference)}, examples={sorted(missing_in_reference)[:5]}"
            )

        reference_by_pid = {patient.pid: patient for patient in reference_dataset.patients}
        source_patients_with_reference = [
            patient for patient in self.patients if patient.pid in reference_by_pid
        ]
        skipped_source_patients = len(self.patients) - len(source_patients_with_reference)
        self.patients = source_patients_with_reference

        matched_patients = []
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
            source_events_no_val = [
                event for event in source_events if id_to_token_source.get(event[0]) != VAL_TOKEN
            ]
            target_events_no_val = [
                event for event in target_events if id_to_token_reference.get(event[0]) != VAL_TOKEN
            ]

            # Count patients whose sequences are not already identical at the start.
            # Same key as matching: concept token, abspos, age (values/segments ignored).
            if len(source_events_no_val) != len(target_events_no_val):
                nonidentical_start_patients += 1
            else:
                is_identical_at_start = all(
                    match_events_equal(
                        se, te, id_to_token_source, id_to_token_reference, code_mapping
                    )
                    for se, te in zip(source_events_no_val, target_events_no_val)
                )
                if not is_identical_at_start:
                    nonidentical_start_patients += 1

            if len(source_events_no_val) < len(target_events_no_val):
                skipped_source_shorter_than_reference += 1
                continue

            try:
                matched_indices = compute_match_source_indices(
                    source_patient,
                    reference_patient,
                    vocab_source,
                    vocab_reference,
                    code_mapping,
                )
            except ValueError as exc:
                skipped_unalignable_patients += 1
                if len(unalignable_examples) < 5:
                    unalignable_examples.append((source_patient.pid, str(exc)))
                continue
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
            matched_patients.append(source_patient)
        self.patients = matched_patients
        self.match_stats = {
            "nonidentical_start_patients": nonidentical_start_patients,
            "total_patients": len(self.patients),
            "skipped_source_patients_without_reference": skipped_source_patients,
            "skipped_source_shorter_than_reference": skipped_source_shorter_than_reference,
            "skipped_unalignable_patients": skipped_unalignable_patients,
        }
        if skipped_source_shorter_than_reference > 0:
            print(
                "Warning: Skipped patients where source sequence is shorter than reference. "
                f"Count={skipped_source_shorter_than_reference}"
            )
        if skipped_unalignable_patients > 0:
            print(
                "Warning: Skipped unalignable patients during matching. "
                f"Count={skipped_unalignable_patients}, examples={unalignable_examples}"
            )
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
