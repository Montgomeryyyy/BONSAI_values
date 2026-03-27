"""Prepare data for training. Run create_data.yaml first to create the dataset and vocabulary."""

import copy
import logging
import os
from os.path import join
from typing import Dict, Optional

import pandas as pd
import torch

from corebehrt.functional.setup.args import get_args
from corebehrt.modules.preparation.prepare_data import DatasetPreparer
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer
from corebehrt.functional.features.split import split_pids_into_train_val
from corebehrt.main.helper.pretrain import (
    load_train_val_split,
    get_splits_path,
)
from corebehrt.functional.io_operations.save import save_pids_splits
from corebehrt.constants.paths import FOLDS_FILE, TEST_PIDS_FILE
from corebehrt.modules.preparation.dataset import (
    PatientDataset,
    translate_concept_token,
)
from corebehrt.constants.paths import PREPARED_ALL_PATIENTS
from corebehrt.constants.data import VAL_TOKEN
from corebehrt.functional.io_operations.load import load_vocabulary
from corebehrt.functional.features.normalize import normalize_segments_for_patient

CONFIG_PATH = "./corebehrt/configs/prepare_pretrain.yaml"


def _invert_vocab(vocab: dict) -> dict:
    return {v: k for k, v in vocab.items()}


def _abbrev_token(token: str, max_len: int = 44) -> str:
    s = str(token)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _fmt_value(v) -> str:
    if v is None:
        return "None"
    if pd.isna(v):
        return "nan"
    return f"{v:.6g}" if isinstance(v, float) else str(v)


def _patient_summary(
    label: str,
    patient,
    idx: int,
    vocab: dict,
    code_mapping: Optional[Dict[str, str]],
) -> None:
    """Print one-line stats for a PatientData sample (debugging)."""
    n = len(patient.concepts)
    head = patient.concepts[: min(5, n)]
    n_no_val = _count_non_val(patient, vocab, code_mapping)
    print(
        f"  [{label}] index={idx} pid={patient.pid} "
        f"seq_len={n} outcome={patient.outcome} "
        f"seq_len_no_val={n_no_val} "
        f"concept_ids[:5]={head}"
    )


def _non_val_rows(
    patient,
    vocab: dict,
    code_mapping: Optional[Dict[str, str]],
    max_rows: int,
):
    """
    Return non-VAL rows with an explicit matching key.
    Key must match what matcher uses: (translated_concept, abspos, age).
    """
    id_to_token = _invert_vocab(vocab)
    rows = []
    for idx, (c, v, a, g) in enumerate(
        zip(patient.concepts, patient.values, patient.abspos, patient.ages)
    ):
        token = id_to_token.get(c, f"<?:{c}>")
        translated = translate_concept_token(token, code_mapping)
        if token == VAL_TOKEN or translated == VAL_TOKEN:
            continue
        key = (translated, a, g)
        rows.append(
            {
                "src_idx": idx,
                "concept": translated,
                "value": v,
                "abspos": a,
                "age": g,
                "key": key,
            }
        )
        if len(rows) >= max_rows:
            break
    return rows


def _count_non_val(patient, vocab: dict, code_mapping: Optional[Dict[str, str]]) -> int:
    id_to_token = _invert_vocab(vocab)
    count = 0
    for c in patient.concepts:
        token = id_to_token.get(c, f"<?:{c}>")
        translated = translate_concept_token(token, code_mapping)
        if token == VAL_TOKEN or translated == VAL_TOKEN:
            continue
        count += 1
    return count


def _print_patient_side_by_side(
    pid,
    source_before,
    source_after,
    reference_patient,
    vocab_source: dict,
    vocab_reference: dict,
    code_mapping: Optional[Dict[str, str]],
    max_rows: int = 8,
) -> None:
    before_n = _count_non_val(source_before, vocab_source, code_mapping)
    after_n = _count_non_val(source_after, vocab_source, code_mapping)
    ref_n = _count_non_val(reference_patient, vocab_reference, code_mapping)
    print(
        f"  PID={pid}: source_non_val {before_n} -> {after_n}; reference_non_val={ref_n}"
    )
    print(
        "    Columns: ref_row# | reference(concept, value, abspos, age) "
        "|| matched_source(concept, value, abspos, age) | key_match"
    )

    ref_rows = _non_val_rows(
        reference_patient, vocab_reference, code_mapping, max_rows=max_rows
    )
    src_rows_all = _non_val_rows(
        source_after, vocab_source, code_mapping, max_rows=10_000_000
    )
    # Multimap: key -> list of available source rows to consume.
    available = {}
    for r in src_rows_all:
        available.setdefault(r["key"], []).append(r)

    used_keys = 0
    for i, ref_r in enumerate(ref_rows):
        key = ref_r["key"]
        if key in available and available[key]:
            src_r = available[key].pop(0)
            used_keys += 1
            key_match = True
            left = (
                ref_r["concept"],
                _fmt_value(ref_r["value"]),
                ref_r["abspos"],
                ref_r["age"],
            )
            right = (
                src_r["concept"],
                _fmt_value(src_r["value"]),
                src_r["abspos"],
                src_r["age"],
            )
        else:
            key_match = False
            left = (
                ref_r["concept"],
                _fmt_value(ref_r["value"]),
                ref_r["abspos"],
                ref_r["age"],
            )
            right = ("MISSING", "MISSING", "MISSING", "MISSING")

        print(f"    {i:2d} | {left} || {right} | {key_match}")

    if used_keys == 0:
        print("    Note: no reference keys found in matched source for these rows.")


def main_match_data(config_path):
    cfg = load_config(config_path)

    DirectoryPreparer(cfg).setup_match_data()

    logger = logging.getLogger("match data")
    logger.info("Matching data")

    # Load prepared data
    prepared_data = torch.load(join(cfg.paths.prepared_data, PREPARED_ALL_PATIENTS))
    prepared_data = PatientDataset(prepared_data)
    vocab_source = load_vocabulary(cfg.paths.prepared_data)

    # Load reference data
    reference_data = torch.load(join(cfg.paths.reference_data, PREPARED_ALL_PATIENTS))
    reference_data = PatientDataset(reference_data)
    vocab_reference = load_vocabulary(cfg.paths.reference_data)

    code_mapping = None
    if "code_mapping" in cfg.paths and cfg.paths["code_mapping"]:
        cm_path = cfg.paths["code_mapping"]
        logger.info("Loading code mapping (same as tokenization) from %s", cm_path)
        code_mapping = torch.load(cm_path)
    else:
        logger.warning(
            "No paths.code_mapping in config: matching compares raw vocabulary tokens. "
            "For cross-vocab alignment, set code_mapping to the same .pt file used in create_data."
        )

    sample_idx = 1
    original_n_patients = len(prepared_data)
    prepared_by_pid_before = {p.pid: copy.deepcopy(p) for p in prepared_data.patients}
    print(
        "Loaded datasets:\n"
        f"  prepared_data: n_patients={len(prepared_data)} "
        f"| vocab (source) size={len(vocab_source)} "
        f"| path={cfg.paths.prepared_data}\n"
        f"  reference_data: n_patients={len(reference_data)} "
        f"| vocab (reference) size={len(vocab_reference)} "
        f"| path={cfg.paths.reference_data}"
    )
    print(f"Sample patient before match (PatientDataset[{sample_idx}]):")
    _patient_summary(
        "prepared (source)",
        prepared_data[sample_idx],
        sample_idx,
        vocab_source,
        code_mapping,
    )
    _patient_summary(
        "reference",
        reference_data[sample_idx],
        sample_idx,
        vocab_reference,
        code_mapping,
    )

    matched_data = prepared_data.match_datasets(
        reference_data, vocab_source, vocab_reference, code_mapping
    )
    matched_data.patients = matched_data.process_in_parallel(
        normalize_segments_for_patient
    )

    stats = getattr(matched_data, "match_stats", None)
    skipped_total = original_n_patients - len(matched_data)
    print(f"Matching result: kept={len(matched_data)}, skipped={skipped_total}")
    if stats:
        print(
            "match_datasets() stats: "
            f"nonidentical_at_start={stats.get('nonidentical_start_patients')}/"
            f"{stats.get('total_patients')} patients, "
            f"skipped_without_reference={stats.get('skipped_source_patients_without_reference', 0)}"
        )
    print(f"Sample patient after match (PatientDataset[{sample_idx}]):")
    _patient_summary(
        "matched (source aligned to reference)",
        matched_data[sample_idx],
        sample_idx,
        vocab_source,
        code_mapping,
    )

    reference_by_pid = {p.pid: p for p in reference_data.patients}
    changed_pids = []
    for p in matched_data.patients:
        before = prepared_by_pid_before.get(p.pid)
        if before is None:
            continue
        before_n = _count_non_val(before, vocab_source, code_mapping)
        after_n = _count_non_val(p, vocab_source, code_mapping)
        if before_n != after_n:
            changed_pids.append(p.pid)
    print(
        f"Patients with changed non-VAL sequence length after matching: {len(changed_pids)}"
    )
    if changed_pids:
        print("Examples (reference vs matched source) for changed patients:")
        for pid in changed_pids[:3]:
            _print_patient_side_by_side(
                pid,
                prepared_by_pid_before[pid],
                next(p for p in matched_data.patients if p.pid == pid),
                reference_by_pid[pid],
                vocab_source,
                vocab_reference,
                code_mapping,
                max_rows=8,
            )

    matched_data.save(cfg.paths.matched_data)

if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    config_path = args.config_path
    main_match_data(config_path)
