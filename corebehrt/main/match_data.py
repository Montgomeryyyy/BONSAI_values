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
    compute_match_source_indices,
    match_events_equal,
    translate_concept_token,
)
from corebehrt.constants.paths import PREPARED_ALL_PATIENTS
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


def _patient_summary(label: str, patient, idx: int) -> None:
    """Print one-line stats for a PatientData sample (debugging)."""
    n = len(patient.concepts)
    head = patient.concepts[: min(5, n)]
    print(
        f"  [{label}] index={idx} pid={patient.pid} "
        f"seq_len={n} outcome={patient.outcome} "
        f"concept_ids[:5]={head}"
    )


def _print_match_alignment_debug(
    reference_patient,
    source_patient_original,
    vocab_reference: dict,
    vocab_source: dict,
    code_mapping: Optional[Dict[str, str]],
    max_rows: int = 40,
) -> None:
    """
    Side-by-side: each reference event vs the source row chosen by subsequence matching.
    Also lists source indices that were skipped (extra events in source only).
    """
    id_ref = _invert_vocab(vocab_reference)
    id_src = _invert_vocab(vocab_source)
    n_ref = len(reference_patient.concepts)
    n_src = len(source_patient_original.concepts)
    if n_src < n_ref:
        print(
            f"  Alignment pid={reference_patient.pid}: skip side-by-side table — "
            f"source_events={n_src} < reference_events={n_ref}. "
            "Subsequence matching needs len(source) >= len(reference) (one source row per reference row)."
        )
        return

    matched = compute_match_source_indices(
        source_patient_original,
        reference_patient,
        vocab_source,
        vocab_reference,
        code_mapping,
    )
    used_src = set(matched)
    skipped_src = [i for i in range(n_src) if i not in used_src]

    print(
        f"  Alignment pid={reference_patient.pid}: "
        f"reference_events={n_ref}, source_events={n_src}, "
        f"matched_pairs={len(matched)}, skipped_source_rows={len(skipped_src)}"
    )
    print(
        "  Columns: ref# | token(ref id) | translated | val | seg | abspos | age "
        "|| src# | token(src id) | translated | val | seg | abspos | age || keys_match"
    )
    show = min(max_rows, len(matched))
    for k in range(show):
        j = matched[k]
        rc = reference_patient.concepts[k]
        ra = reference_patient.abspos[k]
        sc = source_patient_original.concepts[j]
        sa = source_patient_original.abspos[j]
        ref_event = (
            rc,
            reference_patient.values[k],
            ra,
            reference_patient.segments[k],
            reference_patient.ages[k],
        )
        src_event = (
            sc,
            source_patient_original.values[j],
            sa,
            source_patient_original.segments[j],
            source_patient_original.ages[j],
        )
        tok_r = _abbrev_token(id_ref.get(rc, f"<?:{rc}>"))
        tok_s = _abbrev_token(id_src.get(sc, f"<?:{sc}>"))
        tr_r = _abbrev_token(translate_concept_token(id_ref.get(rc, ""), code_mapping))
        tr_s = _abbrev_token(translate_concept_token(id_src.get(sc, ""), code_mapping))
        keys_ok = match_events_equal(
            src_event, ref_event, id_src, id_ref, code_mapping
        )
        rv = reference_patient.values[k]
        rs = reference_patient.segments[k]
        sv = source_patient_original.values[j]
        ss = source_patient_original.segments[j]
        print(
            f"  {k:3d} | {tok_r} ({rc}) | {tr_r} | {_fmt_value(rv)} | {rs} | {ra:.6g} | {reference_patient.ages[k]:.6g} "
            f"|| {j:3d} | {tok_s} ({sc}) | {tr_s} | {_fmt_value(sv)} | {ss} | {sa:.6g} | {source_patient_original.ages[j]:.6g} "
            f"|| {keys_ok}"
        )
    if len(matched) > show:
        print(f"  ... ({len(matched) - show} more aligned rows not shown)")

    if skipped_src:
        preview = skipped_src[: min(15, len(skipped_src))]
        parts = []
        for j in preview:
            c = source_patient_original.concepts[j]
            tok = _abbrev_token(id_src.get(c, f"<?:{c}>"))
            parts.append(f"{j}:{tok}")
        extra = "" if len(skipped_src) <= 15 else f" ... (+{len(skipped_src) - 15} more)"
        print(
            f"  Source-only rows (not aligned to any reference event): "
            f"{len(skipped_src)} total. First indices: {preview}{extra}"
        )
        print(f"    preview: {', '.join(parts)}")
    else:
        print("  Source-only rows: none (source length equals matched span with no extras).")


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
    _patient_summary("prepared (source)", prepared_data[sample_idx], sample_idx)
    _patient_summary("reference", reference_data[sample_idx], sample_idx)

    ref_sample = reference_data[sample_idx]
    source_before = copy.deepcopy(prepared_data[sample_idx])
    print(
        "Side-by-side alignment (reference vs original source; "
        "match key = translated + abspos + age):"
    )
    _print_match_alignment_debug(
        ref_sample,
        source_before,
        vocab_reference,
        vocab_source,
        code_mapping,
        max_rows=40,
    )

    matched_data = prepared_data.match_datasets(
        reference_data, vocab_source, vocab_reference, code_mapping
    )
    matched_data.patients = matched_data.process_in_parallel(
        normalize_segments_for_patient
    )

    stats = getattr(matched_data, "match_stats", None)
    if stats:
        print(
            "match_datasets() stats: "
            f"nonidentical_at_start={stats.get('nonidentical_start_patients')}/"
            f"{stats.get('total_patients')} patients"
        )
    print(f"Sample patient after match (PatientDataset[{sample_idx}]):")
    _patient_summary("matched (source aligned to reference)", matched_data[sample_idx], sample_idx)

if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    config_path = args.config_path
    main_match_data(config_path)
