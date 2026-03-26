"""Prepare data for training. Run create_data.yaml first to create the dataset and vocabulary."""

import logging
import torch
from os.path import join
import os

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
from corebehrt.modules.preparation.dataset import PatientDataset
from corebehrt.constants.paths import PREPARED_ALL_PATIENTS

CONFIG_PATH = "./corebehrt/configs/prepare_pretrain.yaml"


def main_match_data(config_path):
    cfg = load_config(config_path)

    DirectoryPreparer(cfg).setup_match_data()

    logger = logging.getLogger("match data")
    logger.info("Matching data")

    # Load prepared data
    prepared_data = torch.load(join(cfg.paths.prepared_data, PREPARED_ALL_PATIENTS))
    prepared_data = PatientDataset(prepared_data)

    # Load reference data
    reference_data = torch.load(join(cfg.paths.reference_data, PREPARED_ALL_PATIENTS))
    reference_data = PatientDataset(reference_data)
    
    print((prepared_data[1]))
    print((reference_data[1]))

    matched_data = prepared_data.match_datasets(reference_data)
    print(matched_data[1])

if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    config_path = args.config_path
    main_match_data(config_path)
