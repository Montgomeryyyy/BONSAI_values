"""Pretrain BERT model on EHR data. Use config_template pretrain.yaml. Run main_data_pretrain.py first to create the dataset and vocabulary."""

import logging
import torch
from os.path import join
from corebehrt.functional.io_operations.load import load_vocabulary
from corebehrt.functional.setup.args import get_args
from corebehrt.functional.setup.model import load_model_cfg_from_checkpoint
from corebehrt.functional.trainer.setup import replace_steps_with_epochs
from corebehrt.main.helper.pretrain import (
    load_checkpoint_and_epoch,
)
from corebehrt.modules.preparation.dataset import MLMDataset, PatientDataset
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer
from corebehrt.modules.setup.initializer import Initializer
from corebehrt.modules.trainer.trainer import EHRTrainer
from corebehrt.constants.paths import PREPARED_TRAIN_PATIENTS, PREPARED_VAL_PATIENTS

CONFIG_PATH = "./corebehrt/configs/pretrain.yaml"


def main_train(config_path):
    cfg = load_config(config_path)
    logger = logging.getLogger("pretrain")

    # Setup directories
    data_cfg = load_config(join(cfg.paths.prepared_data, "data_config.yaml"))
    value_embedding_mode = data_cfg.features.values.value_type
    logger.info(f"Value embedding mode: {value_embedding_mode}")
    cfg.model.value_embedding_mode = value_embedding_mode
    DirectoryPreparer(cfg).setup_pretrain()

    # Check if we are training from checkpoint, if so, update model config
    restart_path = cfg.paths.get("restart_model")
    if restart_path:
        cfg.model = load_model_cfg_from_checkpoint(restart_path, "pretrain_config")

    # Get data
    train_data = PatientDataset(
        torch.load(
            join(cfg.paths.prepared_data, PREPARED_TRAIN_PATIENTS), weights_only=False
        )
    )
    val_data = PatientDataset(
        torch.load(
            join(cfg.paths.prepared_data, PREPARED_VAL_PATIENTS), weights_only=False
        )
    )
    vocab = load_vocabulary(cfg.paths.prepared_data)

    # Initialize datasets
    train_dataset = MLMDataset(train_data.patients, vocab, **cfg.data.dataset)

    # print(train_dataset[0])
    val_dataset = MLMDataset(val_data.patients, vocab, **cfg.data.dataset)

    if "scheduler" in cfg:
        logger.info("Replacing steps with epochs in scheduler config")
        cfg.scheduler = replace_steps_with_epochs(
            cfg.scheduler, cfg.trainer_args.batch_size, len(train_dataset)
        )

    checkpoint, epoch = None, None
    if restart_path:
        checkpoint, epoch = load_checkpoint_and_epoch(
            restart_path, cfg.paths.get("checkpoint_epoch")
        )
        logger.info(f"Continue training from epoch {epoch}")
    initializer = Initializer(cfg, checkpoint=checkpoint, model_path=restart_path)
    model = initializer.initialize_pretrain_model(train_dataset)
    logger.info("Initializing optimizer")
    optimizer = initializer.initialize_optimizer(model)
    scheduler = initializer.initialize_scheduler(optimizer)

    logger.info("Initialize trainer")
    trainer = EHRTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        args=cfg.trainer_args,
        metrics=cfg.metrics,
        cfg=cfg,
        logger=logger,
        last_epoch=epoch,
    )
    logger.info("Start training")
    trainer.train()
    logger.info("Done")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    config_path = args.config_path
    main_train(config_path)
