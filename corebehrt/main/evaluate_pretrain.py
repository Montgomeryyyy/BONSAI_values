import logging
from os.path import join
import pandas as pd
import torch
import os

from corebehrt.constants.paths import FOLDS_FILE, PREPARED_ALL_PATIENTS
from corebehrt.functional.setup.args import get_args
from corebehrt.main.helper.evaluate_finetune import inference_fold, compute_metrics
from corebehrt.main.helper.finetune_cv import check_for_overlap
from corebehrt.modules.preparation.dataset import BinaryOutcomeDataset, PatientDataset
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer
from corebehrt.modules.setup.config import instantiate_function
from corebehrt.modules.setup.manager import ModelManager
from corebehrt.modules.model.model import (
    CorebehrtForPretraining,
)
from corebehrt.modules.setup.loader import ModelLoader
from corebehrt.functional.io_operations.load import load_vocabulary
from corebehrt.modules.preparation.dataset import MLMDataset, PatientDataset
from corebehrt.modules.trainer.inference import EHRInferenceRunnerPretrain

CONFIG_PATH = "./corebehrt/configs/evaluate_pretrain.yaml"


def main_evaluate_pretrain(config_path):
    # Setup directories
    cfg = load_config(config_path)
    DirectoryPreparer(cfg).setup_evaluate_pretrain()

    # Setup logging and config
    logger = logging.getLogger("evaluate")
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    cfg.trainer_args = {}
    batch_size_value = cfg.get("test_batch_size", 128)
    for key in ["test_batch_size", "effective_batch_size", "batch_size"]:
        cfg.trainer_args[key] = batch_size_value

    # Load model
    model_loader = ModelLoader(cfg.paths.model, cfg.paths.get("checkpoint_epoch"))
    model = model_loader.load_model(
        CorebehrtForPretraining
    )
    model.to(device)
    
    
    print(f"Model loaded from {cfg.paths.model}")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # Load test data
    test_data = PatientDataset(
        torch.load(join(cfg.paths.test_data_dir, PREPARED_ALL_PATIENTS))
    )
    vocab = load_vocabulary(cfg.paths.test_data_dir)
    test_dataset = MLMDataset(test_data.patients, vocab, select_ratio=0)

    # Run inference
    inference_runner = EHRInferenceRunnerPretrain(
        model=model,
        test_dataset=test_dataset,
        args=cfg.trainer_args,
        cfg=cfg,
        logger=logger,
    )
    all_embeddings, all_pids = inference_runner.inference_loop(return_embeddings=True)
    
    # Print information about the flattened embeddings and PIDs
    print(f"Total samples: {len(all_embeddings)}")
    print(f"Total PIDs: {len(all_pids)}")
    print("Sample information:")
    for i in range(min(5, len(all_embeddings))):
        print(f"  Sample {i}: {all_embeddings[i].shape} with PID: {all_pids[i]}")
    
    # Calculate memory usage
    total_parameters = sum(emb.numel() for emb in all_embeddings)
    print(f"Total parameters: {total_parameters:,}")
    print(f"Memory usage: {total_parameters * 4 / 1024**3:.2f} GB (assuming float32)")
    
    # Show sequence length distribution
    seq_lengths = [emb.shape[0] for emb in all_embeddings]
    print(f"Sequence length stats:")
    print(f"  Min: {min(seq_lengths)}")
    print(f"  Max: {max(seq_lengths)}")
    print(f"  Mean: {sum(seq_lengths) / len(seq_lengths):.1f}")

    # Save embeddings and PIDs
    os.makedirs(cfg.paths.embeddings, exist_ok=True)
    
    # Save flattened embeddings and PIDs
    torch.save(all_embeddings, join(cfg.paths.embeddings, "embeddings.pt"))
    torch.save(all_pids, join(cfg.paths.embeddings, "pids.pt"))
    
    print(f"Embeddings and PIDs saved to {cfg.paths.embeddings}")
    print("Files saved:")
    print("  - embeddings.pt: List of embedding tensors (one per sample)")
    print("  - pids.pt: List of patient IDs (one per sample)")
    print("  - Each embedding has shape: (sequence_length, hidden_size)")
    print("  - Each PID is a string corresponding to the embedding at the same index")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_evaluate_pretrain(args.config_path)
