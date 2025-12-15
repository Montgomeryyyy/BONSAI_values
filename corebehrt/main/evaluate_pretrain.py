import logging
from os.path import join
import torch
import os

from corebehrt.constants.paths import PREPARED_ALL_PATIENTS
from corebehrt.functional.setup.args import get_args
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer
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
    model = model_loader.load_model(CorebehrtForPretraining)
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

    # Compute pooled (mean over sequence) embeddings and save in chunks
    os.makedirs(cfg.paths.embeddings, exist_ok=True)
    chunks_dir = join(cfg.paths.embeddings, "embeddings")
    os.makedirs(chunks_dir, exist_ok=True)

    import json

    chunk_size = 1000  # samples per chunk

    # Compute pooled embeddings (mean over sequence dimension)
    print("Computing pooled embeddings (mean over sequence)...")
    pooled_embeddings = []
    for emb in all_embeddings:
        pooled = emb.mean(dim=0)  # Shape: (hidden_size,)
        pooled_embeddings.append(pooled)

    summary = {
        "chunk_size": chunk_size,
        "num_samples": len(pooled_embeddings),
        "hidden_size": pooled_embeddings[0].shape[0] if pooled_embeddings else 0,
        "dtype": str(pooled_embeddings[0].dtype) if pooled_embeddings else "unknown",
        "pooled": True,  # Indicates these are pooled (mean over sequence)
        "chunks": [],
    }

    for i in range(0, len(pooled_embeddings), chunk_size):
        idx = i // chunk_size
        chunk_embeddings = pooled_embeddings[
            i : i + chunk_size
        ]  # List of (hidden_size,) tensors
        chunk_pids = all_pids[i : i + chunk_size]

        chunk_file = join(chunks_dir, f"embeddings_chunk_{idx}.pt")
        torch.save(chunk_embeddings, chunk_file)

        chunk_pids_file = join(chunks_dir, f"pids_chunk_{idx}.pt")
        torch.save(chunk_pids, chunk_pids_file)

        summary["chunks"].append(
            {
                "embeddings": os.path.relpath(chunk_file, cfg.paths.embeddings),
                "pids": os.path.relpath(chunk_pids_file, cfg.paths.embeddings),
                "start": i,
                "end": min(i + chunk_size, len(pooled_embeddings)),
            }
        )
        print(
            f"Saved chunk {idx + 1}/{(len(pooled_embeddings) + chunk_size - 1) // chunk_size}"
        )

    summary_path = join(cfg.paths.embeddings, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f)


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_evaluate_pretrain(args.config_path)
