import logging
from os.path import join
import pandas as pd
import torch
import os
import numpy as np
import json
from typing import List, Dict, Any

from corebehrt.constants.paths import FOLDS_FILE, PREPARED_ALL_PATIENTS
from corebehrt.constants.data import DEFAULT_VOCABULARY, EOS_TOKEN, BOS_TOKEN
from corebehrt.functional.setup.args import get_args
from corebehrt.main.helper.finetune_cv import check_for_overlap
from corebehrt.main.helper.evaluate_decoder import generate_sequences, evaluate_generated_sequences, calculate_outcome_probabilities, analyze_generated_sequences
from corebehrt.modules.preparation.dataset import DecoderDataset, PatientDataset
from corebehrt.modules.setup.config import load_config
from corebehrt.modules.setup.directory import DirectoryPreparer
from corebehrt.functional.io_operations.load import load_vocabulary
from corebehrt.modules.setup.manager import ModelManager

CONFIG_PATH = "./corebehrt/configs/evaluate_decoder.yaml"


def main_evaluate(config_path):
    # Setup directories
    cfg = load_config(config_path)
    DirectoryPreparer(cfg).setup_evaluate_decoder()

    # Logger
    logger = logging.getLogger("evaluate")

    # Setup config
    cfg.trainer_args = {}
    batch_size_value = cfg.get("test_batch_size", 128)
    for key in ["test_batch_size", "effective_batch_size", "batch_size"]:
        cfg.trainer_args[key] = batch_size_value
    cfg.paths.restart_model = cfg.paths.model
    model_cfg = load_config(join(cfg.paths.model, "train_decoder.yaml"))
    predict_all_embeddings = model_cfg.model.get("predict_all_embeddings", False)
    # cfg.sequence_generation.eos_token_id = DEFAULT_VOCABULARY[EOS_TOKEN]
    # cfg.sequence_generation.bos_token_id = DEFAULT_VOCABULARY[BOS_TOKEN]

    # Load data
    loaded_data = torch.load(
        join(cfg.paths.test_data_dir, PREPARED_ALL_PATIENTS), weights_only=False
    )
    test_data = PatientDataset(loaded_data)
    outcomes = test_data.get_outcomes()
    vocab = load_vocabulary(cfg.paths.test_data_dir)
    test_dataset = DecoderDataset(test_data.patients, vocab)
    test_pids = test_data.get_pids()
    folds = torch.load(join(cfg.paths.model, FOLDS_FILE), weights_only=False)
    check_for_overlap(folds, test_pids, logger)
    targets = [0 if np.isnan(x) else 1 for x in test_data.get_outcomes()]
    logger.info(f"Number of test patients: {len(test_pids)}")
    logger.info(f"Number of test positive targets: {sum(targets)}")

    # Debug vocabulary and outcomes
    target_outcomes = cfg.sequence_evaluation.outcomes
    print(f"\n=== VOCABULARY DEBUG ===")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Target outcomes: {target_outcomes}")
    
    # Check if target outcomes are in vocabulary
    for outcome in target_outcomes:
        if outcome in vocab:
            print(f"✓ Outcome '{outcome}' found in vocabulary with ID: {vocab[outcome]}")
        else:
            print(f"✗ Outcome '{outcome}' NOT found in vocabulary!")
    
    # Show some sample vocabulary entries
    print(f"\nSample vocabulary entries:")
    sample_entries = list(vocab.items())[:10]
    for token, token_id in sample_entries:
        print(f"  '{token}' -> {token_id}")
    
    # Check for any tokens that might be outcomes
    outcome_like_tokens = [token for token in vocab.keys() if token.startswith('DE') or token.startswith('DI')]
    print(f"\nOutcome-like tokens in vocabulary: {outcome_like_tokens[:10]}...")

    # Load model from the first fold
    modelmanager_trained = ModelManager(cfg, fold=None)
    checkpoint = modelmanager_trained.load_checkpoint(checkpoints=True)
    model = modelmanager_trained.initialize_decoder_model(checkpoint, [])

    # Generate sequences
    gen_data = generate_sequences(cfg, test_dataset, vocab, logger, model, predict_all_embeddings=predict_all_embeddings)

    # Analyze generated sequences for debugging
    sequence_analysis = analyze_generated_sequences(gen_data['generated_sequences'], test_data, vocab, target_outcomes)
    
    print("\n=== SEQUENCE ANALYSIS ===")
    print(f"Total sequences generated: {sequence_analysis['total_sequences']}")
    print(f"Sequence quality issues: {sequence_analysis['quality_metrics']['issues_detected']}")
    print(f"Outcome generation rate: {sequence_analysis['quality_metrics']['outcome_generation_rate']:.4f}")
    print(f"Outcome detection rates: {sequence_analysis['outcome_analysis']['outcome_detection_rates']}")

    # Evaluate generated sequences
    target_outcomes = cfg.sequence_evaluation.outcomes
    
    # Binary evaluation
    df_results_binary, summary_metrics_binary = evaluate_generated_sequences(gen_data['generated_sequences'], test_data, vocab, target_outcomes)
    
    # Probability-based evaluation
    df_results_prob, summary_metrics_prob = calculate_outcome_probabilities(gen_data['generated_sequences'], test_data, vocab, target_outcomes)
    
    print("\n=== EVALUATION RESULTS ===")
    print("Binary Results:")
    print(f"  Accuracy: {summary_metrics_binary.get('accuracy', 0):.4f}")
    print(f"  Precision: {summary_metrics_binary.get('precision', 0):.4f}")
    print(f"  Recall: {summary_metrics_binary.get('recall', 0):.4f}")
    print(f"  F1-Score: {summary_metrics_binary.get('f1_score', 0):.4f}")
    
    print("\nProbability Results:")
    print(f"  ROC AUC: {summary_metrics_prob.get('auc_score', 0):.4f}")
    print(f"  PR AUC: {summary_metrics_prob.get('pr_auc_score', 0):.4f}")
    print(f"  Best threshold: {summary_metrics_prob.get('best_threshold', 0):.4f}")
    print(f"  Best accuracy: {summary_metrics_prob.get('best_accuracy', 0):.4f}")
    print(f"  Mean outcome probability: {summary_metrics_prob.get('mean_outcome_probability', 0):.4f}")
    
    # Save evaluation results
    df_results_binary.to_csv(join(cfg.paths.predictions, "eval_gen_data_binary.csv"), index=False)
    df_results_prob.to_csv(join(cfg.paths.predictions, "eval_gen_data_probabilities.csv"), index=False)
    
    # Save summary metrics
    with open(join(cfg.paths.predictions, "summary_metrics_binary.json"), 'w') as f:
        json.dump(summary_metrics_binary, f, indent=2)
    
    with open(join(cfg.paths.predictions, "summary_metrics_probabilities.json"), 'w') as f:
        json.dump(summary_metrics_prob, f, indent=2)
    
    # Save sequence analysis
    with open(join(cfg.paths.predictions, "sequence_analysis.json"), 'w') as f:
        json.dump(sequence_analysis, f, indent=2)
    
    print(f"\nDetailed results saved to: {join(cfg.paths.predictions, 'eval_gen_data_*.csv')}")
    print(f"Analysis saved to: {join(cfg.paths.predictions, 'sequence_analysis.json')}")

    # Check if we should run sequence generation evaluation
#     run_sequence_generation_evaluation(cfg, test_dataset, vocab, folds, logger)


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_evaluate(args.config_path)
