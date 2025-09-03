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
    
    # Load vocabulary directly from the model's saved vocabulary file
    # The model was trained with this vocabulary, so we need to use it for decoding
    print(f"\n=== VOCABULARY LOADING DEBUG ===")
    print(f"Looking for vocabulary in: {cfg.paths.model}")
    
    # Check what files exist in the model directory
    import os
    model_files = os.listdir(cfg.paths.model)
    print(f"Files in model directory: {model_files}")
    
    # Try different possible vocabulary file names
    possible_vocab_files = [
        "vocabulary.pt",
        "vocab.pt", 
        "tokenizer.pt",
        "vocab.json",
        "tokenizer.json"
    ]
    
    vocab = None
    vocab_source = None
    
    for vocab_file in possible_vocab_files:
        vocab_path = join(cfg.paths.model, vocab_file)
        if os.path.exists(vocab_path):
            print(f"Found vocabulary file: {vocab_file}")
            try:
                if vocab_file.endswith('.pt'):
                    vocab = torch.load(vocab_path)
                elif vocab_file.endswith('.json'):
                    import json
                    with open(vocab_path, 'r') as f:
                        vocab = json.load(f)
                vocab_source = vocab_path
                print(f"✓ Successfully loaded vocabulary from {vocab_file}")
                break
            except Exception as e:
                print(f"✗ Failed to load {vocab_file}: {e}")
                continue
    
    if vocab is None:
        # Fallback to load_vocabulary function
        from corebehrt.functional.io_operations.load import load_vocabulary
        print("⚠ No vocabulary file found, trying load_vocabulary function...")
        vocab = load_vocabulary(cfg.paths.model)
        vocab_source = "load_vocabulary function"
    
    if vocab is None:
        raise ValueError("Could not load vocabulary from any source!")
    
    print(f"Vocabulary loaded from: {vocab_source}")
    
    # Create reverse vocabulary mapping (id -> token) for decoding
    # The vocabulary is currently token -> id, but we need id -> token for decoding
    reverse_vocab = {int(v): k for k, v in vocab.items()}
    print(f"Created reverse vocabulary mapping (id -> token)")
    print(f"Reverse vocabulary size: {len(reverse_vocab)}")
    
    # Test the reverse mapping with some known tokens
    test_tokens = [0, 1, 2, 5, 681]  # PAD, CLS, SEP, BOS, DE11
    print(f"Testing reverse mapping:")
    for token_id in test_tokens:
        if token_id in reverse_vocab:
            print(f"  {token_id} -> '{reverse_vocab[token_id]}'")
        else:
            print(f"  {token_id} -> NOT FOUND")
    
    # Use reverse_vocab for decoding - this should be id -> token
    vocab = reverse_vocab
    
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
    print(f"Vocabulary loaded from: {cfg.paths.model}")
    
    # Check if target outcomes are in vocabulary (now check by token ID)
    for outcome in target_outcomes:
        # First check if the outcome string exists in the original vocabulary
        if outcome in vocab:
            print(f"✓ Outcome '{outcome}' found in vocabulary with ID: {vocab[outcome]}")
        else:
            # Check if it's a numeric ID
            try:
                outcome_id = int(outcome)
                if outcome_id in vocab:
                    print(f"✓ Outcome ID {outcome_id} found in vocabulary with token: '{vocab[outcome_id]}'")
                else:
                    print(f"✗ Outcome '{outcome}' NOT found in vocabulary!")
            except ValueError:
                print(f"✗ Outcome '{outcome}' NOT found in vocabulary!")
    
    # Show some sample vocabulary entries
    print(f"\nSample vocabulary entries:")
    sample_entries = list(vocab.items())[:10]
    for token_id, token in sample_entries:
        print(f"  {token_id} -> '{token}'")
    
    # Check for any tokens that might be outcomes
    outcome_like_tokens = [token for token in vocab.values() if token.startswith('DE') or token.startswith('DI') or token.startswith('S/')]
    print(f"\nOutcome-like tokens in vocabulary: {outcome_like_tokens[:10]}...")
    
    # Debug: Check if the vocabulary is actually a dict and has the right structure
    print(f"\nVocabulary type: {type(vocab)}")
    if isinstance(vocab, dict):
        print(f"Vocabulary keys type: {type(list(vocab.keys())[0]) if vocab else 'N/A'}")
        print(f"Vocabulary values type: {type(list(vocab.values())[0]) if vocab else 'N/A'}")
        
        # Verify we have the correct structure (id -> token)
        if vocab and isinstance(list(vocab.keys())[0], int):
            print("✓ Vocabulary has correct structure: id -> token")
        else:
            print("⚠ Vocabulary structure issue detected!")
            # Force the correct structure
            if vocab and isinstance(list(vocab.keys())[0], str):
                print("Converting string keys to integer keys...")
                vocab = {int(k): v for k, v in vocab.items()}
                print("✓ Fixed vocabulary structure")

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
