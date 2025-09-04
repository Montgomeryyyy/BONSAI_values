from typing import List, Dict, Any
import torch
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import logging

from corebehrt.modules.preparation.dataset import DecoderDataset, PatientDataset
from corebehrt.constants.data import DEFAULT_VOCABULARY, EOS_TOKEN

logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def _get_generation_config(cfg: dict) -> dict:
    """
    Get the generation configuration from the config.
    """
    # Get generation configuration
    generation_config = cfg.get("sequence_generation")
    if generation_config is None:
        raise ValueError("sequence_generation configuration is required but not found in config")
    
    # Check for required parameters
    required_params = ['max_length', 'do_sample', 'temperature', 'top_p', 'top_k', 'repetition_penalty']
    missing_params = [param for param in required_params if param not in generation_config]
    if missing_params:
        raise ValueError(f"Missing required generation parameters: {missing_params}")
    
    # Add eos_token_id if not present
    if 'eos_token_id' not in generation_config:
        generation_config['eos_token_id'] = DEFAULT_VOCABULARY[EOS_TOKEN]
    
    return generation_config

def generate_sequences(
    cfg: dict,
    test_dataset: DecoderDataset,
    vocab: Dict[str, int],
    logger,
    model: torch.nn.Module,
    predict_all_embeddings: bool = False
) -> Dict[str, Any]:
    """
    Generate sequences using the decoder model.
    
    Args:
        cfg: Configuration dictionary
        test_dataset: Test dataset
        vocab: Vocabulary dictionary
        logger: Logger instance
        model: Pre-loaded model to use for generation
        predict_all_embeddings: Whether to predict all embeddings or just concepts
        
    Returns:
        Dictionary containing generated sequences and metadata
    """
    logger.info("Starting sequence generation")
    
    # Get generation configuration
    generation_config = _get_generation_config(cfg)
    
    # Get dataloader
    from corebehrt.modules.trainer.trainer import EHRTrainer
    trainer = EHRTrainer(
        model=model,
        test_dataset=test_dataset,
        args=cfg.trainer_args,
        cfg=cfg,
    )
    dataloader = trainer.get_dataloader(test_dataset, mode="test")
    
    generated_sequences = []
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            trainer.batch_to_device(batch)
            
            # Debug: Show input batch info
            if batch_idx == 0:
                print(f"\n=== GENERATION DEBUG ===")
                print(f"Input batch shape: {batch['concept'].shape}")
                print(f"Sample input sequence: {batch['concept'][0][:10].cpu().tolist()}")
                print(f"Input tokens: {[vocab.get(token_id, f'<UNK_{token_id}>') for token_id in batch['concept'][0][:10].cpu().tolist()]}")
            
            # Generate sequences using the model's generate method
            generated_outputs = model.generate(
                batch, 
                predict_all_embeddings=predict_all_embeddings,
                **generation_config
            )
            
            # Debug: Show generated output info
            if batch_idx == 0:
                print(f"Generated output shape: {generated_outputs['concepts'].shape}")
                print(f"Sample generated sequence: {generated_outputs['concepts'][0][:20].cpu().tolist()}")
                print(f"Generated tokens: {[vocab.get(token_id, f'<UNK_{token_id}>') for token_id in generated_outputs['concepts'][0][:20].cpu().tolist()]}")
                
                # Check for target outcomes in first few sequences
                for i in range(min(3, batch['concept'].size(0))):
                    generated_tokens = [vocab.get(token_id, f'<UNK_{token_id}>') for token_id in generated_outputs['concepts'][i].cpu().tolist()]
                    print(f"Sequence {i} generated tokens: {generated_tokens}")
            
            # Store generated sequences with metadata
            batch_size = batch["concept"].size(0)
            for i in range(batch_size):
                # Post-process generated sequence to remove excessive padding
                generated_concepts = generated_outputs['concepts'][i].cpu().tolist()
                
                # Find the first EOS token or excessive padding
                eos_token_id = 6  # <|EOS|> token ID
                pad_token_id = 0  # [PAD] token ID
                
                # Look for EOS token first
                eos_pos = -1
                for pos, token_id in enumerate(generated_concepts):
                    if token_id == eos_token_id:
                        eos_pos = pos
                        break
                
                # If no EOS, look for excessive padding (more than 3 consecutive PAD tokens)
                if eos_pos == -1:
                    consecutive_pad = 0
                    for pos, token_id in enumerate(generated_concepts):
                        if token_id == pad_token_id:
                            consecutive_pad += 1
                            if consecutive_pad > 3:  # More than 3 consecutive PADs
                                eos_pos = pos - consecutive_pad
                                break
                        else:
                            consecutive_pad = 0
                
                # Trim the sequence
                if eos_pos > 0:
                    generated_concepts = generated_concepts[:eos_pos]
                
                seq_data = {
                    'patient_index': batch_idx * cfg.trainer_args.get('batch_size', 32) + i,
                    'original_sequence': batch["concept"][i].cpu().tolist(),
                    'generated_sequence': generated_concepts,
                    'original_length': batch["concept"][i].size(0),
                    'generated_length': len(generated_concepts),
                }
                
                # Only include additional embeddings if predict_all_embeddings is True
                if predict_all_embeddings:
                    seq_data.update({
                        'generated_segments': generated_outputs['segments'][i].cpu().tolist(),
                        'generated_ages': generated_outputs['ages'][i].cpu().tolist(),
                        'generated_abspos': generated_outputs['abspos'][i].cpu().tolist()
                    })
                
                generated_sequences.append(seq_data)
    
    # Prepare results
    results = {
        'generated_sequences': generated_sequences,
        'total_sequences': len(generated_sequences),
        'generation_config': generation_config,
        'vocab_size': len(vocab)
    }
    
    logger.info(f"Sequence generation completed. Total sequences: {len(generated_sequences)}")
    
    return results

def evaluate_generated_sequences(
    generated_sequences: List[Dict[str, Any]],
    test_data: PatientDataset,
    vocab: Dict[str, int],
    outcomes: List[str]
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluate generated sequences against real outcomes.
    
    Args:
        generated_sequences: List of generated sequences with metadata
        outcomes: List of outcomes to evaluate against
        
    Returns:
        Tuple of (DataFrame with detailed results, Dictionary with summary metrics)
    """
    
    detailed_results = []
    
    for i, seq_data in enumerate(generated_sequences):
        # Get corresponding patient data
        patient_index = int(seq_data['patient_index'])  # Ensure it's an integer
        patient_data = test_data.patients[patient_index]
        
        # Decode sequences
        original_seq = [vocab.get(token_id, f"<UNK_{token_id}>") for token_id in seq_data['original_sequence']]
        generated_seq = [vocab.get(token_id, f"<UNK_{token_id}>") for token_id in seq_data['generated_sequence']]
        
        target_outcome_detected = 1 if any(outcome in generated_seq for outcome in outcomes) else 0
        
        # Calculate basic sequence quality metrics
        sequence_metrics = {
            'length': len(generated_seq),
            'unique_tokens': len(set(generated_seq)),
            'vocabulary_coverage': len(set(generated_seq)) / len(vocab) if len(vocab) > 0 else 0
        }
        
        detailed_results.append({
            'pid': patient_data.pid,
            'target_outcome_detected': target_outcome_detected,
            'original_outcome': 1 if not pd.isna(patient_data.outcome) else 0,
            'sequence_length': len(generated_seq),
            'sequence_quality_length': sequence_metrics['length'],
            'sequence_quality_unique_tokens': sequence_metrics['unique_tokens'],
            'sequence_quality_vocabulary_coverage': sequence_metrics['vocabulary_coverage'],
        })
    
    # Create DataFrame
    df_results = pd.DataFrame(detailed_results)
    
    # Calculate summary metrics
    summary_metrics = {}
    if len(df_results) > 0:
        total_sequences = len(df_results)
        target_outcomes_detected = df_results['target_outcome_detected'].sum()
        
        # Collect original outcomes and detected outcomes for accuracy calculation
        original_outcomes = df_results['original_outcome'].tolist()
        detected_outcomes = df_results['target_outcome_detected'].tolist()
        
        # Calculate TP, TN, FP, FN
        tp = sum(1 for orig, det in zip(original_outcomes, detected_outcomes) if orig == 1 and det == 1)
        tn = sum(1 for orig, det in zip(original_outcomes, detected_outcomes) if orig == 0 and det == 0)
        fp = sum(1 for orig, det in zip(original_outcomes, detected_outcomes) if orig == 0 and det == 1)
        fn = sum(1 for orig, det in zip(original_outcomes, detected_outcomes) if orig == 1 and det == 0)
        
        # Calculate additional metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        positive_class_ratio = sum(original_outcomes) / len(original_outcomes)
        
        summary_metrics = {
            'total_sequences': total_sequences,
            'target_outcomes_detected': target_outcomes_detected,
            'target_outcome_detection_rate': target_outcomes_detected / total_sequences if total_sequences > 0 else 0,
            'accuracy': accuracy_score(original_outcomes, detected_outcomes),
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'positive_class_ratio': positive_class_ratio,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
        
        # Add warning if all outcomes are the same class
        unique_outcomes = set(original_outcomes)
        if len(unique_outcomes) < 2:
            logger.warning(f"All outcomes are the same class ({unique_outcomes}). This may indicate a data issue.")
        
        # Convert numpy types to native Python types for JSON serialization
        summary_metrics = convert_numpy_types(summary_metrics)
    
    return df_results, summary_metrics

def calculate_outcome_probabilities(
    generated_sequences: List[Dict[str, Any]],
    test_data: PatientDataset,
    vocab: Dict[str, int],
    outcomes: List[str]
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Calculate outcome probabilities from generated sequences.
    
    Args:
        generated_sequences: List of generated sequences with metadata
        outcomes: List of outcomes to evaluate against
        
    Returns:
        Tuple of (DataFrame with detailed results, Dictionary with summary metrics)
    """
    
    detailed_results = []
    
    for i, seq_data in enumerate(generated_sequences):
        # Get corresponding patient data
        patient_index = int(seq_data['patient_index'])  # Ensure it's an integer
        patient_data = test_data.patients[patient_index]
        
        # Decode sequences
        original_seq = [vocab.get(token_id, f"<UNK_{token_id}>") for token_id in seq_data['original_sequence']]
        generated_seq = [vocab.get(token_id, f"<UNK_{token_id}>") for token_id in seq_data['generated_sequence']]
        
        # Calculate probability of each outcome appearing in the sequence
        outcome_probabilities = {}
        for outcome in outcomes:
            # Count occurrences of the outcome in the generated sequence
            outcome_count = generated_seq.count(outcome)
            
            # Improved probability calculation:
            # 1. If outcome appears, use a higher base probability
            # 2. Scale by frequency but with diminishing returns
            if outcome_count > 0:
                # Use a sigmoid-like function: 0.5 + 0.4 * (1 - exp(-count))
                outcome_prob = 0.5 + 0.4 * (1 - np.exp(-outcome_count))
            else:
                outcome_prob = 0.1  # Small baseline probability for no occurrence
            
            outcome_probabilities[outcome] = outcome_prob
        
        # Calculate overall outcome probability (max probability across all outcomes)
        max_outcome_prob = max(outcome_probabilities.values()) if outcome_probabilities else 0.1
        
        # Calculate basic sequence quality metrics
        sequence_metrics = {
            'length': len(generated_seq),
            'unique_tokens': len(set(generated_seq)),
            'vocabulary_coverage': len(set(generated_seq)) / len(vocab) if len(vocab) > 0 else 0
        }
        
        result_row = {
            'pid': patient_data.pid,
            'original_outcome': 1 if not pd.isna(patient_data.outcome) else 0,
            'sequence_length': len(generated_seq),
            'sequence_quality_length': sequence_metrics['length'],
            'sequence_quality_unique_tokens': sequence_metrics['unique_tokens'],
            'sequence_quality_vocabulary_coverage': sequence_metrics['vocabulary_coverage'],
            'max_outcome_probability': max_outcome_prob,
        }
        
        # Add individual outcome probabilities
        for outcome in outcomes:
            result_row[f'prob_{outcome}'] = outcome_probabilities[outcome]
        
        detailed_results.append(result_row)
    
    # Create DataFrame
    df_results = pd.DataFrame(detailed_results)
    
    # Calculate summary metrics
    summary_metrics = {}
    if len(df_results) > 0:
        total_sequences = len(df_results)
        
        # Calculate AUC using probabilities
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
        
        original_outcomes = df_results['original_outcome'].tolist()
        outcome_probs = df_results['max_outcome_probability'].tolist()
        
        # Check if we have both positive and negative classes
        unique_outcomes = set(original_outcomes)
        if len(unique_outcomes) < 2:
            # If all outcomes are the same class, AUC is undefined
            auc_score = 0.5
            pr_auc_score = 0.5
            logger.warning(f"All outcomes are the same class ({unique_outcomes}). AUC set to 0.5.")
        else:
            try:
                auc_score = roc_auc_score(original_outcomes, outcome_probs)
                # Also calculate PR-AUC
                precision, recall, _ = precision_recall_curve(original_outcomes, outcome_probs)
                pr_auc_score = auc(recall, precision)
            except Exception as e:
                logger.warning(f"Error calculating AUC: {e}. Setting to 0.5.")
                auc_score = 0.5
                pr_auc_score = 0.5
        
        # Try different thresholds for binary predictions
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        best_accuracy = 0
        best_threshold = 0.1
        
        for threshold in thresholds:
            binary_predictions = [1 if prob >= threshold else 0 for prob in outcome_probs]
            accuracy = sum(1 for orig, pred in zip(original_outcomes, binary_predictions) if orig == pred) / len(original_outcomes)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        # Use best threshold for final binary predictions
        binary_predictions = [1 if prob >= best_threshold else 0 for prob in outcome_probs]
        
        # Calculate TP, TN, FP, FN
        tp = sum(1 for orig, pred in zip(original_outcomes, binary_predictions) if orig == 1 and pred == 1)
        tn = sum(1 for orig, pred in zip(original_outcomes, binary_predictions) if orig == 0 and pred == 0)
        fp = sum(1 for orig, pred in zip(original_outcomes, binary_predictions) if orig == 0 and pred == 1)
        fn = sum(1 for orig, pred in zip(original_outcomes, binary_predictions) if orig == 1 and pred == 0)
        
        summary_metrics = {
            'total_sequences': total_sequences,
            'auc_score': auc_score,
            'pr_auc_score': pr_auc_score,
            'best_threshold': best_threshold,
            'best_accuracy': best_accuracy,
            'mean_outcome_probability': df_results['max_outcome_probability'].mean(),
            'std_outcome_probability': df_results['max_outcome_probability'].std(),
            'min_outcome_probability': df_results['max_outcome_probability'].min(),
            'max_outcome_probability': df_results['max_outcome_probability'].max(),
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'positive_class_ratio': sum(original_outcomes) / len(original_outcomes)
        }
        
        # Convert numpy types to native Python types for JSON serialization
        summary_metrics = convert_numpy_types(summary_metrics)
    
    return df_results, summary_metrics

def analyze_generated_sequences(
    generated_sequences: List[Dict[str, Any]],
    test_data: PatientDataset,
    vocab: Dict[str, int],
    outcomes: List[str]
) -> Dict[str, Any]:
    """
    Analyze generated sequences to understand model performance and quality.
    
    Args:
        generated_sequences: List of generated sequences with metadata
        test_data: PatientDataset containing original data
        vocab: Vocabulary dictionary
        outcomes: List of outcomes to look for
        
    Returns:
        Dictionary with analysis results
    """
    
    analysis = {
        'total_sequences': len(generated_sequences),
        'outcomes_to_find': outcomes,
        'sequence_analysis': {},
        'outcome_analysis': {},
        'quality_metrics': {}
    }
    
    if len(generated_sequences) == 0:
        return analysis
    
    # Analyze sequence lengths
    lengths = []
    unique_token_counts = []
    vocab_coverage = []
    outcome_counts = {outcome: 0 for outcome in outcomes}
    total_outcome_occurrences = 0
    
    for seq_data in generated_sequences:
        generated_seq = [vocab.get(token_id, f"<UNK_{token_id}>") for token_id in seq_data['generated_sequence']]
        
        # Sequence length analysis
        lengths.append(len(generated_seq))
        unique_token_counts.append(len(set(generated_seq)))
        vocab_coverage.append(len(set(generated_seq)) / len(vocab) if len(vocab) > 0 else 0)
        
        # Outcome analysis
        for outcome in outcomes:
            if outcome in generated_seq:
                outcome_counts[outcome] += 1
                total_outcome_occurrences += generated_seq.count(outcome)
    
    # Sequence quality metrics
    analysis['sequence_analysis'] = {
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'min_length': np.min(lengths),
        'max_length': np.max(lengths),
        'mean_unique_tokens': np.mean(unique_token_counts),
        'mean_vocab_coverage': np.mean(vocab_coverage),
        'sequences_with_outcomes': sum(1 for count in outcome_counts.values() if count > 0),
        'total_outcome_occurrences': total_outcome_occurrences
    }
    
    # Outcome-specific analysis
    analysis['outcome_analysis'] = {
        'outcome_counts': outcome_counts,
        'outcome_detection_rates': {
            outcome: count / len(generated_sequences) 
            for outcome, count in outcome_counts.items()
        }
    }
    
    # Check for common issues
    issues = []
    
    # Check if sequences are too short
    if np.mean(lengths) < 5:
        issues.append(f"Sequences are very short (mean: {np.mean(lengths):.2f})")
    
    # Check if vocabulary coverage is too low
    if np.mean(vocab_coverage) < 0.01:
        issues.append(f"Very low vocabulary coverage (mean: {np.mean(vocab_coverage):.4f})")
    
    # Check if outcomes are rarely generated
    total_outcome_sequences = sum(1 for count in outcome_counts.values() if count > 0)
    if total_outcome_sequences < len(generated_sequences) * 0.1:
        issues.append(f"Outcomes rarely generated ({total_outcome_sequences}/{len(generated_sequences)} sequences)")
    
    # Check for repetitive sequences
    if np.mean(unique_token_counts) < np.mean(lengths) * 0.3:
        issues.append("Sequences appear to be repetitive (low unique token ratio)")
    
    analysis['quality_metrics'] = {
        'issues_detected': issues,
        'sequence_diversity': np.mean(unique_token_counts) / np.mean(lengths) if np.mean(lengths) > 0 else 0,
        'outcome_generation_rate': total_outcome_sequences / len(generated_sequences)
    }
    
    return analysis
