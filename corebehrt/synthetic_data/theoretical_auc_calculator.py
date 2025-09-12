"""
Comprehensive theoretical AUC calculator for synthetic data generation.

This module provides various methods to calculate theoretical ROC AUC values
for different scenarios in synthetic data generation, particularly for
multi-lab switching scenarios.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Tuple, Optional


def calculate_basic_distribution_auc(mean1: float, mean2: float, std: float) -> float:
    """
    Calculate theoretical AUC for two normal distributions with equal variance.
    
    This is the most basic theoretical AUC calculation, assuming perfect
    classification based on individual lab values.
    
    Formula: AUC = Φ((μ₁ - μ₂) / (σ√2))
    where Φ is the standard normal CDF.
    
    Args:
        mean1: Mean of first distribution
        mean2: Mean of second distribution  
        std: Standard deviation (assumed same for both)
        
    Returns:
        float: Theoretical AUC (0.5 to 1.0)
    """
    if std == 0:
        return 1.0 if mean1 != mean2 else 0.5
    
    z_score = (mean1 - mean2) / (std * np.sqrt(2))
    theoretical_auc = norm.cdf(z_score)
    return theoretical_auc


def calculate_switch_detection_auc(mean1: float, mean2: float, std: float, 
                                 n_labs: int = 5, switch_prob: float = 1.0) -> float:
    """
    Calculate theoretical AUC for switch detection between distributions.
    
    This calculates the probability of correctly identifying high-risk patients
    (who switch distributions) vs low-risk patients (who don't switch).
    
    Args:
        mean1: Mean of first distribution
        mean2: Mean of second distribution  
        std: Standard deviation (assumed same for both)
        n_labs: Average number of labs per patient
        switch_prob: Probability of switching for high-risk patients
        
    Returns:
        float: Theoretical AUC for switch detection
    """
    if std == 0:
        return 1.0 if mean1 != mean2 else 0.5
    
    threshold = (mean1 + mean2) / 2
    
    # Probability that a value from distribution 1 is below threshold
    p1_below = norm.cdf((threshold - mean1) / std)
    
    # Probability that a value from distribution 2 is above threshold  
    p2_above = 1 - norm.cdf((threshold - mean2) / std)
    
    # Probability of detecting a switch from dist1 to dist2
    p_switch_detected = p1_below * p2_above
    
    # Probability of false switch detection (due to noise)
    p_false_switch = 2 * norm.pdf((threshold - mean1) / std) * norm.pdf((threshold - mean2) / std) * std
    
    # Calculate theoretical AUC based on switch detection probability
    if abs(mean1 - mean2) > 6 * std:  # Essentially no overlap
        theoretical_auc = 1.0
    else:
        # More sophisticated calculation for overlapping case
        separation = abs(mean1 - mean2) / std
        theoretical_auc = norm.cdf(separation / 2)
    
    return theoretical_auc


def calculate_sequence_variance_auc(mean1: float, mean2: float, std: float, 
                                  n_labs: int = 5) -> float:
    """
    Calculate theoretical AUC based on sequence variance differences.
    
    High-risk patients (who switch) have higher variance in their lab sequences
    compared to low-risk patients (who don't switch).
    
    Args:
        mean1: Mean of first distribution
        mean2: Mean of second distribution  
        std: Standard deviation (assumed same for both)
        n_labs: Average number of labs per patient
        
    Returns:
        float: Theoretical AUC based on variance differences
    """
    if std == 0:
        return 1.0 if mean1 != mean2 else 0.5
    
    # Theoretical variance for high-risk patients (mixture of two distributions)
    # Var(X) = E[X²] - E[X]² = 0.5 * (Var₁ + Var₂ + (μ₁-μ₂)²/4)
    high_risk_variance = std**2 + (mean1 - mean2)**2 / 4
    
    # Theoretical variance for low-risk patients (single distribution)
    low_risk_variance = std**2
    
    # Calculate theoretical AUC based on variance difference
    variance_ratio = high_risk_variance / low_risk_variance
    
    # Convert variance ratio to AUC (approximation)
    if variance_ratio > 2.0:  # Significant difference
        theoretical_auc = 0.8 + 0.2 * min(1.0, (variance_ratio - 2.0) / 3.0)
    else:
        # Use the basic distribution separation
        separation = abs(mean1 - mean2) / std
        theoretical_auc = norm.cdf(separation / 2)
    
    return theoretical_auc


def calculate_trend_change_auc(mean1: float, mean2: float, std: float, 
                             n_labs: int = 5) -> float:
    """
    Calculate theoretical AUC for trend change detection.
    
    This considers the ability to detect trend changes in lab sequences,
    which is characteristic of high-risk patients who switch distributions.
    
    Args:
        mean1: Mean of first distribution
        mean2: Mean of second distribution  
        std: Standard deviation (assumed same for both)
        n_labs: Average number of labs per patient
        
    Returns:
        float: Theoretical AUC for trend change detection
    """
    if std == 0:
        return 1.0 if mean1 != mean2 else 0.5
    
    # Calculate the probability of detecting a trend change
    change_magnitude = abs(mean1 - mean2)
    noise_level = std
    
    # Signal-to-noise ratio for trend detection
    snr = change_magnitude / noise_level
    
    # Theoretical AUC based on signal-to-noise ratio
    if snr > 3.0:  # Strong signal
        theoretical_auc = 0.9 + 0.1 * min(1.0, (snr - 3.0) / 2.0)
    elif snr > 1.0:  # Moderate signal
        theoretical_auc = 0.7 + 0.2 * (snr - 1.0) / 2.0
    else:  # Weak signal
        theoretical_auc = 0.5 + 0.2 * snr
    
    return theoretical_auc


def calculate_mixture_model_auc(mean1: float, mean2: float, std: float, 
                              n_labs: int = 5, mixing_ratio: float = 0.5) -> float:
    """
    Calculate theoretical AUC for mixture model classification.
    
    This considers high-risk patients as a mixture of two distributions,
    while low-risk patients come from a single distribution.
    
    Args:
        mean1: Mean of first distribution
        mean2: Mean of second distribution  
        std: Standard deviation (assumed same for both)
        n_labs: Average number of labs per patient
        mixing_ratio: Ratio of mixing between distributions for high-risk patients
        
    Returns:
        float: Theoretical AUC for mixture model classification
    """
    if std == 0:
        return 1.0 if mean1 != mean2 else 0.5
    
    # High-risk patients: mixture of two distributions
    # Low-risk patients: single distribution (randomly chosen)
    
    # Calculate the probability of correctly classifying based on mixture detection
    # This is complex, so we use an approximation
    
    separation = abs(mean1 - mean2) / std
    
    # For mixture models, the theoretical AUC depends on the separation
    # and the mixing ratio
    if separation > 4.0:  # Well-separated distributions
        theoretical_auc = 0.95 + 0.05 * min(1.0, (separation - 4.0) / 2.0)
    elif separation > 2.0:  # Moderately separated
        theoretical_auc = 0.8 + 0.15 * (separation - 2.0) / 2.0
    elif separation > 1.0:  # Poorly separated
        theoretical_auc = 0.6 + 0.2 * (separation - 1.0)
    else:  # Very poorly separated
        theoretical_auc = 0.5 + 0.1 * separation
    
    return theoretical_auc


def calculate_comprehensive_theoretical_aucs(mean1: float, mean2: float, std: float,
                                           n_labs: int = 5, switch_prob: float = 1.0,
                                           mixing_ratio: float = 0.5) -> Dict[str, float]:
    """
    Calculate all theoretical AUC values for comprehensive analysis.
    
    Args:
        mean1: Mean of first distribution
        mean2: Mean of second distribution  
        std: Standard deviation (assumed same for both)
        n_labs: Average number of labs per patient
        switch_prob: Probability of switching for high-risk patients
        mixing_ratio: Ratio of mixing between distributions for high-risk patients
        
    Returns:
        Dict[str, float]: Dictionary containing all theoretical AUC values
    """
    return {
        "basic_distribution_auc": calculate_basic_distribution_auc(mean1, mean2, std),
        "switch_detection_auc": calculate_switch_detection_auc(mean1, mean2, std, n_labs, switch_prob),
        "sequence_variance_auc": calculate_sequence_variance_auc(mean1, mean2, std, n_labs),
        "trend_change_auc": calculate_trend_change_auc(mean1, mean2, std, n_labs),
        "mixture_model_auc": calculate_mixture_model_auc(mean1, mean2, std, n_labs, mixing_ratio),
    }


def print_theoretical_auc_summary(mean1: float, mean2: float, std: float,
                                n_labs: int = 5, switch_prob: float = 1.0,
                                mixing_ratio: float = 0.5) -> None:
    """
    Print a comprehensive summary of theoretical AUC values.
    
    Args:
        mean1: Mean of first distribution
        mean2: Mean of second distribution  
        std: Standard deviation (assumed same for both)
        n_labs: Average number of labs per patient
        switch_prob: Probability of switching for high-risk patients
        mixing_ratio: Ratio of mixing between distributions for high-risk patients
    """
    aucs = calculate_comprehensive_theoretical_aucs(mean1, mean2, std, n_labs, switch_prob, mixing_ratio)
    
    print("\n" + "="*60)
    print("THEORETICAL AUC CALCULATIONS")
    print("="*60)
    print(f"Distribution Parameters:")
    print(f"  Mean 1: {mean1:.3f}")
    print(f"  Mean 2: {mean2:.3f}")
    print(f"  Std: {std:.3f}")
    print(f"  Separation: {abs(mean1 - mean2) / std:.3f} standard deviations")
    print(f"  N labs: {n_labs}")
    print(f"  Switch probability: {switch_prob:.1%}")
    print(f"  Mixing ratio: {mixing_ratio:.1%}")
    
    print(f"\nTheoretical AUC Values:")
    print(f"  Basic Distribution AUC: {aucs['basic_distribution_auc']:.4f}")
    print(f"  Switch Detection AUC: {aucs['switch_detection_auc']:.4f}")
    print(f"  Sequence Variance AUC: {aucs['sequence_variance_auc']:.4f}")
    print(f"  Trend Change AUC: {aucs['trend_change_auc']:.4f}")
    print(f"  Mixture Model AUC: {aucs['mixture_model_auc']:.4f}")
    
    # Calculate average and range
    values = list(aucs.values())
    avg_auc = np.mean(values)
    min_auc = np.min(values)
    max_auc = np.max(values)
    
    print(f"\nSummary:")
    print(f"  Average AUC: {avg_auc:.4f}")
    print(f"  Range: {min_auc:.4f} - {max_auc:.4f}")
    print(f"  AUC Spread: {max_auc - min_auc:.4f}")
    
    print("="*60)


if __name__ == "__main__":
    # Example usage
    print_theoretical_auc_summary(
        mean1=0.65,
        mean2=0.35, 
        std=0.05,
        n_labs=5,
        switch_prob=1.0,
        mixing_ratio=0.5
    )
