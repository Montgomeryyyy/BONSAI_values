#!/usr/bin/env python3
"""
Example script demonstrating how to calculate theoretical ROC AUCs
for different synthetic data scenarios.
"""

import sys
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from theoretical_auc_calculator import (
    calculate_comprehensive_theoretical_aucs,
    print_theoretical_auc_summary
)


def main():
    """Demonstrate theoretical AUC calculations for different scenarios."""
    
    print("Theoretical AUC Calculator Examples")
    print("=" * 50)
    
    # Scenario 1: Well-separated distributions (high theoretical performance)
    print("\nScenario 1: Well-separated distributions")
    print("-" * 40)
    print_theoretical_auc_summary(
        mean1=0.8,      # High-risk mean
        mean2=0.2,      # Low-risk mean
        std=0.05,       # Low noise
        n_labs=5,
        switch_prob=1.0,
        mixing_ratio=0.5
    )
    
    # Scenario 2: Overlapping distributions (moderate theoretical performance)
    print("\nScenario 2: Overlapping distributions")
    print("-" * 40)
    print_theoretical_auc_summary(
        mean1=0.6,      # High-risk mean
        mean2=0.4,      # Low-risk mean
        std=0.1,        # Higher noise
        n_labs=5,
        switch_prob=1.0,
        mixing_ratio=0.5
    )
    
    # Scenario 3: Poorly separated distributions (low theoretical performance)
    print("\nScenario 3: Poorly separated distributions")
    print("-" * 40)
    print_theoretical_auc_summary(
        mean1=0.55,     # High-risk mean
        mean2=0.45,     # Low-risk mean
        std=0.15,       # High noise
        n_labs=5,
        switch_prob=1.0,
        mixing_ratio=0.5
    )
    
    # Scenario 4: Different number of labs
    print("\nScenario 4: More lab values per patient")
    print("-" * 40)
    print_theoretical_auc_summary(
        mean1=0.65,     # High-risk mean
        mean2=0.35,     # Low-risk mean
        std=0.05,       # Low noise
        n_labs=10,      # More labs
        switch_prob=1.0,
        mixing_ratio=0.5
    )
    
    # Scenario 5: Partial switching probability
    print("\nScenario 5: Partial switching probability")
    print("-" * 40)
    print_theoretical_auc_summary(
        mean1=0.65,     # High-risk mean
        mean2=0.35,     # Low-risk mean
        std=0.05,       # Low noise
        n_labs=5,
        switch_prob=0.7,  # Only 70% of high-risk patients switch
        mixing_ratio=0.5
    )


if __name__ == "__main__":
    main()
