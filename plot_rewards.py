"""
Script to plot and compare learning curves from different reward functions.

This script loads training metrics from CSV files and generates a comparison plot
of the learning performance for baseline vs shaped reward functions.

Usage:
    python plot_rewards.py
"""

import os
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_metrics(log_dir):
    """
    Load metrics from a training log directory.
    
    Args:
        log_dir (str): Path to the log directory
        
    Returns:
        dict: timesteps and rewards arrays, or None if file not found
    """
    csv_path = os.path.join(log_dir, "training_metrics.csv")
    
    if not Path(csv_path).exists():
        print(f"Warning: Metrics file not found at {csv_path}")
        return None
    
    timesteps = []
    rewards = []
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Support both "timesteps" and legacy "timestep" headers.
                    ts_value = row.get('timesteps', row.get('timestep'))
                    if ts_value is None:
                        raise KeyError('timesteps')
                    timesteps.append(int(ts_value))
                    rewards.append(float(row['mean_reward']))
                except (ValueError, KeyError) as e:
                    print(f"Warning: Could not parse row {row}: {e}")
                    continue
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None
    
    if not timesteps:
        print(f"No valid data found in {csv_path}")
        return None
    
    return {
        'timesteps': np.array(timesteps),
        'rewards': np.array(rewards)
    }


def main():
    print("Generating reward comparison plot...")
    
    # Define log directories
    baseline_log_dir = "logs/baseline"
    shaped_log_dir = "logs/shaped"
    output_path = "reward_comparison.png"
    
    # Load metrics
    baseline_metrics = load_metrics(baseline_log_dir)
    shaped_metrics = load_metrics(shaped_log_dir)
    
    if baseline_metrics is None and shaped_metrics is None:
        print("Error: No training data found. Please run training first.")
        print(f"Expected log files at:")
        print(f"  - {baseline_log_dir}/training_metrics.csv")
        print(f"  - {shaped_log_dir}/training_metrics.csv")
        
        # Create a sample plot for demonstration
        print("\nCreating sample plot for demonstration...")
        create_sample_plot(output_path)
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot baseline if available
    if baseline_metrics is not None:
        ax.plot(
            baseline_metrics['timesteps'],
            baseline_metrics['rewards'],
            label='Baseline Reward',
            linewidth=2,
            marker='o',
            markersize=4,
            alpha=0.7
        )
        print(f"Plotted baseline reward: {len(baseline_metrics['timesteps'])} points")
    
    # Plot shaped if available
    if shaped_metrics is not None:
        ax.plot(
            shaped_metrics['timesteps'],
            shaped_metrics['rewards'],
            label='Shaped Reward',
            linewidth=2,
            marker='s',
            markersize=4,
            alpha=0.7
        )
        print(f"Plotted shaped reward: {len(shaped_metrics['timesteps'])} points")
    
    # Customize plot
    ax.set_xlabel('Timesteps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
    ax.set_title('Learning Curves: Baseline vs Shaped Reward Functions', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    
    plt.close()


def create_sample_plot(output_path):
    """Create a sample plot for demonstration when no real data exists."""
    # Generate dummy data
    timesteps = np.linspace(0, 200000, 100)
    
    # Baseline: slower, more linear growth
    baseline_rewards = -2 + 1.5 * (1 - np.exp(-timesteps / 50000)) + np.random.normal(0, 0.1, 100)
    
    # Shaped: faster growth, reaches higher plateau
    shaped_rewards = -2 + 1.8 * (1 - np.exp(-timesteps / 30000)) + np.random.normal(0, 0.15, 100)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(timesteps, baseline_rewards, label='Baseline Reward', linewidth=2, marker='o', markersize=4, alpha=0.7)
    ax.plot(timesteps, shaped_rewards, label='Shaped Reward', linewidth=2, marker='s', markersize=4, alpha=0.7)
    
    ax.set_xlabel('Timesteps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
    ax.set_title('Learning Curves: Baseline vs Shaped Reward Functions\n(Sample Data)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Sample plot saved to {output_path}")
    
    plt.close()


if __name__ == "__main__":
    main()
