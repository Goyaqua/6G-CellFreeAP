"""
Utility Functions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import json
import os


def plot_training_curves(log_dir: str, save_path: str = None):
    """
    Plot training curves from TensorBoard logs
    
    Args:
        log_dir: Directory containing TensorBoard logs
        save_path: Path to save the plot
    """
    # This is a placeholder - you'd typically use tensorboard data
    # For now, we'll create a simple plotting function
    plt.figure(figsize=(12, 4))
    
    # Placeholder for actual implementation
    print(f"Plot training curves from {log_dir}")
    print("Use TensorBoard for detailed visualization: tensorboard --logdir", log_dir)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_comparison(results: Dict[str, Dict], save_path: str = None):
    """
    Plot comparison between different strategies
    
    Args:
        results: Dictionary mapping strategy names to their metrics
        save_path: Path to save the plot
    """
    metrics = ['mean_energy_efficiency', 'mean_qos_satisfaction', 'mean_rate_mbps']
    metric_labels = ['Energy Efficiency\n(bits/Joule)', 'QoS Satisfaction\n(%)', 'Avg Rate\n(Mbps)']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        strategies = list(results.keys())
        values = [results[s][metric] for s in strategies]
        errors = [results[s].get(f'std_{metric.replace("mean_", "")}', 0) for s in strategies]
        
        axes[idx].bar(strategies, values, yerr=errors, capsize=5, alpha=0.7)
        axes[idx].set_ylabel(label, fontsize=12)
        axes[idx].set_xlabel('Strategy', fontsize=12)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def save_results(results: Dict, filename: str):
    """
    Save results to JSON file
    
    Args:
        results: Results dictionary
        filename: Output filename
    """
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        return obj
    
    results_converted = convert(results)
    
    with open(filename, 'w') as f:
        json.dump(results_converted, f, indent=4)
    
    print(f"Results saved to {filename}")


def load_results(filename: str) -> Dict:
    """
    Load results from JSON file
    
    Args:
        filename: Input filename
        
    Returns:
        Results dictionary
    """
    with open(filename, 'r') as f:
        results = json.load(f)
    
    return results


def print_results_table(results: Dict):
    """
    Print results in a formatted table
    
    Args:
        results: Dictionary mapping strategy names to their metrics
    """
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Strategy':<20} {'EE (bits/J)':<15} {'QoS Sat (%)':<15} {'Avg Rate (Mbps)':<15}")
    print("-"*80)
    
    for strategy, metrics in results.items():
        ee = metrics.get('mean_energy_efficiency', 0)
        qos = metrics.get('mean_qos_satisfaction', 0)
        rate = metrics.get('mean_rate_mbps', 0)
        
        print(f"{strategy:<20} {ee:<15.2e} {qos:<15.2f} {rate:<15.2f}")
    
    print("="*80 + "\n")


def plot_channel_gains(network, save_path: str = None):
    """
    Plot channel gain matrix
    
    Args:
        network: CellFreeNetworkSionna instance
        save_path: Path to save the plot
    """
    # Generate channel
    channel_matrix = network.generate_channel_matrix(batch_size=1)
    
    # Get channel gains (magnitude)
    channel_gain = np.abs(channel_matrix[0].numpy())  # (num_users, num_tx)
    
    # Average over antennas to get per-AP gains
    channel_gain_per_ap = np.zeros((network.num_users, network.num_aps))
    for ap_idx in range(network.num_aps):
        ant_start = ap_idx * network.num_antennas_per_ap
        ant_end = (ap_idx + 1) * network.num_antennas_per_ap
        channel_gain_per_ap[:, ap_idx] = np.mean(
            channel_gain[:, ant_start:ant_end],
            axis=1
        )
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        channel_gain_per_ap.T,
        annot=True,
        fmt='.2e',
        cmap='YlOrRd',
        xticklabels=[f'U{i}' for i in range(network.num_users)],
        yticklabels=[f'AP{i}' for i in range(network.num_aps)],
        cbar_kws={'label': 'Channel Gain (linear)'}
    )
    plt.xlabel('Users', fontsize=12)
    plt.ylabel('Access Points', fontsize=12)
    plt.title('Channel Gain Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_sinr_distribution(sinr_values: List[float], save_path: str = None):
    """
    Plot SINR distribution
    
    Args:
        sinr_values: List of SINR values in linear scale
        save_path: Path to save the plot
    """
    sinr_db = 10 * np.log10(sinr_values)
    
    plt.figure(figsize=(10, 6))
    plt.hist(sinr_db, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('SINR (dB)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('SINR Distribution', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_experiment_dir(base_dir: str = './experiments') -> str:
    """
    Create a new experiment directory with timestamp
    
    Args:
        base_dir: Base directory for experiments
        
    Returns:
        Path to new experiment directory
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(base_dir, f'exp_{timestamp}')
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'results'), exist_ok=True)
    
    return exp_dir
