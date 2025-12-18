"""
Run Baseline Strategies Evaluation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import argparse
import yaml
from network.cellfree_network import CellFreeNetworkSionna
from agents.baselines import evaluate_baseline
from utils.plotting import plot_comparison, save_results, print_results_table


def main():
    parser = argparse.ArgumentParser(description='Evaluate Baseline Strategies')
    parser.add_argument('--config', type=str, default='../configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--save_dir', type=str, default='../results',
                       help='Directory to save results')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create network
    print("Creating Cell-Free Network...")
    network = CellFreeNetworkSionna(
        num_aps=config['network']['num_aps'],
        num_users=config['network']['num_users'],
        num_antennas_per_ap=config['network']['num_antennas_per_ap'],
        area_size=config['network']['area_size'],
        seed=config['seed']['env_seed']
    )
    
    print(f"Network Configuration:")
    print(f"  - Number of APs: {network.num_aps}")
    print(f"  - Number of Users: {network.num_users}")
    print(f"  - Coverage Area: {network.area_size}x{network.area_size} m²")
    
    # Visualize network
    network.visualize_network()
    
    # Evaluate baseline strategies
    strategies = config['evaluation']['baseline_strategies']
    results = {}
    
    print(f"\nEvaluating {len(strategies)} baseline strategies...")
    print(f"Episodes per strategy: {args.episodes}\n")
    
    for strategy in strategies:
        print(f"Evaluating: {strategy}...")
        result = evaluate_baseline(network, strategy, num_episodes=args.episodes)
        results[strategy] = result
        print(f"  - Energy Efficiency: {result['mean_energy_efficiency']:.2e} bits/Joule")
        print(f"  - QoS Satisfaction: {result['mean_qos_satisfaction']:.2f}%")
        print(f"  - Average Rate: {result['mean_rate_mbps']:.2f} Mbps")
        print(f"  - Average SINR: {result['mean_sinr_db']:.2f} dB\n")
    
    # Print results table
    print_results_table(results)
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    results_file = os.path.join(args.save_dir, 'baseline_results.json')
    save_results(results, results_file)
    
    # Plot comparison
    plot_path = os.path.join(args.save_dir, 'baseline_comparison.png')
    plot_comparison(results, save_path=plot_path)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Comparison plot saved to: {plot_path}")


if __name__ == '__main__':
    main()
