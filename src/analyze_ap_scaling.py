"""
Advanced AP Scaling Analysis: Performance vs Number of Access Points

This script provides flexible analysis of network performance with:
- Multiple strategies (nearest_ap, equal_power, load_balancing, all)
- Custom AP and user configurations
- Single or multi-configuration analysis
- Command-line interface

Usage:
    # Single configuration
    python src/analyze_ap_scaling.py nearest_ap 16 8

    # Multi-configuration analysis
    python src/analyze_ap_scaling.py equal_power --multi

    # Compare all strategies
    python src/analyze_ap_scaling.py all 25 10

    # Custom scaling analysis
    python src/analyze_ap_scaling.py load_balancing --aps 16,25,36,49,64 --users 8
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from network.cellfree_network import CellFreeNetworkSionna
from agents.baselines import BaselineStrategies
from agents.dqn_agent import DQNAgent
from environment.cellfree_env import CellFreeEnv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional


# Strategy mapping
STRATEGY_MAP = {
    'nearest_ap': ('Nearest AP + Max Power', BaselineStrategies.nearest_ap_max_power),
    'equal_power': ('Equal Power + All Serve', BaselineStrategies.equal_power_all_serve),
    'load_balancing': ('Load Balancing', BaselineStrategies.load_balancing),
}


def evaluate_rl_agent_scaling(
    rl_model_path: str,
    num_aps: int,
    num_users: int,
    n_episodes: int = 10,
    circuit_power: float = 0.2
) -> Dict:
    """
    Evaluate RL agent for AP scaling analysis

    Args:
        rl_model_path: Path to trained RL model
        num_aps: Number of APs
        num_users: Number of users
        n_episodes: Number of evaluation episodes
        circuit_power: Circuit power per AP (Watts)

    Returns:
        Dictionary with performance metrics
    """
    print(f"\n{'='*70}")
    print(f"Testing: RL Agent (DQN)")
    print(f"Configuration: {num_aps} APs, {num_users} Users")
    print(f"{'='*70}")

    # Create environment
    env = CellFreeEnv(
        num_aps=num_aps,
        num_users=num_users,
        qos_min_rate_mbps=5.0,
        qos_weight=10.0,
        episode_length=100,
        action_type='discrete'
    )

    # Override circuit power
    env.network.circuit_power_per_ap = circuit_power

    # Load RL agent
    agent = DQNAgent(env=env, verbose=0)
    agent.load(rl_model_path)
    print(f"✓ RL agent loaded")

    # Evaluate metrics
    all_rates = []
    all_ee = []
    all_qos = []
    all_active_aps = []
    all_sinr = []
    all_total_power = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_metrics = {
            'rate': [],
            'ee': [],
            'qos': [],
            'active_aps': [],
            'sinr': [],
            'total_power': []
        }

        done = False
        steps = 0
        max_steps = 100

        while not done and steps < max_steps:
            try:
                action, _ = agent.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1

                # Collect metrics
                if 'avg_rate_mbps' in info:
                    episode_metrics['rate'].append(info['avg_rate_mbps'])
                if 'energy_efficiency' in info:
                    episode_metrics['ee'].append(info['energy_efficiency'])
                if 'qos_satisfaction' in info:
                    episode_metrics['qos'].append(info['qos_satisfaction'])
                if 'active_aps' in info:
                    episode_metrics['active_aps'].append(info['active_aps'])
                if 'avg_sinr_db' in info:
                    episode_metrics['sinr'].append(info['avg_sinr_db'])
                if 'total_power_w' in info:
                    episode_metrics['total_power'].append(info['total_power_w'])

            except Exception as e:
                print(f"⚠️ Error in episode {episode+1}: {e}")
                break

        # Average metrics over episode
        if episode_metrics['rate']:
            all_rates.append(np.mean(episode_metrics['rate']))
        if episode_metrics['ee']:
            all_ee.append(np.mean(episode_metrics['ee']))
        if episode_metrics['qos']:
            all_qos.append(np.mean(episode_metrics['qos']))
        if episode_metrics['active_aps']:
            all_active_aps.append(np.mean(episode_metrics['active_aps']))
        if episode_metrics['sinr']:
            all_sinr.append(np.mean(episode_metrics['sinr']))
        if episode_metrics['total_power']:
            all_total_power.append(np.mean(episode_metrics['total_power']))

    # Aggregate results
    results = {
        'strategy': 'RL Agent',
        'num_aps': num_aps,
        'num_users': num_users,
        'avg_sinr_db': float(np.mean(all_sinr)) if all_sinr else 0.0,
        'avg_rate_mbps': float(np.mean(all_rates)) if all_rates else 0.0,
        'total_rate_mbps': float(np.mean(all_rates) * num_users) if all_rates else 0.0,
        'energy_efficiency': float(np.mean(all_ee)) if all_ee else 0.0,
        'qos_satisfaction': float(np.mean(all_qos)) if all_qos else 0.0,
        'min_rate_mbps': float(np.min(all_rates)) if all_rates else 0.0,
        'max_rate_mbps': float(np.max(all_rates)) if all_rates else 0.0,
        'total_power_w': float(np.mean(all_total_power)) if all_total_power else 0.0,
        'active_aps': int(np.mean(all_active_aps)) if all_active_aps else 0,
        'avg_aps_per_user': float(np.mean(all_active_aps) / num_users) if all_active_aps else 0.0
    }

    # Print summary
    print(f"\nResults (averaged over {n_episodes} episodes):")
    print(f"  • Average SINR: {results['avg_sinr_db']:.2f} dB")
    print(f"  • Average Rate: {results['avg_rate_mbps']:.2f} Mbps")
    print(f"  • Total Rate: {results['total_rate_mbps']:.2f} Mbps")
    print(f"  • Energy Efficiency: {results['energy_efficiency']:.2e} bits/Joule")
    print(f"  • QoS Satisfaction: {results['qos_satisfaction']:.2f}%")
    print(f"  • Active APs: {results['active_aps']}/{num_aps}")
    print(f"  • Avg APs/User: {results['avg_aps_per_user']:.2f}")

    return results


def run_simulation(
    num_aps: int,
    num_users: int,
    strategy_name: str,
    strategy_func,
    seed: int = 42
) -> Dict:
    """
    Run simulation for a given configuration

    Args:
        num_aps: Number of Access Points
        num_users: Number of users
        strategy_name: Name of the strategy
        strategy_func: Strategy function to apply
        seed: Random seed for reproducibility

    Returns:
        Dictionary with performance metrics
    """
    print(f"\n{'='*70}")
    print(f"Testing: {strategy_name}")
    print(f"Configuration: {num_aps} APs, {num_users} Users")
    print(f"{'='*70}")

    # Create network
    network = CellFreeNetworkSionna(
        num_aps=num_aps,
        num_users=num_users,
        num_antennas_per_ap=1,
        area_size=500.0,
        seed=seed
    )
    print(f"✓ Network created")

    # Generate channel
    channel_matrix = network.generate_channel_matrix(batch_size=1)
    print(f"✓ Channel generated: {channel_matrix.shape}")

    # Apply strategy
    power_allocation, ap_association = strategy_func(network, channel_matrix)
    print(f"✓ Strategy applied")

    # Calculate performance metrics
    sinr, rates = network.calculate_sinr_and_rate(
        channel_matrix,
        power_allocation,
        ap_association
    )

    ee = network.calculate_energy_efficiency(rates, power_allocation, ap_association)
    qos_requirements = np.ones(network.num_users) * 5e6  # 5 Mbps
    qos_sat = network.calculate_qos_satisfaction(rates, qos_requirements)

    # Extract results
    results = {
        'strategy': strategy_name,
        'num_aps': num_aps,
        'num_users': num_users,
        'avg_sinr_db': float(10 * np.log10(np.mean(sinr.numpy()))),
        'avg_rate_mbps': float(np.mean(rates.numpy()) / 1e6),
        'total_rate_mbps': float(np.sum(rates.numpy()) / 1e6),
        'energy_efficiency': float(ee.numpy()[0]),
        'qos_satisfaction': float(qos_sat.numpy()[0]),
        'min_rate_mbps': float(np.min(rates.numpy()) / 1e6),
        'max_rate_mbps': float(np.max(rates.numpy()) / 1e6),
        'total_power_w': float(np.sum(power_allocation)),
        'active_aps': int(np.sum(np.sum(ap_association, axis=1) > 0)),
        'avg_aps_per_user': float(np.mean(np.sum(ap_association, axis=0)))
    }

    # Print summary
    print(f"\nResults:")
    print(f"  • Average SINR: {results['avg_sinr_db']:.2f} dB")
    print(f"  • Average Rate: {results['avg_rate_mbps']:.2f} Mbps")
    print(f"  • Total Rate: {results['total_rate_mbps']:.2f} Mbps")
    print(f"  • Energy Efficiency: {results['energy_efficiency']:.2e} bits/Joule")
    print(f"  • QoS Satisfaction: {results['qos_satisfaction']:.2f}%")
    print(f"  • Active APs: {results['active_aps']}/{num_aps}")
    print(f"  • Avg APs/User: {results['avg_aps_per_user']:.2f}")

    return results


def plot_single_config_comparison(results_list: List[Dict], save_path: str = None):
    """
    Plot comparison of multiple strategies for a single configuration

    Args:
        results_list: List of result dictionaries
        save_path: Optional path to save the figure
    """
    strategies = [r['strategy'] for r in results_list]
    metrics = {
        'avg_rate_mbps': 'Average Rate (Mbps)',
        'energy_efficiency': 'Energy Efficiency (bits/J)',
        'avg_sinr_db': 'Average SINR (dB)',
        'qos_satisfaction': 'QoS Satisfaction (%)',
        'active_aps': 'Active APs',
        'avg_aps_per_user': 'Avg APs per User'
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f'Strategy Comparison: {results_list[0]["num_aps"]} APs, {results_list[0]["num_users"]} Users',
        fontsize=16, fontweight='bold', y=0.995
    )

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']

    for idx, (key, label) in enumerate(metrics.items()):
        row, col = idx // 3, idx % 3
        values = [r[key] for r in results_list]

        bars = axes[row, col].bar(strategies, values, color=colors[:len(strategies)],
                                   alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[row, col].set_ylabel(label, fontsize=12, fontweight='bold')
        axes[row, col].set_title(label, fontsize=13, fontweight='bold')
        axes[row, col].grid(True, alpha=0.3, linestyle='--', axis='y')

        # Rotate x labels (fix warning by setting xticks first)
        axes[row, col].set_xticks(range(len(strategies)))
        axes[row, col].set_xticklabels(strategies, rotation=45, ha='right')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if key == 'energy_efficiency':
                axes[row, col].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{val:.1e}', ha='center', va='bottom', fontsize=9)
            else:
                axes[row, col].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Figure saved to: {save_path}")

    plt.show()


def plot_multi_config_results(results_list: List[Dict], save_path: str = None):
    """
    Plot performance metrics vs number of APs for multiple strategies

    Args:
        results_list: List of result dictionaries
        save_path: Optional path to save the figure
    """
    # Group results by strategy
    strategy_groups = {}
    for r in results_list:
        strategy = r['strategy']
        if strategy not in strategy_groups:
            strategy_groups[strategy] = []
        strategy_groups[strategy].append(r)

    # Sort by num_aps
    for strategy in strategy_groups:
        strategy_groups[strategy].sort(key=lambda x: x['num_aps'])

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Network Performance vs Number of Access Points',
                 fontsize=16, fontweight='bold', y=0.995)

    colors = {'Nearest AP + Max Power': '#2E86AB',
              'Equal Power + All Serve': '#A23B72',
              'Load Balancing': '#F18F01',
              'RL Agent': '#FFD93D'}
    markers = {'Nearest AP + Max Power': 'o',
               'Equal Power + All Serve': 's',
               'Load Balancing': '^',
               'RL Agent': 'D'}

    metrics = [
        ('avg_rate_mbps', 'Average Rate per User (Mbps)', (0, 0)),
        ('total_rate_mbps', 'Total Network Throughput (Mbps)', (0, 1)),
        ('energy_efficiency', 'Energy Efficiency (bits/Joule)', (0, 2)),
        ('avg_sinr_db', 'Average SINR (dB)', (1, 0)),
        ('qos_satisfaction', 'QoS Satisfaction (%)', (1, 1)),
        ('active_aps', 'Active APs', (1, 2))
    ]

    for metric_key, metric_label, (row, col) in metrics:
        ax = axes[row, col]

        for strategy, results in strategy_groups.items():
            num_aps_list = [r['num_aps'] for r in results]
            values = [r[metric_key] for r in results]

            ax.plot(num_aps_list, values,
                   marker=markers.get(strategy, 'o'),
                   linewidth=2.5, markersize=8,
                   label=strategy,
                   color=colors.get(strategy, '#666666'))

        ax.set_xlabel('Number of APs', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
        ax.set_title(metric_label, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=9)

        if metric_key == 'energy_efficiency':
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Figure saved to: {save_path}")

    plt.show()


def save_results_table(results_list: List[Dict], save_path: str = None):
    """
    Save results as a formatted table

    Args:
        results_list: List of result dictionaries
        save_path: Optional path to save the table
    """
    print(f"\n{'='*110}")
    print("SUMMARY TABLE: Analysis Results")
    print(f"{'='*110}")

    # Header
    header = (f"{'Strategy':<25} | {'APs':>4} | {'Users':>5} | {'Avg Rate':>10} | "
              f"{'SINR':>8} | {'Energy Eff':>12} | {'QoS':>7}")
    print(header)
    print(f"{'-'*110}")

    # Data rows
    for r in results_list:
        row = (f"{r['strategy']:<25} | "
               f"{r['num_aps']:>4} | "
               f"{r['num_users']:>5} | "
               f"{r['avg_rate_mbps']:>9.2f}M | "
               f"{r['avg_sinr_db']:>7.2f}dB | "
               f"{r['energy_efficiency']:>11.2e} | "
               f"{r['qos_satisfaction']:>6.1f}%")
        print(row)

    print(f"{'='*110}\n")

    # Save to file if requested
    if save_path:
        with open(save_path, 'w') as f:
            f.write("AP Scaling Analysis Results\n")
            f.write("="*110 + "\n")
            f.write(header + "\n")
            f.write("-"*110 + "\n")
            for r in results_list:
                row = (f"{r['strategy']:<25} | "
                       f"{r['num_aps']:>4} | "
                       f"{r['num_users']:>5} | "
                       f"{r['avg_rate_mbps']:>9.2f}M | "
                       f"{r['avg_sinr_db']:>7.2f}dB | "
                       f"{r['energy_efficiency']:>11.2e} | "
                       f"{r['qos_satisfaction']:>6.1f}%\n")
                f.write(row)
            f.write("="*110 + "\n")
        print(f"✓ Results table saved to: {save_path}")


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Advanced AP Scaling Analysis for Cell-Free Networks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single configuration with one strategy
  python src/analyze_ap_scaling.py nearest_ap 16 8

  # Compare all strategies for one configuration
  python src/analyze_ap_scaling.py all 25 10

  # Multi-configuration scaling analysis
  python src/analyze_ap_scaling.py equal_power --multi

  # Custom AP configurations
  python src/analyze_ap_scaling.py load_balancing --aps 16,25,36 --users 8

  # All strategies with scaling
  python src/analyze_ap_scaling.py all --multi --aps 16,25,36,49,64 --users 8
        """
    )

    parser.add_argument(
        'strategy',
        choices=['nearest_ap', 'equal_power', 'load_balancing', 'all'],
        help='Strategy to analyze (or "all" for comparison)'
    )

    parser.add_argument(
        'num_aps',
        type=int,
        nargs='?',
        default=None,
        help='Number of Access Points (ignored if --multi or --aps is used)'
    )

    parser.add_argument(
        'num_users',
        type=int,
        nargs='?',
        default=8,
        help='Number of users (default: 8)'
    )

    parser.add_argument(
        '--multi',
        action='store_true',
        help='Run multi-configuration scaling analysis'
    )

    parser.add_argument(
        '--aps',
        type=str,
        default='16,25,36,49,64',
        help='Comma-separated list of AP numbers for multi-config (default: 16,25,36,49,64)'
    )

    parser.add_argument(
        '--users',
        type=int,
        default=8,
        help='Number of users for analysis (default: 8)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--rl-model',
        type=str,
        default=None,
        help='Path to trained RL model to include in comparison'
    )

    parser.add_argument(
        '--circuit-power',
        type=float,
        default=0.2,
        help='Circuit power per AP in Watts for RL evaluation (default: 0.2)'
    )

    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()

    print("="*80)
    print("ADVANCED AP SCALING ANALYSIS")
    print("="*80)

    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Determine strategies to test
    if args.strategy == 'all':
        strategies_to_test = list(STRATEGY_MAP.keys())
    else:
        strategies_to_test = [args.strategy]

    print(f"\nStrategies: {', '.join(strategies_to_test)}")

    # Determine configurations
    if args.multi:
        ap_configs = [int(x.strip()) for x in args.aps.split(',')]
        num_users = args.users
        print(f"AP Configurations: {ap_configs}")
        print(f"Users: {num_users}")
        mode = 'multi'
    else:
        if args.num_aps is None:
            print("\n❌ Error: num_aps is required for single-config mode")
            print("Usage: python src/analyze_ap_scaling.py strategy num_aps num_users")
            print("Or use --multi flag for multi-configuration analysis")
            return
        ap_configs = [args.num_aps]
        num_users = args.num_users
        print(f"Configuration: {args.num_aps} APs, {num_users} Users")
        mode = 'single'

    # Run simulations
    results_list = []

    for strategy_key in strategies_to_test:
        strategy_name, strategy_func = STRATEGY_MAP[strategy_key]

        for num_aps in ap_configs:
            try:
                results = run_simulation(
                    num_aps=num_aps,
                    num_users=num_users,
                    strategy_name=strategy_name,
                    strategy_func=strategy_func,
                    seed=args.seed
                )
                results_list.append(results)
            except Exception as e:
                print(f"\n⚠️  Error with {strategy_name}, {num_aps} APs: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Test RL agent if model provided
    if args.rl_model:
        print(f"\n{'='*80}")
        print("Testing RL Agent")
        print(f"{'='*80}")

        for num_aps in ap_configs:
            try:
                results = evaluate_rl_agent_scaling(
                    rl_model_path=args.rl_model,
                    num_aps=num_aps,
                    num_users=num_users,
                    n_episodes=10,
                    circuit_power=args.circuit_power
                )
                results_list.append(results)
            except Exception as e:
                print(f"\n⚠️  Error with RL Agent, {num_aps} APs: {e}")
                import traceback
                traceback.print_exc()
                continue

    if not results_list:
        print("\n❌ No successful simulations. Exiting.")
        return

    # Save results table
    table_filename = f'results/analysis_{args.strategy}_{mode}.txt'
    save_results_table(results_list, save_path=table_filename)

    # Generate plots
    print(f"\nGenerating visualizations...")

    if mode == 'single' and len(strategies_to_test) > 1:
        # Multiple strategies, single configuration
        plot_filename = f'results/comparison_{args.num_aps}aps_{num_users}users.png'
        plot_single_config_comparison(results_list, save_path=plot_filename)

    elif mode == 'multi':
        # Multi-configuration scaling
        plot_filename = f'results/scaling_{args.strategy}_{num_users}users.png'
        plot_multi_config_results(results_list, save_path=plot_filename)

    else:
        # Single strategy, single config - just print results
        print("\n✓ Single configuration analysis completed")
        print("   Use --multi for scaling analysis or 'all' for strategy comparison")

    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED")
    print("="*80)

    if len(results_list) > 1:
        print("\nKey Findings:")
        best_rate = max(results_list, key=lambda x: x['avg_rate_mbps'])
        best_ee = max(results_list, key=lambda x: x['energy_efficiency'])
        best_qos = max(results_list, key=lambda x: x['qos_satisfaction'])

        print(f"  • Best Average Rate: {best_rate['strategy']} "
              f"({best_rate['num_aps']} APs) - {best_rate['avg_rate_mbps']:.2f} Mbps")
        print(f"  • Best Energy Efficiency: {best_ee['strategy']} "
              f"({best_ee['num_aps']} APs) - {best_ee['energy_efficiency']:.2e} bits/J")
        print(f"  • Best QoS Satisfaction: {best_qos['strategy']} "
              f"({best_qos['num_aps']} APs) - {best_qos['qos_satisfaction']:.1f}%")

    print(f"\nOutput files in results/ directory:")
    print(f"  - {table_filename}")
    if 'plot_filename' in locals():
        print(f"  - {plot_filename}")


if __name__ == '__main__':
    main()
