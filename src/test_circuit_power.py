"""
Circuit Power Sensitivity Analysis
Tests how different circuit power values affect strategy performance
Includes RL agent evaluation alongside baseline strategies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from network.cellfree_network import CellFreeNetworkSionna
from agents.baselines import BaselineStrategies
from agents.dqn_agent import DQNAgent
from environment.cellfree_env import CellFreeEnv
import numpy as np
import matplotlib.pyplot as plt
import argparse

def test_circuit_power_sensitivity(rl_model_path=None, num_users=10, num_aps=25):
    """Test strategies with different circuit power values

    Args:
        rl_model_path: Optional path to trained RL model to include in comparison
        num_users: Number of users (default: 10)
        num_aps: Number of APs (default: 25)
    """

    # Configuration
    circuit_powers = [0.1, 0.2, 0.5]  # 100mW, 200mW, 500mW
    circuit_power_labels = ['100mW', '200mW', '500mW']

    strategies = {
        'Nearest AP': BaselineStrategies.nearest_ap_max_power,
        'Equal Power': BaselineStrategies.equal_power_all_serve,
        'Load Balancing': BaselineStrategies.load_balancing
    }

    print("="*80)
    print("CIRCUIT POWER SENSITIVITY ANALYSIS")
    print("="*80)
    print(f"Configuration: {num_aps} APs, {num_users} Users")
    print(f"Testing circuit power values: {circuit_power_labels}")
    if rl_model_path:
        print(f"RL Model: {rl_model_path}")
    print("="*80)

    # Store results (include RL agent if model provided)
    all_strategy_names = list(strategies.keys())
    if rl_model_path:
        all_strategy_names.append('RL Agent')

    results = {strategy: {
        'rate': [],
        'energy_eff': [],
        'active_aps': [],
        'qos_sat': []
    } for strategy in all_strategy_names}

    # Test each circuit power value
    for cp_idx, circuit_power in enumerate(circuit_powers):
        print(f"\n{'='*80}")
        print(f"Testing with Circuit Power = {circuit_power_labels[cp_idx]}")
        print(f"{'='*80}")

        # Create network with specific circuit power
        network = CellFreeNetworkSionna(
            num_aps=num_aps,
            num_users=num_users,
            num_antennas_per_ap=1,
            area_size=500.0,
            circuit_power_per_ap=circuit_power,
            seed=42
        )

        # Generate channel
        channel_matrix = network.generate_channel_matrix(batch_size=1)

        # Test baseline strategies
        for strategy_name, strategy_func in strategies.items():
            print(f"\n  {strategy_name}:")

            # Get allocation
            power_allocation, ap_association = strategy_func(network, channel_matrix)

            # Calculate metrics
            sinr, rates = network.calculate_sinr_and_rate(
                channel_matrix,
                power_allocation,
                ap_association
            )

            ee = network.calculate_energy_efficiency(rates, power_allocation, ap_association)

            qos_requirements = np.ones(network.num_users) * 5e6  # 5 Mbps
            qos_sat = network.calculate_qos_satisfaction(rates, qos_requirements)

            active_aps = np.sum(np.sum(ap_association, axis=1) > 0)
            avg_rate = np.mean(rates.numpy()) / 1e6

            # Store results
            results[strategy_name]['rate'].append(avg_rate)
            results[strategy_name]['energy_eff'].append(ee.numpy()[0])
            results[strategy_name]['active_aps'].append(active_aps)
            results[strategy_name]['qos_sat'].append(qos_sat.numpy()[0])

            # Print results
            print(f"    - Avg Rate: {avg_rate:.2f} Mbps")
            print(f"    - Energy Eff: {ee.numpy()[0]:.2e} bits/J")
            print(f"    - Active APs: {active_aps}/{num_aps}")
            print(f"    - QoS Sat: {qos_sat.numpy()[0]:.1f}%")

        # Test RL agent if model provided
        if rl_model_path:
            print(f"\n  RL Agent:")
            rl_results = evaluate_rl_agent_circuit_power(
                rl_model_path,
                num_aps,
                num_users,
                circuit_power,
                n_episodes=10
            )

            # Store results
            results['RL Agent']['rate'].append(rl_results['mean_rate_mbps'])
            results['RL Agent']['energy_eff'].append(rl_results['mean_energy_efficiency'])
            results['RL Agent']['active_aps'].append(rl_results['mean_active_aps'])
            results['RL Agent']['qos_sat'].append(rl_results['mean_qos_satisfaction'])

            # Print results
            print(f"    - Avg Rate: {rl_results['mean_rate_mbps']:.2f} Mbps")
            print(f"    - Energy Eff: {rl_results['mean_energy_efficiency']:.2e} bits/J")
            print(f"    - Active APs: {rl_results['mean_active_aps']:.1f}/{num_aps}")
            print(f"    - QoS Sat: {rl_results['mean_qos_satisfaction']:.1f}%")

    # Plot results
    plot_sensitivity_analysis(results, circuit_power_labels, num_aps, num_users, include_rl=rl_model_path is not None)

    # Print summary
    print_summary(results, circuit_power_labels)

def evaluate_rl_agent_circuit_power(model_path, num_aps, num_users, circuit_power, n_episodes=10):
    """Evaluate RL agent with specific circuit power setting

    Args:
        model_path: Path to trained RL model
        num_aps: Number of APs
        num_users: Number of users
        circuit_power: Circuit power per AP (Watts)
        n_episodes: Number of evaluation episodes

    Returns:
        Dictionary with mean metrics
    """
    # Create environment with specific circuit power
    env = CellFreeEnv(
        num_aps=num_aps,
        num_users=num_users,
        qos_min_rate_mbps=5.0,
        qos_weight=10.0,
        episode_length=100,
        action_type='discrete'
    )

    # Override circuit power in network
    env.network.circuit_power_per_ap = circuit_power

    # Load RL agent
    agent = DQNAgent(env=env, verbose=0)
    agent.load(model_path)

    # Evaluate with timeout protection
    all_rates = []
    all_ee = []
    all_active_aps = []
    all_qos_sat = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_metrics = {
            'energy_efficiency': [],
            'rate': [],
            'qos_satisfaction': [],
            'active_aps': []
        }

        done = False
        steps = 0
        max_steps = 100  # Safety timeout

        while not done and steps < max_steps:
            try:
                action, _ = agent.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1

                # Collect metrics
                if 'energy_efficiency' in info:
                    episode_metrics['energy_efficiency'].append(info['energy_efficiency'])
                if 'avg_rate_mbps' in info:
                    episode_metrics['rate'].append(info['avg_rate_mbps'])
                if 'qos_satisfaction' in info:
                    episode_metrics['qos_satisfaction'].append(info['qos_satisfaction'])
                if 'active_aps' in info:
                    episode_metrics['active_aps'].append(info['active_aps'])

            except Exception as e:
                print(f"    ⚠️ Error in episode {episode+1}: {e}")
                break

        # Average metrics over episode
        if episode_metrics['energy_efficiency']:
            all_ee.append(np.mean(episode_metrics['energy_efficiency']))
        if episode_metrics['rate']:
            all_rates.append(np.mean(episode_metrics['rate']))
        if episode_metrics['qos_satisfaction']:
            all_qos_sat.append(np.mean(episode_metrics['qos_satisfaction']))
        if episode_metrics['active_aps']:
            all_active_aps.append(np.mean(episode_metrics['active_aps']))

    return {
        'mean_rate_mbps': np.mean(all_rates) if all_rates else 0.0,
        'mean_energy_efficiency': np.mean(all_ee) if all_ee else 0.0,
        'mean_active_aps': np.mean(all_active_aps) if all_active_aps else 0.0,
        'mean_qos_satisfaction': np.mean(all_qos_sat) if all_qos_sat else 0.0
    }

def plot_sensitivity_analysis(results, circuit_power_labels, num_aps, num_users, include_rl=False):
    """Plot circuit power sensitivity analysis

    Args:
        results: Dictionary of results for each strategy
        circuit_power_labels: Labels for circuit power values
        num_aps: Number of APs
        num_users: Number of users
        include_rl: Whether RL agent is included
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title = f'Circuit Power Sensitivity Analysis\n{num_aps} APs, {num_users} Users'
    if include_rl:
        title += ' (with RL Agent)'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    metrics = [
        ('rate', 'Average Rate per User (Mbps)', axes[0, 0]),
        ('energy_eff', 'Energy Efficiency (bits/Joule)', axes[0, 1]),
        ('active_aps', 'Number of Active APs', axes[1, 0]),
        ('qos_sat', 'QoS Satisfaction (%)', axes[1, 1])
    ]

    colors = {
        'Nearest AP': '#FF6B6B',
        'Equal Power': '#4ECDC4',
        'Load Balancing': '#95E1D3',
        'RL Agent': '#FFD93D'  # Yellow for RL agent
    }
    markers = {
        'Nearest AP': 'o',
        'Equal Power': 's',
        'Load Balancing': '^',
        'RL Agent': 'D'  # Diamond for RL agent
    }

    for metric_key, ylabel, ax in metrics:
        for strategy_name, strategy_results in results.items():
            values = strategy_results[metric_key]
            ax.plot(circuit_power_labels, values,
                   marker=markers[strategy_name],
                   color=colors[strategy_name],
                   linewidth=2, markersize=8,
                   label=strategy_name)

        ax.set_xlabel('Circuit Power per AP', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Use log scale for energy efficiency
        if metric_key == 'energy_eff':
            ax.set_yscale('log')

    plt.tight_layout()

    # Save figure
    save_path = 'results/circuit_power_sensitivity.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {save_path}")

    plt.show()

def print_summary(results, circuit_power_labels):
    """Print summary of findings"""

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    # Find best strategies for each metric at each circuit power
    for cp_idx, cp_label in enumerate(circuit_power_labels):
        print(f"\nCircuit Power = {cp_label}:")

        # Best rate
        best_rate_strategy = max(results.keys(),
                                key=lambda s: results[s]['rate'][cp_idx])
        best_rate = results[best_rate_strategy]['rate'][cp_idx]
        print(f"  • Best Rate: {best_rate_strategy} - {best_rate:.2f} Mbps")

        # Best energy efficiency
        best_ee_strategy = max(results.keys(),
                              key=lambda s: results[s]['energy_eff'][cp_idx])
        best_ee = results[best_ee_strategy]['energy_eff'][cp_idx]
        print(f"  • Best Energy Eff: {best_ee_strategy} - {best_ee:.2e} bits/J")

        # Fewest active APs
        min_aps_strategy = min(results.keys(),
                              key=lambda s: results[s]['active_aps'][cp_idx])
        min_aps = results[min_aps_strategy]['active_aps'][cp_idx]
        print(f"  • Fewest Active APs: {min_aps_strategy} - {min_aps} APs")

    # Analyze trends
    print("\n" + "="*80)
    print("TRENDS")
    print("="*80)

    for strategy_name in results.keys():
        ee_100mw = results[strategy_name]['energy_eff'][0]
        ee_500mw = results[strategy_name]['energy_eff'][2]
        ee_change = ((ee_500mw - ee_100mw) / ee_100mw) * 100

        aps_100mw = results[strategy_name]['active_aps'][0]
        aps_500mw = results[strategy_name]['active_aps'][2]

        print(f"\n{strategy_name}:")
        print(f"  • Energy Eff change (100mW → 500mW): {ee_change:+.1f}%")
        print(f"  • Active APs (100mW): {aps_100mw}, (500mW): {aps_500mw}")
        print(f"  • Circuit power impact: {'HIGH' if abs(ee_change) > 30 else 'MODERATE' if abs(ee_change) > 10 else 'LOW'}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Circuit Power Sensitivity Analysis')
    parser.add_argument('--rl-model', type=str, default=None,
                       help='Path to trained RL model (optional)')
    parser.add_argument('--num-users', type=int, default=10,
                       help='Number of users (default: 10)')
    parser.add_argument('--num-aps', type=int, default=25,
                       help='Number of APs (default: 25)')
    args = parser.parse_args()

    test_circuit_power_sensitivity(
        rl_model_path=args.rl_model,
        num_users=args.num_users,
        num_aps=args.num_aps
    )
