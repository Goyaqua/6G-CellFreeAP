"""
Circuit Power Sensitivity Analysis Tool
Tests how different circuit power values affect network performance
Includes comprehensive analysis for RL agents and baseline strategies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from network.cellfree_network import CellFreeNetworkSionna
from agents.baselines import BaselineStrategies
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from environment.cellfree_env import CellFreeEnv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json


# ============================================================================
# CORE EVALUATION FUNCTIONS
# ============================================================================

def evaluate_rl_agent_circuit_power(model_path, agent_type, num_aps, num_users, circuit_power, n_episodes=10):
    """Evaluate RL agent with specific circuit power setting

    Args:
        model_path: Path to trained RL model
        agent_type: 'dqn' or 'ppo'
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
    if agent_type == 'ppo':
        agent = PPOAgent(env=env, verbose=0)
    else:
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
        'std_rate_mbps': np.std(all_rates) if all_rates else 0.0,
        'mean_energy_efficiency': np.mean(all_ee) if all_ee else 0.0,
        'std_energy_efficiency': np.std(all_ee) if all_ee else 0.0,
        'mean_active_aps': np.mean(all_active_aps) if all_active_aps else 0.0,
        'std_active_aps': np.std(all_active_aps) if all_active_aps else 0.0,
        'mean_qos_satisfaction': np.mean(all_qos_sat) if all_qos_sat else 0.0,
        'std_qos_satisfaction': np.std(all_qos_sat) if all_qos_sat else 0.0
    }


def evaluate_baseline_circuit_power(network, strategy_func, strategy_name, n_episodes=10):
    """Evaluate baseline strategy with specific circuit power

    Args:
        network: CellFreeNetworkSionna instance
        strategy_func: Baseline strategy function
        strategy_name: Name of the strategy
        n_episodes: Number of evaluation episodes

    Returns:
        Dictionary with mean metrics
    """
    all_rates = []
    all_ee = []
    all_active_aps = []
    all_qos_sat = []

    for _ in range(n_episodes):
        # Generate channel
        channel_matrix = network.generate_channel_matrix(batch_size=1)

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
        all_rates.append(avg_rate)
        all_ee.append(ee.numpy()[0])
        all_active_aps.append(active_aps)
        all_qos_sat.append(qos_sat.numpy()[0])

    return {
        'mean_rate_mbps': np.mean(all_rates),
        'std_rate_mbps': np.std(all_rates),
        'mean_energy_efficiency': np.mean(all_ee),
        'std_energy_efficiency': np.std(all_ee),
        'mean_active_aps': np.mean(all_active_aps),
        'std_active_aps': np.std(all_active_aps),
        'mean_qos_satisfaction': np.mean(all_qos_sat),
        'std_qos_satisfaction': np.std(all_qos_sat)
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

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

    plt.close()


def plot_detailed_comparison(results, circuit_power_labels, save_dir='results/circuit_power'):
    """Create detailed comparison plots with error bars

    Args:
        results: Dictionary with results including std deviations
        circuit_power_labels: Labels for circuit power values
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Detailed Circuit Power Sensitivity Analysis', fontsize=16, fontweight='bold')

    colors = {
        'Nearest AP': '#FF6B6B',
        'Equal Power': '#4ECDC4',
        'Load Balancing': '#95E1D3',
        'RL Agent': '#FFD93D'
    }
    markers = {
        'Nearest AP': 'o',
        'Equal Power': 's',
        'Load Balancing': '^',
        'RL Agent': 'D'
    }

    # Plot 1: Energy Efficiency with error bars
    ax = axes[0, 0]
    for strategy_name, strategy_results in results.items():
        means = strategy_results['energy_eff']
        stds = strategy_results.get('energy_eff_std', [0] * len(means))
        ax.errorbar(range(len(circuit_power_labels)), means, yerr=stds,
                   marker=markers[strategy_name], color=colors[strategy_name],
                   linewidth=2, markersize=8, label=strategy_name, capsize=5)
    ax.set_xticks(range(len(circuit_power_labels)))
    ax.set_xticklabels(circuit_power_labels)
    ax.set_xlabel('Circuit Power', fontweight='bold')
    ax.set_ylabel('Energy Efficiency (bits/J)', fontweight='bold')
    ax.set_title('Energy Efficiency vs Circuit Power', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')

    # Plot 2: Data Rate with error bars
    ax = axes[0, 1]
    for strategy_name, strategy_results in results.items():
        means = strategy_results['rate']
        stds = strategy_results.get('rate_std', [0] * len(means))
        ax.errorbar(range(len(circuit_power_labels)), means, yerr=stds,
                   marker=markers[strategy_name], color=colors[strategy_name],
                   linewidth=2, markersize=8, label=strategy_name, capsize=5)
    ax.set_xticks(range(len(circuit_power_labels)))
    ax.set_xticklabels(circuit_power_labels)
    ax.set_xlabel('Circuit Power', fontweight='bold')
    ax.set_ylabel('Average Rate (Mbps)', fontweight='bold')
    ax.set_title('Data Rate vs Circuit Power', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Active APs
    ax = axes[1, 0]
    for strategy_name, strategy_results in results.items():
        means = strategy_results['active_aps']
        stds = strategy_results.get('active_aps_std', [0] * len(means))
        ax.errorbar(range(len(circuit_power_labels)), means, yerr=stds,
                   marker=markers[strategy_name], color=colors[strategy_name],
                   linewidth=2, markersize=8, label=strategy_name, capsize=5)
    ax.set_xticks(range(len(circuit_power_labels)))
    ax.set_xticklabels(circuit_power_labels)
    ax.set_xlabel('Circuit Power', fontweight='bold')
    ax.set_ylabel('Active APs', fontweight='bold')
    ax.set_title('Active APs vs Circuit Power', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: QoS Satisfaction
    ax = axes[1, 1]
    for strategy_name, strategy_results in results.items():
        means = strategy_results['qos_sat']
        stds = strategy_results.get('qos_sat_std', [0] * len(means))
        ax.errorbar(range(len(circuit_power_labels)), means, yerr=stds,
                   marker=markers[strategy_name], color=colors[strategy_name],
                   linewidth=2, markersize=8, label=strategy_name, capsize=5)
    ax.set_xticks(range(len(circuit_power_labels)))
    ax.set_xticklabels(circuit_power_labels)
    ax.set_xlabel('Circuit Power', fontweight='bold')
    ax.set_ylabel('QoS Satisfaction (%)', fontweight='bold')
    ax.set_title('QoS Satisfaction vs Circuit Power', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'detailed_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved detailed comparison: {save_path}")
    plt.close()


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

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
        print(f"  • Fewest Active APs: {min_aps_strategy} - {min_aps:.1f} APs")

    # Analyze trends
    print("\n" + "="*80)
    print("TRENDS")
    print("="*80)

    for strategy_name in results.keys():
        ee_first = results[strategy_name]['energy_eff'][0]
        ee_last = results[strategy_name]['energy_eff'][-1]
        ee_change = ((ee_last - ee_first) / ee_first) * 100

        aps_first = results[strategy_name]['active_aps'][0]
        aps_last = results[strategy_name]['active_aps'][-1]

        print(f"\n{strategy_name}:")
        print(f"  • Energy Eff change ({circuit_power_labels[0]} → {circuit_power_labels[-1]}): {ee_change:+.1f}%")
        print(f"  • Active APs ({circuit_power_labels[0]}): {aps_first:.1f}, ({circuit_power_labels[-1]}): {aps_last:.1f}")
        print(f"  • Circuit power impact: {'HIGH' if abs(ee_change) > 30 else 'MODERATE' if abs(ee_change) > 10 else 'LOW'}")


def print_results_table(results, circuit_power_labels):
    """Print formatted results table"""
    print("\n" + "="*100)
    print("CIRCUIT POWER SENSITIVITY RESULTS")
    print("="*100)

    # Print for each circuit power value
    for cp_idx, cp_label in enumerate(circuit_power_labels):
        print(f"\n{'='*100}")
        print(f"Circuit Power = {cp_label}")
        print(f"{'='*100}")
        print(f"\n{'Strategy':<20} {'Rate (Mbps)':<20} {'EE (bits/J)':<20} {'Active APs':<15} {'QoS (%)':<15}")
        print("-" * 100)

        for strategy_name in results.keys():
            rate = results[strategy_name]['rate'][cp_idx]
            rate_std = results[strategy_name].get('rate_std', [0] * len(circuit_power_labels))[cp_idx]
            ee = results[strategy_name]['energy_eff'][cp_idx]
            ee_std = results[strategy_name].get('energy_eff_std', [0] * len(circuit_power_labels))[cp_idx]
            aps = results[strategy_name]['active_aps'][cp_idx]
            aps_std = results[strategy_name].get('active_aps_std', [0] * len(circuit_power_labels))[cp_idx]
            qos = results[strategy_name]['qos_sat'][cp_idx]
            qos_std = results[strategy_name].get('qos_sat_std', [0] * len(circuit_power_labels))[cp_idx]

            rate_str = f"{rate:.2f} ± {rate_std:.2f}"
            ee_str = f"{ee:.2e} ± {ee_std:.2e}"
            aps_str = f"{aps:.1f} ± {aps_std:.1f}"
            qos_str = f"{qos:.1f} ± {qos_std:.1f}"

            print(f"{strategy_name:<20} {rate_str:<20} {ee_str:<20} {aps_str:<15} {qos_str:<15}")

    print("="*100)


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def test_circuit_power_sensitivity(rl_model_path=None, agent_type='dqn', num_users=10, num_aps=25,
                                   circuit_powers=None, n_episodes=10):
    """Test strategies with different circuit power values

    Args:
        rl_model_path: Optional path to trained RL model
        agent_type: 'dqn' or 'ppo'
        num_users: Number of users
        num_aps: Number of APs
        circuit_powers: List of circuit power values in Watts
        n_episodes: Number of evaluation episodes
    """

    # Default circuit power values
    if circuit_powers is None:
        circuit_powers = [0.1, 0.2, 0.5]  # 100mW, 200mW, 500mW
    circuit_power_labels = [f'{int(cp*1000)}mW' for cp in circuit_powers]

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
    print(f"Episodes per configuration: {n_episodes}")
    if rl_model_path:
        print(f"RL Model: {rl_model_path}")
        print(f"Agent Type: {agent_type.upper()}")
    print("="*80)

    # Store results (include RL agent if model provided)
    all_strategy_names = list(strategies.keys())
    if rl_model_path:
        all_strategy_names.append('RL Agent')

    results = {strategy: {
        'rate': [],
        'rate_std': [],
        'energy_eff': [],
        'energy_eff_std': [],
        'active_aps': [],
        'active_aps_std': [],
        'qos_sat': [],
        'qos_sat_std': []
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

        # Test baseline strategies
        for strategy_name, strategy_func in strategies.items():
            print(f"\n  {strategy_name}:")

            res = evaluate_baseline_circuit_power(network, strategy_func, strategy_name, n_episodes=n_episodes)

            # Store results
            results[strategy_name]['rate'].append(res['mean_rate_mbps'])
            results[strategy_name]['rate_std'].append(res['std_rate_mbps'])
            results[strategy_name]['energy_eff'].append(res['mean_energy_efficiency'])
            results[strategy_name]['energy_eff_std'].append(res['std_energy_efficiency'])
            results[strategy_name]['active_aps'].append(res['mean_active_aps'])
            results[strategy_name]['active_aps_std'].append(res['std_active_aps'])
            results[strategy_name]['qos_sat'].append(res['mean_qos_satisfaction'])
            results[strategy_name]['qos_sat_std'].append(res['std_qos_satisfaction'])

            # Print results
            print(f"    - Avg Rate: {res['mean_rate_mbps']:.2f} ± {res['std_rate_mbps']:.2f} Mbps")
            print(f"    - Energy Eff: {res['mean_energy_efficiency']:.2e} ± {res['std_energy_efficiency']:.2e} bits/J")
            print(f"    - Active APs: {res['mean_active_aps']:.1f} ± {res['std_active_aps']:.1f}/{num_aps}")
            print(f"    - QoS Sat: {res['mean_qos_satisfaction']:.1f} ± {res['std_qos_satisfaction']:.1f}%")

        # Test RL agent if model provided
        if rl_model_path:
            print(f"\n  RL Agent ({agent_type.upper()}):")
            rl_results = evaluate_rl_agent_circuit_power(
                rl_model_path,
                agent_type,
                num_aps,
                num_users,
                circuit_power,
                n_episodes=n_episodes
            )

            # Store results
            results['RL Agent']['rate'].append(rl_results['mean_rate_mbps'])
            results['RL Agent']['rate_std'].append(rl_results['std_rate_mbps'])
            results['RL Agent']['energy_eff'].append(rl_results['mean_energy_efficiency'])
            results['RL Agent']['energy_eff_std'].append(rl_results['std_energy_efficiency'])
            results['RL Agent']['active_aps'].append(rl_results['mean_active_aps'])
            results['RL Agent']['active_aps_std'].append(rl_results['std_active_aps'])
            results['RL Agent']['qos_sat'].append(rl_results['mean_qos_satisfaction'])
            results['RL Agent']['qos_sat_std'].append(rl_results['std_qos_satisfaction'])

            # Print results
            print(f"    - Avg Rate: {rl_results['mean_rate_mbps']:.2f} ± {rl_results['std_rate_mbps']:.2f} Mbps")
            print(f"    - Energy Eff: {rl_results['mean_energy_efficiency']:.2e} ± {rl_results['std_energy_efficiency']:.2e} bits/J")
            print(f"    - Active APs: {rl_results['mean_active_aps']:.1f} ± {rl_results['std_active_aps']:.1f}/{num_aps}")
            print(f"    - QoS Sat: {rl_results['mean_qos_satisfaction']:.1f} ± {rl_results['std_qos_satisfaction']:.1f}%")

    # Print results table
    print_results_table(results, circuit_power_labels)

    # Plot results
    plot_sensitivity_analysis(results, circuit_power_labels, num_aps, num_users, include_rl=rl_model_path is not None)
    plot_detailed_comparison(results, circuit_power_labels)

    # Print summary
    print_summary(results, circuit_power_labels)

    # Save results to JSON
    save_dir = 'results/circuit_power'
    os.makedirs(save_dir, exist_ok=True)
    results_file = os.path.join(save_dir, 'circuit_power_results.json')

    # Convert numpy types to native Python types
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_native = convert_to_native(results)

    with open(results_file, 'w') as f:
        json.dump({
            'configuration': {
                'num_aps': num_aps,
                'num_users': num_users,
                'circuit_powers': circuit_powers,
                'circuit_power_labels': circuit_power_labels,
                'n_episodes': n_episodes,
                'rl_model': rl_model_path,
                'agent_type': agent_type
            },
            'results': results_native
        }, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")

    print("\n" + "="*80)
    print("✅ CIRCUIT POWER SENSITIVITY ANALYSIS COMPLETE")
    print("="*80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Circuit Power Sensitivity Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline strategies only
  python src/circuit_power_analyze.py --num-users 10 --num-aps 25

  # Include RL agent (DQN)
  python src/circuit_power_analyze.py --rl-model experiments/model.zip --agent-type dqn

  # Include RL agent (PPO)
  python src/circuit_power_analyze.py --rl-model experiments/model.zip --agent-type ppo

  # Custom circuit power values
  python src/circuit_power_analyze.py --circuit-powers 0.1 0.2 0.3 0.5
        """
    )

    parser.add_argument('--rl-model', type=str, default=None,
                       help='Path to trained RL model (optional)')
    parser.add_argument('--agent-type', type=str, default='dqn', choices=['dqn', 'ppo'],
                       help='Type of RL agent (default: dqn)')
    parser.add_argument('--num-users', type=int, default=10,
                       help='Number of users (default: 10)')
    parser.add_argument('--num-aps', type=int, default=25,
                       help='Number of APs (default: 25)')
    parser.add_argument('--circuit-powers', type=float, nargs='+', default=None,
                       help='Circuit power values in Watts (default: [0.1, 0.2, 0.5])')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes (default: 10)')

    args = parser.parse_args()

    test_circuit_power_sensitivity(
        rl_model_path=args.rl_model,
        agent_type=args.agent_type,
        num_users=args.num_users,
        num_aps=args.num_aps,
        circuit_powers=args.circuit_powers,
        n_episodes=args.episodes
    )


if __name__ == '__main__':
    main()
