"""
Evaluate Trained RL Agent - Safe version with comprehensive visualization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from environment.cellfree_env import CellFreeEnv
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from network.cellfree_network import CellFreeNetworkSionna
from agents.baselines import BaselineStrategies
import json


def safe_evaluate_rl_agent(agent, env, n_episodes=20, max_steps=100):
    """
    Safely evaluate RL agent with timeout protection
    """
    all_rewards = []
    all_ee = []
    all_qos = []
    all_rates = []
    all_sinr = []
    all_active_aps = []

    print(f"Evaluating RL agent for {n_episodes} episodes...")

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        steps = 0

        while not done and steps < max_steps:
            try:
                # Get action from trained model
                action, _ = agent.model.predict(obs, deterministic=True)

                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                steps += 1

                # Store metrics from last step
                if 'energy_efficiency' in info:
                    all_ee.append(info['energy_efficiency'])
                if 'qos_satisfaction' in info:
                    all_qos.append(info['qos_satisfaction'])
                if 'avg_rate_mbps' in info:
                    all_rates.append(info['avg_rate_mbps'])
                if 'sinr_db' in info:
                    all_sinr.append(info['sinr_db'])
                if 'active_aps' in info:
                    all_active_aps.append(info['active_aps'])

            except Exception as e:
                print(f"\n⚠️ Error in episode {episode+1}: {e}")
                break

        all_rewards.append(episode_reward)
        if (episode + 1) % 5 == 0:
            print(f"  Episode {episode+1}/{n_episodes}: Reward={episode_reward:.4f}, Steps={steps}")

    # Calculate statistics
    results = {
        'mean_reward': np.mean(all_rewards) if len(all_rewards) > 0 else 0,
        'std_reward': np.std(all_rewards) if len(all_rewards) > 0 else 0,
        'mean_energy_efficiency': np.mean(all_ee) if len(all_ee) > 0 else 0,
        'std_energy_efficiency': np.std(all_ee) if len(all_ee) > 0 else 0,
        'mean_qos_satisfaction': np.mean(all_qos) if len(all_qos) > 0 else 0,
        'std_qos_satisfaction': np.std(all_qos) if len(all_qos) > 0 else 0,
        'mean_rate_mbps': np.mean(all_rates) if len(all_rates) > 0 else 0,
        'std_rate_mbps': np.std(all_rates) if len(all_rates) > 0 else 0,
        'mean_sinr_db': np.mean(all_sinr) if len(all_sinr) > 0 else 0,
        'std_sinr_db': np.std(all_sinr) if len(all_sinr) > 0 else 0,
        'mean_active_aps': np.mean(all_active_aps) if len(all_active_aps) > 0 else 0,
        'std_active_aps': np.std(all_active_aps) if len(all_active_aps) > 0 else 0,
    }

    return results


def evaluate_baselines(network, n_episodes=20):
    """
    Evaluate baseline strategies
    """
    strategies = {
        'Nearest AP': BaselineStrategies.nearest_ap_max_power,
        'Equal Power': BaselineStrategies.equal_power_all_serve,
        'Load Balancing': BaselineStrategies.load_balancing
    }

    results = {}

    for strategy_name, strategy_func in strategies.items():
        print(f"\nEvaluating baseline: {strategy_name}...")

        all_ee = []
        all_qos = []
        all_rates = []
        all_sinr = []
        all_active_aps = []

        for episode in range(n_episodes):
            # Generate channel
            channel_matrix = network.generate_channel_matrix(batch_size=1)

            # Get allocation
            power_allocation, ap_association = strategy_func(network, channel_matrix)

            # Calculate metrics
            sinr, rates = network.calculate_sinr_and_rate(
                channel_matrix, power_allocation, ap_association
            )
            ee = network.calculate_energy_efficiency(rates, power_allocation, ap_association)
            qos_requirements = np.ones(network.num_users) * 5e6
            qos_sat = network.calculate_qos_satisfaction(rates, qos_requirements)

            # Store
            all_ee.append(ee.numpy()[0])
            all_qos.append(qos_sat.numpy()[0])
            all_rates.append(np.mean(rates.numpy()) / 1e6)
            all_sinr.append(10 * np.log10(np.mean(sinr.numpy())))
            all_active_aps.append(np.sum(np.sum(ap_association, axis=1) > 0))

        results[strategy_name] = {
            'mean_energy_efficiency': np.mean(all_ee),
            'std_energy_efficiency': np.std(all_ee),
            'mean_qos_satisfaction': np.mean(all_qos),
            'std_qos_satisfaction': np.std(all_qos),
            'mean_rate_mbps': np.mean(all_rates),
            'std_rate_mbps': np.std(all_rates),
            'mean_sinr_db': np.mean(all_sinr),
            'std_sinr_db': np.std(all_sinr),
            'mean_active_aps': np.mean(all_active_aps),
            'std_active_aps': np.std(all_active_aps),
        }

        print(f"  - Energy Efficiency: {results[strategy_name]['mean_energy_efficiency']:.2e} bits/Joule")
        print(f"  - QoS Satisfaction: {results[strategy_name]['mean_qos_satisfaction']:.2f}%")
        print(f"  - Avg Rate: {results[strategy_name]['mean_rate_mbps']:.2f} Mbps")

    return results


def plot_comprehensive_comparison(rl_results, baseline_results, save_dir):
    """
    Create comprehensive comparison plots
    """
    os.makedirs(save_dir, exist_ok=True)

    # Prepare data
    all_strategies = ['RL Agent'] + list(baseline_results.keys())
    all_results = {'RL Agent': rl_results, **baseline_results}

    # Metrics to plot
    metrics = [
        ('mean_energy_efficiency', 'Energy Efficiency (bits/Joule)', True),
        ('mean_rate_mbps', 'Average Rate per User (Mbps)', False),
        ('mean_qos_satisfaction', 'QoS Satisfaction (%)', False),
        ('mean_sinr_db', 'Average SINR (dB)', False),
    ]

    # 1. Bar chart comparison (2x2 grid)
    fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle('Performance Comparison: RL Agent vs Baselines', fontsize=16, fontweight='bold')

    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181']

    for idx, (metric_key, ylabel, use_log) in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        values = [all_results[s][metric_key] for s in all_strategies]
        stds = [all_results[s].get(f"std_{metric_key.replace('mean_', '')}", 0) for s in all_strategies]

        bars = ax.bar(all_strategies, values, color=colors[:len(all_strategies)],
                     alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.errorbar(range(len(all_strategies)), values, yerr=stds, fmt='none',
                   ecolor='black', capsize=5, capthick=2)

        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(ylabel, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        if use_log:
            ax.set_yscale('log')

        # Rotate x labels
        ax.set_xticks(range(len(all_strategies)))
        ax.set_xticklabels(all_strategies, rotation=45, ha='right')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if use_log:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1e}', ha='center', va='bottom', fontsize=9)
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plot1_path = os.path.join(save_dir, 'comparison_bar_chart.png')
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot1_path}")
    plt.close()

    # 2. Normalized performance radar chart
    fig2, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Normalize metrics (higher is better)
    metric_names = ['Energy\nEfficiency', 'Rate', 'QoS\nSatisfaction', 'SINR']
    metric_keys_normalized = ['mean_energy_efficiency', 'mean_rate_mbps',
                              'mean_qos_satisfaction', 'mean_sinr_db']

    # Find max values for normalization
    max_values = {k: max([all_results[s][k] for s in all_strategies])
                 for k in metric_keys_normalized}

    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    for idx, strategy in enumerate(all_strategies):
        values = [all_results[strategy][k] / max_values[k] * 100
                 for k in metric_keys_normalized]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=strategy, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax.set_title('Normalized Performance Comparison\n(100% = Best in Each Metric)',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plot2_path = os.path.join(save_dir, 'comparison_radar_chart.png')
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot2_path}")
    plt.close()

    # 3. Active APs comparison (if available)
    if 'mean_active_aps' in baseline_results['Nearest AP']:
        fig3, ax = plt.subplots(figsize=(10, 6))

        baseline_aps = [baseline_results[s]['mean_active_aps'] for s in baseline_results.keys()]
        baseline_names = list(baseline_results.keys())

        bars = ax.bar(baseline_names, baseline_aps, color=colors[1:len(baseline_names)+1],
                     alpha=0.8, edgecolor='black', linewidth=1.5)

        ax.set_ylabel('Number of Active APs', fontsize=12, fontweight='bold')
        ax.set_title('Active APs Comparison (Baselines)', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(baseline_aps) * 1.2)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars, baseline_aps):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plot3_path = os.path.join(save_dir, 'active_aps_comparison.png')
        plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot3_path}")
        plt.close()


def print_results_table(rl_results, baseline_results):
    """
    Print formatted results table with circuit power info
    """
    print("\n" + "="*120)
    print("EVALUATION RESULTS TABLE")
    print("="*120)

    all_strategies = ['RL Agent'] + list(baseline_results.keys())
    all_results = {'RL Agent': rl_results, **baseline_results}

    # Header
    print(f"\n{'Strategy':<20} {'EE (bits/J)':<20} {'Rate (Mbps)':<15} {'QoS (%)':<12} {'SINR (dB)':<12} {'Active APs':<12}")
    print("-" * 120)

    # Data rows
    for strategy in all_strategies:
        r = all_results[strategy]
        ee_str = f"{r['mean_energy_efficiency']:.2e} ± {r['std_energy_efficiency']:.2e}"
        rate_str = f"{r['mean_rate_mbps']:.2f} ± {r['std_rate_mbps']:.2f}"
        qos_str = f"{r['mean_qos_satisfaction']:.1f} ± {r['std_qos_satisfaction']:.1f}"
        sinr_str = f"{r['mean_sinr_db']:.2f} ± {r['std_sinr_db']:.2f}"

        # Add active APs info if available
        if 'mean_active_aps' in r:
            aps_str = f"{r['mean_active_aps']:.1f} ± {r['std_active_aps']:.1f}"
        else:
            aps_str = "N/A"

        print(f"{strategy:<20} {ee_str:<20} {rate_str:<15} {qos_str:<12} {sinr_str:<12} {aps_str:<12}")

    print("="*120)


def calculate_improvements(rl_results, baseline_results):
    """
    Calculate and print improvement percentages
    """
    print("\n" + "="*80)
    print("PERFORMANCE IMPROVEMENTS")
    print("="*80)

    rl_ee = rl_results['mean_energy_efficiency']
    rl_rate = rl_results['mean_rate_mbps']
    rl_qos = rl_results['mean_qos_satisfaction']

    for strategy_name, baseline_result in baseline_results.items():
        baseline_ee = baseline_result['mean_energy_efficiency']
        baseline_rate = baseline_result['mean_rate_mbps']
        baseline_qos = baseline_result['mean_qos_satisfaction']

        ee_improvement = ((rl_ee - baseline_ee) / baseline_ee) * 100
        rate_improvement = ((rl_rate - baseline_rate) / baseline_rate) * 100
        qos_improvement = rl_qos - baseline_qos

        print(f"\nRL Agent vs {strategy_name}:")
        print(f"  - Energy Efficiency Improvement: {ee_improvement:+.2f}%")
        print(f"  - Rate Improvement: {rate_improvement:+.2f}%")
        print(f"  - QoS Improvement: {qos_improvement:+.2f} percentage points")

        if ee_improvement > 0:
            print(f"  ✓ RL Agent is MORE energy efficient")
        else:
            print(f"  ✗ RL Agent is LESS energy efficient")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Trained Agent')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (without extension)')
    parser.add_argument('--agent', type=str, default='dqn', choices=['dqn', 'ppo'],
                       help='Agent type')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--episodes', type=int, default=20,
                       help='Number of evaluation episodes')
    parser.add_argument('--compare_baselines', action='store_true',
                       help='Compare with baseline strategies')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    args = parser.parse_args()

    print("="*80)
    print("RL AGENT EVALUATION")
    print("="*80)

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create environment
    print("\nCreating Environment...")
    env = CellFreeEnv(
        config_path=args.config,
        render_mode=None
    )
    print(f"  ✓ Environment created: {env.num_aps} APs, {env.num_users} users")

    # Create and load agent
    print(f"\nLoading {args.agent.upper()} Agent...")

    if args.agent == 'dqn':
        agent = DQNAgent(env=env, verbose=0)
    else:
        agent = PPOAgent(env=env, verbose=0)

    agent.load(args.model)
    print(f"  ✓ Model loaded from: {args.model}")

    # Evaluate RL agent
    print("\n" + "="*80)
    print("EVALUATING RL AGENT")
    print("="*80)

    rl_results = safe_evaluate_rl_agent(agent, env, n_episodes=args.episodes)

    print("\nRL Agent Performance:")
    print(f"  - Mean Reward: {rl_results['mean_reward']:.4f} ± {rl_results['std_reward']:.4f}")
    print(f"  - Energy Efficiency: {rl_results['mean_energy_efficiency']:.2e} ± {rl_results['std_energy_efficiency']:.2e} bits/Joule")
    print(f"  - QoS Satisfaction: {rl_results['mean_qos_satisfaction']:.2f} ± {rl_results['std_qos_satisfaction']:.2f}%")
    print(f"  - Average Rate: {rl_results['mean_rate_mbps']:.2f} ± {rl_results['std_rate_mbps']:.2f} Mbps")

    # Compare with baselines if requested
    baseline_results = {}
    if args.compare_baselines:
        print("\n" + "="*80)
        print("EVALUATING BASELINE STRATEGIES")
        print("="*80)

        network = CellFreeNetworkSionna(
            num_aps=config['network']['num_aps'],
            num_users=config['network']['num_users'],
            num_antennas_per_ap=config['network']['num_antennas_per_ap'],
            area_size=config['network']['area_size'],
            circuit_power_per_ap=config['network']['circuit_power_per_ap'],
            seed=config['seed']['eval_seed']
        )

        baseline_results = evaluate_baselines(network, n_episodes=args.episodes)

        # Print comparison table
        print_results_table(rl_results, baseline_results)

        # Calculate improvements
        calculate_improvements(rl_results, baseline_results)

        # Plot comprehensive comparison
        print("\n" + "="*80)
        print("GENERATING COMPARISON PLOTS")
        print("="*80)
        plot_comprehensive_comparison(rl_results, baseline_results, args.save_dir)

    # Save results to JSON
    os.makedirs(args.save_dir, exist_ok=True)
    results_file = os.path.join(args.save_dir, 'evaluation_results.json')

    all_results = {
        'RL Agent': rl_results,
        **baseline_results
    }

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    all_results_native = convert_to_native(all_results)

    with open(results_file, 'w') as f:
        json.dump(all_results_native, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")

    print("\n" + "="*80)
    print("✓ EVALUATION COMPLETED")
    print("="*80)


if __name__ == '__main__':
    main()
