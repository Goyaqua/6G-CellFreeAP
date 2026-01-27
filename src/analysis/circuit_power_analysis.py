"""
Circuit Power Sensitivity Analysis

Tests how different circuit power values affect network performance.
Supports both Statistical and Ray Tracing (RT) channel models.

Features:
- Test multiple circuit power values (100mW, 200mW, 500mW)
- RL agent evaluation (optional)
- Baseline strategies evaluation (always)
- Trend analysis and key findings
- RT mode support

Usage:
    # Baseline-only with RT
    python src/analysis/circuit_power_analysis.py --config CONFIG --use-rt

    # RL + Baselines with RT
    python src/analysis/circuit_power_analysis.py --config CONFIG --model MODEL --use-rt
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from environment.cellfree_env import CellFreeEnv
from agents.baselines import BaselineStrategies


def evaluate_rl_agent_circuit_power(agent, env, n_episodes=10):
    """
    Evaluate RL agent with specific circuit power setting

    Args:
        agent: RL agent (DQN or PPO)
        env: CellFreeEnv instance
        n_episodes: Number of evaluation episodes

    Returns:
        Dictionary with mean metrics
    """
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
        max_steps = 100

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


def evaluate_baseline_circuit_power(network, strategy_func, strategy_name, n_episodes=10):
    """
    Evaluate baseline strategy

    Args:
        network: CellFreeNetworkSionna instance
        strategy_func: Baseline strategy function
        strategy_name: Name of the strategy
        n_episodes: Number of evaluation episodes

    Returns:
        Dictionary with mean metrics
    """
    all_ee = []
    all_qos = []
    all_rates = []
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
        all_active_aps.append(np.sum(np.sum(ap_association, axis=1) > 0))

    return {
        'mean_energy_efficiency': np.mean(all_ee),
        'mean_qos_satisfaction': np.mean(all_qos),
        'mean_rate_mbps': np.mean(all_rates),
        'mean_active_aps': np.mean(all_active_aps),
    }


def plot_sensitivity_analysis(results, circuit_power_labels, rt_mode, scene_name, save_dir):
    """
    Plot circuit power sensitivity analysis

    Args:
        results: Dictionary of results for each strategy
        circuit_power_labels: Labels for circuit power values
        rt_mode: Whether RT mode is active
        scene_name: RT scene name
        save_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    title = f'Circuit Power Sensitivity Analysis'
    if rt_mode:
        title += f' - {scene_name.upper()} (RT Mode)'
    else:
        title += ' - Statistical Mode'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    colors = {
        'Nearest AP': '#2E86AB',
        'Equal Power': '#A23B72',
        'Load Balancing': '#F18F01',
        'RL Agent': '#FFD93D'
    }
    markers = {
        'Nearest AP': 'o',
        'Equal Power': 's',
        'Load Balancing': '^',
        'RL Agent': 'D'
    }

    # Plot 1: Energy Efficiency
    ax = axes[0]
    for strategy_name, strategy_results in results.items():
        values = strategy_results['ee']
        ax.plot(circuit_power_labels, values,
               marker=markers.get(strategy_name, 'o'),
               color=colors.get(strategy_name, 'gray'),
               linewidth=2, markersize=8, label=strategy_name)
    ax.set_xlabel('Circuit Power per AP', fontsize=12, fontweight='bold')
    ax.set_ylabel('Energy Efficiency (bits/J)', fontsize=12, fontweight='bold')
    ax.set_title('Energy Efficiency vs Circuit Power', fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Data Rate
    ax = axes[1]
    for strategy_name, strategy_results in results.items():
        values = strategy_results['rate']
        ax.plot(circuit_power_labels, values,
               marker=markers.get(strategy_name, 'o'),
               color=colors.get(strategy_name, 'gray'),
               linewidth=2, markersize=8, label=strategy_name)
    ax.set_xlabel('Circuit Power per AP', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Rate (Mbps)', fontsize=12, fontweight='bold')
    ax.set_title('Data Rate vs Circuit Power', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Active APs
    ax = axes[2]
    for strategy_name, strategy_results in results.items():
        values = strategy_results['active_aps']
        ax.plot(circuit_power_labels, values,
               marker=markers.get(strategy_name, 'o'),
               color=colors.get(strategy_name, 'gray'),
               linewidth=2, markersize=8, label=strategy_name)
    ax.set_xlabel('Circuit Power per AP', fontsize=12, fontweight='bold')
    ax.set_ylabel('Active APs', fontsize=12, fontweight='bold')
    ax.set_title('Active APs vs Circuit Power', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: {save_path}")
    plt.close()


def print_key_findings(results, circuit_power_labels):
    """Print key findings for each circuit power value"""
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    for cp_idx, cp_label in enumerate(circuit_power_labels):
        print(f"\nCircuit Power = {cp_label}:")

        # Best rate
        best_rate_strategy = max(results.keys(),
                                key=lambda s: results[s]['rate'][cp_idx])
        best_rate = results[best_rate_strategy]['rate'][cp_idx]
        print(f"  • Best Rate: {best_rate_strategy} - {best_rate:.2f} Mbps")

        # Best energy efficiency
        best_ee_strategy = max(results.keys(),
                              key=lambda s: results[s]['ee'][cp_idx])
        best_ee = results[best_ee_strategy]['ee'][cp_idx]
        print(f"  • Best Energy Eff: {best_ee_strategy} - {best_ee:.2e} bits/J")

        # Fewest active APs
        min_aps_strategy = min(results.keys(),
                              key=lambda s: results[s]['active_aps'][cp_idx])
        min_aps = results[min_aps_strategy]['active_aps'][cp_idx]
        print(f"  • Fewest Active APs: {min_aps_strategy} - {min_aps:.1f} APs")


def print_trends(results):
    """Print trend analysis"""
    print("\n" + "="*80)
    print("TRENDS (100mW → 500mW)")
    print("="*80)

    for strategy_name in results.keys():
        ee_100mw = results[strategy_name]['ee'][0]
        ee_500mw = results[strategy_name]['ee'][2]
        ee_change = ((ee_500mw - ee_100mw) / ee_100mw) * 100 if ee_100mw > 0 else 0

        aps_100mw = results[strategy_name]['active_aps'][0]
        aps_500mw = results[strategy_name]['active_aps'][2]

        print(f"\n{strategy_name}:")
        print(f"  • Energy Eff change: {ee_change:+.1f}%")
        print(f"  • Active APs (100mW): {aps_100mw:.1f}, (500mW): {aps_500mw:.1f}")
        impact = 'HIGH' if abs(ee_change) > 30 else 'MODERATE' if abs(ee_change) > 10 else 'LOW'
        print(f"  • Circuit power impact: {impact}")


def main():
    parser = argparse.ArgumentParser(
        description='Circuit Power Sensitivity Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline-only with RT mode
  python src/analysis/circuit_power_analysis.py --config CONFIG --use-rt

  # RL + Baselines with RT mode
  python src/analysis/circuit_power_analysis.py --config CONFIG --model MODEL --use-rt
        """
    )

    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained RL model (optional)')
    parser.add_argument('--use-rt', action='store_true',
                       help='Force Ray Tracing mode ON (Paris map)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes (default: 10)')
    args = parser.parse_args()

    print("="*80)
    print("CIRCUIT POWER SENSITIVITY ANALYSIS")
    print("="*80)

    # Circuit power values to test
    circuit_powers = [0.1, 0.2, 0.5]  # 100mW, 200mW, 500mW
    circuit_power_labels = ['100mW', '200mW', '500mW']

    # Load configuration
    print(f"\n📄 Loading config: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Force RT mode if --use-rt flag is set
    rt_mode = False
    scene_name = 'etoile'

    if args.use_rt:
        print("\n" + "="*80)
        print("🗼 FORCE RAY TRACING MODE: WELCOME TO PARIS! 🗼")
        print("="*80)
        config['network']['use_ray_tracing'] = True
        if 'rt_scene_name' not in config['network']:
            config['network']['rt_scene_name'] = 'etoile'
            config['network']['rt_ap_height'] = 8.0
            config['network']['rt_user_height'] = 1.5
        print("⚠️  Config override: use_ray_tracing = TRUE")
        print("-"*80 + "\n")

    rt_mode = config['network'].get('use_ray_tracing', False)
    scene_name = config['network'].get('rt_scene_name', 'etoile')

    if rt_mode:
        print(f"🌍 RT Mode: ENABLED (Scene: {scene_name.upper()})")
    else:
        print("📡 Channel Mode: Statistical (Rayleigh + Path Loss)")

    # Create results directory
    mode_str = f'rt_{scene_name}' if rt_mode else 'statistical'
    results_dir = f'results/circuit_power/{mode_str}'
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 Results directory: {results_dir}")

    # Storage for all strategies
    all_strategy_names = ['Nearest AP', 'Equal Power', 'Load Balancing']
    if args.model:
        all_strategy_names.append('RL Agent')

    results = {strategy: {'ee': [], 'rate': [], 'active_aps': [], 'qos': []}
               for strategy in all_strategy_names}

    # Test each circuit power value
    for cp_idx, circuit_power in enumerate(circuit_powers):
        print(f"\n{'='*60}")
        print(f"Testing Circuit Power: {circuit_power_labels[cp_idx]}")
        print(f"{'='*60}")

        # Update config with current circuit power
        config['network']['circuit_power_per_ap'] = circuit_power

        # Write modified config to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp_config_path = tmp.name

        # Create environment
        env = CellFreeEnv(config_path=tmp_config_path)

        # Clean up temp file
        os.remove(tmp_config_path)

        # Test RL agent (if model provided)
        if args.model:
            print(f"\n  RL Agent:")

            # Auto-detect agent type
            if 'ppo' in args.model.lower():
                from agents.ppo_agent import PPOAgent
                agent = PPOAgent(env=env, verbose=0)
            else:
                from agents.dqn_agent import DQNAgent
                agent = DQNAgent(env=env, verbose=0)

            agent.load(args.model)
            rl_res = evaluate_rl_agent_circuit_power(agent, env, n_episodes=args.episodes)

            results['RL Agent']['ee'].append(rl_res['mean_energy_efficiency'])
            results['RL Agent']['rate'].append(rl_res['mean_rate_mbps'])
            results['RL Agent']['active_aps'].append(rl_res['mean_active_aps'])
            results['RL Agent']['qos'].append(rl_res['mean_qos_satisfaction'])

            print(f"    - Rate: {rl_res['mean_rate_mbps']:.2f} Mbps")
            print(f"    - Energy Eff: {rl_res['mean_energy_efficiency']:.2e} bits/J")
            print(f"    - Active APs: {rl_res['mean_active_aps']:.1f}")

        # Test baseline strategies
        baselines = {
            'Nearest AP': BaselineStrategies.nearest_ap_max_power,
            'Equal Power': BaselineStrategies.equal_power_all_serve,
            'Load Balancing': BaselineStrategies.load_balancing
        }

        for name, func in baselines.items():
            print(f"\n  {name}:")
            res = evaluate_baseline_circuit_power(env.network, func, name, n_episodes=args.episodes)

            results[name]['ee'].append(res['mean_energy_efficiency'])
            results[name]['rate'].append(res['mean_rate_mbps'])
            results[name]['active_aps'].append(res['mean_active_aps'])
            results[name]['qos'].append(res['mean_qos_satisfaction'])

            print(f"    - Rate: {res['mean_rate_mbps']:.2f} Mbps")
            print(f"    - Energy Eff: {res['mean_energy_efficiency']:.2e} bits/J")
            print(f"    - Active APs: {res['mean_active_aps']:.1f}")

    # Plot results
    print("\n📊 Generating plots...")
    plot_sensitivity_analysis(results, circuit_power_labels, rt_mode, scene_name, results_dir)

    # Print analysis
    print_key_findings(results, circuit_power_labels)
    print_trends(results)

    print("\n✅ Circuit power analysis complete!")
    print(f"📁 Results saved to: {results_dir}/")
    print("="*80)


if __name__ == '__main__':
    main()
