"""
User Scaling Analysis

Tests network performance with different numbers of users.
Supports both Statistical and Ray Tracing (RT) channel models.

IMPORTANT: This analysis only supports baseline strategies.
RL models cannot be used due to fixed observation space dimensions.

Features:
- Test different user counts: [10, 16, 25]
- Baseline strategies evaluation (only)
- Comparison plots
- RT mode support

Usage:
    # Baseline-only with RT mode
    python src/analysis/user_scaling_analysis.py --config CONFIG --use-rt

    # Statistical mode
    python src/analysis/user_scaling_analysis.py --config CONFIG
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from network.cellfree_network import CellFreeNetworkSionna
from agents.baselines import BaselineStrategies
from analysis.visualization import plot_network_topology


def evaluate_baseline_user_scaling(network, strategy_func, strategy_name, n_episodes=10):
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


def plot_scaling_analysis(results, user_configs, rt_mode, scene_name, save_dir):
    """
    Plot user scaling analysis

    Args:
        results: Dictionary of results for each strategy
        user_configs: List of user counts tested
        rt_mode: Whether RT mode is active
        scene_name: RT scene name
        save_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    title = f'User Scaling Analysis'
    if rt_mode:
        title += f' - {scene_name.upper()} (RT Mode)'
    else:
        title += ' - Statistical Mode'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    colors = {
        'Nearest AP': '#2E86AB',
        'Equal Power': '#A23B72',
        'Load Balancing': '#F18F01'
    }
    markers = {
        'Nearest AP': 'o',
        'Equal Power': 's',
        'Load Balancing': '^'
    }

    # Plot 1: Energy Efficiency
    ax = axes[0]
    for strategy_name, strategy_results in results.items():
        if strategy_results['ee']:
            x_values = user_configs[:len(strategy_results['ee'])]
            ax.plot(x_values, strategy_results['ee'],
                   marker=markers.get(strategy_name, 'o'),
                   color=colors.get(strategy_name, 'gray'),
                   linewidth=2, markersize=8, label=strategy_name)
    ax.set_xlabel('Number of Users', fontsize=12, fontweight='bold')
    ax.set_ylabel('Energy Efficiency (bits/J)', fontsize=12, fontweight='bold')
    ax.set_title('Energy Efficiency vs Users', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Data Rate
    ax = axes[1]
    for strategy_name, strategy_results in results.items():
        if strategy_results['rate']:
            x_values = user_configs[:len(strategy_results['rate'])]
            ax.plot(x_values, strategy_results['rate'],
                   marker=markers.get(strategy_name, 'o'),
                   color=colors.get(strategy_name, 'gray'),
                   linewidth=2, markersize=8, label=strategy_name)
    ax.set_xlabel('Number of Users', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Rate per User (Mbps)', fontsize=12, fontweight='bold')
    ax.set_title('Average Rate vs Users', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Active APs
    ax = axes[2]
    for strategy_name, strategy_results in results.items():
        if strategy_results['active_aps']:
            x_values = user_configs[:len(strategy_results['active_aps'])]
            ax.plot(x_values, strategy_results['active_aps'],
                   marker=markers.get(strategy_name, 'o'),
                   color=colors.get(strategy_name, 'gray'),
                   linewidth=2, markersize=8, label=strategy_name)
    ax.set_xlabel('Number of Users', fontsize=12, fontweight='bold')
    ax.set_ylabel('Active APs', fontsize=12, fontweight='bold')
    ax.set_title('Active APs vs Users', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='User Scaling Analysis (Baseline Strategies Only)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline-only with RT mode
  python src/analysis/user_scaling_analysis.py --config CONFIG --use-rt

  # Statistical mode
  python src/analysis/user_scaling_analysis.py --config CONFIG

Note: RL models are NOT supported due to observation space dimension mismatch.
      Only baseline strategies will be evaluated.
        """
    )

    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--use-rt', action='store_true',
                       help='Force Ray Tracing mode ON (Paris map)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes (default: 10)')
    args = parser.parse_args()

    print("="*80)
    print("USER SCALING ANALYSIS (BASELINE STRATEGIES ONLY)")
    print("="*80)
    print("⚠️  Note: RL models cannot be used (dimension mismatch)")
    print("="*80)

    # User counts to test
    user_configs = [10, 16, 25]

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

    # Get num_aps from config
    num_aps = config['network']['num_aps']

    # Create results directory
    mode_str = f'rt_{scene_name}' if rt_mode else 'statistical'
    results_dir = f'results/user_scaling/{mode_str}'
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 Results directory: {results_dir}")

    # Storage for baseline strategies
    results = {name: {'ee': [], 'rate': [], 'active_aps': [], 'qos': []}
               for name in ['Nearest AP', 'Equal Power', 'Load Balancing']}

    # Test each user count
    for num_users in user_configs:
        print(f"\n{'='*60}")
        print(f"Testing {num_users} Users")
        print(f"{'='*60}")

        # Update config with current user count
        config['network']['num_users'] = num_users

        # Write modified config to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp_config_path = tmp.name

        # Create network directly (no environment needed for baselines)
        network = CellFreeNetworkSionna(
            num_aps=config['network']['num_aps'],
            num_users=config['network']['num_users'],
            num_antennas_per_ap=config['network']['num_antennas_per_ap'],
            area_size=config['network']['area_size'],
            circuit_power_per_ap=config['network']['circuit_power_per_ap'],
            use_ray_tracing=rt_mode,
            rt_scene_name=scene_name if rt_mode else None,
            rt_ap_height=config['network'].get('rt_ap_height', 8.0) if rt_mode else None,
            rt_user_height=config['network'].get('rt_user_height', 1.5) if rt_mode else None,
            seed=config['seed']['eval_seed']
        )

        # Clean up temp file
        os.remove(tmp_config_path)

        # Visualizations for this config
        print(f"\n📍 Generating network topology ({num_users} users)...")
        topology_title = f"Network Topology: {num_aps} APs, {num_users} Users"
        if rt_mode:
            topology_title += f" - {scene_name.upper()}"
        plot_network_topology(
            network,
            title=topology_title,
            save_path=f'{results_dir}/topology_{num_users}users.png',
            rt_mode=rt_mode
        )

        # Test baseline strategies
        baselines = {
            'Nearest AP': BaselineStrategies.nearest_ap_max_power,
            'Equal Power': BaselineStrategies.equal_power_all_serve,
            'Load Balancing': BaselineStrategies.load_balancing
        }

        for name, func in baselines.items():
            print(f"\n  {name}:")
            res = evaluate_baseline_user_scaling(network, func, name, n_episodes=args.episodes)

            results[name]['ee'].append(res['mean_energy_efficiency'])
            results[name]['rate'].append(res['mean_rate_mbps'])
            results[name]['active_aps'].append(res['mean_active_aps'])
            results[name]['qos'].append(res['mean_qos_satisfaction'])

            print(f"    - Rate: {res['mean_rate_mbps']:.2f} Mbps")
            print(f"    - Energy Eff: {res['mean_energy_efficiency']:.2e} bits/J")
            print(f"    - Active APs: {res['mean_active_aps']:.1f}")

    # Plot results
    print("\n📊 Generating plots...")
    plot_scaling_analysis(results, user_configs, rt_mode, scene_name, results_dir)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for strategy_name in results.keys():
        print(f"\n{strategy_name}:")
        print(f"  10 Users → 25 Users:")
        if len(results[strategy_name]['rate']) >= 3:
            rate_change = results[strategy_name]['rate'][2] - results[strategy_name]['rate'][0]
            ee_change_pct = ((results[strategy_name]['ee'][2] - results[strategy_name]['ee'][0]) /
                            results[strategy_name]['ee'][0] * 100) if results[strategy_name]['ee'][0] > 0 else 0
            print(f"    - Rate change: {rate_change:+.2f} Mbps")
            print(f"    - Energy Eff change: {ee_change_pct:+.1f}%")

    print("\n✅ User scaling analysis complete!")
    print(f"📁 Results saved to: {results_dir}/")
    print("="*80)


if __name__ == '__main__':
    main()
