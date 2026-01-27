"""
Network Evaluation and Visualization

Comprehensive evaluation tool for Cell-Free Massive MIMO networks.
Supports both Statistical and Ray Tracing (RT) channel models.

Features:
- Network topology visualization (2D/3D)
- Channel gain analysis
- RL agent evaluation (optional)
- Baseline strategies evaluation (always)
- Comparison plots

Usage:
    # Baseline-only with RT
    python src/analysis/evaluate_network.py --config CONFIG --use-rt

    # RL + Baselines with RT
    python src/analysis/evaluate_network.py --config CONFIG --model MODEL --use-rt

    # Statistical mode
    python src/analysis/evaluate_network.py --config CONFIG --model MODEL
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import numpy as np
import tempfile
from environment.cellfree_env import CellFreeEnv
from agents.baselines import BaselineStrategies
from analysis.visualization import (
    plot_network_topology,
    plot_rt_scene_3d,
    plot_channel_gain_matrix,
    plot_ap_user_association
)


def evaluate_rl_agent(env, agent, n_episodes=10):
    """
    Evaluate RL agent and return metrics + final state for visualization

    Args:
        env: CellFreeEnv instance
        agent: RL agent (DQN or PPO)
        n_episodes: Number of evaluation episodes

    Returns:
        Dictionary with metrics and association matrix
    """
    all_rates = []
    all_ee = []
    all_qos = []
    all_active_aps = []
    all_rewards = []
    final_association = None

    print(f"  Evaluating RL agent for {n_episodes} episodes...")

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        max_steps = 100

        while not done and steps < max_steps:
            try:
                action, _ = agent.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
                episode_reward += reward

                if 'avg_rate_mbps' in info:
                    all_rates.append(info['avg_rate_mbps'])
                if 'energy_efficiency' in info:
                    all_ee.append(info['energy_efficiency'])
                if 'qos_satisfaction' in info:
                    all_qos.append(info['qos_satisfaction'])
                if 'active_aps' in info:
                    all_active_aps.append(info['active_aps'])

                # Save last association for visualization
                if episode == n_episodes - 1 and steps == max_steps - 1:
                    _, final_association = env._action_to_allocation(action)

            except Exception as e:
                print(f"⚠️ Error in episode {episode+1}: {e}")
                break

        all_rewards.append(episode_reward)

    results = {
        'strategy': 'RL Agent',
        'mean_reward': float(np.mean(all_rewards)) if all_rewards else 0,
        'mean_rate': float(np.mean(all_rates)) if all_rates else 0,
        'mean_ee': float(np.mean(all_ee)) if all_ee else 0,
        'mean_qos': float(np.mean(all_qos)) if all_qos else 0,
        'mean_active_aps': float(np.mean(all_active_aps)) if all_active_aps else 0,
        'association_matrix': final_association
    }

    return results


def evaluate_baseline(network, strategy_func, strategy_name, n_episodes=10):
    """
    Evaluate baseline strategy

    Args:
        network: CellFreeNetworkSionna instance
        strategy_func: Baseline strategy function
        strategy_name: Name of the strategy
        n_episodes: Number of evaluation episodes

    Returns:
        Dictionary with metrics and association matrix
    """
    all_rates = []
    all_ee = []
    all_qos = []
    all_active_aps = []
    qos_threshold = 5.0  # Mbps

    print(f"  Evaluating {strategy_name} for {n_episodes} episodes...")

    for _ in range(n_episodes):
        # Generate channel matrix for this episode
        channel_matrix = network.generate_channel_matrix(batch_size=1)

        # Get baseline decision
        power_allocation, ap_selection = strategy_func(network, channel_matrix)

        # Calculate SINR and rates
        sinr, rates = network.calculate_sinr_and_rate(channel_matrix, power_allocation, ap_selection)
        rates_mbps = (rates.numpy()[0] / 1e6)  # Convert to Mbps

        mean_rate = float(np.mean(rates_mbps))
        total_power = np.sum(power_allocation) + np.sum(np.sum(ap_selection, axis=1) > 0) * network.circuit_power_per_ap
        ee = (mean_rate * 1e6) / total_power if total_power > 0 else 0
        qos = 100.0 * np.mean(rates_mbps >= qos_threshold)
        active_aps = int(np.sum(np.sum(ap_selection, axis=1) > 0))

        all_rates.append(mean_rate)
        all_ee.append(ee)
        all_qos.append(qos)
        all_active_aps.append(active_aps)

    results = {
        'strategy': strategy_name,
        'mean_rate': float(np.mean(all_rates)),
        'mean_ee': float(np.mean(all_ee)),
        'mean_qos': float(np.mean(all_qos)),
        'mean_active_aps': float(np.mean(all_active_aps)),
        'association_matrix': ap_selection
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Network Evaluation and Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline-only with RT mode
  python src/analysis/evaluate_network.py --config configs/ppo_scenarios/1_sweet_spot_balanced.yaml --use-rt

  # RL + Baselines with RT mode
  python src/analysis/evaluate_network.py --config configs/ppo_scenarios/1_sweet_spot_balanced.yaml --model MODEL_PATH --use-rt

  # Statistical mode
  python src/analysis/evaluate_network.py --config configs/ppo_scenarios/1_sweet_spot_balanced.yaml --model MODEL_PATH
        """
    )

    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained RL model (optional - for RL evaluation)')
    parser.add_argument('--use-rt', action='store_true',
                       help='Force Ray Tracing mode ON (Paris map)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes (default: 10)')
    args = parser.parse_args()

    print("="*80)
    print("NETWORK EVALUATION AND VISUALIZATION")
    print("="*80)

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

    # Write modified config to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
        yaml.dump(config, tmp)
        tmp_config_path = tmp.name

    # Create environment from modified config
    env = CellFreeEnv(config_path=tmp_config_path)

    # Clean up temp file
    os.remove(tmp_config_path)

    # Create results directory
    mode_str = f'rt_{scene_name}' if rt_mode else 'statistical'
    results_dir = f'results/evaluate_network/{mode_str}'
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 Results directory: {results_dir}")

    # 1. Plot network topology (2D top-down view)
    print("\n📍 Generating network topology (2D)...")
    topology_title = f"Network Topology: {env.num_aps} APs, {env.num_users} Users"
    if rt_mode:
        topology_title += f" - {scene_name.upper()} Scene"
    plot_network_topology(
        env.network,
        title=topology_title,
        save_path=f'{results_dir}/topology_2d.png',
        rt_mode=rt_mode
    )

    # 2. Plot 3D view if RT mode
    if rt_mode:
        print("\n📍 Generating RT scene (3D)...")
        plot_rt_scene_3d(
            env.network,
            title=f"RT Scene 3D: {scene_name.upper()} - {env.num_aps} APs + {env.num_users} Users",
            save_path=f'{results_dir}/topology_3d.png'
        )

    # 3. Plot channel gain matrix
    print("\n📶 Generating channel gain matrix...")
    channel_gain = env.network.calculate_pathloss().numpy()
    channel_title = "Channel Gain Matrix (Ray Tracing)" if rt_mode else "Channel Gain Matrix (Path Loss)"
    plot_channel_gain_matrix(
        channel_gain,
        title=channel_title,
        save_path=f'{results_dir}/channel_gain.png'
    )

    # 4. Evaluate RL agent (if model provided)
    rl_results = None
    if args.model:
        print("\n🤖 Evaluating RL Agent...")

        # Auto-detect agent type from model filename
        if 'ppo' in args.model.lower():
            print("  Detected: PPO agent")
            from agents.ppo_agent import PPOAgent
            agent = PPOAgent(env=env, verbose=0)
        else:
            print("  Detected: DQN agent")
            from agents.dqn_agent import DQNAgent
            agent = DQNAgent(env=env, verbose=0)

        agent.load(args.model)
        rl_results = evaluate_rl_agent(env, agent, n_episodes=args.episodes)

        print(f"\n  Mean Reward: {rl_results['mean_reward']:.2f}")
        print(f"  Mean Rate: {rl_results['mean_rate']:.2f} Mbps")
        print(f"  Mean EE: {rl_results['mean_ee']:.2e} bits/J")
        print(f"  Mean QoS: {rl_results['mean_qos']:.1f}%")
        print(f"  Mean Active APs: {rl_results['mean_active_aps']:.1f}/{env.num_aps}")

        # Plot RL association
        if rl_results['association_matrix'] is not None:
            plot_ap_user_association(
                rl_results['association_matrix'],
                strategy_name="RL Agent (Trained)" + (" - RT Mode" if rt_mode else ""),
                save_path=f'{results_dir}/association_rl.png'
            )

    # 5. Evaluate baselines (always)
    print("\n📊 Evaluating Baseline Strategies...")
    baselines = {
        'Nearest AP': BaselineStrategies.nearest_ap_max_power,
        'Equal Power': BaselineStrategies.equal_power_all_serve,
        'Load Balancing': BaselineStrategies.load_balancing
    }

    baseline_results = {}
    for name, func in baselines.items():
        results = evaluate_baseline(env.network, func, name, n_episodes=args.episodes)
        baseline_results[name] = results
        print(f"\n  {name}:")
        print(f"    Rate: {results['mean_rate']:.2f} Mbps")
        print(f"    EE: {results['mean_ee']:.2e} bits/J")
        print(f"    QoS: {results['mean_qos']:.1f}%")
        print(f"    Active APs: {results['mean_active_aps']:.1f}/{env.num_aps}")

        # Plot association
        safe_name = name.replace(' ', '_').replace('+', '').lower()
        plot_ap_user_association(
            results['association_matrix'],
            strategy_name=name + (" - RT Mode" if rt_mode else ""),
            save_path=f'{results_dir}/association_{safe_name}.png'
        )

    # 6. Print comparison summary
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    if rt_mode:
        print(f"RT Mode: {scene_name.upper()} Scene (Realistic Channel)")
    else:
        print("Statistical Mode (Rayleigh + Path Loss)")
    print("="*80)
    print(f"{'Strategy':<20} {'Rate (Mbps)':<15} {'EE (bits/J)':<15} {'QoS (%)':<12} {'Active APs':<12}")
    print("-"*80)

    if args.model and rl_results:
        print(f"{'RL Agent':<20} {rl_results['mean_rate']:<15.2f} {rl_results['mean_ee']:<15.2e} "
              f"{rl_results['mean_qos']:<12.1f} {rl_results['mean_active_aps']:<12.1f}")

    for name, results in baseline_results.items():
        print(f"{name:<20} {results['mean_rate']:<15.2f} {results['mean_ee']:<15.2e} "
              f"{results['mean_qos']:<12.1f} {results['mean_active_aps']:<12.1f}")

    print("="*80)

    print(f"\n✅ Evaluation complete! Check {results_dir}/ directory")
    print("="*80)


if __name__ == '__main__':
    main()
