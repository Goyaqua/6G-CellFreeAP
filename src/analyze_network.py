"""
Unified Network Analysis Tool
Comprehensive evaluation and visualization for Cell-Free Massive MIMO with RL

Usage:
    # Basic evaluation with visualizations
    python src/analyze_network.py --mode evaluate --model MODEL_PATH

    # Circuit power sensitivity analysis
    python src/analyze_network.py --mode circuit-power --model MODEL_PATH

    # AP scaling analysis
    python src/analyze_network.py --mode ap-scaling --model MODEL_PATH

    # User scaling analysis
    python src/analyze_network.py --mode user-scaling --model MODEL_PATH
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from environment.cellfree_env import CellFreeEnv
from agents.dqn_agent import DQNAgent
from agents.baselines import BaselineStrategies
from network.cellfree_network import CellFreeNetworkSionna

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_network_topology(network, title="Network Topology", save_path=None):
    """Plot network topology showing AP and user positions"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot APs
    ap_x = network.ap_positions[:, 0]
    ap_y = network.ap_positions[:, 1]
    ax.scatter(ap_x, ap_y, c='red', s=200, marker='^',
               label='Access Points', edgecolors='black', linewidths=2, zorder=3)

    # Plot Users
    user_x = network.user_positions[:, 0]
    user_y = network.user_positions[:, 1]
    ax.scatter(user_x, user_y, c='blue', s=150, marker='o',
               label='Users', edgecolors='black', linewidths=1.5, zorder=3)

    # Add labels
    for i, (x, y) in enumerate(network.ap_positions):
        ax.text(x, y+15, f'AP{i}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    for i, (x, y) in enumerate(network.user_positions):
        ax.text(x, y+15, f'U{i}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-50, network.area_size + 50)
    ax.set_ylim(-50, network.area_size + 50)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved topology plot: {save_path}")
    return fig


def plot_channel_gain_matrix(channel_gain, title="Channel Gain Matrix", save_path=None):
    """Plot channel gain matrix heatmap"""
    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(channel_gain, cmap='hot', aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Channel Gain (linear)', fontsize=11)

    # Add labels
    ax.set_xlabel('Users', fontsize=12)
    ax.set_ylabel('Access Points', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add text annotations
    num_aps, num_users = channel_gain.shape
    for i in range(num_aps):
        for j in range(num_users):
            text = ax.text(j, i, f'{channel_gain[i, j]:.2e}',
                          ha="center", va="center", color="white", fontsize=7)

    # Set ticks
    ax.set_xticks(range(num_users))
    ax.set_yticks(range(num_aps))
    ax.set_xticklabels([f'U{i}' for i in range(num_users)])
    ax.set_yticklabels([f'AP{i}' for i in range(num_aps)])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved channel gain plot: {save_path}")
    return fig


def plot_ap_user_association(association_matrix, strategy_name="Strategy", save_path=None):
    """Plot AP-User association matrix"""
    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(association_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Association (0=inactive, 1=active)', fontsize=11)

    # Add labels
    ax.set_xlabel('Users', fontsize=12)
    ax.set_ylabel('Access Points', fontsize=12)
    ax.set_title(f'AP-User Association: {strategy_name}', fontsize=14, fontweight='bold')

    # Add text annotations
    num_aps, num_users = association_matrix.shape
    for i in range(num_aps):
        for j in range(num_users):
            value = association_matrix[i, j]
            text_color = 'white' if value > 0.5 else 'black'
            text = ax.text(j, i, f'{int(value)}',
                          ha="center", va="center", color=text_color, fontsize=9)

    # Set ticks
    ax.set_xticks(range(num_users))
    ax.set_yticks(range(num_aps))
    ax.set_xticklabels([f'U{i}' for i in range(num_users)])
    ax.set_yticklabels([f'AP{i}' for i in range(num_aps)])

    # Add summary text
    active_aps = np.sum(np.any(association_matrix > 0, axis=1))
    avg_aps_per_user = np.sum(association_matrix) / num_users
    summary_text = f'Active APs: {active_aps}/{num_aps}\nAvg APs/User: {avg_aps_per_user:.1f}'
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved association matrix: {save_path}")
    return fig


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_rl_agent(env, agent, n_episodes=10):
    """Evaluate RL agent and return metrics + final state for visualization"""
    all_rates = []
    all_ee = []
    all_qos = []
    all_active_aps = []
    all_rewards = []
    final_association = None

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
                    # Decode action to get association matrix
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
    """Evaluate baseline strategy"""
    all_rates = []
    all_ee = []
    all_qos = []
    all_active_aps = []
    qos_threshold = 5.0  # Mbps

    for _ in range(n_episodes):
        # Generate channel matrix for this episode
        channel_matrix = network.generate_channel_matrix(batch_size=1)

        # Get baseline decision (now with channel_matrix)
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


# ============================================================================
# MODE 1: BASIC EVALUATION
# ============================================================================

def mode_evaluate(args):
    """Basic evaluation with visualizations"""
    print("="*80)
    print("MODE: BASIC EVALUATION")
    print("="*80)

    # Create environment
    env = CellFreeEnv(
        num_aps=args.num_aps,
        num_users=args.num_users,
        qos_min_rate_mbps=5.0,
        qos_weight=10.0,
        episode_length=100,
        action_type='discrete'
    )
    env.network.circuit_power_per_ap = args.circuit_power

    # Create results directory
    os.makedirs('results/evaluate', exist_ok=True)

    # 1. Plot network topology
    print("\n📍 Generating network topology...")
    plot_network_topology(
        env.network,
        title=f"Network Topology ({args.num_aps} APs, {args.num_users} Users)",
        save_path='results/evaluate/topology.png'
    )

    # 2. Plot channel gain matrix
    print("\n📶 Generating channel gain matrix...")
    # Get large-scale channel gains (path loss) from the network
    channel_gain = env.network.calculate_pathloss().numpy()
    plot_channel_gain_matrix(
        channel_gain,
        title="Channel Gain Matrix (Path Loss)",
        save_path='results/evaluate/channel_gain.png'
    )

    # 3. Evaluate RL agent
    if args.model:
        print("\n🤖 Evaluating RL Agent...")

        # Auto-detect agent type from model filename or path
        if 'ppo' in args.model.lower():
            print("  Detected: PPO agent")
            from agents.ppo_agent import PPOAgent
            agent = PPOAgent(env=env, verbose=0)
        else:
            print("  Detected: DQN agent")
            agent = DQNAgent(env=env, verbose=0)

        agent.load(args.model)
        rl_results = evaluate_rl_agent(env, agent, n_episodes=args.episodes)

        print(f"\n  Mean Reward: {rl_results['mean_reward']:.2f}")
        print(f"  Mean Rate: {rl_results['mean_rate']:.2f} Mbps")
        print(f"  Mean EE: {rl_results['mean_ee']:.2e} bits/J")
        print(f"  Mean QoS: {rl_results['mean_qos']:.1f}%")
        print(f"  Mean Active APs: {rl_results['mean_active_aps']:.1f}")

        # Plot RL association
        if rl_results['association_matrix'] is not None:
            plot_ap_user_association(
                rl_results['association_matrix'],
                strategy_name="RL Agent",
                save_path='results/evaluate/association_rl.png'
            )

    # 4. Evaluate baselines
    print("\n📊 Evaluating Baselines...")
    baselines = {
        'Nearest AP': BaselineStrategies.nearest_ap_max_power,
        'Equal Power': BaselineStrategies.equal_power_all_serve,
        'Load Balancing': BaselineStrategies.load_balancing
    }

    for name, func in baselines.items():
        results = evaluate_baseline(env.network, func, name, n_episodes=args.episodes)
        print(f"\n  {name}:")
        print(f"    Rate: {results['mean_rate']:.2f} Mbps")
        print(f"    EE: {results['mean_ee']:.2e} bits/J")
        print(f"    QoS: {results['mean_qos']:.1f}%")
        print(f"    Active APs: {results['mean_active_aps']:.1f}")

        # Plot association
        safe_name = name.replace(' ', '_').replace('+', '').lower()
        plot_ap_user_association(
            results['association_matrix'],
            strategy_name=name,
            save_path=f'results/evaluate/association_{safe_name}.png'
        )

    print("\n✅ Evaluation complete! Check results/evaluate/ directory")


# ============================================================================
# MODE 2: CIRCUIT POWER SENSITIVITY
# ============================================================================

def mode_circuit_power(args):
    """Circuit power sensitivity analysis"""
    print("="*80)
    print("MODE: CIRCUIT POWER SENSITIVITY")
    print("="*80)

    circuit_powers = [0.1, 0.2, 0.5]  # 100mW, 200mW, 500mW
    labels = ['100mW', '200mW', '500mW']

    os.makedirs('results/circuit_power', exist_ok=True)

    # Storage
    results_data = {name: {'ee': [], 'rate': [], 'active_aps': []}
                    for name in ['RL Agent', 'Nearest AP', 'Equal Power', 'Load Balancing']}

    for cp_idx, circuit_power in enumerate(circuit_powers):
        print(f"\n{'='*60}")
        print(f"Testing Circuit Power: {labels[cp_idx]}")
        print(f"{'='*60}")

        # Create environment
        env = CellFreeEnv(
            num_aps=args.num_aps,
            num_users=args.num_users,
            qos_min_rate_mbps=5.0,
            qos_weight=10.0,
            episode_length=100,
            action_type='discrete'
        )
        env.network.circuit_power_per_ap = circuit_power

        # Test RL
        if args.model:
            # Auto-detect agent type
            if 'ppo' in args.model.lower():
                from agents.ppo_agent import PPOAgent
                agent = PPOAgent(env=env, verbose=0)
            else:
                agent = DQNAgent(env=env, verbose=0)
            agent.load(args.model)
            rl_res = evaluate_rl_agent(env, agent, n_episodes=args.episodes)
            results_data['RL Agent']['ee'].append(rl_res['mean_ee'])
            results_data['RL Agent']['rate'].append(rl_res['mean_rate'])
            results_data['RL Agent']['active_aps'].append(rl_res['mean_active_aps'])
            print(f"  RL Agent: Rate={rl_res['mean_rate']:.2f} Mbps, EE={rl_res['mean_ee']:.2e}")

        # Test baselines
        baselines = {
            'Nearest AP': BaselineStrategies.nearest_ap_max_power,
            'Equal Power': BaselineStrategies.equal_power_all_serve,
            'Load Balancing': BaselineStrategies.load_balancing
        }

        for name, func in baselines.items():
            res = evaluate_baseline(env.network, func, name, n_episodes=args.episodes)
            results_data[name]['ee'].append(res['mean_ee'])
            results_data[name]['rate'].append(res['mean_rate'])
            results_data[name]['active_aps'].append(res['mean_active_aps'])
            print(f"  {name}: Rate={res['mean_rate']:.2f} Mbps, EE={res['mean_ee']:.2e}")

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Circuit Power Sensitivity Analysis', fontsize=16, fontweight='bold')

    colors = {'RL Agent': '#FFD93D', 'Nearest AP': '#2E86AB',
              'Equal Power': '#A23B72', 'Load Balancing': '#F18F01'}
    markers = {'RL Agent': 'D', 'Nearest AP': 'o',
               'Equal Power': 's', 'Load Balancing': '^'}

    # Plot 1: Energy Efficiency
    ax = axes[0]
    for name, data in results_data.items():
        if data['ee']:
            ax.plot(circuit_powers, data['ee'], marker=markers.get(name, 'o'),
                   color=colors.get(name, 'gray'), linewidth=2, markersize=8, label=name)
    ax.set_xlabel('Circuit Power (W)', fontsize=12)
    ax.set_ylabel('Energy Efficiency (bits/J)', fontsize=12)
    ax.set_title('Energy Efficiency vs Circuit Power', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Data Rate
    ax = axes[1]
    for name, data in results_data.items():
        if data['rate']:
            ax.plot(circuit_powers, data['rate'], marker=markers.get(name, 'o'),
                   color=colors.get(name, 'gray'), linewidth=2, markersize=8, label=name)
    ax.set_xlabel('Circuit Power (W)', fontsize=12)
    ax.set_ylabel('Average Rate (Mbps)', fontsize=12)
    ax.set_title('Data Rate vs Circuit Power', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Active APs
    ax = axes[2]
    for name, data in results_data.items():
        if data['active_aps']:
            ax.plot(circuit_powers, data['active_aps'], marker=markers.get(name, 'o'),
                   color=colors.get(name, 'gray'), linewidth=2, markersize=8, label=name)
    ax.set_xlabel('Circuit Power (W)', fontsize=12)
    ax.set_ylabel('Active APs', fontsize=12)
    ax.set_title('Active APs vs Circuit Power', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/circuit_power/comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved comparison plot: results/circuit_power/comparison.png")

    print("\n✅ Circuit power analysis complete!")


# ============================================================================
# MODE 3: AP SCALING
# ============================================================================

def mode_ap_scaling(args):
    """AP scaling analysis"""
    print("="*80)
    print("MODE: AP SCALING ANALYSIS")
    print("="*80)

    ap_configs = [16, 25, 36, 49, 64]
    os.makedirs('results/ap_scaling', exist_ok=True)

    # Storage
    results_data = {name: {'ee': [], 'rate': [], 'active_aps': []}
                    for name in ['RL Agent', 'Nearest AP', 'Equal Power', 'Load Balancing']}

    for num_aps in ap_configs:
        print(f"\n{'='*60}")
        print(f"Testing {num_aps} APs")
        print(f"{'='*60}")

        # Create environment
        env = CellFreeEnv(
            num_aps=num_aps,
            num_users=args.num_users,
            qos_min_rate_mbps=5.0,
            qos_weight=10.0,
            episode_length=100,
            action_type='discrete'
        )
        env.network.circuit_power_per_ap = args.circuit_power

        # Visualizations for this config
        plot_network_topology(
            env.network,
            title=f"Network Topology ({num_aps} APs, {args.num_users} Users)",
            save_path=f'results/ap_scaling/topology_{num_aps}aps.png'
        )

        # Test RL (if model trained for this config)
        if args.model and num_aps == 25:  # Assuming model trained for 25 APs
            # Auto-detect agent type
            if 'ppo' in args.model.lower():
                from agents.ppo_agent import PPOAgent
                agent = PPOAgent(env=env, verbose=0)
            else:
                agent = DQNAgent(env=env, verbose=0)
            agent.load(args.model)
            rl_res = evaluate_rl_agent(env, agent, n_episodes=args.episodes)
            results_data['RL Agent']['ee'].append(rl_res['mean_ee'])
            results_data['RL Agent']['rate'].append(rl_res['mean_rate'])
            results_data['RL Agent']['active_aps'].append(rl_res['mean_active_aps'])
            print(f"  RL Agent: Rate={rl_res['mean_rate']:.2f} Mbps")

        # Test baselines
        baselines = {
            'Nearest AP': BaselineStrategies.nearest_ap_max_power,
            'Equal Power': BaselineStrategies.equal_power_all_serve,
            'Load Balancing': BaselineStrategies.load_balancing
        }

        for name, func in baselines.items():
            res = evaluate_baseline(env.network, func, name, n_episodes=args.episodes)
            results_data[name]['ee'].append(res['mean_ee'])
            results_data[name]['rate'].append(res['mean_rate'])
            results_data[name]['active_aps'].append(res['mean_active_aps'])
            print(f"  {name}: Rate={res['mean_rate']:.2f} Mbps")

    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Network Performance vs Number of Access Points', fontsize=16, fontweight='bold')

    colors = {'RL Agent': '#FFD93D', 'Nearest AP': '#2E86AB',
              'Equal Power': '#A23B72', 'Load Balancing': '#F18F01'}
    markers = {'RL Agent': 'D', 'Nearest AP': 'o',
               'Equal Power': 's', 'Load Balancing': '^'}

    # Energy Efficiency
    ax = axes[0, 2]
    for name, data in results_data.items():
        if data['ee']:
            x_values = ap_configs[:len(data['ee'])]
            ax.plot(x_values, data['ee'], marker=markers.get(name, 'o'),
                   color=colors.get(name, 'gray'), linewidth=2, markersize=8, label=name)
    ax.set_xlabel('Number of APs', fontsize=11)
    ax.set_ylabel('Energy Efficiency (bits/J)', fontsize=11)
    ax.set_title('Energy Efficiency', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Rate
    ax = axes[0, 0]
    for name, data in results_data.items():
        if data['rate']:
            x_values = ap_configs[:len(data['rate'])]
            ax.plot(x_values, data['rate'], marker=markers.get(name, 'o'),
                   color=colors.get(name, 'gray'), linewidth=2, markersize=8, label=name)
    ax.set_xlabel('Number of APs', fontsize=11)
    ax.set_ylabel('Average Rate (Mbps)', fontsize=11)
    ax.set_title('Average Rate per User', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/ap_scaling/comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved comparison plot: results/ap_scaling/comparison.png")

    print("\n✅ AP scaling analysis complete!")


# ============================================================================
# MODE 4: USER SCALING
# ============================================================================

def mode_user_scaling(args):
    """User scaling analysis"""
    print("="*80)
    print("MODE: USER SCALING ANALYSIS")
    print("="*80)

    user_configs = [10, 16, 25]
    os.makedirs('results/user_scaling', exist_ok=True)

    # Storage
    results_data = {name: {'ee': [], 'rate': [], 'active_aps': []}
                    for name in ['RL Agent', 'Nearest AP', 'Equal Power', 'Load Balancing']}

    for num_users in user_configs:
        print(f"\n{'='*60}")
        print(f"Testing {num_users} Users")
        print(f"{'='*60}")

        # Create environment
        env = CellFreeEnv(
            num_aps=args.num_aps,
            num_users=num_users,
            qos_min_rate_mbps=5.0,
            qos_weight=10.0,
            episode_length=100,
            action_type='discrete'
        )
        env.network.circuit_power_per_ap = args.circuit_power

        # Visualizations for this config
        plot_network_topology(
            env.network,
            title=f"Network Topology ({args.num_aps} APs, {num_users} Users)",
            save_path=f'results/user_scaling/topology_{num_users}users.png'
        )

        # Test RL (if model trained for this config)
        if args.model and num_users == 10:  # Assuming model trained for 10 users
            # Auto-detect agent type
            if 'ppo' in args.model.lower():
                from agents.ppo_agent import PPOAgent
                agent = PPOAgent(env=env, verbose=0)
            else:
                agent = DQNAgent(env=env, verbose=0)
            agent.load(args.model)
            rl_res = evaluate_rl_agent(env, agent, n_episodes=args.episodes)
            results_data['RL Agent']['ee'].append(rl_res['mean_ee'])
            results_data['RL Agent']['rate'].append(rl_res['mean_rate'])
            results_data['RL Agent']['active_aps'].append(rl_res['mean_active_aps'])
            print(f"  RL Agent: Rate={rl_res['mean_rate']:.2f} Mbps")

        # Test baselines
        baselines = {
            'Nearest AP': BaselineStrategies.nearest_ap_max_power,
            'Equal Power': BaselineStrategies.equal_power_all_serve,
            'Load Balancing': BaselineStrategies.load_balancing
        }

        for name, func in baselines.items():
            res = evaluate_baseline(env.network, func, name, n_episodes=args.episodes)
            results_data[name]['ee'].append(res['mean_ee'])
            results_data[name]['rate'].append(res['mean_rate'])
            results_data[name]['active_aps'].append(res['mean_active_aps'])
            print(f"  {name}: Rate={res['mean_rate']:.2f} Mbps")

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Network Performance vs Number of Users', fontsize=16, fontweight='bold')

    colors = {'RL Agent': '#FFD93D', 'Nearest AP': '#2E86AB',
              'Equal Power': '#A23B72', 'Load Balancing': '#F18F01'}
    markers = {'RL Agent': 'D', 'Nearest AP': 'o',
               'Equal Power': 's', 'Load Balancing': '^'}

    # Energy Efficiency
    ax = axes[2]
    for name, data in results_data.items():
        if data['ee']:
            x_values = user_configs[:len(data['ee'])]
            ax.plot(x_values, data['ee'], marker=markers.get(name, 'o'),
                   color=colors.get(name, 'gray'), linewidth=2, markersize=8, label=name)
    ax.set_xlabel('Number of Users', fontsize=12)
    ax.set_ylabel('Energy Efficiency (bits/J)', fontsize=12)
    ax.set_title('Energy Efficiency', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Rate
    ax = axes[0]
    for name, data in results_data.items():
        if data['rate']:
            x_values = user_configs[:len(data['rate'])]
            ax.plot(x_values, data['rate'], marker=markers.get(name, 'o'),
                   color=colors.get(name, 'gray'), linewidth=2, markersize=8, label=name)
    ax.set_xlabel('Number of Users', fontsize=12)
    ax.set_ylabel('Average Rate (Mbps)', fontsize=12)
    ax.set_title('Average Rate per User', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Active APs
    ax = axes[1]
    for name, data in results_data.items():
        if data['active_aps']:
            x_values = user_configs[:len(data['active_aps'])]
            ax.plot(x_values, data['active_aps'], marker=markers.get(name, 'o'),
                   color=colors.get(name, 'gray'), linewidth=2, markersize=8, label=name)
    ax.set_xlabel('Number of Users', fontsize=12)
    ax.set_ylabel('Active APs', fontsize=12)
    ax.set_title('Active APs', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/user_scaling/comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved comparison plot: results/user_scaling/comparison.png")

    print("\n✅ User scaling analysis complete!")


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Unified Network Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python src/analyze_network.py --mode evaluate --model experiments/exp_20251210_230304/models/dqn_cellfree_final

  # Circuit power sensitivity
  python src/analyze_network.py --mode circuit-power --model MODEL_PATH

  # AP scaling
  python src/analyze_network.py --mode ap-scaling --model MODEL_PATH

  # User scaling
  python src/analyze_network.py --mode user-scaling --model MODEL_PATH
        """
    )

    parser.add_argument('--mode', type=str, required=True,
                       choices=['evaluate', 'circuit-power', 'ap-scaling', 'user-scaling'],
                       help='Analysis mode')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained RL model')
    parser.add_argument('--num-aps', type=int, default=25,
                       help='Number of APs (default: 25)')
    parser.add_argument('--num-users', type=int, default=10,
                       help='Number of users (default: 10)')
    parser.add_argument('--circuit-power', type=float, default=0.2,
                       help='Circuit power per AP in Watts (default: 0.2)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes (default: 10)')

    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*80)
    print("UNIFIED NETWORK ANALYSIS TOOL")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"APs: {args.num_aps}, Users: {args.num_users}")
    print(f"Circuit Power: {args.circuit_power*1000:.0f} mW")
    print(f"Episodes: {args.episodes}")
    if args.model:
        print(f"RL Model: {args.model}")
    print("="*80)

    # Route to appropriate mode
    if args.mode == 'evaluate':
        mode_evaluate(args)
    elif args.mode == 'circuit-power':
        mode_circuit_power(args)
    elif args.mode == 'ap-scaling':
        mode_ap_scaling(args)
    elif args.mode == 'user-scaling':
        mode_user_scaling(args)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
