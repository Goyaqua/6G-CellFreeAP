"""
Verify Agent Adaptivity - Detailed Behavior Analysis
This script analyzes how the RL agent adapts its decisions across different scenarios
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from environment.cellfree_env import CellFreeEnv
from agents.dqn_agent import DQNAgent
import argparse


def analyze_agent_adaptivity(model_path, num_episodes=100, circuit_power=0.2):
    """
    Analyze how the RL agent adapts its decisions across different scenarios

    Args:
        model_path: Path to trained RL model
        num_episodes: Number of test episodes
        circuit_power: Circuit power per AP (Watts)

    Returns:
        Dictionary with detailed statistics
    """

    print("="*80)
    print(f"AGENT ADAPTIVITY ANALYSIS")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Circuit Power: {circuit_power*1000:.0f} mW")
    print()

    # Create environment
    env = CellFreeEnv(
        num_aps=25,
        num_users=10,
        qos_min_rate_mbps=5.0,
        qos_weight=10.0,
        episode_length=100,
        action_type='discrete'
    )

    # Override circuit power
    env.network.circuit_power_per_ap = circuit_power

    # Load agent
    agent = DQNAgent(env=env, verbose=0)
    agent.load(model_path)

    # Storage for analysis
    all_active_aps = []  # Per-step active AP counts
    episode_avg_active_aps = []  # Per-episode averages
    episode_min_active_aps = []
    episode_max_active_aps = []
    episode_std_active_aps = []
    all_rewards = []
    all_rates = []
    all_ee = []

    print(f"Running {num_episodes} episodes...")

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_active_aps = []
        episode_rewards = []
        episode_rates = []
        episode_ee = []

        done = False
        steps = 0
        max_steps = 100

        while not done and steps < max_steps:
            try:
                action, _ = agent.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1

                # Collect active AP count
                if 'active_aps' in info:
                    active_count = info['active_aps']
                    episode_active_aps.append(active_count)
                    all_active_aps.append(active_count)

                # Collect other metrics
                episode_rewards.append(reward)
                if 'avg_rate_mbps' in info:
                    episode_rates.append(info['avg_rate_mbps'])
                if 'energy_efficiency' in info:
                    episode_ee.append(info['energy_efficiency'])

            except Exception as e:
                print(f"⚠️ Error in episode {episode+1}: {e}")
                break

        # Episode statistics
        if episode_active_aps:
            episode_avg_active_aps.append(np.mean(episode_active_aps))
            episode_min_active_aps.append(np.min(episode_active_aps))
            episode_max_active_aps.append(np.max(episode_active_aps))
            episode_std_active_aps.append(np.std(episode_active_aps))

        if episode_rewards:
            all_rewards.append(np.mean(episode_rewards))
        if episode_rates:
            all_rates.append(np.mean(episode_rates))
        if episode_ee:
            all_ee.append(np.mean(episode_ee))

        # Progress indicator
        if (episode + 1) % 20 == 0:
            print(f"  Progress: {episode+1}/{num_episodes} episodes completed")

    # Statistical Analysis
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)

    if all_active_aps:
        print(f"\n📊 Active AP Count Distribution (All Steps):")
        print(f"  • Mean: {np.mean(all_active_aps):.3f} APs")
        print(f"  • Std Dev: {np.std(all_active_aps):.3f} APs")
        print(f"  • Min: {np.min(all_active_aps)} APs")
        print(f"  • Max: {np.max(all_active_aps)} APs")
        print(f"  • Median: {np.median(all_active_aps):.1f} APs")
        print(f"  • 25th percentile: {np.percentile(all_active_aps, 25):.1f} APs")
        print(f"  • 75th percentile: {np.percentile(all_active_aps, 75):.1f} APs")

        # Unique values frequency
        unique, counts = np.unique(all_active_aps, return_counts=True)
        print(f"\n📈 Frequency Distribution:")
        for val, count in zip(unique, counts):
            percentage = (count / len(all_active_aps)) * 100
            bar = '█' * int(percentage / 2)
            print(f"  {val:2.0f} APs: {count:4d} times ({percentage:5.1f}%) {bar}")

        # Adaptivity metrics
        num_unique_values = len(unique)
        std_dev = np.std(all_active_aps)

        print(f"\n🎯 Adaptivity Metrics:")
        print(f"  • Unique AP counts used: {num_unique_values}")
        print(f"  • Standard deviation: {std_dev:.3f}")

        if std_dev < 0.5:
            adaptivity_level = "LOW - Agent is mostly using fixed AP count"
        elif std_dev < 1.5:
            adaptivity_level = "MODERATE - Agent shows some adaptation"
        else:
            adaptivity_level = "HIGH - Agent actively adapts to scenarios"

        print(f"  • Adaptivity Level: {adaptivity_level}")

        # Episode-level variation
        if episode_avg_active_aps:
            print(f"\n📉 Per-Episode Variation:")
            print(f"  • Avg episode mean: {np.mean(episode_avg_active_aps):.3f} APs")
            print(f"  • Avg within-episode std: {np.mean(episode_std_active_aps):.3f} APs")
            print(f"  • Episode means range: [{np.min(episode_avg_active_aps):.2f}, {np.max(episode_avg_active_aps):.2f}]")

    # Performance Metrics
    if all_rewards:
        print(f"\n⚡ Performance Metrics:")
        print(f"  • Mean Reward: {np.mean(all_rewards):.4f}")
        print(f"  • Mean Rate: {np.mean(all_rates):.2f} Mbps")
        print(f"  • Mean Energy Eff: {np.mean(all_ee):.2e} bits/J")

    # Visualization
    create_adaptivity_plots(
        all_active_aps,
        episode_avg_active_aps,
        episode_std_active_aps,
        circuit_power
    )

    return {
        'all_active_aps': all_active_aps,
        'mean': np.mean(all_active_aps) if all_active_aps else 0,
        'std': np.std(all_active_aps) if all_active_aps else 0,
        'unique_counts': num_unique_values if all_active_aps else 0,
        'adaptivity_level': adaptivity_level if all_active_aps else 'UNKNOWN'
    }


def create_adaptivity_plots(all_active_aps, episode_means, episode_stds, circuit_power):
    """Create visualization plots for adaptivity analysis"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Agent Adaptivity Analysis\nCircuit Power: {circuit_power*1000:.0f} mW',
                 fontsize=14, fontweight='bold')

    # Plot 1: Histogram of Active AP Counts
    ax1 = axes[0, 0]
    bins = np.arange(0, 26) - 0.5
    ax1.hist(all_active_aps, bins=bins, align='mid', rwidth=0.8,
             color='#6a0dad', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(all_active_aps), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(all_active_aps):.2f}')
    ax1.set_xlabel('Number of Active APs', fontsize=11)
    ax1.set_ylabel('Frequency (Steps)', fontsize=11)
    ax1.set_title('Distribution of Active AP Counts', fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xlim(-0.5, 25.5)

    # Plot 2: Time series (first 500 steps)
    ax2 = axes[0, 1]
    plot_length = min(500, len(all_active_aps))
    ax2.plot(all_active_aps[:plot_length], linewidth=0.8, color='#1f77b4', alpha=0.7)
    ax2.axhline(np.mean(all_active_aps), color='red', linestyle='--',
                linewidth=1.5, label=f'Overall Mean: {np.mean(all_active_aps):.2f}')
    ax2.set_xlabel('Step Number', fontsize=11)
    ax2.set_ylabel('Active APs', fontsize=11)
    ax2.set_title(f'Active AP Count Over Time (First {plot_length} Steps)', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Plot 3: Episode-level statistics
    ax3 = axes[1, 0]
    episodes = np.arange(1, len(episode_means) + 1)
    ax3.errorbar(episodes, episode_means, yerr=episode_stds,
                 fmt='o', markersize=3, alpha=0.6, capsize=2,
                 color='#2ca02c', ecolor='gray')
    ax3.axhline(np.mean(episode_means), color='red', linestyle='--',
                linewidth=1.5, label=f'Grand Mean: {np.mean(episode_means):.2f}')
    ax3.set_xlabel('Episode Number', fontsize=11)
    ax3.set_ylabel('Mean Active APs', fontsize=11)
    ax3.set_title('Per-Episode Mean Active APs (± Std Dev)', fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Plot 4: Box plot
    ax4 = axes[1, 1]
    all_active_aps_array = np.array(all_active_aps)
    unique_values = np.unique(all_active_aps_array)
    data_grouped = [all_active_aps_array[all_active_aps_array == val]
                    for val in unique_values]

    bp = ax4.boxplot(data_grouped, positions=unique_values, widths=0.6,
                     patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor('#ff7f0e')
        patch.set_alpha(0.6)

    ax4.set_xlabel('Active AP Count', fontsize=11)
    ax4.set_ylabel('Distribution', fontsize=11)
    ax4.set_title('Statistical Distribution by AP Count', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save plot
    os.makedirs('results', exist_ok=True)
    output_file = f'results/agent_adaptivity_analysis_{circuit_power*1000:.0f}mW.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plots saved to: {output_file}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze Agent Adaptivity')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained RL model')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of test episodes (default: 100)')
    parser.add_argument('--circuit-power', type=float, default=0.2,
                       help='Circuit power per AP in Watts (default: 0.2)')

    args = parser.parse_args()

    analyze_agent_adaptivity(
        model_path=args.model,
        num_episodes=args.episodes,
        circuit_power=args.circuit_power
    )

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
