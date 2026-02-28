#!/usr/bin/env python3
"""
Behavior Analysis Script: Action Distribution & Strategy Analysis
Provides "Explainable AI" insights into what the RL agent learned

Usage:
    python src/analyze_behavior.py \
        --model experiments/exp_20251218_145715/models/ppo_cellfree_final.zip \
        --config configs/ppo_scenarios/1_sweet_spot_balanced.yaml \
        --episodes 100 \
        --save_dir results/behavior_analysis/scenario1
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import json

from environment.cellfree_env import CellFreeEnv
from stable_baselines3 import PPO


# Action Space Mapping (From your methodology)
POWER_LEVELS = ["Very Low", "Low", "Medium", "High", "Max"]  # 5 levels
CLUSTERING_STRATEGIES = ["Nearest-Only", "Top-2", "Top-3", "Top-5", "Top-25%", "Top-50%", "Top-75%", "All-Active"]  # 8 strategies


def decode_action(action_id):
    """
    Decode action ID (0-39) to human-readable (Power Level, Clustering Strategy)

    Assumption: action = power_idx * 8 + cluster_idx
    - Power: 0-4 (5 levels)
    - Clustering: 0-7 (8 strategies)
    """
    power_idx = action_id // 8
    cluster_idx = action_id % 8

    return POWER_LEVELS[power_idx], CLUSTERING_STRATEGIES[cluster_idx]


def collect_action_history(model, env, n_episodes=100, max_steps=100):
    """
    Run agent for n_episodes and collect action distribution + QoS metrics
    """
    action_history = []
    episode_actions = []  # List of lists (each episode's actions)

    # QoS and Performance metrics
    all_user_rates = []  # Per-user rates (Mbps)
    all_qos_satisfaction = []
    all_energy_efficiency = []
    all_active_aps = []
    qos_violations = []

    print(f"\n📊 Collecting action data for {n_episodes} episodes...")

    for episode in range(n_episodes):
        obs, info = env.reset(seed=42 + episode)
        episode_data = []
        done = False
        steps = 0

        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            action_history.append(int(action))
            episode_data.append(int(action))

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

            # Collect QoS metrics from info
            if 'user_rates_mbps' in info:
                all_user_rates.extend(info['user_rates_mbps'])
            if 'qos_satisfaction' in info:
                all_qos_satisfaction.append(info['qos_satisfaction'])
            if 'energy_efficiency' in info:
                all_energy_efficiency.append(info['energy_efficiency'])
            if 'active_aps' in info:
                all_active_aps.append(info['active_aps'])
            if 'qos_violations' in info:
                qos_violations.append(info['qos_violations'])

        episode_actions.append(episode_data)

        if (episode + 1) % 20 == 0:
            print(f"  Episode {episode+1}/{n_episodes} completed")

    print(f"✅ Collected {len(action_history)} action samples\n")

    performance_metrics = {
        'user_rates': all_user_rates,
        'qos_satisfaction': all_qos_satisfaction,
        'energy_efficiency': all_energy_efficiency,
        'active_aps': all_active_aps,
        'qos_violations': qos_violations
    }

    return action_history, episode_actions, performance_metrics


def analyze_action_distribution(action_history):
    """
    Analyze overall action distribution
    """
    action_counts = Counter(action_history)
    total_actions = len(action_history)

    print("="*80)
    print("ACTION DISTRIBUTION ANALYSIS")
    print("="*80)

    # Sort by frequency
    sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'Rank':<6} {'Action ID':<12} {'Power Level':<15} {'Clustering':<15} {'Count':<10} {'%':<8}")
    print("-"*80)

    for rank, (action_id, count) in enumerate(sorted_actions[:10], 1):
        power, cluster = decode_action(action_id)
        percentage = (count / total_actions) * 100
        print(f"{rank:<6} {action_id:<12} {power:<15} {cluster:<15} {count:<10} {percentage:.2f}%")

    print("="*80)

    # Most common action
    most_common_action = sorted_actions[0][0]
    most_common_count = sorted_actions[0][1]
    most_common_power, most_common_cluster = decode_action(most_common_action)

    print(f"\n🎯 DOMINANT STRATEGY:")
    print(f"   Action ID: {most_common_action}")
    print(f"   Power Level: {most_common_power}")
    print(f"   Clustering: {most_common_cluster}")
    print(f"   Frequency: {most_common_count}/{total_actions} ({most_common_count/total_actions*100:.1f}%)")

    return action_counts, sorted_actions


def analyze_power_and_clustering_separately(action_history):
    """
    Separate analysis: Power level distribution and Clustering strategy distribution
    """
    power_distribution = [0] * 5  # 5 power levels
    cluster_distribution = [0] * 8  # 8 clustering strategies

    for action_id in action_history:
        power_idx = action_id // 8
        cluster_idx = action_id % 8
        power_distribution[power_idx] += 1
        cluster_distribution[cluster_idx] += 1

    total = len(action_history)

    print("\n" + "="*80)
    print("POWER LEVEL DISTRIBUTION")
    print("="*80)
    for idx, (power_name, count) in enumerate(zip(POWER_LEVELS, power_distribution)):
        print(f"  {power_name:<12}: {count:>6} ({count/total*100:>5.1f}%)")

    print("\n" + "="*80)
    print("CLUSTERING STRATEGY DISTRIBUTION")
    print("="*80)
    for idx, (cluster_name, count) in enumerate(zip(CLUSTERING_STRATEGIES, cluster_distribution)):
        print(f"  {cluster_name:<15}: {count:>6} ({count/total*100:>5.1f}%)")

    return power_distribution, cluster_distribution


def analyze_qos_performance(performance_metrics, qos_threshold=5.0):
    """
    Analyze QoS performance: user rates, violations, satisfaction
    """
    print("\n" + "="*80)
    print("QoS PERFORMANCE ANALYSIS")
    print("="*80)

    user_rates = np.array(performance_metrics['user_rates'])
    qos_sat = performance_metrics['qos_satisfaction']
    ee = performance_metrics['energy_efficiency']
    active_aps = performance_metrics['active_aps']

    # User rate statistics
    print(f"\n📊 USER RATE STATISTICS (Mbps):")
    print(f"   Mean Rate: {np.mean(user_rates):.2f} Mbps")
    print(f"   Median Rate: {np.median(user_rates):.2f} Mbps")
    print(f"   Min Rate: {np.min(user_rates):.2f} Mbps")
    print(f"   Max Rate: {np.max(user_rates):.2f} Mbps")
    print(f"   Std Dev: {np.std(user_rates):.2f} Mbps")

    # QoS violations
    num_violations = np.sum(user_rates < qos_threshold)
    total_users = len(user_rates)
    violation_rate = (num_violations / total_users) * 100

    print(f"\n⚠️  QoS VIOLATIONS (< {qos_threshold} Mbps):")
    print(f"   Total Violations: {num_violations}/{total_users} ({violation_rate:.2f}%)")
    print(f"   Satisfaction Rate: {100 - violation_rate:.2f}%")

    # Average satisfaction percentage
    if qos_sat:
        avg_qos_sat = np.mean(qos_sat)
        print(f"   Average QoS Satisfaction: {avg_qos_sat:.2f}%")

    # Energy efficiency
    if ee:
        avg_ee = np.mean(ee)
        print(f"\n⚡ ENERGY EFFICIENCY:")
        print(f"   Mean EE: {avg_ee:.2e} bits/Joule")

    # Active APs
    if active_aps:
        avg_active = np.mean(active_aps)
        print(f"\n📡 ACTIVE ACCESS POINTS:")
        print(f"   Mean Active APs: {avg_active:.1f}")

    print("="*80)

    return {
        'mean_rate': float(np.mean(user_rates)),
        'median_rate': float(np.median(user_rates)),
        'min_rate': float(np.min(user_rates)),
        'max_rate': float(np.max(user_rates)),
        'std_rate': float(np.std(user_rates)),
        'violation_rate': float(violation_rate),
        'avg_qos_satisfaction': float(np.mean(qos_sat)) if qos_sat else 0.0,
        'avg_energy_efficiency': float(np.mean(ee)) if ee else 0.0,
        'avg_active_aps': float(np.mean(active_aps)) if active_aps else 0.0
    }


def plot_action_distribution(action_history, save_dir, scenario_name):
    """
    Plot comprehensive action distribution visualizations
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Overall Action Histogram (0-39)
    fig1, ax = plt.subplots(figsize=(14, 6))

    action_counts = Counter(action_history)
    actions = list(range(40))
    counts = [action_counts.get(a, 0) for a in actions]

    bars = ax.bar(actions, counts, color='teal', alpha=0.8, edgecolor='black', linewidth=1.2)

    ax.set_xlabel('Action ID', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax.set_title(f'Action Distribution: {scenario_name}', fontsize=15, fontweight='bold')
    ax.set_xticks(range(40))
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Highlight top-3 most frequent actions
    sorted_by_count = sorted(enumerate(counts), key=lambda x: x[1], reverse=True)
    for rank, (action_id, count) in enumerate(sorted_by_count[:3]):
        bars[action_id].set_color(['gold', 'silver', '#CD7F32'][rank])
        bars[action_id].set_edgecolor('black')
        bars[action_id].set_linewidth(2.5)

    plt.tight_layout()
    plot1_path = os.path.join(save_dir, 'action_distribution_histogram.png')
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {plot1_path}")
    plt.close()

    # 2. Power Level Distribution (Pie Chart)
    power_distribution, cluster_distribution = analyze_power_and_clustering_separately(action_history)

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Power levels
    colors_power = ['#FF6B6B', '#FFA07A', '#FFD700', '#90EE90', '#4169E1']
    wedges1, texts1, autotexts1 = ax1.pie(
        power_distribution,
        labels=POWER_LEVELS,
        autopct='%1.1f%%',
        colors=colors_power,
        startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    ax1.set_title('Power Level Distribution', fontsize=14, fontweight='bold')

    # Clustering strategies
    colors_cluster = ['#FF69B4', '#9370DB', '#20B2AA', '#FF8C00', '#DDA0DD', '#CD5C5C', '#F0E68C', '#87CEEB']
    wedges2, texts2, autotexts2 = ax2.pie(
        cluster_distribution,
        labels=CLUSTERING_STRATEGIES,
        autopct='%1.1f%%',
        colors=colors_cluster,
        startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    ax2.set_title('Clustering Strategy Distribution', fontsize=14, fontweight='bold')

    plt.suptitle(f'Strategy Breakdown: {scenario_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    plot2_path = os.path.join(save_dir, 'power_and_clustering_distribution.png')
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {plot2_path}")
    plt.close()

    # 3. Heatmap: Power x Clustering (2D grid)
    fig3, ax = plt.subplots(figsize=(12, 8))

    # Create 2D matrix (5 power levels x 8 clustering strategies)
    action_matrix = np.zeros((5, 8))
    for action_id in action_history:
        power_idx = action_id // 8
        cluster_idx = action_id % 8
        action_matrix[power_idx, cluster_idx] += 1

    # Normalize to percentages
    action_matrix_pct = (action_matrix / len(action_history)) * 100

    im = ax.imshow(action_matrix_pct, cmap='YlOrRd', aspect='auto')

    # Set ticks and labels
    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels(CLUSTERING_STRATEGIES, fontsize=11, rotation=45, ha='right')
    ax.set_yticklabels(POWER_LEVELS, fontsize=11)

    ax.set_xlabel('Clustering Strategy', fontsize=13, fontweight='bold')
    ax.set_ylabel('Power Level', fontsize=13, fontweight='bold')
    ax.set_title(f'Action Heatmap (2D): {scenario_name}', fontsize=15, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Frequency (%)', fontsize=12, fontweight='bold')

    # Annotate cells with percentages
    for i in range(5):
        for j in range(8):
            text = ax.text(j, i, f'{action_matrix_pct[i, j]:.1f}%',
                          ha='center', va='center', color='black', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plot3_path = os.path.join(save_dir, 'action_heatmap_2d.png')
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {plot3_path}")
    plt.close()


def plot_qos_analysis(performance_metrics, save_dir, scenario_name, qos_threshold=5.0):
    """
    Plot QoS performance visualizations
    """
    # Flatten user_rates in case it's nested
    user_rates_raw = performance_metrics['user_rates']
    if isinstance(user_rates_raw, list) and len(user_rates_raw) > 0:
        if isinstance(user_rates_raw[0], list):
            # Nested list - flatten it
            user_rates = np.array([item for sublist in user_rates_raw for item in sublist])
        else:
            user_rates = np.array(user_rates_raw)
    else:
        user_rates = np.array(user_rates_raw)

    # 1. User Rate Distribution (Histogram)
    fig1, ax = plt.subplots(figsize=(12, 6))

    n, bins, patches = ax.hist(user_rates, bins=50, color='steelblue', alpha=0.7, edgecolor='black')

    # Color bins below QoS threshold in red
    for i, patch in enumerate(patches):
        if bins[i] < qos_threshold:
            patch.set_facecolor('red')
            patch.set_alpha(0.8)

    # Add QoS threshold line
    ax.axvline(qos_threshold, color='red', linestyle='--', linewidth=2.5, label=f'QoS Threshold ({qos_threshold} Mbps)')

    # Add mean line
    mean_rate = np.mean(user_rates)
    ax.axvline(mean_rate, color='green', linestyle='--', linewidth=2.5, label=f'Mean Rate ({mean_rate:.2f} Mbps)')

    ax.set_xlabel('User Rate (Mbps)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax.set_title(f'User Rate Distribution: {scenario_name}', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plot1_path = os.path.join(save_dir, 'user_rate_distribution.png')
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {plot1_path}")
    plt.close()

    # 2. User Rate CDF (Cumulative Distribution Function)
    fig2, ax = plt.subplots(figsize=(10, 6))

    sorted_rates = np.sort(user_rates)
    cdf = np.arange(1, len(sorted_rates) + 1) / len(sorted_rates) * 100

    ax.plot(sorted_rates, cdf, linewidth=2.5, color='darkblue', label='CDF')
    ax.axvline(qos_threshold, color='red', linestyle='--', linewidth=2.5, label=f'QoS Threshold ({qos_threshold} Mbps)')

    # Find percentage of users above QoS
    pct_above_qos = np.sum(user_rates >= qos_threshold) / len(user_rates) * 100
    ax.axhline(pct_above_qos, color='green', linestyle='--', linewidth=2, label=f'{pct_above_qos:.1f}% Above QoS')

    ax.set_xlabel('User Rate (Mbps)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Probability (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'User Rate CDF: {scenario_name}', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plot2_path = os.path.join(save_dir, 'user_rate_cdf.png')
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {plot2_path}")
    plt.close()

    # 3. Box plot for user rates
    fig3, ax = plt.subplots(figsize=(8, 6))

    bp = ax.boxplot([user_rates], vert=True, patch_artist=True, widths=0.5,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2.5),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))

    ax.axhline(qos_threshold, color='red', linestyle='--', linewidth=2.5, label=f'QoS Threshold ({qos_threshold} Mbps)')

    ax.set_ylabel('User Rate (Mbps)', fontsize=13, fontweight='bold')
    ax.set_title(f'User Rate Statistics: {scenario_name}', fontsize=15, fontweight='bold')
    ax.set_xticklabels(['All Users'])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plot3_path = os.path.join(save_dir, 'user_rate_boxplot.png')
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {plot3_path}")
    plt.close()


def export_results_json(action_counts, power_dist, cluster_dist, qos_stats, scenario_name, save_dir):
    """
    Export analysis results to JSON for LaTeX tables
    """
    # Convert action counts to human-readable format
    action_breakdown = []
    for action_id, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        power, cluster = decode_action(action_id)
        action_breakdown.append({
            'action_id': int(action_id),
            'power_level': power,
            'clustering_strategy': cluster,
            'count': int(count),
            'percentage': float(count / sum(action_counts.values()) * 100)
        })

    results = {
        'scenario': scenario_name,
        'total_actions': int(sum(action_counts.values())),
        'action_breakdown': action_breakdown,
        'power_distribution': {
            POWER_LEVELS[i]: {'count': int(power_dist[i]),
                             'percentage': float(power_dist[i] / sum(power_dist) * 100)}
            for i in range(5)
        },
        'clustering_distribution': {
            CLUSTERING_STRATEGIES[i]: {'count': int(cluster_dist[i]),
                                       'percentage': float(cluster_dist[i] / sum(cluster_dist) * 100)}
            for i in range(8)
        },
        'qos_performance': qos_stats
    }

    json_path = os.path.join(save_dir, 'behavior_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze RL Agent Behavior')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained PPO model (.zip file)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes to analyze (default: 100)')
    parser.add_argument('--save_dir', type=str, default='results/behavior_analysis',
                       help='Directory to save analysis results')

    args = parser.parse_args()

    print("="*80)
    print("RL AGENT BEHAVIOR ANALYSIS (Explainable AI)")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print(f"Episodes: {args.episodes}")
    print("="*80)

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Extract scenario name from config path
    scenario_name = os.path.splitext(os.path.basename(args.config))[0]

    # Create environment
    print("\n🌐 Creating environment...")
    env = CellFreeEnv(config_path=args.config, render_mode=None)
    print(f"   {env.num_aps} APs, {env.num_users} Users")

    # Load model
    print(f"\n🤖 Loading PPO model...")
    model = PPO.load(args.model, env=env)
    print(f"   ✅ Model loaded")

    # Collect action data + performance metrics
    action_history, episode_actions, performance_metrics = collect_action_history(model, env, n_episodes=args.episodes)

    # Analyze action distribution
    action_counts, sorted_actions = analyze_action_distribution(action_history)

    # Separate power and clustering analysis
    power_dist, cluster_dist = analyze_power_and_clustering_separately(action_history)

    # Analyze QoS performance
    qos_stats = analyze_qos_performance(performance_metrics, qos_threshold=5.0)

    # Plot action distribution visualizations
    print("\n📊 Generating action distribution visualizations...")
    plot_action_distribution(action_history, args.save_dir, scenario_name)

    # Plot QoS performance visualizations
    print("\n📊 Generating QoS performance visualizations...")
    plot_qos_analysis(performance_metrics, args.save_dir, scenario_name, qos_threshold=5.0)

    # Export to JSON
    print("\n💾 Exporting results...")
    export_results_json(action_counts, power_dist, cluster_dist, qos_stats, scenario_name, args.save_dir)

    print("\n" + "="*80)
    print("✅ BEHAVIOR ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n📁 Results saved to: {args.save_dir}/")
    print("\n📊 Generated files:")
    print("   ACTION DISTRIBUTION:")
    print("   - action_distribution_histogram.png")
    print("   - power_and_clustering_distribution.png")
    print("   - action_heatmap_2d.png")
    print("\n   QoS PERFORMANCE:")
    print("   - user_rate_distribution.png")
    print("   - user_rate_cdf.png")
    print("   - user_rate_boxplot.png")
    print("\n   DATA:")
    print("   - behavior_analysis.json")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
