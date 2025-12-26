#!/usr/bin/env python3
"""
Run experiments with multiple random seeds for statistical reliability

Usage:
    # Baseline evaluation with 5 seeds
    python src/run_multiple_seeds.py \
        --type baseline \
        --config configs/ppo_scenarios/4_massive_mimo_64ap.yaml \
        --seeds 42 101 2023 55 99 \
        --episodes 50

    # PPO evaluation with 5 seeds
    python src/run_multiple_seeds.py \
        --type ppo \
        --model experiments/scenario4/models/ppo_final.zip \
        --config configs/ppo_scenarios/4_massive_mimo_64ap.yaml \
        --seeds 42 101 2023 55 99 \
        --episodes 50
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import json
import yaml
from pathlib import Path
from scipy import stats
import tensorflow as tf

from environment.cellfree_env import CellFreeEnv
from agents.baselines import evaluate_baseline
from stable_baselines3 import PPO


def run_baseline_with_seed(config_path, seed, episodes=50):
    """Run baseline evaluation with specific seed"""

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set seeds
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Create environment
    env = CellFreeEnv(config_path=config_path, render_mode=None)

    # Set environment seed
    env.reset(seed=seed)

    # Evaluate all baseline strategies
    strategies = config['evaluation']['baseline_strategies']
    results = {}

    print(f"\n  Seed {seed}: Evaluating {len(strategies)} strategies...")

    for strategy in strategies:
        result = evaluate_baseline(env.network, strategy, num_episodes=episodes)
        results[strategy] = result
        print(f"    {strategy}: EE={result['mean_energy_efficiency']:.2e}")

    return results


def run_ppo_with_seed(model_path, config_path, seed, episodes=50):
    """Run PPO evaluation with specific seed"""

    # Set seeds
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Create environment
    env = CellFreeEnv(config_path=config_path, render_mode=None)

    # Load PPO model
    model = PPO.load(model_path, env=env)

    # Reset with seed
    obs, info = env.reset(seed=seed)

    # Evaluation metrics
    all_rewards = []
    all_rates = []
    all_ee = []
    all_qos = []
    all_active_aps = []

    print(f"\n  Seed {seed}: Running {episodes} episodes...")

    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)  # Different seed per episode
        episode_reward = 0
        episode_metrics = {'rate': [], 'ee': [], 'qos': [], 'active_aps': []}

        done = False
        steps = 0
        max_steps = 100

        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

            episode_reward += reward

            if 'avg_rate_mbps' in info:
                episode_metrics['rate'].append(info['avg_rate_mbps'])
            if 'energy_efficiency' in info:
                episode_metrics['ee'].append(info['energy_efficiency'])
            if 'qos_satisfaction' in info:
                episode_metrics['qos'].append(info['qos_satisfaction'])
            if 'active_aps' in info:
                episode_metrics['active_aps'].append(info['active_aps'])

        # Store episode averages
        all_rewards.append(episode_reward)
        if episode_metrics['rate']:
            all_rates.append(np.mean(episode_metrics['rate']))
        if episode_metrics['ee']:
            all_ee.append(np.mean(episode_metrics['ee']))
        if episode_metrics['qos']:
            all_qos.append(np.mean(episode_metrics['qos']))
        if episode_metrics['active_aps']:
            all_active_aps.append(np.mean(episode_metrics['active_aps']))

    return {
        'mean_reward': np.mean(all_rewards),
        'mean_rate_mbps': np.mean(all_rates) if all_rates else 0,
        'mean_energy_efficiency': np.mean(all_ee) if all_ee else 0,
        'mean_qos_satisfaction': np.mean(all_qos) if all_qos else 0,
        'mean_active_aps': np.mean(all_active_aps) if all_active_aps else 0
    }


def aggregate_results(all_seed_results, metric_key):
    """Aggregate results across seeds"""
    values = [r[metric_key] for r in all_seed_results]
    return {
        'mean': np.mean(values),
        'std': np.std(values, ddof=1),  # Sample std
        'values': values
    }


def statistical_test(values1, values2, test_name="Method 1 vs Method 2"):
    """Perform statistical significance test"""

    if len(values1) < 2 or len(values2) < 2:
        print(f"  ⚠️  Not enough samples for statistical test")
        return None, None

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(values1, values2)

    print(f"\n  📊 Statistical Test: {test_name}")
    print(f"    T-statistic: {t_stat:.4f}")
    print(f"    P-value: {p_value:.4e}")

    if p_value < 0.001:
        print(f"    ✓ HIGHLY significant (p < 0.001)")
    elif p_value < 0.01:
        print(f"    ✓ Very significant (p < 0.01)")
    elif p_value < 0.05:
        print(f"    ✓ Significant (p < 0.05)")
    else:
        print(f"    ✗ Not significant (p ≥ 0.05)")

    return t_stat, p_value


def format_latex(mean, std, metric_name):
    """Format results for LaTeX"""

    if mean > 1e6:
        # Scientific notation
        order = int(np.floor(np.log10(mean)))
        mean_scaled = mean / (10 ** order)
        std_scaled = std / (10 ** order)
        return f"$({mean_scaled:.2f} \\pm {std_scaled:.2f}) \\times 10^{{{order}}}$"
    else:
        return f"${mean:.2f} \\pm {std:.2f}$"


def main():
    parser = argparse.ArgumentParser(description='Run experiments with multiple seeds')
    parser.add_argument('--type', type=str, required=True, choices=['baseline', 'ppo'],
                       help='Type of experiment')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--model', type=str,
                       help='Path to PPO model (required for --type ppo)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 101, 2023, 55, 99],
                       help='List of random seeds (default: 42 101 2023 55 99)')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Episodes per seed (default: 50)')
    parser.add_argument('--save_dir', type=str, default='results/multi_seed',
                       help='Directory to save results')

    args = parser.parse_args()

    if args.type == 'ppo' and not args.model:
        parser.error("--model is required when --type is ppo")

    print("="*80)
    print("MULTI-SEED EXPERIMENT FOR STATISTICAL RELIABILITY")
    print("="*80)
    print(f"Type: {args.type.upper()}")
    print(f"Config: {args.config}")
    if args.type == 'ppo':
        print(f"Model: {args.model}")
    print(f"Seeds: {args.seeds}")
    print(f"Episodes per seed: {args.episodes}")
    print("="*80)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Run experiments
    all_results = []

    for seed in args.seeds:
        print(f"\n{'='*80}")
        print(f"Running with seed: {seed}")
        print(f"{'='*80}")

        if args.type == 'baseline':
            result = run_baseline_with_seed(args.config, seed, args.episodes)
        else:  # ppo
            result = run_ppo_with_seed(args.model, args.config, seed, args.episodes)

        all_results.append(result)

    # Aggregate results
    print(f"\n{'='*80}")
    print("AGGREGATING RESULTS")
    print(f"{'='*80}")

    if args.type == 'baseline':
        # Aggregate per strategy
        strategies = all_results[0].keys()
        final_results = {}

        for strategy in strategies:
            print(f"\n  Strategy: {strategy}")
            metrics = ['mean_energy_efficiency', 'mean_rate_mbps', 'mean_qos_satisfaction']

            strategy_results = {}
            for metric in metrics:
                values = [r[strategy][metric] for r in all_results]
                strategy_results[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values, ddof=1),
                    'values': values
                }

                print(f"    {metric}: {strategy_results[metric]['mean']:.2e} ± {strategy_results[metric]['std']:.2e}")
                print(f"      LaTeX: {format_latex(strategy_results[metric]['mean'], strategy_results[metric]['std'], metric)}")

            final_results[strategy] = strategy_results

    else:  # ppo
        print(f"\n  PPO Results:")
        metrics = ['mean_energy_efficiency', 'mean_rate_mbps', 'mean_qos_satisfaction', 'mean_reward', 'mean_active_aps']

        final_results = {}
        for metric in metrics:
            values = [r[metric] for r in all_results]
            final_results[metric] = {
                'mean': np.mean(values),
                'std': np.std(values, ddof=1),
                'values': values
            }

            print(f"    {metric}: {final_results[metric]['mean']:.2e} ± {final_results[metric]['std']:.2e}")
            print(f"      LaTeX: {format_latex(final_results[metric]['mean'], final_results[metric]['std'], metric)}")

    # Save results (convert numpy to python native types)
    config_name = Path(args.config).stem
    save_path = os.path.join(args.save_dir, f'{config_name}_{args.type}_multiseed.json')

    def convert_to_json_serializable(obj):
        """Recursively convert numpy types to Python native types"""
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj

    serializable_results = convert_to_json_serializable(final_results)

    with open(save_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ Results saved to: {save_path}")
    print(f"{'='*80}")

    return final_results


if __name__ == '__main__':
    main()
