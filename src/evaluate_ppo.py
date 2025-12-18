"""
Evaluate PPO Agent
Quick evaluation script for PPO models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
from stable_baselines3 import PPO
from environment.cellfree_env import CellFreeEnv

def evaluate_ppo(model_path, n_episodes=50, circuit_power=0.2):
    """
    Evaluate PPO agent

    Args:
        model_path: Path to trained PPO model (.zip file)
        n_episodes: Number of evaluation episodes
        circuit_power: Circuit power per AP (Watts)
    """

    print("="*80)
    print("PPO AGENT EVALUATION")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Circuit Power: {circuit_power*1000:.0f} mW")
    print("="*80)

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

    # Load PPO model
    model = PPO.load(model_path, env=env)

    # Evaluation metrics
    all_rewards = []
    all_rates = []
    all_ee = []
    all_qos = []
    all_active_aps = []

    print(f"\nRunning {n_episodes} episodes...")

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_metrics = {
            'rate': [],
            'ee': [],
            'qos': [],
            'active_aps': []
        }

        done = False
        steps = 0
        max_steps = 100

        while not done and steps < max_steps:
            try:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1

                episode_reward += reward

                # Collect metrics
                if 'avg_rate_mbps' in info:
                    episode_metrics['rate'].append(info['avg_rate_mbps'])
                if 'energy_efficiency' in info:
                    episode_metrics['ee'].append(info['energy_efficiency'])
                if 'qos_satisfaction' in info:
                    episode_metrics['qos'].append(info['qos_satisfaction'])
                if 'active_aps' in info:
                    episode_metrics['active_aps'].append(info['active_aps'])

            except Exception as e:
                print(f"⚠️ Error in episode {episode+1}: {e}")
                break

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

        # Progress
        if (episode + 1) % 10 == 0:
            print(f"  Progress: {episode+1}/{n_episodes} episodes completed")

    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    if all_rewards:
        print(f"\n📊 Performance Metrics:")
        print(f"  • Mean Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
        print(f"  • Min Reward: {np.min(all_rewards):.2f}")
        print(f"  • Max Reward: {np.max(all_rewards):.2f}")

    if all_rates:
        print(f"\n📡 Data Rate:")
        print(f"  • Mean Rate: {np.mean(all_rates):.2f} ± {np.std(all_rates):.2f} Mbps")
        print(f"  • Min Rate: {np.min(all_rates):.2f} Mbps")
        print(f"  • Max Rate: {np.max(all_rates):.2f} Mbps")

    if all_ee:
        print(f"\n⚡ Energy Efficiency:")
        print(f"  • Mean EE: {np.mean(all_ee):.2e} ± {np.std(all_ee):.2e} bits/J")
        print(f"  • Min EE: {np.min(all_ee):.2e} bits/J")
        print(f"  • Max EE: {np.max(all_ee):.2e} bits/J")

    if all_qos:
        print(f"\n✅ QoS Satisfaction:")
        print(f"  • Mean QoS: {np.mean(all_qos):.1f} ± {np.std(all_qos):.1f}%")
        print(f"  • Min QoS: {np.min(all_qos):.1f}%")
        print(f"  • Max QoS: {np.max(all_qos):.1f}%")

    if all_active_aps:
        print(f"\n🔌 Active APs:")
        print(f"  • Mean Active APs: {np.mean(all_active_aps):.2f} ± {np.std(all_active_aps):.2f}")
        print(f"  • Min Active APs: {np.min(all_active_aps):.1f}")
        print(f"  • Max Active APs: {np.max(all_active_aps):.1f}")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

    return {
        'mean_reward': np.mean(all_rewards) if all_rewards else 0,
        'mean_rate': np.mean(all_rates) if all_rates else 0,
        'mean_ee': np.mean(all_ee) if all_ee else 0,
        'mean_qos': np.mean(all_qos) if all_qos else 0,
        'mean_active_aps': np.mean(all_active_aps) if all_active_aps else 0
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate PPO Agent')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained PPO model (.zip)')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of evaluation episodes (default: 50)')
    parser.add_argument('--circuit-power', type=float, default=0.2,
                       help='Circuit power per AP in Watts (default: 0.2)')

    args = parser.parse_args()

    evaluate_ppo(
        model_path=args.model,
        n_episodes=args.episodes,
        circuit_power=args.circuit_power
    )
