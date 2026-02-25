"""
Analyze trained RL models on STAR topology (cross pattern) user distribution
Generate heatmaps showing AP activation patterns to visualize if the agent
learned to turn off APs far from traffic patterns.

This script:
1. Loads trained PPO models
2. Tests them on STAR topology (4-avenue cross pattern) with multiple random seeds
3. Generates heatmaps showing:
   - AP activation frequency
   - Average power allocation per AP
   - Spatial correlation with user traffic patterns
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless operation
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from environment.cellfree_env import CellFreeEnv
from stable_baselines3 import PPO


class StarTopologyAnalyzer:
    """Analyzer for testing trained models on star topology using CellFreeEnv"""

    def __init__(
        self,
        model_path: str,
        num_aps: int = 64,
        num_users: int = 16,
        area_size: float = 500.0,
        num_episodes: int = 10,
        episode_length: int = 100
    ):
        """
        Initialize analyzer

        Args:
            model_path: Path to trained PPO model
            num_aps: Number of APs (must match training)
            num_users: Number of users
            area_size: Coverage area size
            num_episodes: Number of test episodes with different random seeds
            episode_length: Steps per episode
        """
        self.model_path = model_path
        self.num_aps = num_aps
        self.num_users = num_users
        self.area_size = area_size
        self.num_episodes = num_episodes
        self.episode_length = episode_length

        # Load model
        print(f"Loading model from: {model_path}")
        self.model = PPO.load(model_path)

        # Create environment matching training config
        self.env = CellFreeEnv(
            num_aps=num_aps,
            num_users=num_users,
            qos_min_rate_mbps=5.0,
            qos_weight=5.0,
            episode_length=episode_length,
            action_type='discrete'
        )

        # Switch network to star topology
        self.env.network.reset_topology(topology_mode='star')

        # Storage for analysis
        self.ap_activation_counts = np.zeros(num_aps)
        self.ap_power_sum = np.zeros(num_aps)
        self.total_steps = 0

    def run_analysis(self) -> Dict:
        """
        Run analysis over multiple episodes with different random user positions
        All episodes use STAR topology with different random seeds

        Returns:
            Dictionary with analysis results
        """
        print(f"\nRunning analysis on STAR topology...")
        print(f"Episodes: {self.num_episodes}")
        print(f"Steps per episode: {self.episode_length}")
        print("=" * 80)

        episode_rewards = []
        episode_rates = []
        episode_ee = []

        for ep in range(self.num_episodes):
            print(f"\n[Episode {ep+1}/{self.num_episodes}]")

            # Reset with unique seed and re-apply star topology
            seed = 42 + ep
            self.env.network.reset_topology(topology_mode='star')

            print(f"  Seed: {seed}")
            print(f"  User positions (first 3): {self.env.network.user_positions[:3]}")

            # Run episode
            ep_reward, ep_rate, ep_ee = self._run_episode(seed)

            episode_rewards.append(ep_reward)
            episode_rates.append(ep_rate)
            episode_ee.append(ep_ee)

            print(f"  Total Reward: {ep_reward:.2f}")
            print(f"  Avg Rate: {ep_rate/1e6:.2f} Mbps")
            print(f"  Avg EE: {ep_ee:.2f} bits/J")

        # Calculate statistics
        results = {
            'model_path': self.model_path,
            'num_episodes': self.num_episodes,
            'total_steps': self.total_steps,
            'ap_activation_frequency': self.ap_activation_counts / max(self.total_steps, 1),
            'ap_average_power': self.ap_power_sum / max(self.total_steps, 1),
            'episode_rewards': episode_rewards,
            'episode_rates': episode_rates,
            'episode_ee': episode_ee,
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_rate': float(np.mean(episode_rates)),
            'mean_ee': float(np.mean(episode_ee))
        }

        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"Mean Episode Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Mean Data Rate: {results['mean_rate']/1e6:.2f} Mbps")
        print(f"Mean Energy Efficiency: {results['mean_ee']:.2f} bits/J")
        print(f"Active APs (avg): {np.sum(results['ap_activation_frequency'] > 0.1)}/{self.num_aps}")

        return results

    def _run_episode(self, seed: int) -> Tuple[float, float, float]:
        """
        Run single episode using CellFreeEnv and collect AP usage statistics

        Returns:
            Tuple of (total_reward, avg_rate, avg_ee)
        """
        obs, info = self.env.reset(seed=seed)

        episode_reward = 0.0
        episode_rate = 0.0
        episode_ee = 0.0
        steps = 0

        done = False
        while not done:
            # Get action from trained model
            action, _ = self.model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Extract AP activation from info
            if 'active_aps' in info:
                active_count = info['active_aps']

            # Track power allocation from the environment's last action
            # We need to decode the action to get per-AP power
            power_allocation, ap_association = self.env._action_to_allocation(action)

            active_aps = (power_allocation > 0.001).astype(float)
            self.ap_activation_counts += active_aps
            self.ap_power_sum += power_allocation
            self.total_steps += 1

            # Accumulate metrics
            episode_reward += reward
            if 'avg_rate_mbps' in info:
                episode_rate += info['avg_rate_mbps'] * 1e6
            if 'energy_efficiency' in info:
                episode_ee += info['energy_efficiency']

            steps += 1

        # Average over episode
        if steps > 0:
            episode_rate /= steps
            episode_ee /= steps

        return episode_reward, episode_rate, episode_ee

    def visualize_heatmaps(self, results: Dict, save_dir: str = "."):
        """
        Generate comprehensive heatmap visualizations

        Args:
            results: Results dictionary from run_analysis()
            save_dir: Directory to save figures
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)

        # Get grid dimensions for APs
        side_length = int(np.ceil(np.sqrt(self.num_aps)))

        # Reshape AP statistics to 2D grid
        activation_freq = results['ap_activation_frequency']
        avg_power = results['ap_average_power']

        # Pad to square grid
        padded_activation = np.zeros(side_length * side_length)
        padded_power = np.zeros(side_length * side_length)
        padded_activation[:self.num_aps] = activation_freq
        padded_power[:self.num_aps] = avg_power

        activation_grid = padded_activation.reshape(side_length, side_length)
        power_grid = padded_power.reshape(side_length, side_length)

        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 12))

        # 1. AP Activation Frequency Heatmap
        ax1 = plt.subplot(2, 3, 1)
        sns.heatmap(
            activation_grid,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            vmin=0,
            vmax=1,
            square=True,
            cbar_kws={'label': 'Activation Frequency'},
            ax=ax1
        )
        ax1.set_title('AP Activation Frequency\n(Star Topology)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Grid X')
        ax1.set_ylabel('Grid Y')

        # 2. Average Power Allocation Heatmap
        ax2 = plt.subplot(2, 3, 2)
        sns.heatmap(
            power_grid * 1000,  # Convert to mW
            annot=True,
            fmt='.1f',
            cmap='plasma',
            square=True,
            cbar_kws={'label': 'Power (mW)'},
            ax=ax2
        )
        ax2.set_title('Average Power Allocation\n(Star Topology)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Grid X')
        ax2.set_ylabel('Grid Y')

        # 3. Binary Active/Inactive (threshold at 10%)
        ax3 = plt.subplot(2, 3, 3)
        active_grid = (activation_grid > 0.1).astype(int)
        sns.heatmap(
            active_grid,
            annot=True,
            fmt='d',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            square=True,
            cbar_kws={'label': 'Active (1) / Inactive (0)'},
            ax=ax3
        )
        ax3.set_title('AP Active Status\n(Threshold: 10%)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Grid X')
        ax3.set_ylabel('Grid Y')

        # 4. Network Topology Visualization with AP activation overlay
        ax4 = plt.subplot(2, 3, 4)

        # Draw star pattern (cross)
        center_x = self.area_size / 2
        center_y = self.area_size / 2
        avenue_angles = np.array([45, 135, 225, 315])
        max_radius = 240.0

        for angle_deg in avenue_angles:
            angle_rad = np.deg2rad(angle_deg)
            x_end = center_x + max_radius * np.cos(angle_rad)
            y_end = center_y + max_radius * np.sin(angle_rad)
            ax4.plot(
                [center_x, x_end],
                [center_y, y_end],
                'gray',
                alpha=0.3,
                linestyle='--',
                linewidth=2
            )

        # Plot APs colored by activation frequency
        ap_positions = self.env.network.ap_positions
        scatter = ax4.scatter(
            ap_positions[:, 0],
            ap_positions[:, 1],
            c=activation_freq,
            cmap='YlOrRd',
            s=300,
            marker='^',
            edgecolors='black',
            linewidths=2,
            vmin=0,
            vmax=1,
            zorder=5
        )
        plt.colorbar(scatter, ax=ax4, label='Activation Frequency')

        # Mark center
        ax4.scatter([center_x], [center_y], c='gold', marker='X', s=500,
                   edgecolors='black', linewidths=2, zorder=6)

        ax4.set_xlim(0, self.area_size)
        ax4.set_ylim(0, self.area_size)
        ax4.set_xlabel('X (meters)', fontsize=12)
        ax4.set_ylabel('Y (meters)', fontsize=12)
        ax4.set_title('Spatial AP Activation Pattern', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal')

        # 5. Performance Metrics
        ax5 = plt.subplot(2, 3, 5)
        metrics = ['Reward', 'Rate\n(Mbps)', 'Energy Eff.\n(Mbit/J)']
        values = [
            results['mean_reward'],
            results['mean_rate'] / 1e6,
            results['mean_ee'] / 1e6
        ]
        bars = ax5.bar(metrics, values, color=['skyblue', 'lightgreen', 'coral'])
        ax5.set_title('Average Performance Metrics', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Value')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}',
                    ha='center', va='bottom', fontweight='bold')

        # 6. AP Activation Distribution
        ax6 = plt.subplot(2, 3, 6)
        ax6.hist(activation_freq, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax6.axvline(0.1, color='red', linestyle='--', linewidth=2, label='Active Threshold (10%)')
        ax6.set_xlabel('Activation Frequency', fontsize=12)
        ax6.set_ylabel('Number of APs', fontsize=12)
        ax6.set_title('Distribution of AP Activation', fontsize=14, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # Add overall statistics text
        active_aps = np.sum(activation_freq > 0.1)
        avg_active = active_aps / self.num_aps * 100

        fig.text(0.5, 0.02,
                f'Model: {Path(self.model_path).name} | Episodes: {self.num_episodes} | '
                f'Active APs: {active_aps}/{self.num_aps} ({avg_active:.1f}%) | '
                f'Topology: STAR (4-Avenue Cross)',
                ha='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        # Save figure
        model_name = Path(self.model_path).stem
        save_file = save_path / f'{model_name}_star_topology_heatmap.png'
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"\nSaved heatmap: {save_file}")

        plt.close()

    def save_results(self, results: Dict, save_dir: str = "."):
        """Save analysis results to JSON"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)

        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {
            'model_path': results['model_path'],
            'num_episodes': results['num_episodes'],
            'total_steps': results['total_steps'],
            'ap_activation_frequency': results['ap_activation_frequency'].tolist(),
            'ap_average_power': results['ap_average_power'].tolist(),
            'episode_rewards': [float(x) for x in results['episode_rewards']],
            'episode_rates': [float(x) for x in results['episode_rates']],
            'episode_ee': [float(x) for x in results['episode_ee']],
            'mean_reward': results['mean_reward'],
            'std_reward': results['std_reward'],
            'mean_rate': results['mean_rate'],
            'mean_ee': results['mean_ee']
        }

        model_name = Path(self.model_path).stem
        save_file = save_path / f'{model_name}_star_topology_analysis.json'

        with open(save_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"Saved results: {save_file}")


def analyze_multiple_models(model_paths: List[str], output_dir: str = "star_topology_analysis",
                           num_aps: int = 64, num_users: int = 16,
                           num_episodes: int = 10, episode_length: int = 100):
    """
    Analyze multiple trained models on star topology

    Args:
        model_paths: List of paths to trained models
        output_dir: Directory to save all results
        num_aps: Number of APs
        num_users: Number of users
        num_episodes: Number of test episodes
        episode_length: Steps per episode
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("MULTI-MODEL STAR TOPOLOGY ANALYSIS")
    print("=" * 80)
    print(f"Number of models: {len(model_paths)}")
    print(f"Output directory: {output_dir}")

    all_results = []

    for i, model_path in enumerate(model_paths, 1):
        print(f"\n{'='*80}")
        print(f"Analyzing Model {i}/{len(model_paths)}: {model_path}")
        print(f"{'='*80}")

        try:
            analyzer = StarTopologyAnalyzer(
                model_path=model_path,
                num_aps=num_aps,
                num_users=num_users,
                area_size=500.0,
                num_episodes=num_episodes,
                episode_length=episode_length
            )

            results = analyzer.run_analysis()
            analyzer.visualize_heatmaps(results, save_dir=output_dir)
            analyzer.save_results(results, save_dir=output_dir)

            all_results.append(results)

        except Exception as e:
            print(f"ERROR analyzing {model_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate comparison summary
    if len(all_results) > 1:
        generate_comparison_summary(all_results, output_dir)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")


def generate_comparison_summary(all_results: List[Dict], output_dir: str):
    """Generate comparison figure for multiple models"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    model_names = [Path(r['model_path']).stem for r in all_results]

    # 1. Mean Rewards
    ax = axes[0, 0]
    rewards = [r['mean_reward'] for r in all_results]
    stds = [r['std_reward'] for r in all_results]
    ax.bar(range(len(model_names)), rewards, yerr=stds, capsize=5, color='skyblue', edgecolor='black')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Model Comparison: Rewards', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 2. Active APs
    ax = axes[0, 1]
    active_aps = [np.sum(np.array(r['ap_activation_frequency']) > 0.1) for r in all_results]
    ax.bar(range(len(model_names)), active_aps, color='lightgreen', edgecolor='black')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel('Number of Active APs')
    ax.set_title('Model Comparison: Active APs', fontweight='bold')
    ax.axhline(64, color='red', linestyle='--', label='Total APs')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Mean Rate
    ax = axes[1, 0]
    rates = [r['mean_rate'] / 1e6 for r in all_results]
    ax.bar(range(len(model_names)), rates, color='coral', edgecolor='black')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel('Mean Rate (Mbps)')
    ax.set_title('Model Comparison: Data Rate', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 4. Energy Efficiency
    ax = axes[1, 1]
    ee = [r['mean_ee'] / 1e6 for r in all_results]
    ax.bar(range(len(model_names)), ee, color='plum', edgecolor='black')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel('Energy Efficiency (Mbit/J)')
    ax.set_title('Model Comparison: Energy Efficiency', fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_file = Path(output_dir) / 'model_comparison_star_topology.png'
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison figure: {save_file}")
    plt.close()


if __name__ == '__main__':
    # Example usage
    print("Star Topology Analysis Tool")
    print("=" * 80)

    # Example: Analyze a single model
    model_path = "path/to/your/trained_model.zip"

    # Check if model exists
    if not Path(model_path).exists():
        print(f"\nModel not found: {model_path}")
        print("\nPlease provide the path to your trained PPO model.")
        print("\nExample usage:")
        print("  python analyze_star_topology_heatmap.py")
        print("\nThen edit the model_path variable in the script.")
        sys.exit(1)

    # Run analysis
    analyzer = StarTopologyAnalyzer(
        model_path=model_path,
        num_aps=64,
        num_users=16,
        area_size=500.0,
        num_episodes=10,
        episode_length=100
    )

    results = analyzer.run_analysis()
    analyzer.visualize_heatmaps(results)
    analyzer.save_results(results)
