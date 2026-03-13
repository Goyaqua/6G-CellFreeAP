"""
Gymnasium Environment for Cell-Free Network Resource Allocation
"""

import gymnasium as gym
import numpy as np
import tensorflow as tf
from gymnasium import spaces
from typing import Optional, Dict, Tuple, Any
import yaml

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from network.cellfree_network import CellFreeNetworkSionna


class CellFreeEnv(gym.Env):
    """
    Gymnasium Environment for Cell-Free Network Resource Allocation
    
    State Space: Channel gains + User QoS requirements
    Action Space: Discrete or Continuous (Power allocation + AP association)
    Reward: Energy efficiency with QoS penalty
    """
    
    metadata = {'render_modes': ['human'], 'render_fps': 1}
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        num_aps: int = 25,
        num_users: int = 10,
        qos_min_rate_mbps: float = 5.0,
        qos_weight: float = 10.0,
        episode_length: int = 100,
        action_type: str = 'discrete',  # 'discrete'
        render_mode: Optional[str] = None,
        randomize_circuit_power: bool = False,  # NEW: Enable circuit power randomization
        circuit_power_range: Optional[Tuple[float, float]] = None  # NEW: Range for randomization
    ):
        """
        Initialize Cell-Free RL Environment

        Args:
            config_path: Path to YAML config file
            num_aps: Number of Access Points
            num_users: Number of users
            qos_min_rate_mbps: Minimum QoS requirement (Mbps)
            qos_weight: Weight for QoS penalty in reward
            episode_length: Number of steps per episode
            action_type: 'discrete' or 'continuous'
            render_mode: Rendering mode
            randomize_circuit_power: If True, randomize circuit power each episode
            circuit_power_range: Tuple (min, max) for circuit power randomization (Watts)
        """
        super().__init__()
        
        self.render_mode = render_mode

        # Circuit power randomization settings
        self.randomize_circuit_power = randomize_circuit_power
        if circuit_power_range is None:
            self.circuit_power_range = (0.1, 0.5)  # Default: 100mW to 500mW
        else:
            self.circuit_power_range = circuit_power_range

        # Load config if provided
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.num_aps = config['network']['num_aps']
            self.num_users = config['network']['num_users']
            self.qos_min_rate = config['environment']['qos_min_rate_mbps'] * 1e6
            self.qos_weight = config['environment']['qos_weight']
            self.episode_length = config['environment']['episode_length']
            self.action_type = config['environment']['action_type']

            # Unified reward weights
            self.rate_weight = config['environment'].get('rate_weight', 1.0)
            self.power_penalty_weight = config['environment'].get('power_penalty_weight', 1.0)
            self.fairness_weight = config['environment'].get('fairness_weight', 0.0)

            # Override with config if available
            if 'randomize_circuit_power' in config['environment']:
                self.randomize_circuit_power = config['environment']['randomize_circuit_power']
            if 'circuit_power_range' in config['environment']:
                self.circuit_power_range = tuple(config['environment']['circuit_power_range'])
        else:
            self.num_aps = num_aps
            self.num_users = num_users
            self.qos_min_rate = qos_min_rate_mbps * 1e6  # Convert to bps
            self.qos_weight = qos_weight
            self.episode_length = episode_length
            self.action_type = action_type

            # Unified reward weights (defaults)
            self.rate_weight = 1.0
            self.power_penalty_weight = 1.0
            self.fairness_weight = 0.0
        
        # Create network
        # Load circuit power from config if available
        if config_path and 'circuit_power_per_ap' in config['network']:
            circuit_power = config['network']['circuit_power_per_ap']
        else:
            circuit_power = 0.2  # Default 200mW

        self.network = CellFreeNetworkSionna(
            num_aps=self.num_aps,
            num_users=self.num_users,
            num_antennas_per_ap=1,
            area_size=500.0,
            circuit_power_per_ap=circuit_power,
            seed=42
        )
        
        # QoS requirements for each user
        self.qos_requirements = np.ones(self.num_users) * self.qos_min_rate
        
        # Episode tracking
        self.current_step = 0
        self.current_channel = None
        
        # Define observation space
        # State: Flattened channel gains (num_aps x num_users) + QoS requirements (num_users) + circuit_power (1)
        obs_dim = self.num_aps * self.num_users + self.num_users + 1  # +1 for circuit power
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Define action space
        if self.action_type == 'discrete':
            # IMPROVED: Factored discrete action space
            # 5 power strategies × 8 AP selection strategies = 40 total actions
            self.num_power_levels = 5
            self.num_ap_strategies = 8
            self.action_space = spaces.Discrete(self.num_power_levels * self.num_ap_strategies)
        else:
            # Continuous: [power_ap1, ..., power_apN, assoc_threshold]
            # Power normalized to [0, 1]
            self.action_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.num_aps + 1,),
                dtype=np.float32
            )
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)

        # Reset episode tracking
        self.current_step = 0

        # Randomize circuit power if enabled
        if self.randomize_circuit_power:
            new_circuit_power = np.random.uniform(
                self.circuit_power_range[0],
                self.circuit_power_range[1]
            )
            self.network.circuit_power_per_ap = new_circuit_power

        # Generate new channel realization
        self.current_channel = self.network.generate_channel_matrix(batch_size=1)

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info
    
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        # Convert action to power allocation and AP association
        power_allocation, ap_association = self._action_to_allocation(action)
        
        # Calculate SINR and rates
        sinr, rates = self.network.calculate_sinr_and_rate(
            self.current_channel,
            power_allocation,
            ap_association
        )
        
        # Calculate reward (with ap_association for circuit power)
        reward = self._calculate_reward(rates, power_allocation, ap_association)
        
        # Update step counter
        self.current_step += 1
        
        # Check if episode is done
        terminated = False
        truncated = self.current_step >= self.episode_length
        
        # Generate new channel for next step (time-varying channel)
        self.current_channel = self.network.generate_channel_matrix(batch_size=1)
        
        # Get new observation
        obs = self._get_observation()
        info = self._get_info()
        
        # Add performance metrics to info
        active_aps_count = np.sum(np.sum(ap_association, axis=1) > 0)

        info.update({
            'sinr_db': 10 * np.log10(max(tf.reduce_mean(sinr).numpy(), 1e-12)),
            'avg_rate_mbps': tf.reduce_mean(rates).numpy() / 1e6,
            'user_rates_mbps': (rates.numpy() / 1e6).tolist(),  # Per-user rates in Mbps
            'energy_efficiency': self.network.calculate_energy_efficiency(
                rates, power_allocation, ap_association
            ).numpy()[0],
            'qos_satisfaction': self.network.calculate_qos_satisfaction(
                rates, self.qos_requirements
            ).numpy()[0],
            'qos_violations': int(np.sum(rates.numpy() < self.qos_min_rate)),  # Number of users below QoS
            'active_aps': active_aps_count
        })
        
        return obs, float(reward), terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation

        Observation: [normalized_channel_gains, normalized_qos_requirements, normalized_circuit_power]
        """
        # Get channel gains per AP
        channel_gain_per_ap = self.network.get_channel_gain_per_ap(self.current_channel)

        # Flatten (num_aps, num_users)
        channel_gain_flat = channel_gain_per_ap.flatten()

        # FIXED: Use theoretical absolute bounds for channel gain normalization (dB scale)
        # Prevents loss of absolute channel quality information across time steps
        channel_gain_db = 20 * np.log10(channel_gain_flat + 1e-10)
        MIN_GAIN_DB = -150.0  # Max distance (~500m) + heavy shadowing + deep fade
        MAX_GAIN_DB = -20.0   # Min distance (~1m) + favorable fading
        
        channel_gain_normalized = (channel_gain_db - MIN_GAIN_DB) / (MAX_GAIN_DB - MIN_GAIN_DB)
        channel_gain_normalized = np.clip(channel_gain_normalized, 0.0, 1.0)

        # Normalize QoS requirements
        qos_normalized = self.qos_requirements / (self.qos_min_rate * 2)

        # Normalize circuit power to [0, 1]
        # Range: 0.1W to 3.0W -> normalized to [0, 1]
        circuit_power_min = 0.1
        circuit_power_max = 3.0
        circuit_power_normalized = np.array([
            (self.network.circuit_power_per_ap - circuit_power_min) /
            (circuit_power_max - circuit_power_min)
        ], dtype=np.float32)

        # Concatenate
        obs = np.concatenate([
            channel_gain_normalized,
            qos_normalized,
            circuit_power_normalized
        ]).astype(np.float32)

        return obs
    
    def _action_to_allocation(
        self,
        action: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert action to power allocation and AP association
        
        Returns:
            power_allocation: (num_aps,) power in Watts
            ap_association: (num_aps, num_users) binary matrix
        """
        if self.action_type == 'discrete':
            # IMPROVED: Decode factored action
            # action = power_idx * num_ap_strategies + ap_strategy_idx
            power_idx = action // self.num_ap_strategies
            ap_strategy_idx = action % self.num_ap_strategies

            # Apply power strategy
            power_allocation = self._apply_power_strategy(power_idx)

            # Apply AP selection strategy
            ap_association = self._apply_ap_strategy(ap_strategy_idx)
            
        else:  # continuous
            # First num_aps values are power allocation
            power_allocation = action[:self.num_aps] * self.network.max_power_per_ap
            
            # Last value is association threshold
            threshold = action[-1]
            
            # Get channel gains for association
            channel_gain_per_ap = self.network.get_channel_gain_per_ap(self.current_channel)
            
            # Threshold-based association
            ap_association = np.zeros((self.num_aps, self.num_users))
            for user_idx in range(self.num_users):
                # Normalize gains for this user
                gains = channel_gain_per_ap[:, user_idx]
                
                # FIXED: Use theoretical absolute bounds for consistency with state observation
                gains_db = 20 * np.log10(gains + 1e-10)
                MIN_GAIN_DB = -150.0
                MAX_GAIN_DB = -20.0
                gains_norm = (gains_db - MIN_GAIN_DB) / (MAX_GAIN_DB - MIN_GAIN_DB)
                gains_norm = np.clip(gains_norm, 0.0, 1.0)
                
                # Associate with APs above threshold
                serving_aps = gains_norm >= threshold
                if not serving_aps.any():
                    # If no AP above threshold, use nearest
                    serving_aps[np.argmax(gains)] = True
                
                ap_association[serving_aps, user_idx] = 1
        
        return power_allocation, ap_association
    
    def _apply_power_strategy(self, power_idx: int) -> np.ndarray:
        """
        Apply power allocation strategy based on index

        Strategies:
        0: Very Low (20%)
        1: Low (40%)
        2: Medium (60%)
        3: High (80%)
        4: Maximum (100%)
        """
        power_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
        power_level = power_levels[power_idx]

        # All APs use same power level (could be extended to per-AP in future)
        power_allocation = np.ones(self.num_aps) * power_level * self.network.max_power_per_ap

        return power_allocation

    def _apply_ap_strategy(self, strategy_idx: int) -> np.ndarray:
        """
        Apply AP selection strategy based on index

        Strategies:
        0: Nearest-only (1 AP per user, most energy efficient)
        1: Top-2 nearest
        2: Top-3 nearest
        3: Top-5 nearest
        4: Top-25% APs
        5: Top-50% APs
        6: Top-75% APs
        7: All APs (maximum cooperation, highest power)
        """
        # Get channel gains
        channel_gain_per_ap = self.network.get_channel_gain_per_ap(self.current_channel)

        ap_association = np.zeros((self.num_aps, self.num_users))

        if strategy_idx == 0:  # Ultra-Green (Minimum possible APs)
            # Find the absolute strongest AP for each user
            best_aps_per_user = np.argmax(channel_gain_per_ap, axis=0)
            
            # UNIQUE APs only! If 3 users have the same best AP, we only turn on 1 AP!
            unique_best_aps = np.unique(best_aps_per_user)
            
            for user_idx in range(self.num_users):
                ap_association[best_aps_per_user[user_idx], user_idx] = 1

        elif strategy_idx == 1:  # Top-2
            num_serving = min(2, self.num_aps)
            for user_idx in range(self.num_users):
                top_aps = np.argsort(channel_gain_per_ap[:, user_idx])[-num_serving:]
                ap_association[top_aps, user_idx] = 1

        elif strategy_idx == 2:  # Top-3
            num_serving = min(3, self.num_aps)
            for user_idx in range(self.num_users):
                top_aps = np.argsort(channel_gain_per_ap[:, user_idx])[-num_serving:]
                ap_association[top_aps, user_idx] = 1

        elif strategy_idx == 3:  # Top-5
            num_serving = min(5, self.num_aps)
            for user_idx in range(self.num_users):
                top_aps = np.argsort(channel_gain_per_ap[:, user_idx])[-num_serving:]
                ap_association[top_aps, user_idx] = 1

        elif strategy_idx == 4:  # Top-25%
            num_serving = max(1, self.num_aps // 4)
            for user_idx in range(self.num_users):
                top_aps = np.argsort(channel_gain_per_ap[:, user_idx])[-num_serving:]
                ap_association[top_aps, user_idx] = 1

        elif strategy_idx == 5:  # Top-50%
            num_serving = max(1, self.num_aps // 2)
            for user_idx in range(self.num_users):
                top_aps = np.argsort(channel_gain_per_ap[:, user_idx])[-num_serving:]
                ap_association[top_aps, user_idx] = 1

        elif strategy_idx == 6:  # Top-75%
            num_serving = max(1, (3 * self.num_aps) // 4)
            for user_idx in range(self.num_users):
                top_aps = np.argsort(channel_gain_per_ap[:, user_idx])[-num_serving:]
                ap_association[top_aps, user_idx] = 1

        else:  # strategy_idx == 7: All APs
            ap_association = np.ones((self.num_aps, self.num_users))

        return ap_association

    def _nearest_ap_association(self) -> np.ndarray:
        """Nearest AP association strategy (legacy, kept for compatibility)"""
        return self._apply_ap_strategy(0)
    
    def _calculate_reward(
        self,
        rates: tf.Tensor,
        power_allocation: np.ndarray,
        ap_association: np.ndarray
    ) -> float:
        """
        Calculate unified reward function for all scenarios

        Reward = α×log_sum_rate - β×normalized_power - γ×qos_penalty - δ×fairness_penalty

        Components:
        - log_sum_rate: Proportional fairness (magnitude ~20-50)
        - normalized_power: Power cost in [0, 1] range
        - qos_penalty: QoS violations penalty
        - fairness_penalty: Jain's Index based fairness (0 = fair, 1 = unfair)

        Weights (α, β, γ, δ) are scenario-specific from YAML config
        """
        # Extract rates as numpy array
        rates_np = rates.numpy()[0]  # Shape: (num_users,)
        user_rates_mbps = rates_np / 1e6  # Convert bps to Mbps

        # 1. Log Sum-Rate (Proportional Fairness)
        # log1p(x) = log(1 + x) → avoids log(0) issues
        log_sum_rate = np.sum(np.log1p(user_rates_mbps))

        # 2. QoS Penalty
        qos_violations = np.maximum(0, self.qos_min_rate / 1e6 - user_rates_mbps)
        qos_penalty = self.qos_weight * np.sum(qos_violations)

        # 3. Normalized Power Cost
        active_aps_mask = np.sum(ap_association, axis=1) > 0
        total_tx_power = np.sum(power_allocation[active_aps_mask])
        active_aps = np.sum(active_aps_mask)
        total_circuit_power = active_aps * self.network.circuit_power_per_ap
        total_power = total_tx_power + total_circuit_power

        max_possible_power = self.num_aps * (
            self.network.max_power_per_ap + self.network.circuit_power_per_ap
        )
        normalized_power = total_power / max_possible_power

        # 4. Fairness Penalty (Jain's Index)
        sum_rates = np.sum(user_rates_mbps)
        sum_rates_squared = np.sum(user_rates_mbps ** 2)

        if sum_rates_squared > 1e-9:  # Avoid division by zero
            jains_index = (sum_rates ** 2) / (self.num_users * sum_rates_squared)
        else:
            jains_index = 0.0

        fairness_penalty = self.fairness_weight * (1.0 - jains_index)

        # Final Reward (Configurable Weights)
        reward = (
            self.rate_weight * log_sum_rate
            - self.power_penalty_weight * normalized_power
            - qos_penalty
            - fairness_penalty
        )

        return reward
    
    def _get_info(self) -> Dict:
        """Get additional information"""
        return {
            'step': self.current_step,
            'num_aps': self.num_aps,
            'num_users': self.num_users
        }
    
    def render(self):
        """Render the environment"""
        if self.render_mode == 'human':
            self.network.visualize_network()
    
    def close(self):
        """Clean up resources"""
        pass