"""
Baseline Strategies for Comparison
"""

import numpy as np
import tensorflow as tf
from typing import Tuple


class BaselineStrategies:
    """
    Collection of baseline resource allocation strategies
    """
    
    @staticmethod
    def nearest_ap_max_power(
        network,
        channel_matrix: tf.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Nearest AP with Maximum Power
        - Each user associates with nearest AP (highest channel gain)
        - All APs transmit at maximum power
        
        Args:
            network: CellFreeNetworkSionna instance
            channel_matrix: Channel matrix (1, num_users, num_tx)
            
        Returns:
            power_allocation: (num_aps,) power in Watts
            ap_association: (num_aps, num_users) binary matrix
        """
        # Maximum power for all APs
        power_allocation = np.ones(network.num_aps) * network.max_power_per_ap
        
        # Average channel gain per AP
        channel_gain_per_ap = network.get_channel_gain_per_ap(channel_matrix)
        
        # Nearest AP association
        ap_association = np.zeros((network.num_aps, network.num_users))
        for user_idx in range(network.num_users):
            nearest_ap = np.argmax(channel_gain_per_ap[:, user_idx])
            ap_association[nearest_ap, user_idx] = 1
        
        return power_allocation, ap_association
    
    @staticmethod
    def random_allocation(
        network,
        channel_matrix: tf.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Random Power and Association
        - Random power allocation
        - Random AP-user association
        
        Args:
            network: CellFreeNetworkSionna instance
            channel_matrix: Channel matrix (1, num_users, num_tx)
            
        Returns:
            power_allocation: (num_aps,) power in Watts
            ap_association: (num_aps, num_users) binary matrix
        """
        # Random power allocation (20% to 100% of max power)
        power_allocation = np.random.uniform(
            0.2 * network.max_power_per_ap,
            network.max_power_per_ap,
            size=network.num_aps
        )
        
        # Random association (each user connects to 1-3 random APs)
        ap_association = np.zeros((network.num_aps, network.num_users))
        for user_idx in range(network.num_users):
            num_serving_aps = np.random.randint(1, min(4, network.num_aps + 1))
            serving_aps = np.random.choice(
                network.num_aps,
                size=num_serving_aps,
                replace=False
            )
            ap_association[serving_aps, user_idx] = 1
        
        return power_allocation, ap_association
    
    @staticmethod
    def equal_power_all_serve(
        network,
        channel_matrix: tf.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Equal Power, All APs Serve All Users
        - All APs use equal power (50% of max)
        - All APs serve all users (full cooperation)
        
        Args:
            network: CellFreeNetworkSionna instance
            channel_matrix: Channel matrix (1, num_users, num_tx)
            
        Returns:
            power_allocation: (num_aps,) power in Watts
            ap_association: (num_aps, num_users) binary matrix
        """
        # Equal power at 50% of maximum
        power_allocation = np.ones(network.num_aps) * (0.5 * network.max_power_per_ap)
        
        # All APs serve all users
        ap_association = np.ones((network.num_aps, network.num_users))
        
        return power_allocation, ap_association
    
    @staticmethod
    def distance_based_power(
        network,
        channel_matrix: tf.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Distance-based Power Control
        - Nearest AP association
        - Power allocation proportional to average distance to SERVED users
        
        Args:
            network: CellFreeNetworkSionna instance
            channel_matrix: Channel matrix (1, num_users, num_tx)
            
        Returns:
            power_allocation: (num_aps,) power in Watts
            ap_association: (num_aps, num_users) binary matrix
        """
        # First, determine nearest AP association
        _, ap_association = BaselineStrategies.nearest_ap_max_power(
            network, channel_matrix
        )
        
        power_allocation = np.zeros(network.num_aps)
        avg_served_distances = np.zeros(network.num_aps)
        
        # Calculate average distance from each AP to its SERVED users only
        for ap_idx in range(network.num_aps):
            served_users = np.where(ap_association[ap_idx] > 0)[0]
            if len(served_users) > 0:
                avg_served_distances[ap_idx] = np.mean(network.distances[ap_idx, served_users])
        
        # Normalize distances based on maximum active distance
        max_dist = np.max(avg_served_distances) if np.max(avg_served_distances) > 0 else 1.0
        
        # Allocate power only to active APs
        for ap_idx in range(network.num_aps):
            if np.sum(ap_association[ap_idx]) > 0:
                normalized_dist = avg_served_distances[ap_idx] / max_dist
                # Original baseline logic: closer APs to their users get relatively more power
                power_factor = 1.0 - 0.5 * normalized_dist  # Range: [0.5, 1.0]
                power_allocation[ap_idx] = power_factor * network.max_power_per_ap
        
        return power_allocation, ap_association
    
    @staticmethod
    def load_balancing(
        network,
        channel_matrix: tf.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load Balancing Strategy
        - Distribute users evenly across APs
        - Power proportional to number of users served
        
        Args:
            network: CellFreeNetworkSionna instance
            channel_matrix: Channel matrix (1, num_users, num_tx)
            
        Returns:
            power_allocation: (num_aps,) power in Watts
            ap_association: (num_aps, num_users) binary matrix
        """
        # Get channel gains
        channel_gain_per_ap = network.get_channel_gain_per_ap(channel_matrix)
        
        # Load balancing: assign each user to top-K APs, preferring least loaded
        ap_association = np.zeros((network.num_aps, network.num_users))
        ap_loads = np.zeros(network.num_aps)
        num_serving_per_user = min(5, network.num_aps)  # Each user served by top-5 APs

        # Sort users by their best channel gain (prioritize users with poor channels)
        user_best_gains = np.max(channel_gain_per_ap, axis=0)
        user_order = np.argsort(user_best_gains)

        for user_idx in user_order:
            gains = channel_gain_per_ap[:, user_idx]
            # Load-aware scoring: higher gain + lower load = better
            max_gain = np.max(gains) + 1e-10
            scores = gains / max_gain - 0.3 * (ap_loads / (np.max(ap_loads) + 1))
            top_aps = np.argsort(scores)[-num_serving_per_user:]

            ap_association[top_aps, user_idx] = 1
            ap_loads[top_aps] += 1

        # Power allocation proportional to load (only active APs)
        active_mask = (np.sum(ap_association, axis=1) > 0).astype(float)
        max_load = np.max(ap_loads)
        if max_load > 0:
            power_factors = 0.3 + 0.7 * (ap_loads / max_load)
        else:
            power_factors = np.ones(network.num_aps)

        power_allocation = power_factors * network.max_power_per_ap * active_mask

        return power_allocation, ap_association




def evaluate_baseline(
    network,
    strategy_name: str,
    target_qos_bps: float,
    num_episodes: int = 100
) -> dict:
    """
    Evaluate a baseline strategy
    
    Args:
        network: CellFreeNetworkSionna instance
        strategy_name: Name of baseline strategy
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary with performance metrics
    """
    strategy_map = {
        'nearest_max': BaselineStrategies.nearest_ap_max_power,
        'random': BaselineStrategies.random_allocation,
        'equal_all': BaselineStrategies.equal_power_all_serve,
        'distance': BaselineStrategies.distance_based_power,
        'load_balance': BaselineStrategies.load_balancing
    }
    
    if strategy_name not in strategy_map:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    strategy_func = strategy_map[strategy_name]
    
    # QoS requirements from argument
    qos_requirements = np.ones(network.num_users) * target_qos_bps
    
    # Metrics
    energy_efficiencies = []
    qos_satisfactions = []
    avg_rates = []
    sinr_values = []
    
    for episode in range(num_episodes):
        # Generate channel
        channel_matrix = network.generate_channel_matrix(batch_size=1)
        
        # Get allocation from strategy
        power_allocation, ap_association = strategy_func(network, channel_matrix)
        
        # Calculate performance
        sinr, rates = network.calculate_sinr_and_rate(
            channel_matrix,
            power_allocation,
            ap_association
        )
        
        ee = network.calculate_energy_efficiency(rates, power_allocation, ap_association)
        qos_sat = network.calculate_qos_satisfaction(rates, qos_requirements)
        
        # Store metrics
        energy_efficiencies.append(ee.numpy()[0])
        qos_satisfactions.append(qos_sat.numpy()[0])
        avg_rates.append(tf.reduce_mean(rates).numpy() / 1e6)
        sinr_values.append(10 * np.log10(max(tf.reduce_mean(sinr).numpy(), 1e-12)))
    
    return {
        'strategy': strategy_name,
        'mean_energy_efficiency': np.mean(energy_efficiencies),
        'std_energy_efficiency': np.std(energy_efficiencies),
        'mean_qos_satisfaction': np.mean(qos_satisfactions),
        'std_qos_satisfaction': np.std(qos_satisfactions),
        'mean_rate_mbps': np.mean(avg_rates),
        'std_rate_mbps': np.std(avg_rates),
        'mean_sinr_db': np.mean(sinr_values),
        'std_sinr_db': np.std(sinr_values)
    }
