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
        
        # Get channel gains
        channel_gain = tf.abs(channel_matrix[0]).numpy()  # (num_users, num_tx)
        
        # Average channel gain per AP
        channel_gain_per_ap = np.zeros((network.num_aps, network.num_users))
        for ap_idx in range(network.num_aps):
            ant_start = ap_idx * network.num_antennas_per_ap
            ant_end = (ap_idx + 1) * network.num_antennas_per_ap
            channel_gain_per_ap[ap_idx] = np.mean(
                channel_gain[:, ant_start:ant_end],
                axis=1
            )
        
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
        - Power allocation proportional to average distance to users
        - Nearest AP association
        
        Args:
            network: CellFreeNetworkSionna instance
            channel_matrix: Channel matrix (1, num_users, num_tx)
            
        Returns:
            power_allocation: (num_aps,) power in Watts
            ap_association: (num_aps, num_users) binary matrix
        """
        # Calculate average distance from each AP to all users
        avg_distances = np.mean(network.distances, axis=1)
        
        # Normalize distances and invert (closer APs get more power)
        max_dist = np.max(avg_distances)
        normalized_distances = avg_distances / max_dist
        power_factors = 1.0 - 0.5 * normalized_distances  # Range: [0.5, 1.0]
        
        # Allocate power
        power_allocation = power_factors * network.max_power_per_ap
        
        # Nearest AP association
        _, ap_association = BaselineStrategies.nearest_ap_max_power(
            network, channel_matrix
        )
        
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
        channel_gain = tf.abs(channel_matrix[0]).numpy()
        channel_gain_per_ap = np.zeros((network.num_aps, network.num_users))
        for ap_idx in range(network.num_aps):
            ant_start = ap_idx * network.num_antennas_per_ap
            ant_end = (ap_idx + 1) * network.num_antennas_per_ap
            channel_gain_per_ap[ap_idx] = np.mean(
                channel_gain[:, ant_start:ant_end],
                axis=1
            )
        
        # Load balancing: assign users to least loaded AP with good channel
        ap_association = np.zeros((network.num_aps, network.num_users))
        ap_loads = np.zeros(network.num_aps)
        
        # Sort users by their best channel gain (prioritize users with poor channels)
        user_best_gains = np.max(channel_gain_per_ap, axis=0)
        user_order = np.argsort(user_best_gains)
        
        for user_idx in user_order:
            # Find APs with good channel (top 50%)
            gains = channel_gain_per_ap[:, user_idx]
            threshold = np.median(gains)
            good_aps = np.where(gains >= threshold)[0]
            
            if len(good_aps) == 0:
                good_aps = [np.argmax(gains)]
            
            # Among good APs, choose least loaded
            least_loaded_ap = good_aps[np.argmin(ap_loads[good_aps])]
            ap_association[least_loaded_ap, user_idx] = 1
            ap_loads[least_loaded_ap] += 1
        
        # Power allocation proportional to load
        max_load = np.max(ap_loads)
        if max_load > 0:
            power_factors = 0.3 + 0.7 * (ap_loads / max_load)  # Range: [0.3, 1.0]
        else:
            power_factors = np.ones(network.num_aps)
        
        power_allocation = power_factors * network.max_power_per_ap
        
        return power_allocation, ap_association


def evaluate_baseline(
    network,
    strategy_name: str,
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
    
    # QoS requirements (5 Mbps per user)
    qos_requirements = np.ones(network.num_users) * 5e6
    
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
        
        ee = network.calculate_energy_efficiency(rates, power_allocation)
        qos_sat = network.calculate_qos_satisfaction(rates, qos_requirements)
        
        # Store metrics
        energy_efficiencies.append(ee.numpy()[0])
        qos_satisfactions.append(qos_sat.numpy()[0])
        avg_rates.append(tf.reduce_mean(rates).numpy() / 1e6)
        sinr_values.append(10 * np.log10(tf.reduce_mean(sinr).numpy()))
    
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
