"""
Cell-Free Network Simulation using Sionna
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
import sionna
from sionna.channel import RayleighBlockFading, AWGN
from sionna.utils import QAMSource


class CellFreeNetworkSionna:
    """
    Cell-Free Massive MIMO Network Simulator using Sionna
    
    This class simulates a cell-free network where multiple Access Points (APs)
    cooperatively serve users without cell boundaries.
    """
    
    def __init__(
        self,
        num_aps: int = 25,
        num_users: int = 10,
        num_antennas_per_ap: int = 1,
        area_size: float = 500.0,
        carrier_frequency: float = 3.5e9,
        bandwidth: float = 10e6,
        max_power_per_ap: float = 200e-3,
        noise_power_dbm: float = -94,
        circuit_power_per_ap: float = 200e-3,
        seed: Optional[int] = None
    ):
        """
        Initialize Cell-Free Network

        Args:
            num_aps: Number of Access Points
            num_users: Number of users
            num_antennas_per_ap: Number of antennas per AP
            area_size: Coverage area size (meters)
            carrier_frequency: Carrier frequency (Hz)
            bandwidth: System bandwidth (Hz)
            max_power_per_ap: Maximum transmit power per AP (Watts)
            noise_power_dbm: Noise power (dBm)
            circuit_power_per_ap: Circuit power consumption per active AP (Watts)
            seed: Random seed for reproducibility
        """
        self.num_aps = num_aps
        self.num_users = num_users
        self.num_antennas_per_ap = num_antennas_per_ap
        self.area_size = area_size
        self.carrier_frequency = carrier_frequency
        self.bandwidth = bandwidth
        self.max_power_per_ap = max_power_per_ap
        self.noise_power_dbm = noise_power_dbm
        self.circuit_power_per_ap = circuit_power_per_ap
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
        
        # Total transmit antennas
        self.num_tx = num_aps * num_antennas_per_ap
        self.num_rx_per_user = 1  # Single antenna users
        
        # Deploy network
        self.ap_positions = self._deploy_aps()
        self.user_positions = self._deploy_users()
        self.distances = self._calculate_distances()
        
        # Setup channel model
        self._setup_channel()
        
        # Convert noise power to linear scale (Watts)
        self.noise_power_linear = 10**(self.noise_power_dbm / 10) / 1000
        
    def _deploy_aps(self) -> np.ndarray:
        """Deploy APs in a regular grid pattern"""
        side_length = int(np.ceil(np.sqrt(self.num_aps)))
        spacing = self.area_size / (side_length + 1)
        
        positions = []
        count = 0
        for i in range(side_length):
            for j in range(side_length):
                if count >= self.num_aps:
                    break
                x = (i + 1) * spacing
                y = (j + 1) * spacing
                positions.append([x, y])
                count += 1
            if count >= self.num_aps:
                break
        
        return np.array(positions)
    
    def _deploy_users(self) -> np.ndarray:
        """Deploy users randomly in the coverage area"""
        return np.random.uniform(0, self.area_size, (self.num_users, 2))
    
    def _calculate_distances(self) -> np.ndarray:
        """Calculate distance matrix between all APs and users"""
        distances = np.zeros((self.num_aps, self.num_users))
        for i in range(self.num_aps):
            for j in range(self.num_users):
                distances[i, j] = np.linalg.norm(
                    self.ap_positions[i] - self.user_positions[j]
                )
        return distances
    
    def _setup_channel(self):
        """Setup Sionna channel models"""
        # Rayleigh block fading for small-scale fading
        self.channel = RayleighBlockFading(
            num_rx=self.num_users,
            num_rx_ant=self.num_rx_per_user,
            num_tx=self.num_tx,
            num_tx_ant=1
        )
        
        # AWGN channel
        self.awgn = AWGN()
    
    @tf.function(experimental_relax_shapes=True)
    def calculate_pathloss(self) -> tf.Tensor:
        """
        Calculate path loss using 3GPP-like model
        OPTIMIZED: @tf.function decorator prevents memory leak

        Returns:
            Path loss in linear scale, shape (num_aps, num_users)
        """
        # Path loss parameters
        d0 = 1.0  # Reference distance (meters)
        PL0 = 30.0  # Path loss at reference distance (dB)
        alpha = 3.5  # Path loss exponent
        sigma_shadowing = 8.0  # Shadowing standard deviation (dB)

        # Convert distances to TensorFlow tensor
        distances_tf = tf.constant(self.distances, dtype=tf.float32)

        # Ensure minimum distance
        distances_safe = tf.maximum(distances_tf, d0)

        # Path loss in dB: PL(d) = PL0 + 10*alpha*log10(d/d0)
        pathloss_db = PL0 + 10.0 * alpha * tf.experimental.numpy.log10(distances_safe / d0)

        # Add log-normal shadowing
        shadowing_db = tf.random.normal(
            tf.shape(distances_safe),
            mean=0.0,
            stddev=sigma_shadowing,
            dtype=tf.float32
        )

        # Total large-scale fading
        total_loss_db = pathloss_db + shadowing_db

        # Convert to linear scale
        pathloss_linear = tf.pow(10.0, -total_loss_db / 10.0)

        return pathloss_linear
    
    @tf.function(experimental_relax_shapes=True)
    def generate_channel_matrix(self, batch_size: int = 1) -> tf.Tensor:
        """
        Generate channel matrix with both small-scale and large-scale fading.
        FIXED: Added num_time_steps=1 argument for Sionna compatibility.
        FIXED: Explicit casting for Complex64 compatibility.
        OPTIMIZED: @tf.function decorator prevents memory leak from graph rebuilding.
        """
        # 1. Generate small-scale fading (Rayleigh) -> COMPLEX64
        # RayleighBlockFading returns a tuple: (channel_coefficients, delays)
        h, _ = self.channel(batch_size, num_time_steps=1)

        # 2. Squeeze to remove single dimensions
        # Sionna output: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, delays]
        # After squeeze: [num_users, num_tx]
        h = tf.squeeze(h)

        # Expand batch dimension if needed
        if len(h.shape) == 2:
            h = tf.expand_dims(h, axis=0)
        # Result shape: [batch_size, num_users, num_tx]
        
        # 3. Calculate Large-Scale Fading (Path Loss) -> FLOAT32
        pathloss = self.calculate_pathloss()
        
        # Expand path loss for batch dimension and repeat for antennas
        pathloss_per_antenna = tf.repeat(
            pathloss,
            repeats=self.num_antennas_per_ap,
            axis=0
        )  # Shape: (num_tx, num_users)
        
        pathloss_per_antenna = tf.transpose(pathloss_per_antenna) # Shape: (num_users, num_tx)
        
        pathloss_batched = tf.expand_dims(pathloss_per_antenna, axis=0)
        pathloss_batched = tf.repeat(pathloss_batched, repeats=batch_size, axis=0)
        
        # 4. Apply path loss to small-scale fading
        # ÖNCEKİ FIX: Float olan pathloss'u Complex64'e çeviriyoruz
        pathloss_complex = tf.cast(tf.sqrt(pathloss_batched), dtype=tf.complex64)
        
        h_with_pathloss = h * pathloss_complex
        
        return h_with_pathloss
    
    def calculate_sinr_and_rate(
        self,
        channel_matrix: tf.Tensor,
        power_allocation: np.ndarray,
        ap_association: np.ndarray
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Calculate SINR and achievable data rate for each user
        OPTIMIZED: Converts to TensorFlow and calls optimized function to prevent memory leak

        Args:
            channel_matrix: Complex channel matrix (batch_size, num_users, num_tx)
            power_allocation: Power per AP in Watts, shape (num_aps,)
            ap_association: Binary matrix indicating AP-user association (num_aps, num_users)

        Returns:
            sinr: SINR for each user (batch_size, num_users)
            rates: Data rate in bps (batch_size, num_users)
        """
        # Convert numpy to TensorFlow tensors
        power_allocation_tf = tf.constant(power_allocation, dtype=tf.float32)
        ap_association_tf = tf.constant(ap_association, dtype=tf.float32)

        # Call optimized TensorFlow function
        return self._calculate_sinr_and_rate_tf(channel_matrix, power_allocation_tf, ap_association_tf)

    @tf.function(experimental_relax_shapes=True, reduce_retracing=True)
    def _calculate_sinr_and_rate_tf(
        self,
        channel_matrix: tf.Tensor,
        power_allocation: tf.Tensor,
        ap_association: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        TensorFlow-optimized SINR and rate calculation
        Prevents memory leak by caching the computation graph
        """
        # Power per antenna (split AP power equally among antennas)
        power_per_antenna = tf.repeat(
            power_allocation / tf.cast(self.num_antennas_per_ap, tf.float32),
            self.num_antennas_per_ap
        )  # Shape: (num_tx,)

        # Channel gain magnitude squared
        channel_gain = tf.abs(channel_matrix) ** 2
        # Shape: (batch_size, num_users, num_tx)

        # Expand power for broadcasting: (num_tx,) -> (1, 1, num_tx)
        power_expanded = tf.reshape(power_per_antenna, [1, 1, -1])

        # Received power from all antennas: (batch_size, num_users, num_tx)
        received_power_all = channel_gain * power_expanded

        # Reshape for AP-based calculation
        # (batch_size, num_users, num_tx) -> (batch_size, num_users, num_aps, num_antennas_per_ap)
        batch_size = tf.shape(channel_matrix)[0]
        received_power_reshaped = tf.reshape(
            received_power_all,
            [batch_size, self.num_users, self.num_aps, self.num_antennas_per_ap]
        )

        # Sum power across antennas of each AP: (batch_size, num_users, num_aps)
        power_per_ap = tf.reduce_sum(received_power_reshaped, axis=-1)

        # Transpose association matrix for broadcasting: (num_aps, num_users) -> (num_users, num_aps)
        ap_association_t = tf.transpose(ap_association)

        # Expand for batch: (num_users, num_aps) -> (1, num_users, num_aps)
        ap_association_expanded = tf.expand_dims(ap_association_t, axis=0)

        # Calculate signal and interference power
        signal_power = tf.reduce_sum(power_per_ap * ap_association_expanded, axis=-1)  # (batch_size, num_users)
        total_power = tf.reduce_sum(power_per_ap, axis=-1)  # (batch_size, num_users)
        interference_power = total_power - signal_power  # (batch_size, num_users)

        # Calculate SINR with epsilon to prevent division by zero
        epsilon = 1e-12  # Small constant to prevent numerical instability
        sinr_tensor = signal_power / (
            interference_power + tf.constant(self.noise_power_linear, dtype=tf.float32) + epsilon
        )

        # Clip SINR to prevent extremely large values (max SINR = 60 dB = 1e6)
        sinr_clipped = tf.clip_by_value(sinr_tensor, 0.0, 1e6)

        # Calculate achievable rate using Shannon formula
        # Add epsilon inside log to prevent log(0) or log(negative)
        rates = self.bandwidth * tf.math.log(1.0 + tf.maximum(sinr_clipped, 0.0) + epsilon) / tf.math.log(2.0)

        return sinr_tensor, rates
    
    def calculate_energy_efficiency(
        self,
        rates: tf.Tensor,
        power_allocation: np.ndarray,
        ap_association: Optional[np.ndarray] = None
    ) -> tf.Tensor:
        """
        Calculate network energy efficiency in bits/Joule

        P_total = P_tx + P_circuit
        where P_circuit = circuit_power_per_ap * num_active_APs

        Args:
            rates: Data rates in bps, shape (batch_size, num_users)
            power_allocation: Transmit power per AP in Watts, shape (num_aps,)
            ap_association: Binary matrix (num_aps, num_users) indicating active APs

        Returns:
            Energy efficiency in bits/Joule, shape (batch_size,)
        """
        # Total network throughput
        total_rate = tf.reduce_sum(rates, axis=-1)  # (batch_size,)

        # Calculate total power consumption
        # 1. Transmit power
        total_tx_power = np.sum(power_allocation)

        # 2. Circuit power (only for active APs)
        if ap_association is not None:
            # An AP is active if it serves at least one user
            active_aps = np.sum(np.sum(ap_association, axis=1) > 0)
            total_circuit_power = active_aps * self.circuit_power_per_ap
        else:
            # If no association matrix, assume all APs are active
            total_circuit_power = self.num_aps * self.circuit_power_per_ap

        # Total power = Transmit + Circuit
        total_power = total_tx_power + total_circuit_power

        # Energy efficiency with numerical stability
        epsilon = 1e-8  # Prevent division by zero
        # Clip total_rate to prevent inf propagation
        # Max realistic rate: 10 users × 100 Mbps = 1000 Mbps = 1e9 bps
        total_rate_clipped = tf.clip_by_value(total_rate, 0.0, 1e10)
        ee = total_rate_clipped / (total_power + epsilon)

        return ee
    
    def calculate_qos_satisfaction(
        self,
        rates: tf.Tensor,
        qos_requirements: np.ndarray
    ) -> tf.Tensor:
        """
        Calculate QoS satisfaction rate
        
        Args:
            rates: Data rates in bps, shape (batch_size, num_users)
            qos_requirements: Required rate per user in bps, shape (num_users,)
            
        Returns:
            QoS satisfaction rate (percentage), shape (batch_size,)
        """
        qos_req_tensor = tf.constant(qos_requirements, dtype=tf.float32)
        qos_req_tensor = tf.expand_dims(qos_req_tensor, axis=0)
        
        # Check if each user meets QoS
        meets_qos = tf.cast(rates >= qos_req_tensor, tf.float32)
        
        # Calculate satisfaction rate
        satisfaction_rate = tf.reduce_mean(meets_qos, axis=-1) * 100
        
        return satisfaction_rate
    
    def visualize_association_matrix(
        self,
        ap_association: np.ndarray,
        title: str = "AP-User Association Matrix",
        save_path: Optional[str] = None
    ):
        """
        Visualize AP-User association matrix as a heatmap

        Args:
            ap_association: Binary matrix (num_aps, num_users) indicating associations
            title: Title for the plot
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=(12, 8))

        # Create heatmap
        im = plt.imshow(ap_association, cmap='YlOrRd', aspect='auto', interpolation='nearest')

        # Add colorbar
        cbar = plt.colorbar(im, label='Association (0=No, 1=Yes)')
        cbar.set_ticks([0, 1])

        # Set ticks and labels
        plt.xticks(range(self.num_users), [f'U{i}' for i in range(self.num_users)], fontsize=10)
        plt.yticks(range(self.num_aps), [f'AP{i}' for i in range(self.num_aps)], fontsize=10)

        # Add labels
        plt.xlabel('Users', fontsize=14, fontweight='bold')
        plt.ylabel('Access Points', fontsize=14, fontweight='bold')
        plt.title(title, fontsize=16, fontweight='bold', pad=20)

        # Add text annotations showing the values
        for i in range(self.num_aps):
            for j in range(self.num_users):
                text_color = 'white' if ap_association[i, j] > 0.5 else 'black'
                plt.text(j, i, int(ap_association[i, j]),
                        ha="center", va="center", color=text_color, fontsize=9)

        # Add grid
        plt.grid(False)

        # Add statistics
        active_aps = np.sum(np.sum(ap_association, axis=1) > 0)
        avg_aps_per_user = np.mean(np.sum(ap_association, axis=0))
        plt.text(0.02, 0.98,
                f'Active APs: {active_aps}/{self.num_aps}\nAvg APs/User: {avg_aps_per_user:.2f}',
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def visualize_network(self, save_path: Optional[str] = None):
        """Visualize network topology"""
        plt.figure(figsize=(10, 10))
        
        # Plot APs
        plt.scatter(
            self.ap_positions[:, 0],
            self.ap_positions[:, 1],
            c='red',
            marker='^',
            s=300,
            label='Access Points',
            edgecolors='black',
            linewidths=2,
            zorder=5
        )
        
        # Plot users
        plt.scatter(
            self.user_positions[:, 0],
            self.user_positions[:, 1],
            c='blue',
            marker='o',
            s=150,
            label='Users',
            edgecolors='black',
            linewidths=1.5,
            zorder=5
        )
        
        # Add labels
        for i, pos in enumerate(self.ap_positions):
            plt.text(pos[0], pos[1] + 15, f'AP{i}', ha='center', fontsize=8)
        
        for i, pos in enumerate(self.user_positions):
            plt.text(pos[0], pos[1] - 15, f'U{i}', ha='center', fontsize=8)
        
        plt.xlabel('X (meters)', fontsize=14)
        plt.ylabel('Y (meters)', fontsize=14)
        plt.title('Cell-Free Network Topology', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12, loc='upper right')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xlim(-20, self.area_size + 20)
        plt.ylim(-20, self.area_size + 20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_network_info(self) -> Dict:
        """Get network configuration information"""
        return {
            'num_aps': self.num_aps,
            'num_users': self.num_users,
            'num_antennas_per_ap': self.num_antennas_per_ap,
            'area_size': self.area_size,
            'carrier_frequency': self.carrier_frequency,
            'bandwidth': self.bandwidth,
            'max_power_per_ap': self.max_power_per_ap,
            'noise_power_dbm': self.noise_power_dbm,
            'total_tx_antennas': self.num_tx
        }
