"""
Quick Demo: Cell-Free Network Simulation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from network.cellfree_network import CellFreeNetworkSionna
from agents.baselines import BaselineStrategies
from utils.plotting import plot_channel_gains
import numpy as np


def main():
    print("="*80)
    print("CELL-FREE NETWORK SIMULATION DEMO")
    print("="*80)
    
    # Create network
    print("\n1. Creating Cell-Free Network...")
    network = CellFreeNetworkSionna(
        num_aps=16,
        num_users=8,
        num_antennas_per_ap=1,
        area_size=500.0,
        seed=42
    )
    
    info = network.get_network_info()
    print(f"   ✓ Network created with {info['num_aps']} APs and {info['num_users']} users")
    
    # Visualize network
    print("\n2. Visualizing Network Topology...")
    network.visualize_network()
    
    # Generate channel
    print("\n3. Generating Channel Matrix...")
    channel_matrix = network.generate_channel_matrix(batch_size=1)
    print(f"   ✓ Channel shape: {channel_matrix.shape}")
    
    # Plot channel gains
    print("\n4. Plotting Channel Gains...")
    plot_channel_gains(network)
    
    # Test baseline strategies
    print("\n5. Testing Baseline Strategies...")
    
    strategies = {
        'Nearest AP + Max Power': BaselineStrategies.nearest_ap_max_power,
        'Equal Power + All Serve': BaselineStrategies.equal_power_all_serve,
        'Load Balancing': BaselineStrategies.load_balancing
    }
    
    for name, strategy_func in strategies.items():
        print(f"\n   Testing: {name}")

        # Get allocation
        power_allocation, ap_association = strategy_func(network, channel_matrix)

        # Calculate performance
        sinr, rates = network.calculate_sinr_and_rate(
            channel_matrix,
            power_allocation,
            ap_association
        )

        ee = network.calculate_energy_efficiency(rates, power_allocation, ap_association)
        qos_requirements = np.ones(network.num_users) * 5e6  # 5 Mbps
        qos_sat = network.calculate_qos_satisfaction(rates, qos_requirements)

        # Print results
        print(f"   - Average SINR: {10 * np.log10(np.mean(sinr.numpy())):.2f} dB")
        print(f"   - Average Rate: {np.mean(rates.numpy()) / 1e6:.2f} Mbps")
        print(f"   - Energy Efficiency: {ee.numpy()[0]:.2e} bits/Joule")
        print(f"   - QoS Satisfaction: {qos_sat.numpy()[0]:.2f}%")
        print(f"   - Active APs: {np.sum(np.sum(ap_association, axis=1) > 0)}/{network.num_aps}")

        # Visualize AP-User Association Matrix
        print(f"   - Visualizing Association Matrix...")
        network.visualize_association_matrix(
            ap_association,
            title=f"AP-User Association: {name}"
        )
    
    print("\n" + "="*80)
    print("DEMO COMPLETED")
    print("="*80)
    print("\nNext steps:")
    print("  1. Run baseline evaluation: python run_baseline.py")
    print("  2. Train RL agent: python train_agent.py --agent dqn --timesteps 50000")
    print("  3. Evaluate trained agent: python evaluate.py --model path/to/model --compare_baselines")


if __name__ == '__main__':
    main()
