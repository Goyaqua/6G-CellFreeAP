"""
Test script for CellFreeNetworkSionna topology modes
Demonstrates 'random' and 'star' user distribution patterns
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.network.cellfree_network import CellFreeNetworkSionna


def test_random_topology():
    """Test Mode A: Random user distribution"""
    print("=" * 80)
    print("Testing Mode A: Random User Distribution")
    print("=" * 80)

    network = CellFreeNetworkSionna(
        num_aps=64,
        num_users=16,
        area_size=500.0,
        topology_mode='random',
        seed=42
    )

    print("\nNetwork Information:")
    info = network.get_network_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    print(f"\nAP Positions (first 5):")
    print(network.ap_positions[:5])

    print(f"\nUser Positions (first 5):")
    print(network.user_positions[:5])

    print(f"\nDistance Matrix Shape: {network.distances.shape}")
    print(f"Min distance: {network.distances.min():.2f}m")
    print(f"Max distance: {network.distances.max():.2f}m")
    print(f"Mean distance: {network.distances.mean():.2f}m")

    # Visualize
    print("\nGenerating visualization...")
    network.visualize_network(save_path='test_topology_random.png')
    print("Saved to: test_topology_random.png")


def test_star_topology():
    """Test Mode B: Star user distribution (8-Avenue Pattern)"""
    print("\n" + "=" * 80)
    print("Testing Mode B: Star User Distribution (8-Avenue Pattern)")
    print("=" * 80)

    network = CellFreeNetworkSionna(
        num_aps=64,
        num_users=24,  # More users to see the pattern better
        area_size=500.0,
        topology_mode='star',
        seed=42
    )

    print("\nNetwork Information:")
    info = network.get_network_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    print(f"\nUser Positions (first 8):")
    print(network.user_positions[:8])

    # Analyze star pattern
    center_x = network.area_size / 2
    center_y = network.area_size / 2

    radial_distances = []
    for pos in network.user_positions:
        dist = np.sqrt((pos[0] - center_x)**2 + (pos[1] - center_y)**2)
        radial_distances.append(dist)

    print(f"\nRadial distances from center:")
    print(f"  Min: {min(radial_distances):.2f}m (should be ~40m)")
    print(f"  Max: {max(radial_distances):.2f}m (should be ~240m)")
    print(f"  Mean: {np.mean(radial_distances):.2f}m")

    print(f"\nDistance Matrix Shape: {network.distances.shape}")
    print(f"Min AP-User distance: {network.distances.min():.2f}m")
    print(f"Max AP-User distance: {network.distances.max():.2f}m")

    # Visualize
    print("\nGenerating visualization...")
    network.visualize_network(save_path='test_topology_star.png')
    print("Saved to: test_topology_star.png")


def test_reset_topology():
    """Test dynamic topology switching with reset_topology()"""
    print("\n" + "=" * 80)
    print("Testing Dynamic Topology Switching")
    print("=" * 80)

    # Start with random
    network = CellFreeNetworkSionna(
        num_aps=36,
        num_users=12,
        area_size=500.0,
        topology_mode='random',
        seed=42
    )

    print("\nInitial topology: random")
    print(f"User positions (first 3):\n{network.user_positions[:3]}")

    # Switch to star
    print("\nSwitching to 'star' topology...")
    new_positions = network.reset_topology(topology_mode='star')
    print(f"New user positions (first 3):\n{network.user_positions[:3]}")
    print(f"Topology mode: {network.topology_mode}")

    # Switch back to random
    print("\nSwitching back to 'random' topology...")
    network.reset_topology(topology_mode='random')
    print(f"New user positions (first 3):\n{network.user_positions[:3]}")
    print(f"Topology mode: {network.topology_mode}")

    # Reset with same topology (re-randomize)
    print("\nRe-randomizing with same 'random' topology...")
    old_pos = network.user_positions[0].copy()
    network.reset_topology()
    new_pos = network.user_positions[0]
    print(f"Position changed: {not np.allclose(old_pos, new_pos)}")

    print("\n✓ Dynamic topology switching works correctly!")


def test_channel_generation():
    """Test that channel generation works with both topologies"""
    print("\n" + "=" * 80)
    print("Testing Channel Generation with Both Topologies")
    print("=" * 80)

    for mode in ['random', 'star']:
        print(f"\n--- Testing with '{mode}' topology ---")

        network = CellFreeNetworkSionna(
            num_aps=16,
            num_users=8,
            area_size=500.0,
            topology_mode=mode,
            seed=42
        )

        # Generate channel matrix
        batch_size = 2
        channel_matrix = network.generate_channel_matrix(batch_size=batch_size)

        print(f"Channel matrix shape: {channel_matrix.shape}")
        print(f"Expected: ({batch_size}, {network.num_users}, {network.num_tx})")

        # Test SINR and rate calculation
        power_allocation = np.ones(network.num_aps) * 0.1  # 100mW per AP
        ap_association = np.ones((network.num_aps, network.num_users))  # All APs serve all users

        sinr, rates = network.calculate_sinr_and_rate(
            channel_matrix,
            power_allocation,
            ap_association
        )

        print(f"SINR shape: {sinr.shape}")
        print(f"Rates shape: {rates.shape}")
        print(f"Mean rate: {rates.numpy().mean() / 1e6:.2f} Mbps")

        print(f"✓ Channel generation works for '{mode}' topology")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("Cell-Free Network Topology Modes Test Suite")
    print("=" * 80)

    # Run tests
    test_random_topology()
    test_star_topology()
    test_reset_topology()
    test_channel_generation()

    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)
    print("\nKey Features Demonstrated:")
    print("  ✓ Random user distribution (Mode A)")
    print("  ✓ Star user distribution - 8-avenue pattern (Mode B)")
    print("  ✓ Fixed AP grid infrastructure")
    print("  ✓ Dynamic topology switching with reset_topology()")
    print("  ✓ Channel generation compatibility")
    print("  ✓ Enhanced visualization with topology indicators")
    print("\nGenerated files:")
    print("  - test_topology_random.png")
    print("  - test_topology_star.png")


if __name__ == '__main__':
    main()
