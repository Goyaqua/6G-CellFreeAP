"""
Shared Visualization Functions for Cell-Free Network Analysis

Supports both Statistical and Ray Tracing (RT) channel models.
All plotting functions handle 2D (statistical) and 3D (RT) positions.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_network_topology(network, title="Network Topology", save_path=None, rt_mode=False):
    """
    Plot network topology showing AP and user positions (2D top-down view)

    Args:
        network: CellFreeNetworkSionna instance
        title: Plot title
        save_path: Path to save the plot
        rt_mode: Whether RT mode is active (affects styling)
    """
    fig, ax = plt.subplots(figsize=(12, 12) if rt_mode else (10, 8))

    # Plot APs (handle both 2D and 3D positions)
    ap_x = network.ap_positions[:, 0]
    ap_y = network.ap_positions[:, 1]
    ap_label = 'Access Points'
    if network.ap_positions.shape[1] == 3:  # RT mode (3D)
        ap_label = f'Access Points (h={network.ap_positions[0, 2]:.1f}m)'

    ax.scatter(ap_x, ap_y, c='red', s=300 if rt_mode else 200, marker='^',
               label=ap_label, edgecolors='black', linewidths=2, zorder=3, alpha=0.8)

    # Plot Users (handle both 2D and 3D positions)
    user_x = network.user_positions[:, 0]
    user_y = network.user_positions[:, 1]
    user_label = 'Users'
    if network.user_positions.shape[1] == 3:  # RT mode (3D)
        user_label = f'Users (h={network.user_positions[0, 2]:.1f}m)'

    ax.scatter(user_x, user_y, c='blue', s=200 if rt_mode else 150, marker='o',
               label=user_label, edgecolors='black', linewidths=1.5, zorder=3, alpha=0.8)

    # Add labels (fewer labels in RT mode to avoid clutter)
    label_offset = 5 if rt_mode else 15
    if not rt_mode or network.num_aps <= 36:
        for i, pos in enumerate(network.ap_positions[:, :2]):
            ax.text(pos[0], pos[1]+label_offset, f'AP{i}', ha='center', va='bottom',
                   fontsize=8, fontweight='bold', color='darkred')

    if not rt_mode or network.num_users <= 20:
        for i, pos in enumerate(network.user_positions[:, :2]):
            ax.text(pos[0], pos[1]+label_offset, f'U{i}', ha='center', va='bottom',
                   fontsize=8, fontweight='bold', color='darkblue')

    ax.set_xlabel('X (meters)', fontsize=14 if rt_mode else 12, fontweight='bold' if rt_mode else 'normal')
    ax.set_ylabel('Y (meters)', fontsize=14 if rt_mode else 12, fontweight='bold' if rt_mode else 'normal')
    ax.set_title(title, fontsize=16 if rt_mode else 14, fontweight='bold', pad=20 if rt_mode else 10)
    ax.legend(fontsize=12 if rt_mode else 11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--' if rt_mode else '-')

    # Set limits based on mode
    if rt_mode:
        ax.set_xlim(-120, 120)
        ax.set_ylim(-120, 120)
        ax.set_aspect('equal')
        # Add scene boundary
        ax.axhline(y=-100, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=-100, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=100, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    else:
        ax.set_xlim(-50, network.area_size + 50)
        ax.set_ylim(-50, network.area_size + 50)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved topology plot: {save_path}")

    plt.close()
    return fig


def plot_rt_scene_3d(network, title="RT Scene 3D View", save_path=None):
    """
    Plot RT scene in 3D with APs and users

    Args:
        network: CellFreeNetworkSionna instance (must have 3D positions)
        title: Plot title
        save_path: Path to save the plot
    """
    if network.ap_positions.shape[1] != 3:
        print("⚠️  3D plotting only available for RT mode")
        return None

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    ap_positions = network.ap_positions
    user_positions = network.user_positions

    # Plot APs
    ax.scatter(ap_positions[:, 0], ap_positions[:, 1], ap_positions[:, 2],
              c='red', s=300, marker='^',
              label=f'APs (h={ap_positions[0, 2]:.1f}m)',
              edgecolors='black', linewidths=2, alpha=0.9)

    # Plot Users
    ax.scatter(user_positions[:, 0], user_positions[:, 1], user_positions[:, 2],
              c='blue', s=200, marker='o',
              label=f'Users (h={user_positions[0, 2]:.1f}m)',
              edgecolors='black', linewidths=1.5, alpha=0.9)

    # Plot vertical lines from ground to APs/Users
    for pos in ap_positions:
        ax.plot([pos[0], pos[0]], [pos[1], pos[1]], [0, pos[2]],
               'r--', alpha=0.3, linewidth=1)

    for pos in user_positions:
        ax.plot([pos[0], pos[0]], [pos[1], pos[1]], [0, pos[2]],
               'b--', alpha=0.3, linewidth=1)

    # Ground plane (semi-transparent)
    xx, yy = np.meshgrid(np.linspace(-100, 100, 10), np.linspace(-100, 100, 10))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')

    ax.set_xlabel('X (meters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (meters)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Height (meters)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11)

    # Set equal aspect ratio
    ax.set_xlim(-120, 120)
    ax.set_ylim(-120, 120)
    ax.set_zlim(0, 20)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved 3D scene plot: {save_path}")

    plt.close()
    return fig


def plot_channel_gain_matrix(channel_gain, title="Channel Gain Matrix", save_path=None):
    """
    Plot channel gain matrix heatmap

    Args:
        channel_gain: (num_aps, num_users) channel gain matrix
        title: Plot title
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(channel_gain, cmap='hot', aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Channel Gain (linear)', fontsize=11)

    # Add labels
    ax.set_xlabel('Users', fontsize=12)
    ax.set_ylabel('Access Points', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add text annotations
    num_aps, num_users = channel_gain.shape
    for i in range(num_aps):
        for j in range(num_users):
            text = ax.text(j, i, f'{channel_gain[i, j]:.2e}',
                          ha="center", va="center", color="white", fontsize=7)

    # Set ticks
    ax.set_xticks(range(num_users))
    ax.set_yticks(range(num_aps))
    ax.set_xticklabels([f'U{i}' for i in range(num_users)])
    ax.set_yticklabels([f'AP{i}' for i in range(num_aps)])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved channel gain plot: {save_path}")

    plt.close()
    return fig


def plot_ap_user_association(association_matrix, strategy_name="Strategy", save_path=None):
    """
    Plot AP-User association matrix

    Args:
        association_matrix: (num_aps, num_users) binary association matrix
        strategy_name: Name of the strategy
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(association_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Association (0=inactive, 1=active)', fontsize=11)

    # Add labels
    ax.set_xlabel('Users', fontsize=12)
    ax.set_ylabel('Access Points', fontsize=12)
    ax.set_title(f'AP-User Association: {strategy_name}', fontsize=14, fontweight='bold')

    # Add text annotations
    num_aps, num_users = association_matrix.shape
    for i in range(num_aps):
        for j in range(num_users):
            value = association_matrix[i, j]
            text_color = 'white' if value > 0.5 else 'black'
            text = ax.text(j, i, f'{int(value)}',
                          ha="center", va="center", color=text_color, fontsize=9)

    # Set ticks
    ax.set_xticks(range(num_users))
    ax.set_yticks(range(num_aps))
    ax.set_xticklabels([f'U{i}' for i in range(num_users)])
    ax.set_yticklabels([f'AP{i}' for i in range(num_aps)])

    # Add summary text
    active_aps = np.sum(np.any(association_matrix > 0, axis=1))
    avg_aps_per_user = np.sum(association_matrix) / num_users
    summary_text = f'Active APs: {active_aps}/{num_aps}\nAvg APs/User: {avg_aps_per_user:.1f}'
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved association matrix: {save_path}")

    plt.close()
    return fig


__all__ = [
    'plot_network_topology',
    'plot_rt_scene_3d',
    'plot_channel_gain_matrix',
    'plot_ap_user_association',
]
