"""
Train RL Agent for Cell-Free Resource Allocation
"""

import sys
import os

# Force CPU-only mode to avoid TensorFlow Metal deadlock issues on M3 Pro
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import yaml
from environment.cellfree_env import CellFreeEnv
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from utils.plotting import create_experiment_dir


def main():
    parser = argparse.ArgumentParser(description='Train RL Agent')
    parser.add_argument('--agent', type=str, default='dqn', choices=['dqn', 'ppo'],
                       help='RL agent type')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--timesteps', type=int, default=None,
                       help='Total training timesteps (overrides config)')
    parser.add_argument('--exp_dir', type=str, default='experiments',
                       help='Base directory for experiments')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create experiment directory
    exp_dir = create_experiment_dir(args.exp_dir)
    print(f"Experiment directory: {exp_dir}")
    
    # Create environment
    print("\nCreating Environment...")
    env = CellFreeEnv(
        config_path=args.config,
        render_mode=None
    )
    
    print(f"Environment Configuration:")
    print(f"  - Observation Space: {env.observation_space.shape}")
    print(f"  - Action Space: {env.action_space}")
    print(f"  - Number of APs: {env.num_aps}")
    print(f"  - Number of Users: {env.num_users}")
    print(f"  - QoS Requirement: {env.qos_min_rate / 1e6:.1f} Mbps")
    print(f"  - Episode Length: {env.episode_length}")
    
    # Create agent
    print(f"\nCreating {args.agent.upper()} Agent...")
    
    tensorboard_log = os.path.join(exp_dir, 'tensorboard')
    
    if args.agent == 'dqn':
        agent = DQNAgent(
            env=env,
            learning_rate=config['training']['dqn']['learning_rate'],
            buffer_size=config['training']['dqn']['buffer_size'],
            learning_starts=config['training']['dqn']['learning_starts'],
            batch_size=config['training']['dqn']['batch_size'],
            gamma=config['training']['dqn']['gamma'],
            tau=config['training']['dqn']['tau'],
            train_freq=config['training']['dqn']['train_freq'],
            gradient_steps=config['training']['dqn']['gradient_steps'],
            target_update_interval=config['training']['dqn']['target_update_interval'],
            exploration_fraction=config['training']['dqn']['exploration_fraction'],
            exploration_initial_eps=config['training']['dqn']['exploration_initial_eps'],
            exploration_final_eps=config['training']['dqn']['exploration_final_eps'],
            verbose=config['output']['verbose'],
            tensorboard_log=tensorboard_log,
            seed=config['seed']['agent_seed']
        )
    else:  # ppo
        agent = PPOAgent(
            env=env,
            learning_rate=config['training']['ppo']['learning_rate'],
            n_steps=config['training']['ppo']['n_steps'],
            batch_size=config['training']['ppo']['batch_size'],
            n_epochs=config['training']['ppo']['n_epochs'],
            gamma=config['training']['ppo']['gamma'],
            gae_lambda=config['training']['ppo']['gae_lambda'],
            clip_range=config['training']['ppo']['clip_range'],
            ent_coef=config['training']['ppo']['ent_coef'],
            vf_coef=config['training']['ppo']['vf_coef'],
            max_grad_norm=config['training']['ppo']['max_grad_norm'],
            verbose=config['output']['verbose'],
            tensorboard_log=tensorboard_log,
            seed=config['seed']['agent_seed']
        )
    
    # Training parameters
    total_timesteps = args.timesteps if args.timesteps else config['training']['total_timesteps']
    
    print(f"\nTraining Configuration:")
    print(f"  - Total Timesteps: {total_timesteps}")
    print(f"  - Evaluation Frequency: {config['training']['eval_freq']}")
    print(f"  - Save Frequency: {config['training']['save_freq']}")
    print(f"  - TensorBoard Log: {tensorboard_log}")
    
    # Train
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")
    
    model_save_dir = os.path.join(exp_dir, 'models')
    log_dir = os.path.join(exp_dir, 'logs')
    
    train_info = agent.train(
        total_timesteps=total_timesteps,
        log_dir=log_dir,
        model_save_dir=model_save_dir,
        save_freq=config['training']['save_freq'],
        eval_freq=config['training']['eval_freq'],
        n_eval_episodes=config['training']['n_eval_episodes']
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Final model saved to: {train_info['final_model_path']}")

    print(f"\n✓ Training completed successfully!")
    print(f"✓ Results saved to: {exp_dir}")
    print(f"\nTo view training progress, run:")
    print(f"  tensorboard --logdir {tensorboard_log}")
    print(f"\nTo evaluate the trained model, run:")
    print(f"  python src/analyze_network.py --mode evaluate --model {train_info['final_model_path']} --episodes 100")


if __name__ == '__main__':
    main()
