"""
Proximal Policy Optimization (PPO) Agent for Cell-Free Resource Allocation
"""

import numpy as np
import tensorflow as tf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from typing import Optional, Dict
import os


class PPOAgent:
    """
    PPO Agent for continuous action space
    """
    
    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize PPO Agent
        
        Args:
            env: Gymnasium environment
            learning_rate: Learning rate
            n_steps: Steps to collect before update
            batch_size: Batch size for training
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Max gradient norm
            verbose: Verbosity level
            tensorboard_log: TensorBoard log directory
            seed: Random seed
        """
        self.env = Monitor(env)
        
        # Create PPO model
        self.model = PPO(
            policy='MlpPolicy',
            env=self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            seed=seed,
            device='auto'
        )
    
    def train(
        self,
        total_timesteps: int = 100000,
        log_dir: str = './logs',
        model_save_dir: str = './results',
        save_freq: int = 10000,
        eval_freq: int = 5000,
        n_eval_episodes: int = 10
    ) -> Dict:
        """
        Train the PPO agent
        
        Args:
            total_timesteps: Total training timesteps
            log_dir: Directory for logs
            model_save_dir: Directory for saving models
            save_freq: Frequency of model saving
            eval_freq: Frequency of evaluation
            n_eval_episodes: Number of evaluation episodes
            
        Returns:
            Training information dictionary
        """
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=model_save_dir,
            name_prefix='ppo_cellfree'
        )
        
        eval_callback = EvalCallback(
            self.env,
            best_model_save_path=model_save_dir,
            log_path=log_dir,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False
        )
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
        
        # Save final model
        final_model_path = os.path.join(model_save_dir, 'ppo_cellfree_final')
        self.model.save(final_model_path)
        
        return {
            'total_timesteps': total_timesteps,
            'final_model_path': final_model_path
        }
    
    def evaluate(
        self,
        n_episodes: int = 100,
        deterministic: bool = True
    ) -> Dict:
        """
        Evaluate the trained agent
        
        Args:
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic policy
            
        Returns:
            Evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        energy_efficiencies = []
        qos_satisfactions = []
        avg_rates = []
        
        for episode in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                # Collect metrics
                if 'energy_efficiency' in info:
                    energy_efficiencies.append(info['energy_efficiency'])
                if 'qos_satisfaction' in info:
                    qos_satisfactions.append(info['qos_satisfaction'])
                if 'avg_rate_mbps' in info:
                    avg_rates.append(info['avg_rate_mbps'])
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_energy_efficiency': np.mean(energy_efficiencies) if energy_efficiencies else 0,
            'mean_qos_satisfaction': np.mean(qos_satisfactions) if qos_satisfactions else 0,
            'mean_rate_mbps': np.mean(avg_rates) if avg_rates else 0
        }
    
    def predict(self, observation, deterministic: bool = True):
        """Make prediction given observation"""
        return self.model.predict(observation, deterministic=deterministic)
    
    def load(self, path: str):
        """Load model from file"""
        self.model = PPO.load(path, env=self.env)
    
    def save(self, path: str):
        """Save model to file"""
        self.model.save(path)
