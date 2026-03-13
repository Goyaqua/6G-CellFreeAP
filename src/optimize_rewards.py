import os
import sys

# Add project root to path so 'src.' imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import optuna
import numpy as np
import tensorflow as tf
from optuna_integration import TensorBoardCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from environment.cellfree_env import CellFreeEnv
from network.cellfree_network import CellFreeNetworkSionna

class OptunaEvalCallback(BaseCallback):
    """
    Custom callback for evaluating during Optuna optimization.
    It stops training early if the model is performing terribly to save time.
    """
    def __init__(self, eval_env, trial, eval_freq=2000, n_eval_episodes=5, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.trial = trial
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            rewards = []
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                episode_reward = 0.0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    # For DummyVecEnv in sb3, step returns (obs, rewards, dones, infos)
                    obs, reward, done, info = self.eval_env.step(action)
                    done = done[0]
                    episode_reward += reward[0]
                rewards.append(episode_reward)
            
            mean_reward = np.mean(rewards)
            self.trial.report(mean_reward, self.n_calls)
            
            # Print live progress
            print(f"  [Trial {self.trial.number}] Step {self.n_calls} - Mean Reward: {mean_reward:.2f}")
            
            # Prune trial if it's unpromising
            if self.trial.should_prune():
                raise optuna.TrialPruned()
                
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                
        return True

def load_base_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_model(model, env, num_episodes=10):
    """Evaluate a trained model to see its true metrics (EE, Rate, QoS)"""
    all_ee = []
    all_rate = []
    all_qos = []
    all_rewards = []
    
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        
        ep_ee = []
        ep_rates = []
        ep_qos = []
        ep_active_aps = []
        ep_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # DummyVecEnv in sb3 returns 4 values for step: (obs, rewards, dones, infos)
            obs, reward, done, info = env.step(action)
            
            # DummyVecEnv returns arrays/lists for parallel environments
            done = done[0]
            ep_reward += reward[0]
            info_dict = info[0]
            
            if 'energy_efficiency' in info_dict:
                ep_ee.append(info_dict['energy_efficiency'])
            if 'avg_rate_mbps' in info_dict:
                ep_rates.append(info_dict['avg_rate_mbps'])
            if 'qos_satisfaction' in info_dict:
                ep_qos.append(info_dict['qos_satisfaction'])
            if 'active_aps' in info_dict:
                ep_active_aps.append(info_dict['active_aps'])
            
        all_rewards.append(ep_reward)
        all_ee.append(np.mean(ep_ee) if ep_ee else 0)
        all_rate.append(np.mean(ep_rates) if ep_rates else 0)
        all_qos.append(np.mean(ep_qos) if ep_qos else 0)
        
    return {
        'mean_reward': np.mean(all_rewards),
        'mean_ee': np.mean(all_ee),
        'mean_rate': np.mean(all_rate),
        'mean_qos': np.mean(all_qos),
        'mean_active_aps': np.mean(ep_active_aps) if ep_active_aps else 64
    }

def optimize_agent(trial, base_config, args):
    """
    Objective function for Optuna.
    Samples reward weights, trains a short PPO model, and evaluates its true performance.
    Returns a combined score to maximize.
    """
    # Adapt the search bounds based on the scenario
    config_name = os.path.basename(args.config).lower()
    
    # Alpha: (Sum-Rate is log scale, usually yields ~20-60 points) -> Needs small-to-mid weights
    alpha = trial.suggest_float("reward_weight_sum_rate", 0.01, 50.0, log=True)
    
    # Beta: (Power is normalized [0, 1] scale) -> Needs massive weights to be felt!
    beta = trial.suggest_float("reward_weight_power_cost", 10.0, 500.0, log=True)
    
    # Gamma: (QoS penalty is absolute Mbps lost, e.g. 5.0) -> Needs tiny weights or it destroys the agent's sanity
    gamma = trial.suggest_float("reward_weight_qos_penalty", 0.001, 5.0, log=True)
    
    # Delta: (Fairness is [0, 1] scale) 
    delta = trial.suggest_float("reward_weight_fairness", 0.0, 50.0)

    print(f"\n" + "-"*60)
    print(f"🚀 STARTING TRIAL {trial.number}")
    print(f"   Testing Weights -> Alpha (Rate): {alpha:.2f}, Beta (Power): {beta:.2f}, Gamma (QoS): {gamma:.2f}, Delta (Fairness): {delta:.2f}")
    print("-"*60)

    # 2. Setup Environment with Sampled Weights
    env_config = base_config['environment']
    net_config = base_config['network']
    
    def make_env():
        # Inject the sampled weights into the environment's reward function
        # We subclass CellFreeEnv dynamically to override the weights 
        # (since they are currently hardcoded in the config loading)
        
        class OptunaCellFreeEnv(CellFreeEnv):
            def _calculate_reward(self, rates, power_allocation, ap_association):
                rates_np = rates.numpy()[0]
                user_rates_mbps = rates_np / 1e6
                
                log_sum_rate = np.sum(np.log1p(user_rates_mbps))
                
                qos_violations = np.maximum(0, self.qos_min_rate / 1e6 - user_rates_mbps)
                qos_penalty = np.sum(qos_violations)
                
                active_aps_mask = np.sum(ap_association, axis=1) > 0
                total_tx_power = np.sum(power_allocation[active_aps_mask])
                active_aps = np.sum(active_aps_mask)
                total_circuit_power = active_aps * self.network.circuit_power_per_ap
                total_power = total_tx_power + total_circuit_power
                
                # Normalize power back to [0, 1] so beta can be logically bounded across scales
                max_possible_power = self.num_aps * (self.network.max_power_per_ap + self.network.circuit_power_per_ap)
                normalized_power = total_power / max_possible_power
                
                sum_rates = np.sum(user_rates_mbps)
                sq_sum_rates = np.sum(user_rates_mbps ** 2)
                jains_index = (sum_rates ** 2) / (self.num_users * sq_sum_rates + 1e-10)
                fairness_penalty = 1.0 - jains_index
                
                # Apply Optuna's sampled weights
                reward = (alpha * log_sum_rate) - (beta * normalized_power) - (gamma * qos_penalty) - (delta * fairness_penalty)
                return float(reward)
                
        return OptunaCellFreeEnv(
            config_path=args.config,
            num_aps=net_config['num_aps'],
            num_users=net_config['num_users'],
            qos_min_rate_mbps=env_config['qos_min_rate_mbps'],
            episode_length=env_config['episode_length'],
            action_type=env_config.get('action_type', 'discrete')
        )

    # We use a single env for training to save compute during hyperparameter tuning
    train_env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    # 3. Create PPO Agent
    # For speed during HPO, we use a smaller network and fewer timesteps
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=1024,
        batch_size=256,
        n_epochs=4,
        learning_rate=3e-4,
        verbose=0,
        policy_kwargs=dict(net_arch=[128, 128]) # Smaller net for fast tuning
    )

    # 4. Train the model
    eval_callback = OptunaEvalCallback(eval_env, trial, eval_freq=2000, n_eval_episodes=5)
    
    try:
        model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    except (AssertionError, ValueError) as e:
        # Sometimes divergent params cause NaN crashes
        print(f"Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned()

    print(f"  [Trial {trial.number}] Training finished. Evaluating true performance on 10 episodes...")
    
    # 5. Evaluate True Performance (Not just the arbitrary reward score)
    # We want to optimize the actual physics metrics
    metrics = evaluate_model(model, eval_env, num_episodes=10)
    
    # Clean up
    train_env.close()
    eval_env.close()
    
    # 6. Define the Target Metric
    # We want high Energy Efficiency, but ONLY if QoS is acceptable (e.g. >95%)
    # We create a composite score where QoS failure heavily penalizes the score
    
    qos = metrics['mean_qos']
    ee_mbit = metrics['mean_ee'] / 1e6 # Scale to make it comparable
    rate = metrics['mean_rate']
    active_aps_val = metrics.get('mean_active_aps', 64)
    
    # %95 üstü QoS'i tam başarı say (çarpan 1.0)
    # %95'ten aşağı düştükçe skoru yavaş yavaş, ama yumuşakça kırp
    # Örneğin QoS %85 ise çarpan ~0.89 olur, böylece model ceza yediği i̇çin tüm sistemi yıkmaz
    qos_multiplier = min(1.0, qos / 95.0)
        
    # Adapt the Objective combination based on the scenario
    # config_name is already defined above
    
    if "4_qos_priority" in config_name:
        # Scenario 4: YALNIZCA HIZ ve QoS. EE'yi tamamen çöpe atıyoruz. (100 Mbps = 100 puan)
        composite_score = (rate) * qos_multiplier
    elif "5_green_network" in config_name:
        # Scenario 5: YALNIZCA ENERJİ VERİMLİLİĞİ ve QoS. Hızı dahil bile etmiyoruz.
        composite_score = (ee_mbit) * qos_multiplier
    else:
        # Scenarios 1, 2, 3: Dengeli (Hem EE, hem Rate önemli)
        composite_score = (ee_mbit + rate) * qos_multiplier
    
    # Save the true physics metrics as user attributes so we can see them in the DB
    trial.set_user_attr("EE (Mbits/J)", ee_mbit)
    trial.set_user_attr("Rate (Mbps)", rate)
    trial.set_user_attr("QoS (%)", qos)
    trial.set_user_attr("Active APs", active_aps_val)
    
    print(f"✅ [Trial {trial.number}] Result: EE= {ee_mbit:.2f} Mbits/J | Rate= {rate:.2f} Mbps | QoS= {qos:.2f}% | APs= {active_aps_val:.1f} -> Score: {composite_score:.2f}\n")
    
    return composite_score


def main():
    parser = argparse.ArgumentParser(description='Optuna HPO for Reward Function Weights')
    parser.add_argument('--config', type=str, required=True, help='Path to base YAML config')
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials to run')
    parser.add_argument('--timesteps', type=int, default=30000, help='Timesteps per trial (keep small, e.g. 30k-50k for speed)')
    parser.add_argument('--study-name', type=str, default='cellfree_reward_optimization', help='Name of the Optuna study')
    args = parser.parse_args()

    # Disable TF logging spam
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    
    # Load the base config architecture
    base_config = load_base_config(args.config)

    print("="*80)
    print(f"STARTING REWARD FUNCTION OPTIMIZATION")
    print("="*80)
    print(f"Base Config: {args.config}")
    print(f"Trials: {args.trials}")
    print(f"Timesteps per trial: {args.timesteps}")
    print("Optimizing: Alpha (Rate), Beta (EE), Gamma (QoS), Delta (Fairness)\n")

    # Create Optuna study to maximize our composite score
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    # Run optimization
    study.optimize(lambda trial: optimize_agent(trial, base_config, args), n_trials=args.trials, show_progress_bar=True)

    # Print Results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    
    best_trial = study.best_trial
    
    print("\n🏆 BEST REWARD WEIGHTS FOUND:")
    print(f"  alpha (Sum-Rate)  = {best_trial.params['reward_weight_sum_rate']:.4f}")
    print(f"  beta (Power Cost) = {best_trial.params['reward_weight_power_cost']:.4f}")
    print(f"  gamma (QoS Pen.)  = {best_trial.params['reward_weight_qos_penalty']:.4f}")
    print(f"  delta (Fairness)  = {best_trial.params['reward_weight_fairness']:.4f}")
    
    print("\n📊 BEST PHYSICS PERFORMANCE (Validation):")
    print(f"  Energy Efficiency: {best_trial.user_attrs.get('EE (Mbits/J)', 0):.2f} Mbits/Joule")
    print(f"  Average Rate:      {best_trial.user_attrs.get('Rate (Mbps)', 0):.2f} Mbps")
    print(f"  QoS Satisfaction:  {best_trial.user_attrs.get('QoS (%)', 0):.2f} %")
    
    print("\nUpdate your YAML configs with these new weight values to beat the Nearest AP baseline!")

if __name__ == '__main__':
    main()
