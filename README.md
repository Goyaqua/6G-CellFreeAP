# AI-Driven Energy-Efficient Resource Allocation in 6G Cell-Free Networks

Deep Reinforcement Learning (PPO) framework for energy-efficient power control in Cell-Free Massive MIMO systems using NVIDIA Sionna.

## Overview

This project implements a **unified, scalable reward function** for Cell-Free network resource allocation that adapts to diverse topologies (10-64 APs, 5-264 users) through configurable weights. The agent learns to maximize energy efficiency while maintaining strict QoS requirements.

## Key Features

- **Unified Reward Function**: Multi-objective optimization (throughput, power, QoS, fairness)
- **Topology-Agnostic**: Single framework adapts to 11 diverse scenarios
- **Realistic Physical Layer**: NVIDIA Sionna simulation (Rayleigh fading, path loss, SINR)
- **Surgical Power Control**: Agent learns deep sleep mode for redundant APs
- **PPO Agent**: Proximal Policy Optimization with Actor-Critic architecture
- **Baseline Comparisons**: EPA, Nearest-AP, Load Balance, WMMSE

## Quick Start

### Installation

```bash
# Clone repository
git clone [your-repo-url]
cd ceng505_cellfree_rl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Train Single Scenario

```bash
# Train Sweet Spot scenario (36 APs, 10 users)
python src/train_agent.py \
  --config configs/ppo_scenarios/1_sweet_spot_balanced.yaml \
  --agent ppo
```

### Run All Experiments

```bash
# Train all 11 scenarios + analysis (6-12 hours)
bash run_all_experiments.sh

# Monitor progress
tail -f pipeline_output.log
```

### Evaluate Trained Model

```bash
# Network performance analysis
python src/analyze_network.py --mode evaluate \
  --model experiments/exp_*/models/ppo_cellfree_final \
  --episodes 100 --num-aps 36 --num-users 10

# Behavior analysis
python src/analyze_behavior.py \
  --model experiments/exp_*/models/ppo_cellfree_final \
  --config configs/ppo_scenarios/1_sweet_spot_balanced.yaml \
  --episodes 100
```

## Project Structure

```
ceng505_cellfree_rl/
├── src/
│   ├── network/           # Sionna-based physical layer simulation
│   │   └── cellfree_network.py  # SINR, channel model, power consumption
│   ├── environment/       # Gymnasium RL environment
│   │   └── cellfree_env.py      # Unified reward function
│   ├── agents/            # Baseline heuristics (EPA, Nearest, WMMSE)
│   ├── train_agent.py     # PPO training script
│   ├── analyze_network.py # Performance evaluation
│   └── analyze_behavior.py# Action distribution analysis
├── configs/
│   └── ppo_scenarios/     # 11 scenario configurations
├── run_all_experiments.sh # Automated pipeline
└── RUN_EXPERIMENTS_README.md  # Detailed pipeline documentation
```

## Unified Reward Function

```
reward = α × log_sum_rate - β × normalized_power - γ × qos_penalty - δ × fairness_penalty
```

### Scenario Weights

| Scenario | APs | Users | α (rate) | β (power) | γ (QoS) | δ (fairness) |
|----------|-----|-------|----------|-----------|---------|--------------|
| 1. Sweet Spot | 36 | 10 | 1.0 | 1.0 | 2.0 | 0.0 |
| 3. High Interference | 36 | 20 | 1.0 | 0.8 | 3.0 | 1.5 |
| 8. Crowded | 64 | 264 | 1.0 | 0.5 | 10.0 | 2.0 |
| 9. Low Load Green | 36 | 5 | 0.5 | 2.0 | 2.0 | 0.0 |
| 10. Eco Mode | 36 | 10 | 0.5 | 3.0 | 2.0 | 0.0 |
| 11. Performance Mode | 36 | 10 | 1.5 | 0.3 | 2.0 | 0.0 |

See [RUN_EXPERIMENTS_README.md](RUN_EXPERIMENTS_README.md) for all 11 scenarios.

## Results

### Energy Efficiency Gains

PPO agent achieves **8-69x** energy efficiency improvement over Equal Power Allocation (EPA):

- **Sweet Spot**: 76.8 Mbit/J (+716% vs EPA)
- **Crowded (264 users)**: 391 Mbit/J (+6820% vs EPA)
- **Eco Mode**: 83.1 Mbit/J (+783% vs EPA)

### Key Insights

1. **Surgical Power Control**: Agent activates only ~9 APs in crowded scenario (86% reduction from 64 APs)
2. **Denominator Effect**: Massive EE gains stem from eliminating circuit power (>80% of total consumption)
3. **100% QoS Satisfaction**: All users maintain >5 Mbps across all scenarios
4. **Topology Adaptation**: Same reward framework works for 10-64 APs, 5-264 users

## Configuration

Each YAML file defines scenario-specific parameters:

```yaml
environment:
  qos_min_rate_mbps: 5.0
  qos_weight: 2.0                  # QoS penalty weight (γ)
  rate_weight: 1.0                 # Log sum-rate importance (α)
  power_penalty_weight: 1.0        # Normalized power penalty (β)
  fairness_weight: 0.0             # Jain's Index penalty (δ)
  episode_length: 50
  action_type: 'discrete'
  num_power_levels: 5
  randomize_circuit_power: true    # Adaptive circuit power
  circuit_power_range: [0.1, 0.5]  # 100-500mW per AP

training:
  total_timesteps: 200000
  ppo:
    learning_rate: 0.0003
    n_steps: 2048
    batch_size: 256
    n_epochs: 10
```

## System Model

- **Downlink Cell-Free Massive MIMO**
- **Non-coherent Joint Transmission** (power domain combining)
- **Channel Model**: Rayleigh fading + path loss
- **SINR Formula**: γ_k = Σ(m∈M_k) P_m|g_mk|² / (interference + noise)
- **Power Consumption**: P_total = Σ P_m + P_circ × N_active
- **Deep Sleep**: Inactive APs consume zero power

## State Space (371 dimensions)

```python
state = [
    channel_gains,      # M×K flattened (360 for 36 APs, 10 users)
    qos_requirements,   # K users (10)
    circuit_power       # 1 scalar
]
```

## Action Space (20 discrete actions)

- **Power Levels**: {0, 0.25, 0.5, 0.75, 1.0}
- **Clustering**: {Nearest-Only, Top-3, Top-50%, All-Active}
- **Total**: 5 × 4 = 20 actions

## Performance Metrics

1. **Spectral Efficiency**: Sum-Rate (Mbps)
2. **Energy Efficiency**: Sum-Rate / Total Power (Mbit/J)
3. **QoS Satisfaction**: % of users ≥ 5 Mbps
4. **Fairness**: Jain's Index
5. **Active APs**: Sleep mode efficiency

## Citation

```bibtex
@article{ozturk2025cellfree,
  title={AI-Driven Energy-Efficient Resource Allocation in 6G Cell-Free Networks},
  author={Öztürk, Bengisu},
  journal={CENG505 Project},
  year={2025}
}
```

## References

- [NVIDIA Sionna](https://nvlabs.github.io/sionna/) - Physical layer simulation
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - PPO implementation
- [Gymnasium](https://gymnasium.farama.org/) - RL environment standard

## License

Academic use only - MIT License

## Contact

Bengisu Öztürk - beozturk@std.iyte.edu.tr
Izmir Institute of Technology
