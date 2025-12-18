# CENG505: AI-Driven Energy-Efficient Resource Allocation in 6G Cell-Free Networks

## Project Overview

This project implements Reinforcement Learning (RL) based resource allocation for 6G Cell-Free Massive MIMO networks using NVIDIA Sionna. The goal is to optimize energy efficiency while maintaining Quality of Service (QoS) requirements.

## Features

- **Realistic Channel Modeling**: Using Sionna's advanced channel models
- **RL-based Optimization**: Deep Q-Network (DQN) and PPO agents
- **Baseline Comparisons**: Nearest-AP, Max-Power, Random strategies
- **Performance Metrics**: Energy Efficiency, SINR, Data Rate, QoS satisfaction

## Project Structure

```
ceng505_cellfree_rl/
├── src/
│   ├── network/          # Network simulation classes
│   ├── environment/      # Gymnasium RL environment
│   ├── agents/           # RL agent implementations
│   └── utils/            # Helper functions
├── configs/              # Configuration files
├── results/              # Simulation results
├── logs/                 # Training logs
├── notebooks/            # Jupyter notebooks for analysis
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

### Prerequisites
- Python 3.12+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Setup

1. Clone or download this project:
```bash
cd ceng505_cellfree_rl
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Run Baseline Simulations
```bash
python src/run_baseline.py --config configs/default.yaml
```

### 2. Train RL Agent
```bash
python src/train_agent.py --agent dqn --episodes 1000
```

### 3. Evaluate Performance
```bash
python src/evaluate.py --model results/dqn_model.zip
```

### 4. Visualize Results
```bash
python src/visualize_results.py --results results/comparison.npz
```

## Configuration

Edit `configs/default.yaml` to change:
- Network parameters (number of APs, users)
- RL hyperparameters
- Channel model settings
- Training parameters

## Results

Results will be saved in:
- `results/`: Model checkpoints and performance data
- `logs/`: TensorBoard logs for training visualization
- `notebooks/`: Analysis and visualization notebooks

## Usage Examples

### Example 1: Custom Network Configuration
```python
from src.network.cellfree_network import CellFreeNetworkSionna

network = CellFreeNetworkSionna(
    num_aps=25,
    num_users=10,
    num_antennas_per_ap=4,
    area_size=500
)
```

### Example 2: Train Custom Agent
```python
from src.agents.dqn_agent import DQNAgent
from src.environment.cellfree_env import CellFreeEnv

env = CellFreeEnv(config='configs/default.yaml')
agent = DQNAgent(env)
agent.train(episodes=1000)
```

## Methodology

### Phase 1: Network Modeling
- Cell-free network topology
- Sionna channel models (path loss, shadowing, fading)
- SINR and rate calculations

### Phase 2: RL Strategy
- State: Channel gains, user QoS requirements
- Action: Power allocation, AP-user association
- Reward: Energy efficiency with QoS penalty

### Phase 3: Evaluation
- Compare against baselines
- Analyze convergence
- Measure adaptability

## Performance Metrics

1. **Energy Efficiency (EE)**: bits/Joule
2. **Average Data Rate**: Mbps per user
3. **QoS Satisfaction Rate**: % of users meeting requirements
4. **SINR**: Signal-to-Interference-plus-Noise Ratio

## References

See `docs/references.bib` for full bibliography.

## License

MIT License - Academic use only

## Contact

For questions or issues, contact: [your email]

## Acknowledgments

- NVIDIA Sionna team for the wireless simulation framework
- OpenAI Gymnasium for RL environment standards
- Stable-Baselines3 for RL implementations
