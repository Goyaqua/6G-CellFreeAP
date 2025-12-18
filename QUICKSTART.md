# Quick Start Guide
## CENG505: Cell-Free Network RL Project

### Installation

1. **Create Virtual Environment**
```bash
cd ceng505_cellfree_rl
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Verify Installation**
```bash
python -c "import sionna; import tensorflow; import gymnasium; print('✓ All packages installed successfully!')"
```

### Quick Demo

Run the demo to verify everything works:
```bash
cd src
python demo.py
```

This will:
- Create a cell-free network
- Visualize the topology
- Test baseline strategies
- Display performance metrics

### Running Experiments

#### 1. Evaluate Baseline Strategies

```bash
cd src
python run_baseline.py --config ../configs/default.yaml --episodes 100
```

Expected output: Performance comparison of 5 baseline strategies

#### 2. Train RL Agent (DQN)

```bash
cd src
python train_agent.py --agent dqn --timesteps 50000
```

Training options:
- `--agent`: Choose `dqn` or `ppo`
- `--timesteps`: Total training steps (default: 100000)
- `--config`: Custom config file

Monitor training with TensorBoard:
```bash
tensorboard --logdir experiments/exp_YYYYMMDD_HHMMSS/tensorboard
```

#### 3. Train RL Agent (PPO) for Continuous Actions

First, modify `configs/default.yaml`:
```yaml
environment:
  action_type: 'continuous'  # Change from 'discrete'
```

Then train:
```bash
cd src
python train_agent.py --agent ppo --timesteps 50000
```

#### 4. Evaluate Trained Model

```bash
cd src
python evaluate.py \
    --model ../experiments/exp_YYYYMMDD_HHMMSS/models/dqn_cellfree_final \
    --agent dqn \
    --episodes 100 \
    --compare_baselines
```

This will:
- Evaluate your trained agent
- Compare with baseline strategies
- Generate comparison plots
- Calculate performance improvements

### Using Jupyter Notebook

```bash
cd notebooks
jupyter notebook analysis.ipynb
```

The notebook includes:
- Network setup and visualization
- Channel analysis
- Baseline strategy comparison
- SINR distribution analysis
- Energy efficiency vs QoS trade-off plots

### Project Structure Overview

```
ceng505_cellfree_rl/
├── src/
│   ├── network/              # Sionna-based network simulation
│   │   └── cellfree_network.py
│   ├── environment/          # Gymnasium RL environment
│   │   └── cellfree_env.py
│   ├── agents/               # RL agents + baselines
│   │   ├── dqn_agent.py
│   │   ├── ppo_agent.py
│   │   └── baselines.py
│   ├── utils/                # Plotting and utilities
│   │   └── plotting.py
│   ├── demo.py               # Quick demo
│   ├── run_baseline.py       # Baseline evaluation
│   ├── train_agent.py        # RL training
│   └── evaluate.py           # Model evaluation
├── configs/
│   └── default.yaml          # Configuration file
├── notebooks/
│   └── analysis.ipynb        # Jupyter notebook
└── requirements.txt          # Dependencies
```

### Configuration

Edit `configs/default.yaml` to customize:

**Network Parameters:**
- `num_aps`: Number of Access Points
- `num_users`: Number of users
- `area_size`: Coverage area (meters)
- `max_power_per_ap`: Maximum power per AP (Watts)

**Training Parameters:**
- `total_timesteps`: Total training steps
- `learning_rate`: Agent learning rate
- `batch_size`: Batch size for training

**Environment Parameters:**
- `qos_min_rate_mbps`: Minimum QoS requirement
- `episode_length`: Steps per episode
- `action_type`: 'discrete' or 'continuous'

### Common Issues

**1. CUDA/GPU Issues**
If you see CUDA errors, TensorFlow will automatically fall back to CPU. For GPU support:
```bash
pip install tensorflow[and-cuda]
```

**2. Import Errors**
Make sure you're in the `src/` directory when running scripts:
```bash
cd src
python demo.py
```

**3. Sionna Installation**
If Sionna installation fails:
```bash
pip install --upgrade pip setuptools wheel
pip install sionna --no-cache-dir
```

### Expected Results

**Baseline Strategies** (100 episodes):
- Energy Efficiency: 10^6 - 10^7 bits/Joule
- QoS Satisfaction: 60-95%
- Average Rate: 8-15 Mbps

**RL Agent** (after training):
- Should outperform baselines in energy efficiency
- Should maintain high QoS satisfaction
- Training time: ~30-60 minutes for 50K steps (CPU)

### Next Steps for Your Project

1. **Experiment with different configurations**
   - Change number of APs/users
   - Adjust QoS requirements
   - Try different power levels

2. **Analyze results**
   - Use the Jupyter notebook for detailed analysis
   - Generate plots for your report
   - Compare convergence behavior

3. **Write your report**
   - Use results from `results/` directory
   - Include plots from experiments
   - Discuss trade-offs and improvements

### Getting Help

- Check the README.md for detailed documentation
- Look at code comments in source files
- Run scripts with `--help` flag for options

### For Your CENG505 Report

Key sections to include:
1. **Introduction**: Cell-free networks and RL motivation
2. **Methodology**: Network model, RL formulation, Sionna usage
3. **Experiments**: Baseline comparison, RL training results
4. **Results**: Performance metrics, plots, analysis
5. **Conclusion**: Findings and future work

Good luck with your project! 🚀
