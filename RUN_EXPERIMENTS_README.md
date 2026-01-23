# Unified Reward Function - Complete Experiment Pipeline

## Overview

This automated pipeline trains all 11 scenarios with the new unified reward function and generates comprehensive analysis reports.

## What the Script Does

### Phase 1: Training (6-12 hours total)
- Trains all 11 scenarios sequentially
- Saves models and training logs
- Total timesteps range: 100k-250k per scenario

### Phase 2: Network Performance Analysis
- Evaluates each trained model (100 episodes)
- Compares RL agent vs baselines (Nearest Max, Equal All, Load Balance, WMMSE)
- Generates performance plots

### Phase 3: Behavior Analysis
- Analyzes agent's action distribution
- Generates action heatmaps, power/clustering distributions
- Analyzes QoS performance (user rates, violations, fairness)
- Exports JSON data for LaTeX tables

## Usage

### Basic Usage
```bash
bash run_all_experiments.sh
```

### Run in Background (Recommended for long runs)
```bash
nohup bash run_all_experiments.sh > pipeline_output.log 2>&1 &

# Monitor progress
tail -f pipeline_output.log
```

### Run with tmux (Best for remote sessions)
```bash
tmux new -s experiments
bash run_all_experiments.sh
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t experiments
```

## Output Structure

After completion, results are organized in:

```
results_unified_reward_YYYYMMDD_HHMMSS/
├── PIPELINE_SUMMARY.txt                    # Overall summary
│
├── scenario_1_training.log                 # Training logs
├── scenario_1_model_path.txt               # Saved model location
├── scenario_1_network_analysis.log         # Performance analysis log
├── scenario_1_network_plots/               # Baseline comparison plots
│   ├── comparison_ee.png
│   ├── comparison_rates.png
│   ├── comparison_power.png
│   └── ...
├── scenario_1_behavior_analysis.log        # Behavior analysis log
├── scenario_1_behavior/                    # Action & QoS analysis
│   ├── action_distribution_histogram.png
│   ├── power_and_clustering_distribution.png
│   ├── action_heatmap_2d.png
│   ├── user_rate_distribution.png
│   ├── user_rate_cdf.png
│   ├── user_rate_boxplot.png
│   └── behavior_analysis.json              # Data for LaTeX tables
│
... (repeated for scenarios 2-11)
```

## Scenario List

| # | Name | APs | Users | Training Time (Est.) |
|---|------|-----|-------|----------------------|
| 1 | Sweet Spot Balanced | 36 | 10 | ~2 hours |
| 2 | Fix Rate 25AP | 25 | 10 | ~1.5 hours |
| 3 | Scalability High Int | 36 | 20 | ~2.5 hours |
| 4 | Massive MIMO 64AP | 64 | 10 | ~2 hours |
| 5 | QoS Max Speed | 36 | 10 | ~1.5 hours |
| 6 | Sparse Network 10AP | 10 | 5 | ~1 hour |
| 7 | MIMO Collocated 16AP | 16 | 10 | ~2 hours |
| 8 | MIMO Crowded 264 Users | 64 | 264 | ~2 hours |
| 9 | Low Load Green | 36 | 5 | ~1.5 hours |
| 10 | Eco Mode Aggressive | 36 | 10 | ~2 hours |
| 11 | Performance Mode | 36 | 10 | ~2 hours |

**Total Estimated Time**: 6-12 hours (sequential execution)

## Monitoring Progress

### Check Current Status
```bash
# View pipeline output
tail -f pipeline_output.log

# Check which scenarios are training
ps aux | grep train_agent.py

# Monitor GPU/CPU usage
htop  # or nvidia-smi for GPU
```

### Check TensorBoard During Training
```bash
tensorboard --logdir=./tensorboard_unified_YYYYMMDD_HHMMSS
# Open browser: http://localhost:6006
```

## Expected Results

### Before (Old Reward - EE Dominated)
- **Sweet Spot**: ~30 Mbps, EE dominates
- **Crowded**: Agent fails (ignores throughput)
- **Low Load**: Wastes energy (too many APs active)

### After (New Unified Reward)
- **Sweet Spot**: 40-50 Mbps, balanced EE/throughput
- **Crowded**: All 264 users satisfied, fair distribution
- **Low Load**: Only 6-8 APs active (sleep mode works)
- **Performance Mode**: 50-80 Mbps (beats Equal Power)
- **Eco Mode**: 40% power savings while maintaining QoS

## Troubleshooting

### Training Fails
```bash
# Check log for specific scenario
cat results_unified_reward_*/scenario_X_training.log

# Common issues:
# - Out of memory: Reduce batch_size in YAML
# - NaN/Inf values: Check reward scaling
# - CUDA errors: Check GPU availability
```

### Analysis Fails
```bash
# Check if model was saved
ls experiments/exp_*/models/ppo_cellfree_final.zip

# Check analysis logs
cat results_unified_reward_*/scenario_X_network_analysis.log
cat results_unified_reward_*/scenario_X_behavior_analysis.log
```

### Run Specific Scenario Only
```bash
# Train only Scenario 1
python src/train_agent.py \
  --config configs/ppo_scenarios/1_sweet_spot_balanced.yaml \
  --agent ppo

# Then analyze manually
python src/analyze_network.py --mode evaluate \
  --model experiments/exp_*/models/ppo_cellfree_final \
  --episodes 100 --num-aps 36 --num-users 10

python src/analyze_behavior.py \
  --model experiments/exp_*/models/ppo_cellfree_final \
  --config configs/ppo_scenarios/1_sweet_spot_balanced.yaml \
  --episodes 100 --save_dir ./scenario_1_behavior
```

## Post-Processing

### Generate Comparison Plots
After all experiments complete, compare scenarios:

```bash
# Compare all scenarios side-by-side
python src/compare_all_scenarios.py \
  --results_dir results_unified_reward_YYYYMMDD_HHMMSS

# Generate LaTeX tables from JSON
python src/generate_latex_tables.py \
  --results_dir results_unified_reward_YYYYMMDD_HHMMSS
```

### Archive Results
```bash
# Compress results for backup
tar -czf unified_reward_results_YYYYMMDD.tar.gz \
  results_unified_reward_YYYYMMDD_HHMMSS/

# Upload to cloud/server
# rsync -avz results_unified_reward_* user@server:/backup/
```

## Next Steps

1. **Review Results**: Check `PIPELINE_SUMMARY.txt` for overview
2. **Analyze Failures**: If any scenarios failed, check logs
3. **Compare with Baselines**: Review network_plots for each scenario
4. **Extract Insights**: Use behavior JSON files for thesis/paper
5. **Generate Paper Figures**: Use comparison plots for publications

## Key Files for Thesis

- **Performance**: `scenario_X_network_plots/comparison_*.png`
- **Behavior**: `scenario_X_behavior/action_heatmap_2d.png`
- **QoS Analysis**: `scenario_X_behavior/user_rate_cdf.png`
- **Data Tables**: `scenario_X_behavior/behavior_analysis.json`
- **Summary Stats**: `PIPELINE_SUMMARY.txt`

## Contact

For issues or questions about the unified reward implementation:
- Check: `/Users/bengi/.claude/plans/structured-growing-wolf.md`
- Review: Implementation details in plan file
