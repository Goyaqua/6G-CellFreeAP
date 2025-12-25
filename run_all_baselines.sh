#!/bin/bash
# Re-run all baseline evaluations with corrected circuit power calculation
# Bug fix: baselines now correctly calculate circuit power using ap_association

set -e  # Exit on error

echo "=========================================="
echo "Re-running Baseline Evaluations"
echo "Bug Fixed: Circuit power now uses active APs only"
echo "=========================================="
echo ""

# Results directory
RESULTS_BASE="results/baselines_corrected"
mkdir -p "$RESULTS_BASE"

# Number of episodes
EPISODES=100

echo "Configuration:"
echo "  - Episodes per scenario: $EPISODES"
echo "  - Results directory: $RESULTS_BASE"
echo ""

# Scenario 1: Sweet Spot (36 AP, 10 users) - MAIN MODEL
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Scenario 1: Sweet Spot Balanced (36 AP, 10 users)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/run_baseline.py \
  --config configs/ppo_scenarios/1_sweet_spot_balanced.yaml \
  --episodes $EPISODES \
  --save_dir "$RESULTS_BASE/scenario1_36ap_10u"
echo ""

# Scenario 2: Fix Rate (25 AP, 10 users) - COMPARISON WITH OLD MODEL
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Scenario 2: Fix Rate Balanced (25 AP, 10 users)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/run_baseline.py \
  --config configs/ppo_scenarios/2_fix_rate_balanced_25ap.yaml \
  --episodes $EPISODES \
  --save_dir "$RESULTS_BASE/scenario2_25ap_10u"
echo ""

# Scenario 3: High Interference (36 AP, 20 users) - SCALABILITY
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Scenario 3: Scalability High Interference (36 AP, 20 users)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/run_baseline.py \
  --config configs/ppo_scenarios/3_scalability_high_int.yaml \
  --episodes $EPISODES \
  --save_dir "$RESULTS_BASE/scenario3_36ap_20u"
echo ""

# Scenario 4: Massive MIMO (64 AP, 10 users) - GENERALIZATION
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Scenario 4: Massive MIMO (64 AP, 10 users)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/run_baseline.py \
  --config configs/ppo_scenarios/4_massive_mimo_64ap.yaml \
  --episodes $EPISODES \
  --save_dir "$RESULTS_BASE/scenario4_64ap_10u"
echo ""

# Scenario 5: QoS Max Speed (36 AP, 10 users) - UPPER BOUND
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Scenario 5: QoS Max Speed (36 AP, 10 users)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/run_baseline.py \
  --config configs/ppo_scenarios/5_qos_max_speed_36ap.yaml \
  --episodes $EPISODES \
  --save_dir "$RESULTS_BASE/scenario5_36ap_10u_qos"
echo ""

# Scenario 6: Sparse Network (10 AP, 5 users) - ROBUSTNESS
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Scenario 6: Sparse Network (10 AP, 5 users)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/run_baseline.py \
  --config configs/ppo_scenarios/6_sparse_network_10ap.yaml \
  --episodes $EPISODES \
  --save_dir "$RESULTS_BASE/scenario6_10ap_5u"
echo ""

# Summary
echo "=========================================="
echo "✅ All Baseline Evaluations Completed!"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_BASE/"
echo ""
echo "Directory structure:"
tree -L 2 "$RESULTS_BASE" 2>/dev/null || ls -R "$RESULTS_BASE"
echo ""
echo "Next steps:"
echo "1. Compare with PPO results: python src/evaluate.py --model <ppo_model_path>"
echo "2. Generate comparison plots"
echo "3. Update thesis figures with corrected baseline values"
echo ""
echo "Expected changes:"
echo "  - nearest_max EE: +150% improvement ⬆️"
echo "  - equal_all EE: +25% improvement ⬆️"
echo "  - Reason: Circuit power now counts only ACTIVE APs"
