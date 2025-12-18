# Training Plan: 9 Circuit Power-Adaptive Models

## 📋 Overview

All models use **circuit power randomization** during training (`randomize_circuit_power: true`, range: 100mW-500mW).

This ensures the agent learns to adapt to different circuit power costs.

---

## 🎯 Model Matrix

| # | Scenario | APs | Users | qos_weight | Timesteps | Buffer | Purpose |
|---|----------|-----|-------|------------|-----------|--------|---------|
| **Baseline** | **Standard (Green)** | 25 | 10 | 10 | 150k | 100k | **✅ Already Trained** |
| 1 | High Interference (Green) | 25 | **20** | 10 | **200k** | **150k** | Test interference management |
| 2 | Sparse Network (Green) | **10** | **5** | 10 | 150k | 100k | Test resource constraints |
| 3 | Massive MIMO (Green) | **50** | 10 | 10 | **200k** | **150k** | Test AP selection efficiency |
| 4 | Balanced (25AP) | 25 | 10 | **50** | 150k | 100k | Balanced energy-performance |
| 5 | QoS-Focused (25AP) | 25 | 10 | **100** | 150k | 100k | Performance priority |
| 6 | High Interference (Balanced) | 25 | **20** | **50** | **200k** | **150k** | Balanced in dense network |
| 7 | High Interference (QoS) | 25 | **20** | **100** | **200k** | **150k** | Performance in dense network |
| 8 | Massive MIMO (Balanced) | **50** | 10 | **50** | **200k** | **150k** | Balanced with many APs |
| 9 | Massive MIMO (QoS) | **50** | 10 | **100** | **200k** | **150k** | Performance with many APs |

---

## 🚀 Training Commands

### Group 1: Network Scale Variations (Green Reward)

```bash
# Model 1: High Interference (25 APs, 20 Users, Green)
python src/train_agent.py \
  --agent dqn \
  --config configs/scenarios/high_interference_green.yaml

# Model 2: Sparse Network (10 APs, 5 Users, Green)
python src/train_agent.py \
  --agent dqn \
  --config configs/scenarios/sparse_network_green.yaml

# Model 3: Massive MIMO (50 APs, 10 Users, Green)
python src/train_agent.py \
  --agent dqn \
  --config configs/scenarios/massive_mimo_green.yaml
```

### Group 2: Reward Strategy Variations (25 APs, 10 Users)

```bash
# Model 4: Balanced Reward
python src/train_agent.py \
  --agent dqn \
  --config configs/scenarios/balanced_reward_25ap.yaml

# Model 5: QoS-Focused Reward
python src/train_agent.py \
  --agent dqn \
  --config configs/scenarios/qos_focused_25ap.yaml
```

### Group 3: Combined Scenarios

```bash
# Model 6: High Interference + Balanced
python src/train_agent.py \
  --agent dqn \
  --config configs/scenarios/high_interference_balanced.yaml

# Model 7: High Interference + QoS-Focused
python src/train_agent.py \
  --agent dqn \
  --config configs/scenarios/high_interference_qos.yaml

# Model 8: Massive MIMO + Balanced
python src/train_agent.py \
  --agent dqn \
  --config configs/scenarios/massive_mimo_balanced.yaml

# Model 9: Massive MIMO + QoS-Focused
python src/train_agent.py \
  --agent dqn \
  --config configs/scenarios/massive_mimo_qos.yaml
```

---

## ⏱️ Estimated Training Times

| Timesteps | Estimated Time (M3 Pro) | Estimated Time (GPU) |
|-----------|------------------------|----------------------|
| 150,000 | ~4-6 hours | ~2-3 hours |
| 200,000 | ~6-8 hours | ~3-4 hours |

**Total training time (9 models):** ~50-70 hours on M3 Pro

---

## 📊 Expected Outcomes

### Green Reward (qos_weight=10)
- **Focus:** Energy Efficiency
- **Expected:** Low active APs (~7-8), high EE, moderate QoS satisfaction
- **Use case:** Energy-constrained scenarios

### Balanced Reward (qos_weight=50)
- **Focus:** Trade-off between energy and performance
- **Expected:** Medium active APs (~10-12), balanced EE and QoS
- **Use case:** Practical deployment scenarios

### QoS-Focused Reward (qos_weight=100)
- **Focus:** Performance (Rate & QoS)
- **Expected:** Higher active APs (~12-15), high QoS, lower EE
- **Use case:** Performance-critical applications

---

## 🔬 Testing After Training

After each model is trained, run:

```bash
# Replace MODEL_PATH with actual experiment path
MODEL_PATH="experiments/exp_YYYYMMDD_HHMMSS/models/dqn_cellfree_final"

# 1. Basic evaluation
python src/analyze_network.py --mode evaluate --model $MODEL_PATH --episodes 50

# 2. Circuit power sensitivity test
python src/analyze_network.py --mode circuit-power --model $MODEL_PATH --episodes 20

# 3. Adaptivity analysis
python src/verify_adaptivity.py --model $MODEL_PATH --episodes 100 --circuit-power 0.2
```

---

## 📝 Notes

- All models use **circuit power randomization** (100mW - 500mW)
- Training with `--agent dqn` (can also try `--agent ppo`)
- Models are saved every 15k-20k timesteps
- TensorBoard logs available in `./tensorboard/`
- Results saved in `./experiments/exp_YYYYMMDD_HHMMSS/`

---

**Created:** 2025-12-14
**Purpose:** Comprehensive circuit power-adaptive RL training across multiple scenarios
