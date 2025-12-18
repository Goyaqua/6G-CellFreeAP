# Discrete Action Space Improvement

## Problem Identified

**Original Implementation**:
- Only 5 discrete actions (power levels 0-4)
- All APs used the same power
- Fixed "nearest AP" association strategy
- Agent could NOT learn AP selection

**Limitations**:
- ❌ Cannot shut down specific APs
- ❌ Cannot learn load balancing
- ❌ Cannot optimize per-AP power
- ❌ Only learns global power control

## Solution: Factored Discrete Actions

### New Action Space
**Total Actions: 20** (up from 5)

**Factorization**: `action = power_idx × 4 + ap_strategy_idx`

### Power Strategies (5 options)
| Index | Power Level | Description |
|-------|-------------|-------------|
| 0 | 20% | Very Low - Maximum energy savings |
| 1 | 40% | Low - Good for sparse networks |
| 2 | 60% | Medium - Balanced |
| 3 | 80% | High - Good coverage |
| 4 | 100% | Maximum - Best performance |

### AP Selection Strategies (4 options)
| Index | Strategy | Active APs | Use Case |
|-------|----------|------------|----------|
| 0 | Nearest-only | 1 per user | **Energy efficient**, may drop coverage |
| 1 | Top-3 | 3 per user | **Balanced** - good cooperation |
| 2 | Top-50% | ~18 (for 36 APs) | Better performance, more power |
| 3 | All APs | All | **Maximum performance**, highest power |

### Example Actions

| Action | Power | AP Strategy | Description |
|--------|-------|-------------|-------------|
| 0 | 20% | Nearest-only | **Ultra Green** - min power, min APs |
| 4 | 20% | Top-3 | Green with cooperation |
| 10 | 60% | Top-3 | **Balanced** - moderate power & APs |
| 15 | 80% | All APs | High performance mode |
| 19 | 100% | All APs | **Maximum speed** - full power, all APs |

## Benefits

### ✅ Solves Original Problems:
1. **AP Selection**: Agent can now learn which AP strategy works best
2. **Energy Control**: 4 different AP activation levels (1, 3, ~18, 36)
3. **Power-Performance Tradeoff**: 20 combinations to explore
4. **Massive MIMO**: Can learn to use only 50% of 64 APs (Priority 4 goal!)

### ✅ Scientific Advantages:
- Agent can learn **when to shut down APs** (sparse vs dense strategies)
- Can discover **cooperation benefits** (top-3 vs nearest-only)
- Can trade **energy for performance** in principled way
- **Still computationally efficient** (20 actions vs 5^36)

## Code Changes

### 1. Action Space Definition
```python
# Before:
self.action_space = spaces.Discrete(5)

# After:
self.num_power_levels = 5
self.num_ap_strategies = 4
self.action_space = spaces.Discrete(20)  # 5 × 4
```

### 2. Action Decoding
```python
# Decode factored action
power_idx = action // self.num_ap_strategies
ap_strategy_idx = action % self.num_ap_strategies

power_allocation = self._apply_power_strategy(power_idx)
ap_association = self._apply_ap_strategy(ap_strategy_idx)
```

### 3. New Methods
- `_apply_power_strategy(power_idx)` - Returns power allocation array
- `_apply_ap_strategy(strategy_idx)` - Returns AP association matrix

## Impact on Existing Models

**⚠️ IMPORTANT**: This changes the action space!

### Already Trained Models:
- Old models trained with 5 actions **will NOT work** with new 20-action space
- Need to **retrain all models** from scratch

### Configuration Files:
- ✅ No changes needed - configs are compatible
- Training will automatically use new 20-action space

## Expected Results Improvement

### Before Fix:
- Agent only learns: "Use 20% power" vs "Use 100% power"
- All scenarios behave similarly
- Cannot achieve "AP shutdown" goal

### After Fix:
- Agent learns: "Nearest-only + 40% power" for sparse
- Agent learns: "Top-3 + 80% power" for balanced
- Agent learns: "All APs + 60% power" when high QoS needed
- **Can achieve Priority 4 goal**: Shut down 50% of APs in 64-AP scenario

## Recommendation

### ✅ Proceed with retraining:
All 6 PPO models should be retrained with the improved action space. The results will be **scientifically stronger** and more interesting.

### Expected Training Time:
Same as before - the action space is still small (20 vs 5), so training speed should be similar.

### Verification:
After training, check that models learn to use different AP strategies:
```bash
# Action distribution should NOT be uniform
tensorboard --logdir ./tensorboard
# Look for "actions/action_distribution"
```

---

**Status**: ✅ Implementation Complete
**Next Step**: Retrain all 6 PPO models with improved action space
**Expected Outcome**: Better results + ability to demonstrate AP selection learning
