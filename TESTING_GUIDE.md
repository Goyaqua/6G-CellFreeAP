# Cell-Free Network RL - Testing Guide

Bu dokümanda, projedeki tüm test scriptleri, komutları, çıktıları ve ne test ettikleri detaylı olarak açıklanmıştır.

---

## 🚀 Quick Reference - Tüm Komutlar

| # | Script | Temel Komut | Süre | Ne Test Ediyor? |
|---|--------|-------------|------|-----------------|
| 1 | **Demo** | `python src/demo.py` | ~30s | Sistem çalışıyor mu? |
| 2 | **Network Test** | `python src/network/cellfree_network.py` | ~10s | Sionna simulation doğru mu? |
| 3 | **Baselines** | `python src/agents/baselines.py` | ~2min | 5 baseline strateji karşılaştırma |
| 4 | **Circuit Power** | `python src/test_circuit_power.py` | ~3min | Circuit power etkisi (baselines) |
| 4b | **Circuit Power + RL** | `python src/test_circuit_power.py --rl-model MODEL_PATH` | ~5min | Circuit power + RL adaptasyonu |
| 5 | **AP Scaling** | `python src/analyze_ap_scaling.py` | ~5min | AP sayısının etkisi |
| 6 | **Train DQN** | `python src/train_agent.py --config CONFIG --timesteps 150000` | ~3-6 hours | RL agent eğitimi |
| 7 | **Evaluate** | `python src/evaluate.py --model MODEL --n-episodes 100` | ~10min | Comprehensive evaluation |
| 8 | **Quick Eval** | `python src/quick_eval.py` | ~1min | Hızlı model testi |
| 9 | **Adaptivity Check** | `python src/verify_adaptivity.py --model MODEL --episodes 100` | ~8min | Agent karar dağılımı analizi |

### 🎯 Training Başlamadan Önce Çalıştır (Sırasıyla):
```bash
# 1. System check
python src/demo.py

# 2. Baseline understanding
python src/agents/baselines.py

# 3. Circuit power effect
python src/test_circuit_power.py

# 4. Old RL model circuit power test
python src/test_circuit_power.py --rl-model experiments/exp_20251205_143919/models/dqn_cellfree_final

# 5. AP scaling analysis
python src/analyze_ap_scaling.py
```

### 🔥 Training Başlat:
```bash
# Circuit power-aware model eğitimi
python src/train_agent.py --config configs/circuit_power_adaptive.yaml --agent dqn --timesteps 150000
```

### 📊 Training Bittikten Sonra:
```bash
# 1. Comprehensive evaluation
python src/evaluate.py --model experiments/exp_NEW/models/dqn_cellfree_final --n-episodes 100

# 2. Circuit power sensitivity test
python src/test_circuit_power.py --rl-model experiments/exp_NEW/models/dqn_cellfree_final
```

---

## 📋 İçindekiler

1. [Temel Demo](#1-temel-demo)
2. [Network Simülasyonu Testi](#2-network-simülasyonu-testi)
3. [Baseline Stratejiler Karşılaştırması](#3-baseline-stratejiler-karşılaştırması)
4. [Circuit Power Sensitivity Analizi](#4-circuit-power-sensitivity-analizi)
5. [AP Scaling Analizi](#5-ap-scaling-analizi)
6. [RL Agent Training](#6-rl-agent-training)
7. [RL Agent Evaluation](#7-rl-agent-evaluation)
8. [Quick Evaluation](#8-quick-evaluation)
9. [Agent Adaptivity Analysis](#9-agent-adaptivity-analysis)

---

## 1. Temel Demo

### Komut
```bash
cd /Users/bengi/ceng505_cellfree_rl
python src/demo.py
```

### Ne Test Ediyor?
- Sionna kütüphanesinin doğru kurulumunu
- Cell-Free network simülasyonunun çalışmasını
- 3 baseline stratejinin temel performansını

### Çıktılar
**Terminal Output:**
- Her strateji için:
  - Average SINR (dB)
  - Average Rate (Mbps)
  - Energy Efficiency (bits/Joule)
  - QoS Satisfaction (%)
  - Active APs sayısı

**Grafikler:**
- 3 adet AP-User Association Matrix heatmap (her strateji için)
- Her grafik hangi AP'lerin hangi kullanıcılara hizmet ettiğini gösterir

### Ne Zaman Kullanılır?
- Projeyi ilk kurduğunda
- Sistemin çalıştığını doğrulamak için
- Baseline stratejilerin temel davranışını görmek için

### Örnek Çıktı
```
================================================================================
CELL-FREE NETWORK RL - DEMONSTRATION
================================================================================
Configuration: 25 APs, 10 Users

   Testing: Nearest AP + Max Power
   - Average SINR: 13.74 dB
   - Average Rate: 25.69 Mbps
   - Energy Efficiency: 3.85e+07 bits/Joule
   - QoS Satisfaction: 95.50%
   - Active APs: 8/25
```

---

## 2. Network Simülasyonu Testi

### Komut
```bash
python src/network/cellfree_network.py
```

### Ne Test Ediyor?
- Channel generation (Rayleigh fading + path loss)
- SINR hesaplamaları
- Rate hesaplamaları
- Energy efficiency hesaplamaları
- Circuit power modelinin doğruluğu

### Çıktılar
**Terminal Output:**
- Network configuration details
- Channel matrix shape ve statistics
- SINR values (batch)
- Rate values (batch)
- Energy efficiency
- Circuit power contribution

### Ne Zaman Kullanılır?
- Network simülasyonunu debug etmek için
- Matematiksel hesaplamaları doğrulamak için
- Yeni özellikler ekledikten sonra test için

---

## 3. Baseline Stratejiler Karşılaştırması

### Komut 1: Tek Strateji Test (Quick)
```bash
# Sadece Nearest AP stratejisini test et
python src/agents/baselines.py --strategy nearest_ap
```

### Komut 2: Birkaç Strateji Karşılaştır
```bash
# İki stratejiyi karşılaştır
python src/agents/baselines.py --strategy nearest_ap equal_power
```

### Komut 3: Tüm Baseline Stratejiler (Full Comparison)
```bash
# Tüm 5 stratejiyi karşılaştır (default)
python src/agents/baselines.py
```

### Komut 4: Farklı Episode Sayısı
```bash
# 50 episode ile test et (daha accurate results)
python src/agents/baselines.py --n-episodes 50
```

### Komut 5: Detaylı Grafik Kaydetme
```bash
# Sonuçları belirli bir klasöre kaydet
python src/agents/baselines.py --save-dir results/baseline_comparison
```

### Komut 6: Farklı Network Konfigürasyonu
```bash
# 30 AP, 15 user ile test et
python src/agents/baselines.py --num-aps 30 --num-users 15
```

### Komut 7: Full Parametreli Test
```bash
# Tüm parametrelerle detaylı test
python src/agents/baselines.py \
  --n-episodes 50 \
  --num-aps 25 \
  --num-users 10 \
  --save-dir results/baseline_full \
  --plot
```

### Ne Test Ediyor?
- 5 farklı baseline stratejiyi karşılaştırır:
  1. **Nearest AP + Max Power**: Her kullanıcı en yakın AP'ye bağlanır, max power
  2. **Equal Power + All Serve**: Tüm AP'ler her kullanıcıya hizmet eder
  3. **Random**: Random power allocation ve AP selection
  4. **Distance-Based**: Mesafeye göre power allocation
  5. **Load Balancing**: AP'ler arasında kullanıcıları dengeli dağıt

### Çıktılar
**Terminal Output:**
```
================================================================================
BASELINE STRATEGIES COMPARISON
================================================================================
Configuration: 25 APs, 10 Users
Number of Episodes: 20

Testing Strategy: Nearest AP + Max Power
  Episode 1/20: Rate=26.14 Mbps, EE=4.51e+07 bits/J, QoS=100.0%
  Episode 5/20: Rate=25.89 Mbps, EE=4.48e+07 bits/J, QoS=100.0%
  ...

Average Results:
  - Average Rate: 25.69 ± 1.23 Mbps
  - Energy Efficiency: 3.85e+07 ± 2.31e+06 bits/J
  - QoS Satisfaction: 95.5 ± 2.8%
  - SINR: 13.74 ± 1.2 dB
  - Active APs: 8.0 ± 0.0

Testing Strategy: Equal Power + All Serve
  Episode 1/20: Rate=114.23 Mbps, EE=1.52e+08 bits/J, QoS=100.0%
  ...

================================================================================
COMPARISON TABLE
================================================================================

Strategy              | EE (bits/J)       | Rate (Mbps)   | QoS (%)      | Active APs
----------------------------------------------------------------------------------
Nearest AP            | 3.85e+07 ± 2.3e+6 | 25.69 ± 1.23  | 95.5 ± 2.8   | 8.0 ± 0.0
Equal Power           | 1.52e+08 ± 5.6e+6 | 114.23 ± 4.5  | 100.0 ± 0.0  | 25.0 ± 0.0
Random                | 2.14e+07 ± 3.2e+6 | 18.45 ± 2.67  | 78.3 ± 5.1   | 12.3 ± 1.2
Distance-Based        | 3.12e+07 ± 2.8e+6 | 22.34 ± 1.89  | 88.7 ± 3.5   | 9.5 ± 0.7
Load Balancing        | 3.87e+06 ± 8.2e+5 | 1.89 ± 0.45   | 7.0 ± 5.2    | 10.0 ± 0.0

Best Energy Efficiency: Equal Power (1.52e+08 bits/J)
Best Rate: Equal Power (114.23 Mbps)
Best QoS: Equal Power (100.0%)
Least Active APs: Nearest AP (8.0)

Trade-off Analysis:
  • Equal Power: Highest performance but uses all 25 APs (high circuit power)
  • Nearest AP: Good balance - 95.5% QoS with only 8 APs
  • Load Balancing: Poor performance - users spread too thin
```

**Grafikler:**
1. **`results/baseline_comparison_metrics.png`** (2x2 grid):
   - Energy Efficiency (bar chart with error bars)
   - Average Rate (bar chart with error bars)
   - QoS Satisfaction (bar chart with error bars)
   - Active APs Count (bar chart with error bars)

2. **`results/baseline_association_nearest_ap.png`**:
   - Heatmap: AP-User association for Nearest AP
   - 25 (APs) x 10 (Users) matrix
   - Color intensity: Association strength

3. **`results/baseline_association_equal_power.png`**:
   - Heatmap: All APs serve all users (fully populated)

4. **`results/baseline_association_load_balance.png`**:
   - Heatmap: Users distributed across APs

5. **`results/baseline_radar_chart.png`**:
   - Normalized performance comparison
   - 4 axes: EE, Rate, QoS, -Active APs (inverted)

**JSON Export:**
- `results/baseline_comparison_results.json`:
```json
{
  "nearest_ap": {
    "mean_energy_efficiency": 3.85e+07,
    "std_energy_efficiency": 2.31e+06,
    "mean_rate_mbps": 25.69,
    "std_rate_mbps": 1.23,
    "mean_qos_satisfaction": 95.5,
    "std_qos_satisfaction": 2.8,
    "mean_active_aps": 8.0,
    "std_active_aps": 0.0
  },
  ...
}
```

### Ne Zaman Kullanılır?
- RL agent'ı train etmeden önce baseline'ları anlamak için
- Farklı stratejilerin trade-off'larını görmek için
- Hangi baseline'ın comparison için best match olduğunu belirlemek için

### Command Chaining Örnekleri

**Test → Analyze → Report Pipeline:**
```bash
# 1. Test all baselines
python src/agents/baselines.py --n-episodes 50 --save-dir results/baseline_v1

# 2. Test with different network config
python src/agents/baselines.py --num-aps 30 --num-users 15 --save-dir results/baseline_v2

# 3. Compare results (manuel olarak JSON'ları karşılaştır)
cat results/baseline_v1/baseline_comparison_results.json
cat results/baseline_v2/baseline_comparison_results.json
```

**Sequential Testing (Farklı Konfigürasyonlar):**
```bash
# Test 1: Default config
python src/agents/baselines.py --save-dir results/baseline_25aps_10users

# Test 2: More APs
python src/agents/baselines.py --num-aps 40 --save-dir results/baseline_40aps_10users

# Test 3: More users
python src/agents/baselines.py --num-users 20 --save-dir results/baseline_25aps_20users

# Sonuçları karşılaştır
ls -la results/baseline_*/
```

### Örnek Çıktı Analizi

**Equal Power vs Nearest AP Trade-off:**
```
Equal Power:
  ✅ Pros: Highest EE (1.52e+08), Best QoS (100%), Highest Rate (114 Mbps)
  ❌ Cons: Uses ALL 25 APs (5W circuit power @ 200mW/AP)

Nearest AP:
  ✅ Pros: Only 8 APs (1.6W circuit power), Still 95.5% QoS
  ❌ Cons: Lower EE (3.85e+07), Lower Rate (25.69 Mbps)

Winner: Depends on optimization goal
  - Maximize Performance → Equal Power
  - Minimize Circuit Power → Nearest AP
  - Balance Both → RL Agent (to be trained)
```

---

## 4. Circuit Power Sensitivity Analizi

### Komut 1: Sadece Baseline Stratejiler
```bash
# Default: 3 circuit power değeri (100mW, 200mW, 500mW)
python src/test_circuit_power.py
```

### Komut 2: RL Agent Dahil (Old Model)
```bash
# Eski circuit power-unaware model ile test
python src/test_circuit_power.py \
  --rl-model experiments/exp_20251205_143919/models/dqn_cellfree_final
```

### Komut 3: RL Agent Dahil (New Circuit Power-Aware Model)
```bash
# Yeni eğitilmiş circuit power-aware model ile test
python src/test_circuit_power.py \
  --rl-model experiments/exp_YYYYMMDD_HHMMSS/models/dqn_cellfree_final
```

### Komut 4: Multiple Model Comparison
```bash
# Önce eski model test et
python src/test_circuit_power.py \
  --rl-model experiments/exp_20251205_143919/models/dqn_cellfree_final

# Grafikleri kaydet
mv results/circuit_power_sensitivity.png results/circuit_power_sensitivity_old_model.png

# Sonra yeni model test et
python src/test_circuit_power.py \
  --rl-model experiments/exp_NEW/models/dqn_cellfree_final

# Grafikleri kaydet
mv results/circuit_power_sensitivity.png results/circuit_power_sensitivity_new_model.png

# İki grafiği karşılaştır
open results/circuit_power_sensitivity_old_model.png
open results/circuit_power_sensitivity_new_model.png
```

### Komut 5: Extended Circuit Power Range
```bash
# Script içinde circuit_powers listesini değiştirerek
# Örnek: [0.05, 0.1, 0.2, 0.3, 0.5, 0.8] ile test et
# (Kod modifikasyonu gerektirir)
```

### Ne Test Ediyor?
- **Circuit power değerinin stratejilere etkisini** analiz eder
- 3 farklı circuit power değeri test edilir:
  - 100mW (düşük)
  - 200mW (default)
  - 500mW (yüksek)
- Her strateji için circuit power değişiminin etkisini gösterir

### Çıktılar
**Terminal Output:**
```
Circuit Power = 100mW:
  Nearest AP:
    - Avg Rate: 26.14 Mbps
    - Energy Eff: 4.51e+07 bits/J
    - Active APs: 8/25
    - QoS Sat: 100.0%

  RL Agent:
    - Avg Rate: 32.55 Mbps
    - Energy Eff: 7.18e+07 bits/J
    - Active APs: 7.9/25
    - QoS Sat: 95.6%

TRENDS:
  RL Agent:
    • Energy Eff change (100mW → 500mW): -40.8%
    • Active APs (100mW): 7.868, (500mW): 7.868
    • Circuit power impact: HIGH
```

**Grafikler:**
- `results/circuit_power_sensitivity.png` (2x2 grid):
  1. **Average Rate per User (Mbps)** - Circuit power etkisi
  2. **Energy Efficiency (bits/Joule)** - Log scale
  3. **Number of Active APs** - Strateji davranışları
  4. **QoS Satisfaction (%)** - QoS compliance

**Key Findings:**
- Her circuit power değeri için en iyi strateji
- Stratejilerin circuit power'a adaptasyonu
- Circuit power impact seviyesi (HIGH/MODERATE/LOW)

### Ne Zaman Kullanılır?
- Circuit power modelini doğrulamak için
- RL agent'ın adaptasyon yeteneğini test etmek için
- Farklı circuit power senaryolarını analiz etmek için

### Grafik Açıklaması
- **X-axis**: Circuit power değerleri (100mW, 200mW, 500mW)
- **Y-axis**: Metrik değerleri
- **Çizgiler**: Her strateji farklı renk ve marker ile gösterilir
  - 🔴 Nearest AP (circle)
  - 🔵 Equal Power (square)
  - 🟢 Load Balancing (triangle)
  - 🟡 RL Agent (diamond)

---

## 5. AP Scaling Analizi

### Komut
```bash
python src/analyze_ap_scaling.py
```

### Ne Test Ediyor?
- **AP sayısının** network performansına etkisini analiz eder
- 5 farklı AP sayısı test edilir: 10, 15, 20, 25, 30
- Sabit 10 kullanıcı ile test edilir

### Çıktılar
**Terminal Output:**
```
Testing: 10 APs, 10 Users
  Nearest AP: EE=2.34e+07, Rate=18.45 Mbps, QoS=85.2%, Active APs=7
  Equal Power: EE=8.23e+07, Rate=87.12 Mbps, QoS=100%, Active APs=10

Testing: 30 APs, 10 Users
  Nearest AP: EE=5.12e+07, Rate=31.89 Mbps, QoS=100%, Active APs=9
  Equal Power: EE=1.89e+08, Rate=132.45 Mbps, QoS=100%, Active APs=30
```

**Grafikler:**
- `results/ap_scaling_analysis.png` (2x2 grid):
  1. **Average Rate (Mbps)** vs Number of APs
  2. **Energy Efficiency (bits/J)** vs Number of APs
  3. **Active APs** vs Total APs
  4. **QoS Satisfaction (%)** vs Number of APs

### Ne Zaman Kullanılır?
- Network capacity planning için
- Optimal AP deployment sayısını bulmak için
- Scalability analizi için

### İncelenen Sorular
- Daha fazla AP = daha iyi performans mı?
- Hangi strateji AP sayısından en çok faydalanır?
- Diminishing returns ne zaman başlar?

---

## 6. RL Agent Training

### Komut 1: DQN - Default Config (Circuit Power-Unaware)
```bash
# Original training (observation space: 260 features)
python src/train_agent.py \
  --agent dqn \
  --config configs/default.yaml \
  --timesteps 100000
```

### Komut 2: DQN - Circuit Power Adaptive (RECOMMENDED)
```bash
# Circuit power-aware training (observation space: 261 features)
python src/train_agent.py \
  --agent dqn \
  --config configs/circuit_power_adaptive.yaml \
  --timesteps 150000
```

### Komut 3: DQN - Quick Test (Fast Training)
```bash
# Hızlı test için az timesteps
python src/train_agent.py \
  --agent dqn \
  --config configs/circuit_power_adaptive.yaml \
  --timesteps 10000
```

### Komut 4: DQN - Long Training (High Quality)
```bash
# Daha uzun training for better convergence
python src/train_agent.py \
  --agent dqn \
  --config configs/circuit_power_adaptive.yaml \
  --timesteps 300000
```

### Komut 5: PPO - Circuit Power Adaptive
```bash
# PPO algorithm (different from DQN)
python src/train_agent.py \
  --agent ppo \
  --config configs/circuit_power_adaptive.yaml \
  --timesteps 150000
```

### Komut 6: Custom Experiment Directory
```bash
# Specific experiment name/location
python src/train_agent.py \
  --agent dqn \
  --config configs/circuit_power_adaptive.yaml \
  --timesteps 150000 \
  --exp_dir experiments_adaptive
```

### Komut 7: Resume Training (if supported)
```bash
# Load checkpoint ve devam et
# (train_agent.py'de --load-model parametresi eklenirse)
python src/train_agent.py \
  --agent dqn \
  --config configs/circuit_power_adaptive.yaml \
  --timesteps 150000 \
  --load-model experiments/exp_OLD/models/dqn_cellfree_50000
```

### Komut 8: Curriculum Learning (Manual 2-Stage)
```bash
# Stage 1: Fixed 200mW circuit power (50k steps)
python src/train_agent.py \
  --agent dqn \
  --config configs/circuit_power_curriculum.yaml \
  --timesteps 50000

# Model kaydedilir: experiments/exp_STAGE1/models/dqn_cellfree_final

# Stage 2: Config'i güncelle (randomize_circuit_power: true yap)
# Sonra Stage 1 model'inden devam et
python src/train_agent.py \
  --agent dqn \
  --config configs/circuit_power_adaptive.yaml \
  --timesteps 100000 \
  --load-model experiments/exp_STAGE1/models/dqn_cellfree_final
```

### Komut 9: Parallel Training (Multiple Configs)
```bash
# Terminal 1: Default config
python src/train_agent.py --agent dqn --config configs/default.yaml --timesteps 100000 &

# Terminal 2: Adaptive config
python src/train_agent.py --agent dqn --config configs/circuit_power_adaptive.yaml --timesteps 150000 &

# Terminal 3: PPO
python src/train_agent.py --agent ppo --config configs/circuit_power_adaptive.yaml --timesteps 150000 &

# wait for all to finish
wait
```

### Komut 10: Training with Real-time TensorBoard
```bash
# Terminal 1: Start training
python src/train_agent.py \
  --agent dqn \
  --config configs/circuit_power_adaptive.yaml \
  --timesteps 150000

# Terminal 2: Monitor with TensorBoard (başka terminal'de)
# Find experiment directory first
ls -lt experiments/ | head -5

# Then start TensorBoard
tensorboard --logdir experiments/exp_20251205_HHMMSS/tensorboard --port 6006

# Open browser: http://localhost:6006
```

### Ne Test Ediyor?
- RL algoritmasının öğrenme yeteneği
- Farklı hyperparameter kombinasyonları
- Circuit power'a adaptasyon (adaptive config ile)

### Çıktılar
**Terminal Output:**
```
Creating Environment...
Environment Configuration:
  - Observation Space: (261,)  # Circuit power dahil!
  - Action Space: Discrete(5)
  - Number of APs: 25
  - Number of Users: 10
  - QoS Requirement: 5.0 Mbps
  - Episode Length: 100

Creating DQN Agent...

================================================================================
STARTING TRAINING
================================================================================

[DQN training progress...]
```

**Dosyalar:**
- `experiments/exp_YYYYMMDD_HHMMSS/models/dqn_cellfree_final.zip` - Trained model
- `experiments/exp_YYYYMMDD_HHMMSS/tensorboard/` - TensorBoard logs
- `experiments/exp_YYYYMMDD_HHMMSS/logs/` - Training logs
- `experiments/exp_YYYYMMDD_HHMMSS/results/eval_results.json` - Evaluation results

**TensorBoard Görselleştirme:**
```bash
tensorboard --logdir experiments/exp_YYYYMMDD_HHMMSS/tensorboard
# http://localhost:6006 adresinden erişilebilir
```

**TensorBoard Grafikleri:**
- **rollout/ep_rew_mean**: Episode reward (ortalama)
- **rollout/ep_len_mean**: Episode length
- **train/loss**: Training loss
- **train/learning_rate**: Learning rate schedule
- **train/exploration_rate**: Epsilon (DQN için)

### Ne Zaman Kullanılır?
- Yeni model train etmek için
- Farklı config/hyperparameter denemek için
- Circuit power-adaptive model eğitmek için

---

## 7. RL Agent Evaluation (Comprehensive)

### Komut 1: Basic Evaluation (Default 20 Episodes)
```bash
# Quick evaluation
python src/evaluate.py \
  --model experiments/exp_20251205_143919/models/dqn_cellfree_final
```

### Komut 2: Comprehensive Evaluation (100 Episodes)
```bash
# More accurate results with 100 episodes
python src/evaluate.py \
  --model experiments/exp_20251205_143919/models/dqn_cellfree_final \
  --n-episodes 100
```

### Komut 3: Save to Custom Directory
```bash
# Save results to specific folder
python src/evaluate.py \
  --model experiments/exp_20251205_143919/models/dqn_cellfree_final \
  --n-episodes 100 \
  --save-dir results/evaluation_old_model
```

### Komut 4: Evaluate New Circuit Power-Aware Model
```bash
# Test the newly trained circuit power-aware model
python src/evaluate.py \
  --model experiments/exp_YYYYMMDD_HHMMSS/models/dqn_cellfree_final \
  --n-episodes 100 \
  --save-dir results/evaluation_new_model
```

### Komut 5: Side-by-Side Model Comparison
```bash
# Evaluate old model
python src/evaluate.py \
  --model experiments/exp_20251205_143919/models/dqn_cellfree_final \
  --n-episodes 100 \
  --save-dir results/old_model

# Evaluate new model
python src/evaluate.py \
  --model experiments/exp_NEW/models/dqn_cellfree_final \
  --n-episodes 100 \
  --save-dir results/new_model

# Compare JSON results
diff results/old_model/evaluation_results.json results/new_model/evaluation_results.json

# Or use Python to compare
python -c "
import json
old = json.load(open('results/old_model/evaluation_results.json'))
new = json.load(open('results/new_model/evaluation_results.json'))
print('Old EE:', old['RL Agent']['mean_energy_efficiency'])
print('New EE:', new['RL Agent']['mean_energy_efficiency'])
improvement = ((new['RL Agent']['mean_energy_efficiency'] - old['RL Agent']['mean_energy_efficiency']) / old['RL Agent']['mean_energy_efficiency']) * 100
print(f'Improvement: {improvement:.2f}%')
"
```

### Komut 6: Evaluate Multiple Models in Loop
```bash
# Evaluate all models in experiments directory
for exp_dir in experiments/exp_*/; do
  echo "Evaluating: $exp_dir"
  python src/evaluate.py \
    --model "${exp_dir}models/dqn_cellfree_final" \
    --n-episodes 50 \
    --save-dir "results/eval_$(basename $exp_dir)"
done
```

### Komut 7: Evaluation with Different Baseline Subsets
```bash
# Only compare with Nearest AP (fastest)
# (Requires code modification to select baselines)
python src/evaluate.py \
  --model experiments/exp_NEW/models/dqn_cellfree_final \
  --n-episodes 100 \
  --baselines nearest_ap
```

### Ne Test Ediyor?
- Trained RL agent'ın performansını
- 5 baseline strateji ile detaylı karşılaştırma
- 100 episode boyunca ortalama performans
- Circuit power bilgisi (active APs)

### Çıktılar
**Terminal Output:**
```
========================================================
EVALUATION RESULTS TABLE
========================================================

Strategy              EE (bits/J)            Rate (Mbps)      QoS (%)       SINR (dB)     Active APs
------------------------------------------------------------------------------------------------------------------------
RL Agent              6.15e+07 ± 4.23e+06    32.68 ± 2.15     95.7 ± 3.2    21.44 ± 1.5   7.9 ± 0.5
Nearest AP            3.85e+07 ± 2.31e+06    25.69 ± 1.89     95.5 ± 2.8    13.74 ± 1.2   8.0 ± 0.0
Equal Power           1.52e+08 ± 5.67e+06    114.23 ± 4.56    100.0 ± 0.0   43.23 ± 2.1   25.0 ± 0.0
Load Balancing        3.87e+06 ± 8.23e+05    1.89 ± 0.45      7.0 ± 5.2     -9.35 ± 2.3   10.0 ± 0.0

========================================================
PERFORMANCE IMPROVEMENTS
========================================================

vs Nearest AP:
  • Energy Efficiency: +59.77%
  • Average Rate: +27.21%
  • QoS Satisfaction: +0.21%
  • SINR: +56.05%
```

**Grafikler:**
1. **`results/comparison_metrics.png`** (2x2 grid):
   - Energy Efficiency (bar chart with error bars)
   - Average Rate (bar chart with error bars)
   - QoS Satisfaction (bar chart with error bars)
   - SINR (bar chart with error bars)

2. **`results/performance_radar.png`** (Radar/Spider chart):
   - 4 metriklerin normalize edilmiş karşılaştırması
   - Her strateji farklı renkte çizgi ile gösterilir
   - RL agent'ın hangi metriklerde güçlü/zayıf olduğunu görsel olarak gösterir

3. **`results/active_aps_comparison.png`** (Bar chart):
   - Her stratejinin kullandığı active AP sayısı
   - Circuit power consumption'ın indirect göstergesi
   - RL agent'ın efficiency'si

**JSON Dosyası:**
- `results/evaluation_results.json`:
  - Tüm stratejiler için detaylı metrikler
  - Mean ve std değerleri
  - Programatik analiz için kullanılabilir

### Ne Zaman Kullanılır?
- Training bittikten sonra final evaluation için
- Farklı modelleri karşılaştırmak için
- Rapor/paper için detaylı sonuçlar almak için

### Grafik Açıklaması
**Comparison Metrics:**
- Her metrik için bar chart
- Error bars: Standard deviation
- Y-axis: Metrik değeri
- X-axis: Stratejiler

**Radar Chart:**
- Merkezden dışa: Daha iyi performans
- 4 eksen: EE, Rate, QoS, SINR
- Normalize edilmiş [0, 1] scale

---

## 8. Quick Evaluation

### Komut
```bash
python src/quick_eval.py
```

### Ne Test Ediyor?
- Trained model'in hızlı test edilmesi (5 episode)
- Freeze olmadan güvenli evaluation
- Temel metrikler + improvement yüzdesi

### Çıktılar
**Terminal Output:**
```
Quick Evaluation (5 episodes):

RL Agent Results:
  Mean EE: 6.19e+07 bits/Joule
  Mean Rate: 32.45 Mbps
  Mean QoS: 94.8%

Baseline (Nearest AP):
  Mean EE: 3.96e+07 bits/Joule
  Mean Rate: 25.89 Mbps

Improvement: +56.16% Energy Efficiency
```

**Grafikler:** Yok (sadece terminal output)

### Ne Zaman Kullanılır?
- Training sırasında intermediate checkpoints test etmek için
- Hızlı sanity check için
- Full evaluation freeze oluyorsa alternatif olarak

---

## 9. Agent Adaptivity Analysis

### Komut 1: Basic Adaptivity Check (100 Episodes)
```bash
# Default: 100 episode, 200mW circuit power
python src/verify_adaptivity.py \
  --model experiments/exp_20251210_230304/models/dqn_cellfree_final \
  --episodes 100 \
  --circuit-power 0.2
```

### Komut 2: Quick Adaptivity Check (20 Episodes)
```bash
# Hızlı test için daha az episode
python src/verify_adaptivity.py \
  --model experiments/exp_YYYYMMDD_HHMMSS/models/dqn_cellfree_final \
  --episodes 20 \
  --circuit-power 0.2
```

### Komut 3: Comprehensive Analysis (200 Episodes)
```bash
# Daha detaylı istatistikler için çok episode
python src/verify_adaptivity.py \
  --model experiments/exp_YYYYMMDD_HHMMSS/models/dqn_cellfree_final \
  --episodes 200 \
  --circuit-power 0.2
```

### Komut 4: Multi-Circuit Power Adaptivity
```bash
# 100mW circuit power ile test
python src/verify_adaptivity.py \
  --model experiments/exp_YYYYMMDD_HHMMSS/models/dqn_cellfree_final \
  --episodes 100 \
  --circuit-power 0.1

# 200mW circuit power ile test
python src/verify_adaptivity.py \
  --model experiments/exp_YYYYMMDD_HHMMSS/models/dqn_cellfree_final \
  --episodes 100 \
  --circuit-power 0.2

# 500mW circuit power ile test
python src/verify_adaptivity.py \
  --model experiments/exp_YYYYMMDD_HHMMSS/models/dqn_cellfree_final \
  --episodes 100 \
  --circuit-power 0.5

# Grafikleri karşılaştır
open results/agent_adaptivity_analysis_100mW.png
open results/agent_adaptivity_analysis_200mW.png
open results/agent_adaptivity_analysis_500mW.png
```

### Komut 5: Compare Old vs New Model Adaptivity
```bash
# Old model (circuit power-unaware)
python src/verify_adaptivity.py \
  --model experiments/exp_20251205_143919/models/dqn_cellfree_final \
  --episodes 100 \
  --circuit-power 0.2

mv results/agent_adaptivity_analysis_200mW.png results/adaptivity_old_model.png

# New model (circuit power-aware)
python src/verify_adaptivity.py \
  --model experiments/exp_NEW/models/dqn_cellfree_final \
  --episodes 100 \
  --circuit-power 0.2

mv results/agent_adaptivity_analysis_200mW.png results/adaptivity_new_model.png

# Karşılaştır
open results/adaptivity_old_model.png
open results/adaptivity_new_model.png
```

### Ne Test Ediyor?
- **Agent'ın karar dağılımını** (decision distribution) analiz eder
- Agent'ın **sabit bir strateji mi yoksa adaptif mi** olduğunu belirler
- **Kaç farklı AP konfigürasyonu** kullandığını gösterir
- **Standart sapma** ile adaptasyon seviyesini ölçer
- **Frequency distribution** ile tercih edilen AP sayılarını gösterir

### Çıktılar

**Terminal Output:**
```
================================================================================
AGENT ADAPTIVITY ANALYSIS
================================================================================
Model: experiments/exp_20251210_230304/models/dqn_cellfree_final
Episodes: 100
Circuit Power: 200 mW

Running 100 episodes...
  Progress: 20/100 episodes completed
  Progress: 40/100 episodes completed
  ...
  Progress: 100/100 episodes completed

================================================================================
STATISTICAL ANALYSIS
================================================================================

📊 Active AP Count Distribution (All Steps):
  • Mean: 7.862 APs
  • Std Dev: 0.821 APs
  • Min: 5 APs
  • Max: 10 APs
  • Median: 8.0 APs
  • 25th percentile: 7.0 APs
  • 75th percentile: 8.0 APs

📈 Frequency Distribution:
   5 APs:   15 times (  0.1%)
   6 APs:  468 times (  4.7%) ██
   7 APs: 2603 times ( 26.0%) █████████████
   8 APs: 4770 times ( 47.7%) ███████████████████████
   9 APs: 2084 times ( 20.8%) ██████████
  10 APs:   60 times (  0.6%)

🎯 Adaptivity Metrics:
  • Unique AP counts used: 6
  • Standard deviation: 0.821
  • Adaptivity Level: MODERATE - Agent shows some adaptation

📉 Per-Episode Variation:
  • Avg episode mean: 7.862 APs
  • Avg within-episode std: 0.814 APs
  • Episode means range: [7.63, 8.10]

⚡ Performance Metrics:
  • Mean Reward: 115.3545
  • Mean Rate: 32.41 Mbps
  • Mean Energy Eff: 1.15e+08 bits/J
```

**Grafikler:**
- `results/agent_adaptivity_analysis_200mW.png` (2x2 grid):
  1. **Histogram (Top Left)**:
     - Active AP count frequency distribution
     - Shows which AP counts agent prefers
     - Mean line overlaid (red dashed)

  2. **Time Series (Top Right)**:
     - First 500 steps showing AP count over time
     - Shows temporal variation
     - Mean line overlaid

  3. **Episode Statistics (Bottom Left)**:
     - Per-episode mean ± std dev
     - Shows episode-to-episode variation
     - Grand mean overlaid

  4. **Box Plot (Bottom Right)**:
     - Statistical distribution for each AP count
     - Shows variance within each category

### Metrik Yorumlama

**Adaptivity Level Classification:**
- **LOW (std < 0.5)**:
  - Agent ezberci, hep aynı AP sayısını kullanıyor
  - Örnek: Std=0.2, sadece 8 AP kullanıyor (histogram'da tek çubuk)

- **MODERATE (0.5 < std < 1.5)**:
  - Agent duruma göre değişiklik gösteriyor ama tutucu
  - Örnek: Std=0.8, çoğunlukla 7-8-9 AP kullanıyor (3-4 çubuk)

- **HIGH (std > 1.5)**:
  - Agent çok esnek, geniş bir range'de karar veriyor
  - Örnek: Std=2.3, 5-15 AP arası geniş dağılım (birçok çubuk)

**Örnek Yorumlar:**

*Senaryou 1: Circuit Power-Unaware Model*
```
Mean: 7.9, Std: 0.1
Unique: 2 (7 ve 8 AP)
Adaptivity: LOW

Yorum: "Agent sabit bir strateji öğrenmiş, hep 7-8 AP açıyor.
Circuit power değişikliklerine adapte olmuyor."
```

*Senaryo 2: Circuit Power-Aware Model (Beklenen)*
```
Mean: 7.86, Std: 0.82
Unique: 6 (5-10 arası)
Adaptivity: MODERATE

Yorum: "Agent duruma göre 5-10 AP arası esnek karar veriyor.
Çoğunlukla 7-8-9 tercih ediyor ama nadir durumlarda 5 veya 10'a kadar çıkabiliyor."
```

### Ne Zaman Kullanılır?
- **Tezde "Agent adaptif mi?" sorusunu cevaplamak için**
- Training sonrası agent davranışını anlamak için
- Old vs new model karşılaştırmasında adaptasyon farkını göstermek için
- Farklı circuit power değerlerinde agent'ın nasıl davrandığını görmek için

### Beklenen Sonuçlar

**Old Model (Circuit Power-Unaware):**
- Mean: ~7.9 APs
- Std: **< 0.5** (LOW adaptivity)
- Unique: 1-2 (sadece 7 ve 8 AP)
- Frequency: Tek bir büyük çubuk (8 AP'de)

**New Model (Circuit Power-Aware) - 100mW:**
- Mean: ~10-12 APs (circuit power ucuz → daha çok AP)
- Std: 1.0-2.0 (MODERATE-HIGH adaptivity)
- Unique: 4-6 (8-14 arası)

**New Model (Circuit Power-Aware) - 500mW:**
- Mean: ~5-6 APs (circuit power pahalı → daha az AP)
- Std: 1.0-2.0 (MODERATE-HIGH adaptivity)
- Unique: 4-6 (3-8 arası)

### Grafik Açıklaması

**Histogram (Sol Üst):**
- X-axis: Active AP count (0-25)
- Y-axis: Frequency (kaç kere seçildi)
- Kırmızı çizgi: Ortalama
- **Yorumlama**:
  - Tek çubuk → Ezberci
  - Birkaç çubuk → Orta seviye adaptasyon
  - Geniş dağılım → Yüksek adaptasyon

**Time Series (Sağ Üst):**
- X-axis: Step number (0-500)
- Y-axis: Active APs
- Kırmızı çizgi: Genel ortalama
- **Yorumlama**:
  - Düz çizgi → Hiç değişmiyor
  - Hafif dalgalı → Bazen değişiyor
  - Çok dalgalı → Sürekli adapte oluyor

**Episode Stats (Sol Alt):**
- X-axis: Episode number
- Y-axis: Mean active APs per episode
- Error bars: Within-episode std
- **Yorumlama**:
  - Küçük error bars → Episode içinde sabit
  - Büyük error bars → Episode içinde değişken

**Box Plot (Sağ Alt):**
- Her AP count için variance gösterir
- **Yorumlama**:
  -좁은 kutular → O AP sayısı az kullanılmış
  - Geniş kutular → O AP sayısı çeşitli senaryolarda kullanılmış

### Tezde Kullanım Örnekleri

**Şekil Başlığı:**
```
Figure X: Agent Adaptivity Analysis for Circuit Power-Aware DQN Model
The agent demonstrates moderate adaptivity (std=0.82) across 100 test episodes,
utilizing 6 different AP configurations (5-10 APs) with preference for 7-9 APs (94%).
```

**Metin İçinde:**
```
The trained DQN agent exhibited moderate adaptivity, using an average of 7.86 ± 0.82
active APs across 100 evaluation episodes. The decision distribution (Figure X) shows
that while the agent predominantly selects 7-9 APs (94% of decisions), it demonstrates
flexibility by occasionally using 5-6 or 10 APs (6% of decisions) in specific scenarios.
This adaptive behavior contrasts with the circuit power-unaware baseline model, which
consistently used 7.9 ± 0.1 APs regardless of circuit power cost, indicating learned
rigidity rather than scenario-specific optimization.
```

### Command Chaining - Full Adaptivity Analysis Pipeline

```bash
# =============================================================================
# COMPLETE ADAPTIVITY ANALYSIS WORKFLOW
# =============================================================================

# Step 1: Test at 3 different circuit powers
echo "=== Testing adaptivity at 100mW ==="
python src/verify_adaptivity.py \
  --model experiments/exp_NEW/models/dqn_cellfree_final \
  --episodes 100 \
  --circuit-power 0.1 > logs/adaptivity_100mW.log

echo "=== Testing adaptivity at 200mW ==="
python src/verify_adaptivity.py \
  --model experiments/exp_NEW/models/dqn_cellfree_final \
  --episodes 100 \
  --circuit-power 0.2 > logs/adaptivity_200mW.log

echo "=== Testing adaptivity at 500mW ==="
python src/verify_adaptivity.py \
  --model experiments/exp_NEW/models/dqn_cellfree_final \
  --episodes 100 \
  --circuit-power 0.5 > logs/adaptivity_500mW.log

# Step 2: Archive results
mkdir -p results/adaptivity_analysis
cp results/agent_adaptivity_analysis_*.png results/adaptivity_analysis/
cp logs/adaptivity_*.log results/adaptivity_analysis/

# Step 3: Compare with old model (optional)
echo "=== Testing old model for comparison ==="
python src/verify_adaptivity.py \
  --model experiments/exp_20251205_143919/models/dqn_cellfree_final \
  --episodes 100 \
  --circuit-power 0.2 > logs/adaptivity_old_model.log

# Step 4: Generate summary report
echo "=== Generating Summary Report ==="
python3 << 'EOF'
import re

# Parse log files
circuit_powers = ['100mW', '200mW', '500mW']
results = {}

for cp in circuit_powers:
    with open(f'logs/adaptivity_{cp}.log', 'r') as f:
        content = f.read()

        # Extract metrics
        mean = re.search(r'Mean: ([\d.]+) APs', content)
        std = re.search(r'Std Dev: ([\d.]+) APs', content)
        unique = re.search(r'Unique AP counts used: (\d+)', content)
        level = re.search(r'Adaptivity Level: (.+)', content)

        results[cp] = {
            'mean': float(mean.group(1)) if mean else None,
            'std': float(std.group(1)) if std else None,
            'unique': int(unique.group(1)) if unique else None,
            'level': level.group(1).strip() if level else None
        }

# Print summary
print("\n" + "="*80)
print("CIRCUIT POWER ADAPTIVITY SUMMARY")
print("="*80)

for cp in circuit_powers:
    r = results[cp]
    print(f"\n{cp}:")
    print(f"  Mean APs: {r['mean']:.2f} ± {r['std']:.2f}")
    print(f"  Unique Configs: {r['unique']}")
    print(f"  Adaptivity: {r['level']}")

print("\n" + "="*80)
EOF

echo ""
echo "✅ Adaptivity analysis complete!"
echo "Results: results/adaptivity_analysis/"
echo "Logs: logs/adaptivity_*.log"
```

---

## 🎯 Test Sıralaması Önerisi

Training başlamadan önce bu sırayla test et:

1. ✅ **Demo** - Sistem çalışıyor mu?
```bash
python src/demo.py
```

2. ✅ **Baseline Comparison** - Baseline stratejileri anla
```bash
python src/agents/baselines.py
```

3. ✅ **Circuit Power Analysis (Baselines)** - Circuit power etkisini gör
```bash
python src/test_circuit_power.py
```

4. ✅ **Circuit Power Analysis (Old RL Model)** - Eski model ne yapıyor?
```bash
python src/test_circuit_power.py --rl-model experiments/exp_20251205_143919/models/dqn_cellfree_final
```

5. ✅ **AP Scaling** - AP sayısı etkisini anla
```bash
python src/analyze_ap_scaling.py
```

---

## 📊 Training Sırasında İzleme

Training devam ederken:

### TensorBoard (Real-time)
```bash
tensorboard --logdir experiments/exp_YYYYMMDD_HHMMSS/tensorboard
```
- http://localhost:6006 adresinden izle
- Reward trend'ini gör
- Loss'un düştüğünü doğrula
- Exploration rate'i izle

### Quick Checkpoint Test (Her 25k steps)
```bash
# Training durduğunda checkpoint'ten test et
python src/quick_eval.py  # Model path'i içeride güncelle
```

---

## 🔍 Training Bittikten Sonra

1. **Comprehensive Evaluation**
```bash
python src/evaluate.py --model experiments/exp_YYYYMMDD_HHMMSS/models/dqn_cellfree_final --n-episodes 100
```

2. **Circuit Power Sensitivity (New Model)**
```bash
python src/test_circuit_power.py --rl-model experiments/exp_YYYYMMDD_HHMMSS/models/dqn_cellfree_final
```

3. **Adaptivity Analysis**
```bash
# Agent'ın karar dağılımını analiz et
python src/verify_adaptivity.py \
  --model experiments/exp_YYYYMMDD_HHMMSS/models/dqn_cellfree_final \
  --episodes 100 \
  --circuit-power 0.2
```

4. **Compare with Old Model**
```bash
# Old model
python src/evaluate.py --model experiments/exp_20251205_143919/models/dqn_cellfree_final --save-dir results/old_model

# New model
python src/evaluate.py --model experiments/exp_YYYYMMDD_HHMMSS/models/dqn_cellfree_final --save-dir results/new_model

# Manuel karşılaştır veya results/*.json dosyalarını analiz et
```

---

## 📝 Notlar

### Observation Space Değişikliği
- **Old models (exp_20251205_143919)**: 260 features (no circuit power)
- **New models**: 261 features (with circuit power)
- Old model'i test ederken observation space uyumsuzluğu **olabilir**

### Model Compatibility
- Circuit power-aware config ile eğitilen model sadece circuit power bilgisi içeren env ile çalışır
- Eski modeller yeni environment'ta çalışmaz (observation space farklı)

### Performance Beklentileri

**Old Model (Circuit Power-Unaware):**
- 100mW, 200mW, 500mW: Aynı sayıda AP kullanır
- Circuit power'a adapte olmaz

**New Model (Circuit Power-Aware):**
- 100mW: Daha fazla AP (10-12)
- 200mW: Orta seviye (7-8)
- 500mW: Daha az AP (5-6)
- Circuit power'a göre strateji değiştirir

---

## 🚀 Hızlı Başlangıç Checklist

Training başlamadan önce:

- [ ] Demo çalışıyor
- [ ] Baseline stratejiler test edildi
- [ ] Circuit power sensitivity analizi yapıldı (baselines)
- [ ] Eski RL model circuit power sensitivity test edildi
- [ ] Config dosyası doğru (`circuit_power_adaptive.yaml`)
- [ ] Virtual environment aktif
- [ ] Disk space yeterli (TensorBoard logs büyük olabilir)

Training bittikten sonra:

- [ ] TensorBoard logları incelendi
- [ ] Comprehensive evaluation yapıldı
- [ ] Circuit power sensitivity test edildi (new model)
- [ ] Results grafikler kaydedildi
- [ ] JSON results export edildi

---

## 📞 Sorun Giderme

### "Environment freeze oluyor"
→ `quick_eval.py` kullan veya `max_steps=100` ekle

### "Observation space mismatch"
→ Eski model yeni environment'ta çalışmaz (260 vs 261 features)

### "TensorBoard açılmıyor"
→ Port 6006 kullanımda olabilir: `tensorboard --logdir ... --port 6007`

### "Training çok yavaş"
→ CPU kullanıyorsun, GPU kullan veya timesteps azalt

---

## 📚 Ek Kaynaklar

- **Config Files**: `configs/` dizininde 3 config var
  - `default.yaml`: Original (circuit power-unaware)
  - `circuit_power_adaptive.yaml`: Randomized circuit power
  - `circuit_power_curriculum.yaml`: Curriculum learning

- **Model Storage**: `experiments/` dizininde timestamped directories
  - `models/`: Trained models (.zip)
  - `tensorboard/`: TensorBoard logs
  - `logs/`: Text logs
  - `results/`: Evaluation results (JSON)

- **Results**: `results/` dizininde generated plots ve JSON files

---

---

## 🎬 Complete Testing Workflow Example

### Senaryo: Yeni Circuit Power-Aware Model Eğitimi ve Karşılaştırması

```bash
# =============================================================================
# PHASE 1: PRE-TRAINING TESTS (Training başlamadan önce)
# =============================================================================

# 1.1. System Check
echo "=== System Check ==="
python src/demo.py
# Output: 3 baseline strateji test edilir, association heatmaps

# 1.2. Baseline Comparison
echo "=== Baseline Comparison ==="
python src/agents/baselines.py --n-episodes 50 --save-dir results/baseline_reference
# Output: 5 strateji detaylı karşılaştırma, grafikler kaydedilir

# 1.3. Circuit Power Sensitivity (Baselines Only)
echo "=== Circuit Power Test (Baselines) ==="
python src/test_circuit_power.py
# Output: 3 circuit power değeri, 3 baseline, results/circuit_power_sensitivity.png
mv results/circuit_power_sensitivity.png results/circuit_power_baseline_only.png

# 1.4. Old RL Model Circuit Power Test
echo "=== Circuit Power Test (Old RL Model) ==="
python src/test_circuit_power.py --rl-model experiments/exp_20251205_143919/models/dqn_cellfree_final
# Output: Old model circuit power adaptasyonu (yok, hep 7.9 APs)
mv results/circuit_power_sensitivity.png results/circuit_power_old_rl.png

# 1.5. AP Scaling Analysis
echo "=== AP Scaling ==="
python src/analyze_ap_scaling.py
# Output: 10-30 APs arası scaling, results/ap_scaling_analysis.png

# =============================================================================
# PHASE 2: TRAINING (3-6 saat)
# =============================================================================

# 2.1. Start Training (Terminal 1)
echo "=== Starting Circuit Power-Aware Training ==="
python src/train_agent.py \
  --agent dqn \
  --config configs/circuit_power_adaptive.yaml \
  --timesteps 150000 \
  --exp_dir experiments

# Model save edilir: experiments/exp_YYYYMMDD_HHMMSS/models/dqn_cellfree_final

# 2.2. Monitor with TensorBoard (Terminal 2 - training devam ederken)
# Find the experiment directory
EXP_DIR=$(ls -td experiments/exp_* | head -1)
echo "Monitoring: $EXP_DIR"

tensorboard --logdir $EXP_DIR/tensorboard --port 6006
# Browser: http://localhost:6006

# 2.3. Quick Checkpoint Test (Her 30 dakikada bir - opsiyonel)
# Training durdur, quick test yap, devam et

# =============================================================================
# PHASE 3: POST-TRAINING EVALUATION
# =============================================================================

# 3.1. Find the trained model
NEW_MODEL=$(ls -td experiments/exp_*/models/dqn_cellfree_final | head -1)
echo "New Model: $NEW_MODEL"

# 3.2. Quick Evaluation (Hızlı test)
echo "=== Quick Evaluation ==="
python src/quick_eval.py
# Output: 5 episode quick test, terminal output only

# 3.3. Comprehensive Evaluation
echo "=== Comprehensive Evaluation (New Model) ==="
python src/evaluate.py \
  --model $NEW_MODEL \
  --n-episodes 100 \
  --save-dir results/eval_new_model
# Output: 3 grafikler, JSON results, detailed comparison

# 3.4. Circuit Power Sensitivity (New Model)
echo "=== Circuit Power Test (New Model) ==="
python src/test_circuit_power.py --rl-model $NEW_MODEL
# Output: New model circuit power adaptasyonu
mv results/circuit_power_sensitivity.png results/circuit_power_new_rl.png

# =============================================================================
# PHASE 4: COMPARISON & ANALYSIS
# =============================================================================

# 4.1. Side-by-Side Model Comparison
echo "=== Comparing Old vs New Model ==="

# Evaluate old model
python src/evaluate.py \
  --model experiments/exp_20251205_143919/models/dqn_cellfree_final \
  --n-episodes 100 \
  --save-dir results/eval_old_model

# Results already saved for new model in Phase 3.3

# 4.2. Compare JSON Results
python3 << 'EOF'
import json
import numpy as np

# Load results
old = json.load(open('results/eval_old_model/evaluation_results.json'))
new = json.load(open('results/eval_new_model/evaluation_results.json'))

# Compare RL Agent performance
old_rl = old['RL Agent']
new_rl = new['RL Agent']

print("\n" + "="*80)
print("OLD MODEL vs NEW MODEL COMPARISON")
print("="*80)

metrics = [
    ('mean_energy_efficiency', 'Energy Efficiency (bits/J)'),
    ('mean_rate_mbps', 'Average Rate (Mbps)'),
    ('mean_qos_satisfaction', 'QoS Satisfaction (%)'),
    ('mean_active_aps', 'Active APs')
]

for key, name in metrics:
    old_val = old_rl[key]
    new_val = new_rl[key]
    improvement = ((new_val - old_val) / old_val) * 100

    print(f"\n{name}:")
    print(f"  Old: {old_val:.2e}" if 'efficiency' in key else f"  Old: {old_val:.2f}")
    print(f"  New: {new_val:.2e}" if 'efficiency' in key else f"  New: {new_val:.2f}")
    print(f"  Improvement: {improvement:+.2f}%")

print("\n" + "="*80)
EOF

# 4.3. Visual Comparison (Open all graphs)
echo "=== Opening Comparison Graphs ==="
open results/circuit_power_old_rl.png
open results/circuit_power_new_rl.png
open results/eval_old_model/comparison_metrics.png
open results/eval_new_model/comparison_metrics.png

# 4.4. Generate Comparison Report
echo "=== Generating Report ==="
cat > results/COMPARISON_REPORT.md << 'EOF'
# Circuit Power-Aware Training Results

## Model Comparison

### Old Model (Circuit Power-Unaware)
- Training: 100k timesteps, default config
- Observation Space: 260 features (no circuit power)
- Circuit Power Adaptation: **NO**

### New Model (Circuit Power-Aware)
- Training: 150k timesteps, adaptive config
- Observation Space: 261 features (with circuit power)
- Circuit Power Adaptation: **YES**

## Key Findings

### Circuit Power Adaptivity:
**Old Model:**
- 100mW → 7.9 APs
- 200mW → 7.9 APs
- 500mW → 7.9 APs
- **No adaptation!**

**New Model:**
- 100mW → ~10-12 APs (expected)
- 200mW → ~7-8 APs (expected)
- 500mW → ~5-6 APs (expected)
- **Adapts to circuit power cost!**

## Conclusion
[To be filled with actual results]
EOF

echo "Report saved to: results/COMPARISON_REPORT.md"

# =============================================================================
# PHASE 5: ARCHIVE & DOCUMENTATION
# =============================================================================

# 5.1. Create Archive
ARCHIVE_DIR="results/experiment_archive_$(date +%Y%m%d_%H%M%S)"
mkdir -p $ARCHIVE_DIR

# Copy all results
cp -r results/eval_* $ARCHIVE_DIR/
cp results/circuit_power_*.png $ARCHIVE_DIR/
cp results/COMPARISON_REPORT.md $ARCHIVE_DIR/
cp $EXP_DIR/tensorboard/* $ARCHIVE_DIR/tensorboard/

echo "Results archived to: $ARCHIVE_DIR"

# 5.2. Generate Final Summary
echo "=== EXPERIMENT COMPLETE ==="
echo "Training Time: [Check TensorBoard]"
echo "New Model Path: $NEW_MODEL"
echo "Results Directory: results/eval_new_model"
echo "Archive: $ARCHIVE_DIR"
echo ""
echo "Next Steps:"
echo "  1. Review TensorBoard logs"
echo "  2. Analyze comparison graphs"
echo "  3. Update COMPARISON_REPORT.md with findings"
echo "  4. Prepare presentation/paper materials"
```

### Beklenen Süre Tahmini:
- **Phase 1** (Pre-training tests): ~15 dakika
- **Phase 2** (Training): ~3-6 saat
- **Phase 3** (Post-training eval): ~20 dakika
- **Phase 4** (Comparison): ~5 dakika
- **Phase 5** (Archive): ~2 dakika

**Toplam**: ~4-7 saat (çoğu training)

---

## 📝 Önemli Notlar

### Model Compatibility Uyarıları

1. **Observation Space Mismatch**:
   - Old models: 260 features
   - New models: 261 features
   - **UYUMSUZ!** Old model new env'de çalışmaz

2. **Config Dependency**:
   - `circuit_power_adaptive.yaml` ile eğitilen model
   - `randomize_circuit_power: true` flag'i gerektirir
   - Evaluation sırasında doğru config kullan

3. **TensorBoard Port Conflicts**:
   - Default port: 6006
   - Port kullanımda ise: `--port 6007` kullan

### Troubleshooting

| Problem | Çözüm |
|---------|-------|
| Evaluation freeze | `quick_eval.py` kullan |
| Observation space mismatch | Old model old env ile test et |
| TensorBoard açılmıyor | Port değiştir (`--port 6007`) |
| Training çok yavaş | GPU kullan veya timesteps azalt |
| JSON serialization error | Already fixed (convert_to_native) |

---

**Son Güncelleme**: 2025-12-11
**Proje**: Cell-Free Massive MIMO Resource Allocation with RL
**Config**: Circuit Power-Aware Training
**Toplam Test Scripts**: 9
**Toplam Komut Varyasyonları**: 50+

---

## 🎯 Quick Command Reference Card

### Most Important Commands (Copy-Paste Ready)

**1. Pre-Training Check:**
```bash
python src/demo.py && python src/agents/baselines.py && python src/test_circuit_power.py
```

**2. Train New Model:**
```bash
python src/train_agent.py --agent dqn --config configs/circuit_power_adaptive.yaml --timesteps 150000
```

**3. Post-Training Analysis (Replace exp_YYYYMMDD_HHMMSS with your experiment):**
```bash
# Set your experiment directory
EXP_DIR="experiments/exp_YYYYMMDD_HHMMSS"
MODEL_PATH="${EXP_DIR}/models/dqn_cellfree_final"

# 1. Comprehensive evaluation
python src/evaluate.py --model $MODEL_PATH --n-episodes 100

# 2. Circuit power sensitivity
python src/test_circuit_power.py --rl-model $MODEL_PATH

# 3. Adaptivity analysis
python src/verify_adaptivity.py --model $MODEL_PATH --episodes 100 --circuit-power 0.2

echo "✅ All analyses complete! Check results/ directory"
```

**4. Compare Old vs New:**
```bash
# Old model
OLD_MODEL="experiments/exp_20251205_143919/models/dqn_cellfree_final"
NEW_MODEL="experiments/exp_YYYYMMDD_HHMMSS/models/dqn_cellfree_final"

# Evaluate both
python src/evaluate.py --model $OLD_MODEL --save-dir results/eval_old
python src/evaluate.py --model $NEW_MODEL --save-dir results/eval_new

# Adaptivity comparison
python src/verify_adaptivity.py --model $OLD_MODEL --episodes 100 --circuit-power 0.2
mv results/agent_adaptivity_analysis_200mW.png results/adaptivity_old.png

python src/verify_adaptivity.py --model $NEW_MODEL --episodes 100 --circuit-power 0.2
mv results/agent_adaptivity_analysis_200mW.png results/adaptivity_new.png

# Open all comparison graphs
open results/eval_old/*.png results/eval_new/*.png results/adaptivity_*.png
```
