# Advanced AP Scaling Analysis

Gelişmiş AP sayısı ve strateji analiz aracı. Command-line interface ile esnek kullanım.

## 🚀 Yeni Özellikler

- ✅ **Tek komut satırı**: Tüm parametreler komut satırından ayarlanabilir
- ✅ **Strateji seçimi**: `nearest_ap`, `equal_power`, `load_balancing`, veya `all`
- ✅ **Esnek konfigürasyon**: AP ve kullanıcı sayısı özelleştirilebilir
- ✅ **2 analiz modu**: Single-config (karşılaştırma) ve Multi-config (scaling)
- ✅ **Otomatik grafik**: Moda göre uygun görselleştirme

## 📋 Kullanım Örnekleri

### 1. Tek Konfigürasyon - Tek Strateji
İstediğiniz formatı kullanabilirsiniz:
```bash
# Format: strategy AP_sayısı kullanıcı_sayısı
python src/analyze_ap_scaling.py nearest_ap 16 8
python src/analyze_ap_scaling.py equal_power 25 10
python src/analyze_ap_scaling.py load_balancing 36 12
```

**Çıktı:**
- Konsol raporu
- `results/analysis_nearest_ap_single.txt`

### 2. Tek Konfigürasyon - Tüm Stratejiler (Karşılaştırma)
```bash
# Format: all AP_sayısı kullanıcı_sayısı
python src/analyze_ap_scaling.py all 25 10
```

**Çıktı:**
- 3 stratejinin karşılaştırma grafiği (bar chart, 6 panel)
- `results/comparison_25aps_10users.png`
- `results/analysis_all_single.txt`

### 3. Multi-Konfigürasyon - Tek Strateji (Scaling)
```bash
# Default: 16, 25, 36, 49, 64 AP
python src/analyze_ap_scaling.py equal_power --multi

# Özel AP listesi
python src/analyze_ap_scaling.py nearest_ap --multi --aps 10,20,30,40

# Kullanıcı sayısı değiştirme
python src/analyze_ap_scaling.py load_balancing --multi --users 12
```

**Çıktı:**
- AP sayısına göre scaling grafiği (line plot, 6 panel)
- `results/scaling_equal_power_8users.png`
- `results/analysis_equal_power_multi.txt`

### 4. Multi-Konfigürasyon - Tüm Stratejiler (Tam Analiz)
```bash
# Tüm stratejiler + scaling
python src/analyze_ap_scaling.py all --multi

# Özel konfigürasyon
python src/analyze_ap_scaling.py all --multi --aps 16,25,36,49,64 --users 10
```

**Çıktı:**
- 3 stratejiyi karşılaştıran scaling grafiği (multi-line plot)
- `results/scaling_all_10users.png`
- `results/analysis_all_multi.txt`

## 🎯 Komut Yapısı

```bash
python src/analyze_ap_scaling.py STRATEGY [APs] [USERS] [OPTIONS]
```

### Parametreler

| Parametre | Tip | Açıklama | Örnek |
|-----------|-----|----------|-------|
| `STRATEGY` | **Zorunlu** | Strateji: `nearest_ap`, `equal_power`, `load_balancing`, `all` | `equal_power` |
| `APs` | Opsiyonel | AP sayısı (single-config için) | `16` |
| `USERS` | Opsiyonel | Kullanıcı sayısı | `8` |
| `--multi` | Flag | Multi-config scaling modu | |
| `--aps` | String | Virgülle ayrılmış AP listesi | `--aps 16,25,36` |
| `--users` | Integer | Kullanıcı sayısı (multi-config için) | `--users 10` |
| `--seed` | Integer | Random seed | `--seed 42` |

## 📊 Çıktı Tipleri

### Single-Config Mode
**Tek strateji:**
- Sadece konsol raporu

**Çoklu strateji (`all`):**
- Bar chart (2x3 panel)
- Metrikler: Rate, Energy Eff, SINR, QoS, Active APs, APs/User

### Multi-Config Mode
**Tek strateji:**
- Line plot (2x3 panel)
- X ekseni: AP sayısı
- Her metrik için ayrı grafik

**Çoklu strateji (`all`):**
- Multi-line plot (2x3 panel)
- 3 strateji aynı grafiklerde karşılaştırılır
- Farklı renk ve marker'lar

## 📁 Dosya Adlandırma

Script otomatik olarak dosya isimleri oluşturur:

```
results/
├── analysis_[strategy]_[mode].txt          # Tablo
├── comparison_[X]aps_[Y]users.png          # Single-config comparison
└── scaling_[strategy]_[Y]users.png         # Multi-config scaling
```

**Örnekler:**
- `analysis_nearest_ap_single.txt`
- `comparison_25aps_10users.png`
- `scaling_all_8users.png`

## 🎨 Grafik Açıklamaları

### Bar Chart (Strategy Comparison)
- **Ne zaman:** `all` stratejisi + single-config
- **Gösterir:** 3 stratejinin tek konfigürasyondaki performansı
- **Format:** Bar chart, her metrik ayrı panel

### Line Plot (Single Strategy Scaling)
- **Ne zaman:** Tek strateji + multi-config
- **Gösterir:** AP sayısı arttıkça performans değişimi
- **Format:** Line plot, trend analizi

### Multi-Line Plot (All Strategies Scaling)
- **Ne zaman:** `all` stratejisi + multi-config
- **Gösterir:** 3 stratejinin scaling davranışı karşılaştırması
- **Format:** Renkli çoklu çizgiler, legend ile

## 💡 Pratik Senaryolar

### Senaryo 1: Hangi strateji en iyi?
```bash
python src/analyze_ap_scaling.py all 25 10
```
**Sonuç:** 25 AP ve 10 user için en iyi stratejiyi görürsünüz.

### Senaryo 2: Equal Power stratejisinin scalability'si?
```bash
python src/analyze_ap_scaling.py equal_power --multi
```
**Sonuç:** AP sayısı arttıkça performansın nasıl değiştiğini görürsünüz.

### Senaryo 3: Hangi strategi en iyi scale oluyor?
```bash
python src/analyze_ap_scaling.py all --multi
```
**Sonuç:** 3 stratejinin scaling davranışını karşılaştırırsınız.

### Senaryo 4: Özel analiz (10, 15, 20, 25, 30 AP)
```bash
python src/analyze_ap_scaling.py all --multi --aps 10,15,20,25,30 --users 6
```
**Sonuç:** Özel AP aralığında 6 kullanıcılı analiz.

## 🔍 Key Findings Analizi

Script bittiğinde otomatik olarak özetler:

```
Key Findings:
  • Best Average Rate: Equal Power + All Serve (64 APs) - 95.42 Mbps
  • Best Energy Efficiency: Load Balancing (36 APs) - 5.23e+08 bits/J
  • Best QoS Satisfaction: Equal Power + All Serve (49 APs) - 100.0%
```

Bu size:
- En yüksek hız hangi stratejide
- En verimli konfigürasyon hangisi
- QoS için minimum AP sayısı

gibi bilgileri verir.

## 📈 Ölçülen Metrikler

1. **Average Rate per User** - Kullanıcı başına hız (Mbps)
2. **Total Network Throughput** - Toplam kapasitet (Mbps)
3. **Energy Efficiency** - Enerji verimliliği (bits/Joule)
4. **Average SINR** - Sinyal kalitesi (dB)
5. **QoS Satisfaction** - Minimum hız garantisi (%)
6. **Active APs** - Aktif AP sayısı
7. **Avg APs per User** - Kullanıcı başına ortalama AP

## 🐛 Hata Ayıklama

**Hata: "num_aps is required"**
```bash
# Single-config modunda AP sayısı zorunlu:
python src/analyze_ap_scaling.py nearest_ap 16 8
# VEYA multi-config kullanın:
python src/analyze_ap_scaling.py nearest_ap --multi
```

**Hata: "ModuleNotFoundError"**
```bash
# Environment'ı aktif edin:
conda activate 6g_project
```

**Grafik açılmıyor**
```python
# analyze_ap_scaling.py içinde plt.show() satırlarını yorum yapın
# Sadece save_path ile çalışır
```

## 🎓 İleri Seviye Kullanım

### Kendi Seed'iniz ile
```bash
python src/analyze_ap_scaling.py all --multi --seed 123
```

### Çok sayıda konfigürasyon
```bash
python src/analyze_ap_scaling.py all --multi --aps 5,10,15,20,25,30,35,40 --users 12
```

### Batch analiz (script)
```bash
#!/bin/bash
for users in 4 8 12 16; do
    python src/analyze_ap_scaling.py all --multi --users $users
done
```

## 📞 Yardım

Tüm seçenekleri görmek için:
```bash
python src/analyze_ap_scaling.py --help
```

## ✅ Hızlı Başlangıç

```bash
# 1. Temel karşılaştırma
python src/analyze_ap_scaling.py all 25 8

# 2. Scaling analizi
python src/analyze_ap_scaling.py all --multi

# 3. Sonuçları inceleyin
ls results/
```

3 komut ile tüm analizi yapabilirsiniz! 🎉
