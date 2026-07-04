# 🎙️ Interview Feature Analysis — Statistical Report for Evaluation Schema Design

**Dataset:** 86 successful interview recordings  
**Analysis Date:** May 2026  
**Goal:** Use descriptive statistics, normality testing, and percentile analysis to design a data-driven evaluation rubric for interview candidates.

---

## 1. Dataset Overview

| Metric | Value |
|--------|-------|
| Total records processed | 101 |
| Successfully extracted | **86** (85.1%) |
| Failed / archived recordings | 15 (14.9%) |
| Mean interview duration | **26.0 minutes** |
| Median interview duration | 26.2 minutes |
| Std of duration | 8.1 minutes |
| Range | 2.2 min – 43.5 min |
| P10 duration | 15.0 min |
| P90 duration | 36.0 min |

> **Note:** Interview length varies significantly. Shorter recordings (< 10 min) may produce less reliable feature estimates, particularly for lexical diversity (MATTR, TTR) and discourse connector counts.

---

## 2. Normality Analysis — Which Features Follow a Bell Curve?

Before deciding on scoring thresholds, we tested every feature for normality using the **Shapiro-Wilk test** (α = 0.05).

- **Normally distributed** → thresholds can be set using **z-scores** (mean ± std)
- **Non-normal / skewed** → thresholds must use **empirical percentiles** (p20/p40/p60/p80)

### ✅ Normal Features (z-score thresholds valid)

| Feature | Shapiro W | p-value | Skewness |
|---------|-----------|---------|----------|
| `intel_confidence` (ASR confidence) | 0.982 | 0.270 | −0.32 |
| `pronunciation_score` | 0.988 | 0.623 | +0.11 |
| `discourse_connectors` | 0.972 | 0.061 | −0.10 |
| `pitch_mean` | 0.973 | 0.068 | +0.26 |
| `pitch_std` | 0.981 | 0.250 | +0.23 |
| `lexical_ttr` | 0.983 | 0.315 | +0.32 |
| `lexical_mattr` | 0.982 | 0.288 | −0.34 |
| `lexical_unique` | 0.983 | 0.298 | +0.33 |
| `lexical_rare` | 0.990 | 0.726 | −0.09 |
| `duration_s` | 0.988 | 0.651 | +0.09 |

### ❌ Non-Normal Features (percentile thresholds required)

| Feature | p-value | Skewness | Interpretation |
|---------|---------|----------|---------------|
| `fluency_wpm` | 0.000 | **+1.40** | Right-skewed — most candidates speak slowly; a few speak fast |
| `fluency_filler_rate` | 0.000 | **+3.65** | Heavily right-skewed — most have few fillers, outliers many |
| `fluency_pause_dur` | 0.000 | **+1.46** | Right-skewed — some candidates have very long pauses |
| `fluency_pause_freq` | 0.043 | +0.43 | Mildly skewed |
| `discourse_tier1` | 0.000 | +0.82 | Most candidates use few high-tier connectors |
| `discourse_tier2` | 0.000 | **+1.78** | Usage varies widely |
| `sentiment_compound` | 0.000 | **+1.92** | Mostly neutral/positive, skewed right |
| `sentiment_assertive` | 0.000 | **+2.67** | Most use few assertive phrases |
| `sentiment_hedge_rate` | 0.000 | **+4.20** | Extreme right skew — most don't hedge at all |
| `voiced_fraction` | 0.000 | +1.46 | Right-skewed speaking density |
| `pitch_range` | 0.000 | −2.36 | Very tight cluster with outliers below |

> **Key Insight:** Most fluency and sentiment features are right-skewed. The majority of candidates cluster at the lower end of each range — percentile-based thresholds will better reflect real performance differences.

---

## 3. Descriptive Statistics & Percentile Thresholds

### 3.1 Fluency

#### Words Per Minute (`fluency_wpm`)
| mean=70.6 | median=61.1 | std=38.0 | skew=+1.40 | ❌ Non-normal |

| Band | WPM Range | Percentile |
|------|-----------|------------|
| 🔴 Poor | < 44.7 WPM | < p20 |
| 🟠 Below Average | 44.7 – 55.0 WPM | p20–p40 |
| 🟡 Average | 55.0 – 67.3 WPM | p40–p60 |
| 🟢 Good | 67.3 – 93.3 WPM | p60–p80 |
| ✅ Excellent | > 93.3 WPM | > p80 |

> Low median (61 WPM) reflects deliberate pacing in interview contexts. Native professional rate is 130–160 WPM, but candidates responding in a second language or thinking through answers will naturally speak slower.

#### Filler Rate (`fluency_filler_rate`) — Lower is Better
| mean=0.040 | median=0.034 | std=0.032 | skew=+3.65 | ❌ Non-normal |

| Band | Filler Rate |
|------|------------|
| ✅ Excellent | < 0.018 |
| 🟢 Good | 0.018 – 0.025 |
| 🟡 Average | 0.025 – 0.043 |
| 🟠 Below Average | 0.043 – 0.058 |
| 🔴 Poor | > 0.058 |

#### Mean Pause Duration (`fluency_pause_dur`) — Lower is Better
| mean=6.2s | median=5.0s | std=4.3s | skew=+1.46 | ❌ Non-normal |

| Band | Mean Pause Duration |
|------|---------------------|
| ✅ Excellent | < 2.8s |
| 🟢 Good | 2.8 – 4.4s |
| 🟡 Average | 4.4 – 5.9s |
| 🟠 Below Average | 5.9 – 8.8s |
| 🔴 Poor | > 8.8s |

#### Pause Frequency (`fluency_pause_freq`) — Higher is Better (more structured pausing)
| mean=146.8 | median=139.0 | std=65.2 |

| Band | Pause Count |
|------|------------|
| 🔴 Poor | < 86 |
| 🟠 Below Average | 86 – 116 |
| 🟡 Average | 116 – 164 |
| 🟢 Good | 164 – 205 |
| ✅ Excellent | > 205 |

---

### 3.2 Intelligibility

#### ASR Confidence (`intel_confidence`) — ✅ Normal
| mean=0.741 | median=0.743 | std=0.079 | Shapiro p=0.270 |

| Band | Confidence | Z-score |
|------|-----------|---------|
| 🔴 Poor | < 0.679 | z < −0.8 |
| 🟠 Below Average | 0.679 – 0.722 | −0.8 to −0.2 |
| 🟡 Average | 0.722 – 0.759 | ±0.2 |
| 🟢 Good | 0.759 – 0.799 | +0.2 to +0.7 |
| ✅ Excellent | > 0.799 | z > +0.7 |

#### Pronunciation Score (`pronunciation_score`) — ✅ Normal
| mean=0.625 | median=0.619 | std=0.113 | Shapiro p=0.623 |

| Band | Score |
|------|-------|
| 🔴 Poor | < 0.537 |
| 🟠 Below Average | 0.537 – 0.600 |
| 🟡 Average | 0.600 – 0.651 |
| 🟢 Good | 0.651 – 0.706 |
| ✅ Excellent | > 0.706 |

> ⚠️ `intel_confidence` and `pronunciation_score` have **r = 0.979** — almost perfectly correlated. Use only ONE in the rubric (recommend `intel_confidence`).

---

### 3.3 Lexical Resource

#### MATTR — Moving Average Type-Token Ratio (`lexical_mattr`) — ✅ Normal
| mean=0.762 | median=0.764 | std=0.040 | Shapiro p=0.288 |

| Band | MATTR |
|------|-------|
| 🔴 Poor | < 0.732 |
| 🟠 Below Average | 0.732 – 0.751 |
| 🟡 Average | 0.751 – 0.779 |
| 🟢 Good | 0.779 – 0.794 |
| ✅ Excellent | > 0.794 |

> MATTR is preferred over TTR or unique word count — it is **not biased by interview length**.

#### Rare Word Ratio (`lexical_rare`) — ✅ Normal
| mean=0.141 | median=0.145 | std=0.041 | Shapiro p=0.726 |

| Band | Rare Word Ratio |
|------|----------------|
| 🔴 Poor | < 0.101 |
| 🟠 Below Average | 0.101 – 0.132 |
| 🟡 Average | 0.132 – 0.155 |
| 🟢 Good | 0.155 – 0.173 |
| ✅ Excellent | > 0.173 |

---

### 3.4 Discourse

#### Connector Count (`discourse_connectors`) — ✅ Normal
| mean=11.4 | median=11.5 | std=2.4 | Shapiro p=0.061 |

| Band | Count |
|------|-------|
| 🔴 Poor | < 9 |
| 🟠 Below Average | 9 – 11 |
| 🟡 Average | 11 – 12 |
| 🟢 Good | 12 – 13 |
| ✅ Excellent | > 13 |

> ⚠️ Very narrow range (std=2.4). Weak differentiator alone. Combine with Tier-1 connectors for better signal.

#### Tier-1 Connectors (`discourse_tier1`) — Academic/High-quality connectors
| mean=2.5 | median=2.0 | std=2.3 | ❌ Non-normal |

| Band | Tier-1 Count |
|------|-------------|
| 🔴 Poor | 0 |
| 🟠 Below Average | 1 |
| 🟡 Average | 2–3 |
| 🟢 Good | 3–4 |
| ✅ Excellent | ≥ 5 |

---

### 3.5 Sentiment & Confidence

#### Compound Sentiment (`sentiment_compound`) — ❌ Non-normal
| mean=0.207 | median=0.174 | std=0.131 | skew=+1.92 |

| Band | Score |
|------|-------|
| 🔴 Poor | < 0.105 |
| 🟠 Below Average | 0.105 – 0.147 |
| 🟡 Average | 0.147 – 0.209 |
| 🟢 Good | 0.209 – 0.278 |
| ✅ Excellent | > 0.278 |

#### Assertiveness (`sentiment_assertive`) — ❌ Non-normal
| mean=6.3 | median=4.0 | std=7.8 | skew=+2.67 |

| Band | Assertive Phrases |
|------|-----------------|
| 🔴 Poor | ≤ 2 |
| 🟠 Below Average | 2 – 3 |
| 🟡 Average | 3 – 5 |
| 🟢 Good | 5 – 9 |
| ✅ Excellent | > 9 |

> **Hedge rate** (mean=0.193, skew=+4.20): Use as a **penalty flag** — hedge_rate > 0.303 (p80) signals excessive uncertainty and should reduce sentiment score.

---

### 3.6 Voice Modulation

#### Pitch Std / Expressiveness (`pitch_std`) — ✅ Normal
| mean=50.6 Hz | median=49.9 Hz | std=9.3 Hz | Shapiro p=0.250 |

| Band | Pitch Std (Hz) | Interpretation |
|------|---------------|---------------|
| 🔴 Poor | < 42.3 | Monotone |
| 🟠 Below Average | 42.3 – 47.3 | Low variation |
| 🟡 Average | 47.3 – 51.6 | Normal |
| 🟢 Good | 51.6 – 59.1 | Expressive |
| ✅ Excellent | > 59.1 | Highly expressive |

> ⚠️ `pitch_mean` (absolute Hz) is **gender-dependent** — avoid using it as a scoring feature directly. Use `pitch_std` (variation) which is gender-neutral.

#### Voiced Fraction (`voiced_fraction`) — ❌ Non-normal
| mean=0.273 | median=0.239 | std=0.150 | skew=+1.46 |

| Band | Voiced Fraction |
|------|----------------|
| 🔴 Poor | < 0.170 |
| 🟠 Below Average | 0.170 – 0.216 |
| 🟡 Average | 0.216 – 0.264 |
| 🟢 Good | 0.264 – 0.345 |
| ✅ Excellent | > 0.345 |

---

## 4. Feature Redundancy — Correlation Analysis

Pairs with |r| ≥ 0.65 measure the same underlying behaviour. **Use only one from each pair.**

| Feature A | Feature B | r | Recommendation |
|-----------|-----------|---|---------------|
| `intel_confidence` | `pronunciation_score` | **+0.979** | Drop `pronunciation_score` |
| `fluency_wpm` | `voiced_fraction` | **+0.959** | Keep `wpm`; drop `voiced_fraction` from fluency |
| `fluency_total_words` | `discourse_tier2` | **+0.951** | Both length-driven; normalize by duration |
| `intel_confidence` | `intel_var_conf` | **−0.908** | Keep `intel_confidence`; drop `var_conf` |
| `fluency_total_words` | `lexical_unique` | **+0.892** | Drop `lexical_unique`; use `mattr` |
| `sentiment_hedging` | `sentiment_hedge_rate` | **+0.824** | Keep `hedge_rate` (rate is normalized) |
| `discourse_connectors` | `lexical_unique` | **+0.836** | Both driven by total talk time |
| `fluency_wpm` | `fluency_pause_dur` | **−0.725** | Expected inverse — both valid if scored correctly |

---

## 5. Recommended Evaluation Schema

### Scoring Dimensions & Primary Features

| Dimension | Primary Feature | Secondary | Method |
|-----------|----------------|-----------|--------|
| **Fluency** | `fluency_wpm` | `fluency_filler_rate` (inverted) | Percentile |
| **Intelligibility** | `intel_confidence` | `fluency_pause_dur` (inverted) | Z-score |
| **Lexical Resource** | `lexical_mattr` | `lexical_rare` | Z-score |
| **Discourse** | `discourse_connectors` | `discourse_tier1` | Z-score + Percentile |
| **Sentiment** | `sentiment_compound` | `sentiment_assertive` | Percentile |
| **Voice Modulation** | `pitch_std` | `voiced_fraction` | Z-score |

### Band-to-Score Mapping (0–5 scale per dimension)

| Band | Score | Percentile |
|------|-------|------------|
| Poor | 0 – 1 | < p20 |
| Below Average | 1 – 2 | p20 – p40 |
| Average | 2 – 3 | p40 – p60 |
| Good | 3 – 4 | p60 – p80 |
| Excellent | 4 – 5 | > p80 |

### Composite Score Formula (0–100)

```
overall_score = (
    fluency_score         × 0.20 +
    intelligibility_score × 0.20 +
    lexical_score         × 0.15 +
    discourse_score       × 0.15 +
    sentiment_score       × 0.15 +
    voice_score           × 0.15
) × 20
```

> **Weight rationale:** Fluency and Intelligibility weighted highest (20% each) — most objectively measurable and most predictive of communication ability. All other dimensions contribute equally at 15% each.

---

## 6. Flags & Edge Cases

| Flag | Condition | Action |
|------|-----------|--------|
| Short response | duration < 10 min | Lower confidence on lexical scores |
| Scripted speech | `fluency_scripted_score` > 0 | Apply penalty; flag for review |
| Excessive hedging | `sentiment_hedge_rate` > 0.303 (p80) | Reduce sentiment score |
| Pitch range anomaly | `pitch_range` near constant | Known pipeline issue — exclude from scoring |
| Outlier candidate | > 3 features at \|z\| > 3 | Flag for manual review |

---

## 7. Data Limitations

| Limitation | Impact |
|-----------|--------|
| n = 86 recordings | Thresholds are indicative; re-calibrate after 500+ recordings |
| Grammar check mocked (all = 0) | Language control dimension cannot be scored yet |
| No ground-truth scores | Cannot validate feature→score mapping without human-annotated calibration set |
| `pitch_range` near-constant | Likely a pipeline ceiling issue in parselmouth — investigate |
| 15 archived recordings excluded | May represent a non-random subset (older interviews) |
