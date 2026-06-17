# 🎙️ Interview Audio — Evaluation Schema

**Version:** 2.0 (Statistically Calibrated)  
**Calibrated on:** 86 real interview recordings (May 2026)  
**Scoring scale:** 0–100 overall | 0–5 per dimension

---

## Overview

Each candidate is scored across **6 dimensions**. Every dimension score (0–5) is computed from raw acoustic and linguistic features using **percentile-band scoring** — thresholds derived empirically from the distribution of 86 real interviews, not arbitrary heuristics.

| Dimension | Weight | Primary Signal |
|-----------|--------|---------------|
| Fluency | **20%** | Words per minute, filler rate, pause duration |
| Intelligibility | **20%** | ASR confidence (speech clarity) |
| Lexical Resource | **15%** | Vocabulary diversity (MATTR), rare word usage |
| Discourse | **15%** | Discourse connectors, high-quality connectors |
| Sentiment & Confidence | **15%** | Positive sentiment, assertiveness, hedging |
| Voice Modulation | **15%** | Pitch variation, voiced speech fraction |
| ~~Language Control~~ | ~~0%~~ | ~~Grammar errors~~ — *excluded (mocked, unreliable)* |

### Overall Score Formula

```
overall_score (0–100) = (
    fluency_score         × 0.20  +
    intelligibility_score × 0.20  +
    lexical_score         × 0.15  +
    discourse_score       × 0.15  +
    sentiment_score       × 0.15  +
    voice_score           × 0.15
) × 20
```

---

## Scoring Method: Percentile Bands

All features are scored using **5 performance bands** mapped to a 0–5 scale with linear interpolation within each band:

| Band | Percentile Range | Score Range | What it means |
|------|-----------------|-------------|---------------|
| 🔴 Poor | < 20th | 0.0 – 1.0 | Bottom 20% of all candidates |
| 🟠 Below Average | 20th – 40th | 1.0 – 2.0 | Below median |
| 🟡 Average | 40th – 60th | 2.0 – 3.0 | Around the median |
| 🟢 Good | 60th – 80th | 3.0 – 4.0 | Above median |
| ✅ Excellent | > 80th | 4.0 – 5.0 | Top 20% of all candidates |

> Scores are smoothly interpolated within each band — a candidate at exactly the 70th percentile will score 3.5, not a hard jump to 4.0.

---

## Dimension 1 — Fluency (Weight: 20%)

**What it measures:** How smoothly and at what pace the candidate communicates.

### Sub-features & Weights

| Feature | Direction | Weight | Percentile Thresholds (p20 / p40 / p60 / p80) |
|---------|-----------|--------|-----------------------------------------------|
| `fluency_wpm` (words per minute) | Higher → Better | 50% | 44.7 / 55.0 / 67.3 / 93.3 |
| `fluency_filler_rate` (fillers per word) | **Lower → Better** | 30% | 0.058 / 0.043 / 0.025 / 0.018 |
| `fluency_mean_pause_duration` (seconds) | **Lower → Better** | 20% | 8.8s / 5.9s / 4.4s / 2.8s |

### Band Reference Table

| Band | WPM | Filler Rate | Mean Pause |
|------|-----|-------------|------------|
| 🔴 Poor | < 44.7 | > 5.8% | > 8.8s |
| 🟠 Below Average | 44.7 – 55.0 | 4.3% – 5.8% | 5.9s – 8.8s |
| 🟡 Average | 55.0 – 67.3 | 2.5% – 4.3% | 4.4s – 5.9s |
| 🟢 Good | 67.3 – 93.3 | 1.8% – 2.5% | 2.8s – 4.4s |
| ✅ Excellent | > 93.3 | < 1.8% | < 2.8s |

### Notes
- Median WPM in our dataset is **61 WPM** — much lower than native conversational speech (130–160 WPM) because candidates think through answers in real-time.
- Filler rate is heavily right-skewed: most candidates have low fillers; a few outliers drag the mean up.
- Filler words detected: *um, uh, like, you know, basically, actually, literally, kind of, sort of, right, okay*.

---

## Dimension 2 — Intelligibility (Weight: 20%)

**What it measures:** How clearly the candidate's speech is understood by the ASR engine — a proxy for pronunciation clarity and articulation.

### Sub-feature

| Feature | Direction | Weight | Thresholds |
|---------|-----------|--------|-----------|
| `intelligibility_mean_confidence` (ASR word confidence, 0–1) | Higher → Better | 100% | 0.679 / 0.722 / 0.759 / 0.799 |

### Band Reference Table

| Band | Mean Confidence |
|------|----------------|
| 🔴 Poor | < 0.679 |
| 🟠 Below Average | 0.679 – 0.722 |
| 🟡 Average | 0.722 – 0.759 |
| 🟢 Good | 0.759 – 0.799 |
| ✅ Excellent | > 0.799 |

*Dataset stats: mean = 0.741, std = 0.079, Shapiro-Wilk p = 0.270 (normally distributed)*

### Dropped Features
- ~~`pronunciation_score`~~ — r = **0.979** correlation with `mean_confidence`. Perfectly redundant; using both double-counts the same signal.
- ~~`variance_confidence`~~ — r = **−0.908** with `mean_confidence`. Also redundant.

---

## Dimension 3 — Lexical Resource (Weight: 15%)

**What it measures:** Vocabulary richness and range — does the candidate use varied, sophisticated language?

### Sub-features & Weights

| Feature | Direction | Weight | Thresholds (p20/p40/p60/p80) |
|---------|-----------|--------|------------------------------|
| `lexical_mattr` (Moving Average Type-Token Ratio) | Higher → Better | 65% | 0.732 / 0.751 / 0.779 / 0.794 |
| `lexical_rare_word_ratio` (fraction of rare words) | Higher → Better | 35% | 0.101 / 0.132 / 0.155 / 0.173 |

### Band Reference Table

| Band | MATTR | Rare Word Ratio |
|------|-------|-----------------|
| 🔴 Poor | < 0.732 | < 0.101 |
| 🟠 Below Average | 0.732 – 0.751 | 0.101 – 0.132 |
| 🟡 Average | 0.751 – 0.779 | 0.132 – 0.155 |
| 🟢 Good | 0.779 – 0.794 | 0.155 – 0.173 |
| ✅ Excellent | > 0.794 | > 0.173 |

### Why MATTR (not TTR or unique word count)?
MATTR measures vocabulary diversity in a **sliding window** — it is **not biased by interview length**. A 5-minute response and a 30-minute response are compared fairly. TTR and raw unique-word count both inflate for longer interviews.

### Short Transcript Gate
If `total_words < 100`, the lexical score is scaled down proportionally. Very short responses do not provide enough text for reliable lexical diversity measurement.

### Dropped Features
- ~~`unique_words`~~ — r = 0.892 with total_words; length-driven, not a true richness signal.
- ~~`avg_word_frequency`~~ — removes discrimination; high-frequency words dominate in all professional speech.

---

## Dimension 4 — Discourse (Weight: 15%)

**What it measures:** Structural coherence — does the candidate use connective language to organise their thoughts?

### Sub-features & Weights

| Feature | Direction | Weight | Thresholds |
|---------|-----------|--------|-----------|
| `discourse_connector_count` (unique connector types used) | Higher → Better | 50% | 9 / 11 / 12 / 13 |
| `discourse_tier1_count` (high-quality connectors) | Higher → Better | 50% | 0 / 1 / 2–3 / ≥5 |

### Band Reference Table

| Band | Connector Count | Tier-1 Count |
|------|----------------|--------------|
| 🔴 Poor | < 9 | 0 |
| 🟠 Below Average | 9 – 11 | 1 |
| 🟡 Average | 11 – 12 | 2 – 3 |
| 🟢 Good | 12 – 13 | 3 – 4 |
| ✅ Excellent | > 13 | ≥ 5 |

### Connector Tiers

**Tier 1 — High-quality / Academic connectors** (weighted 2×):
> *furthermore, consequently, nevertheless, however, therefore, moreover, nonetheless, subsequently, conversely, accordingly, alternatively, predominantly, specifically, additionally*

**Tier 2 — Common connectors** (weighted 1×):
> *but, and, so, because, although, while, since, also, then, thus, yet, hence, whether, unless, meanwhile, besides, instead, despite, rather*

### Special Rule
If `tier1_count == 0`, the discourse score is **capped at 3.5** regardless of connector count — using only basic connectors (*and, but, so*) is insufficient for a Good+ rating.

---

## Dimension 5 — Sentiment & Confidence (Weight: 15%)

**What it measures:** Candidate's positivity, assertiveness, and avoidance of excessive hedging.

### Sub-features & Weights

| Feature | Direction | Weight | Thresholds (p20/p40/p60/p80) |
|---------|-----------|--------|------------------------------|
| `sentiment_mean_compound` (VADER compound, −1 to +1) | Higher → Better | 60% | 0.105 / 0.147 / 0.209 / 0.278 |
| `sentiment_assertive_count` (assertive phrase count) | Higher → Better | 40% | 2 / 3 / 5 / 9 |

### Band Reference Table

| Band | Mean Compound | Assertive Phrases |
|------|--------------|-------------------|
| 🔴 Poor | < 0.105 | ≤ 2 |
| 🟠 Below Average | 0.105 – 0.147 | 2 – 3 |
| 🟡 Average | 0.147 – 0.209 | 3 – 5 |
| 🟢 Good | 0.209 – 0.278 | 5 – 9 |
| ✅ Excellent | > 0.278 | > 9 |

### Hedging Penalty
If `sentiment_hedge_rate > 0.303` (above the 80th percentile), **subtract 0.5** from the sentiment score.

- Hedge rate = (hedging phrase count / total words)
- Common hedging phrases: *I think, I believe, I guess, maybe, perhaps, kind of, sort of, I'm not sure, possibly, probably*
- Excessive hedging signals lack of confidence

### Assertive Phrases Detected
> *I am, I have, I will, I can, I did, I led, I built, I achieved, I know, I believe strongly, definitely, certainly, absolutely, clearly, I am confident*

---

## Dimension 6 — Voice Modulation (Weight: 15%)

**What it measures:** Expressive range in speech — does the candidate speak in a monotone, or do they vary pitch to convey meaning?

### Sub-features & Weights

| Feature | Direction | Weight | Thresholds (p20/p40/p60/p80) |
|---------|-----------|--------|------------------------------|
| `voice_pitch_std` (pitch standard deviation, Hz) | Higher → Better | 70% | 42.3 / 47.3 / 51.6 / 59.1 |
| `voice_voiced_fraction` (fraction of audio that is voiced) | Higher → Better | 30% | 0.170 / 0.216 / 0.264 / 0.345 |

### Band Reference Table

| Band | Pitch Std (Hz) | Voiced Fraction |
|------|---------------|-----------------|
| 🔴 Poor (Monotone) | < 42.3 | < 0.170 |
| 🟠 Below Average | 42.3 – 47.3 | 0.170 – 0.216 |
| 🟡 Average | 47.3 – 51.6 | 0.216 – 0.264 |
| 🟢 Good | 51.6 – 59.1 | 0.264 – 0.345 |
| ✅ Excellent | > 59.1 | > 0.345 |

*Dataset stats: pitch_std mean = 50.6 Hz, std = 9.3 Hz, Shapiro-Wilk p = 0.250 (normally distributed)*

### Dropped Features
- ~~`pitch_mean`~~ — **gender-dependent**. Female speakers naturally have higher absolute pitch. Using `pitch_std` (variation around the speaker's own baseline) is gender-neutral.
- ~~`pitch_range`~~ — near-constant in our dataset (std = 1.07), likely a pipeline ceiling bug in Parselmouth. Excluded.

---

## Flags & Edge Cases

| Flag | Condition | Action |
|------|-----------|--------|
| ⚠️ Short Response | `duration < 10 min` | Lexical score unreliable — lower confidence |
| 🤖 Scripted Speech | `fluency_ai_scripted_score > 0` | Flag for manual review; apply 10% penalty to fluency |
| 🔇 Excessive Hedging | `sentiment_hedge_rate > 0.303` | −0.5 penalty on sentiment score |
| 📊 Outlier Candidate | > 3 features at \|z\| > 3 | Flag for manual review |
| ❌ No Tier-1 Connectors | `discourse_tier1_count == 0` | Cap discourse score at 3.5 |
| 📁 Archived Recording | 403 Forbidden on download | Exclude from scoring; mark status = skipped |

---

## Overall Score Bands (0–100)

| Band | Score Range | Recommendation |
|------|-------------|----------------|
| 🔴 Poor | 0 – 40 | Do not advance |
| 🟠 Below Average | 40 – 55 | Strong pass-not-recommend |
| 🟡 Average | 55 – 65 | Consider for next round |
| 🟢 Good | 65 – 75 | Recommend for interview |
| ✅ Excellent | 75 – 100 | Priority candidate |

*Based on observed score distribution: range 44.2 – 86.6, median ~68, mean ~67 across 86 calibration candidates.*

---

## Calibration Limitations

| Limitation | Impact |
|-----------|--------|
| n = 86 recordings | Thresholds are indicative — recalibrate after 500+ recordings |
| Grammar check mocked (all = 0) | Language control excluded from scoring until a working grammar checker is integrated |
| No human ground-truth scores | Cannot validate feature→score correlation without annotated benchmark |
| Diarization removed | New pipeline (MongoDB) processes pre-separated audio — no diarization bias |
| 15 archived recordings excluded (403) | May represent a non-random subset; older interviews |

---

## Feature Redundancy Map

The following pairs are **highly correlated (|r| ≥ 0.65)** — only one from each pair is used in scoring:

| Kept | Dropped | Correlation | Reason |
|------|---------|-------------|--------|
| `mean_confidence` | `pronunciation_score` | r = **+0.979** | Same signal |
| `mean_confidence` | `variance_confidence` | r = **−0.908** | Inverse of same signal |
| `fluency_wpm` | `voiced_fraction` (in fluency) | r = **+0.959** | WPM already captures speaking density |
| `lexical_mattr` | `lexical_unique_words` | r = **+0.892** | Both length-driven; MATTR is length-corrected |
| `sentiment_hedge_rate` | `sentiment_hedging_count` | r = **+0.824** | Rate is normalized; count is not |

---

*Schema version 2.0 — calibrated May 2026 on 86 interview recordings. Next recalibration target: 500 recordings.*
