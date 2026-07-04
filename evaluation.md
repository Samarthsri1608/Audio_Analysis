# AI Speaking Assessment Engine — Evaluation Criteria

This document explains every dimension on which a candidate is evaluated, what each feature means, how it is extracted from the audio, and how it is converted into a score between **0 and 5** (5 = best).

---

## Scoring System Overview

| Property | Detail |
|---|---|
| **Score range** | 0.0 – 5.0 per dimension |
| **Primary method** | Trained ML models (`{dim}_model.pkl`) |
| **Fallback method** | Discriminative heuristic rules (v2.1) |
| **Input** | Raw audio file — transcribed via OpenAI Whisper |
| **Overall Score** | Weighted sum out of 100 |
| **Dimensions** | 7 (Fluency, Intelligibility, Language Control, Lexical Resource, Discourse, Voice Modulation, Sentiment) |

---

## Overall Assessment Weightage

The final score is calculated as a weighted average of individual dimensions, then scaled to **100**. Dimensions are weighted by their linguistic grounding and signal reliability.

| Dimension | Weight | Primary Signal |
|---|---|---|
| **Fluency** | 0.22 (22%) | Pause density & WPM |
| **Intelligibility** | 0.22 (22%) | Pronunciation clarity (% high-conf words) |
| **Language Control** | 0.20 (20%) | Grammar error density |
| **Lexical Resource** | 0.18 (18%) | MATTR Diversity & Sophistication |
| **Discourse** | 0.12 (12%) | Weighted connector density |
| **Voice Modulation** | 0.03 (3%) | Pitch standard deviation |
| **Sentiment & confidence**| 0.03 (3%) | Assertive vs hedging ratio |

---

---

## 1. Fluency

### What It Means
Fluency measures how smoothly and naturally a candidate speaks — their pace, flow, and the absence of unnecessary hesitation or filler words. A fluent speaker maintains a comfortable, consistent speed without stumbling, over-pausing, or leaning on verbal crutches.

### How Features Are Extracted
The audio is transcribed by Whisper, which returns per-word timestamps. The following metrics are derived:

| Feature | Definition | How It Is Computed |
|---|---|---|
| `wpm` | Words Per Minute | Total word count ÷ audio duration in minutes |
| `pause_frequency` | Number of pauses | Count of inter-word gaps ≥ **0.3 seconds** |
| `mean_pause_duration` | Average pause length | Mean of all detected pause durations (seconds) |
| `filler_count` | Number of filler word occurrences | Matches against a fixed lexicon of filler words and phrases |
| `filler_rate` | Proportion of words that are fillers | `filler_count / total_words` |
| `filler_words_found` | Which fillers were used | Deduplicated set of filler types detected |

**Filler word lexicon includes:** *um, uh, uhh, umm, hmm, like, basically, literally, actually, you know, I mean, sort of, kind of, right, okay, so, well*

### Scoring Formula (Heuristic v2.1)
```
# Pause density: 4–10 pauses/min = optimal
if pauses/min <= 4:
    pause_score = 1.0 + (pauses/min / 4.0) * 2.0    # Scripted penalty
elif pauses/min <= 10:
    pause_score = 3.0 + (pauses/min - 4.0) / 6.0 * 2.0
else:
    pause_score = 5.0 - (pauses/min - 10.0) / 8.0 * 4.0

# WPM Penalty: Penalty for being too slow (<80) or too fast (>170)
wpm_penalty = clamp((80 - wpm)/80 * 2.0) if wpm < 80 else clamp((wpm - 170)/70 * 1.5)

# Filler Penalty: 5% filler rate = −1 point
filler_penalty = filler_rate * 20.0

fluency_score = clamp(pause_score - wpm_penalty - filler_penalty, 0, 5)
```

### Interpretation
| Score | Meaning |
|---|---|
| 5.0 | 140–165 WPM, zero fillers, natural pausing |
| 3.0–4.0 | Acceptable pace, occasional fillers or pauses |
| 1.0–2.0 | Very slow/fast, frequent fillers, or excessive pausing |

---

## 2. Intelligibility

### What It Means
Intelligibility measures how clearly and accurately the candidate pronounces words — whether a listener (or the speech recognition engine) can understand them without difficulty. Poor intelligibility may stem from a strong accent, mumbling, unclear articulation, or mispronunciation.

### How Features Are Extracted
Whisper assigns a `probability` value to every recognised word — this is its confidence that it transcribed the word correctly. Low confidence implies the word was hard to understand.

| Feature | Definition | How It Is Computed |
|---|---|---|
| `mean_confidence` | Average ASR confidence across all words | Mean of per-word Whisper probability scores (0–1) |
| `variance_confidence` | Inconsistency in clarity | Variance of per-word confidence scores |
| `pronunciation_score` | Fraction of clearly-spoken words | % of words with confidence ≥ **0.75** |
| `mispronounced_words` | Words that were unclear | All words with confidence < 0.75, sorted worst-first |
| `low_confidence_flag` | Overall poor intelligibility | `True` if `mean_confidence < 0.60` |

### Scoring Formula (Heuristic v2.1)
```
# Pronunciation score: fraction of words with confidence >= 0.75
# Map 0.60 -> 1.0, 1.0 -> 5.0
pron_component = clamp(1.0 + (pronunciation_score - 0.60) / 0.40 * 4.0)

# Variance penalty: high variance (>0.05) means inconsistent clarity
var_penalty = clamp(variance_confidence / 0.05 * 1.5, 0, 1.5)

intelligibility_score = clamp(pron_component - var_penalty, 0, 5)
```

**Weights:** Mean confidence contributes 60%; pronunciation score (fraction of clear words) contributes 40%.

### Interpretation
| Score | Meaning |
|---|---|
| 5.0 | Near-perfect clarity; Whisper understands every word with high confidence |
| 3.0–4.0 | Mostly clear with a few ambiguous words |
| 1.0–2.0 | Significant pronunciation issues; many words flagged as unclear |

---

## 3. Language Control

### What It Means
Language control measures grammatical accuracy — whether the candidate uses correct sentence structure, verb agreement, tense, and syntax. It reflects the candidate's command over English grammar as used in spoken communication.

### How Features Are Extracted
The Whisper transcript is run through `LanguageTool` (an open-source grammar checker). Because the input is a *spoken* transcript (no punctuation, informal register), the following error categories are **excluded** from scoring to avoid false positives:

| Excluded Category | Reason |
|---|---|
| `PUNCTUATION` | Speech has no punctuation |
| `TYPOGRAPHY` | Irrelevant to spoken output |
| `CASING` | Transcripts are lower-case by default |
| `STYLE` | Style is subjective for spoken English |
| `REDUNDANCY` | Too many false positives in informal speech |
| `MISC` | Covers trivial/debatable rules |

Only **GRAMMAR-category** errors are counted and used for scoring.

| Feature | Definition |
|---|---|
| `grammar_error_count` | Number of real grammar violations (used for scoring) |
| `error_count` | Total errors across all categories (for reference only) |
| `errors` | List of error details: message, context, rule ID, suggestions |

### Scoring Formula (Heuristic v2.1)
```
error_density = grammar_error_count / total_words
# 0% errors -> 5.0,  >= 8% errors -> 1.0
lc_score = 5.0 - (error_density / 0.08) * 4.0

# Fragment penalty: avg sentence < 5 words = penalty
if avg_sentence_length < 5:
    lc_score -= (5 - avg_sentence_length) * 0.3

language_control_score = clamp(lc_score, 0, 5)
```

### Interpretation
| Score | Meaning |
|---|---|
| 5.0 | Zero grammar errors |
| 3.0–4.0 | Minor errors (~3–8% error density) |
| 1.0–2.0 | Frequent grammar mistakes (≥10% error density) |

**Example:** 10 grammar errors in a 200-word answer = 5% density → score ≈ 3.7

---

## 4. Lexical Resource

### What It Means
Lexical resource assesses the richness, variety, and sophistication of the candidate's vocabulary. A high-scoring candidate uses a broad and varied set of words, including some advanced or domain-specific terms, rather than repeating simple, common words.

### How Features Are Extracted
The transcript is processed using **spaCy** (for tokenisation and filtering) and **wordfreq** (for word frequency lookup in English).

| Feature | Definition | How It Is Computed |
|---|---|---|
| `mattr` | Moving Average Type-Token Ratio | Lexical diversity computed over a sliding 50-word window — length-invariant |
| `type_token_ratio` | Plain TTR (fallback) | Unique words ÷ total words |
| `rare_word_ratio` | Vocabulary sophistication | % of content words rarer than 1-in-100,000 in English |
| `avg_word_frequency` | Average word commonness | Mean `wordfreq` frequency of content words |
| `sophisticated_words_sample` | Example advanced words used | Up to 10 rare content words found in the response |

> **Why MATTR over plain TTR?**  
> Plain TTR is biased against longer responses (naturally more repetition in longer speech). MATTR uses a sliding 50-word window and averages per-window TTRs, making it length-invariant and fairer across responses of different lengths.

Stopwords (*the, a, and, is, I, you, …*) are excluded from vocabulary sophistication scoring — only meaningful content words are evaluated.

### Scoring Formula (Heuristic v2.1)
```
# MATTR: Tightened range [0.45 -> 1.0, 0.72 -> 5.0]
mattr_component = clamp(1.0 + (mattr - 0.45) / (0.72 - 0.45) * 4.0)

# Sophistication: 15% rare content words -> 5.0
rare_component  = clamp(1.0 + (rare_word_ratio / 0.15) * 4.0)

# Frequency penalty: common-word usage over 0.005 threshold
freq_penalty    = clamp((avg_word_freq - 0.005) / 0.005 * 0.8)

lexical_resource_score = clamp(0.55 * mattr_component + 0.45 * rare_component - freq_penalty, 0, 5)
```

**Weights:** MATTR contributes 60%; rare word ratio contributes 40%.

### Interpretation
| Score | Meaning |
|---|---|
| 5.0 | High diversity (MATTR ≥ 0.70) and ≥20% advanced vocabulary |
| 3.0–4.0 | Moderate variety with some sophisticated words |
| 1.0–2.0 | Repetitive, limited vocabulary; mostly common words |

---

## 5. Discourse

### What It Means
Discourse measures how well-organised and logically structured the candidate's response is. A strong discourse score indicates the candidate connects ideas explicitly, uses transitions effectively, and guides the listener through a coherent argument or narrative.

### How Features Are Extracted
The transcript is searched (case-insensitive) for discourse connectors — words and phrases that signal logical relationships between ideas.

| Feature | Definition |
|---|---|
| `connector_count` | Total occurrences of discourse connectors |
| `connectors_used` | Set of unique connectors detected |

### Connector Quality Tiers
Connectors are weighted by quality to reward more sophisticated language use:

| Tier | Weight | Examples |
|---|---|---|
| **Tier 1** — Logical/Academic | ×2 | *therefore, consequently, furthermore, nevertheless, nonetheless, in contrast, on the other hand, in conclusion, to summarise, as a result, moreover, however, although, whereas, subsequently, in particular, specifically, for instance* |
| **Tier 2** — Basic/Conversational | ×1 | *because, so, but, also, and, then, first, second, finally, next, additionally, though, yet, still, thus* |

### Scoring Formula (Heuristic v2.1)
```
# weighted_count = Tier1_count * 2.0 + Tier2_count * 1.0
# variety_bonus  = 0.08 per unique connector, capped at 0.5

wdph = weighted_count / total_words * 100
disc_score = 1.0 + (wdph / 4.0) * 3.0 + variety_bonus

# Ceiling: If no Tier-1 connectors used, cap score at 3.5
if tier1_count == 0:
    disc_score = min(disc_score, 3.5)

discourse_score = clamp(disc_score, 0, 5)
```

### Interpretation
| Score | Meaning |
|---|---|
| 5.0 | Rich use of varied Tier-1 connectors throughout the response |
| 3.0–4.0 | Several connectors used, some Tier-1; decent structure |
| 1.0–2.0 | Minimal connectors; response feels disjointed or unstructured |

> **Quality over quantity:** Using *"therefore"* once counts the same as using *"so"* twice — the system rewards academic/logical connectors more highly.

---

## 6. Voice Modulation

### What It Means
Voice modulation measures the expressiveness and dynamism of the candidate's voice. A well-modulated speaker varies their pitch naturally — rising for questions, falling for conclusions, emphasising key words. A flat, monotone delivery signals lack of engagement or confidence.

### How Features Are Extracted
The raw audio file is analysed using **Praat** (via the `parselmouth` Python library), which extracts the fundamental frequency (F0/pitch) frame by frame at 10 ms intervals.

| Feature | Definition | How It Is Computed |
|---|---|---|
| `pitch_mean` | Average speaking pitch (Hz) | Mean F0 across all voiced frames |
| `pitch_std` | Vocal expressiveness (Hz) | Standard deviation of F0 — **core metric** |
| `pitch_range` | Pitch span (Hz) | Max F0 − Min F0 across voiced frames |
| `voiced_fraction` | Proportion of audio that is speech | Voiced frames ÷ total frames |
| `monotone_flag` | Flat delivery detected | `True` if `pitch_std < 20 Hz` |

**Analysis settings:** pitch floor 75 Hz, ceiling 500 Hz (covers both male and female speakers).

### Scoring Formula
```
vm_score = 1.0 + (pitch_std - 20.0) / (60.0 - 20.0) × 4.0   # 20 Hz → 1.0,  ≥60 Hz → 5.0

if voiced_fraction < 0.30:
    vm_score -= 1.0     # penalty: candidate spoke too little / mostly silence

voice_modulation_score = clamp(vm_score, 0, 5)
```

### Interpretation
| Score | Meaning |
|---|---|
| 5.0 | Highly expressive; pitch varies by ≥60 Hz std dev |
| 3.0–4.0 | Natural variation; moderately engaging delivery |
| 1.0–2.0 | Monotone or near-monotone; flat, unexpressive voice |

> **Why pitch standard deviation?** It is the most reliable single indicator of vocal expressiveness. A naturally engaging speaker's pitch fluctuates significantly; a nervous or disengaged speaker tends to speak in a narrow, flat range.

---

## 7. Sentiment & Confidence

### What It Means
Sentiment & Confidence evaluates the *emotional tone* and *self-assurance* of a candidate's spoken response. It does **not** simply reward positivity — in an interview context, professional neutrality is perfectly acceptable. Instead, it penalises clear negativity (self-doubt, complaining), rewards moderate positive engagement, and strongly rewards confident, assertive language while penalising excessive hedging.

> ⚠️ **Important design principle:** "More positive ≠ better." A perfectly neutral technical answer scores well. Overly effusive or artificially positive responses do not.

### The Three Sub-Signals

| Signal | Weight | What It Captures |
|---|---|---|
| **Positivity** | 40% | Overall emotional tone — moderately positive is ideal |
| **Confidence** | 40% | Assertive vs. hedging language ratio |
| **Composure** | 20% | Emotional stability across the full response |

### How Features Are Extracted
The transcript text is analysed using **VADER** (Valence Aware Dictionary and sEntiment Reasoner) from NLTK — a rule-based sentiment analyser designed for conversational/informal text, sentence by sentence.

| Feature | Definition | How It Is Computed |
|---|---|---|
| `mean_compound` | Overall tone (−1 to 1) | Mean VADER compound score across all sentences |
| `std_compound` | Emotional consistency | Standard deviation of sentence-level compound scores |
| `neg_sentiment_ratio` | Fraction of negative sentences | % of sentences with VADER `neg` component > 0.3 |
| `positive_ratio` | Fraction of positive sentences | % of sentences with VADER `pos` component > 0.3 |
| `assertive_count` | Confident language | Count of unique assertive phrases detected |
| `hedging_count` | Uncertain language | Count of unique hedging phrases detected |
| `hedge_rate` | Hedging density | Hedging phrases per 100 words |
| `assertive_phrases_found` | Which assertive phrases used | Deduplicated list |
| `hedging_phrases_found` | Which hedging phrases used | Deduplicated list |

**Hedging phrases (penalised):** *"I guess", "I suppose", "maybe", "perhaps", "sort of", "kind of", "I'm not sure", "not sure", "I don't know", "probably", "might be", "could be", "I feel like", "hopefully", "I assume"*

**Assertive phrases (rewarded):** *"I know", "I am confident", "definitely", "certainly", "I led", "I built", "I achieved", "I demonstrated", "I managed", "I delivered", "I have experience", "my expertise", "I successfully", "I proved"*

### Scoring Formula
```
# Signal 1 — Positivity: peaks at compound ≈ 0.35 (moderately positive)
positivity = 5.0 - |mean_compound - 0.35| / 0.35 × 3.0
positivity -= neg_sentiment_ratio × 4.0     # heavy negativity = big deduction

# Signal 2 — Confidence
confidence = 2.5 + assertive_count × 0.5 - hedge_rate × 0.4

# Signal 3 — Composure: std_compound 0.0 → 5.0,  ≥0.5 → 1.0
composure = 5.0 - (std_compound / 0.5) × 4.0

sentiment_score = clamp(0.4 × positivity + 0.4 × confidence + 0.2 × composure, 0, 5)
```

### Interpretation
| Score | Meaning |
|---|---|
| 5.0 | Moderately positive tone, multiple assertive phrases, emotionally consistent |
| 3.0–4.0 | Professional neutral-to-positive, some hedging, mostly composed |
| 1.0–2.0 | Heavy self-doubt or negativity, frequent hedging, erratic emotional variation |

### Worked Examples

| Response Type | Expected Score | Reason |
|---|---|---|
| *"I led the redesign of our entire data pipeline and delivered it two weeks ahead of schedule."* | ~4.5 | Strongly assertive, positive, stable |
| *"The system architecture uses event sourcing and eventual consistency."* | ~3.5 | Neutral but professional — not penalised |
| *"I'm not sure... I think maybe it could work, but I'm not confident."* | ~1.5 | Multiple hedges, low compound, lacks confidence |
| *"I absolutely love everything about this job and it's the best company ever!"* | ~3.0 | Overly effusive — compound too high, sounds rehearsed |

---

## 8. Intensive Testing Results (March 2026)

The engine was subjected to intensive testing using a benchmark suite of **5 full-length interview recordings** (approx. 4.3 hours of audio total) to validate heuristic v2.1 and reproducibility.

### Performance Benchmark Summary

| Test ID | Duration | Fluency | Intel. | Grammar | Voice | Sentiment | **Overall / 100** |
|---|---|---|---|---|---|---|---|
| **Test_1** | 46.8 min | 1.55 | 3.25 | 5.00 | 3.36 | 3.35 | **48.3** |
| **Test_2** | 28.1 min | 0.30 | 2.92 | 5.00 | 2.34 | 2.86 | **40.4** |
| **Test_3** | 29.8 min | 0.00 | 3.65 | 5.00 | 3.19 | 3.81 | **44.0** |
| **Test_4** | 63.4 min | 0.00 | 2.83 | 5.00 | 5.00 | 3.83 | **42.5** |
| **Test_5** | 32.8 min | 3.27 | 3.25 | 5.00 | 5.00 | 3.67 | **57.1** |

> **Key Observations:**
> - **Language Control** is consistently high (5.0), suggesting very low grammar error density in the test recordings or potential for further threshold tightening.
> - **Fluency** shows significant variance, correctly penalizing very slow speaking rates (Test_3, Test_4) vs natural speech.
> - **Scoring Compressed:** Lexical Resource and Discourse features are currently yielding baseline scores (0.4 and 1.0 respectively) in several tests, indicating a need to investigate feature extraction pipelines for long-form recordings.

---

## 9. System Reliability & Reproducibility

Automated tests are run to ensure that for identical input audio, the engine returns identical features and scores. This is critical for auditing and client trust.

| Test Category | Status | Count | Avg Coverage |
|---|---|---|---|
| **API Endpoint Stability** | ✅ PASSED | 65 | 100% |
| **Score Reproducibility** | ✅ PASSED | 7 | 100% |
| **Transcoding Integrity** | ✅ PASSED | 12 | 100% |

**Latest Test Session:** `passed in 3193.29s (0:53:13)`

---

## Technical Stack

| Component | Tool / Library |
|---|---|
| Speech-to-Text | OpenAI Whisper (with word-level timestamps and probabilities) |
| Grammar Checking | `language-tool-python` (LanguageTool, Java-based) |
| NLP / Tokenisation | spaCy (`en_core_web_sm`) |
| Word Frequency | `wordfreq` |
| Pitch Analysis | Praat via `parselmouth` |
| Sentiment Analysis | VADER via `nltk` (`SentimentIntensityAnalyzer`) |
| Report Reasoning | **Gemini 2.5 Flash** (via `google-genai` SDK) |
| ML Scoring (when trained) | scikit-learn models (`.pkl`) loaded via `joblib` |

---

*Document updated: 9th March 2026*  
*Project: AI Speaking Assessment Engine — Voice-projects*
