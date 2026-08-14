# V3 Speech Feature & Delivery Analysis Pipeline — Replication & Reference Manual

This document provides a complete, self-contained functional specification, mathematical formulation, and architecture design for the **V3 Audio-Text Hybrid Speech Feature & Delivery Analysis Pipeline**. It is structured specifically to enable an AI agent to fully understand, replicate, debug, and maintain the feature extraction, speech deviation scoring, and naturalness evaluation model without needing to reference any other files.

---

## 1. What It Is
The V3 Speech Analysis Pipeline is a post-interview batch processing system designed to extract, analyze, and score candidate vocal delivery characteristics, prosody, fluency, and speech patterns. 

Rather than relying on raw absolute features which vary naturally by candidate, this pipeline evaluates:
1.  **Track A (Self-Relative Deviation)**: How much a specific question response deviates from the candidate's own baseline speech profile established across the rest of the interview.
2.  **Track C (Speech Naturalness Rules)**: Inherent delivery dynamics such as pacing uniformity, pause interval regularities, pitch expression ranges, and interview-wide stability/flatness metrics.

---

## 2. Core Architecture & Data Flow

Because the pipeline evaluates interview-wide characteristics (such as cross-question latency variance and overall speech flatness), it is structured as a post-interview analysis job running after the interview is complete.

```
                         Candidate Audio Waveform
                                    │
                                    ├──> [1] Audio Converter (MP4/WAV to 16kHz mono WAV)
                                    └──> [2] ASR Transcriber (Get text & word-level timestamps)
                                                 │
                                                 ▼
                                     Text transcript & word timestamps
                                                 │
                                                 ▼
      ┌──────────────────────────────────────────┴──────────────────────────────────────────┐
      ▼                                                                                     ▼
  [3] Text Feature Extractor                                                    [4] Vocal Feature Extractor
  - speech_rate_wpm                                                             - pitch_variation_cv (F0 CV)
  - filler_word_ratio                                                           - vocal_confidence (power stability)
  - speech_fluency (fraction without gaps)                                      - stress_markers (fry & tremor inverse)
  - discourse_organization                                                      - pause_features (pause freq & dur)
  - sbert_coherence (sentence similarity)                                       - voiced_fraction (speaking ratio)
      │                                                                                     │
      └──────────────────────────────────────────┬──────────────────────────────────────────┘
                                                 ▼
                                      [5] Eligibility Gate
                                                 │
                                                 ▼
                                      [6] Interview Aggregator
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
                     Track A                                           Track C
                 (Self-Relative)                                   (Naturalness)
             - Leave-one-out baseline                          - Bucket pace CV
             - Percentile rank (discourse)                     - Pause regularity CV
             - Robust Z-Scores                                 - Sigmoid latency Z-Score
                                                               - Interview-wide flatness
                                                               - Interview-wide latency variance
                        │                                                 │
                        └────────────────────────┬────────────────────────┘
                                                 ▼
                                      [7] Composite Scorer
                                                 │
                                                 ▼
                                    Consolidated Evidence Payload
```

---

## 3. How It Works: Step-by-Step

### Step 1: Audio Processing & Transcription
1.  **Audio Conversion**: Input media is converted to **16kHz mono WAV** format (pcm_s16le) via ffmpeg.
2.  **ASR Transcription**: The WAV is processed to obtain the word-level transcript alongside precise start/end timestamps for every word.

### Step 2: Speech Eligibility Filtering
Before feature extraction, each response is evaluated by an eligibility gate. A response is marked `evaluable = True` only if it passes all the following:
*   `skipped` or `off_camera` is `False`.
*   ASR transcript text is non-empty.
*   Total duration of the audio response $\ge 3.0\text{ seconds}$.
*   Word count $\ge 10$ words (either from word timestamps or space-separated transcript text).
*   Mean ASR decoding confidence $\ge 0.40$.

If a response fails any check, it is marked `evaluable = False` with a specific reason (`"too_short"`, `"low_word_count"`, `"low_asr_confidence"`, etc.), and features/scores are set to `None`.

### Step 3: Feature-Specific Availability Checks
To prevent short or sparse responses from generating corrupted "zero" values, features are set to `None` if the response fails these minimum criteria:

*   **`lexical_mattr`**: Requires $\ge 50\text{ words}$ (below this, moving-average TTR degrades to plain TTR).
*   **`sbert_coherence`**: Requires $\ge 20\text{ words}$ (minimum needed to yield at least 2 sentences for comparison).
*   **`discourse_organization`**: Requires $\ge 15\text{ words}$ (below this, connector counts are artificially sparse).
*   **`speech_rate_wpm`**: Requires $\ge 5\text{ words}$.
*   **`filler_word_ratio`**: Requires $\ge 10\text{ words}$.
*   **`intra_answer_pace_variance`**: Requires response duration $\ge 30.0\text{ seconds}$ (needs at least two 15s buckets).
*   **`pause_regularity`**: Requires response duration $\ge 10.0\text{ seconds}$ (needs $\ge 2$ pauses $> 300\text{ ms}$).

---

## 4. Why It Works: Feature Formulations & Logic

### 4.1 Text Features
*   **`speech_rate_wpm`**:
    $$\text{speech\_rate\_wpm} = \frac{\text{word\_count}}{\text{duration\_s}} \times 60$$
*   **`filler_word_ratio`**:
    $$\text{filler\_word\_ratio} = \frac{\text{filler\_words\_count}}{\text{word\_count}}$$
    *Filler vocabulary*: `"like"`, `"um"`, `"uh"`, `"ah"`, `"eh"`, and sentence-initial `"so"`.
*   **`lexical_mattr` (Moving Average Type-Token Ratio)**:
    Measures vocabulary diversity over a moving window of $W = 50$ words:
    $$\text{MATTR} = \frac{1}{N - W + 1} \sum_{i=1}^{N - W + 1} \frac{\text{UniqueWords}(w_i \dots w_{i+W-1})}{W}$$
*   **`discourse_organization`**:
    Evaluates usage of transition and logic words:
    $$\text{discourse\_score} = \text{connectors\_count} + 1.5 \times \text{tier1\_connectors\_count}$$
    *Tier 1 Connectors*: `"specifically"`, `"consequently"`, `"subsequently"`, `"furthermore"`, `"moreover"`.
    *General Connectors*: `"however"`, `"therefore"`, `"because"`, `"since"`, `"although"`, `"besides"`, `"thus"`, `"hence"`.
*   **`sbert_coherence`**:
    Calculates the mean cosine similarity between SBERT embeddings of consecutive sentences:
    $$\text{Coherence} = \frac{1}{S-1} \sum_{j=1}^{S-1} \frac{\mathbf{e}_j \cdot \mathbf{e}_{j+1}}{\|\mathbf{e}_j\| \|\mathbf{e}_{j+1}\|}$$
    If sentence extraction fails or cosine similarity returns exactly $0.65$ (neutral sentinel), it is treated as `None` (unavailable).

### 4.2 Acoustic Prosody Features
*   **`pitch_variation_cv`** (Dialect-Neutral Pitch Variation):
    Uses the Coefficient of Variation (CV) rather than absolute Hz standard deviation. This provides speaker-relative, dialect-neutral scaling (Indian English dialect patterns naturally feature narrower pitch ranges, which raw Hz std dev would falsely penalize as monotone):
    $$\text{F0} = \text{YIN\_Algorithm}(x, \text{fmin}=75\text{Hz}, \text{fmax}=400\text{Hz})$$
    A 5-frame rolling median filter is applied to $\text{F0}$ to remove octave tracking jumps. Let $\text{F0}_{\text{smoothed}}$ represent the filtered voiced frames:
    $$\text{CV}_{\text{pitch}} = \text{clip}\left(\frac{\text{std}(\text{F0}_{\text{smoothed}})}{\text{mean}(\text{F0}_{\text{smoothed}})}, 0.0, 0.45\right)$$
*   **`vocal_confidence`** (Stability of power in voiced segments):
    Energy (STFT power spectral density) is segmented. For voiced segments $> 10$ frames:
    $$\text{stability} = 1.0 - \frac{\text{std}(\text{power}_{\text{voiced}})}{\text{mean}(\text{power}_{\text{voiced}})}$$
    $$\text{vocal\_confidence} = \text{mean}(\mathbf{stability})$$
*   **`speech_fluency`**:
    The fraction of word transitions without a long gap:
    $$\text{speech\_fluency} = 1.0 - \frac{\text{gaps} > 200\text{ ms}}{\text{total\_gaps}}$$
*   **`stress_markers`**:
    Combines vocal fry (disproportionate low-frequency energy) and pitch tremor (cv of voiced pitch):
    $$\text{fry} = \text{clip}\left(\frac{\text{mean}(\text{Spectrogram}_{\text{low\_freq}})}{\text{mean}(\text{Spectrogram}_{\text{mid\_freq}})}, 0.0, 1.0\right)$$
    $$\text{tremor} = \text{clip}\left(\frac{\text{std}(\text{F0})}{\text{mean}(\text{F0})}, 0.0, 0.45\right)$$
    $$\text{stress\_markers} = 1.0 - (0.6 \times \text{fry} + 0.4 \times \text{tremor})$$
*   **`pause_duration` & `pause_frequency`**:
    Word gap durations $> 300\text{ ms}$ are collected. Gaps below $300\text{ ms}$ are ignored as normal inter-word transitions.
*   **`voiced_fraction`**:
    The ratio of active voiced duration to total response duration:
    $$\text{voiced\_fraction} = \frac{\sum (\text{word\_end} - \text{word\_start})}{\text{duration\_seconds}}$$

---

## 5. Track A: Self-Relative Deviation (Leave-One-Out)

Track A compares the candidate's current response features against a baseline built from **all other evaluable questions** in the candidate's interview (excluding the current question).
*Requirement*: Minimum of 4 evaluable answers in the interview. If $< 4$, Track A is skipped.

### 5.1 Robust Z-Score Formulation
$$\text{median} = \text{median}(\mathbf{X}_{\text{peers}})$$
$$\text{MAD} = \text{median}(\{ |x_j - \text{median}| : x_j \in \mathbf{X}_{\text{peers}} \})$$
$$\text{MAD}_{\text{eff}} = \max(\text{MAD}, \text{MAD\_Floor}_{\text{feature}})$$

The robust z-score is directional:
$$z_{\text{raw}} = 0.6745 \cdot \frac{x_i - \text{median}}{\text{MAD}_{\text{eff}}}$$
$$z = \text{clip}(z_{\text{raw}}, -10.0, 10.0)$$

Depending on the feature, only the target direction contributes to deviation (the opposite direction is clipped to $0.0$):

| Feature Name | Target Direction | MAD Floor |
|:---|:---:|:---:|
| `speech_rate_wpm` | Increase (Speeding up) | $1.0$ |
| `filler_word_ratio` | Decrease (Atypically fluent) | $0.002$ |
| `lexical_mattr` | Increase (Atypically high density) | $0.005$ |
| `sbert_coherence` | Increase (Unusually uniform similarity) | $0.005$ |
| *Default Floor* | — | $0.001$ |

### 5.2 Discourse Organization Percentile Rank
Because discourse connectors are sparse and cause MAD to collapse to 0, it uses a percentile rank instead of a z-score:
$$\text{rank} = \frac{\text{Count}(x_j < x_i) + 0.5 \cdot \text{Count}(x_j == x_i)}{N}$$
$$z_{\text{discourse}} = \text{clip}(10.0 \cdot (\text{rank} - 0.5) \cdot 2.0, 0.0, 10.0)$$

---

## 6. Track C: Naturalness Rules

These features assess candidate delivery patterns and natural speech variance:

1.  **`intra_answer_pace_variance`**:
    Word rate (WPM) is computed for each $15\text{-second}$ window. The coefficient of variation (CV) of these rates is calculated. A low CV indicates highly uniform, mechanical pacing. A robust z-score is computed, flagging a **decrease** in CV.
2.  **`pause_regularity`**:
    The CV of pause durations $> 300\text{ ms}$. A low CV indicates mechanically structured pause intervals. A robust z-score is computed, flagging a **decrease** in CV.
3.  **`pitch_variance_ratio`**:
    $$\text{pitch\_variance\_ratio} = \frac{\text{CV}_{\text{pitch}}}{\max(\mathbf{CV}_{\text{pitch\_interview}}) - \min(\mathbf{CV}_{\text{pitch\_interview}})}$$
    A robust z-score is computed, flagging a **decrease** in pitch variation (flattened intonation).
4.  **`response_latency_sec`**:
    Start timestamp of the first word. The latency z-score is transformed via a sigmoid function:
    $$z_{\text{sig}} = \frac{1}{1 + \exp(-1.2 \cdot |z_{\text{latency}}|)}$$
5.  **`latency_variance_across_questions`** (Provisional Interview-Wide Flag):
    Measures standard deviation of latency across all questions in the interview. If std dev $< 0.8\text{ s}$ (with $\ge 3$ questions), it flags a lack of normal thinking time variation. Adds **+2.5** to the composite score.
6.  **`cross_question_naturalness_flatness`** (Provisional Interview-Wide Flag):
    Measures the standard deviation of `speech_rate_wpm`, `lexical_mattr`, and `sbert_coherence` across the entire interview, normalized by their mean:
    $$\text{flatness}(X) = \frac{\text{std}(X)}{\text{mean}(X) + 10^{-6}}$$
    If the average flatness index is $< 0.05$ (with $\ge 3$ questions), it indicates the candidate's speech is abnormally uniform across easy/hard questions. Adds **+3.0** to the composite score.

---

## 7. Composite Scoring Schema

### 7.1 Composite Score Formula
$$\text{Composite\_Score} = \sum (\text{Weight}_A \cdot z_A) + \sum (\text{Weight}_C \cdot z_C) + \text{Latency\_sig\_contrib} + \text{Provisional\_bonuses}$$

Where:
*   $\text{Latency\_sig\_contrib} = 2.0 \cdot (z_{\text{sig}} - 0.5) \cdot 2.0$
*   Track A Weights: WPM ($0.6$), Fillers ($0.6$), MATTR ($0.6$), Discourse ($0.5$), Coherence ($0.5$).
*   Track C Weights: Pace Variance ($1.0$), Pause Regularity ($1.5$), Pitch ($1.5$).
*   Provisional bonuses: Latency variance bonus ($2.5$), Flatness bonus ($3.0$).

### 7.2 Core Config Constants
*   `COMPOSITE_THRESHOLD = 6.0`
*   `HARD_SIGNAL_THRESHOLD = 2.0`
*   `HYSTERESIS_BAND = 0.25`
*   `LATENCY_SIGMOID_SCALE = 1.2`
*   `TRACK_A_MIN_EVALUABLE_ANSWERS = 4`
