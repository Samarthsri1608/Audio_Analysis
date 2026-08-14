# V4 Audio-Only Proctoring Pipeline — AI Agent Reference & Replication Manual

This document provides a complete, self-contained functional specification, mathematical formulation, and architecture design for the **V4 Audio-Only Proctoring Pipeline**. It is structured specifically to enable an AI agent to fully understand, replicate, debug, and maintain the model and API service.

---

## 1. Pipeline Purpose & Rationale
The V4 proctoring pipeline is a stateless, audit-friendly anomaly detector designed to flag potential academic violations in spoken interviews using **audio features only**.

### Why the ASR/Transcription was Removed
In previous iterations (v3), the pipeline relied on Automatic Speech Recognition (ASR) to compute text-based features (e.g., SBERT semantic coherence, grammatical complexity, and answer-question relevance). This resulted in several fatal issues:
1. **False Positives on Technical Jargon**: Lexical-rich candidate responses containing advanced terminology (e.g., "MongoDB", "Kubernetes") were flagged as anomalies because text models could not distinguish domain vocabulary from generic delivery deviations.
2. **Redundancy**: Answer relevancy is already evaluated by the scoring engine; duplicate calculation in proctoring was redundant.
3. **Execution Bottlenecks**: ASR APIs introduced significant network latency, cost, and a 26% unexplained request failure rate.

### The Audio-Only Solution
V4 operates entirely on the raw audio waveform. A transcript is **never** used as a model input. (Transcripts may be generated post-flagging for human auditors, but have zero influence on the detection path). By shifting to raw acoustic properties (latency, energy, and spectral environment), the pipeline runs fast, costs less, and is immune to vocabulary biases.

---

## 2. Core Architecture & Data Flow

An interview consists of up to $N$ questions (typically $N \le 25$). Each question has a separate recording file fetched from Zeko's server.

```
                  Raw Question Audio Files (MP4/WAV)
                                │
                                ▼
         Audio Fetcher: Download & Convert to WAV (16kHz, mono)
                                │
                                ▼
      Feature Extractor: Parallel Vectorized RMS, VAD & Spectral Roll
                                │
          ┌─────────────────────┴─────────────────────┐
          ▼                                           ▼
      Track A: Self-Baseline                     Track C: Naturalness Rules
   - Compare `response_latency`               - Evaluate `latency_fluency_mismatch`
   - Calculate Robust Median & MAD            - Check `acoustic_environment_shift`
   - Compute Clipped Z-Score ([-8.0, 8.0])   - Check `background_noise_shift`
          │                                           │
          └─────────────────────┬─────────────────────┘
                                ▼
                      OR-Gate Corroboration
                                │
                                ▼
                     Confidence Score Mapper
                                │
                                ▼
                    Evidence Payload Generator
```

---

## 3. Data Schemas (Pydantic Models)

An agent replicating this system must match the following schema structures for database and endpoint compatibility:

### 3.1 AudioFeatures
Extracted features for a single question response. Null/None values represent "uncomputable" features (e.g., pitch on complete silence) and must not be filled with default zeros.
*   `f0_mean` / `f0_std`: `Optional[float] = None` *(Retained as None for V3 schema compatibility)*
*   `speech_rate_proxy`: `Optional[float] = None` *(Retained as None for V3 schema compatibility)*
*   `mfcc_mean` / `mfcc_std`: `Optional[list[float]] = None` *(Retained as None for V3 schema compatibility)*
*   `spectral_flatness_mean`: `Optional[float] = None` *(Retained as None for Track C rolling average compatibility)*
*   `pause_ratio`: `Optional[float]` (Fraction of answer spent in silence)
*   `pause_duration_mean`: `Optional[float]` (Mean silence-segment duration in seconds)
*   `pause_duration_max`: `Optional[float]` (Max silence-segment duration in seconds)
*   `response_latency`: `Optional[float]` (Seconds from audio start to first voiced frame)
*   `energy_mean` / `energy_std`: `Optional[float]` (RMS energy of voiced frames)
*   `room_fingerprint`: `Optional[float]` (Spectral rolloff mean over silence segments in Hz)
*   `noise_floor_centroid`: `Optional[float]` (Spectral centroid of silence segments in Hz)
*   `total_duration_s`: `float` (Total file length)
*   `speech_duration_s`: `float` (Total voiced speech length)

### 3.2 EvaluabilityResult
Gate evaluation representing whether the question audio is usable.
*   `evaluable`: `bool`
*   `not_evaluable_reason`: `Optional[str] = None`
    *   *Allowed values*: `"file_not_found"`, `"corrupt_audio"`, `"insufficient_speech_duration"`, `"no_speech_detected"`, `"low_signal_quality"`, `"audio_quality_issue"`, `"unknown"`

### 3.3 TrackAResult
Output of the self-relative deviation track.
*   `available`: `bool = False` (Set to `False` if cold-started due to insufficient history)
*   `deviation_score`: `Optional[float] = None` (The clipped z-score)
*   `flagged`: `bool = False`
*   `z_scores`: `dict[str, Optional[float]]` (Key-value mapping of feature keys to z-scores)

### 3.4 TrackCResult
Output of the naturalness/mechanism rules track.
*   `flagged`: `bool = False`
*   `signals`: `list[str]` (Triggered signals list: `"latency_fluency_mismatch"`, `"acoustic_environment_shift"`, `"background_noise_shift"`)
*   `signal_details`: `dict[str, float]` (Surfaces raw numeric values that triggered the signals)

### 3.5 QuestionEvidencePayload
Final output structure for a single question.
*   `q_no`: `int`
*   `evaluable`: `bool`
*   `not_evaluable_reason`: `Optional[str] = None`
*   `flagged_for_review`: `bool = False`
*   `confidence`: `str = "low"` (One of: `"high"`, `"medium"`, `"medium_provisional"`, `"low"`)
*   `track_a`: `Optional[TrackAResult] = None`
*   `track_c`: `Optional[TrackCResult] = None`
*   `is_cold_start`: `bool = False`
*   `contributing_features`: `dict[str, Optional[float]]`

---

## 4. Feature Extraction Mathematics & Logic

All feature extraction must be computed at a target sample rate of **$16000\text{ Hz}$** in mono.

### 4.1 Frame Segmentation & Vectorized VAD
A vectorized Voice Activity Detection (VAD) algorithm is computed to find the voiced mask:
1.  **Frame Size**: $20\text{ ms}$ (`VAD_FRAME_S = 0.02`). For $16\text{ kHz}$ audio, this corresponds to $320\text{ samples}$ per frame.
2.  **Vectorized RMS Computation**:
    $$\text{RMS}_t = \sqrt{\frac{1}{M} \sum_{i=t \cdot M}^{(t+1) \cdot M - 1} x[i]^2}$$
    where $M = 320$ samples, and $x[i]$ represents the audio amplitude array.
3.  **Threshold Gate**:
    $$\text{VoicedMask}_t = \text{RMS}_t > P_{60}(\mathbf{RMS})$$
    where $P_{60}$ is the 60th percentile of all RMS values across the entire file.

### 4.2 Latency and Silence Run Lengths
*   **Response Latency**: Seconds from start to first speech:
    $$\text{response\_latency} = t_{\text{first}} \cdot \text{VAD\_FRAME\_S}$$
    where $t_{\text{first}}$ is the smallest index where $\text{VoicedMask}_t = \text{True}$. If no frames are voiced, $\text{response\_latency} = \text{total\_duration\_s}$.
*   **Pause Duration Runs**:
    Consecutive `False` values in $\text{VoicedMask}$ are grouped into runs. The duration of each run is:
    $$\text{Duration}_{\text{run}} = \text{Length}_{\text{run}} \cdot \text{VAD\_FRAME\_S}$$
    Runs with $\text{Duration}_{\text{run}} \ge 0.05\text{ s}$ are classified as meaningful pauses and used to compute `pause_duration_mean` and `pause_duration_max`.

### 4.3 Silence Fingerprinting (Track C Inputs)
Silence segments are extracted by concatenating all segments of $x$ corresponding to frames where $\text{VoicedMask}_t = \text{False}$.
1.  **Room Fingerprint**: The mean of the 85% spectral rolloff over the silence segments:
    $$\text{room\_fingerprint} = \text{mean}(\text{spectral\_rolloff}(x_{\text{silence}}, \text{roll\_percent}=0.85))$$
2.  **Noise Floor Centroid**: The mean spectral centroid over the silence segments:
    $$\text{noise\_floor\_centroid} = \text{mean}(\text{spectral\_centroid}(x_{\text{silence}}))$$

---

## 5. Track A: Self-Baseline Deviation

Track A compares the candidate's current latency against their own interview history.

### 5.1 Baseline Pool
Collect response latencies from all evaluable questions in the interview:
$$\mathbf{L} = \{l_1, l_2, \dots, l_k\}$$
*Cold-Start Guard*: If $k < 3$ (`TRACK_A_MIN_ANSWERS`), Track A terminates and returns `available = False`.

### 5.2 Median & Median Absolute Deviation (MAD)
To ensure outlier resilience, we use the median and median absolute deviation instead of mean and standard deviation:
$$\text{median} = \text{median}(\mathbf{L})$$
$$\text{MAD} = \text{median}(\{ |l_i - \text{median}| : l_i \in \mathbf{L} \})$$

### 5.3 MAD Floor & Clipped Z-Score
To prevent division-by-zero or inflation of z-scores on extremely flat baselines (e.g., when a user responds with identical latencies, resulting in $\text{MAD} = 0$), we apply an effective MAD floor of $0.1\text{ s}$:
$$\text{MAD}_{\text{eff}} = \max(\text{MAD}, 0.1\text{ s})$$

The robust z-score for latency $x$ is computed as:
$$z_{\text{raw}} = 0.6745 \cdot \frac{x - \text{median}}{\text{MAD}_{\text{eff}}}$$
where $0.6745$ is the scaling factor converting MAD to standard deviation equivalents.

To prevent numerical overflow or runaway scores on massive outliers, $z$ is clipped to $[-8.0, 8.0]$:
$$z = \text{clip}(z_{\text{raw}}, -8.0, 8.0)$$

### 5.4 Flagging Condition
$$\text{flag\_A} = z \ge \text{TRACK\_A\_THRESHOLD} \quad (\text{Default } 1.75)$$

---

## 6. Track C: Naturalness Rules & Rolling Interview Baseline

Track C targets cheating mechanisms directly using rolling comparison baselines.

### 6.1 Rolling Interview Baseline State
An `InterviewBaseline` class maintains:
*   $\mathbf{F}_{\text{room}}$: Room fingerprints of previous evaluable questions.
*   $\mathbf{F}_{\text{noise}}$: Noise floor centroids of previous evaluable questions.
*   $\mathbf{F}_{\text{flat}}$: Spectral flatness values of previous evaluable questions.

For any question $m$, the features of question $m$ are evaluated using the baseline state *prior* to question $m$, and the baseline is updated *after* evaluation.

### 6.2 Latency-Fluency Mismatch Rule
This flags candidates who wait for a pasted answer and then read it monotonically:
*   $\text{flatness\_threshold}$:
    $$\theta_{\text{flat}} = \begin{cases}
    \min(0.15, \text{mean}(\mathbf{F}_{\text{flat}}) \cdot 0.7) & \text{if } |\mathbf{F}_{\text{flat}}| \ge 3 \\
    0.15 & \text{otherwise}
    \end{cases}$$
*   **Trigger Condition**:
    All three conditions must be true:
    $$\text{response\_latency} > 4.0\text{ s} \quad \land \quad \text{spectral\_flatness\_mean} < \theta_{\text{flat}} \quad \land \quad \text{pause\_ratio} < 0.05$$

### 6.3 Acoustic Environment Shift Rule
Detects mid-interview device swaps or room transitions:
*   *Requirement*: $|\mathbf{F}_{\text{room}}| \ge 3$
*   *Trigger Condition*:
    $$\frac{|\text{room\_fingerprint} - \text{mean}(\mathbf{F}_{\text{room}})|}{\text{mean}(\mathbf{F}_{\text{room}})} > 0.20 \quad (\text{TRACK\_C\_ROOM\_SHIFT\_THRESHOLD})$$

### 6.4 Background Noise Signature Change Rule
Detects new background audio sources or ambient changes:
*   *Requirement*: $|\mathbf{F}_{\text{noise}}| \ge 3$
*   *Trigger Condition*:
    $$\frac{|\text{noise\_floor\_centroid} - \text{mean}(\mathbf{F}_{\text{noise}})|}{\text{mean}(\mathbf{F}_{\text{noise}})} > 0.25 \quad (\text{TRACK\_C\_NOISE\_SHIFT\_THRESHOLD})$$

---

## 7. OR-Gate & Confidence Truth Table

The outputs of both tracks are merged via a logical OR gate to calculate the final question flag:
$$\text{flagged\_for\_review} = \text{flag\_A} \lor \text{flag\_C}$$

### Confidence Mapping Matrix
The confidence field provides context on flag reliability:

| Track A Flagged (`flag_A`) | Track C Flagged (`flag_C`) | Track A Available (`available`) | Output `confidence` |
|:---:|:---:|:---:|:---|
| `True` | `True` | `True` | `"high"` |
| `False` | `True` | `True` | `"medium"` |
| `True` | `False` | `True` | `"medium"` |
| `False` | `True` | `False` (Cold Start) | `"medium_provisional"` |
| `False` | `False` | Any | `"low"` |

---

## 8. Evaluability Gate Flow Chart

If any check fails, the question is marked `evaluable = False` and processing immediately proceeds to the next question.

```
                  Question WAV Audio
                           │
                           ▼
                 Is file found on disk?
                  ├── No  ──────> False ("file_not_found")
                  └── Yes
                           │
                           ▼
                  Can audio be decoded?
                  ├── No  ──────> False ("corrupt_audio")
                  └── Yes
                           │
                           ▼
                Is total duration >= 1.0s?
                  ├── No  ──────> False ("insufficient_speech_duration")
                  └── Yes
                           │
                           ▼
                Is speech duration >= 2.0s?
                  ├── No  ──────> False ("insufficient_speech_duration")
                  └── Yes
                           │
                           ▼
               Are voiced frames detected?
                  ├── No  ──────> False ("no_speech_detected")
                  └── Yes
                           │
                           ▼
            Is mean RMS energy >= 1e-5 (10^-5)?
                  ├── No  ──────> False ("low_signal_quality")
                  └── Yes ──────> True (evaluable = True)
```

*Note*: If `evaluable = False`, `not_evaluable_reason` is set to the respective reason, and `features` is set to `None`. The corroboration layer defensively defaults any missing reason on a non-evaluable question to `"unknown"`.

---

## 9. Performance Optimization & Calibration

### The 120x Compute Optimization
The V4 feature extraction was profiling-optimized by isolating and removing heavy calls that did not contribute to Track A or C:
*   **Pitch Tracking (`librosa.pyin`)**: Spent ~688 ms in a CPU-bound Python loop per question, making up **97% of the total pipeline duration**. Dropped since F0 is not consumed by the calibrated features.
*   **MFCCs**: Removed (saved ~6 ms per question).
*   **Spectral Flatness**: Removed from active question calculations, though rolling averages are maintained where compatible.
*   **Vectorization**: Swapped Python frame loops for `librosa.feature.rms` (C-backed).
*   **Results**: Total DSP execution time per question dropped from **~707 ms to ~5.9 ms**.

### Target Operating Point Calibration
Calibrated on 332 audited responses (78 High / 253 Low ground truth):
*   **TRACK_A_THRESHOLD**: **1.75** (Env overrideable)
*   **Recall**: **~74.5%**
*   **Precision**: **~25%**
*   This represents a **review-filter operating point**. It maximizes recall to ensure violations are captured for audit, while maintaining a low false alarm rate relative to random screening.

---

## 10. Automated Recovery & State Persistence

For batch running (`run_audio_batch.py`), the runner uses a state-recovery pattern to survive API outages or terminal crashes:
1.  **State Loading**: Reads `output_audio.json` on startup. Populates a map keyed by `(candidate_id, job_id)`.
2.  **Filter Logic**: Identifies entries with `"status": "success"` and excludes them from the execution queue (`to_process`).
3.  **Process Retries**: For every candidate in the execution queue, the script retries processing up to **3 times** with a $2\text{ s}$ delay if a transient API error occurs.
4.  **In-Place Save**: Merges new successes or persistent failures in-place in the map, writing back to `output_audio.json` after every single candidate to guarantee no data loss.
5.  **CSV Sync**: Generates the consolidated `output_audio.csv` only at the end.
