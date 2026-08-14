# Audio-Only Proctoring Pipeline — V4

> **Status: validated on a 300-interview production run.** See §14 for the full confusion matrix and what's still open. Headline result: **76.6% recall** (target was ≥70%), at 23.4% precision and a 79.2% flag rate. Two items remain unresolved: the `evaluable`/`not_evaluable_reason` gate (§9) and question-level localization accuracy (§14.3).

## 1. Why this exists

The v3 academic violation detector relied on a transcription step (ASR) to feed text-derived signals — SBERT coherence, grammar-error scoring, answer-question relevance. In practice this created three problems:

1. **Content-based checks produced false positives on legitimate technical vocabulary** (e.g. flagging use of "MongoDB" as a deviation), because lexical/complexity-based features don't distinguish "unusual word" from "unusual for this candidate."
2. **Answer relevance is already computed by the scoring service** — duplicating it here added no value.
3. **Cross-candidate plagiarism checks don't hold up** as a defensible signal at this stage, so they're out of scope entirely.

What's left once those are removed is a set of signals that don't actually need a transcript: delivery timing patterns and room/device changes mid-interview. All of these live in the audio itself. This pipeline detects academic violations using **audio features only — no ASR, no transcript, in the detection path.**

A transcript may still be generated *after* a response is flagged, purely so a human reviewer has context — but it is never a model input.

---

## 2. Architecture overview

```
Answer audio (one WAV per question, downloaded from Zeko API)
     │
     ▼
Feature extraction
  - VAD (vectorized RMS) → voiced/silence mask
  - Response latency (time to first voiced frame)  ← Track A input
  - Pause ratio, energy                            ← reviewer context
  - Room fingerprint, noise floor centroid         ← Track C input
     │
     ├──────────────────────┐
     ▼                      ▼
Track A:               Track C:
Self-baseline          Naturalness / mechanism
(latency z-score       (room-acoustics shift,
vs own interview       background noise shift,
history)               latency-fluency mismatch)
     │                      │
     └──────────┬────────────┘
                ▼
       OR-gate corroboration
                │
                ▼
         Evidence payload
         (per-question flagged_questions
          list surfaced to proctors)
```

- **Feature extraction** runs once per question-WAV file and produces a shared feature object.
- **Track A** compares `response_latency_s` for each question against the *same candidate's* median latency across their interview (self-relative, robust z-score). This is the sole Track A signal.
- **Track C** looks for delivery patterns that indicate environment changes or reading cues — independent of any baseline, so it works from question 1.
- **OR-gate corroboration** combines both tracks into a final decision: either track alone can trigger a flag; agreement between both raises confidence to `"high"`.
- **Evidence payload** is the structured output — every flagged question's `q_no` is surfaced in `flagged_questions` so proctors know exactly which question to verify.

---

## 3. Feature extraction layer

Computed per question-WAV file using only the audio waveform (no ASR). The extractor is optimised: only features that are actually consumed by Track A or Track C are computed.

### Active features (computed every call)

| Feature | What it captures | How computed | Used by |
|---|---|---|---|
| `response_latency` | Time from audio start to first voiced frame | `argmax(voiced_mask) × frame_hop` | **Track A** (sole signal) |
| `pause_ratio` | Fraction of answer spent in silence | `silence_frames / total_frames` via VAD | Track C + reviewer context |
| `pause_duration_mean/max` | Shape of pausing behaviour | Run-length of silence segments | Reviewer context |
| `energy_mean`, `energy_std` | Vocal effort and SNR check | RMS over voiced frames (vectorized `librosa.feature.rms`) | Evaluability SNR gate + reviewer context |
| `room_fingerprint` | Spectral rolloff over silence segments | `librosa.feature.spectral_rolloff` over noise-floor audio | **Track C** environment-shift detection |
| `noise_floor_centroid` | Spectral centroid of noise floor | `librosa.feature.spectral_centroid` over silence segments | **Track C** background-noise shift |

### Removed features (profiling confirmed not needed)

| Feature | Why removed | Old cost |
|---|---|---|
| `f0_mean`, `f0_std` (pyin) | **97% of per-question wall-clock time.** AUC ~0.53 — near noise. Not consumed by Track A (now latency-only) or Track C. | **688 ms / question** |
| `mfcc_mean`, `mfcc_std` | Track A dropped multi-feature scoring; no Track C consumer. | ~6 ms / question |
| `spectral_flatness_mean` | Track C latency-fluency mismatch signal had near-zero discriminative power; not used. | ~4 ms / question |
| `speech_rate_proxy` | AUC ~0.51 — removed from Track A; not needed elsewhere. | <1 ms / question |

> All removed fields remain as `None` in `AudioFeatures` for schema and API backward compatibility.

**Per-question compute time:** ~700 ms → **~6 ms** after removing `pyin` (120× speedup confirmed via profiling).

### VAD implementation

Energy-percentile VAD using `librosa.feature.rms` (C-backed, vectorized):
- A frame is **voiced** if its RMS exceeds the 60th percentile of all frame RMS values.
- Frame length: 20 ms (`VAD_FRAME_S = 0.02`).
- Audio loaded at `sr=16000` directly (avoids a redundant 22050→16000 resample pass).

---

## 4. Track A — self-baseline deviation (latency only)

**Goal:** flag a question where `response_latency_s` is anomalously high compared to how this *same candidate* starts their other answers in the same interview.

**Signal:** `response_latency_s` only. All other features were dropped — empirical AUC on the labeled set was 0.51–0.58 (near noise) and diluted the one feature that actually works.

**Method:**
1. Collect `response_latency` from all evaluable questions for the candidate.
2. Compute robust baseline: **median** and **MAD** (median absolute deviation) — not mean/std, to resist a small number of outlier answers skewing the baseline.
3. Per-question robust z-score:
   ```
   z = clip(0.6745 × (latency − median) / max(MAD, 0.1), −8, 8)
   ```
   - `0.6745`: consistency factor — makes MAD-based z-scores comparable to σ-based ones under normality.
   - `max(MAD, 0.1)`: **MAD floor at 0.1 s** — prevents collapse to z=0 when all baseline latencies are identical (e.g. pre-recorded responses). 0.1 s is well below typical human answer-to-answer variation (~0.3–1.5 s).
   - `clip(−8, 8)`: **hard cap** — prevents blow-up when a single extreme outlier vs. a flat baseline would otherwise produce z-scores in the hundreds/thousands.
4. **Flag the question** if `z ≥ TRACK_A_THRESHOLD` (default: **1.75**).
   - Predicted operating point (pre-deployment tuning): ~74.5% recall, ~25% precision, ~72% flag rate.
   - **Validated operating point (300-interview production run, see §14): 76.6% recall, 23.4% precision, 79.2% flag rate** — consistent with the predicted point.
   - This is a **review-filter** operating point, not a final-verdict one.
   - Threshold is env-var configurable (`TRACK_A_THRESHOLD`) — no code change needed to re-tune.

**Flagging is per-question, not per-interview.** Each question gets its own `flagged=True/False`. The interview-level `flagged_questions` list is the union of all per-question flags. This surfaces specific question numbers to proctors.

**Cold-start:** if a candidate has fewer than `TRACK_A_MIN_ANSWERS = 3` evaluable questions, Track A returns `available=False` for all questions in that interview. There is no reliable baseline with fewer than 3 data points.

---

## 5. Track C — naturalness / mechanism signals

**Goal:** catch cheating mechanisms directly, independent of any personal baseline. This is the **cold-start-safe** track — it fires from question 1.

| Signal | What it flags | Detection approach |
|---|---|---|
| `latency_fluency_mismatch` | Long silence before answer + unusually smooth, monotone delivery — signature of reading a pasted/prepared answer | `response_latency > 4.0 s` AND `spectral_flatness_mean < 0.15` AND `pause_ratio < 0.05` — natural spontaneous speech has mid-answer hesitation; scripted reading doesn't |
| `acoustic_environment_shift` | Candidate moved, switched device, or left/re-entered mid-interview | Per-answer `room_fingerprint` (spectral rolloff over silence frames) compared to the interview's running mean. Flag if relative shift > 20% |
| `background_noise_shift` | Typing sounds, a second device, ambient shift suggesting external help | Per-answer `noise_floor_centroid` (spectral centroid over silence frames) compared to running mean. Flag if relative shift > 25% |

> **Second-voice detection is explicitly out of scope for V4.** Not enough labeled acoustic-cheating data to train reliably; no rule-based heuristic has been validated. Hold for a later phase.

Track C uses a **rolling `InterviewBaseline`** that updates after each evaluable answer. The current answer is always compared against the history of *prior* answers only (baseline is updated after scoring, not before).

Signals are implemented as **rule-based thresholds** — auditable and tunable against the labeled set, with no trained model to overfit. All thresholds are in `config.py`.

---

## 6. Cold-start handling

Track A needs history; Track C doesn't. Behavior by track:

| Track | Behaviour | Output |
|---|---|---|
| Track A | Does **not** fire until ≥ 3 evaluable questions exist | `available=False`, `deviation_score=None` |
| Track C | Runs from question 1 | Full signal output |
| Combined confidence | If Track C fires and Track A has no baseline | `"medium_provisional"` |

Track A activates from question 4 onwards (for that interview). It does **not** retroactively re-score already-processed questions — the pipeline is stateless and auditable.

---

## 7. OR-gate corroboration logic

```
flag_A = Track_A.available AND Track_A.deviation_score >= TRACK_A_THRESHOLD
flag_C = Track_C.any_signal_fires

final_flag = flag_A OR flag_C
```

Confidence mapping:

| Condition | `confidence` |
|---|---|
| `flag_A AND flag_C` | `"high"` |
| `flag_C` only (baseline exists) | `"medium"` |
| `flag_A` only | `"medium"` |
| `flag_C` only, cold-start | `"medium_provisional"` |
| Neither | `"low"` |

Either track alone is sufficient to flag — this preserves recall (the v3 system missed ~77% of confirmed cases; a single-track-sufficient gate is a deliberate design choice). Corroboration from both tracks sets *confidence*, not whether a flag happens.

---

## 8. Evidence payload — per-question output

Every question produces a `QuestionEvidencePayload`. The full list of payloads is returned in `question_evidence`; flagged question numbers are surfaced in `flagged_questions` at the interview level so proctors can go directly to the right question.

```json
{
  "response_id": "...",
  "status": "success",
  "flagged_questions": [4, 7],          // q_nos where z >= threshold or Track C fired
  "question_evidence": [
    {
      "q_no": 4,
      "evaluable": true,
      "not_evaluable_reason": null,
      "flagged_for_review": true,
      "confidence": "medium",
      "is_cold_start": false,
      "track_a": {
        "available": true,
        "deviation_score": 2.31,         // latency z-score (clipped to [-8, 8])
        "flagged": true,
        "z_scores": { "response_latency": 2.31 }
      },
      "track_c": {
        "flagged": false,
        "signals": [],
        "signal_details": {}
      },
      "contributing_features": {
        "response_latency_s": 8.42,
        "latency_z_score": 2.31,         // z-score alongside raw value for reviewer
        "pause_ratio": 0.12,
        "energy_mean": 0.0043,
        "speech_duration_s": 18.4,
        "total_duration_s": 20.9
      }
    }
  ],
  "total_questions_evaluated": 8,
  "total_questions_flagged": 2,
  "schema_version": "v4-audio-only"
}
```

---

## 9. Evaluability gates

Non-evaluability is an explicit, reasoned output — never a silent absence:

| Condition | `evaluable` | `not_evaluable_reason` |
|---|---|---|
| Audio file missing | `false` | `"file_not_found"` |
| Audio decode failed | `false` | `"corrupt_audio"` |
| Total audio duration < 1 s | `false` | `"insufficient_speech_duration"` |
| No VAD-detected speech | `false` | `"no_speech_detected"` |
| Voiced speech < 2 s | `false` | `"insufficient_speech_duration"` |
| RMS energy < 1e-5 (digital silence) | `false` | `"low_signal_quality"` |
| Audio glitch/quality issue | `false` | `"audio_quality_issue"` |
| All checks pass | `true` | `null` |

`evaluable` must always be explicitly `true` or `false` — never left blank. `not_evaluable_reason` must always be non-null when `evaluable=false`. The corroboration layer enforces this as a defensive fallback (defaults to `"unknown"` if the extractor somehow omitted it).

> **⚠️ Implementation status: NOT YET LANDED.** This gate is spec'd above but not implemented in production. Across all three validation runs to date (initial audio-only run, threshold-tuning run, and the 300-interview run in §14), `not_evaluable_reason` is **100% null** and `evaluable` has **never once been explicitly `false`** — it is only ever `true` or blank. This has been flagged in every review round; treat as a standing, unresolved defect, not a design gap.

---

## 10. Confirmed out of scope for V4

These were explicitly reviewed and deferred:

| Feature | Decision |
|---|---|
| Second-voice / overlap detection | **Deferred.** No validated heuristic; not enough labeled data for a model. Hold for a later phase. |
| Cross-candidate / same-question baseline | **Out of scope.** Self-baseline only; cross-candidate comparison is not part of this pipeline. |
| Population-wide absolute-latency flag | **Hold for Phase 2.** For "consistent reader, no single standout question" cases. Pending more labeled data. |
| ASR / transcription reintroduction | **Out of scope.** Audio-only detection path is a hard constraint. |
| Multi-feature Track A z-score | **Removed.** Empirical AUC for all features except `response_latency` was 0.51–0.58 (near noise). |

---

## 11. Reference implementation (current, optimised)

```python
# feature_extractor.py (key path — simplified)
import librosa
import numpy as np

VAD_FRAME_S = 0.02   # 20 ms frames
SR = 16_000

def extract_features(wav_path):
    # 1. Load at target SR directly (avoids redundant resample pass)
    y, sr = librosa.load(wav_path, sr=SR, mono=True)
    total_s = len(y) / sr

    # 2. Vectorized VAD (C-backed, ~0.1ms vs ~2ms Python loop)
    frame_len = int(VAD_FRAME_S * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=frame_len)[0]
    voiced_mask = rms > np.percentile(rms, 60)
    speech_s = voiced_mask.sum() * VAD_FRAME_S

    # 3. Response latency (sole Track A signal)
    first_voiced = int(np.argmax(voiced_mask)) if voiced_mask.any() else None
    response_latency = first_voiced * VAD_FRAME_S if first_voiced else total_s

    # 4. Room fingerprint (Track C)
    silence_mask = ~voiced_mask
    # ... build noise_y from silence segments ...
    room_fingerprint = float(np.mean(librosa.feature.spectral_rolloff(y=noise_y, sr=sr)))
    noise_floor_centroid = float(np.mean(librosa.feature.spectral_centroid(y=noise_y, sr=sr)))

    return AudioFeatures(response_latency=response_latency,
                         room_fingerprint=room_fingerprint, ...)


# track_a.py (key path — simplified)
import numpy as np

MAD_FLOOR = 0.1   # seconds — prevents collapse to z=0 on flat baselines
Z_CLIP    = 8.0   # prevents blow-up on extreme outliers

def score_all_answers(features_list, min_history=3):
    latencies = [f.response_latency for f in features_list if f is not None]
    if len(latencies) < min_history:
        return [TrackAResult(available=False)] * len(features_list)

    latencies_arr = np.array(latencies, dtype=float)
    median = float(np.median(latencies_arr))
    mad    = float(np.median(np.abs(latencies_arr - median)))
    mad_eff = max(mad, MAD_FLOOR)

    results = []
    for latency in latencies:
        raw_z = 0.6745 * (latency - median) / mad_eff
        z = float(np.clip(raw_z, -Z_CLIP, Z_CLIP))
        results.append(TrackAResult(
            available=True,
            deviation_score=z,
            flagged=(z >= TRACK_A_THRESHOLD),   # default 1.75
            z_scores={"response_latency": z},
        ))
    return results
```

---

## 12. Thresholds (current calibration)

All thresholds live in `v4_proctoring/config.py` and are env-var overridable without code changes.

| Parameter | Value | Note |
|---|---|---|
| `TRACK_A_THRESHOLD` | `1.75` | Validated on 300-interview run: 76.6% recall, 23.4% precision, 79.2% flag rate (§14). Review-filter, not final-verdict. Re-tune when new labeled data arrives — precision/flag-rate trade-off is an open business decision. |
| `TRACK_A_MIN_ANSWERS` | `3` | Minimum evaluable questions before Track A fires |
| `TRACK_A_Z_CLIP` | `8.0` | Hard cap on z-score to prevent blow-up |
| `MAD_FLOOR` | `0.1 s` | Minimum effective MAD to prevent z-score collapse on flat baselines |
| `MIN_SPEECH_SECONDS` | `2.0 s` | Minimum voiced speech for evaluability |
| `MIN_AUDIO_DURATION_SECONDS` | `1.0 s` | Minimum total audio duration for evaluability |
| `TRACK_C_LATENCY_THRESHOLD_S` | `4.0 s` | Latency-fluency mismatch trigger |
| `TRACK_C_FLATNESS_LOW_THRESHOLD` | `0.15` | Spectral flatness below this = very monotone |
| `TRACK_C_PAUSE_RATIO_LOW_THRESHOLD` | `0.05` | Pause ratio below this = near-zero in-answer pauses |
| `TRACK_C_ROOM_SHIFT_THRESHOLD` | `0.20` | 20% relative shift in room fingerprint = environment change |
| `TRACK_C_NOISE_SHIFT_THRESHOLD` | `0.25` | 25% relative shift in noise floor centroid |

---

## 13. Data and privacy notes

Audio features here still constitute **voice/biometric data** under India's DPDP Act — dropping transcription does not remove this obligation, since pitch and vocal-tract characteristics are themselves biometric identifiers. Consent language, storage limits, and access controls that apply to raw audio apply equally to derived feature vectors, not just the audio file itself.

> Note: `f0_mean`, `f0_std`, `mfcc_mean`, `mfcc_std` are no longer computed, which reduces the biometric-data surface area compared to v3. `response_latency` and room acoustic fingerprints are lower-sensitivity features (timing and environment, not vocal identity), though they remain personal data under DPDP.

---

## 14. Validation results

### 14.1 Version history — confusion matrix comparison

| Metric | v3 (transcript-based) | v4 audio-only, pre-fix (z-explosion bug) | v4 audio-only, tuned threshold (predicted) | **v4 audio-only, 300-interview run (validated)** |
|---|---|---|---|---|
| n (clean responses) | 245 | 253 | 230 | **318** |
| Accuracy | 45.7% | 24.1% | ~39.6% | **33.6%** |
| Precision | 15.5% | 23.9% | ~24.7% | **23.4%** |
| **Recall** | 34.0% | 98.4% | ~74.5% | **76.6%** |
| Specificity | 49.0% | 0.5% | — | **19.9%** |
| F1 | 21.3% | 38.5% | — | **35.9%** |
| Flag rate | — | 99.2% | ~72% | **79.2%** |
| Fail rate | 26% | 26% | — | **4.2%** |

**Confusion matrix, 300-interview run** (318 successful responses, 77 High-audited / 241 Low-audited):

|  | Audited High | Audited Low |
|---|---|---|
| **Model flagged** | TP = 59 | FP = 193 |
| **Model didn't flag** | FN = 18 | TN = 48 |

### 14.2 What improved

- **Fail rate: 26% → 4.2%.** The performance/loading fixes (single audio load per response, dropped unused feature extraction) had a large side effect of also reducing pipeline failures, not just wall-clock time.
- **Z-score explosion bug confirmed fixed.** `track_a_score` now sits in the intended bounded range (capped at 8) instead of the hundreds/thousands seen in the pre-fix run.
- **Recall target met.** 76.6% ≥ the 70% target, up from 34% on the original transcript-based pipeline.

### 14.3 What's still open

- **Question-level localization is weak.** Across 44 High-audited responses where the auditor named specific question numbers in `Remarks`, only **10 (22.7%)** had *any* overlap between the audited question(s) and the pipeline's `flagged_questions`. The response-level triage decision is usually right; the specific question numbers handed to proctors are usually wrong. This matters directly for the review workflow, since proctors are given question numbers to verify, not just a flagged/not-flagged verdict.
- **Precision/specificity are low at this operating point.** 79.2% flag rate and 19.9% specificity mean roughly 4 in 5 interviews get sent to review. Needs a business decision: confirm with the proctor team whether this volume is workable, or dial `TRACK_A_THRESHOLD` up to trade recall back for precision.
- **`evaluable` / `not_evaluable_reason` gate is unimplemented** (see §9 callout) — outstanding across all three validation runs.
- **Root-cause diagnostics on per-candidate baseline stability are still unverified.** The CSV export used for the 300-interview analysis only contains flagged-question rows (plus a placeholder row for zero-flag responses), not a full per-question dump — so questions like "does MAD collapse for candidates with very few evaluable answers" can't be checked from that export. A full per-question dump (all questions, flagged or not) across a batch is needed to validate this properly.

## 15. Rollout plan

1. ~~**Shadow mode**: run this pipeline against the existing 332 audited/labeled responses (78 High / 253 Low ground truth) without affecting production flags.~~ **Done** — see §14.
2. ~~**Compare** the resulting confusion matrix against the v3 baseline.~~ **Done** — see §14.1.
3. **Tune thresholds** further based on the open items in §14.3 — in particular, decide on the precision/recall trade-off with the proctor team before wider rollout.
4. **Investigate question-level localization** before treating the flagged `q_no` list as reliable for reviewers — consider surfacing all moderately-elevated questions rather than only the single max-z question per response.
5. **Cut over gradually** — run both old and new pipelines in parallel for one full customer cycle, with new pipeline flags visible to reviewers but not yet authoritative, before retiring the transcript-based path.
6. **Re-validate thresholds periodically** as labeled volume grows (more labeled data has been committed and is pending).

---

## 16. Known limitations

- Track A cannot help until a candidate has answered ≥ 3 questions (cold-start gap, mitigated but not eliminated by Track C).
- `response_latency_s` is the sole Track A signal — it is a strong signal for the labeled set but may not generalize perfectly to all interview formats (e.g. very long think-time questions where latency is intentionally high). Monitor the flag rate on clean candidates after cutover.
- The 0.1 s MAD floor is a pragmatic choice; if a candidate genuinely has very consistent latencies (tight cluster near zero), a 10 s outlier will still get a high z-score. This is the intended behavior for the review-filter operating point, but proctors should understand that high flag rates on a given interview may simply reflect a consistent fast-responder with one slow question, not guaranteed cheating.
- Environmental/hardware variance across candidates is not fully controlled for by self-baseline comparison within one interview — a candidate who changes devices mid-interview for an innocent reason (e.g. dropped call, reconnect) could trigger `acoustic_environment_shift` and needs a clear reviewer-facing explanation.
- **Question-level localization is weak (validated finding, §14.3):** only 22.7% of audited-flagged questions overlap with the pipeline's flagged questions. Treat `flagged_questions` as reliable for "should this interview go to review," not yet reliable for "these are definitely the exact suspicious moments."
- **High flag rate needs a business decision, not just a technical one:** 79.2% of interviews are currently flagged. This pipeline has been validated end-to-end on real data (§14) — the numbers above are no longer a design target, they are measured results — but whether this operating point is workable for the proctor team's review capacity is still an open question.
- `evaluable` / `not_evaluable_reason` remains unimplemented (§9) — non-evaluable questions still fail silently rather than with a diagnosable reason.