# Audio-Only Academic Violation Detection Pipeline

## 1. Why this exists

The current academic violation detector relies on a transcription step (ASR) to feed text-derived signals — SBERT coherence, grammar-error scoring, answer-question relevance. In practice this created three problems:

1. **Content-based checks produced false positives on legitimate technical vocabulary** (e.g. flagging use of "MongoDB" as a deviation), because lexical/complexity-based features don't distinguish "unusual word" from "unusual for this candidate."
2. **Answer relevance is already computed by the scoring service** — duplicating it here added no value.
3. **Cross-candidate plagiarism checks don't hold up** as a defensible signal at this stage, so they're out of scope entirely.

What's left once those are removed is a set of signals that don't actually need a transcript: reading-vs-speaking delivery patterns, second-voice presence, room/device changes mid-interview, and per-candidate delivery consistency. All of these live in the audio itself. This document specs a pipeline that detects academic violations using **audio features only — no ASR, no transcript, in the detection path.**

A transcript may still be generated *after* a response is flagged, purely so a human reviewer has context — but it is never a model input.

---

## 2. Architecture overview

```
Answer audio
     │
     ▼
Feature extraction (pitch, pace, pauses, energy, timbre)
     │
     ├──────────────┬──────────────┐
     ▼              ▼
Track A:        Track C:
self-baseline   naturalness
(z-score vs     (second voice,
own history)    reading cues)
     │              │
     └──────┬───────┘
            ▼
   OR-gate corroboration
            │
            ▼
     Evidence payload
```

- **Feature extraction** runs once per answer and produces a shared feature vector.
- **Track A** compares an answer against the *same candidate's* other answers in the interview (self-relative, robust z-scores).
- **Track C** looks for delivery patterns that indicate reading, prompting, or environment changes — independent of any baseline, so it works from question 1.
- **OR-gate corroboration** combines both tracks into a final decision: either track alone can trigger a flag; agreement between both raises confidence.
- **Evidence payload** is the structured output attached to the interview report.

---

## 3. Feature extraction layer

Computed per answer, using only the audio waveform (no ASR):

| Feature | What it captures | How to compute |
|---|---|---|
| `f0_mean`, `f0_std` | Pitch level and variation | Pitch tracking (e.g. `librosa.pyin` or CREPE) over voiced frames |
| `speech_rate_proxy` | Speaking pace without needing words | Syllable-rate proxy via energy-envelope peak detection (peaks/sec in voiced segments), or VAD-segment count normalized by duration |
| `pause_ratio` | Fraction of answer spent in silence | (Total silence duration) / (total answer duration), via VAD |
| `pause_duration_dist` | Shape of pausing behavior | Mean, std, and max of individual silence-segment lengths |
| `response_latency` | Time from question end to first speech | Timestamp of first VAD-detected speech minus question-end timestamp |
| `energy_mean`, `energy_std` | Vocal effort and consistency | RMS energy over voiced frames |
| `mfcc_summary` | Timbre/vocal-tract consistency | Mean + std of MFCC coefficients (1–13) over the answer |
| `spectral_flatness` | Monotone vs. expressive delivery | Mean spectral flatness over voiced frames — flat/scripted speech tends toward lower variance here |

All of these are computed from the raw waveform; none require knowing what was said.

---

## 4. Track A — self-baseline deviation

**Goal:** flag an answer that behaves very differently from how this candidate delivers *their other* answers in the same interview.

**Method:**
1. For each candidate, collect the feature vectors from all their evaluable answers in the interview.
2. Compute a robust baseline per feature: median and MAD (median absolute deviation), not mean/std — robust to a few outlier answers skewing the baseline.
3. For each answer, compute a robust z-score per feature:
   `z = 0.6745 * (x - median) / MAD`
4. Aggregate into a per-answer deviation score (e.g. max or top-2-mean of the absolute z-scores across features).
5. Flag if the aggregate score exceeds a threshold (start conservative, tune against labeled data).

**Requires:** a minimum number of the candidate's *own* answers to build a stable baseline (see cold-start handling below). This is the track that structurally cannot help on interview #1.

---

## 5. Track C — naturalness / mechanism signals

**Goal:** catch cheating mechanisms directly, independent of any personal baseline. This is the cold-start-safe track.

| Signal | What it flags | Rough detection approach |
|---|---|---|
| Second voice / overlap | Someone else speaking in the background (prompting, whispering) | Speaker-change / overlap detection on the raw waveform (e.g. diarization confidence, or simple pitch-range/formant discontinuity within a single "turn") |
| Room acoustics shift | Candidate moved, switched device, or left/re-entered frame mid-interview | Estimate reverberation (RT60 proxy) per answer; flag a significant shift versus the interview's running average |
| Latency-fluency mismatch | Long pause followed by unnaturally smooth, monotone delivery — signature of reading a prepared/pasted answer aloud | Combine `response_latency` (high) with `spectral_flatness` (low variance) and `pause_ratio` *within* the answer (near zero) — natural spontaneous speech has some mid-answer hesitation; scripted reading typically doesn't |
| Background noise signature change | Typing sounds, a second device, ambient shift suggesting external help | Spectral profile of the noise floor between voiced segments, compared across the interview |

Start these as **rule-based thresholds**, not a trained classifier — there isn't enough labeled acoustic-cheating data yet to train reliably, and thresholds are auditable and tunable against your existing 78 known-High / 253 known-Low labeled responses. Move to a learned model only once enough flagged/confirmed cases accumulate.

---

## 6. Cold-start handling

Track A needs history; Track C doesn't. So:

- If a candidate has **fewer than N answered questions** (suggest N=3 as a starting point) in the interview so far, Track A does not fire — there's no reliable baseline yet.
- Track C runs regardless, and its flags for cold-start candidates are marked **provisional** rather than final — visible to reviewers, but treated with slightly higher scrutiny before action, since there's no corroborating self-baseline signal yet to confirm it.
- Once enough answers accumulate mid-interview, Track A activates retroactively for that candidate's remaining answers (but doesn't retroactively re-score already-passed answers, to keep the pipeline stateless and auditable).

---

## 7. OR-gate corroboration logic

```
flag_A = Track_A.deviation_score > threshold_A   (only if baseline exists)
flag_C = Track_C.any_signal_fires

final_flag = flag_A OR flag_C
confidence = "high" if (flag_A AND flag_C) else "medium" if flag_C else "low-provisional" if flag_A only and cold-start else "standard"
```

Either track alone is sufficient to flag — this preserves recall (the current system's biggest weakness was missing ~77% of confirmed cases; a single-track-sufficient gate is a deliberate choice to not repeat that). Corroboration from both tracks is used to set *confidence*, not to gate whether a flag happens at all.

---

## 8. Handling non-evaluable audio (fixing the known bug)

The current system silently leaves `evaluable`, `flagged_for_review`, and `suspicion_score` all null for unevaluable answers, with `not_evaluable_reason` never populated at all. This pipeline fixes that by making non-evaluability an explicit, reasoned output rather than an absence of output:

| Condition | `evaluable` | `not_evaluable_reason` |
|---|---|---|
| Audio file missing/corrupted | `false` | `"file_not_found"` / `"corrupt_audio"` |
| Answer shorter than minimum duration (e.g. <2s of detected speech) | `false` | `"insufficient_speech_duration"` |
| Entirely silence / no VAD-detected speech | `false` | `"no_speech_detected"` |
| Audio present but SNR too low to extract reliable pitch/energy features | `false` | `"low_signal_quality"` |
| Feature extraction succeeds | `true` | `null` |

`evaluable` must always be explicitly `true` or `false` — never left blank. This alone should explain and reduce a chunk of the current pipeline's 26% unexplained failure rate.

---

## 9. Other edge cases to design for

- **Legitimate background noise (pets, siblings, traffic) that isn't a second voice**: overlap detection should specifically target *speech-shaped* overlap (formant structure consistent with a second speaker), not any non-silence background noise — otherwise this becomes a new false-positive source, possibly correlated with home environment/socioeconomic status, which is its own fairness risk to watch.
- **Candidates with genuinely flat/monotone speaking styles** (this can correlate with certain speech patterns, non-native fluency, or disability): Track C's "spectral flatness" signal should always be evaluated as a *deviation from the interview's own running baseline*, not an absolute threshold — a naturally monotone speaker isn't a deviation from themselves.
- **Short, correct answers that are simply confident and quick**: don't let low `response_latency` alone be penalized — the risk signal is the *combination* of fast latency AND unnaturally flat delivery, not speed alone.
- **Connection/audio glitches mid-answer** (dropped frames, robotic artifacts): these should route to `not_evaluable_reason: "audio_quality_issue"` rather than being scored and potentially misread as a "naturalness" anomaly.
- **Multi-language or code-switching candidates**: pitch/prosody baselines should still hold since these are language-agnostic acoustic features — but validate this assumption specifically before rollout, since it's an assumption, not a guarantee.

---

## 10. Reference implementation

```python
import numpy as np
import librosa

# ---------- 1. Feature extraction ----------

def extract_features(audio_path, question_end_ts, sr=16000):
    """Extract audio-only features for a single answer. No ASR involved."""
    y, sr = librosa.load(audio_path, sr=sr)
    duration = len(y) / sr

    # Voice activity detection (simple energy-based; swap for webrtcvad in prod)
    frame_len = int(0.02 * sr)
    hop = frame_len
    energy = np.array([
        np.sqrt(np.mean(y[i:i+frame_len]**2))
        for i in range(0, len(y) - frame_len, hop)
    ])
    voiced_thresh = np.percentile(energy, 60)
    voiced_mask = energy > voiced_thresh

    if voiced_mask.sum() == 0:
        return None  # no speech detected -> not_evaluable

    # Pitch (F0) via pyin, restricted to voiced frames
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
    )
    f0_voiced = f0[~np.isnan(f0)]

    # Response latency: time to first voiced frame after question end
    first_speech_frame = np.argmax(voiced_mask) if voiced_mask.any() else None
    response_latency = (first_speech_frame * hop / sr) if first_speech_frame is not None else duration

    # Pause ratio + pause durations
    silence_mask = ~voiced_mask
    pause_ratio = silence_mask.sum() / len(voiced_mask)
    pause_runs = _run_lengths(silence_mask) * (hop / sr)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Spectral flatness (monotone/scripted-delivery proxy)
    flatness = librosa.feature.spectral_flatness(y=y)

    return {
        "f0_mean": float(np.mean(f0_voiced)) if len(f0_voiced) else None,
        "f0_std": float(np.std(f0_voiced)) if len(f0_voiced) else None,
        "speech_rate_proxy": float(voiced_mask.sum() / duration),
        "pause_ratio": float(pause_ratio),
        "pause_duration_mean": float(np.mean(pause_runs)) if len(pause_runs) else 0.0,
        "pause_duration_max": float(np.max(pause_runs)) if len(pause_runs) else 0.0,
        "response_latency": float(response_latency),
        "energy_mean": float(np.mean(energy)),
        "energy_std": float(np.std(energy)),
        "mfcc_mean": mfcc.mean(axis=1).tolist(),
        "mfcc_std": mfcc.std(axis=1).tolist(),
        "spectral_flatness_mean": float(np.mean(flatness)),
    }


def _run_lengths(bool_mask):
    """Return lengths (in frames) of consecutive True runs."""
    runs = []
    count = 0
    for v in bool_mask:
        if v:
            count += 1
        elif count > 0:
            runs.append(count)
            count = 0
    if count > 0:
        runs.append(count)
    return np.array(runs)


# ---------- 2. Evaluability gate ----------

MIN_SPEECH_SECONDS = 2.0

def check_evaluable(audio_path, features):
    if features is None:
        return False, "no_speech_detected"
    speech_seconds = features["speech_rate_proxy"] * 1.0  # proxy already normalized; adjust as needed
    if speech_seconds < MIN_SPEECH_SECONDS:
        return False, "insufficient_speech_duration"
    return True, None


# ---------- 3. Track A: self-baseline z-score ----------

def track_a_scores(candidate_answers, min_history=3):
    """
    candidate_answers: list of feature dicts for one candidate's evaluable answers
    Returns per-answer deviation scores; None if history is insufficient (cold-start).
    """
    if len(candidate_answers) < min_history:
        return [None] * len(candidate_answers)

    keys = ["f0_mean", "speech_rate_proxy", "pause_ratio", "response_latency",
            "energy_mean", "spectral_flatness_mean"]

    matrix = np.array([[a[k] for k in keys] for a in candidate_answers])
    median = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - median), axis=0)
    mad = np.where(mad == 0, 1e-6, mad)  # avoid divide-by-zero

    robust_z = 0.6745 * (matrix - median) / mad
    per_answer_score = np.max(np.abs(robust_z), axis=1)  # max deviation across features
    return per_answer_score.tolist()


TRACK_A_THRESHOLD = 3.5  # tune against labeled data


# ---------- 4. Track C: naturalness / mechanism rules ----------

def track_c_flags(features, interview_running_baseline):
    """Rule-based, cold-start-safe. interview_running_baseline holds running
    averages for reverb/noise-floor computed across the interview so far."""
    flags = []

    # Latency-fluency mismatch: long pause before answer, then very smooth delivery
    if features["response_latency"] > 4.0 and features["spectral_flatness_mean"] < 0.15 \
       and features["pause_ratio"] < 0.05:
        flags.append("latency_fluency_mismatch")

    # Room/device acoustics shift (placeholder: reverb estimate would be computed
    # upstream and compared to interview_running_baseline["reverb_rt60"])
    if interview_running_baseline.get("reverb_rt60") is not None:
        current_rt60 = features.get("reverb_rt60")
        if current_rt60 and abs(current_rt60 - interview_running_baseline["reverb_rt60"]) > 0.15:
            flags.append("acoustic_environment_shift")

    # Second voice / overlap (placeholder: would come from a diarization/overlap
    # detector run alongside feature extraction)
    if features.get("overlap_detected"):
        flags.append("second_voice_detected")

    return flags


# ---------- 5. OR-gate corroboration + evidence payload ----------

def build_evidence_payload(candidate_id, response_id, q_no, features,
                            track_a_score, track_c_flags_list, is_cold_start):
    flag_a = track_a_score is not None and track_a_score > TRACK_A_THRESHOLD
    flag_c = len(track_c_flags_list) > 0
    final_flag = flag_a or flag_c

    if flag_a and flag_c:
        confidence = "high"
    elif flag_c and is_cold_start:
        confidence = "medium_provisional"
    elif flag_c:
        confidence = "medium"
    elif flag_a:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "candidate_id": candidate_id,
        "response_id": response_id,
        "q_no": q_no,
        "evaluable": True,
        "not_evaluable_reason": None,
        "flagged_for_review": final_flag,
        "confidence": confidence if final_flag else "low",
        "track_a_score": track_a_score,
        "track_c_signals": track_c_flags_list,
        "is_cold_start": is_cold_start,
        "contributing_features": {
            k: v for k, v in features.items()
            if k in ("response_latency", "spectral_flatness_mean", "pause_ratio")
        },
    }
```

---

## 11. Data and privacy notes

Audio features here still constitute **voice/biometric data** under India's DPDP Act — dropping transcription does not remove this obligation, since pitch, MFCCs, and vocal-tract characteristics are themselves biometric identifiers. Consent language, storage limits, and access controls that currently apply to raw audio should apply equally to the derived feature vectors, not just the audio file itself.

---

## 12. Rollout plan

1. **Shadow mode**: run this pipeline against the existing 331 audited/labeled responses (78 High / 253 Low ground truth) without affecting production flags.
2. **Compare** the resulting confusion matrix (accuracy, precision, recall) against the current baseline (45.7% accuracy, 15.5% precision, 34% recall on clean data) to confirm this is actually an improvement before cutover.
3. **Tune thresholds** (`TRACK_A_THRESHOLD`, Track C rule thresholds) against that labeled set — start conservative (favor precision) if false-positive cost is currently the bigger customer-trust risk, or favor recall if missed cases are the bigger concern; this is a product call, not a purely technical one.
4. **Cut over gradually** — e.g. run both old and new pipelines in parallel for one full customer cycle, with the new pipeline's flags visible to reviewers but not yet authoritative, before fully retiring the transcript-based path.
5. **Re-validate thresholds periodically** as labeled volume grows — especially once Track C accumulates enough confirmed cases to consider a learned classifier instead of hand-tuned rules.

---

## 13. Known limitations (be upfront about these)

- Track A cannot help at all until a candidate has answered enough questions (cold-start gap, mitigated but not eliminated by Track C).
- Acoustic features can correlate with accent, disability, or speaking style — every threshold must be validated against self-baseline deviation, not absolute cutoffs, to avoid a new fairness risk replacing the old grammar-based one.
- Environmental/hardware variance (different microphones, room types across candidates) is not fully controlled for by self-baseline comparison within one interview — a candidate who changes devices mid-interview for an innocent reason (e.g. dropped call, reconnect) could trigger `acoustic_environment_shift` and needs a clear reviewer-facing explanation, not an automatic high-severity flag.
- This pipeline has not yet been validated end-to-end on real data — the rollout plan above is not optional; treat the numbers in this document as a design target, not a proven result.