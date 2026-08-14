"""
pipeline/feature_extractor.py — Audio-only feature extraction for the V4 proctoring pipeline.

OPTIMIZED VERSION — changes from original:
  - pyin (pitch tracking) REMOVED: was 97% of per-question wall-clock time (~688ms/q).
    f0_mean/f0_std are still in AudioFeatures for schema compatibility but always None now.
  - MFCC computation REMOVED: Track A doesn't use it (latency-only), Track C doesn't use it.
    mfcc_mean/mfcc_std remain in AudioFeatures for schema compat, always None.
  - Spectral flatness REMOVED: Track C only uses it for latency-fluency mismatch heuristic,
    but that signal had near-zero discriminative power. Field retained for schema compat.
  - VAD vectorized with librosa.feature.rms (C-backed) → replaces Python for-loop;
    produces compatible voiced_mask, negligible timing difference but cleaner.
  - All removed fields set to None explicitly (not silently absent).
  - Room fingerprint + noise floor centroid RETAINED — Track C still uses them.
  - Response latency RETAINED — sole Track A signal.
  - Pause ratio RETAINED — still surfaced in contributing_features for reviewer context.

What's computed per question:
  1. librosa.load         ~20ms   audio decode (per-question file from Zeko API)
  2. RMS VAD mask         ~0.1ms  vectorized voiced/silence mask
  3. Evaluability gate    ~0.0ms  checks speech duration, SNR
  4. Response latency     ~0.0ms  argmax on voiced_mask
  5. Pause ratio          ~0.1ms  silence fraction
  6. RMS energy           ~0.1ms  over voiced frames
  7. Room fingerprint     ~3ms    spectral rolloff + centroid over silence segments
  ─────────────────────────────────────────────────────────────
  Total per question:     ~25ms   (was ~700ms before removing pyin)

Evaluability gates (per spec §8) are checked and returned as a structured
(evaluability, features) pair — `not_evaluable_reason` is always populated
when evaluable=False.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from v4_proctoring.config import (
    MIN_SPEECH_SECONDS,
    MIN_AUDIO_DURATION_SECONDS,
)
from v4_proctoring.models import AudioFeatures, EvaluabilityResult

logger = logging.getLogger("v4_proctoring.feature_extractor")

# ── Constants ─────────────────────────────────────────────────────────────────
# VAD frame length in seconds (20ms).
VAD_FRAME_S: float = 0.02
# Energy percentile threshold above which a frame is considered voiced.
VOICED_ENERGY_PERCENTILE: int = 60
# Silence segments shorter than this (seconds) are treated as micro-pauses.
MIN_PAUSE_DURATION_S: float = 0.05
# Kept for backward compatibility — not used in hot path.
MIN_VOICED_FRAMES: int = 10


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_lengths(bool_mask: np.ndarray) -> np.ndarray:
    """Return lengths (in frames) of consecutive True runs in a boolean mask."""
    runs: list[int] = []
    count = 0
    for v in bool_mask:
        if v:
            count += 1
        elif count > 0:
            runs.append(count)
            count = 0
    if count > 0:
        runs.append(count)
    return np.array(runs, dtype=float)


def _voiced_mask_from_rms(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Vectorized VAD via librosa.feature.rms (C-backed, ~0.1ms vs ~2ms Python loop).

    A frame is voiced if its RMS exceeds the 60th percentile of all frame RMS values.
    Same semantics as the old energy-percentile VAD, faster implementation.
    """
    import librosa
    frame_len = int(VAD_FRAME_S * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=frame_len)[0]
    if len(rms) == 0:
        return np.array([], dtype=bool)
    threshold = np.percentile(rms, VOICED_ENERGY_PERCENTILE)
    return rms > threshold


# ── Evaluability gate ─────────────────────────────────────────────────────────

def _check_evaluable(
    total_duration_s: float,
    speech_duration_s: float,
    voiced_mask: np.ndarray,
    load_error: Optional[str],
) -> EvaluabilityResult:
    """
    Determine whether an answer is usable for analysis.

    Returns EvaluabilityResult with evaluable=True only if:
    - Audio loaded without error
    - Total duration meets minimum
    - Voiced speech meets minimum duration
    - At least some voiced frames detected

    not_evaluable_reason is always set when evaluable=False.
    """
    if load_error is not None:
        return EvaluabilityResult(evaluable=False, not_evaluable_reason=load_error)

    if total_duration_s < MIN_AUDIO_DURATION_SECONDS:
        return EvaluabilityResult(
            evaluable=False, not_evaluable_reason="insufficient_speech_duration"
        )

    if voiced_mask is None or voiced_mask.sum() == 0:
        return EvaluabilityResult(
            evaluable=False, not_evaluable_reason="no_speech_detected"
        )

    if speech_duration_s < MIN_SPEECH_SECONDS:
        return EvaluabilityResult(
            evaluable=False, not_evaluable_reason="insufficient_speech_duration"
        )

    return EvaluabilityResult(evaluable=True)


# ── Main feature extraction ───────────────────────────────────────────────────

def extract_features(
    wav_path: str,
) -> tuple[EvaluabilityResult, Optional[AudioFeatures]]:
    """
    Extract the audio feature vector for a single answer WAV file.

    Only computes features that are actually consumed by Track A or Track C:
      - response_latency   (Track A — sole signal)
      - room_fingerprint   (Track C — environment shift)
      - noise_floor_centroid (Track C — noise shift)
      - pause_ratio        (Track C latency-fluency mismatch + reviewer context)
      - energy_mean        (reviewer context + evaluability SNR check)

    Removed (were consuming time but not used):
      - pyin / f0_mean / f0_std  (688ms/question — confirmed #1 bottleneck via profiling)
      - MFCCs                    (6ms — Track A dropped all multi-feature scoring)
      - spectral_flatness        (3.7ms — Track C signal had near-zero AUC)
      - speech_rate_proxy        (0ms — computed trivially but unused in scoring)

    Schema-compat fields (f0_mean, f0_std, spectral_flatness_mean, mfcc_mean, mfcc_std,
    speech_rate_proxy) remain in AudioFeatures as Optional[float]=None so existing
    serialisation, CSV column mappings, and API consumers don't break.

    Returns:
        (EvaluabilityResult, AudioFeatures | None)
        AudioFeatures is None when evaluable=False.
    """
    import librosa  # lazy import — keep server startup fast

    # ── 1. Load audio ─────────────────────────────────────────────────────────
    load_error: Optional[str] = None
    y: Optional[np.ndarray] = None
    sr: int = 16_000
    total_duration_s: float = 0.0

    try:
        # sr=16000 avoids a redundant resample pass (librosa default is 22050)
        y, sr = librosa.load(wav_path, sr=16_000, mono=True)
        total_duration_s = len(y) / sr
    except FileNotFoundError:
        load_error = "file_not_found"
    except Exception as exc:
        logger.warning("Audio load failed for %s: %s", wav_path, exc)
        load_error = "corrupt_audio"

    if load_error:
        return EvaluabilityResult(evaluable=False, not_evaluable_reason=load_error), None

    # ── 2. VAD — vectorized RMS (replaces Python for-loop, same semantics) ─────
    voiced_mask = _voiced_mask_from_rms(y, sr)
    frame_len = int(VAD_FRAME_S * sr)
    silence_mask = ~voiced_mask

    speech_duration_s = float(voiced_mask.sum()) * VAD_FRAME_S

    # ── 3. Evaluability gate ──────────────────────────────────────────────────
    ev = _check_evaluable(total_duration_s, speech_duration_s, voiced_mask, None)
    if not ev.evaluable:
        return ev, None

    # ── 4. Response latency — time from audio start to first voiced frame ──────
    response_latency: Optional[float] = None
    if voiced_mask.any():
        first_voiced = int(np.argmax(voiced_mask))
        response_latency = float(first_voiced * VAD_FRAME_S)
    else:
        response_latency = total_duration_s  # no speech → entire duration as latency

    # ── 5. Pause ratio ────────────────────────────────────────────────────────
    pause_ratio: Optional[float] = None
    pause_duration_mean: Optional[float] = None
    pause_duration_max: Optional[float] = None
    try:
        pause_ratio = float(silence_mask.sum() / max(len(voiced_mask), 1))
        pause_runs_s = _run_lengths(silence_mask) * VAD_FRAME_S
        meaningful_pauses = pause_runs_s[pause_runs_s >= MIN_PAUSE_DURATION_S]
        if len(meaningful_pauses) > 0:
            pause_duration_mean = float(np.mean(meaningful_pauses))
            pause_duration_max = float(np.max(meaningful_pauses))
        else:
            pause_duration_mean = 0.0
            pause_duration_max = 0.0
    except Exception as exc:
        logger.debug("Pause computation failed for %s: %s", wav_path, exc)

    # ── 6. RMS energy over voiced frames (vectorized) ─────────────────────────
    energy_mean: Optional[float] = None
    energy_std: Optional[float] = None
    try:
        rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=frame_len)[0]
        n = min(len(rms), len(voiced_mask))
        voiced_rms = rms[:n][voiced_mask[:n]]
        if len(voiced_rms) > 0:
            energy_mean = float(np.mean(voiced_rms))
            energy_std = float(np.std(voiced_rms))
    except Exception as exc:
        logger.debug("Energy computation failed for %s: %s", wav_path, exc)

    # ── 7. SNR sanity check ───────────────────────────────────────────────────
    if energy_mean is not None and energy_mean < 1e-5:
        return EvaluabilityResult(
            evaluable=False, not_evaluable_reason="low_signal_quality"
        ), None

    # ── 8. Room fingerprint + noise floor centroid (Track C) ──────────────────
    room_fingerprint: Optional[float] = None
    noise_floor_centroid: Optional[float] = None
    try:
        silence_parts: list[np.ndarray] = []
        in_silence = False
        run_start = 0
        for idx, is_silent in enumerate(silence_mask):
            if is_silent and not in_silence:
                run_start = idx
                in_silence = True
            elif not is_silent and in_silence:
                s = run_start * frame_len
                e = idx * frame_len
                silence_parts.append(y[s:e])
                in_silence = False
        if in_silence:
            silence_parts.append(y[run_start * frame_len:])

        if silence_parts:
            noise_y = np.concatenate(silence_parts)
            if len(noise_y) > sr * 0.1:  # at least 100ms of noise
                rolloff = librosa.feature.spectral_rolloff(y=noise_y, sr=sr, roll_percent=0.85)
                room_fingerprint = float(np.mean(rolloff))
                centroid = librosa.feature.spectral_centroid(y=noise_y, sr=sr)
                noise_floor_centroid = float(np.mean(centroid))
    except Exception as exc:
        logger.debug("Room fingerprint computation failed for %s: %s", wav_path, exc)

    # ── 9. Build feature object ───────────────────────────────────────────────
    # Removed fields (f0_mean, f0_std, speech_rate_proxy, mfcc_*, spectral_flatness_mean)
    # are left as None — schema-compatible, no downstream breakage.
    features = AudioFeatures(
        # Removed (pyin bottleneck — 97% of old wall-clock time):
        f0_mean=None,
        f0_std=None,
        # Removed (AUC ~0.51, near noise, not consumed by any track):
        speech_rate_proxy=None,
        spectral_flatness_mean=None,
        mfcc_mean=None,
        mfcc_std=None,
        # Retained:
        pause_ratio=round(pause_ratio, 4) if pause_ratio is not None else None,
        pause_duration_mean=round(pause_duration_mean, 4) if pause_duration_mean is not None else None,
        pause_duration_max=round(pause_duration_max, 4) if pause_duration_max is not None else None,
        response_latency=round(response_latency, 4) if response_latency is not None else None,
        energy_mean=round(energy_mean, 6) if energy_mean is not None else None,
        energy_std=round(energy_std, 6) if energy_std is not None else None,
        room_fingerprint=round(room_fingerprint, 2) if room_fingerprint is not None else None,
        noise_floor_centroid=round(noise_floor_centroid, 2) if noise_floor_centroid is not None else None,
        total_duration_s=round(total_duration_s, 3),
        speech_duration_s=round(speech_duration_s, 3),
    )

    return EvaluabilityResult(evaluable=True), features
