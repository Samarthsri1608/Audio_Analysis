"""
pipeline/feature_extractor.py — Audio-only feature extraction for the V4 proctoring pipeline.

Computes a shared feature vector per answer from the raw waveform alone.
No ASR is invoked here — all signals are purely acoustic.

Features computed (per pipeline spec §3):
  - f0_mean, f0_std          — pitch level and variation (pyin)
  - speech_rate_proxy        — voiced-frame fraction / duration
  - pause_ratio              — silence fraction of total duration
  - pause_duration_mean/max  — shape of pausing behaviour
  - response_latency         — time from audio start to first voiced frame
  - energy_mean, energy_std  — RMS energy over voiced frames
  - mfcc_mean, mfcc_std      — timbre / vocal-tract consistency (13 coefficients)
  - spectral_flatness_mean   — monotone / scripted-delivery proxy

  Additional (Track C support):
  - room_fingerprint         — mean spectral rolloff over silence segments (room proxy)
  - noise_floor_centroid     — spectral centroid of noise floor (background-change proxy)

Evaluability gates (per pipeline spec §8) are checked here and returned as
a structured (evaluable, reason, features) triple so the caller can always
populate `not_evaluable_reason` rather than silently omitting fields.
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
# 60th percentile — same heuristic as the reference implementation in spec §10.
VOICED_ENERGY_PERCENTILE: int = 60
# Minimum voiced frames required to compute pitch reliably.
MIN_VOICED_FRAMES: int = 10
# Silence segments shorter than this (seconds) are treated as micro-pauses,
# not actual pauses (avoids inflating pause counts on natural inter-syllable gaps).
MIN_PAUSE_DURATION_S: float = 0.05


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


def _voiced_mask_from_energy(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Build a binary voiced/silence mask using energy-percentile VAD.

    A frame is voiced if its RMS energy exceeds the 60th percentile of
    all frame energies — simple, fast, no external VAD library required.
    Replace with webrtcvad in production for more reliable VAD.
    """
    frame_len = int(VAD_FRAME_S * sr)
    hop = frame_len
    energy = np.array([
        np.sqrt(np.mean(y[i: i + frame_len] ** 2))
        for i in range(0, len(y) - frame_len, hop)
    ])
    if len(energy) == 0:
        return np.array([], dtype=bool)
    threshold = np.percentile(energy, VOICED_ENERGY_PERCENTILE)
    return energy > threshold


# ── Evaluability gate ─────────────────────────────────────────────────────────

def _check_evaluable(
    total_duration_s: float,
    speech_duration_s: float,
    voiced_mask: np.ndarray,
    load_error: Optional[str],
) -> EvaluabilityResult:
    """
    Determine whether an answer is usable for analysis.

    Returns an EvaluabilityResult with evaluable=True only if:
    - Audio loaded without error
    - Total duration meets minimum
    - Voiced speech meets minimum duration
    - At least some voiced frames detected
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
    Extract the full audio-only feature vector for a single answer WAV file.

    Returns:
        (EvaluabilityResult, AudioFeatures | None)

    AudioFeatures is None when evaluable=False. Individual feature fields
    may still be None within a valid AudioFeatures object when they cannot
    be computed for that answer (e.g. no voiced frames → no pitch).

    All exceptions are caught and returned as an evaluability failure rather
    than propagated, to keep the per-question pipeline stateless.
    """
    import librosa  # lazy import — keep server startup fast

    # ── 1. Load audio ─────────────────────────────────────────────────────────
    load_error: Optional[str] = None
    y: Optional[np.ndarray] = None
    sr: int = 16_000
    total_duration_s: float = 0.0

    try:
        y, sr = librosa.load(wav_path, sr=16_000, mono=True)
        total_duration_s = len(y) / sr
    except FileNotFoundError:
        load_error = "file_not_found"
    except Exception as exc:
        logger.warning("Audio load failed for %s: %s", wav_path, exc)
        load_error = "corrupt_audio"

    if load_error:
        ev = EvaluabilityResult(evaluable=False, not_evaluable_reason=load_error)
        return ev, None

    # ── 2. VAD — build voiced/silence masks ───────────────────────────────────
    voiced_mask = _voiced_mask_from_energy(y, sr)
    frame_hop_s = VAD_FRAME_S  # seconds per frame
    silence_mask = ~voiced_mask

    speech_duration_s = float(voiced_mask.sum()) * frame_hop_s

    # ── 3. Evaluability gate ──────────────────────────────────────────────────
    ev = _check_evaluable(total_duration_s, speech_duration_s, voiced_mask, None)
    if not ev.evaluable:
        return ev, None

    # ── 4. Pitch (F0) via pyin over voiced frames ─────────────────────────────
    f0_mean: Optional[float] = None
    f0_std: Optional[float] = None
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),   # ~65 Hz
            fmax=librosa.note_to_hz("C7"),   # ~2093 Hz
            sr=sr,
        )
        f0_voiced = f0[~np.isnan(f0)]
        if len(f0_voiced) >= MIN_VOICED_FRAMES:
            f0_mean = float(np.mean(f0_voiced))
            f0_std = float(np.std(f0_voiced))
    except Exception as exc:
        logger.debug("pyin failed for %s: %s", wav_path, exc)

    # ── 5. Speaking-rate proxy (voiced-frame fraction / duration) ─────────────
    speech_rate_proxy: Optional[float] = None
    if total_duration_s > 0:
        speech_rate_proxy = float(voiced_mask.sum() / total_duration_s)

    # ── 6. Response latency — time from audio start to first voiced frame ─────
    response_latency: Optional[float] = None
    first_voiced = int(np.argmax(voiced_mask)) if voiced_mask.any() else None
    if first_voiced is not None:
        response_latency = float(first_voiced * frame_hop_s)
    else:
        response_latency = total_duration_s  # no speech → treat entire duration as latency

    # ── 7. Pause ratio + pause duration distribution ──────────────────────────
    pause_ratio: Optional[float] = None
    pause_duration_mean: Optional[float] = None
    pause_duration_max: Optional[float] = None
    try:
        pause_ratio = float(silence_mask.sum() / max(len(voiced_mask), 1))
        pause_runs_frames = _run_lengths(silence_mask)
        pause_runs_s = pause_runs_frames * frame_hop_s
        # Filter out sub-perceptual micro-pauses
        meaningful_pauses = pause_runs_s[pause_runs_s >= MIN_PAUSE_DURATION_S]
        if len(meaningful_pauses) > 0:
            pause_duration_mean = float(np.mean(meaningful_pauses))
            pause_duration_max = float(np.max(meaningful_pauses))
        else:
            pause_duration_mean = 0.0
            pause_duration_max = 0.0
    except Exception as exc:
        logger.debug("Pause computation failed for %s: %s", wav_path, exc)

    # ── 8. RMS energy over voiced frames ─────────────────────────────────────
    energy_mean: Optional[float] = None
    energy_std: Optional[float] = None
    try:
        frame_len = int(VAD_FRAME_S * sr)
        hop = frame_len
        all_energies = np.array([
            np.sqrt(np.mean(y[i: i + frame_len] ** 2))
            for i in range(0, len(y) - frame_len, hop)
        ])
        # Restrict to voiced frames only (mask may be shorter due to boundary)
        n = min(len(all_energies), len(voiced_mask))
        voiced_energies = all_energies[:n][voiced_mask[:n]]
        if len(voiced_energies) > 0:
            energy_mean = float(np.mean(voiced_energies))
            energy_std = float(np.std(voiced_energies))
    except Exception as exc:
        logger.debug("Energy computation failed for %s: %s", wav_path, exc)

    # ── 9. MFCCs (timbre / vocal-tract consistency) ───────────────────────────
    mfcc_mean: Optional[list[float]] = None
    mfcc_std: Optional[list[float]] = None
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = mfcc.mean(axis=1).tolist()
        mfcc_std = mfcc.std(axis=1).tolist()
    except Exception as exc:
        logger.debug("MFCC computation failed for %s: %s", wav_path, exc)

    # ── 10. Spectral flatness (monotone/scripted-delivery proxy) ─────────────
    spectral_flatness_mean: Optional[float] = None
    try:
        flatness = librosa.feature.spectral_flatness(y=y)
        spectral_flatness_mean = float(np.mean(flatness))
    except Exception as exc:
        logger.debug("Spectral flatness computation failed for %s: %s", wav_path, exc)

    # ── 11. Room fingerprint — spectral rolloff over silence frames ───────────
    # Used by Track C to detect mid-interview acoustic environment changes.
    # We compute the spectral rolloff (frequency below which 85% of energy lies)
    # over silence segments — this captures room reverb signature without ASR.
    room_fingerprint: Optional[float] = None
    noise_floor_centroid: Optional[float] = None
    try:
        # Extract silence (noise floor) audio segments
        silence_frames = _run_lengths(silence_mask)
        if len(silence_frames) > 0 and silence_mask.any():
            # Build a noise-floor signal by concatenating silence frame regions
            frame_len = int(VAD_FRAME_S * sr)
            silence_y_parts: list[np.ndarray] = []
            in_silence = False
            run_start = 0
            for idx, is_silent in enumerate(silence_mask):
                if is_silent and not in_silence:
                    run_start = idx
                    in_silence = True
                elif not is_silent and in_silence:
                    start_sample = run_start * frame_len
                    end_sample = idx * frame_len
                    silence_y_parts.append(y[start_sample:end_sample])
                    in_silence = False
            if in_silence:
                start_sample = run_start * frame_len
                silence_y_parts.append(y[start_sample:])

            if silence_y_parts:
                noise_y = np.concatenate(silence_y_parts)
                if len(noise_y) > sr * 0.1:  # at least 100ms of noise
                    rolloff = librosa.feature.spectral_rolloff(y=noise_y, sr=sr, roll_percent=0.85)
                    room_fingerprint = float(np.mean(rolloff))
                    centroid = librosa.feature.spectral_centroid(y=noise_y, sr=sr)
                    noise_floor_centroid = float(np.mean(centroid))
    except Exception as exc:
        logger.debug("Room fingerprint computation failed for %s: %s", wav_path, exc)

    # ── SNR sanity check ──────────────────────────────────────────────────────
    # If energy_mean is extremely low (near-digital-silence), flag as low quality.
    if energy_mean is not None and energy_mean < 1e-5:
        return EvaluabilityResult(
            evaluable=False, not_evaluable_reason="low_signal_quality"
        ), None

    features = AudioFeatures(
        f0_mean=round(f0_mean, 4) if f0_mean is not None else None,
        f0_std=round(f0_std, 4) if f0_std is not None else None,
        speech_rate_proxy=round(speech_rate_proxy, 4) if speech_rate_proxy is not None else None,
        pause_ratio=round(pause_ratio, 4) if pause_ratio is not None else None,
        pause_duration_mean=round(pause_duration_mean, 4) if pause_duration_mean is not None else None,
        pause_duration_max=round(pause_duration_max, 4) if pause_duration_max is not None else None,
        response_latency=round(response_latency, 4) if response_latency is not None else None,
        energy_mean=round(energy_mean, 6) if energy_mean is not None else None,
        energy_std=round(energy_std, 6) if energy_std is not None else None,
        mfcc_mean=[round(v, 4) for v in mfcc_mean] if mfcc_mean is not None else None,
        mfcc_std=[round(v, 4) for v in mfcc_std] if mfcc_std is not None else None,
        spectral_flatness_mean=round(spectral_flatness_mean, 6) if spectral_flatness_mean is not None else None,
        room_fingerprint=round(room_fingerprint, 2) if room_fingerprint is not None else None,
        noise_floor_centroid=round(noise_floor_centroid, 2) if noise_floor_centroid is not None else None,
        total_duration_s=round(total_duration_s, 3),
        speech_duration_s=round(speech_duration_s, 3),
    )

    return EvaluabilityResult(evaluable=True), features
