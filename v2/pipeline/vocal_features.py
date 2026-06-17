"""
pipeline/vocal_features.py — Extract all audio-exclusive vocal features.

Implements the framework feature inventory (Section 1.1):
  F03 fluency_pause_dur    — mean pause duration in seconds
  F04 fluency_pause_freq   — pause count (gaps > 300ms)
  F10 pitch_std            — pitch variation (CV-normalized for Indian English)
  F17 vocal_confidence     — spectral stability (composure proxy)
  F18 speech_fluency_score — fraction of word transitions without long gaps
  F19 stress_markers       — composure: inverse of vocal fry + pitch tremor

Indian English correction (Framework §2.2, Axis 5):
  pitch_variation now uses the Coefficient of Variation (CV = std/mean)
  rather than raw Hz std dev. CV is speaker-relative and dialect-neutral —
  syllable-timed Indian English produces narrower absolute pitch swings in
  formal speech, which the old raw Hz metric misclassified as monotone delivery.

All functions are CPU-bound and run in a ThreadPoolExecutor.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from v2.pipeline.transcriber import WordTimestamp

logger = logging.getLogger("v2.vocal_features")


def _load_audio(wav_path: str):
    """Lazy import librosa and load audio. Returns (y, sr, librosa)."""
    import librosa  # noqa: PLC0415 — lazy import to keep startup fast
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    return y, sr, librosa


# ── F10: Pitch Variation (CV-normalized) ──────────────────────────────────────

def pitch_variation_cv(wav_path: str) -> float:
    """
    F10 pitch_std — Coefficient of Variation of fundamental frequency (F0).

    CV = std(F0) / mean(F0) across voiced frames.

    Using CV instead of raw Hz std dev makes the signal speaker-relative
    and dialect-neutral. Indian English syllable-timing produces narrower
    absolute pitch swings, which raw Hz would misclassify as monotone.
    The CV captures *relative* expressiveness independent of baseline pitch.

    Returns:
        float — CV ratio (typically 0.15–0.60). Higher = more expressive.
        0.0 if insufficient voiced frames.
    """
    try:
        y, sr, librosa = _load_audio(wav_path)
        f0 = librosa.yin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
        voiced = f0[f0 > 0]
        if len(voiced) < 10:
            return 0.0
        f0_mean = float(np.mean(voiced))
        f0_std = float(np.std(voiced))
        if f0_mean <= 0:
            return 0.0
        cv = f0_std / f0_mean
        return round(float(cv), 4)
    except Exception as exc:
        logger.warning("pitch_variation_cv failed: %s", exc)
        return 0.0


# ── F17: Vocal Confidence ─────────────────────────────────────────────────────

def vocal_confidence(wav_path: str) -> float:
    """
    F17 vocal_confidence — Stability of vocal power in voiced segments.

    Proxy for speaker composure and confidence: a steady, stable vocal energy
    pattern correlates with lower anxiety and higher assertiveness.

    Returns:
        float — 0–1 composure score. Higher = steadier, more confident.
    """
    try:
        y, sr, librosa = _load_audio(wav_path)
        D = librosa.stft(y)
        power = np.mean(np.abs(D) ** 2, axis=0)

        threshold = np.mean(power) * 0.5
        voiced_mask = power > threshold

        if not voiced_mask.any():
            return 0.5

        indices = np.where(voiced_mask)[0]
        splits = np.where(np.diff(indices) > 1)[0] + 1
        segments = np.split(indices, splits)

        stability_scores: list[float] = []
        for seg in segments:
            if len(seg) > 10:
                seg_power = power[seg]
                mean_p = np.mean(seg_power)
                if mean_p > 1e-10:
                    stability = 1.0 - np.std(seg_power) / mean_p
                    stability_scores.append(float(max(0.0, min(1.0, stability))))

        if not stability_scores:
            return 0.5
        return round(float(np.mean(stability_scores)), 4)
    except Exception as exc:
        logger.warning("vocal_confidence failed: %s", exc)
        return 0.5


# ── F18: Speech Fluency (from word timestamps) ────────────────────────────────

def speech_fluency(word_timestamps: "list[WordTimestamp]") -> float:
    """
    F18 speech_fluency_score — Fraction of word transitions without a gap > 200ms.

    Uses Whisper word timestamps — no audio processing needed.

    Returns:
        float — 0–1. Higher = smoother, fewer hesitations.
    """
    if len(word_timestamps) < 5:
        return 0.5

    gaps: list[float] = [
        word_timestamps[i]["start"] - word_timestamps[i - 1]["end"]
        for i in range(1, len(word_timestamps))
    ]

    long_gap_count = sum(1 for g in gaps if g > 0.2)
    pause_ratio = long_gap_count / len(gaps)
    return round(float(1.0 - min(pause_ratio, 1.0)), 4)


# ── F03/F04: Pause Duration & Frequency ───────────────────────────────────────

def pause_features(word_timestamps: "list[WordTimestamp]") -> dict[str, float]:
    """
    F03 fluency_pause_dur — Mean pause duration (seconds, pauses > 300ms only).
    F04 fluency_pause_freq — Count of pauses > 300ms.

    Pauses < 300ms are considered normal inter-word transitions and excluded.
    """
    if len(word_timestamps) < 2:
        return {"fluency_pause_dur": 0.0, "fluency_pause_freq": 0.0}

    pauses = [
        word_timestamps[i]["start"] - word_timestamps[i - 1]["end"]
        for i in range(1, len(word_timestamps))
        if word_timestamps[i]["start"] - word_timestamps[i - 1]["end"] > 0.3
    ]

    pause_count = len(pauses)
    mean_pause = round(float(sum(pauses) / pause_count), 4) if pauses else 0.0

    return {
        "fluency_pause_dur":  mean_pause,
        "fluency_pause_freq": float(pause_count),
    }


# ── F19: Stress / Anxiety Markers ─────────────────────────────────────────────

def stress_markers(wav_path: str) -> float:
    """
    F19 stress_markers — Composure score: inverse of vocal stress indicators.

    Measures:
    - Vocal fry: disproportionate low-frequency energy (stress indicator)
    - Pitch tremor: high coefficient of variation in F0 (anxiety marker)

    Returns:
        float — 0–1 composure. Higher = calmer, less stressed.
    """
    try:
        y, sr, librosa = _load_audio(wav_path)

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        low_freq_energy = float(np.mean(S[:10, :]))
        mid_freq_energy = float(np.mean(S[10:100, :]))
        energy_ratio = low_freq_energy / (mid_freq_energy + 1e-10)

        stress_fry = max(0.0, min(1.0, (energy_ratio - 0.5) / 2.0))

        f0 = librosa.yin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
        voiced = f0[f0 > 0]
        if len(voiced) > 10:
            f0_mean = float(np.mean(voiced))
            f0_std = float(np.std(voiced))
            cv = (f0_std / f0_mean) if f0_mean > 0 else 0.0
            stress_tremor = max(0.0, min(1.0, (cv - 0.05) / 0.35))
        else:
            stress_tremor = 0.5

        stress_score = stress_fry * 0.6 + stress_tremor * 0.4
        return round(float(1.0 - stress_score), 4)
    except Exception as exc:
        logger.warning("stress_markers failed: %s", exc)
        return 0.5


# ── Voiced fraction ───────────────────────────────────────────────────────────

def voiced_fraction(word_timestamps: "list[WordTimestamp]", duration_seconds: float) -> float:
    """
    Fraction of total audio time that is actively voiced speech.

    Computed from word timestamps rather than raw audio — accent-neutral.

    Returns:
        float — 0–1. Higher = more time spent speaking vs. silence.
    """
    if not word_timestamps or duration_seconds <= 0:
        return 0.0
    voiced_time = sum(wt["end"] - wt["start"] for wt in word_timestamps)
    return round(min(voiced_time / duration_seconds, 1.0), 4)


# ── Convenience: extract all vocal features ───────────────────────────────────

def extract_all_vocal_features(
    wav_path: str,
    word_timestamps: "list[WordTimestamp]",
    duration_seconds: float = 0.0,
) -> dict[str, float]:
    """
    Return all vocal features as a flat dict.
    Keys map directly to RawFeatures field names.
    """
    pauses = pause_features(word_timestamps)
    return {
        "pitch_variation":      pitch_variation_cv(wav_path),   # CV-normalized (Indian English safe)
        "vocal_confidence":     vocal_confidence(wav_path),
        "speech_fluency":       speech_fluency(word_timestamps),
        "stress_markers":       stress_markers(wav_path),
        "fluency_pause_dur":    pauses["fluency_pause_dur"],
        "fluency_pause_freq":   pauses["fluency_pause_freq"],
        "voiced_fraction":      voiced_fraction(word_timestamps, duration_seconds),
    }
