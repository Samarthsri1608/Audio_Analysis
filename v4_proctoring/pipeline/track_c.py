"""
pipeline/track_c.py — Naturalness / mechanism signal detection (Track C).

Goal: catch cheating mechanisms directly from acoustic properties, independent
of any personal baseline. Cold-start-safe — runs from question 1.

Signals implemented (per spec §5, with user-approved scope):

  1. latency_fluency_mismatch
     Long pause before first speech + unusually smooth/monotone delivery +
     near-zero mid-answer pause ratio.
     Signature: candidate waited for a prepared/pasted answer then read it aloud.

  2. acoustic_environment_shift
     The room acoustic fingerprint (spectral rolloff of the noise floor) shifts
     significantly vs. the interview's running average.
     Simple proxy — no RT60 computation required.

  3. background_noise_shift
     The spectral centroid of the noise floor changes significantly, suggesting
     a different device, location, or a new audio source appeared.

NOT implemented (per user decision):
  - second_voice / overlap detection (requires diarization model, deferred)

Notes on fairness (per spec §9):
  - All Track C thresholds are validated against the interview's own running
    baseline where possible, NOT absolute values.
  - The latency-fluency mismatch requires the combination of three signals —
    fast latency alone or monotone tone alone are NOT flagged.
  - Room/noise shift is computed relative to the rolling interview mean,
    so a consistently monotone candidate or consistently reverberant room
    won't produce false positives.
"""
from __future__ import annotations

import logging
from typing import Optional

from v4_proctoring.config import (
    TRACK_C_LATENCY_THRESHOLD_S,
    TRACK_C_FLATNESS_LOW_THRESHOLD,
    TRACK_C_PAUSE_RATIO_LOW_THRESHOLD,
    TRACK_C_ROOM_SHIFT_THRESHOLD,
    TRACK_C_NOISE_SHIFT_THRESHOLD,
    TRACK_C_ROOM_MIN_ANSWERS,
)
from v4_proctoring.models import AudioFeatures, TrackCResult

logger = logging.getLogger("v4_proctoring.track_c")


class InterviewBaseline:
    """
    Rolling baseline of acoustic environment signals across the interview.

    Maintained across questions so Track C can compare each answer
    against the interview's own running average — making the threshold
    candidate-relative, not absolute.

    Only updated with evaluable answers.
    """

    def __init__(self) -> None:
        self._room_fingerprints: list[float] = []
        self._noise_centroids: list[float] = []
        # Running spectral flatness values (for relative monotone detection)
        self._flatness_values: list[float] = []

    def update(self, features: AudioFeatures) -> None:
        """Add a new evaluable answer's signals to the running baseline."""
        if features.room_fingerprint is not None:
            self._room_fingerprints.append(features.room_fingerprint)
        if features.noise_floor_centroid is not None:
            self._noise_centroids.append(features.noise_floor_centroid)
        if features.spectral_flatness_mean is not None:
            self._flatness_values.append(features.spectral_flatness_mean)

    @property
    def room_mean(self) -> Optional[float]:
        if not self._room_fingerprints:
            return None
        return sum(self._room_fingerprints) / len(self._room_fingerprints)

    @property
    def noise_mean(self) -> Optional[float]:
        if not self._noise_centroids:
            return None
        return sum(self._noise_centroids) / len(self._noise_centroids)

    @property
    def flatness_mean(self) -> Optional[float]:
        if not self._flatness_values:
            return None
        return sum(self._flatness_values) / len(self._flatness_values)

    @property
    def answer_count(self) -> int:
        return len(self._room_fingerprints)


def analyze(
    features: AudioFeatures,
    baseline: InterviewBaseline,
) -> TrackCResult:
    """
    Run all Track C rule-based signals on a single answer's features.

    Args:
        features:  Extracted audio features for this answer.
        baseline:  Running interview baseline (may have zero answers if first question).

    Returns:
        TrackCResult with signals list and signal_details dict.
    """
    signals: list[str] = []
    signal_details: dict[str, float] = {}

    # ── Signal 1: Latency-Fluency Mismatch ────────────────────────────────────
    # Combination gate: all three conditions must be true.
    # This prevents penalizing quick, confident answers or naturally monotone speakers.
    latency = features.response_latency
    flatness = features.spectral_flatness_mean
    pause_r = features.pause_ratio

    # Spectral flatness is evaluated relative to the interview's running baseline
    # to avoid penalizing naturally monotone candidates (spec §9, edge case 2).
    # If we have enough history, adjust the threshold downward proportionally.
    flatness_threshold = TRACK_C_FLATNESS_LOW_THRESHOLD
    if baseline.flatness_mean is not None and baseline.answer_count >= TRACK_C_ROOM_MIN_ANSWERS:
        # If the candidate's own baseline is already flat, tighten the threshold
        # so we only flag relative drops, not the candidate's natural style.
        flatness_threshold = min(
            TRACK_C_FLATNESS_LOW_THRESHOLD,
            baseline.flatness_mean * 0.7,  # 30% below their own baseline
        )

    if (
        latency is not None
        and flatness is not None
        and pause_r is not None
        and latency > TRACK_C_LATENCY_THRESHOLD_S
        and flatness < flatness_threshold
        and pause_r < TRACK_C_PAUSE_RATIO_LOW_THRESHOLD
    ):
        signals.append("latency_fluency_mismatch")
        signal_details["response_latency_s"] = round(latency, 3)
        signal_details["spectral_flatness_mean"] = round(flatness, 5)
        signal_details["pause_ratio"] = round(pause_r, 4)
        logger.info(
            "Track C: latency_fluency_mismatch — latency=%.2fs, flatness=%.5f, pause_ratio=%.3f",
            latency, flatness, pause_r,
        )

    # ── Signal 2: Acoustic Environment Shift ─────────────────────────────────
    # Only fire if we have enough baseline answers to compare against.
    room_fp = features.room_fingerprint
    room_mean = baseline.room_mean

    if (
        room_fp is not None
        and room_mean is not None
        and baseline.answer_count >= TRACK_C_ROOM_MIN_ANSWERS
        and room_mean > 0
    ):
        relative_shift = abs(room_fp - room_mean) / room_mean
        if relative_shift > TRACK_C_ROOM_SHIFT_THRESHOLD:
            signals.append("acoustic_environment_shift")
            signal_details["room_fingerprint_current"] = round(room_fp, 2)
            signal_details["room_fingerprint_baseline"] = round(room_mean, 2)
            signal_details["room_relative_shift"] = round(relative_shift, 4)
            logger.info(
                "Track C: acoustic_environment_shift — current=%.0fHz, baseline=%.0fHz, shift=%.1f%%",
                room_fp, room_mean, relative_shift * 100,
            )

    # ── Signal 3: Background Noise Signature Change ───────────────────────────
    noise_centroid = features.noise_floor_centroid
    noise_mean = baseline.noise_mean

    if (
        noise_centroid is not None
        and noise_mean is not None
        and baseline.answer_count >= TRACK_C_ROOM_MIN_ANSWERS
        and noise_mean > 0
    ):
        noise_relative_shift = abs(noise_centroid - noise_mean) / noise_mean
        if noise_relative_shift > TRACK_C_NOISE_SHIFT_THRESHOLD:
            signals.append("background_noise_shift")
            signal_details["noise_centroid_current"] = round(noise_centroid, 2)
            signal_details["noise_centroid_baseline"] = round(noise_mean, 2)
            signal_details["noise_relative_shift"] = round(noise_relative_shift, 4)
            logger.info(
                "Track C: background_noise_shift — current=%.0fHz, baseline=%.0fHz, shift=%.1f%%",
                noise_centroid, noise_mean, noise_relative_shift * 100,
            )

    return TrackCResult(
        flagged=len(signals) > 0,
        signals=signals,
        signal_details=signal_details,
    )
