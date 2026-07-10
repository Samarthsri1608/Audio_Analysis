"""
config.py — V4 Audio-Only Proctoring pipeline settings.

All defaults are safe for local development. No ASR keys required —
this pipeline uses audio features exclusively; no transcription is performed
in the detection path.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Interview API (same as v3) ────────────────────────────────────────────────
INTERVIEW_API_BASE: str = os.getenv(
    "INTERVIEW_API_BASE",
    "https://interview-api.zeko.ai/dashboard/api/v2/report/recordings/question",
)
MAX_QUESTIONS: int = int(os.getenv("MAX_QUESTIONS", "25"))

# ── Audio processing ──────────────────────────────────────────────────────────
SAMPLE_RATE: int = 16_000        # Hz — 16kHz mono WAV
MAX_DURATION_MINUTES: int = int(os.getenv("MAX_DURATION_MINUTES", "60"))

# ── Temp dir prefix ───────────────────────────────────────────────────────────
TEMP_DIR_PREFIX: str = "v4_proctoring_"

# ── Concurrency ───────────────────────────────────────────────────────────────
MAX_CONCURRENT_JOBS: int = int(os.getenv("MAX_CONCURRENT_JOBS", "10"))
THREAD_POOL_WORKERS: int = int(os.getenv("THREAD_POOL_WORKERS", "4"))

# ── Security ──────────────────────────────────────────────────────────────────
INTERNAL_PROCTORING_TOKEN: str = os.getenv("INTERNAL_PROCTORING_TOKEN", "")

# ── Evaluability gates ────────────────────────────────────────────────────────
# Minimum duration (seconds of detected speech) for an answer to be evaluable.
MIN_SPEECH_SECONDS: float = 2.0
# Minimum total audio duration (seconds) — answers shorter than this are skipped.
MIN_AUDIO_DURATION_SECONDS: float = 1.0

# ── Track A — self-baseline deviation ─────────────────────────────────────────
# Minimum number of the candidate's own evaluable answers before Track A fires.
TRACK_A_MIN_ANSWERS: int = 3
# Robust z-score threshold (max absolute z across features) to flag an answer.
# UNCALIBRATED — tune against labeled data (78 High / 253 Low ground truth).
TRACK_A_THRESHOLD: float = 3.5

# Features used in Track A z-score (keys match AudioFeatures fields).
TRACK_A_FEATURE_KEYS: list[str] = [
    "f0_mean",
    "f0_std",
    "speech_rate_proxy",
    "pause_ratio",
    "response_latency",
    "energy_mean",
    "spectral_flatness_mean",
]

# ── Track C — naturalness / mechanism rules ───────────────────────────────────
# Latency-fluency mismatch: long silence before answer + unusually smooth delivery
# UNCALIBRATED thresholds — starting point only.
TRACK_C_LATENCY_THRESHOLD_S: float = 4.0       # response latency > this (seconds)
TRACK_C_FLATNESS_LOW_THRESHOLD: float = 0.15   # spectral flatness mean < this (very monotone)
TRACK_C_PAUSE_RATIO_LOW_THRESHOLD: float = 0.05 # pause_ratio < this within the answer

# Room / acoustic-environment shift (spectral rolloff fingerprint).
# Flag if per-answer room-fingerprint deviates > this fraction from interview running mean.
TRACK_C_ROOM_SHIFT_THRESHOLD: float = 0.20     # relative shift (0–1 range normalized)

# Background noise signature change: spectral flux of noise floor between voiced segments.
# Flag if noise-floor spectral centroid shifts by more than this fraction.
TRACK_C_NOISE_SHIFT_THRESHOLD: float = 0.25

# Minimum answers needed to build a running room-fingerprint baseline.
TRACK_C_ROOM_MIN_ANSWERS: int = 2
