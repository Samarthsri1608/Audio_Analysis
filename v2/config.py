"""
config.py — V2 pipeline settings loaded from environment variables.
All defaults are safe for local development.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── API keys ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# ── Local Whisper settings ────────────────────────────────────────────────────
WHISPER_LOCAL_MODEL: str = os.getenv("WHISPER_LOCAL_MODEL", "base")
WHISPER_LOCAL_DEVICE: str = os.getenv("WHISPER_LOCAL_DEVICE", "auto")
WHISPER_LOCAL_COMPUTE_TYPE: str = os.getenv("WHISPER_LOCAL_COMPUTE_TYPE", "default")
WHISPER_LOCAL_CPU_THREADS: int = int(os.getenv("WHISPER_LOCAL_CPU_THREADS", "0"))
WHISPER_LOCAL_WORKERS: int = int(os.getenv("WHISPER_LOCAL_WORKERS", "1"))

# ── Interview API ─────────────────────────────────────────────────────────────
INTERVIEW_API_BASE: str = os.getenv(
    "INTERVIEW_API_BASE",
    "https://interview-api.zeko.ai/dashboard/api/v2/report/recordings/question",
)
MAX_QUESTIONS: int = int(os.getenv("MAX_QUESTIONS", "25"))

# ── Audio processing ──────────────────────────────────────────────────────────
SAMPLE_RATE: int = 16_000        # Hz — 16kHz mono WAV for Whisper
MAX_DURATION_MINUTES: int = int(os.getenv("MAX_DURATION_MINUTES", "60"))

# ── Concurrency ───────────────────────────────────────────────────────────────
MAX_CONCURRENT_JOBS: int = int(os.getenv("MAX_CONCURRENT_JOBS", "10"))
THREAD_POOL_WORKERS: int = int(os.getenv("THREAD_POOL_WORKERS", "4"))

# ── Feature normalization bounds (from framework typical ranges) ──────────────
# Used for min-max normalization to 0-100 scale (System B normalizer).
FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    # Legacy aliases (System B compatibility)
    "logical_connector_density":   (0.0,  0.20),
    "avg_sentence_length":         (5.0,  25.0),
    "filler_word_ratio":           (0.0,  0.15),
    "collaborative_language_ratio":(0.0,  1.0),
    "question_density":            (0.0,  2.0),
    "empathetic_language_score":   (0.0,  1.0),
    "vocabulary_density":          (0.5,  1.0),   # MATTR range (content words)
    "metric_density":              (0.0,  6.0),
    "speech_rate_wpm":             (80.0, 200.0),
    "speech_rate_variability":     (0.0,  1.0),
    # New fields (System A + System B)
    "lexical_mattr":               (0.50, 0.95),  # MATTR 50-word window
    "lexical_rare_word_ratio":     (0.0,  0.30),
    "discourse_connectors":        (0.0,  20.0),
    "discourse_tier1":             (0.0,  10.0),
    "sbert_coherence":             (0.30, 1.0),   # calibrated targets after -0.06 offset
    "ner_entity_density":          (0.0,  8.0),
    "narrative_arc_score":         (0.0,  1.0),
    # Vocal — CV-normalized pitch (dialect-neutral)
    "pitch_variation":             (0.0,  0.70),  # CV ratio (not Hz)
    "vocal_confidence":            (0.0,  1.0),
    "speech_fluency":              (0.0,  1.0),
    "stress_markers":              (0.0,  1.0),
    "fluency_pause_dur":           (0.0,  15.0),
    "fluency_pause_freq":          (0.0,  60.0),
    "voiced_fraction":             (0.0,  1.0),
}

# ── LLM description ───────────────────────────────────────────────────────────
DESCRIPTION_MODEL: str = os.getenv("DESCRIPTION_MODEL", "gpt-4o-mini")
DESCRIPTION_MAX_TOKENS: int = int(os.getenv("DESCRIPTION_MAX_TOKENS", "250"))

# ── Temp directory ────────────────────────────────────────────────────────────
TEMP_DIR_PREFIX: str = "v2_pipeline_"
