"""
config.py — V3 pipeline settings loaded from environment variables.
All defaults are safe for local development.

V3 change: Uses AssemblyAI for transcription (replaces OpenAI Whisper + faster-whisper).
"""
from __future__ import annotations

import os
from pathlib import Path

# ── API keys ──────────────────────────────────────────────────────────────────
ASSEMBLYAI_API_KEY: str = os.getenv("ASSEMBLYAI_API_KEY", "")
AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

# ── Gemini (Google) GenAI ──────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
# Safeguard: replace underscores with hyphens (e.g. gemini_2.5_flash_lite -> gemini-2.5-flash-lite)
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").replace("_", "-")

# ── Interview API ─────────────────────────────────────────────────────────────
INTERVIEW_API_BASE: str = os.getenv(
    "INTERVIEW_API_BASE",
    "https://interview-api.zeko.ai/dashboard/api/v2/report/recordings/question",
)
MAX_QUESTIONS: int = int(os.getenv("MAX_QUESTIONS", "25"))

# ── Audio processing ──────────────────────────────────────────────────────────
SAMPLE_RATE: int = 16_000        # Hz — 16kHz mono WAV for audio feature extraction
MAX_DURATION_MINUTES: int = int(os.getenv("MAX_DURATION_MINUTES", "60"))

# ── Temp dir prefix ───────────────────────────────────────────────────────────
TEMP_DIR_PREFIX: str = "v3_pipeline_"

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
    "fluency_pause_dur":           (0.0,  10.0),
    "fluency_pause_freq":          (0.0,  50.0),
    "voiced_fraction":             (0.0,  1.0),
    # ASR quality
    "intel_confidence":            (0.5,  1.0),
}

# ── Description generation ───────────────────────────────────────────────────
DESCRIPTION_MAX_TOKENS: int = int(os.getenv("DESCRIPTION_MAX_TOKENS", "500"))
COMMUNICATION_SUMMARY_MAX_TOKENS: int = int(os.getenv("COMMUNICATION_SUMMARY_MAX_TOKENS", "220"))

# ── Internal proctoring suspicion baseline ────────────────────────────────────
# Hardcoded defaults for v1. Tune as the review team gathers calibration data.
PROCTORING_BASELINE: dict[str, float] = {
    "speech_rate_wpm": 110.0,
    "filler_word_ratio": 0.02,
    "lexical_mattr": 0.84,
    "lexical_rare_word_ratio": 0.18,
    "discourse_connectors": 6.0,
    "discourse_tier1": 3.0,
    "sbert_coherence": 0.78,
    "avg_sentence_length": 22.0,
    "narrative_arc_score": 0.80,
    "preparedness_score": 55.0,
    "session_flag_ratio": 0.60,
    "session_uniform_avg": 0.65,
    "session_uniform_std": 0.20,
    "session_strong_ratio": 0.45,
}

# ── Academic violation framework (Track A + Track C) ─────────────────────────
ACADEMIC_VIOLATION: dict[str, float] = {
    "track_a_min_answers": 4.0,
    "track_a_z_threshold": 1.35,
    "track_a_score_threshold": 4.0,
    "track_c_flatness_threshold": 0.18,
    "track_c_pace_var_threshold": 0.18,
    "track_c_pause_regularity_threshold": 0.20,
    "track_c_pitch_flatness_threshold": 0.12,
    "track_c_question_threshold": 35.0,
}
