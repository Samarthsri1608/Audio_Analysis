"""
pipeline/transcriber.py — Transcribe per-question WAV files using
AssemblyAI's pre-recorded Speech-to-Text API (Universal model with disfluencies).

Key improvements over the v2 Whisper implementation:
  1. Real per-word confidence scores — AssemblyAI returns true per-token
     probabilities; no more flat 0.75 default that gave everyone perfect
     Intelligibility scores.
  2. Native disfluency capture — with disfluencies=True, AssemblyAI
     preserves 'um', 'uh', 'hmm' etc. verbatim in the transcript text,
     so compute_filler_rate in text_features.py sees them directly.
  3. No local GPU/CPU model required — fully API-driven via the SDK.
  4. No +0.06 accent correction needed — AssemblyAI's confidence is
     already well-calibrated across accents (unlike Whisper logprob).

AssemblyAI docs: https://www.assemblyai.com/docs
Auth: SDK uses the raw API key, no 'Bearer' prefix (Voice Agent API only).
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TypedDict

import assemblyai as aai

from v3.config import ASSEMBLYAI_API_KEY

logger = logging.getLogger("v3.transcriber")

# ── SDK initialisation ────────────────────────────────────────────────────────
# The SDK singleton is configured once at import time; it is thread-safe.
aai.settings.api_key = ASSEMBLYAI_API_KEY

# ── Domain vocabulary boost (Framework §1.2) ──────────────────────────────────
# Steer the model toward technical terms and Indian-English proper nouns common
# in backend-engineering / HR interview contexts.
_WORD_BOOST: list[str] = [
    "FastAPI", "MongoDB", "PostgreSQL", "Redis", "Kubernetes",
    "microservices", "async", "pydantic", "decorator",
    "dependency injection", "LangChain", "LangGraph",
    "Zeko", "BPCL", "IIT", "NIT", "BITS",
    "FastAPI", "uvicorn", "Celery", "RabbitMQ", "Kafka",
]

# ── Transcription config ──────────────────────────────────────────────────────
# speech_models                      → Universal-3 Pro & Universal-2 for best accuracy/coverage
# disfluencies=True                  → preserve 'um', 'uh', 'hmm' in output text
# punctuate + format_text            → clean, natural punctuation for downstream NLP
# word_boost                         → improve accuracy on domain-specific terms
_TRANSCRIPTION_CONFIG = aai.TranscriptionConfig(
    speech_models=["universal-3-pro", "universal-2"],
    disfluencies=True,
    punctuate=True,
    format_text=True,
    word_boost=_WORD_BOOST,
)


# ── Types (kept identical to v2 for downstream compatibility) ─────────────────
class WordTimestamp(TypedDict):
    word: str
    start: float   # seconds from start of audio
    end: float     # seconds from start of audio
    confidence: float  # real per-word AssemblyAI confidence (0–1)


class TranscriptResult(TypedDict):
    text: str
    word_timestamps: list[WordTimestamp]
    duration_seconds: float
    mean_confidence: float  # true mean of per-word confidence scores


# ── Public API ────────────────────────────────────────────────────────────────
async def transcribe(wav_path: str) -> TranscriptResult:
    """
    Transcribe a WAV file via the AssemblyAI pre-recorded API.

    The SDK handles upload → submit → polling internally.
    Returns a TranscriptResult compatible with the v3 downstream pipeline
    (text_features.py, vocal_features.py, skills_scorer.py).

    Args:
        wav_path: Absolute path to a 16 kHz mono WAV file.

    Returns:
        TranscriptResult with text, word-level timestamps (in seconds),
        audio duration, and the true mean per-word confidence score.

    Raises:
        RuntimeError: if AssemblyAI returns an error status.
    """
    # Run the blocking SDK call in a thread pool to keep the asyncio event loop free.
    return await asyncio.to_thread(_transcribe_sync, wav_path)


def _handle_silent_audio(path: Path) -> TranscriptResult:
    """Helper to compute duration of a silent audio file and return empty transcript."""
    import wave
    duration = 0.0
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
    except Exception as e:
        logger.warning("Could not read silent audio duration via wave: %s", e)
    
    return TranscriptResult(
        text="",
        word_timestamps=[],
        duration_seconds=duration,
        mean_confidence=0.80,
    )


# ── Synchronous implementation (runs in thread pool) ─────────────────────────
def _transcribe_sync(wav_path: str) -> TranscriptResult:
    path = Path(wav_path)
    logger.info(
        "[AssemblyAI] Transcribing: %s (%.1f MB) …",
        path.name,
        path.stat().st_size / 1_000_000,
    )

    transcriber = aai.Transcriber(config=_TRANSCRIPTION_CONFIG)
    
    try:
        transcript = transcriber.transcribe(str(path))
    except Exception as exc:
        err_msg = str(exc).lower()
        if "language_detection" in err_msg or "no spoken audio" in err_msg or "silence" in err_msg:
            logger.warning("[AssemblyAI] Silence/No spoken audio exception for %s: %s", path.name, exc)
            return _handle_silent_audio(path)
        raise

    if transcript.status == aai.TranscriptStatus.error:
        err_msg = str(transcript.error).lower()
        if "language_detection" in err_msg or "no spoken audio" in err_msg or "silence" in err_msg:
            logger.warning("[AssemblyAI] Silence/No spoken audio status for %s: %s", path.name, transcript.error)
            return _handle_silent_audio(path)
        raise RuntimeError(
            f"AssemblyAI transcription error for {path.name}: {transcript.error}"
        )

    text: str = transcript.text or ""

    # ── Word-level timestamps and confidence ──────────────────────────────────
    # AssemblyAI returns timestamps in milliseconds; we convert to seconds for
    # downstream compatibility with vocal_features.py pause detection.
    word_timestamps: list[WordTimestamp] = []
    all_confidences: list[float] = []

    words = transcript.words or []
    for w in words:
        # w.confidence is a float 0–1 (real per-word probability from the model)
        conf = float(w.confidence) if w.confidence is not None else 0.80
        word_timestamps.append(
            WordTimestamp(
                word=w.text,
                start=w.start / 1000.0,   # ms → seconds
                end=w.end / 1000.0,        # ms → seconds
                confidence=conf,
            )
        )
        all_confidences.append(conf)

    # ── Duration ──────────────────────────────────────────────────────────────
    # AssemblyAI audio_duration is returned in seconds as a float.
    duration = float(transcript.audio_duration or 0.0)

    # ── Mean confidence ───────────────────────────────────────────────────────
    # Use real per-word probabilities. No +0.06 bias-correction offset needed —
    # AssemblyAI's word-level confidence is already well-calibrated across
    # accents (unlike the Whisper segment avg_logprob proxy used in v2).
    mean_confidence = (
        float(sum(all_confidences) / len(all_confidences))
        if all_confidences
        else 0.80
    )

    logger.info(
        "[AssemblyAI] Done — %d words, %.1fs, mean_conf=%.3f",
        len(word_timestamps),
        duration,
        mean_confidence,
    )

    return TranscriptResult(
        text=text,
        word_timestamps=word_timestamps,
        duration_seconds=duration,
        mean_confidence=mean_confidence,
    )
