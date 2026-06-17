"""
pipeline/transcriber.py — Call the Whisper API or run a local Whisper model
(via faster-whisper) depending on whether OPENAI_API_KEY is set.

Indian English accent priming is applied to both API and local paths per
Section 1.2 of the Zeko Unified Communication Framework v1.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from openai import AsyncOpenAI

from v2.config import OPENAI_API_KEY

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger("v2.transcriber")

# ── Indian English accent priming (Framework §1.2) ────────────────────────────
# This is an instrument correction, not a scoring adjustment. The goal is to
# make ASR equally accurate for Indian English speakers by biasing the
# decoder's language model toward en-IN phonetics and vocabulary.
_ACCENT_PROMPT = (
    "The speaker has an Indian accent. "
    "Transcription of technical interview response in Indian English."
)

# ── Clients / Models lazy singletons ──────────────────────────────────────────
_client: AsyncOpenAI | None = None
_local_model: "WhisperModel | None" = None
_model_lock = threading.Lock()


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _client


def _get_local_model() -> "WhisperModel":
    global _local_model
    if _local_model is None:
        with _model_lock:
            if _local_model is None:
                from faster_whisper import WhisperModel
                from v2.config import (
                    WHISPER_LOCAL_COMPUTE_TYPE,
                    WHISPER_LOCAL_CPU_THREADS,
                    WHISPER_LOCAL_DEVICE,
                    WHISPER_LOCAL_MODEL,
                    WHISPER_LOCAL_WORKERS,
                )

                logger.info(
                    "Loading local Whisper model '%s' (device=%s, compute_type=%s, cpu_threads=%d, workers=%d) ...",
                    WHISPER_LOCAL_MODEL,
                    WHISPER_LOCAL_DEVICE,
                    WHISPER_LOCAL_COMPUTE_TYPE,
                    WHISPER_LOCAL_CPU_THREADS,
                    WHISPER_LOCAL_WORKERS,
                )
                _local_model = WhisperModel(
                    WHISPER_LOCAL_MODEL,
                    device=WHISPER_LOCAL_DEVICE,
                    compute_type=WHISPER_LOCAL_COMPUTE_TYPE,
                    cpu_threads=WHISPER_LOCAL_CPU_THREADS,
                    num_workers=WHISPER_LOCAL_WORKERS,
                )
    return _local_model


# ── Types ─────────────────────────────────────────────────────────────────────
class WordTimestamp(TypedDict):
    word: str
    start: float
    end: float
    confidence: float  # per-token Whisper probability (0–1); used for intel_confidence


class TranscriptResult(TypedDict):
    text: str
    word_timestamps: list[WordTimestamp]
    duration_seconds: float
    mean_confidence: float   # average Whisper token confidence across all words


# ── Public API ────────────────────────────────────────────────────────────────
async def transcribe(wav_path: str) -> TranscriptResult:
    """
    Transcribe a WAV file. Uses Whisper API if OPENAI_API_KEY is set,
    otherwise falls back to a local Whisper model (via faster-whisper).

    Indian English accent priming is applied automatically (Framework §1.2).
    The returned `mean_confidence` already has the +0.06 offset applied so
    downstream scoring uses a bias-corrected value out of the box.

    Args:
        wav_path: Absolute path to a 16kHz mono WAV file.

    Returns:
        TranscriptResult with text, word-level timestamps, duration, and
        bias-corrected mean_confidence.
    """
    if OPENAI_API_KEY:
        return await _transcribe_api(wav_path)
    else:
        return await _transcribe_local(wav_path)


def _apply_accent_correction(raw_confidence: float) -> float:
    """
    Apply the +0.06 Indian English instrument correction to raw Whisper
    token confidence before scoring (Framework §2.2, Intelligibility axis).

    This corrects for ASR bias at the extraction layer so all downstream
    thresholds remain universal.
    """
    return min(raw_confidence + 0.06, 1.0)


# ── Implementations ───────────────────────────────────────────────────────────
async def _transcribe_api(wav_path: str) -> TranscriptResult:
    client = _get_client()
    path = Path(wav_path)

    logger.info("Transcribing via API: %s (%.1f MB) …", path.name, path.stat().st_size / 1e6)

    with open(wav_path, "rb") as audio_file:
        response = await client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["word"],
            # Indian English accent priming (Framework §1.2)
            prompt=_ACCENT_PROMPT,
        )

    text: str = response.text or ""
    duration: float = getattr(response, "duration", 0.0) or 0.0

    # Whisper API returns word timestamps as objects; normalise to dicts.
    # NOTE: The API verbose_json does not expose per-word probability. We use
    # the segment-level avg_logprob as a proxy confidence for API calls.
    raw_words = getattr(response, "words", None) or []
    word_timestamps: list[WordTimestamp] = [
        WordTimestamp(
            word=w.word if hasattr(w, "word") else w.get("word", ""),
            start=w.start if hasattr(w, "start") else w.get("start", 0.0),
            end=w.end if hasattr(w, "end") else w.get("end", 0.0),
            confidence=0.75,  # API doesn't expose word-level prob; use neutral default
        )
        for w in raw_words
    ]

    # Derive mean_confidence from segment avg_logprob when available
    segments = getattr(response, "segments", None) or []
    if segments:
        import math
        logprobs = [getattr(s, "avg_logprob", -0.5) for s in segments]
        raw_conf = float(sum(math.exp(lp) for lp in logprobs) / len(logprobs))
    else:
        raw_conf = 0.75  # neutral fallback

    mean_confidence = _apply_accent_correction(raw_conf)

    logger.info(
        "API transcription done — %d words, %.1fs, conf=%.3f (corrected)",
        len(word_timestamps), duration, mean_confidence,
    )

    return TranscriptResult(
        text=text,
        word_timestamps=word_timestamps,
        duration_seconds=duration,
        mean_confidence=mean_confidence,
    )


def _transcribe_local_sync(wav_path: str) -> TranscriptResult:
    model = _get_local_model()
    path = Path(wav_path)

    logger.info("Transcribing locally: %s (%.1f MB) …", path.name, path.stat().st_size / 1e6)

    # Run local Whisper with accent priming and word-level probabilities
    segments, info = model.transcribe(
        wav_path,
        word_timestamps=True,
        initial_prompt=_ACCENT_PROMPT,   # Indian English accent priming (Framework §1.2)
        language="en",
        temperature=0.0,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=200,
        ),
        condition_on_previous_text=False,
    )

    text_pieces = []
    word_timestamps: list[WordTimestamp] = []
    all_confidences: list[float] = []

    for segment in segments:
        text_pieces.append(segment.text)
        if segment.words:
            for w in segment.words:
                prob = getattr(w, "probability", 0.75)
                word_timestamps.append(
                    WordTimestamp(
                        word=w.word,
                        start=w.start,
                        end=w.end,
                        confidence=prob,
                    )
                )
                all_confidences.append(prob)

    full_text = "".join(text_pieces).strip()
    duration = info.duration

    raw_conf = float(sum(all_confidences) / len(all_confidences)) if all_confidences else 0.75
    mean_confidence = _apply_accent_correction(raw_conf)

    logger.info(
        "Local transcription done — %d words, %.1fs, conf=%.3f (corrected)",
        len(word_timestamps), duration, mean_confidence,
    )

    return TranscriptResult(
        text=full_text,
        word_timestamps=word_timestamps,
        duration_seconds=duration,
        mean_confidence=mean_confidence,
    )


async def _transcribe_local(wav_path: str) -> TranscriptResult:
    # Run the CPU/GPU-bound synchronous transcription in a background thread to prevent event loop block.
    return await asyncio.to_thread(_transcribe_local_sync, wav_path)
