"""
asr_service.py
Local ASR using faster-whisper (CTranslate2 backend).

Replaces the previous Gemini-based transcription — zero API calls.

Model selection (via env var WHISPER_MODEL, default "small"):
  tiny   ~75 MB  – fastest, lower accuracy
  base   ~145 MB – good balance
  small  ~465 MB – recommended for production (default)
  medium ~1.5 GB – high accuracy, slower
  large  ~3.0 GB – best accuracy, slowest

Model files are downloaded once to ~/.cache/huggingface/hub/ (or
WHISPER_CACHE_DIR env var) and reused across runs.
In Docker, mount that directory as a volume so the image stays small.

Output format is identical to the previous Gemini-based service so
all downstream feature extractors remain unchanged.
"""

import logging
import os
from functools import lru_cache

from faster_whisper import WhisperModel

from app.shared_models import get_diarization_cache, get_file_id

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration  (override via environment variables)
# ──────────────────────────────────────────────────────────────────────────────
WHISPER_MODEL     = os.getenv("WHISPER_MODEL", "base")        # tiny/base/small/medium/large
WHISPER_DEVICE    = os.getenv("WHISPER_DEVICE", "cpu")         # "cpu" or "cuda"
WHISPER_COMPUTE   = os.getenv("WHISPER_COMPUTE", "int8")       # "int8" keeps RAM low on CPU
WHISPER_CACHE_DIR = os.getenv("WHISPER_CACHE_DIR", None)       # None → HF default cache
WHISPER_LANGUAGE  = os.getenv("WHISPER_LANGUAGE", "en")        # fix language → faster
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "2"))   # 2 = fast greedy; 5 = accurate
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "2")) # per-worker CPU threads


@lru_cache(maxsize=1)
def _get_whisper_model() -> WhisperModel:
    """
    Load the WhisperModel once and cache it in memory.
    lru_cache ensures a single instance is shared across all calls
    (avoids reloading ~500 MB from disk for every interview).
    """
    logger.info(
        f"Loading Whisper model '{WHISPER_MODEL}' "
        f"[device={WHISPER_DEVICE}, compute={WHISPER_COMPUTE}]..."
    )
    model = WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE,
        download_root=WHISPER_CACHE_DIR,
        cpu_threads=WHISPER_CPU_THREADS,   # limits per-instance thread consumption
        num_workers=1,                     # 1 = use single internal worker thread
    )
    logger.info("Whisper model loaded and cached.")
    return model


def transcribe_audio(file_path: str) -> dict:
    """
    Transcribe an audio file using faster-whisper (local, no API calls).

    Checks the in-memory diarization cache first (populated by speaker_filter
    in earlier pipeline stages).  On a cache miss, runs whisper transcription
    with word-level timestamps enabled.

    Returns a dict with the same shape as the previous Gemini-based service:
    {
        "text": "<full transcript>",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 4.2,
                "text": "Hello world",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.98},
                    {"word": "world", "start": 0.5, "end": 1.1, "probability": 0.97},
                ]
            },
            ...
        ]
    }
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # ── Cache check ────────────────────────────────────────────────────────────
    file_id = get_file_id(file_path)
    cache = get_diarization_cache()
    if file_id in cache:
        logger.info(f"ASR Cache HIT for file ID: {file_id}")
        return cache[file_id]

    logger.info(f"ASR Cache MISS — running faster-whisper on: {file_path}")

    # ── Transcribe ─────────────────────────────────────────────────────────────
    model = _get_whisper_model()

    segments_iter, info = model.transcribe(
        file_path,
        language=WHISPER_LANGUAGE,
        beam_size=WHISPER_BEAM_SIZE,
        word_timestamps=True,          # enables per-word start/end times
        vad_filter=True,               # skip silent regions automatically
        vad_parameters=dict(
            min_silence_duration_ms=500,   # skip pauses > 500ms (was 300ms)
            speech_pad_ms=200,
        ),
        condition_on_previous_text=False,  # prevents hallucination loops on long audio
        temperature=0.0,                   # greedy decode — fastest, deterministic
    )

    logger.info(
        f"Detected language '{info.language}' "
        f"(probability {info.language_probability:.2f}), "
        f"duration {info.duration:.1f}s"
    )

    reconstructed_segments = []
    full_text_parts = []

    for seg in segments_iter:   # generator — lazy evaluation
        text = (seg.text or "").strip()
        if not text:
            continue

        full_text_parts.append(text)

        # Word-level timestamps
        segment_words = []
        if seg.words:
            for w in seg.words:
                segment_words.append({
                    "word":        w.word.strip(),
                    "start":       round(w.start, 3),
                    "end":         round(w.end, 3),
                    "probability": round(w.probability, 4),
                })

        reconstructed_segments.append({
            "id":    len(reconstructed_segments),
            "start": round(seg.start, 3),
            "end":   round(seg.end, 3),
            "text":  text,
            "words": segment_words,
        })

    result = {
        "text":     " ".join(full_text_parts),
        "segments": reconstructed_segments,
    }

    logger.info(
        f"Transcription complete: {len(reconstructed_segments)} segments, "
        f"{len(full_text_parts)} non-empty."
    )
    return result
