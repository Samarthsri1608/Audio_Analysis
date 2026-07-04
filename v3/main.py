"""
main.py — V3 FastAPI application.

Endpoints:
  GET  /health                                    → liveness check
  GET  /v3/analyse/{response_id}/personality      → System B: style / archetype / personality eval
  GET  /v3/analyse/{response_id}/communication    → System A: skills scoring / communication eval
  DELETE /v3/analyse/{response_id}/cache          → evict cached raw features for a response_id

Architecture:
  Feature extraction (audio → transcribe → aggregate) is shared and cached in
  _FEATURES_CACHE as a lightweight FeatureCacheEntry (raw features only).
  System A and System B evaluations are computed on-demand per endpoint call
  using the rule-based engines — no full result objects are stored in cache.
  This reduces cache size and avoids running both evaluation systems when only
  one is needed.

V3 changes vs V2:
  - Transcription: AssemblyAI pre-recorded API (disfluencies=True) replaces Whisper.
  - Real per-word confidence scores — no flat 0.75 proxy, no +0.06 offset.
  - Filler words: multi-word phrases ('i mean', 'you know') now counted correctly.
  - Temp dir prefix: v3_pipeline_ (vs v2_pipeline_).
  - Routes: /v3/ prefix (vs /v2/).
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import tempfile
from statistics import median as stats_median
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent  # Audio Analysis/
load_dotenv(ROOT / ".env", override=False)

# Ensure v3 package is importable when run from project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("v3.main")

from v3.config import MAX_CONCURRENT_JOBS                              # noqa: E402
from v3.models import (                                                # noqa: E402
    AnalyzeRequest,
    CommunicationResult,
    CommunicationSummary,
    FeatureCacheEntry,
    PersonalityResult,
    RawFeatures,
)
from v3.pipeline.communication_summary import generate_communication_summary  # noqa: E402
from v3.pipeline.audio_fetcher import fetch_and_prepare_audio          # noqa: E402
from v3.pipeline.transcriber import transcribe                         # noqa: E402
from v3.pipeline.text_features import extract_all_text_features        # noqa: E402
from v3.pipeline.vocal_features import extract_all_vocal_features      # noqa: E402
from v3.pipeline.normalizer import normalize, aggregate_signals        # noqa: E402
from v3.pipeline.archetype import classify, dominant, build_style_profile  # noqa: E402
from v3.pipeline.description import generate_description_from_style_profile  # noqa: E402
from v3.pipeline.skills_scorer import score as score_skills            # noqa: E402
from v3.pipeline.violation_detector import score_interview              # noqa: E402

# ── in-memory features cache ──────────────────────────────────────────────────
# Stores only raw features + metadata per response_id.
# Evaluation (System A / System B) is computed on demand per endpoint.
_FEATURES_CACHE: dict[str, FeatureCacheEntry] = {}
_INTERNAL_PROCTORING_TOKEN = os.getenv("INTERNAL_PROCTORING_TOKEN", "")

# ── thread pool for CPU-bound librosa work ────────────────────────────────────
_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("THREAD_POOL_WORKERS", "4")))

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Audio Analysis V3",
    version="3.0.0",
    description=(
        "Communication & Personality Evaluation Pipeline (AssemblyAI transcription) — "
        "raw features cached → System A (communication) or System B (personality) on demand"
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── helpers ───────────────────────────────────────────────────────────────────

async def _run_in_thread(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_EXECUTOR, lambda: fn(*args, **kwargs))


async def _process_single_question(wav_path: str, q_no: int) -> dict:
    """
    Transcribe a single WAV file, then extract its text and vocal features.

    V3: Uses AssemblyAI pre-recorded API with disfluencies=True.
    mean_confidence is the true mean per-word AssemblyAI confidence — no
    +0.06 correction offset is applied (that was Whisper-specific).
    """
    logger.info("Starting transcription & feature extraction for Q%d: %s", q_no, wav_path)

    # 1. Transcribe via AssemblyAI
    transcript_result = await transcribe(wav_path)
    text = transcript_result["text"]
    word_timestamps = transcript_result["word_timestamps"]
    duration_s = transcript_result["duration_seconds"]
    mean_confidence = transcript_result["mean_confidence"]

    # 2. Extract text + vocal features in parallel threads
    loop = asyncio.get_event_loop()
    text_future = loop.run_in_executor(
        _EXECUTOR,
        lambda: extract_all_text_features(text, duration_s, word_timestamps),
    )
    vocal_future = loop.run_in_executor(
        _EXECUTOR,
        lambda: extract_all_vocal_features(wav_path, word_timestamps, duration_s),
    )

    text_feats, vocal_feats = await asyncio.gather(text_future, vocal_future)

    logger.info("Finished feature extraction for Q%d", q_no)
    return {
        "q_no": q_no,
        "text": text,
        "word_timestamps": word_timestamps,
        "duration_s": duration_s,
        "mean_confidence": mean_confidence,
        "text_feats": text_feats,
        "vocal_feats": vocal_feats,
    }


def _median(values: list[float]) -> float:
    """Simple median helper retained for legacy use inside this module."""
    return float(stats_median(values)) if values else 0.0


# ── core feature extraction pipeline ─────────────────────────────────────────

async def _extract_and_cache_features(response_id: str) -> FeatureCacheEntry:
    """
    Run the shared feature extraction pipeline for a response_id and store
    the result in _FEATURES_CACHE.

    Steps:
      1. Download recordings and convert each to WAV.
      2. Transcribe and extract features per question in parallel.
      3. Aggregate and average features across all questions.
      4. Store a FeatureCacheEntry (raw features + transcript + duration).

    Returns the cached FeatureCacheEntry.
    """
    temp_dir = tempfile.mkdtemp(prefix="v3_pipeline_")
    try:
        # ── Step 1: Audio ──────────────────────────────────────────────────
        wav_paths = await fetch_and_prepare_audio(response_id, temp_dir)

        # ── Step 2: Parallel transcription and feature extraction ──────────
        sem = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

        async def sem_process(wav_path, q_no):
            async with sem:
                return await _process_single_question(wav_path, q_no)

        tasks = [
            sem_process(wav_path, q_no)
            for q_no, wav_path in enumerate(wav_paths, start=1)
        ]
        question_results = await asyncio.gather(*tasks)

        # ── Step 3: Aggregate question features ───────────────────────────
        num_q = len(question_results)
        question_results.sort(key=lambda x: x["q_no"])

        transcript = " ".join(q["text"] for q in question_results if q["text"])
        total_duration_s = sum(q["duration_s"] for q in question_results)
        duration_ms = total_duration_s * 1000.0

        sum_total_words = sum(q["text_feats"]["total_words"] for q in question_results)

        # Discourse features computed on the combined transcript for global accuracy
        from v3.pipeline.text_features import compute_discourse_features
        discourse = compute_discourse_features(transcript)
        sum_connectors = float(discourse["discourse_connectors"])
        sum_tier1 = float(discourse["discourse_tier1"])

        # Accumulate per-question averages
        avg_feats: dict[str, float] = {
            "speech_rate_wpm": 0.0,
            "speech_rate_variability": 0.0,
            "filler_word_ratio": 0.0,
            "lexical_mattr": 0.0,
            "lexical_rare_word_ratio": 0.0,
            "sbert_coherence": 0.0,
            "ner_entity_density": 0.0,
            "metric_density": 0.0,
            "narrative_arc_score": 0.0,
            "collaborative_language_ratio": 0.0,
            "question_density": 0.0,
            "empathetic_language_score": 0.0,
            "avg_sentence_length": 0.0,
            "pitch_variation": 0.0,
            "vocal_confidence": 0.0,
            "speech_fluency": 0.0,
            "stress_markers": 0.0,
            "fluency_pause_dur": 0.0,
            "fluency_pause_freq": 0.0,
            "voiced_fraction": 0.0,
            "intel_confidence": 0.0,
        }

        for q in question_results:
            t_f = q["text_feats"]
            v_f = q["vocal_feats"]

            avg_feats["speech_rate_wpm"]            += t_f.get("speech_rate_wpm", 0.0)
            avg_feats["speech_rate_variability"]    += t_f.get("speech_rate_variability", 0.0)
            avg_feats["filler_word_ratio"]          += t_f.get("filler_word_ratio", 0.0)
            avg_feats["lexical_mattr"]              += t_f.get("lexical_mattr", 0.0)
            avg_feats["lexical_rare_word_ratio"]    += t_f.get("lexical_rare_word_ratio", 0.0)
            avg_feats["sbert_coherence"]            += t_f.get("sbert_coherence", 0.65)
            avg_feats["ner_entity_density"]         += t_f.get("ner_entity_density", 0.0)
            avg_feats["metric_density"]             += t_f.get("metric_density", 0.0)
            avg_feats["narrative_arc_score"]        += t_f.get("narrative_arc_score", 0.0)
            avg_feats["collaborative_language_ratio"] += t_f.get("collaborative_language_ratio", 0.0)
            avg_feats["question_density"]           += t_f.get("question_density", 0.0)
            avg_feats["empathetic_language_score"]  += t_f.get("empathetic_language_score", 0.0)
            avg_feats["avg_sentence_length"]        += t_f.get("avg_sentence_length", 0.0)

            avg_feats["pitch_variation"]   += v_f.get("pitch_variation", 0.0)
            avg_feats["vocal_confidence"]  += v_f.get("vocal_confidence", 0.0)
            avg_feats["speech_fluency"]    += v_f.get("speech_fluency", 0.0)
            avg_feats["stress_markers"]    += v_f.get("stress_markers", 0.0)
            avg_feats["fluency_pause_dur"] += v_f.get("fluency_pause_dur", 0.0)
            avg_feats["fluency_pause_freq"] += v_f.get("fluency_pause_freq", 0.0)
            avg_feats["voiced_fraction"]   += v_f.get("voiced_fraction", 0.0)

            # V3: use AssemblyAI's real per-word confidence directly (no offset)
            avg_feats["intel_confidence"]  += q.get("mean_confidence", 0.80)

        for k in avg_feats:
            avg_feats[k] = round(avg_feats[k] / num_q, 4)

        # Count questions shorter than 1 minute (60 seconds)
        num_short_questions = sum(1 for q in question_results if q.get("duration_s", 0.0) < 60.0)
        is_short_duration = (num_short_questions / max(num_q, 1)) > 0.60

        raw = RawFeatures(
            # Summed / recalculated text stats
            total_words=float(sum_total_words),
            discourse_connectors=float(sum_connectors),
            discourse_tier1=float(sum_tier1),
            logical_connector_density=round(float(sum_connectors) / max(sum_total_words, 1), 4),
            vocabulary_density=avg_feats["lexical_mattr"],

            # Averaged text stats
            speech_rate_wpm=avg_feats["speech_rate_wpm"],
            speech_rate_variability=avg_feats["speech_rate_variability"],
            filler_word_ratio=avg_feats["filler_word_ratio"],
            lexical_mattr=avg_feats["lexical_mattr"],
            lexical_rare_word_ratio=avg_feats["lexical_rare_word_ratio"],
            sbert_coherence=avg_feats["sbert_coherence"],
            ner_entity_density=avg_feats["ner_entity_density"],
            metric_density=avg_feats["metric_density"],
            narrative_arc_score=avg_feats["narrative_arc_score"],
            collaborative_language_ratio=avg_feats["collaborative_language_ratio"],
            question_density=avg_feats["question_density"],
            empathetic_language_score=avg_feats["empathetic_language_score"],
            avg_sentence_length=avg_feats["avg_sentence_length"],

            # Averaged vocal stats
            pitch_variation=avg_feats["pitch_variation"],
            vocal_confidence=avg_feats["vocal_confidence"],
            speech_fluency=avg_feats["speech_fluency"],
            stress_markers=avg_feats["stress_markers"],
            fluency_pause_dur=avg_feats["fluency_pause_dur"],
            fluency_pause_freq=avg_feats["fluency_pause_freq"],
            voiced_fraction=avg_feats["voiced_fraction"],

            # V3: real AssemblyAI per-word confidence (no +0.06 offset)
            intel_confidence=avg_feats["intel_confidence"],
            is_short_duration=is_short_duration,
        )

        entry = FeatureCacheEntry(
            response_id=response_id,
            raw_features=raw,
            transcript=transcript,
            duration_ms=duration_ms,
            question_results=question_results,
        )

    except Exception as exc:
        logger.error("[%s] Feature extraction failed: %s", response_id, exc, exc_info=True)
        raise
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    _FEATURES_CACHE[response_id] = entry
    logger.info("[%s] Feature extraction complete — stored in cache", response_id)
    return entry


async def _get_or_extract_features(response_id: str) -> FeatureCacheEntry:
    """Return cached features if available, otherwise run extraction."""
    if response_id in _FEATURES_CACHE:
        logger.info("[%s] Feature cache hit", response_id)
        return _FEATURES_CACHE[response_id]
    return await _extract_and_cache_features(response_id)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _build_tags(raw: RawFeatures, skills) -> list[str]:
    tags: list[str] = []

    if skills.fluency.score >= 3.5:
        tags.append("steady-delivery")
    if skills.intelligibility.score >= 4.0:
        tags.append("clear-pronunciation")
    if skills.lexical_structural.score >= 3.5:
        tags.append("clear-explanation")
    if skills.narrative_evidence.score >= 3.5:
        tags.append("organized")
    if skills.vocal_delivery.score < 3.0:
        tags.append("flat-prosody")
    if raw.filler_word_ratio > 0.07 or skills.fluency.score < 3.0:
        tags.append("hesitant-flow")
    if raw.intel_confidence < 0.75:
        tags.append("asr-uncertain")
    if raw.total_words >= 180:
        tags.append("long-answer-drift")

    return _dedupe(tags)


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "up & running", "version": "3.1.2", "transcriber": "assemblyai"}


@app.get(
    "/v3/analyse/{response_id}/personality",
    response_model=PersonalityResult,
    summary="System B — Personality / Communication Style Evaluation",
    description=(
        "Extracts raw features (or uses cache) then runs System B: "
        "normalize → 5 signals → archetype blend → style profile. "
        "Non-evaluative — describes HOW the candidate communicates."
    ),
)
async def analyse_personality(
    response_id: str,
    include_description: bool = True,
    style_role: str = "default",
):
    """
    Return the personality / communication-style evaluation for a response.

    - Runs feature extraction if not already cached.
    - Runs System B (normalizer → archetype) on the cached raw features.
    - Optionally generates an LLM description of the style profile.
    """
    try:
        entry = await _get_or_extract_features(response_id)
    except Exception as exc:
        return PersonalityResult(
            response_id=response_id,
            status="error",
            error=str(exc),
        )

    raw = entry.raw_features

    # System B — normalize → signals → archetype blend → style profile
    normalized = normalize(raw)
    signals = aggregate_signals(normalized)
    blend = classify(signals, role=style_role)
    dom = dominant(blend)
    style_profile = build_style_profile(signals, blend, role=style_role)

    # Optional LLM description
    description = ""
    if include_description:
        description = await generate_description_from_style_profile(style_profile)

    logger.info(
        "[%s] Personality eval complete — dominant archetype: %s", response_id, dom
    )

    return PersonalityResult(
        response_id=response_id,
        status="success",
        duration_ms=entry.duration_ms,
        transcript=entry.transcript,
        style_profile=style_profile,
        signals=signals,
        archetype_blend=blend,
        dominant_archetype=dom,
        description=description,
    )


@app.get(
    "/v3/analyse/{response_id}/communication",
    response_model=CommunicationResult,
    summary="System A — Communication Skills Evaluation",
    description=(
        "Extracts raw features (or uses cache) then runs System A: "
        "5-axis skills scoring (Fluency, Intelligibility, Lexical/Structural, "
        "Narrative/Evidence, Vocal Delivery) → composite score."
    ),
)
async def analyse_communication(
    response_id: str,
    role_profile: str = "default",
):
    """
    Return the communication skills evaluation for a response.

    - Runs feature extraction if not already cached.
    - Runs System A (skills scorer) on the cached raw features.
    - Role profile controls axis weights: default / client_facing / technical / leadership.
    """
    try:
        entry = await _get_or_extract_features(response_id)
    except Exception as exc:
        return CommunicationResult(
            response_id=response_id,
            status="error",
            duration=0.0,
            result=CommunicationSummary(
                summary=[
                    "The analysis could not be completed because feature extraction failed.",
                    str(exc),
                ]
            ),
            tags=["error"],
        )

    raw = entry.raw_features

    # System A — 5-axis skills scoring
    skills = score_skills(raw, role_profile=role_profile)
    result = await generate_communication_summary(raw, skills)
    tags = _build_tags(raw, skills)

    logger.info(
        "[%s] Communication eval complete — composite: %.1f/100 (%s), review=%s",
        response_id, skills.composite_score, skills.composite_band, skills.review_required,
    )

    return CommunicationResult(
        response_id=response_id,
        status="success",
        duration=entry.duration_ms / 1000.0,
        result=result,
        tags=tags,
    )


@app.delete(
    "/v3/analyse/{response_id}/cache",
    summary="Evict cached features for a response",
    description="Removes the cached raw features so the next request re-runs extraction.",
)
async def evict_features_cache(response_id: str):
    """Evict a specific response_id from the features cache."""
    evicted = _FEATURES_CACHE.pop(response_id, None) is not None
    logger.info("[%s] Cache evicted=%s", response_id, evicted)
    return {"response_id": response_id, "evicted": evicted}


@app.get("/v3/analyse/{response_id}/raw")
async def get_raw_features(response_id: str):
    entry = await _get_or_extract_features(response_id)
    return entry.raw_features


@app.get(
    "/v3/internal/analyse/{response_id}/proctoring",
    summary="Internal proctor review",
    description=(
        "Question-level academic violation evidence for internal proctoring only. "
        "Output is an evidence payload per question, not a verdict. "
        "Non-evaluable questions (skipped, silent, low-confidence) are excluded from "
        "scoring and from the baseline/flatness series."
    ),
)
async def analyse_proctoring(
    response_id: str,
    internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
):
    if not _INTERNAL_PROCTORING_TOKEN:
        raise HTTPException(status_code=503, detail="Internal proctoring is not configured.")
    if internal_token != _INTERNAL_PROCTORING_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    try:
        entry = await _get_or_extract_features(response_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # score_interview handles sorting, eligibility gate, and off-by-one safety
    # internally — it will never produce a result for a q_no beyond what exists
    # in entry.question_results.
    evidence_payloads = await _run_in_thread(score_interview, entry.question_results)

    flagged_questions = [
        p["q_no"]
        for p in evidence_payloads
        if p.get("flagged_for_review") is True
    ]

    logger.info(
        "[%s] Proctoring complete — %d questions scored, %d flagged",
        response_id,
        sum(1 for p in evidence_payloads if p.get("evaluable")),
        len(flagged_questions),
    )

    return {
        "response_id": response_id,
        "flagged_questions": flagged_questions,
        "question_evidence": evidence_payloads,
        "schema_version": "v3",
    }


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the V3 audio analysis service.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "v3.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
