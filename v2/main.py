"""
main.py — V2 FastAPI application.

Endpoints:
  GET  /health                   → liveness check
  POST /v2/analyze               → start analysis (body: {response_id, include_description, role_profile})
  GET  /v2/result/{response_id}  → poll for cached result
  GET  /v2/analyze/{response_id} → one-shot: run + return result (idempotent)
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent  # Audio Analysis/
load_dotenv(ROOT / ".env", override=False)

# Ensure v2 package is importable when run from project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("v2.main")

from v2.config import MAX_CONCURRENT_JOBS                       # noqa: E402
from v2.models import AnalysisResult, AnalyzeRequest, RawFeatures  # noqa: E402
from v2.pipeline.audio_fetcher import fetch_and_prepare_audio       # noqa: E402
from v2.pipeline.transcriber import transcribe                       # noqa: E402
from v2.pipeline.text_features import extract_all_text_features     # noqa: E402
from v2.pipeline.vocal_features import extract_all_vocal_features   # noqa: E402
from v2.pipeline.normalizer import normalize, aggregate_signals      # noqa: E402
from v2.pipeline.archetype import classify, dominant, build_style_profile  # noqa: E402
from v2.pipeline.description import generate_description_from_style_profile  # noqa: E402
from v2.pipeline.skills_scorer import score as score_skills         # noqa: E402

# ── in-memory cache (swap for Redis in production) ───────────────────────────
_CACHE: dict[str, AnalysisResult] = {}

# ── thread pool for CPU-bound librosa work ────────────────────────────────────
_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("THREAD_POOL_WORKERS", "4")))

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Audio Analysis V2",
    version="2.0.0",
    description=(
        "Communication Style Evaluation Pipeline — "
        "13 features → 5 signals → archetype blend"
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── core pipeline ─────────────────────────────────────────────────────────────

async def _run_in_thread(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_EXECUTOR, lambda: fn(*args, **kwargs))


async def process_single_question_features(wav_path: str, q_no: int) -> dict:
    """
    Transcribe a single WAV file, then extract its text and vocal features.
    """
    logger.info("Starting transcription & feature extraction for Q%d: %s", q_no, wav_path)
    
    # 1. Transcribe
    transcript_result = await transcribe(wav_path)
    text = transcript_result["text"]
    word_timestamps = transcript_result["word_timestamps"]
    duration_s = transcript_result["duration_seconds"]
    mean_confidence = transcript_result["mean_confidence"]

    # 2. Extract features in parallel in threads
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

    logger.info("Finished transcription & feature extraction for Q%d", q_no)

    return {
        "q_no": q_no,
        "text": text,
        "word_timestamps": word_timestamps,
        "duration_s": duration_s,
        "mean_confidence": mean_confidence,
        "text_feats": text_feats,
        "vocal_feats": vocal_feats,
    }


async def run_pipeline(
    response_id: str,
    include_description: bool = True,
    role_profile: str = "default",
    style_role: str = "default",
) -> AnalysisResult:
    """
    Full pipeline:
      1. Download question recordings in parallel and convert each to WAV.
      2. Transcribe and extract features for each question in parallel.
      3. Aggregate and average features across questions.
      4. System A: Skills scoring (5-axis, band scored)
      5. System B: Normalize → signals → archetype blend (sync, fast)
      6. LLM description (async, optional)
    """
    temp_dir = tempfile.mkdtemp(prefix="v2_pipeline_")
    try:
        # ── Step 1: Audio ──────────────────────────────────────────────────
        # Returns a list of WAV paths for each question
        wav_paths = await fetch_and_prepare_audio(response_id, temp_dir)

        # ── Step 2: Parallel transcription and feature extraction ──────────
        # Cap concurrent question processing
        sem = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

        async def sem_process(wav_path, q_no):
            async with sem:
                return await process_single_question_features(wav_path, q_no)

        tasks = [
            sem_process(wav_path, q_no)
            for q_no, wav_path in enumerate(wav_paths, start=1)
        ]

        question_results = await asyncio.gather(*tasks)

        # ── Step 3: Aggregate Question Features ─────────────────────────────
        num_q = len(question_results)
        
        # Sort results by question number to ensure sequential transcript assembly
        question_results.sort(key=lambda x: x["q_no"])
        
        transcript = " ".join(q["text"] for q in question_results if q["text"])
        total_duration_s = sum(q["duration_s"] for q in question_results)
        duration_ms = total_duration_s * 1000.0

        sum_total_words = sum(q["text_feats"]["total_words"] for q in question_results)

        # For unique connectors, we compute them on the combined transcript to match the global target thresholds
        from v2.pipeline.text_features import compute_discourse_features
        discourse = compute_discourse_features(transcript)
        sum_connectors = float(discourse["discourse_connectors"])
        sum_tier1 = float(discourse["discourse_tier1"])

        # Collect averaged metrics
        avg_feats = {
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

            avg_feats["speech_rate_wpm"] += t_f.get("speech_rate_wpm", 0.0)
            avg_feats["speech_rate_variability"] += t_f.get("speech_rate_variability", 0.0)
            avg_feats["filler_word_ratio"] += t_f.get("filler_word_ratio", 0.0)
            avg_feats["lexical_mattr"] += t_f.get("lexical_mattr", 0.0)
            avg_feats["lexical_rare_word_ratio"] += t_f.get("lexical_rare_word_ratio", 0.0)
            avg_feats["sbert_coherence"] += t_f.get("sbert_coherence", 0.65)
            avg_feats["ner_entity_density"] += t_f.get("ner_entity_density", 0.0)
            avg_feats["metric_density"] += t_f.get("metric_density", 0.0)
            avg_feats["narrative_arc_score"] += t_f.get("narrative_arc_score", 0.0)
            avg_feats["collaborative_language_ratio"] += t_f.get("collaborative_language_ratio", 0.0)
            avg_feats["question_density"] += t_f.get("question_density", 0.0)
            avg_feats["empathetic_language_score"] += t_f.get("empathetic_language_score", 0.0)
            avg_feats["avg_sentence_length"] += t_f.get("avg_sentence_length", 0.0)

            avg_feats["pitch_variation"] += v_f.get("pitch_variation", 0.0)
            avg_feats["vocal_confidence"] += v_f.get("vocal_confidence", 0.0)
            avg_feats["speech_fluency"] += v_f.get("speech_fluency", 0.0)
            avg_feats["stress_markers"] += v_f.get("stress_markers", 0.0)
            avg_feats["fluency_pause_dur"] += v_f.get("fluency_pause_dur", 0.0)
            avg_feats["fluency_pause_freq"] += v_f.get("fluency_pause_freq", 0.0)
            avg_feats["voiced_fraction"] += v_f.get("voiced_fraction", 0.0)

            avg_feats["intel_confidence"] += q.get("mean_confidence", 0.75)

        for k in avg_feats:
            avg_feats[k] /= num_q
            avg_feats[k] = round(avg_feats[k], 4)

        # Assemble RawFeatures
        raw = RawFeatures(
            # Summed / Recalculated text stats
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

            # Averaged ASR confidence
            intel_confidence=avg_feats["intel_confidence"],
        )

        # ── Step 4: System A — Skills scoring ────────────────────────────
        skills = score_skills(raw, role_profile=role_profile)
        logger.info(
            "[%s] Skills score: %.1f/100 (%s), review=%s",
            response_id, skills.composite_score, skills.composite_band,
            skills.review_required,
        )

        # ── Step 5: System B — Normalize + signals + archetype + style profile ──
        normalized = normalize(raw)
        signals = aggregate_signals(normalized)
        blend = classify(signals, role=style_role)
        dom = dominant(blend)
        style_profile = build_style_profile(signals, blend, role=style_role)

        # ── Step 6: LLM description (optional) ─────────────────────────
        description = ""
        if include_description:
            description = await generate_description_from_style_profile(style_profile)

        result = AnalysisResult(
            response_id=response_id,
            status="success",
            duration_ms=duration_ms,
            transcript=transcript,
            raw_features=raw,
            skills=skills,
            style_profile=style_profile,
            # Legacy flat fields (backward compatibility)
            signals=signals,
            archetype_blend=blend,
            dominant_archetype=dom,
            description=description,
        )

    except Exception as exc:
        logger.error("[%s] Pipeline failed: %s", response_id, exc, exc_info=True)
        result = AnalysisResult(
            response_id=response_id,
            status="error",
            error=str(exc),
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    logger.info("[%s] Pipeline execution finished with status: %s", response_id, result.status)
    _CACHE[response_id] = result
    return result


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.1.0"}


@app.post("/v2/analyze")
async def analyze_post(req: AnalyzeRequest):
    """
    Idempotent: returns cached result instantly if already processed.
    Otherwise runs the full pipeline and caches the result.
    """
    if req.response_id in _CACHE:
        logger.info("[%s] Cache hit — returning cached result", req.response_id)
        return _CACHE[req.response_id]

    res = await run_pipeline(
        req.response_id, req.include_description, req.role_profile, req.style_role
    )
    logger.info("[%s] Returning POST /v2/analyze response to client", req.response_id)
    return res


@app.get("/v2/analyze/{response_id}")
async def analyze_get(response_id: str, include_description: bool = True):
    """
    GET convenience endpoint — same semantics as POST.
    Idempotent: cached results are returned immediately.
    """
    if response_id in _CACHE:
        logger.info("[%s] Cache hit — returning cached result", response_id)
        return _CACHE[response_id]
    
    res = await run_pipeline(response_id, include_description)
    logger.info("[%s] Returning GET /v2/analyze/%s response to client", response_id, response_id)
    return res


@app.get("/v2/result/{response_id}")
async def get_cached_result(response_id: str):
    """Return a previously computed result. 404 if not yet computed."""
    if response_id not in _CACHE:
        raise HTTPException(
            status_code=404,
            detail=f"No result found for response_id={response_id!r}. "
                   "Call POST /v2/analyze first.",
        )
    return _CACHE[response_id]


@app.delete("/v2/result/{response_id}")
async def evict_cache(response_id: str):
    """Remove a cached result so the pipeline will re-run on next request."""
    _CACHE.pop(response_id, None)
    return {"evicted": response_id}


@app.get("/v2/cache/keys")
async def list_cache():
    """List all currently cached response IDs."""
    return {"cached_ids": list(_CACHE.keys()), "count": len(_CACHE)}


@app.post("/v2/cache/clear")
async def clear_cache_post():
    """Clear all cached response results."""
    count = len(_CACHE)
    _CACHE.clear()
    logger.info("Cache cleared via POST. Removed %d entries.", count)
    return {"status": "success", "message": "Cache cleared successfully", "count": count}


@app.delete("/v2/cache/clear/{response_id}")
async def clear_cache_delete(response_id: str):
    """Clear a specific cached response result."""
    if response_id in _CACHE:
        _CACHE.pop(response_id, None)
        logger.info("Cache entry cleared for response_id=%s.", response_id)
    return {"status": "success", "message": f"Cache entry for {response_id} cleared."}


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the V2 audio analysis service.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "v2.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
