"""
main.py — V4 Audio-Only Proctoring FastAPI application.

Endpoints:
  GET  /health
      → liveness check

  GET  /v4/internal/analyse/{response_id}/proctoring
      → Audio-only academic violation detection per question.
        Requires X-Internal-Token header.
        Compatible with TEST_Proctoring/run_proctoring_batch.py.

Architecture:
  Audio-only pipeline — no ASR, no transcription in the detection path.
  For each question:
    1. Download per-question recording via Zeko interview API.
    2. Convert to 16kHz mono WAV.
    3. Extract audio features (pitch, pace, pauses, energy, timbre, room fingerprint).
    4. Run Track A (self-baseline deviation — requires ≥3 evaluable answers).
    5. Run Track C (naturalness / mechanism rules — works from Q1).
    6. OR-gate corroboration → QuestionEvidencePayload.

  A transcript is NEVER a model input. Transcripts may only be fetched
  downstream by a human reviewer, and only after a flag is raised.

V4 vs V3:
  - No ASR step (no AssemblyAI call).
  - No text-based features (SBERT, grammar, lexical scores all removed).
  - Per-question feature extraction instead of interview-averaged aggregation.
  - Explicit evaluability gate per spec §8 (evaluable always True or False, never null).
  - InterviewBaseline rolling accumulator for Track C environment shift.
  - NaN-aware robust z-score (fixes the zero-masking bug class from v3).
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
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent  # Audio_Analysis/
load_dotenv(ROOT / ".env", override=False)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("v4_proctoring.main")

from v4_proctoring.config import (                              # noqa: E402
    INTERNAL_PROCTORING_TOKEN,
    MAX_CONCURRENT_JOBS,
    THREAD_POOL_WORKERS,
)
from v4_proctoring.models import (                              # noqa: E402
    AudioFeatures,
    EvaluabilityResult,
    ProctoringResponse,
    QuestionEvidencePayload,
)
from v4_proctoring.pipeline.audio_fetcher import fetch_and_prepare_audio  # noqa: E402
from v4_proctoring.pipeline.feature_extractor import extract_features     # noqa: E402
from v4_proctoring.pipeline.corroboration import build_interview_evidence  # noqa: E402

# ── thread pool for CPU-bound librosa work ────────────────────────────────────
_EXECUTOR = ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Audio Analysis V4 — Proctoring",
    version="4.0.0",
    description=(
        "Audio-only academic violation detection pipeline. "
        "No ASR — all signals derived from pitch, pace, pauses, energy, and timbre. "
        "Track A: self-baseline deviation (robust z-score). "
        "Track C: naturalness/mechanism rules (cold-start safe)."
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
    """Run a CPU-bound function in the thread pool without blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_EXECUTOR, lambda: fn(*args, **kwargs))


async def _extract_features_for_question(
    q_no: int,
    wav_path: str,
) -> dict:
    """
    Extract audio features for a single question in a thread pool worker.

    Returns a dict suitable for use in build_interview_evidence:
        {
            "q_no": int,
            "evaluability": EvaluabilityResult,
            "features": Optional[AudioFeatures],
        }
    """
    logger.info("Extracting features for Q%d: %s", q_no, wav_path)
    evaluability, features = await _run_in_thread(extract_features, wav_path)
    logger.info(
        "Q%d — evaluable=%s, reason=%s",
        q_no, evaluability.evaluable, evaluability.not_evaluable_reason,
    )
    return {
        "q_no": q_no,
        "evaluability": evaluability,
        "features": features,
    }


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "up & running",
        "version": "4.0.0",
        "pipeline": "audio-only",
        "asr": "none",
    }


@app.get(
    "/v4/internal/analyse/{response_id}/proctoring",
    response_model=ProctoringResponse,
    summary="V4 Audio-Only Internal Proctoring",
    description=(
        "Runs the audio-only academic violation detection pipeline for a response. "
        "No ASR is performed — all signals are derived from pitch, pace, pauses, "
        "energy, timbre, and room acoustics. "
        "Requires X-Internal-Token header. "
        "Compatible with the existing batch runner in TEST_Proctoring/."
    ),
)
async def analyse_proctoring(
    response_id: str,
    internal_token: Optional[str] = Header(default=None, alias="X-Internal-Token"),
):
    """
    Audio-only proctoring analysis for a single interview response.

    Pipeline:
      1. Authenticate via X-Internal-Token header.
      2. Fetch all per-question recordings from the Zeko API.
      3. Convert each to 16kHz mono WAV.
      4. Extract audio features per question (parallel, CPU-bound).
      5. Run Track A (self-baseline z-score across all evaluable answers).
      6. Run Track C (naturalness rules, cold-start safe, question by question).
      7. OR-gate corroboration → confidence levels → evidence payloads.
      8. Return structured ProctoringResponse.
    """
    # ── Auth ──────────────────────────────────────────────────────────────────
    if not INTERNAL_PROCTORING_TOKEN:
        raise HTTPException(status_code=503, detail="Internal proctoring is not configured.")
    if internal_token != INTERNAL_PROCTORING_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    logger.info("[%s] Starting V4 audio-only proctoring analysis", response_id)

    temp_dir = tempfile.mkdtemp(prefix="v4_proctoring_")
    try:
        # ── Step 1: Fetch and convert per-question recordings ─────────────────
        try:
            question_wav_list = await fetch_and_prepare_audio(response_id, temp_dir)
        except ValueError as exc:
            # No recordings found
            logger.warning("[%s] No recordings found: %s", response_id, exc)
            return ProctoringResponse(
                response_id=response_id,
                status="fail",
                error={"type": "not_found", "detail": str(exc)},
                flagged_questions=[],
                question_evidence=[],
                total_questions_evaluated=0,
                total_questions_flagged=0,
            )
        except Exception as exc:
            logger.error("[%s] Audio fetch error: %s", response_id, exc, exc_info=True)
            return ProctoringResponse(
                response_id=response_id,
                status="fail",
                error={"type": "fetch_error", "detail": str(exc)},
                flagged_questions=[],
                question_evidence=[],
            )

        # ── Step 2: Extract features per question (parallel) ──────────────────
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

        async def sem_extract(q_no: int, wav_path: str) -> dict:
            async with semaphore:
                return await _extract_features_for_question(q_no, wav_path)

        extract_tasks = [
            sem_extract(q_no, wav_path)
            for q_no, wav_path in question_wav_list
        ]
        question_data: list[dict] = await asyncio.gather(*extract_tasks)

        # Sort by question number to ensure correct order for Track A baseline
        question_data.sort(key=lambda d: d["q_no"])

        # ── Step 3: Corroboration (Track A + Track C + OR-gate) ───────────────
        evidence_payloads: list[QuestionEvidencePayload] = await _run_in_thread(
            build_interview_evidence, question_data
        )

        # ── Step 4: Summarise results ─────────────────────────────────────────
        flagged_questions = [
            p.q_no for p in evidence_payloads if p.flagged_for_review
        ]
        total_evaluated = sum(1 for p in evidence_payloads if p.evaluable)
        total_flagged = len(flagged_questions)

        logger.info(
            "[%s] V4 proctoring complete — %d questions evaluated, %d flagged",
            response_id, total_evaluated, total_flagged,
        )

        return ProctoringResponse(
            response_id=response_id,
            status="success",
            flagged_questions=flagged_questions,
            question_evidence=evidence_payloads,
            total_questions_evaluated=total_evaluated,
            total_questions_flagged=total_flagged,
        )

    except Exception as exc:
        logger.error(
            "[%s] Unexpected error during V4 proctoring: %s",
            response_id, exc, exc_info=True,
        )
        err_msg = str(exc)
        err_type = "request_error"
        if "not found" in err_msg.lower() or "no recordings" in err_msg.lower() or "404" in err_msg:
            err_type = "not_found"
        return ProctoringResponse(
            response_id=response_id,
            status="fail",
            error={"type": err_type, "detail": err_msg},
            flagged_questions=[],
            question_evidence=[],
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the V4 Audio-Only Proctoring service.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "v4_proctoring.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
