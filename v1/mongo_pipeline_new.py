#!/usr/bin/env python3
"""
mongo_pipeline_new.py — MongoDB audio pipeline with better error handling,
connection management, and Plotly spider chart output via FastAPI.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

# Ensure backend package imports work from the repository root.
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

load_dotenv(dotenv_path=ROOT_DIR / ".env", override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("audio_pipeline")

try:
    from app.utils.audio_processor import preprocess_audio
    from app.services.asr_service import transcribe_audio
    from app.services.feature_extraction.fluency import extract_fluency_features
    from app.services.feature_extraction.intelligibility import extract_intelligibility_features
    from app.services.feature_extraction.lexical_resource import extract_lexical_features
    from app.services.feature_extraction.discourse import extract_discourse_features
    from app.services.feature_extraction.voice_modulation import extract_voice_modulation_features
    from app.services.feature_extraction.sentiment import extract_sentiment_features
    from app.services.scoring_engine.predict_model import predict_scores
except ImportError as exc:
    logger.error(
        "Failed to import backend modules. Ensure the backend directory exists and dependencies are installed: %s",
        exc,
    )
    raise

MAX_DURATION_S = int(os.getenv("MAX_DURATION_MINUTES", "60")) * 60

app = FastAPI(
    title="Audio Recording Pipeline",
    version="1.0.0",
    description="Fetch interview question recordings, merge them, compute audio/transcription features, and render a Plotly spider chart.",
)

RESULT_CACHE: dict[str, dict[str, Any]] = {}


def _run_in_executor(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, lambda: fn(*args, **kwargs))


async def _probe_duration(file_path: str) -> float:
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            file_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        return float(stdout.decode().strip() or "0")
    except (ValueError, OSError):
        return 0.0


async def _download_file(url: str, dest_path: str, response_id: str, label: str) -> None:
    async with httpx.AsyncClient(timeout=180) as http:
        async with http.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest_path, "wb") as fh:
                async for chunk in resp.aiter_bytes(chunk_size=16_384):
                    fh.write(chunk)


async def _webm_to_wav(input_path: str, output_path: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-i",
        input_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        output_path,
        "-y",
        "-loglevel",
        "error",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg conversion failed (exit {proc.returncode}): "
            f"{stderr.decode().strip()}"
        )


def _combine_transcriptions(question_results: list[dict[str, Any]]) -> tuple[dict[str, Any], float]:
    all_texts: list[str] = []
    all_segments: list[dict[str, Any]] = []
    combined_duration_ms: float = 0.0
    time_offset_s: float = 0.0

    for question in question_results:
        transcription = question["transcription"]
        dur_ms = float(question.get("duration_ms", 0.0))
        all_texts.append(transcription.get("text", "").strip())

        for segment in transcription.get("segments", []):
            shifted = dict(segment)
            shifted["start"] = segment.get("start", 0.0) + time_offset_s
            shifted["end"] = segment.get("end", 0.0) + time_offset_s
            all_segments.append(shifted)

        time_offset_s += dur_ms / 1000.0
        combined_duration_ms += dur_ms

    return {
        "text": " ".join(t for t in all_texts if t),
        "segments": all_segments,
    }, combined_duration_ms


def mock_extract_language_control_features(text: str) -> dict[str, Any]:
    return {
        "error_count": 0,
        "grammar_error_count": 0,
        "errors": [],
    }



async def process_response(response_id: str) -> dict[str, Any]:
    import shutil
    temp_dir = tempfile.mkdtemp(prefix="mongo_pipeline_")
    downloaded_files: list[str] = []
    try:
        logger.info("Fetching question recordings from REST API for response_id=%s", response_id)

        # Run loop for up to 25 questions to fetch recordings
        for q_no in range(1, 26):
            url = f"https://interview-api.zeko.ai/dashboard/api/v2/report/recordings/question?responseId={response_id}&quesNo={q_no}"
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.get(url)
                    if resp.status_code != 200:
                        logger.info(f"Stop loop at Q{q_no}: API returned status {resp.status_code}")
                        break
                    resp_data = resp.json()
                    data = resp_data.get("data")
                    if not data or not isinstance(data, dict):
                        logger.info(f"Stop loop at Q{q_no}: No data dictionary in response")
                        break
                    recording_link = data.get("recordingLink")
                    if not recording_link:
                        logger.info(f"Stop loop at Q{q_no}: No recordingLink in data")
                        break

                ext = ".mp4"
                if "?" in recording_link:
                    url_path = recording_link.split("?")[0]
                else:
                    url_path = recording_link
                if url_path.endswith((".mp4", ".webm", ".wav", ".mp3", ".ogg")):
                    ext = os.path.splitext(url_path)[1]

                local_path = os.path.join(temp_dir, f"q_{q_no}{ext}")
                logger.info("Downloading recording for Q%d: %s", q_no, recording_link)

                async with httpx.AsyncClient(timeout=120) as client:
                    dl = await client.get(recording_link)
                    dl.raise_for_status()

                with open(local_path, "wb") as f:
                    f.write(dl.content)

                downloaded_files.append(local_path)
            except Exception as e:
                logger.info(f"Break loop for response_id {response_id} at Q{q_no}: {e}")
                break

        if not downloaded_files:
            raise ValueError(f"No recordings found for response_id {response_id}")

        # Concatenate downloaded video files
        if len(downloaded_files) == 1:
            merged_video_path = downloaded_files[0]
        else:
            first_ext = os.path.splitext(downloaded_files[0])[1] or ".mp4"
            merged_video_path = os.path.join(temp_dir, f"merged{first_ext}")
            list_file_path = os.path.join(temp_dir, "files.txt")
            with open(list_file_path, "w") as lf:
                for filepath in downloaded_files:
                    lf.write(f"file '{filepath}'\n")

            proc = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file_path,
                "-c", "copy",
                merged_video_path,
                "-y",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg video concat failed (exit {proc.returncode}): {stderr.decode()}")

        # Convert merged video to WAV audio
        wav_path = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")
        await _webm_to_wav(merged_video_path, wav_path)

        # Preprocess and transcribe
        prep_result = await _run_in_executor(preprocess_audio, wav_path)
        processed_path = prep_result["processed_file_path"]

        transcription = await _run_in_executor(transcribe_audio, processed_path)
        combined_text = transcription.get("text", "")
        combined_duration_ms = float(prep_result.get("duration_ms", 0.0))

        # Parallel feature extraction
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                executor.submit(extract_fluency_features, transcription, combined_duration_ms): "fluency",
                executor.submit(extract_intelligibility_features, transcription): "intelligibility",
                executor.submit(mock_extract_language_control_features, combined_text): "language_control",
                executor.submit(extract_lexical_features, combined_text): "lexical_resource",
                executor.submit(extract_discourse_features, combined_text): "discourse",
                executor.submit(extract_voice_modulation_features, processed_path): "voice_modulation",
                executor.submit(extract_sentiment_features, combined_text): "sentiment",
            }

            feature_results: dict[str, Any] = {}
            for future, name in futures.items():
                try:
                    feature_results[name] = future.result()
                except Exception as exc:
                    logger.warning("Feature extraction '%s' failed: %s", name, exc)
                    feature_results[name] = {}

        scoring_input = {
            **feature_results,
            "raw_text": combined_text,
            "duration_ms": combined_duration_ms,
        }
        scores: dict[str, float] = {}
        try:
            scoring_output = await _run_in_executor(predict_scores, scoring_input)
            if isinstance(scoring_output, dict):
                scores = {k: float(v) for k, v in scoring_output.items() if k != "raw_text"}
        except Exception as exc:
            logger.error("Score prediction failed for %s: %s", response_id, exc)
            scores = {}

        result = {
            "response_id": response_id,
            "status": "success",
            "scores": scores,
            "overall_score": float(scores.get("overall_score", 0.0)),
            "features": feature_results,
            "raw_text": combined_text,
            "duration_ms": combined_duration_ms,
        }
        RESULT_CACHE[response_id] = result
        return result
    except Exception as exc:
        logger.error("Processing failed for %s: %s", response_id, exc)
        return {
            "response_id": response_id,
            "status": "error",
            "error": str(exc),
        }
    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/response/{response_id}")
async def response_summary(response_id: str):
    if response_id in RESULT_CACHE:
        return RESULT_CACHE[response_id]

    result = await process_response(response_id)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("error", "Unknown error"))

    return result




if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the Mongo audio pipeline service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run("mongo_pipeline_new:app", host=args.host, port=args.port, reload=True)
