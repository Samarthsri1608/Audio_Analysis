import os
import uuid
import json
import shutil
import time
import asyncio
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from fastapi import UploadFile, HTTPException

from app.settings import settings
from app.utils.type_convert import convert_video_to_audio
from app.services.speaker_filter import extract_interviewee_audio
from app.utils.audio_processor import preprocess_audio
from app.services.asr_service import transcribe_audio

from app.services.feature_extraction.fluency import extract_fluency_features
from app.services.feature_extraction.intelligibility import extract_intelligibility_features
from app.services.feature_extraction.language_control import extract_language_control_features
from app.services.feature_extraction.lexical_resource import extract_lexical_features
from app.services.feature_extraction.discourse import extract_discourse_features
from app.services.feature_extraction.segmentation import segment_transcript
from app.services.feature_extraction.voice_modulation import extract_voice_modulation_features
from app.services.feature_extraction.sentiment import extract_sentiment_features
from app.services.scoring_engine.predict_model import predict_scores
from app.services.report_generator import generate_report

logger = logging.getLogger(__name__)

# Ensure temp and cache directories exist
os.makedirs(settings.TEMP_DIR, exist_ok=True)
os.makedirs(settings.CACHE_DIR, exist_ok=True)


async def _run_in_executor(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


def _cache_path(response_id: str) -> str:
    safe = hashlib.sha256(response_id.encode()).hexdigest()[:16]
    return os.path.join(settings.CACHE_DIR, f"{safe}.json")


def _cache_is_valid(data: dict) -> bool:
    if not isinstance(data, dict) or not data:
        return False

    features = data.get("features", {})
    scores   = data.get("scores", {})

    if not features or not scores:
        return False

    # Ensure scores has new keys
    required_score_keys = {
        "logical_cohesion", "delivery_fluency", "pronunciation_clarity",
        "vocal_dynamism", "collaborative_tone", "lexical_precision"
    }
    if not required_score_keys.issubset(scores.keys()):
        return False

    # Detect NaN or None in scores
    import math
    for k, v in scores.items():
        if v is None:
            return False
        if isinstance(v, float) and math.isnan(v):
            return False

    # Detect zeroed lexical_resource
    lr = features.get("lexical_resource", {})
    lr_is_zeroed = (
        lr.get("unique_words", 0) == 0 and
        lr.get("type_token_ratio", 0) == 0.0 and
        lr.get("mattr", 0) == 0.0
    )
    if lr_is_zeroed:
        return False

    # Detect zeroed discourse
    disc = features.get("discourse", {})
    disc_is_zeroed = (
        disc.get("connector_count", 0) == 0 and
        "tier1_count" not in disc
    )
    if disc_is_zeroed:
        return False

    return True


def cache_get(response_id: str) -> dict | None:
    path = _cache_path(response_id)
    if not os.path.exists(path):
        return None
    age_days = (time.time() - os.path.getmtime(path)) / 86400
    if age_days > settings.CACHE_TTL_DAYS:
        try:
            os.remove(path)
        except Exception:
            pass
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if not _cache_is_valid(data):
            logger.warning(
                f"Cache EVICT for response_id={response_id}: "
                "entry is a corrupt/legacy record with zeroed features. Re-evaluating."
            )
            try:
                os.remove(path)
            except Exception:
                pass
            return None
        logger.info(f"Cache HIT for response_id={response_id} (age={age_days:.1f}d)")
        return data
    except Exception:
        return None


def cache_put(response_id: str, report: dict) -> None:
    path = _cache_path(response_id)
    try:
        with open(path, "w") as f:
            json.dump(report, f)
        logger.info(f"Cache WRITE for response_id={response_id}")
    except Exception as e:
        logger.warning(f"Cache write failed for {response_id}: {e}")


async def fetch_recording_urls(response_id: str) -> list[str]:
    API_URL = (
        f"https://interview-api.zeko.ai/dashboard/api/v2/report/recordings/screen"
        f"?responseId={response_id}"
    )
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.get(API_URL)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Failed to fetch recordings for '{response_id}': {e}")

    data  = resp.json().get("data", resp.json())
    links = data.get("screenRecordingLinks", [])
    if not links:
        raise HTTPException(status_code=404, detail=f"No screen recording links found for responseId '{response_id}'.")
    return links


async def _get_content_length(client: httpx.AsyncClient, url: str) -> int:
    try:
        head = await client.head(url, timeout=10)
        return int(head.headers.get("Content-Length", 0))
    except Exception:
        return 0


async def pick_best_url(urls: list[str]) -> str:
    if len(urls) == 1:
        return urls[0]

    async with httpx.AsyncClient() as client:
        sizes = await asyncio.gather(*[_get_content_length(client, url) for url in urls])

    best_url, best_size = max(zip(urls, sizes), key=lambda x: x[1])
    for url, size in zip(urls, sizes):
        filename = url.split("?")[0].split("/")[-1]
        marker = " ← selected" if url == best_url else ""
        logger.info(f"  {filename}: {size:,} bytes{marker}")

    return best_url


async def run_evaluation(file: UploadFile, skip_diarization: bool = False) -> dict:
    is_video          = file.filename.lower().endswith((".mp4", ".webm", ".mov", ".avi", ".mkv"))
    temp_video_path   = None
    temp_file_path    = None
    processed_path    = None
    interviewee_path  = None

    try:
        if is_video:
            temp_video_path = os.path.join(settings.TEMP_DIR, f"{uuid.uuid4()}.mp4")
            temp_file_path  = os.path.join(settings.TEMP_DIR, f"{uuid.uuid4()}.wav")
            with open(temp_video_path, "wb") as buf:
                shutil.copyfileobj(file.file, buf)
            logger.info(f"Converting video → audio: {temp_video_path}")
            await convert_video_to_audio(temp_video_path, temp_file_path)
        else:
            ext            = os.path.splitext(file.filename)[1]
            temp_file_path = os.path.join(settings.TEMP_DIR, f"{uuid.uuid4()}{ext}")
            with open(temp_file_path, "wb") as buf:
                shutil.copyfileobj(file.file, buf)

        logger.info(f"Processing audio: {temp_file_path}")

        if skip_diarization:
            logger.info("Skipping speaker diarization (using full recording).")
            interviewee_path = temp_file_path
        else:
            # Step 1: Speaker diarization
            logger.info("Running speaker diarization...")
            interviewee_path = await _run_in_executor(extract_interviewee_audio, temp_file_path)

        # Step 2: Preprocess + transcribe
        prep_result          = await _run_in_executor(preprocess_audio, interviewee_path)
        processed_path       = prep_result["processed_file_path"]
        transcription_result = await _run_in_executor(transcribe_audio, processed_path)

        duration_ms = prep_result["duration_ms"]
        text        = transcription_result["text"]

        # Step 3: Parallel feature extraction in thread pool
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {
                pool.submit(extract_fluency_features,         transcription_result, duration_ms): "fluency",
                pool.submit(extract_intelligibility_features, transcription_result):              "intelligibility",
                pool.submit(extract_language_control_features, text):                             "language_control",
                pool.submit(extract_lexical_features,          text):                             "lexical_resource",
                pool.submit(extract_discourse_features,        text):                             "discourse",
                pool.submit(extract_voice_modulation_features, interviewee_path):                 "voice_modulation",
                pool.submit(extract_sentiment_features,        text):                             "sentiment",
                pool.submit(segment_transcript,                transcription_result):             "_segments",
            }
            feature_results: dict = {}
            for future in as_completed(futures):
                dim = futures[future]
                try:
                    feature_results[dim] = future.result()
                except Exception as e:
                    logger.error(f"Feature extraction failed for '{dim}': {e}")
                    feature_results[dim] = {}

        segments = feature_results.pop("_segments", [])

        features_dict = {
            **feature_results,
            "raw_text":    text,
            "audio_path":  interviewee_path,
            "duration_ms": float(duration_ms),
        }

        # Step 4: Score
        scores = predict_scores(features_dict)

        return {
            "filename":            file.filename,
            "duration_ms":         float(duration_ms),
            "preprocessing_flags": [str(f) for f in prep_result.get("flags", [])],
            "segments":            segments,
            "features":            features_dict,
            "scores":              scores,
        }

    finally:
        for path in [temp_video_path, temp_file_path, interviewee_path, processed_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
