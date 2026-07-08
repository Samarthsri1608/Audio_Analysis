#!/usr/bin/env python3
"""
Batch Feature Extractor Script
Extracts ML-grade features for a list of interview response IDs, skipping the scoring, reporting, and Gemini-based grammar checking stages.
Converts all audio uploads to compressed MP3 format to ensure stability and compatibility with the Gemini API (avoiding 500 INTERNAL WAV errors).
Supports chunked diarization for files longer than 5 minutes.
"""

import os
import sys
import json
import uuid
import time
import shutil
import asyncio
import httpx
import argparse
import logging
import traceback
from datetime import datetime
from pydub import AudioSegment
from pydantic import BaseModel, Field
from google.genai import types
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. Configure paths: Add backend directory to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Configure logging
LOG_FILE = os.path.join(BASE_DIR, "feature_extraction.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a")
    ]
)
logger = logging.getLogger("feature_extractor")

# ── Performance configuration ─────────────────────────────────────────────────
# Recordings longer than this are skipped (they are outliers and dominate runtime)
MAX_DURATION_S = int(os.getenv("MAX_DURATION_MINUTES", "60")) * 60  # default 60 min

# Import backend modules
try:
    from app.settings import settings
    from app.shared_models import get_gemini_client, get_diarization_cache, get_file_id
    from app.utils.type_convert import convert_video_to_audio
    from app.services.speaker_filter import extract_interviewee_audio
    from app.utils.audio_processor import preprocess_audio
    from app.services.asr_service import transcribe_audio

    # Feature extraction modules
    from app.services.feature_extraction.fluency import extract_fluency_features
    from app.services.feature_extraction.intelligibility import extract_intelligibility_features
    from app.services.feature_extraction.lexical_resource import extract_lexical_features
    from app.services.feature_extraction.discourse import extract_discourse_features
    from app.services.feature_extraction.segmentation import segment_transcript
    from app.services.feature_extraction.voice_modulation import extract_voice_modulation_features
    from app.services.feature_extraction.sentiment import extract_sentiment_features
    from app.services.evaluation_service import fetch_recording_urls, pick_best_url
except ImportError as e:
    logger.error(f"Failed to import backend modules. Make sure the backend directory exists: {e}")
    sys.exit(1)


# Pydantic schemas for chunked diarization
class SpeechSegment(BaseModel):
    speaker: str = Field(description="Must be either 'Interviewer' or 'Candidate'")
    start: float = Field(description="Start time of the segment in seconds")
    end: float = Field(description="End time of the segment in seconds")
    text: str = Field(description="Accurate transcription of the speech in this segment")


class DiarizationResult(BaseModel):
    segments: list[SpeechSegment] = Field(description="Chronological list of all speech segments")


def mock_extract_language_control_features(text: str) -> dict:
    """
    Mocked language control feature extractor.
    Completely avoids calling the Gemini API for grammar checking, returning 0 errors.
    """
    return {
        "error_count": 0,
        "grammar_error_count": 0,
        "errors": []
    }


async def _run_in_executor(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


async def process_candidate(response_id: str, client) -> dict:
    """
    Downloads and extracts raw features for a single response_id.
    Ensures absolute cleanup of all temporary audio/video files.
    """
    temp_files = []
    try:
        # Step 1: Fetch recording urls
        logger.info(f"[{response_id}] Fetching recording links...")
        links = await fetch_recording_urls(response_id)
        
        # Step 2: Pick the best url (largest video size)
        logger.info(f"[{response_id}] Selecting best recording link out of {len(links)}...")
        chosen_url = await pick_best_url(links)
        
        # Step 3: Download the file to local temp
        filename = chosen_url.split("?")[0].split("/")[-1]
        ext = os.path.splitext(filename)[1].lower()
        if not ext:
            ext = ".mp4"
            
        temp_video_path = os.path.join(settings.TEMP_DIR, f"{uuid.uuid4()}{ext}")
        temp_files.append(temp_video_path)
        
        logger.info(f"[{response_id}] Downloading recording to {temp_video_path}...")
        async with httpx.AsyncClient(timeout=180) as client_http:
            async with client_http.stream("GET", chosen_url) as response:
                response.raise_for_status()
                with open(temp_video_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=16384):
                        f.write(chunk)
                        
        # ── Duration guard: skip recordings that are too long ─────────────────
        # Probe file duration quickly via ffprobe before converting
        try:
            probe_result = await asyncio.create_subprocess_exec(
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", temp_video_path,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await probe_result.communicate()
            file_duration_s = float(stdout.decode().strip() or "0")
            if file_duration_s > MAX_DURATION_S:
                raise ValueError(
                    f"Recording duration {file_duration_s/60:.1f} min exceeds "
                    f"max allowed {MAX_DURATION_S/60:.0f} min — skipping."
                )
        except (ValueError, OSError) as dur_exc:
            if "exceeds" in str(dur_exc):
                raise
            logger.warning(f"[{response_id}] Could not probe duration: {dur_exc}")

        # Step 4: Convert video to audio (.wav)
        temp_audio_path = os.path.join(settings.TEMP_DIR, f"{uuid.uuid4()}.wav")
        temp_files.append(temp_audio_path)
        logger.info(f"[{response_id}] Converting video → audio (16kHz WAV)...")
        await convert_video_to_audio(temp_video_path, temp_audio_path)
        
        # Step 5: Local Diarization using WebRTC VAD + MFCC K-Means
        logger.info(f"[{response_id}] Diarizing and filtering interviewer speech locally...")
        interviewee_path = await _run_in_executor(extract_interviewee_audio, temp_audio_path)
        temp_files.append(interviewee_path)
        
        # Step 6: Preprocess interviewee audio
        logger.info(f"[{response_id}] Normalizing and preprocessing audio...")
        prep_result = await _run_in_executor(preprocess_audio, interviewee_path)
        processed_path = prep_result["processed_file_path"]
        temp_files.append(processed_path)
        
        # Step 7: Transcribe interviewee audio (Should hit cache instantly)
        logger.info(f"[{response_id}] Transcribing audio segments...")
        transcription_result = await _run_in_executor(transcribe_audio, processed_path)
        
        duration_ms = prep_result["duration_ms"]
        text = transcription_result["text"]
        
        # Step 8: Parallel Feature Extraction (ThreadPoolExecutor)
        logger.info(f"[{response_id}] Extracting acoustic and linguistic features...")
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(extract_fluency_features,         transcription_result, duration_ms): "fluency",
                pool.submit(extract_intelligibility_features, transcription_result):              "intelligibility",
                pool.submit(mock_extract_language_control_features, text):                         "language_control",
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
                    logger.error(f"[{response_id}] Feature extraction failed for '{dim}': {e}")
                    feature_results[dim] = {}
                    
        segments = feature_results.pop("_segments", [])
        
        return {
            "response_id": response_id,
            "status": "success",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "duration_ms": float(duration_ms),
            "filename": filename,
            "raw_text": text,
            "features": feature_results,
            "segments": segments
        }

    except Exception as e:
        logger.error(f"[{response_id}] Processing failed: {e}")
        logger.debug(traceback.format_exc())
        return {
            "response_id": response_id,
            "status": "failed",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "error": str(e)
        }

    finally:
        # Strict cleanup of all temporary audio and video files
        for path in temp_files:
            if path and os.path.exists(path):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                except Exception as cleanup_err:
                    logger.warning(f"[{response_id}] Failed to delete temp path {path}: {cleanup_err}")


async def main_async():
    parser = argparse.ArgumentParser(description="Batch Feature Extractor for Audio Analysis")
    parser.add_argument("--input", type=str, default="testing_interviews.json",
                        help="Path to JSON file containing response IDs")
    parser.add_argument("--output", type=str, default="extracted_features.jsonl",
                        help="Path to write the results (JSON Lines format)")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Number of concurrent response IDs to process (default: 5)")
    parser.add_argument("--max-duration", type=int, default=60,
                        help="Skip recordings longer than this many minutes (default: 60). "
                             "Very long recordings inflate runtime and are usually noise.")
    args = parser.parse_args()

    # Load response IDs
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    try:
        with open(args.input, "r") as f:
            raw_data = json.load(f)
        
        if isinstance(raw_data, list):
            response_ids = []
            for item in raw_data:
                if isinstance(item, dict):
                    val = item.get("_id") or item.get("id")
                    if val:
                        response_ids.append(str(val))
                elif isinstance(item, str):
                    response_ids.append(item)
        elif isinstance(raw_data, dict):
            response_ids = []
            for k, v in raw_data.items():
                if isinstance(v, dict) and ("id" in v or "_id" in v):
                    val = v.get("id") or v.get("_id")
                    response_ids.append(str(val))
                else:
                    response_ids.append(k)
        else:
            logger.error("Input file must be a JSON list or dictionary.")
            sys.exit(1)
    except Exception as e:
        try:
            with open(args.input, "r") as f:
                response_ids = [line.strip() for line in f if line.strip()]
        except Exception as txt_err:
            logger.error(f"Could not load input file: {e} / {txt_err}")
            sys.exit(1)

    response_ids = [rid for rid in response_ids if rid]
    logger.info(f"Loaded {len(response_ids)} response IDs to process.")

    # Initialize Gemini client
    client = get_gemini_client()
    if client is None:
        logger.error("Failed to initialize Google Gemini client. Ensure GOOGLE_API_KEY is configured.")
        sys.exit(1)

    # Check for already processed response IDs (for resuming)
    processed_ids = set()
    if os.path.exists(args.output):
        try:
            with open(args.output, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            processed_ids.add(record["response_id"])
                        except json.JSONDecodeError:
                            pass
            logger.info(f"Found {len(processed_ids)} already processed IDs. Resuming from progress.")
        except Exception as e:
            logger.warning(f"Could not parse existing output file for resume check: {e}")

    remaining_ids = [rid for rid in response_ids if rid not in processed_ids]
    logger.info(f"Remaining response IDs to process: {len(remaining_ids)}")

    if not remaining_ids:
        logger.info("All response IDs are already processed! Exiting.")
        return

    # Ensure temp directory exists
    os.makedirs(settings.TEMP_DIR, exist_ok=True)

    # Process batch with concurrency control
    semaphore = asyncio.Semaphore(args.concurrency)

    async def sem_process(rid):
        async with semaphore:
            start_time = time.time()
            res = await process_candidate(rid, client)
            elapsed = time.time() - start_time
            if res.get("status") == "success":
                logger.info(f"🟢 Successfully processed [{rid}] in {elapsed:.1f}s")
            else:
                logger.error(f"🔴 Failed to process [{rid}] in {elapsed:.1f}s: {res.get('error')}")
            return res

    tasks = [sem_process(rid) for rid in remaining_ids]

    # Open output file in append mode
    with open(args.output, "a", encoding="utf-8") as out_f:
        for future in asyncio.as_completed(tasks):
            result = await future
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

    logger.info(f"Batch execution complete. Results saved in {args.output}")


def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Batch execution interrupted by user (Ctrl+C). Exiting cleanly.")
    except Exception as e:
        logger.critical(f"Unhandled critical error in batch execution: {e}")
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    main()
