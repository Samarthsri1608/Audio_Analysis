#!/usr/bin/env python3
"""
mongo_pipeline.py — Batch Feature Extraction from MongoDB
==========================================================
Standalone replacement for feature_extractor.py that sources audio recordings
directly from MongoDB instead of a custom API.

Key differences from feature_extractor.py:
  - Recordings are fetched from the `dynamic_ai_test_responses` MongoDB collection.
  - There is NO diarization — each recording is already candidate-only per-question audio.
  - Response IDs can be supplied as a comma-separated CLI string OR as a JSON file.
  - Multiple per-question audio clips are downloaded, transcribed individually, then
    combined into a single unified transcription before feature extraction & scoring.

Usage:
    uv run python mongo_pipeline.py --response-ids testing_interviews.json \\
        --output mongo_extracted.jsonl --concurrency 5

    uv run python mongo_pipeline.py \\
        --response-ids "6a1976d176ab6743bb8aa3ee,6b0abc123" \\
        --output out.jsonl
"""

from __future__ import annotations

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
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── 1. Path setup: must happen before any backend imports ─────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# ── 2. Load .env before settings are imported by backend ─────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"), override=False)
except ImportError:
    pass  # python-dotenv not installed; rely on shell environment

# ── 3. Logging ────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(BASE_DIR, "mongo_pipeline.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger("mongo_pipeline")

# ── 4. Performance / guard config ─────────────────────────────────────────────
# Recordings longer than this (per question) are flagged; the whole response is
# skipped if ANY individual clip exceeds the ceiling.
MAX_DURATION_S = int(os.getenv("MAX_DURATION_MINUTES", "60")) * 60  # default 60 min

# ── 5. Backend imports ────────────────────────────────────────────────────────
try:
    from app.settings import settings
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
        f"Failed to import backend modules — ensure the backend directory exists "
        f"and all dependencies are installed: {exc}"
    )
    sys.exit(1)

# ── 6. MongoDB import (optional at top-level; fail gracefully later) ──────────
try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
    from bson import ObjectId
    from bson.errors import InvalidId
    _PYMONGO_AVAILABLE = True

    # Setting up mongo client 
    try:
        CLIENT = MongoClient(os.getenv("MONGODB_CONNECTION_STRING"))
        DB = CLIENT[os.getenv("MONGODB_DB", "zeko")]
        collection = DB["dynamic_ai_test_responses"]
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        sys.exit(1)

except ImportError:
    _PYMONGO_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def mock_extract_language_control_features(text: str) -> dict:
    """
    Mocked language-control feature extractor.
    Skips the Gemini grammar-check API entirely (avoids 500 errors on batch runs).
    Returns a neutral zero-error structure that is compatible with predict_scores.
    """
    return {
        "error_count": 0,
        "grammar_error_count": 0,
        "errors": [],
    }


async def _run_in_executor(fn, *args, **kwargs):
    """Run a synchronous callable in the default ThreadPoolExecutor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))



def _fetch_recordings(response_id: str) -> list[dict] | None:
    """
    Attempt to fetch a recording of document by ObjectId.
    Returns the raw document dict or None if not found.
    """
    global collection
    recordings = collection.find_one({"_id": ObjectId(response_id)})['recordings']
    return recordings if recordings else None


def _flatten_audio_links(recordings_field) -> list[dict]:
    """
    Flatten the nested `recordings` array-of-arrays into a flat list of
    ``{"questionNo": int, "audioRecordingLink": str}`` dicts, preserving order
    and skipping any entry that lacks an `audioRecordingLink`.
    """
    flat: list[dict] = []
    if not isinstance(recordings_field, list):
        return flat
    for outer in recordings_field:
        if isinstance(outer, list):
            for entry in outer:
                if isinstance(entry, dict) and entry.get("audioRecordingLink"):
                    flat.append(entry)
        elif isinstance(outer, dict) and outer.get("audioRecordingLink"):
            # Handle the case where `recordings` is a flat array of objects
            flat.append(outer)
    return flat


async def _probe_duration(file_path: str) -> float:
    """
    Use ffprobe to get the duration (in seconds) of a media file.
    Returns 0.0 if ffprobe fails.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        return float(stdout.decode().strip() or "0")
    except (ValueError, OSError):
        return 0.0


async def _download_file(url: str, dest_path: str, response_id: str, label: str) -> None:
    """Stream-download `url` to `dest_path` using httpx (timeout 180 s)."""
    async with httpx.AsyncClient(timeout=180) as http:
        async with http.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest_path, "wb") as fh:
                async for chunk in resp.aiter_bytes(chunk_size=16_384):
                    fh.write(chunk)


async def _webm_to_wav(input_path: str, output_path: str) -> None:
    """
    Convert a WebM audio file to a 16 kHz mono PCM WAV using ffmpeg.
    Raises RuntimeError if ffmpeg exits non-zero.
    """
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path,
        "-y",
        "-loglevel", "error",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg conversion failed (exit {proc.returncode}): "
            f"{stderr.decode().strip()}"
        )


def _combine_transcriptions(
    question_results: list[dict],
) -> tuple[dict, float]:
    """
    Merge per-question transcription results into a single unified transcription
    dict suitable for passing to all feature extractors.

    Each element of `question_results` is:
        {
            "transcription": {"text": str, "segments": list[dict]},
            "duration_ms": float,
        }

    Segment timestamps are shifted by the cumulative duration of all preceding
    questions so that the combined timeline is monotonically increasing.

    Returns:
        combined_transcription  — {"text": str, "segments": [...]}
        combined_duration_ms    — sum of all question durations (float)
    """
    all_texts: list[str] = []
    all_segments: list[dict] = []
    combined_duration_ms: float = 0.0
    time_offset_s: float = 0.0

    for q in question_results:
        tr = q["transcription"]
        dur_ms = float(q.get("duration_ms", 0))

        all_texts.append(tr.get("text", "").strip())

        for seg in tr.get("segments", []):
            shifted = dict(seg)
            shifted["start"] = seg.get("start", 0.0) + time_offset_s
            shifted["end"] = seg.get("end", 0.0) + time_offset_s
            all_segments.append(shifted)

        # Advance the clock by this question's duration (convert ms → s)
        time_offset_s += dur_ms / 1000.0
        combined_duration_ms += dur_ms

    combined_transcription = {
        "text": " ".join(t for t in all_texts if t),
        "segments": all_segments,
    }
    return combined_transcription, combined_duration_ms


# ─────────────────────────────────────────────────────────────────────────────
# Core per-response processing
# ─────────────────────────────────────────────────────────────────────────────

async def process_response(
    response_id: str,
    collection,
    temp_base: str,
) -> dict:
    """
    Download, transcribe, and extract features for a single response_id.

    Steps
    -----
    1  Fetch document from MongoDB.
    2  Flatten and collect all audioRecordingLink values.
    3  For each question:
       a  Download the .webm file.
       b  Probe duration (skip if > MAX_DURATION_S).
       c  Convert webm → 16 kHz WAV.
       d  Preprocess WAV.
       e  Transcribe WAV.
    4  Combine all per-question transcriptions into one unified result.
    5  Run the full feature extraction pipeline in parallel threads.
    6  Run predict_scores on extracted features.
    7  Return the output record dict.
    """
    temp_files: list[str] = []

    try:
        # ── Step 1: Fetch MongoDB document ────────────────────────────────────
        logger.info(f"[{response_id}] Step 1/6: Fetching document from MongoDB...")
        doc = await _run_in_executor(_fetch_doc, collection, response_id)
        if doc is None:
            raise ValueError(f"Document not found in MongoDB for response_id={response_id!r}")

        # ── Step 2: Parse audio links ─────────────────────────────────────────
        logger.info(f"[{response_id}] Step 2/6: Parsing recordings field...")
        audio_entries = _flatten_audio_links(doc.get("recordings", []))
        if not audio_entries:
            raise ValueError("No valid audioRecordingLink entries found in document.")

        # Sort by questionNo for stable ordering
        audio_entries.sort(key=lambda e: e.get("questionNo", 0))
        logger.info(
            f"[{response_id}]   Found {len(audio_entries)} question recording(s): "
            f"Q{[e.get('questionNo') for e in audio_entries]}"
        )

        # ── Step 3: Download, convert, and transcribe each question ───────────
        logger.info(f"[{response_id}] Step 3/6: Processing per-question audio...")
        question_results: list[dict] = []

        for idx, entry in enumerate(audio_entries, start=1):
            q_no = entry.get("questionNo", idx)
            url = entry["audioRecordingLink"]

            # 3a — Download .webm
            webm_path = os.path.join(temp_base, f"{uuid.uuid4()}.webm")
            temp_files.append(webm_path)
            logger.info(f"[{response_id}]   Q{q_no} [{idx}/{len(audio_entries)}]: Downloading...")
            await _download_file(url, webm_path, response_id, f"Q{q_no}")

            # 3b — Probe duration
            duration_s = await _probe_duration(webm_path)
            if duration_s > MAX_DURATION_S:
                raise ValueError(
                    f"Q{q_no} audio duration {duration_s / 60:.1f} min exceeds "
                    f"maximum allowed {MAX_DURATION_S / 60:.0f} min — skipping response."
                )
            logger.info(f"[{response_id}]   Q{q_no}: Duration {duration_s:.1f}s — OK.")

            # 3c — Convert webm → WAV
            wav_path = os.path.join(temp_base, f"{uuid.uuid4()}.wav")
            temp_files.append(wav_path)
            logger.info(f"[{response_id}]   Q{q_no}: Converting webm → 16kHz WAV...")
            await _webm_to_wav(webm_path, wav_path)

            # 3d — Preprocess WAV
            logger.info(f"[{response_id}]   Q{q_no}: Preprocessing audio...")
            prep_result = await _run_in_executor(preprocess_audio, wav_path)
            processed_path = prep_result["processed_file_path"]
            temp_files.append(processed_path)
            duration_ms = float(prep_result.get("duration_ms", duration_s * 1000))

            # 3e — Transcribe
            logger.info(f"[{response_id}]   Q{q_no}: Transcribing audio...")
            transcription_result = await _run_in_executor(transcribe_audio, processed_path)

            question_results.append(
                {
                    "question_no": q_no,
                    "transcription": transcription_result,
                    "duration_ms": duration_ms,
                    "wav_path": processed_path,  # kept alive until cleanup
                }
            )

        # ── Step 4: Combine transcriptions ────────────────────────────────────
        logger.info(f"[{response_id}] Step 4/6: Combining transcriptions from {len(question_results)} question(s)...")
        combined_transcription, combined_duration_ms = _combine_transcriptions(question_results)
        combined_text = combined_transcription["text"]
        logger.info(
            f"[{response_id}]   Combined text length: {len(combined_text)} chars | "
            f"total duration: {combined_duration_ms / 1000:.1f}s"
        )

        if not combined_text.strip():
            logger.warning(f"[{response_id}]   Combined transcription is empty — features may be sparse.")

        # Use the processed wav path of the LAST question for voice modulation
        # (voice_modulation expects a single file path; the last question is representative
        #  since questions are independent candidate answers).
        # For a more rigorous approach this could be a concatenated WAV, but the
        # current backend API only accepts a file path.
        last_wav_path = question_results[-1]["wav_path"]

        # ── Step 5: Parallel feature extraction ───────────────────────────────
        logger.info(f"[{response_id}] Step 5/6: Running parallel feature extraction...")
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(
                    extract_fluency_features,
                    combined_transcription,
                    combined_duration_ms,
                ): "fluency",
                pool.submit(
                    extract_intelligibility_features,
                    combined_transcription,
                ): "intelligibility",
                pool.submit(
                    mock_extract_language_control_features,
                    combined_text,
                ): "language_control",
                pool.submit(
                    extract_lexical_features,
                    combined_text,
                ): "lexical_resource",
                pool.submit(
                    extract_discourse_features,
                    combined_text,
                ): "discourse",
                pool.submit(
                    extract_voice_modulation_features,
                    last_wav_path,
                ): "voice_modulation",
                pool.submit(
                    extract_sentiment_features,
                    combined_text,
                ): "sentiment",
            }

            feature_results: dict = {}
            for future in as_completed(futures):
                dim = futures[future]
                try:
                    feature_results[dim] = future.result()
                except Exception as feat_exc:
                    logger.error(
                        f"[{response_id}] Feature extraction failed for '{dim}': {feat_exc}"
                    )
                    feature_results[dim] = {}

        # ── Step 6: Scoring ───────────────────────────────────────────────────
        logger.info(f"[{response_id}] Step 6/6: Running scoring model...")
        scores: dict = {}
        overall_score: float = 0.0
        try:
            scoring_output = await _run_in_executor(predict_scores, feature_results)
            scores = scoring_output.get("scores", scoring_output)
            overall_score = float(scoring_output.get("overall_score", 0.0))
        except Exception as score_exc:
            logger.error(f"[{response_id}] Scoring failed: {score_exc}")

        return {
            "response_id": response_id,
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question_count": len(question_results),
            "duration_ms": combined_duration_ms,
            "raw_text": combined_text,
            "features": feature_results,
            "scores": scores,
            "overall_score": overall_score,
            "segments": combined_transcription.get("segments", []),
        }

    except Exception as exc:
        logger.error(f"[{response_id}] Processing failed: {exc}")
        logger.debug(traceback.format_exc())
        return {
            "response_id": response_id,
            "status": "error",
            "error": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    finally:
        # Always clean up ALL temporary files regardless of success/failure
        for path in temp_files:
            if path and os.path.exists(path):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                except Exception as cleanup_err:
                    logger.warning(
                        f"[{response_id}] Could not remove temp file {path!r}: {cleanup_err}"
                    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing & ID loading
# ─────────────────────────────────────────────────────────────────────────────

def _parse_response_ids(raw: str) -> list[str]:
    """
    Accept either:
      - A comma-separated string of IDs:   "id1,id2,id3"
      - A path to a JSON file containing one of:
          * A list of ID strings:                ["id1", "id2"]
          * A list of dicts with "_id" or "id":  [{"_id": "id1"}, ...]
          * {"response_ids": ["id1", "id2"]}
          * {"response_ids": {"id1": {...}, ...}}   (keys are IDs)

    Returns a deduplicated list of non-empty string IDs, preserving first-seen order.
    """
    ids: list[str] = []

    # ── Case A: file path ─────────────────────────────────────────────────────
    if os.path.isfile(raw):
        try:
            with open(raw, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError:
            # Last resort: plain text, one ID per line
            with open(raw, "r", encoding="utf-8") as fh:
                return [line.strip() for line in fh if line.strip()]

        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    ids.append(item)
                elif isinstance(item, dict):
                    val = item.get("_id") or item.get("id")
                    if val:
                        ids.append(str(val))
        elif isinstance(data, dict):
            inner = data.get("response_ids", data)
            if isinstance(inner, list):
                for item in inner:
                    if isinstance(item, str):
                        ids.append(item)
                    elif isinstance(item, dict):
                        val = item.get("_id") or item.get("id")
                        if val:
                            ids.append(str(val))
            elif isinstance(inner, dict):
                # Keys are the IDs
                ids.extend(str(k) for k in inner.keys())
            else:
                # Top-level dict keys
                ids.extend(str(k) for k in data.keys())
        else:
            logger.error("Unrecognised JSON structure in response IDs file.")
            sys.exit(1)

    # ── Case B: comma-separated inline string ─────────────────────────────────
    else:
        ids = [part.strip() for part in raw.split(",") if part.strip()]

    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for rid in ids:
        if rid and rid not in seen:
            seen.add(rid)
            result.append(rid)
    return result


def _load_already_processed(output_path: str) -> set[str]:
    """
    Scan an existing JSONL output file and return the set of response_ids that
    have already been written (for resume support).
    """
    processed: set[str] = set()
    if not os.path.exists(output_path):
        return processed
    try:
        with open(output_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    rid = record.get("response_id")
                    if rid:
                        processed.add(str(rid))
                except json.JSONDecodeError:
                    pass
        logger.info(f"Found {len(processed)} already-processed ID(s) in {output_path!r} — resuming.")
    except OSError as exc:
        logger.warning(f"Could not read existing output file for resume check: {exc}")
    return processed


# ─────────────────────────────────────────────────────────────────────────────
# Main async entrypoint
# ─────────────────────────────────────────────────────────────────────────────

async def main_async() -> None:
    parser = argparse.ArgumentParser(
        prog="mongo_pipeline",
        description=(
            "Batch feature extraction pipeline — pulls candidate audio from MongoDB, "
            "transcribes each question's recording, combines them, then runs the full "
            "feature extraction & scoring pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--response-ids",
        required=True,
        metavar="IDS_OR_FILE",
        help=(
            "Comma-separated response IDs  OR  path to a JSON file "
            "(list of IDs / {\"response_ids\": [...]} / {\"response_ids\": {\"id\": ...}})."
        ),
    )
    parser.add_argument(
        "--output",
        default="mongo_extracted_features.jsonl",
        help="Path to write results (JSON Lines, one record per response ID).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of response IDs processed in parallel.",
    )
    parser.add_argument(
        "--mongodb-uri",
        default=None,
        help=(
            "MongoDB connection URI. "
            "Defaults to the MONGODB_URI environment variable."
        ),
    )
    parser.add_argument(
        "--mongodb-db",
        default=None,
        help=(
            "MongoDB database name. "
            "Defaults to the MONGODB_DB environment variable, then 'zeko'."
        ),
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=60,
        help="Skip individual question recordings longer than this many minutes.",
    )
    args = parser.parse_args()

    # ── Resolve MongoDB credentials ───────────────────────────────────────────
    mongodb_uri = os.getenv("MONGODB_CONNECTION_STRING")
    mongodb_db = args.mongodb_db or os.getenv("MONGODB_DB", "zeko")

    if not mongodb_uri:
        logger.error(
            "MongoDB URI not provided. Set --mongodb-uri or the MONGODB_URI "
            "environment variable (e.g. in a .env file)."
        )
        sys.exit(1)

    if not _PYMONGO_AVAILABLE:
        logger.error(
            "pymongo is not installed. Run: pip install pymongo  or  "
            "add 'pymongo' to requirements.txt and reinstall."
        )
        sys.exit(1)

    # ── Apply runtime MAX_DURATION_S override ─────────────────────────────────
    global MAX_DURATION_S
    MAX_DURATION_S = args.max_duration * 60

    # ── Load and validate response IDs ────────────────────────────────────────
    response_ids = _parse_response_ids(args.response_ids)
    if not response_ids:
        logger.error("No response IDs found — nothing to do.")
        sys.exit(1)
    logger.info(f"Loaded {len(response_ids)} response ID(s).")

    # ── Resume: skip already-processed IDs ───────────────────────────────────
    processed_ids = _load_already_processed(args.output)
    remaining_ids = [rid for rid in response_ids if rid not in processed_ids]
    logger.info(f"Remaining IDs to process: {len(remaining_ids)}")

    if not remaining_ids:
        logger.info("All response IDs are already processed. Exiting.")
        return

    # ── Ensure temp directory exists ──────────────────────────────────────────
    temp_dir = settings.TEMP_DIR
    os.makedirs(temp_dir, exist_ok=True)

    # ── Connect to MongoDB ────────────────────────────────────────────────────
    logger.info(f"Connecting to MongoDB ({mongodb_db})...")
    try:
        collection = await _run_in_executor(
            _get_mongo_collection, mongodb_uri, mongodb_db
        )
        # Quick connectivity check
        await _run_in_executor(collection.database.client.admin.command, "ping")
        logger.info("MongoDB connection OK.")
    except Exception as conn_exc:
        logger.error(f"Failed to connect to MongoDB: {conn_exc}")
        sys.exit(1)

    # ── Concurrency-controlled batch processing ───────────────────────────────
    semaphore = asyncio.Semaphore(args.concurrency)

    async def sem_process(rid: str) -> dict:
        async with semaphore:
            start = time.monotonic()
            result = await process_response(rid, collection, temp_dir)
            elapsed = time.monotonic() - start
            status = result.get("status", "unknown")
            if status == "success":
                logger.info(f"🟢 [{rid}] Completed in {elapsed:.1f}s")
            else:
                logger.error(
                    f"🔴 [{rid}] Failed in {elapsed:.1f}s — {result.get('error', 'unknown error')}"
                )
            return result

    tasks = [sem_process(rid) for rid in remaining_ids]

    # Write results to JSONL as each task completes (streaming, not buffered)
    success_count = 0
    error_count = 0
    with open(args.output, "a", encoding="utf-8") as out_fh:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_fh.flush()
            if result.get("status") == "success":
                success_count += 1
            else:
                error_count += 1

    logger.info(
        f"Batch complete — {success_count} succeeded, {error_count} failed. "
        f"Results written to {args.output!r}."
    )


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C). Exiting cleanly.")
    except Exception as exc:
        logger.critical(f"Unhandled critical error: {exc}")
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
