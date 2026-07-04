"""
pipeline/audio_fetcher.py — Download question recordings via the REST API loop,
concat them, and convert to 16kHz mono WAV.

Identical pattern to mongo_pipeline_new.py: loop from quesNo=1 until the API
returns a non-200 status or an empty/missing recordingLink, then stitch.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import httpx

from v2.config import (
    INTERVIEW_API_BASE,
    MAX_QUESTIONS,
    SAMPLE_RATE,
    TEMP_DIR_PREFIX,
    MAX_CONCURRENT_JOBS,
)

logger = logging.getLogger("v2.audio_fetcher")


# ── helpers ───────────────────────────────────────────────────────────────────


async def _download_stream(url: str, dest: str) -> None:
    """Stream-download a URL to a local file."""
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                async for chunk in resp.aiter_bytes(chunk_size=32_768):
                    fh.write(chunk)


async def _to_wav(input_path: str, output_path: str) -> None:
    """Convert any media file to 16kHz mono WAV."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-i", input_path,
        "-ar", str(SAMPLE_RATE),
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
            f"ffmpeg WAV conversion failed (exit {proc.returncode}): {stderr.decode().strip()}"
        )


# ── main entry ────────────────────────────────────────────────────────────────


async def fetch_and_prepare_audio(
    response_id: str,
    temp_dir: str,
) -> list[str]:
    """
    Query the API in parallel for quesNo 1 to MAX_QUESTIONS to find all valid recording links,
    download them, convert each to a 16kHz mono WAV file, and return the list of paths.

    Returns:
        List of absolute paths to individual 16kHz mono WAV files.

    Raises:
        ValueError: if no recordings are found for the response_id.
    """
    logger.info("[%s] Probing all questions in parallel (max %d questions)", response_id, MAX_QUESTIONS)

    async def probe_question(q_no: int) -> dict | None:
        url = f"{INTERVIEW_API_BASE}?responseId={response_id}&quesNo={q_no}"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                link = data.get("recordingLink")
                if link:
                    return {"q_no": q_no, "link": link}
        except Exception as exc:
            logger.debug("[%s] Failed probing Q%d: %s", response_id, q_no, exc)
        return None

    tasks = [probe_question(q_no) for q_no in range(1, MAX_QUESTIONS + 1)]
    probe_results = await asyncio.gather(*tasks)

    # Filter out None and sort by question number
    valid_questions = sorted(
        [r for r in probe_results if r is not None],
        key=lambda x: x["q_no"]
    )

    if not valid_questions:
        raise ValueError(f"No recordings found for response_id={response_id!r}")

    logger.info("[%s] Found %d valid recording(s)", response_id, len(valid_questions))

    # Process download and conversion with concurrency limit
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

    async def process_question(q_no: int, link: str) -> str:
        async with semaphore:
            # Detect file extension
            url_path = link.split("?")[0]
            ext = Path(url_path).suffix or ".mp4"
            local_raw_path = os.path.join(temp_dir, f"q_{q_no:02d}{ext}")
            local_wav_path = os.path.join(temp_dir, f"q_{q_no:02d}.wav")

            logger.info("[%s] Downloading Q%d …", response_id, q_no)
            await _download_stream(link, local_raw_path)

            logger.info("[%s] Converting Q%d → WAV …", response_id, q_no)
            await _to_wav(local_raw_path, local_wav_path)

            # Clean up raw download
            try:
                os.remove(local_raw_path)
            except Exception as e:
                logger.debug("[%s] Failed to delete raw Q%d file: %s", response_id, q_no, e)

            return local_wav_path

    process_tasks = [process_question(q["q_no"], q["link"]) for q in valid_questions]
    wav_paths = await asyncio.gather(*process_tasks)
    return wav_paths
