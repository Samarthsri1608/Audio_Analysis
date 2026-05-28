import logging
import os
import time
import json
from google.genai import types
from pydantic import BaseModel, Field

from app.shared_models import get_gemini_client, get_diarization_cache, get_file_id

logger = logging.getLogger(__name__)

class TranscribeSegment(BaseModel):
    start: float = Field(description="Start time of the segment in seconds")
    end: float = Field(description="End time of the segment in seconds")
    text: str = Field(description="Accurate transcription of this segment")

class TranscriptionResultSchema(BaseModel):
    segments: list[TranscribeSegment] = Field(description="Chronological list of all transcribed segments")


def transcribe_audio(file_path: str) -> dict:
    """
    Transcribes an audio file. Checks the in-memory diarization cache first.
    If not found, uploads the file to the Gemini API and transcribes it directly.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # 1. Check cache first
    file_id = get_file_id(file_path)
    cache = get_diarization_cache()
    if file_id in cache:
        logger.info(f"ASR Cache HIT for file ID: {file_id}")
        return cache[file_id]

    logger.info(f"ASR Cache MISS for file ID: {file_id}. Running direct Gemini transcription...")

    # 2. Call Gemini API directly
    client = get_gemini_client()
    if client is None:
        raise RuntimeError("GOOGLE_API_KEY is not configured. Cannot call Gemini API.")

    logger.info(f"Uploading {file_path} to Gemini File API for ASR...")
    uploaded_file = client.files.upload(file=file_path)
    logger.info(f"Uploaded ASR file name: {uploaded_file.name}")

    try:
        # Poll file state
        while uploaded_file.state.name == "PROCESSING":
            logger.info("Waiting for ASR file to be processed by Gemini...")
            time.sleep(2)
            uploaded_file = client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name == "FAILED":
            raise RuntimeError(f"Gemini File API ASR processing failed: {uploaded_file.error.message}")

        logger.info("Running API-driven transcription...")
        prompt = (
            "You are an expert audio transcription system. "
            "Transcribe the spoken content in this audio recording. "
            "Segment the transcription and provide accurate start and end timestamps in seconds for each segment."
        )

        max_retries = 3
        backoff = 2
        response = None
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=[uploaded_file, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=TranscriptionResultSchema,
                    ),
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Gemini transcription failed after {max_retries} attempts: {e}")
                    raise
                logger.warning(f"Gemini transcription attempt {attempt+1} failed: {e}. Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2

        try:
            result_data = json.loads(response.text)
        except Exception as e:
            logger.error(f"Failed to parse Gemini ASR JSON response: {response.text}")
            raise RuntimeError(f"Gemini ASR response parsing failed: {e}")

        raw_segments = result_data.get("segments", [])
        reconstructed_segments = []
        full_text_list = []

        for seg in raw_segments:
            start_s = float(seg.get("start", 0.0))
            end_s = float(seg.get("end", 0.0))
            text = seg.get("text", "").strip()

            if not text:
                continue

            full_text_list.append(text)
            duration_s = end_s - start_s

            # Interpolate word-level timestamps to match the expected format
            segment_words = []
            words_text = text.split()
            if words_text and duration_s > 0:
                word_duration_s = duration_s / len(words_text)
                for i, w in enumerate(words_text):
                    w_start = start_s + (i * word_duration_s)
                    w_end = w_start + word_duration_s
                    segment_words.append({
                        "word": w,
                        "start": w_start,
                        "end": w_end,
                        "probability": 0.95
                    })

            reconstructed_segments.append({
                "id": len(reconstructed_segments),
                "start": start_s,
                "end": end_s,
                "text": text,
                "words": segment_words
            })

        return {
            "text": " ".join(full_text_list),
            "segments": reconstructed_segments
        }

    finally:
        # Delete file from Gemini
        try:
            client.files.delete(name=uploaded_file.name)
            logger.info(f"Deleted file {uploaded_file.name} from Gemini File API.")
        except Exception as e:
            logger.warning(f"Failed to delete file from Gemini: {e}")
