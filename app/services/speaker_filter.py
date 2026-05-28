import os
import json
import logging
import time
import warnings
from pydub import AudioSegment
from google.genai import types
from pydantic import BaseModel, Field

from app.settings import settings
from app.shared_models import get_gemini_client, get_diarization_cache, get_file_id

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message="std\\(\\): degrees of freedom is <= 0",
    category=UserWarning,
)

class SpeechSegment(BaseModel):
    speaker: str = Field(description="Must be either 'Interviewer' or 'Candidate'")
    start: float = Field(description="Start time of the segment in seconds")
    end: float = Field(description="End time of the segment in seconds")
    text: str = Field(description="Accurate transcription of the speech in this segment")

class DiarizationResult(BaseModel):
    segments: list[SpeechSegment] = Field(description="Chronological list of all speech segments")


def extract_interviewee_audio(audio_path: str) -> str:
    """
    Runs speaker diarization on the entire audio_path using Gemini API to identify
    the interviewee (candidate), then extracts all candidate speech segments into a new .wav file.
    Also caches the transcription result to avoid duplicate transcription calls.
    """
    client = get_gemini_client()
    if client is None:
        raise RuntimeError("GOOGLE_API_KEY is not configured. Cannot call Gemini API.")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(f"Uploading {audio_path} to Gemini File API...")
    uploaded_file = client.files.upload(file=audio_path)
    logger.info(f"Uploaded file name: {uploaded_file.name}")

    try:
        # Poll file state until active
        while uploaded_file.state.name == "PROCESSING":
            logger.info("Waiting for audio file to be processed by Gemini...")
            time.sleep(2)
            uploaded_file = client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name == "FAILED":
            raise RuntimeError(f"Gemini File API processing failed: {uploaded_file.error.message}")

        logger.info("Running API-driven speaker diarization and transcription...")
        prompt = (
            "You are an expert audio diarization and transcription system. "
            "Analyze this interview recording. Separate the conversation into speech segments. "
            "Identify the two speakers: the 'Interviewer' and the 'Candidate' (the interviewee). "
            "For each speech segment, return the speaker, start time in seconds, end time in seconds, and text transcription. "
            "Ensure that the start and end times are as precise as possible, and the text is transcribed accurately."
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
                        response_schema=DiarizationResult,
                    ),
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Gemini diarization failed after {max_retries} attempts: {e}")
                    raise
                logger.warning(f"Gemini diarization attempt {attempt+1} failed: {e}. Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2

        # Parse diarization result
        try:
            result_data = json.loads(response.text)
        except Exception as e:
            logger.error(f"Failed to parse Gemini diarization JSON response: {response.text}")
            raise RuntimeError(f"Gemini response parsing failed: {e}")

        raw_segments = result_data.get("segments", [])
        if not raw_segments:
            raise ValueError("Gemini returned empty speaker segments.")

        logger.info(f"Diarization completed. Total segments found: {len(raw_segments)}")

        # Filter candidate segments
        candidate_segments = []
        for seg in raw_segments:
            spk = seg.get("speaker", "").lower()
            if "cand" in spk or "interviewee" in spk or "speaker 2" in spk:
                candidate_segments.append(seg)

        # Fallback to duration heuristic if no segments explicitly matched
        if not candidate_segments:
            logger.warning("No segments explicitly labeled as 'Candidate'. Falling back to duration heuristic.")
            speaker_durations = {}
            for seg in raw_segments:
                spk = seg.get("speaker", "Unknown")
                dur = float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))
                speaker_durations[spk] = speaker_durations.get(spk, 0.0) + dur
            if speaker_durations:
                interviewee_id = max(speaker_durations, key=speaker_durations.get)
                candidate_segments = [seg for seg in raw_segments if seg.get("speaker") == interviewee_id]
                logger.info(f"Identified interviewee as '{interviewee_id}' via duration fallback.")

        if not candidate_segments:
            raise ValueError("Could not identify candidate/interviewee segments.")

        # Load audio locally using pydub
        audio = AudioSegment.from_file(audio_path)
        logger.info(f"Loaded audio locally. Duration: {len(audio)/1000:.1f}s.")

        # Reconstruct interviewee-only audio
        interviewee_audio = AudioSegment.empty()
        reconstructed_segments = []
        current_time_ms = 0.0

        for seg in candidate_segments:
            start_s = float(seg.get("start", 0.0))
            end_s = float(seg.get("end", 0.0))
            text = seg.get("text", "").strip()

            start_ms = int(start_s * 1000)
            end_ms = int(end_s * 1000)

            # Clamp boundaries
            start_ms = max(0, min(start_ms, len(audio)))
            end_ms = max(0, min(end_ms, len(audio)))

            if end_ms > start_ms:
                duration_ms = end_ms - start_ms
                interviewee_audio += audio[start_ms:end_ms]

                # Reconstruct segments for ASR mapping with interpolated word timestamps
                segment_words = []
                words_text = text.split()
                if words_text:
                    word_duration_s = (duration_ms / 1000.0) / len(words_text)
                    for i, w in enumerate(words_text):
                        w_start = (current_time_ms / 1000.0) + (i * word_duration_s)
                        w_end = w_start + word_duration_s
                        segment_words.append({
                            "word": w,
                            "start": w_start,
                            "end": w_end,
                            "probability": 0.95  # Simulated high confidence
                        })

                reconstructed_segments.append({
                    "id": len(reconstructed_segments),
                    "start": current_time_ms / 1000.0,
                    "end": (current_time_ms + duration_ms) / 1000.0,
                    "text": text,
                    "words": segment_words
                })
                current_time_ms += duration_ms

        if len(interviewee_audio) == 0:
            raise ValueError("No candidate audio extracted.")

        # Save to output path
        base, _ = os.path.splitext(audio_path)
        output_path = f"{base}_interviewee.wav"
        interviewee_audio.export(output_path, format="wav")
        logger.info(f"Interviewee audio saved: {output_path} ({len(interviewee_audio)/1000:.1f}s)")

        # Save transcript to in-memory cache for ASR reuse
        file_id = get_file_id(output_path)
        cache_data = {
            "text": " ".join([seg["text"] for seg in reconstructed_segments]),
            "segments": reconstructed_segments
        }
        get_diarization_cache()[file_id] = cache_data
        logger.info(f"Cached transcription data for file ID: {file_id}")

        return output_path

    finally:
        # Delete file from Gemini File API
        try:
            client.files.delete(name=uploaded_file.name)
            logger.info(f"Deleted file {uploaded_file.name} from Gemini File API.")
        except Exception as e:
            logger.warning(f"Failed to delete file from Gemini: {e}")
