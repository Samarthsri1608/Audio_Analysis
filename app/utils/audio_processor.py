import logging
import os
from pydub import AudioSegment

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE  = 16000
MIN_AUDIO_LENGTH_MS = 30000


def preprocess_audio(file_path: str) -> dict:
    """
    Normalise audio for Whisper:
    - Convert to 16 kHz mono .wav
    - Check duration
    - Basic quality/SNR estimation
    """
    flags: list[str] = []
    try:
        audio = AudioSegment.from_file(file_path)
    except Exception as e:
        raise ValueError(f"Failed to load audio: {e}")

    duration_ms = len(audio)

    if duration_ms < MIN_AUDIO_LENGTH_MS:
        flags.append("AUDIO_TOO_SHORT")

    audio = audio.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(1)

    avg_loudness = audio.dBFS
    if avg_loudness < -40.0:
        flags.append("LOW_LOUDNESS")

    if audio.max_dBFS - avg_loudness < 10:
        flags.append("POSSIBLE_LOW_SNR")

    processed_path = f"{file_path}_processed.wav"
    audio.export(processed_path, format="wav")

    if flags:
        logger.warning(f"Audio quality flags: {flags} for {os.path.basename(file_path)}")

    return {
        "processed_file_path": processed_path,
        "duration_ms":         duration_ms,
        "flags":               flags,
        "avg_loudness_dbfs":   avg_loudness,
    }
