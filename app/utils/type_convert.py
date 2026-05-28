import logging
import os
import subprocess

logger = logging.getLogger(__name__)


async def convert_video_to_audio(video_path: str, audio_path: str) -> None:
    """
    Convert a video file (.mp4) to a 16 kHz mono WAV using ffmpeg.
    Raises RuntimeError on conversion failure.
    """
    command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        logger.info(f"Converted {os.path.basename(video_path)} → {os.path.basename(audio_path)}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg conversion failed: {e.stderr}") from e
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found. Ensure it is installed and on PATH.") from e