import logging

logger = logging.getLogger(__name__)

try:
    import parselmouth
    import numpy as np
    _parselmouth_available = True
except ImportError:
    logger.warning("parselmouth not installed. Voice modulation features will be skipped.")
    _parselmouth_available = False


def extract_voice_modulation_features(audio_path: str) -> dict:
    if not _parselmouth_available:
        return {
            "pitch_mean":      0.0,
            "pitch_std":       0.0,
            "pitch_range":     0.0,
            "monotone_flag":   False,
            "voiced_fraction": 0.0,
        }

    try:
        sound = parselmouth.Sound(audio_path)

        # pitch_ceiling lowered from 500 → 350 Hz.
        # Human conversational speech sits between 75–300 Hz for most speakers.
        # Using 500 Hz caused pitch_range to saturate at ceiling−floor ≈ 425 Hz
        # for the majority of recordings, making the metric meaningless.
        pitch_obj = sound.to_pitch(
            time_step=0.01,
            pitch_floor=75,
            pitch_ceiling=350,
        )

        pitch_values  = pitch_obj.selected_array["frequency"]
        voiced_values = pitch_values[pitch_values > 0]

        total_frames  = len(pitch_values)
        voiced_frames = len(voiced_values)
        voiced_fraction = round(voiced_frames / total_frames, 4) if total_frames > 0 else 0.0

        if voiced_frames == 0:
            return {
                "pitch_mean":      0.0,
                "pitch_std":       0.0,
                "pitch_range":     0.0,
                "monotone_flag":   True,
                "voiced_fraction": 0.0,
            }

        pitch_mean  = float(round(float(np.mean(voiced_values)), 2))
        pitch_std   = float(round(float(np.std(voiced_values)), 2))
        pitch_range = float(round(float(np.max(voiced_values) - np.min(voiced_values)), 2))

        return {
            "pitch_mean":      pitch_mean,
            "pitch_std":       pitch_std,
            "pitch_range":     pitch_range,
            "monotone_flag":   bool(pitch_std < 20.0),
            "voiced_fraction": voiced_fraction,
        }

    except Exception as e:
        logger.error(f"Voice modulation extraction failed: {e}")
        return {
            "pitch_mean":      0.0,
            "pitch_std":       0.0,
            "pitch_range":     0.0,
            "monotone_flag":   False,
            "voiced_fraction": 0.0,
        }
