import logging

logger = logging.getLogger(__name__)

FILLER_WORDS = {
    "um", "uh", "uhh", "umm", "hmm", "hm",
    "like", "basically", "literally", "actually",
    "you know", "i mean", "sort of", "kind of",
    "right", "okay", "so", "well",
}

FILLER_SINGLE = {w for w in FILLER_WORDS if " " not in w}
FILLER_PHRASES = {w for w in FILLER_WORDS if " " in w}

# Scripted-speech detection thresholds
MIN_FILLER_RATE   = 0.008
MAX_NATURAL_WPM   = 185
MIN_PAUSE_PER_MIN = 3  # Lowered from 6 → 3: Indian English speakers naturally pause
                       # less than Western benchmarks. 6/min was a false-positive epidemic
                       # (67% flagged). 3/min correctly targets only unusually smooth delivery.


def _compute_scripted_speech_score(
    filler_rate: float,
    wpm: float,
    pause_frequency: int,
    duration_minutes: float,
    total_words: int,
) -> dict:
    signals: dict = {}
    score = 0

    if total_words >= 30:
        if filler_rate < MIN_FILLER_RATE:
            pts = int(round(40 * (1 - filler_rate / MIN_FILLER_RATE)))
            score += min(pts, 40)
            signals["low_filler_rate"] = {
                "flagged": True,
                "value": round(filler_rate, 4),
                "threshold": MIN_FILLER_RATE,
                "description": (
                    f"Filler rate {round(filler_rate * 100, 2)}% is below the "
                    f"natural minimum of {MIN_FILLER_RATE * 100}%. "
                    "Very few hesitation words detected — speech may be scripted."
                ),
            }
        else:
            signals["low_filler_rate"] = {"flagged": False, "value": round(filler_rate, 4)}

    if duration_minutes >= 0.3:
        if wpm > MAX_NATURAL_WPM:
            excess_ratio = min((wpm - MAX_NATURAL_WPM) / MAX_NATURAL_WPM, 1.0)
            pts = int(round(30 * excess_ratio))
            score += min(pts, 30)
            signals["high_wpm"] = {
                "flagged": True,
                "value": round(wpm, 1),
                "threshold": MAX_NATURAL_WPM,
                "description": (
                    f"Speech rate of {round(wpm, 1)} WPM exceeds the natural "
                    f"spontaneous speech ceiling of {MAX_NATURAL_WPM} WPM. "
                    "Candidate may be reading rather than thinking aloud."
                ),
            }
        else:
            signals["high_wpm"] = {"flagged": False, "value": round(wpm, 1)}

        pauses_per_minute = pause_frequency / duration_minutes if duration_minutes > 0 else 0
        if pauses_per_minute < MIN_PAUSE_PER_MIN:
            deficit_ratio = min(
                (MIN_PAUSE_PER_MIN - pauses_per_minute) / MIN_PAUSE_PER_MIN, 1.0
            )
            pts = int(round(30 * deficit_ratio))
            score += min(pts, 30)
            signals["low_pause_frequency"] = {
                "flagged": True,
                "value": round(pauses_per_minute, 2),
                "threshold": MIN_PAUSE_PER_MIN,
                "description": (
                    f"Only {round(pauses_per_minute, 1)} pauses/minute detected "
                    f"(minimum natural threshold: {MIN_PAUSE_PER_MIN}/min). "
                    "Unusually smooth delivery may indicate scripted reading."
                ),
            }
        else:
            signals["low_pause_frequency"] = {
                "flagged": False,
                "value": round(pauses_per_minute, 2),
            }

    score = min(score, 100)

    if score >= 70:
        severity = "high"
    elif score >= 40:
        severity = "medium"
    elif score >= 15:
        severity = "low"
    else:
        severity = "none"

    return {
        "ai_scripted_speech_score": int(score),
        "severity": severity,
        "flagged": score >= 40,
        "flagged_signals": [k for k, v in signals.items() if v.get("flagged")],
        "signals": signals,
    }


def extract_fluency_features(transcription_result: dict, duration_ms: float) -> dict:
    words = []
    for segment in transcription_result.get("segments", []):
        words.extend(segment.get("words", []))

    total_words    = len(words)
    duration_minutes = duration_ms / 60000.0
    wpm = total_words / duration_minutes if duration_minutes > 0 else 0

    pauses: list[float] = []
    PAUSE_THRESHOLD = 0.3

    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap >= PAUSE_THRESHOLD:
            pauses.append(gap)

    pause_frequency     = len(pauses)
    mean_pause_duration = float(sum(pauses)) / float(len(pauses)) if pauses else 0.0

    word_texts = [w.get("word", "").strip().lower().strip(".,!?;:") for w in words]

    filler_hits: list[str] = []
    for token in word_texts:
        if token in FILLER_SINGLE:
            filler_hits.append(token)

    full_text_lower = " ".join(word_texts)
    for phrase in FILLER_PHRASES:
        count = full_text_lower.count(phrase)
        filler_hits.extend([phrase] * count)

    filler_count = len(filler_hits)
    filler_rate  = round(filler_count / total_words, 4) if total_words > 0 else 0.0

    return {
        "wpm":                 float(round(wpm, 2)),
        "pause_frequency":     int(pause_frequency),
        "mean_pause_duration": float(round(mean_pause_duration, 2)),
        "total_words":         int(total_words),
        "filler_count":        int(filler_count),
        "filler_rate":         float(filler_rate),
        "filler_words_found":  sorted(set(filler_hits)),
        "scripted_detection":  _compute_scripted_speech_score(
            filler_rate=filler_rate,
            wpm=wpm,
            pause_frequency=pause_frequency,
            duration_minutes=duration_minutes,
            total_words=total_words,
        ),
    }
