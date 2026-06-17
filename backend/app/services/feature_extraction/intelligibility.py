import logging
import re

logger = logging.getLogger(__name__)

PRONUNCIATION_THRESHOLD = 0.75

# Words that Whisper frequently gives low confidence NOT due to mispronunciation,
# but because they are acoustically short/ambiguous in context. Excluding these
# from the mispronounced list and from the pronunciation_score denominator prevents
# false-positive penalties on the most common English words.
_EXCLUDE_FROM_MISPRONOUNCED = {
    # Ultra-common function words
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "i", "you", "he", "she",
    "we", "they", "it", "this", "that", "so", "not", "just", "if",
    # Common spoken fillers that always appear
    "okay", "ok", "yeah", "yes", "no", "oh", "ah", "hmm", "um", "uh",
    "right", "well", "like", "also", "then", "now", "here", "there",
    "what", "when", "where", "who", "how", "which", "your", "my", "our",
    "their", "its", "me", "him", "her", "us", "them", "some", "any",
    "all", "by", "up", "out", "as", "from", "about", "into", "through",
    "because", "while", "though", "than", "both", "each", "more", "most",
    "very", "just", "too", "new", "one", "two", "he", "she", "we",
    "can", "get", "got", "go", "see", "know", "think", "want", "need",
    "thank", "thanks", "okay", "bye",
}


def _is_excluded(token: str) -> bool:
    """Return True if the token should be excluded from the mispronounced list."""
    clean = token.lower().strip(".,!?;:\"'")
    # Exclude stop-words
    if clean in _EXCLUDE_FROM_MISPRONOUNCED:
        return True
    # Exclude pure numbers (digits only, possibly with punctuation)
    if re.match(r"^[\d\s\.\,\%\$]+$", clean):
        return True
    # Exclude very short tokens (single character or empty)
    if len(clean) <= 1:
        return True
    return False


def extract_intelligibility_features(transcription_result: dict) -> dict:
    """
    Calculate mean and variance of ASR word confidences, and flag individual
    content words that are likely mispronounced (low Whisper confidence).

    Stop-words, numbers, and single-character tokens are excluded from the
    mispronounced list AND from the pronunciation_score denominator — Whisper
    routinely assigns low confidence to these not due to mispronunciation but
    due to acoustic ambiguity (short vowel tokens in connected speech).
    """
    all_confidences: list[float] = []     # all words — for mean/variance
    content_details: list[tuple[str, float]] = []  # content words only — for pronunciation_score

    for segment in transcription_result.get("segments", []):
        for word in segment.get("words", []):
            if "probability" in word:
                prob = word["probability"]
                text = word.get("word", "").strip()
                all_confidences.append(prob)
                if not _is_excluded(text):
                    content_details.append((text, prob))

    if not all_confidences:
        return {
            "mean_confidence":     0.0,
            "variance_confidence": 0.0,
            "low_confidence_flag": True,
            "pronunciation_score": 0.0,
            "mispronounced_words": [],
        }

    mean_conf     = sum(all_confidences) / len(all_confidences)
    variance_conf = sum((c - mean_conf) ** 2 for c in all_confidences) / len(all_confidences)

    # pronunciation_score: fraction of CONTENT words clearly pronounced (≥ threshold)
    if content_details:
        clearly_pronounced  = sum(1 for _, c in content_details if c >= PRONUNCIATION_THRESHOLD)
        pronunciation_score = round(clearly_pronounced / len(content_details), 4)
    else:
        # Fall back to all words if no content words found (very short transcript)
        clearly_pronounced  = sum(1 for c in all_confidences if c >= PRONUNCIATION_THRESHOLD)
        pronunciation_score = round(clearly_pronounced / len(all_confidences), 4)

    # mispronounced: content words below threshold only, sorted by confidence ascending
    mispronounced = sorted(
        [
            {"word": w, "confidence": round(c, 4)}
            for w, c in content_details
            if c < PRONUNCIATION_THRESHOLD
        ],
        key=lambda x: x["confidence"],
    )

    return {
        "mean_confidence":     round(mean_conf, 4),
        "variance_confidence": round(variance_conf, 4),
        "low_confidence_flag": bool(mean_conf < 0.60),
        "pronunciation_score": pronunciation_score,
        "mispronounced_words": mispronounced[:20],
    }
