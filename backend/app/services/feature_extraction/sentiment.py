import logging

logger = logging.getLogger(__name__)

try:
    import nltk
    nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _sia = SentimentIntensityAnalyzer()
    _vader_available = True
except Exception as e:
    logger.error(f"VADER sentiment analyser could not be loaded: {e}")
    _vader_available = False

import re

HEDGING_PHRASES = {
    "i guess", "i suppose", "i think maybe", "i'm not sure", "i am not sure",
    "not sure", "sort of", "kind of", "maybe", "perhaps", "possibly",
    "probably", "might be", "could be", "i feel like", "i don't know",
    "i do not know", "unsure", "uncertain", "i hope", "hopefully",
    "i believe maybe", "i assume",
}

ASSERTIVE_PHRASES = {
    "i know", "i am confident", "i'm confident", "definitely", "certainly",
    "absolutely", "clearly", "i have", "i led", "i built", "i created",
    "i achieved", "i demonstrated", "i designed", "i managed", "i delivered",
    "i ensured", "i developed", "i implemented", "i established",
    "i specialise", "i specialize", "my expertise", "i have experience",
    "i am skilled", "i'm skilled", "i am experienced", "i'm experienced",
    "i successfully", "i consistently", "i proved", "i showed",
}


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5


def _empty_result() -> dict:
    return {
        "mean_compound":           0.0,
        "std_compound":            0.0,
        "neg_sentiment_ratio":     0.0,
        "positive_ratio":          0.0,
        "hedging_count":           0,
        "assertive_count":         0,
        "hedge_rate":              0.0,
        "hedging_phrases_found":   [],
        "assertive_phrases_found": [],
    }


def extract_sentiment_features(text: str) -> dict:
    if not _vader_available or not text.strip():
        return _empty_result()

    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return _empty_result()

    vader_scores = [_sia.polarity_scores(s) for s in sentences]
    compounds    = [s["compound"] for s in vader_scores]

    mean_compound = round(sum(compounds) / len(compounds), 4)
    std_compound  = round(_std(compounds), 4)

    neg_sent_count = sum(1 for s in vader_scores if s["neg"] > 0.3)
    neg_sent_ratio = round(neg_sent_count / len(vader_scores), 4)

    pos_sent_count = sum(1 for s in vader_scores if s["pos"] > 0.3)
    positive_ratio = round(pos_sent_count / len(vader_scores), 4)

    text_lower  = text.lower()
    total_words = max(len(text.split()), 1)

    # Use regex word boundaries to count all occurrences of each phrase
    hedging_hits = []
    for phrase in HEDGING_PHRASES:
        pattern = re.compile(rf"\b{re.escape(phrase)}\b")
        matches = pattern.findall(text_lower)
        if matches:
            hedging_hits.extend([phrase] * len(matches))

    assertive_hits = []
    for phrase in ASSERTIVE_PHRASES:
        pattern = re.compile(rf"\b{re.escape(phrase)}\b")
        matches = pattern.findall(text_lower)
        if matches:
            assertive_hits.extend([phrase] * len(matches))

    hedging_found = sorted(set(hedging_hits))
    assertive_found = sorted(set(assertive_hits))

    hedge_count  = len(hedging_hits)
    assert_count = len(assertive_hits)
    hedge_rate   = round(hedge_count / total_words * 100, 4)

    return {
        "mean_compound":           float(mean_compound),
        "std_compound":            float(std_compound),
        "neg_sentiment_ratio":     float(neg_sent_ratio),
        "positive_ratio":          float(positive_ratio),
        "hedging_count":           int(hedge_count),
        "assertive_count":         int(assert_count),
        "hedge_rate":              float(hedge_rate),
        "hedging_phrases_found":   hedging_found,
        "assertive_phrases_found": assertive_found,
    }
