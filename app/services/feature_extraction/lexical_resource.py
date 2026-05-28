import logging

logger = logging.getLogger(__name__)

try:
    from wordfreq import word_frequency
    _wordfreq_available = True
except ImportError:
    logger.warning("wordfreq not installed. Vocabulary sophistication metrics will be skipped.")
    _wordfreq_available = False

# Attempt spaCy — may fail on Python 3.14+ (pydantic v1 incompatibility)
try:
    from app.shared_models import get_spacy_nlp as _get_nlp
    _spacy_ok = True
except Exception:
    _spacy_ok = False

_RARE_THRESHOLD = 1e-5

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "i", "you", "he", "she",
    "we", "they", "it", "this", "that", "so", "not", "just",
}

_MATTR_WINDOW = 50


def _compute_mattr(tokens: list[str], window: int = _MATTR_WINDOW) -> float:
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    ttrs = [
        len(set(tokens[i: i + window])) / window
        for i in range(len(tokens) - window + 1)
    ]
    return sum(ttrs) / len(ttrs)


def _simple_tokenise(text: str) -> list[str]:
    """
    Fallback tokeniser when spaCy is unavailable.
    Strips punctuation and lowercases — sufficient for MATTR/frequency computation.
    """
    import re
    tokens = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    return tokens


def extract_lexical_features(text: str) -> dict:
    if not text.strip():
        return {
            "type_token_ratio":           0.0,
            "mattr":                      0.0,
            "unique_words":               0,
            "rare_word_ratio":            0.0,
            "avg_word_frequency":         0.0,
            "sophisticated_words_sample": [],
        }

    # Tokenise — prefer spaCy, fall back to regex
    tokens: list[str] = []
    if _spacy_ok:
        nlp = _get_nlp()
        if nlp is not None:
            doc    = nlp(text)
            tokens = [t.text.lower() for t in doc if not t.is_punct and not t.is_space]

    if not tokens:
        tokens = _simple_tokenise(text)

    if not tokens:
        return {
            "type_token_ratio":           0.0,
            "mattr":                      0.0,
            "unique_words":               0,
            "rare_word_ratio":            0.0,
            "avg_word_frequency":         0.0,
            "sophisticated_words_sample": [],
        }

    unique_words = set(tokens)
    ttr = float(len(unique_words)) / float(len(tokens))

    if _wordfreq_available:
        content_tokens = [t for t in tokens if t not in _STOPWORDS and t.isalpha()]
        if content_tokens:
            freq_scores  = [word_frequency(t, "en") for t in content_tokens]
            rare_words   = [t for t, f in zip(content_tokens, freq_scores) if f < _RARE_THRESHOLD and f > 0]
            rare_ratio   = round(len(rare_words) / len(content_tokens), 4)
            avg_freq     = round(sum(freq_scores) / len(freq_scores), 8)
            sophisticated_sample = sorted(set(rare_words))[:10]
        else:
            rare_ratio = avg_freq = 0.0
            sophisticated_sample = []
    else:
        rare_ratio = avg_freq = 0.0
        sophisticated_sample = []

    return {
        "type_token_ratio":           round(ttr, 4),
        "mattr":                      round(_compute_mattr(tokens), 4),
        "unique_words":               len(unique_words),
        "rare_word_ratio":            float(rare_ratio),
        "avg_word_frequency":         float(avg_freq),
        "sophisticated_words_sample": sophisticated_sample,
    }
