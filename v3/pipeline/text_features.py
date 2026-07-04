"""
pipeline/text_features.py — Extract all text-based features from transcript.

Implements the full feature inventory from Zeko Unified Communication
Framework v1 (Section 1.1):
  F01 fluency_wpm              — from word timestamps
  F02 fluency_filler_rate      — filler words / total words
  F06 lexical_mattr            — MATTR (replaces TTR per framework §3.2)
  F07 lexical_rare             — rare word ratio via wordfreq
  F08 discourse_connectors     — unique connector types used
  F09 discourse_tier1          — Tier-1 high-quality connector count
  F11 ner_entity_density       — entities/min via SpaCy + Indian gazetteer
  F12 metric_density           — numbers + % patterns / minute
  F13 sbert_coherence          — sentence-level cosine similarity (multilingual SBERT)
  F14 collaborative_ratio      — team pronoun ratio
  F15 question_density         — questions per minute
  F16 empathetic_markers       — empathetic word density
  F20 avg_sentence_length      — words per sentence
  narrative_arc_score          — Labovian arc completeness (0–1)

Indian English corrections applied at extraction layer:
  - Homophone correction dict applied to transcript before connector counting
  - Indian proper noun gazetteer injected into SpaCy NER
  - SBERT cosine similarity targets adjusted by -0.06 offset
  - Discourse connector count target scaled by 0.90 (10% compensation)

V3 fix: compute_filler_rate now correctly handles multi-word filler phrases
  ('i mean', 'you know') using full-string matching in addition to per-word
  single-word filler matching. In v2, the _tokenize() split approach silently
  ignored all multi-word entries in FILLER_WORDS.

V3 note: With AssemblyAI disfluencies=True, 'um', 'uh' etc. appear verbatim
  in the transcript text, so FILLER_WORDS_SINGLE now captures them directly
  without any post-hoc pattern matching workaround.
"""
from __future__ import annotations

import re
import logging
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from v3.pipeline.transcriber import WordTimestamp

logger = logging.getLogger("v3.text_features")

# ── Homophone correction (Framework §1.2, Indian English ASR errors) ──────────
_HOMOPHONE_CORRECTIONS: dict[str, str] = {
    "hens": "hence",
    "more over": "moreover",
    "consequent league": "consequently",
    "there for": "therefore",
    "how ever": "however",
    "never the less": "nevertheless",
    "further more": "furthermore",
    "in stead": "instead",
    "al though": "although",
    "be cause": "because",
}


def _apply_homophone_corrections(text: str) -> str:
    """Pre-process transcript to fix common Indian English ASR mismatches."""
    corrected = text.lower()
    for wrong, right in _HOMOPHONE_CORRECTIONS.items():
        corrected = corrected.replace(wrong, right)
    return corrected


# ── Discourse connector tiers (Framework §2.2, Axis 3) ───────────────────────
# Tier 1 — High-quality / academic connectors (weighted 2×)
CONNECTORS_TIER1: frozenset[str] = frozenset({
    "furthermore", "consequently", "nevertheless", "however", "therefore",
    "moreover", "nonetheless", "subsequently", "conversely", "accordingly",
    "alternatively", "predominantly", "specifically", "additionally",
    "in conclusion", "as a result", "in contrast", "on the other hand",
    "to summarize", "ultimately",
})

# Tier 2 — Common connectors (weighted 1×)
CONNECTORS_TIER2: frozenset[str] = frozenset({
    "but", "and", "so", "because", "although", "while", "since", "also",
    "then", "thus", "yet", "hence", "whether", "unless", "meanwhile",
    "besides", "instead", "despite", "rather", "first", "second", "third",
    "finally", "lastly", "next", "after", "before",
})

ALL_CONNECTORS: frozenset[str] = CONNECTORS_TIER1 | CONNECTORS_TIER2

# ── Filler words ──────────────────────────────────────────────────────────────
# V3 FIX: Separated into single-word set and multi-word phrase tuple.
# The old FILLER_WORDS frozenset contained multi-word entries ('i mean',
# 'you know') which _tokenize() (word-split) can never match — they were
# silently ignored, causing under-counting of fillers.
#
# With AssemblyAI disfluencies=True, 'um'/'uh' now appear in the transcript
# text directly so single-word matching catches them without workarounds.

FILLER_WORDS_SINGLE: frozenset[str] = frozenset({
    "um", "uh", "like", "basically", "literally", "actually",
    "sort", "kinda", "gonna", "wanna", "gotta",
    "right", "well", "so", "okay", "uhh", "umm", "hmm",
})

FILLER_PHRASES: tuple[str, ...] = (
    "i mean",
    "you know",
)

# Backward-compatibility alias (used in external tests/logging)
FILLER_WORDS: frozenset[str] = FILLER_WORDS_SINGLE

# ── Pronouns ───────────────────────────────────────────────────────────────────
COLLABORATIVE_PRONOUNS: frozenset[str] = frozenset({"we", "us", "our", "ourselves", "ours"})
INDIVIDUAL_PRONOUNS: frozenset[str] = frozenset({"i", "me", "my", "mine", "myself"})

# ── Empathetic words ───────────────────────────────────────────────────────────
EMPATHETIC_WORDS: frozenset[str] = frozenset({
    "understand", "appreciate", "empathize", "feel", "recognize",
    "acknowledge", "support", "help", "care", "concern",
    "agree", "tough", "challenging", "difficult", "struggle",
    "listen", "together", "community",
})

# ── Narrative arc markers (Labov & Waletzky 1967, Framework §2.2 Axis 4) ──────
_ORIENTATION_MARKERS: list[str] = [
    "when i", "at the time", "the situation was", "we were", "i was working",
    "the context was", "in my previous", "at my last", "back then",
]
_COMPLICATION_MARKERS: list[str] = [
    "however", "but then", "the challenge was", "the problem", "unfortunately",
    "the issue was", "we faced", "it was difficult", "the blocker",
]
_ACTION_MARKERS: list[str] = [
    "so i", "what i did", "my approach", "i decided", "we implemented",
    "i took", "i proposed", "i led", "we built", "i designed",
]
_RESOLUTION_MARKERS: list[str] = [
    "as a result", "ultimately", "the outcome", "we achieved", "this led to",
    "in the end", "finally", "we managed", "successfully", "the result was",
]

# ── Metric pattern (numbers + quantifiers) ─────────────────────────────────────
_METRIC_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*"
    r"(%|x|times|percent|increase|growth|improvement|rise|jump|boost|fold"
    r"|million|billion|crore|lakh|k\b)",
    re.IGNORECASE,
)

# ── Sentence splitter ──────────────────────────────────────────────────────────
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

# ── Indian NER gazetteer (Framework §1.2) ─────────────────────────────────────
# Loaded lazily via _get_nlp() to avoid import-time cost.
_INDIAN_PATTERNS: list[dict] = [
    # Educational institutions
    {"label": "ORG", "pattern": "IIT"},
    {"label": "ORG", "pattern": "NIT"},
    {"label": "ORG", "pattern": "IIM"},
    {"label": "ORG", "pattern": "BITS Pilani"},
    {"label": "ORG", "pattern": "IIT Madras"},
    {"label": "ORG", "pattern": "IIT Bombay"},
    {"label": "ORG", "pattern": "IIT Delhi"},
    {"label": "ORG", "pattern": "IIT Kharagpur"},
    {"label": "ORG", "pattern": "IIT Roorkee"},
    {"label": "ORG", "pattern": "IISC"},
    {"label": "ORG", "pattern": "VIT"},
    {"label": "ORG", "pattern": "SRM"},
    {"label": "ORG", "pattern": "BITS"},
    # Companies — large IT
    {"label": "ORG", "pattern": "Infosys"},
    {"label": "ORG", "pattern": "TCS"},
    {"label": "ORG", "pattern": "Wipro"},
    {"label": "ORG", "pattern": "HCL"},
    {"label": "ORG", "pattern": "Tech Mahindra"},
    {"label": "ORG", "pattern": "Cognizant"},
    {"label": "ORG", "pattern": "Mphasis"},
    {"label": "ORG", "pattern": "NIIT"},
    # Companies — startups / consumer tech
    {"label": "ORG", "pattern": "Razorpay"},
    {"label": "ORG", "pattern": "Zerodha"},
    {"label": "ORG", "pattern": "Flipkart"},
    {"label": "ORG", "pattern": "Swiggy"},
    {"label": "ORG", "pattern": "Zomato"},
    {"label": "ORG", "pattern": "BYJU"},
    {"label": "ORG", "pattern": "CRED"},
    {"label": "ORG", "pattern": "PhonePe"},
    {"label": "ORG", "pattern": "Paytm"},
    {"label": "ORG", "pattern": "Meesho"},
    {"label": "ORG", "pattern": "OYO"},
    {"label": "ORG", "pattern": "Ola"},
    {"label": "ORG", "pattern": "Nykaa"},
    # Geographies — Tier 1
    {"label": "GPE", "pattern": "Bengaluru"},
    {"label": "GPE", "pattern": "Bangalore"},
    {"label": "GPE", "pattern": "Pune"},
    {"label": "GPE", "pattern": "Hyderabad"},
    {"label": "GPE", "pattern": "Chennai"},
    {"label": "GPE", "pattern": "Noida"},
    {"label": "GPE", "pattern": "Gurugram"},
    {"label": "GPE", "pattern": "Gurgaon"},
    {"label": "GPE", "pattern": "Ahmedabad"},
    {"label": "GPE", "pattern": "Kolkata"},
    # Finance vocabulary (Indian context)
    {"label": "MONEY", "pattern": [{"LIKE_NUM": True}, {"LOWER": "crore"}]},
    {"label": "MONEY", "pattern": [{"LIKE_NUM": True}, {"LOWER": "lakh"}]},
]

_nlp = None
_nlp_lock = __import__("threading").Lock()


def _get_nlp():
    """Lazily initialise SpaCy with Indian NER gazetteer."""
    global _nlp
    if _nlp is None:
        with _nlp_lock:
            if _nlp is None:
                try:
                    import spacy
                    from spacy.pipeline import EntityRuler
                    _nlp_instance = spacy.load("en_core_web_sm", disable=["parser", "tagger"])
                    ruler = _nlp_instance.add_pipe("entity_ruler", before="ner")
                    ruler.add_patterns(_INDIAN_PATTERNS)
                    _nlp = _nlp_instance
                    logger.info("SpaCy NLP loaded with Indian gazetteer (%d patterns)", len(_INDIAN_PATTERNS))
                except Exception as exc:
                    logger.warning("SpaCy unavailable (%s) — NER will return 0", exc)
                    _nlp = "unavailable"
    return _nlp if _nlp != "unavailable" else None


# ── SBERT model (multilingual, Framework §1.2 + §2.2 Axis 3) ─────────────────
_sbert = None
_sbert_lock = __import__("threading").Lock()
_SBERT_COHERENCE_OFFSET = -0.06   # Indian English calibration: Framework §2.2

def _get_sbert():
    """Lazily load multilingual SBERT model."""
    global _sbert
    if _sbert is None:
        with _sbert_lock:
            if _sbert is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    _sbert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
                    logger.info("SBERT model loaded: paraphrase-multilingual-MiniLM-L12-v2")
                except Exception as exc:
                    logger.warning("sentence-transformers unavailable (%s) — sbert_coherence will return 0.65", exc)
                    _sbert = "unavailable"
    return _sbert if _sbert != "unavailable" else None


# ── Tokenizers ────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric tokens only."""
    return re.findall(r"\b[a-z]+\b", text.lower())


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END_RE.split(text.strip()) if s.strip()]


# ── Minimal stopword set ───────────────────────────────────────────────────────
_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "was", "are", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "shall",
    "it", "its", "this", "that", "these", "those", "there", "their",
    "they", "he", "she", "we", "you", "i", "me", "my", "your", "his",
    "her", "our", "what", "which", "who", "whom", "when", "where",
    "how", "why", "not", "no", "up", "out", "about", "just", "more",
    "also", "so", "if", "then", "than", "as", "into", "through",
})


# ── Feature functions ─────────────────────────────────────────────────────────

def compute_mattr(transcript: str, window_size: int = 50) -> float:
    """
    F06 lexical_mattr — Moving Average Type-Token Ratio.

    Length-independent vocabulary diversity (Framework §3.2, Change 1).
    Replaces plain TTR which inflates for longer responses.
    Window of 50 words is standard in L2 research literature.
    """
    words = _tokenize(transcript)
    words = [w for w in words if w not in _STOPWORDS and len(w) > 1]
    if not words:
        return 0.0
    if len(words) < window_size:
        return len(set(words)) / len(words)
    ttrs: list[float] = []
    for i in range(len(words) - window_size + 1):
        window = words[i:i + window_size]
        ttrs.append(len(set(window)) / window_size)
    return round(float(sum(ttrs) / len(ttrs)), 4)


def compute_rare_word_ratio(transcript: str) -> float:
    """
    F07 lexical_rare — Fraction of content words that are rare in English.

    Uses wordfreq (Brysbaert et al. 2014) to identify low-frequency words.
    Words with Zipf frequency < 4.0 (~1 in 10,000) are counted as rare.
    Returns 0.0 if wordfreq is unavailable.
    """
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        logger.warning("wordfreq not installed — rare_word_ratio returns 0.0")
        return 0.0
    words = _tokenize(transcript)
    content = [w for w in words if w not in _STOPWORDS and len(w) > 2]
    if not content:
        return 0.0
    rare = sum(1 for w in content if zipf_frequency(w, "en") < 4.0)
    return round(rare / len(content), 4)


def compute_discourse_features(transcript: str) -> dict[str, int | float]:
    """
    F08/F09 — Discourse connector count and Tier-1 count.

    Applies Indian English homophone correction before counting (Framework §1.2).
    The connector count target is scaled by 0.90 at scoring time (not here)
    to compensate for ASR transcription losses in Indian English.
    """
    corrected = _apply_homophone_corrections(transcript)
    words = _tokenize(corrected)
    full_text = corrected.lower()

    tier1_found: set[str] = set()
    tier2_found: set[str] = set()

    for phrase in CONNECTORS_TIER1:
        if phrase in full_text:
            tier1_found.add(phrase)

    for phrase in CONNECTORS_TIER2:
        if f" {phrase} " in f" {full_text} ":
            tier2_found.add(phrase)

    connector_count = len(tier1_found) + len(tier2_found)
    tier1_count = len(tier1_found)

    return {
        "discourse_connectors": connector_count,
        "discourse_tier1": tier1_count,
    }


def compute_sbert_coherence(transcript: str) -> float:
    """
    F13 sbert_coherence — Mean cosine similarity between adjacent sentences.

    Uses multilingual SBERT (paraphrase-multilingual-MiniLM-L12-v2) which
    maps Indian English spelling variations and ASR phonetic errors to the
    same semantic space (Framework §1.2).

    The raw similarity is returned here; the -0.06 offset is applied at the
    scoring layer (skills_scorer.py) as a calibration adjustment.

    Returns 0.65 (neutral Average band) if SBERT is unavailable.
    """
    sents = _sentences(transcript)
    if len(sents) < 2:
        return 0.65  # neutral fallback — too short to measure coherence

    model = _get_sbert()
    if model is None:
        return 0.65  # neutral fallback — SBERT unavailable

    try:
        import numpy as np
        embeddings = model.encode(sents, convert_to_numpy=True, normalize_embeddings=True)
        # Cosine similarity = dot product of L2-normalised vectors
        sims = [
            float(embeddings[i] @ embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]
        return round(float(sum(sims) / len(sims)), 4)
    except Exception as exc:
        logger.warning("sbert_coherence failed: %s", exc)
        return 0.65


def compute_ner_entity_density(transcript: str, duration_seconds: float) -> float:
    """
    F11 ner_entity_density — Named entities per minute.

    Uses SpaCy with Indian NER gazetteer. Entities matched via the Indian
    gazetteer receive a 1.3× multiplier to correct for NER model blind spots
    on Indian proper nouns (Framework §2.2, Axis 4).
    """
    minutes = duration_seconds / 60.0
    if minutes <= 0:
        return 0.0

    nlp = _get_nlp()
    if nlp is None:
        return 0.0

    try:
        doc = nlp(transcript[:50_000])  # SpaCy cap
        indian_labels = {"ORG", "GPE", "MONEY"}
        total_weight = 0.0
        for ent in doc.ents:
            # Check if it was matched by the Indian gazetteer
            is_indian = ent.label_ in indian_labels and any(
                ent.text.lower() in p["pattern"].lower()
                if isinstance(p.get("pattern"), str) else False
                for p in _INDIAN_PATTERNS
            )
            total_weight += 1.3 if is_indian else 1.0

        return round(total_weight / minutes, 4)
    except Exception as exc:
        logger.warning("ner_entity_density failed: %s", exc)
        return 0.0


def compute_narrative_arc(transcript: str) -> float:
    """
    Narrative arc completeness score (0–1) based on Labov & Waletzky (1967).

    Checks for the 4 core stages of a spoken narrative:
    Orientation → Complication → Action → Resolution.
    Each stage present contributes 0.25 to the score.
    """
    t = transcript.lower()
    stages_present = sum([
        any(m in t for m in _ORIENTATION_MARKERS),
        any(m in t for m in _COMPLICATION_MARKERS),
        any(m in t for m in _ACTION_MARKERS),
        any(m in t for m in _RESOLUTION_MARKERS),
    ])
    return round(stages_present / 4.0, 2)


def compute_metric_density(transcript: str, duration_seconds: float) -> float:
    """F12 metric_density — Specific data metrics per minute (numbers + quantifiers)."""
    matches = _METRIC_RE.findall(transcript)
    minutes = duration_seconds / 60.0
    return round(len(matches) / minutes, 4) if minutes > 0 else 0.0


def compute_filler_rate(transcript: str) -> float:
    """
    F02 fluency_filler_rate — Filler word + phrase ratio.

    V3 FIX: Counts both single-word fillers (via tokenized word list) and
    multi-word filler phrases (via substring matching on the raw lowercased
    transcript). Previously, multi-word entries like 'i mean' / 'you know'
    in FILLER_WORDS were silently ignored because _tokenize() splits into
    individual words.

    With AssemblyAI disfluencies=True, 'um', 'uh', 'hmm' appear verbatim
    in the transcript text, so single-word matching in FILLER_WORDS_SINGLE
    correctly captures them.
    """
    words = _tokenize(transcript)
    if not words:
        return 0.0

    total = len(words)

    # Single-word filler count (tokenized, lowercase)
    single_count = sum(1 for w in words if w in FILLER_WORDS_SINGLE)

    # Multi-word filler phrase count (substring match on raw lowercased text)
    lowered = transcript.lower()
    phrase_count = sum(lowered.count(phrase) for phrase in FILLER_PHRASES)

    return round((single_count + phrase_count) / max(total, 1), 4)


def compute_collaborative_ratio(transcript: str) -> float:
    """F14 collaborative_ratio — Team pronoun ratio (team / all pronouns)."""
    words = _tokenize(transcript)
    collab = sum(1 for w in words if w in COLLABORATIVE_PRONOUNS)
    indiv = sum(1 for w in words if w in INDIVIDUAL_PRONOUNS)
    total = collab + indiv
    if total == 0:
        return 0.5
    return round(collab / total, 4)


def compute_question_density(transcript: str, duration_seconds: float) -> float:
    """F15 question_density — Questions asked per minute."""
    sents = _sentences(transcript)
    q_count = sum(1 for s in sents if s.rstrip().endswith("?"))
    minutes = duration_seconds / 60.0
    return round(q_count / minutes, 4) if minutes > 0 else 0.0


def compute_empathetic_markers(transcript: str) -> float:
    """F16 empathetic_markers — Density of empathetic language (0–1, saturates at 5%)."""
    words = _tokenize(transcript)
    if not words:
        return 0.0
    count = sum(1 for w in words if w in EMPATHETIC_WORDS)
    density = count / len(words)
    return round(min(density / 0.05, 1.0), 4)


def compute_avg_sentence_length(transcript: str) -> float:
    """F20 avg_sentence_length — Average words per sentence."""
    sents = _sentences(transcript)
    if not sents:
        return 0.0
    words = _tokenize(transcript)
    return round(len(words) / len(sents), 2)


def compute_speech_rate_stats(
    word_timestamps: "list[WordTimestamp]",
    duration_seconds: float,
) -> dict[str, float]:
    """F01 fluency_wpm + wpm variability across 30-second segments."""
    if not word_timestamps:
        return {"wpm": 0.0, "wpm_std_dev": 0.0, "variability_score": 0.0}

    minutes = duration_seconds / 60.0
    wpm = len(word_timestamps) / minutes if minutes > 0 else 0.0

    segment_wpm_list: list[float] = []
    seg_start = 0.0
    seg_words = 0

    for wt in word_timestamps:
        seg_words += 1
        t = wt["end"]
        if t - seg_start >= 30.0:
            segment_wpm_list.append((seg_words / (t - seg_start)) * 60.0)
            seg_start = t
            seg_words = 0

    if len(segment_wpm_list) < 2:
        std_dev = 0.0
    else:
        mean = sum(segment_wpm_list) / len(segment_wpm_list)
        variance = sum((x - mean) ** 2 for x in segment_wpm_list) / len(segment_wpm_list)
        std_dev = variance ** 0.5

    variability = min(std_dev / 50.0, 1.0)

    return {
        "wpm": round(wpm, 1),
        "wpm_std_dev": round(std_dev, 1),
        "variability_score": round(variability, 3),
    }


# ── Convenience: extract all text features in one call ───────────────────────

def extract_all_text_features(
    transcript: str,
    duration_seconds: float,
    word_timestamps: "list[WordTimestamp]",
) -> dict[str, float]:
    """
    Return all text-based features as a flat dict.
    Keys map directly to RawFeatures field names.
    """
    rate = compute_speech_rate_stats(word_timestamps, duration_seconds)
    discourse = compute_discourse_features(transcript)

    total_words = len(_tokenize(transcript))

    return {
        # Fluency
        "speech_rate_wpm":              rate["wpm"],
        "speech_rate_variability":      rate["variability_score"],
        "filler_word_ratio":            compute_filler_rate(transcript),
        # Lexical
        "lexical_mattr":                compute_mattr(transcript),
        "lexical_rare_word_ratio":      compute_rare_word_ratio(transcript),
        "total_words":                  float(total_words),
        # Discourse
        "discourse_connectors":         float(discourse["discourse_connectors"]),
        "discourse_tier1":              float(discourse["discourse_tier1"]),
        "sbert_coherence":              compute_sbert_coherence(transcript),
        # Narrative & Evidence
        "ner_entity_density":           compute_ner_entity_density(transcript, duration_seconds),
        "metric_density":               compute_metric_density(transcript, duration_seconds),
        "narrative_arc_score":          compute_narrative_arc(transcript),
        # Style signals
        "collaborative_language_ratio": compute_collaborative_ratio(transcript),
        "question_density":             compute_question_density(transcript, duration_seconds),
        "empathetic_language_score":    compute_empathetic_markers(transcript),
        "avg_sentence_length":          compute_avg_sentence_length(transcript),
        # Legacy (kept for System B normalizer compatibility)
        "logical_connector_density":    float(discourse["discourse_connectors"]) / max(total_words, 1),
        "vocabulary_density":           compute_mattr(transcript),  # MATTR replaces TTR
    }
