import logging
import numpy as np
import re

logger = logging.getLogger(__name__)


def _clamp(value: float, lo: float = 0.0, hi: float = 5.0) -> float:
    return float(np.clip(value, lo, hi))


def _band_score(value: float, thresholds: tuple, higher_is_better: bool = True) -> float:
    """
    Map a raw feature value onto a 0.0–5.0 scale using percentile-band boundaries
    derived from the statistical analysis of 86 interview recordings.

    Band → score centre mapping:
        Poor          (< p20)   → ~0.5
        Below Average (p20–p40) → ~1.5
        Average       (p40–p60) → ~2.5
        Good          (p60–p80) → ~3.5
        Excellent     (> p80)   → ~4.5

    Linear interpolation is applied *within* each band for smooth, continuous
    scoring.  The Excellent band uses a soft extrapolation capped at 5.0.

    Parameters
    ----------
    value : float
        Observed feature value.
    thresholds : tuple of four floats
        (p20, p40, p60, p80) percentile cutpoints in ascending order.
    higher_is_better : bool
        When False the value is mirrored around the midpoint of thresholds
        so that lower values earn higher scores (e.g. filler_rate, pause_dur).
    """
    p20, p40, p60, p80 = thresholds

    if not higher_is_better:
        # Mirror value around the midpoint so that low raw values map to high values
        midpoint = (p20 + p80) / 2.0
        value = 2.0 * midpoint - value

    # Band boundaries → score boundaries (linear within each band)
    bands = [
        (float("-inf"), p20, 0.0, 1.0),   # Poor:          score 0 → 1
        (p20,           p40, 1.0, 2.0),   # Below Average: score 1 → 2
        (p40,           p60, 2.0, 3.0),   # Average:       score 2 → 3
        (p60,           p80, 3.0, 4.0),   # Good:          score 3 → 4
        (p80, float("inf"), 4.0, 5.0),    # Excellent:     score 4 → 5
    ]

    for lo, hi, score_lo, score_hi in bands:
        if lo <= value < hi:
            if lo == float("-inf"):
                # Extrapolate downward using the next band's width
                band_width = p40 - p20 if (p40 - p20) > 0 else 1.0
                t = max(0.0, 1.0 - (hi - value) / band_width)
                return _clamp(score_lo + t * (score_hi - score_lo))
            if hi == float("inf"):
                # Soft extrapolation above p80: one band-width past p80 → 5.0
                band_width = p80 - p60 if (p80 - p60) > 0 else 1.0
                t = min((value - lo) / band_width, 1.0)
                return _clamp(score_lo + t * (score_hi - score_lo))
            t = (value - lo) / (hi - lo)
            return score_lo + t * (score_hi - score_lo)

    return 5.0  # value == p80 exactly falls through; cap at top


def compute_fluency_score(
    wpm: float,
    pause_frequency: int,
    duration_seconds: float,
    filler_rate: float,
    pause_dur: float = 0.0,
) -> float:
    """
    Fluency scoring — data-driven thresholds from 86-interview analysis.

    Features and percentile thresholds (non-normal distribution):
        wpm          (higher is better) — p20=44.7, p40=55.0, p60=67.3, p80=93.3
        filler_rate  (lower is better)  — p20=0.058, p40=0.043, p60=0.025, p80=0.018
        pause_dur    (lower is better)  — p20=8.8,  p40=5.9,  p60=4.4,  p80=2.8

    Weights: wpm=0.50, filler_rate=0.30, pause_dur=0.20

    Note: ``pause_frequency`` and ``duration_seconds`` are retained in the
    signature for backwards compatibility but are not used when ``pause_dur``
    is provided.  If ``pause_dur`` is not supplied (default 0.0) the score
    contribution from that component defaults to 2.5 (Average).
    """
    if duration_seconds < 5:
        return 0.0

    wpm_score = _band_score(
        wpm,
        thresholds=(44.7, 55.0, 67.3, 93.3),
        higher_is_better=True,
    )

    filler_score = _band_score(
        filler_rate,
        thresholds=(0.018, 0.025, 0.043, 0.058),  # ascending order of original percentiles
        higher_is_better=False,
    )

    # pause_dur: total pause duration in seconds (lower is better)
    if pause_dur > 0.0:
        pause_score = _band_score(
            pause_dur,
            thresholds=(2.8, 4.4, 5.9, 8.8),  # original p20..p80 ascending
            higher_is_better=False,
        )
    else:
        # Fallback when pause_dur not provided: derive rough estimate or use Average
        pause_score = 2.5

    return _clamp(0.50 * wpm_score + 0.30 * filler_score + 0.20 * pause_score)


def compute_intelligibility_score(
    pronunciation_score: float = 0.0,
    variance_confidence: float = 0.0,
    mean_confidence: float = 0.0,
) -> float:
    """
    Intelligibility scoring — z-score bands from 86-interview analysis.

    Primary feature: ``mean_confidence`` (Whisper per-token confidence mean).
        Distribution: Normal  mean=0.741, std=0.079
        Band thresholds derived at ±0.5σ / ±1.0σ boundaries:
            < 0.679  → Poor
            0.679–0.722 → Below Average
            0.722–0.759 → Average
            0.759–0.799 → Good
            > 0.799  → Excellent

    Dropped features (statistical analysis):
        pronunciation_score  — r=0.979 correlation with mean_confidence (redundant)
        variance_confidence  — fully captured by mean_confidence

    Backwards compatibility: if mean_confidence is not supplied (default 0.0)
    the function falls back to using pronunciation_score so that call-sites
    that have not yet been updated continue to return a plausible value.
    """
    if mean_confidence == 0.0 and pronunciation_score > 0.0:
        # Legacy fallback: rough mapping from old pronunciation_score range
        mean_confidence = 0.600 + pronunciation_score * 0.200  # approximate

    score = _band_score(
        mean_confidence,
        thresholds=(0.679, 0.722, 0.759, 0.799),
        higher_is_better=True,
    )
    return _clamp(score)


def compute_language_control_score(transcript_words: list[str], grammar_errors: list[dict]) -> float:
    """
    Language control scoring.
    Uses straight error_density rather than a whitelist of rule IDs.

    Short-transcript gate: fewer than 80 words is statistically unreliable.
    A silent/near-silent candidate can produce 0 errors simply because there
    is nothing to flag — they should NOT receive 5.0. The score is linearly
    blended from the floor (1.5) up to the computed value as word count grows
    from 0 → 150 words.

    NOTE: This function is intentionally left unchanged. The dimension is
    currently mocked (all-zeros in the pipeline) and is excluded from the
    overall score weights (weight = 0.00) pending a grammar-checker fix.
    """
    total_words = max(len(transcript_words), 1)
    error_density = len(grammar_errors) / total_words

    # 0% errors → 5.0,  ≥ 10% errors → 1.0
    lc_score = 5.0 - (error_density / 0.10) * 4.0

    # Fragment penalty (avg sentence < 5 words)
    sentences = " ".join(transcript_words).split(".")
    sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
    if sentence_lengths:
        avg_len = sum(sentence_lengths) / len(sentence_lengths)
        if avg_len < 5:
            lc_score -= (5 - avg_len) * 0.25

    lc_score = _clamp(lc_score)

    # Short-transcript gate: blend from floor → computed as words → 150
    MIN_WORDS = 80
    MAX_WORDS = 150
    FLOOR     = 1.5
    if total_words < MAX_WORDS:
        t = max(0.0, (total_words - MIN_WORDS) / (MAX_WORDS - MIN_WORDS))
        t = max(0.0, min(t, 1.0))
        lc_score = FLOOR + t * (lc_score - FLOOR)

    return _clamp(lc_score)


def compute_lexical_resource_score(
    mattr: float,
    rare_word_ratio: float,
    avg_word_frequency: float = 0.0,
    total_words: int = 9999,
) -> float:
    """
    Lexical resource scoring — z-score bands from 86-interview analysis.

    Features and thresholds (Normal distributions):

        mattr (Moving Average Type-Token Ratio, higher is better):
            mean=0.762, std=0.040
            < 0.732          → Poor
            0.732–0.751      → Below Average
            0.751–0.779      → Average
            0.779–0.794      → Good
            > 0.794          → Excellent

        rare_word_ratio (higher is better):
            mean=0.141, std=0.041
            < 0.101          → Poor
            0.101–0.132      → Below Average
            0.132–0.155      → Average
            0.155–0.173      → Good
            > 0.173          → Excellent

    Weights: mattr=0.65, rare_word_ratio=0.35

    Short-transcript gate: if total_words < 100, scale score down
    proportionally (sparse transcripts produce unreliable lexical metrics).
    """
    mattr_score = _band_score(
        mattr,
        thresholds=(0.732, 0.751, 0.779, 0.794),
        higher_is_better=True,
    )

    rare_score = _band_score(
        rare_word_ratio,
        thresholds=(0.101, 0.132, 0.155, 0.173),
        higher_is_better=True,
    )

    lex_score = _clamp(0.65 * mattr_score + 0.35 * rare_score)

    # Short-transcript gate: proportional scale-down below 100 words
    if total_words < 100:
        scale = max(0.0, total_words / 100.0)
        lex_score = lex_score * scale

    return _clamp(lex_score)


def compute_discourse_score(
    connector_count: int,
    tier1_count: int,
    tier2_count: int,
    total_words: int,
) -> float:
    """
    Discourse / cohesion scoring — mixed distribution thresholds from analysis.

    Features and thresholds:

        connector_count (total connectives used, Normal):
            mean=11.4, std=2.4
            < 9     → Poor
            9–11    → Below Average
            11–12   → Average
            12–13   → Good
            > 13    → Excellent

        tier1_count (high-quality connectors, Non-normal):
            0       → Poor
            1       → Below Average
            2–3     → Average
            3–4     → Good
            ≥ 5     → Excellent

    Weights: connector_count=0.50, tier1_count=0.50

    Hard cap: score is capped at 3.5 if tier1_count == 0
    (no high-quality connectors used, regardless of raw count).
    """
    if total_words < 30:
        return 1.0

    conn_score = _band_score(
        float(connector_count),
        thresholds=(9.0, 11.0, 12.0, 13.0),
        higher_is_better=True,
    )

    # tier1_count uses fixed ordinal thresholds (non-normal, sparse integer)
    t1 = tier1_count
    if t1 == 0:
        tier1_score = 0.5   # Poor centre
    elif t1 == 1:
        tier1_score = 1.5   # Below Average centre
    elif t1 <= 3:
        tier1_score = 2.0 + (t1 - 2) / 1.0 * 1.0  # 2→2.0, 3→3.0
    elif t1 == 4:
        tier1_score = 3.5   # Good centre
    else:
        tier1_score = min(4.0 + (t1 - 5) * 0.2, 5.0)  # Excellent, soft cap

    tier1_score = _clamp(tier1_score)

    disc_score = _clamp(0.50 * conn_score + 0.50 * tier1_score)

    # Hard cap if no Tier-1 connectors used
    if tier1_count == 0:
        disc_score = min(disc_score, 3.5)

    return _clamp(disc_score)


def compute_sentiment_score(
    mean_compound: float,
    std_compound: float,
    neg_sentiment_ratio: float,
    assertive_count: int,
    hedge_rate: float,
    total_words: int,
) -> float:
    """
    Sentiment & confidence scoring — percentile bands from 86-interview analysis.

    Features and thresholds (Non-normal):

        mean_compound (higher is better):
            p20=0.105, p40=0.147, p60=0.209, p80=0.278

        assertive_count (higher is better):
            p20=2, p40=3, p60=5, p80=9

    Weights: mean_compound=0.60, assertive_count=0.40

    Penalty: if hedge_rate > 0.303 (p80 of hedge_rate distribution),
             subtract 0.5 from the final score.

    Note: ``std_compound`` and ``neg_sentiment_ratio`` are kept in the
    signature for backwards compatibility but are no longer used directly;
    the percentile bands on mean_compound already capture distributional spread.
    """
    compound_score = _band_score(
        mean_compound,
        thresholds=(0.105, 0.147, 0.209, 0.278),
        higher_is_better=True,
    )

    assert_score = _band_score(
        float(assertive_count),
        thresholds=(2.0, 3.0, 5.0, 9.0),
        higher_is_better=True,
    )

    sent_score = _clamp(0.60 * compound_score + 0.40 * assert_score)

    # Hedge penalty: excessive hedging above p80 (0.303) → −0.5
    if hedge_rate > 0.303:
        sent_score = max(0.0, sent_score - 0.5)

    return _clamp(sent_score)


def compute_voice_modulation_score(pitch_std: float, voiced_fraction: float) -> float:
    """
    Voice modulation scoring — data-driven thresholds from 86-interview analysis.

    Features and thresholds:

        pitch_std (Hz, higher is better — Normal):
            mean=50.6, std=9.3
            < 42.3          → Poor
            42.3–47.3       → Below Average
            47.3–51.6       → Average
            51.6–59.1       → Good
            > 59.1          → Excellent

        voiced_fraction (higher is better — Non-normal):
            p20=0.170, p40=0.216, p60=0.264, p80=0.345

    Weights: pitch_std=0.70, voiced_fraction=0.30

    Note: pitch_mean is intentionally NOT used — it is gender-biased and does
    not contribute meaningfully to perceived modulation quality.
    """
    pitch_score = _band_score(
        pitch_std,
        thresholds=(42.3, 47.3, 51.6, 59.1),
        higher_is_better=True,
    )

    voiced_score = _band_score(
        voiced_fraction,
        thresholds=(0.170, 0.216, 0.264, 0.345),
        higher_is_better=True,
    )

    return _clamp(0.70 * pitch_score + 0.30 * voiced_score)


def compute_collaborative_tone_score(
    mean_compound: float,
    text: str,
) -> float:
    """
    Collaborative tone scoring — combines compound sentiment and team pronoun ratio.
    """
    compound_score = _band_score(
        mean_compound,
        thresholds=(0.105, 0.147, 0.209, 0.278),
        higher_is_better=True,
    )
    
    text_lower = text.lower()
    team_words = ["we", "our", "us", "ours", "ourselves"]
    self_words = ["i", "my", "me", "mine", "myself"]
    
    team_count = sum(len(re.findall(rf"\b{w}\b", text_lower)) for w in team_words)
    self_count = sum(len(re.findall(rf"\b{w}\b", text_lower)) for w in self_words)
    
    total_pronouns = team_count + self_count
    if total_pronouns > 0:
        team_ratio = team_count / total_pronouns
        team_score = 1.0 + min(team_ratio / 0.5, 1.0) * 4.0
    else:
        team_score = 2.5
        
    return _clamp(0.60 * compound_score + 0.40 * team_score)


def compute_overall_score(
    fluency: float = 0.0,
    intelligibility: float = 0.0,
    language_control: float = 0.0,
    lexical: float = 0.0,
    discourse: float = 0.0,
    voice_modulation: float = 0.0,
    sentiment: float = 0.0,
    # New axes names:
    logical_cohesion: float = None,
    delivery_fluency: float = None,
    pronunciation_clarity: float = None,
    vocal_dynamism: float = None,
    collaborative_tone: float = None,
    lexical_precision: float = None,
) -> float:
    """
    Calculate the overall score.
    If the 6-Axis fingerprint dimensions are provided, returns their simple average scaled to 100.
    Otherwise, falls back to the legacy weighted overall average.
    """
    new_axes = [
        logical_cohesion,
        delivery_fluency,
        pronunciation_clarity,
        vocal_dynamism,
        collaborative_tone,
        lexical_precision
    ]
    if all(x is not None for x in new_axes):
        overall = np.mean(new_axes) * 20.0
        return float(max(0.0, min(overall, 100.0)))

    # Legacy fallback
    overall = (
        (fluency          * 0.20) +
        (intelligibility  * 0.20) +
        (language_control * 0.00) +
        (lexical          * 0.15) +
        (discourse        * 0.15) +
        (voice_modulation * 0.15) +
        (sentiment        * 0.15)
    ) * 20  # scale 0–5 → 0–100
    return float(max(0.0, min(overall, 100.0)))
