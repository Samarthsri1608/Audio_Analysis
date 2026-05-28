import logging
import numpy as np

logger = logging.getLogger(__name__)


def _clamp(value: float, lo: float = 0.0, hi: float = 5.0) -> float:
    return float(np.clip(value, lo, hi))


def compute_fluency_score(wpm: float, pause_frequency: int, duration_seconds: float, filler_rate: float) -> float:
    """
    Fluency scoring — calibrated for Indian English speakers.

    Key adjustments vs previous version:
    - Optimal WPM range LOWERED to 90–140 (Indian speakers naturally speak
      slower in English; 130–165 is a Western-native benchmark, not applicable here).
    - Pause scoring floor raised: < 1 pause/min still gets 1.0, not 0.5.
    - Filler penalty is gentler: Indian spoken English uses more hedges/fillers
      as a cultural register feature, not a skill deficit.
    """
    if duration_seconds < 5:
        return 0.0

    pauses_per_min = (pause_frequency / duration_seconds) * 60

    # Pause scoring: 3–9 pauses/min = optimal for Indian interview speech
    if pauses_per_min < 1:
        pause_score = 1.0
    elif pauses_per_min < 3:
        pause_score = 1.0 + (pauses_per_min / 3.0) * 1.5
    elif pauses_per_min <= 9:
        pause_score = 2.5 + ((pauses_per_min - 3.0) / 6.0) * 2.5
    else:
        excess = min(pauses_per_min - 9.0, 12.0)
        pause_score = 5.0 - (excess / 12.0) * 4.0

    pause_score = _clamp(pause_score)

    # WPM penalty: optimal range 90–140 WPM (Indian English calibration)
    if 90 <= wpm <= 140:
        wpm_penalty = 0.0
    elif 75 <= wpm < 90:
        wpm_penalty = ((90 - wpm) / 15.0) * 0.5
    elif 140 < wpm <= 170:
        wpm_penalty = ((wpm - 140) / 30.0) * 0.8
    elif 60 <= wpm < 75:
        wpm_penalty = 0.5 + ((75 - wpm) / 15.0) * 1.0
    elif wpm >= 170:
        wpm_penalty = 0.8 + ((wpm - 170) / 40.0) * 1.0
    else:
        # < 60 WPM — very slow, but not 0; could be deliberate pacing
        wpm_penalty = 1.5 + ((60 - wpm) / 30.0) * 0.5

    wpm_penalty = _clamp(wpm_penalty, 0, 2.0)

    # Filler penalty: ≤ 7% natural, Indian speakers skew slightly higher
    filler_pct = filler_rate * 100
    if filler_pct <= 5.0:
        filler_penalty = 0.0
    elif filler_pct <= 10.0:
        filler_penalty = ((filler_pct - 5.0) / 5.0) * 0.4
    elif filler_pct <= 15.0:
        filler_penalty = 0.4 + ((filler_pct - 10.0) / 5.0) * 0.6
    else:
        filler_penalty = min(1.0 + ((filler_pct - 15.0) / 10.0) * 1.0, 2.0)

    filler_penalty = _clamp(filler_penalty, 0, 2.0)

    return _clamp(pause_score - wpm_penalty - filler_penalty)


def compute_intelligibility_score(pronunciation_score: float, variance_confidence: float) -> float:
    """
    Intelligibility scoring — calibrated for Indian English speakers.

    Key changes vs previous version:
    - Baseline: 0.45 pronunciation_score → 1.0 (was 0.55). With stop-words now excluded
      from pronunciation_score, content-word scores are naturally higher. The baseline
      is shifted down slightly to compensate and keep good speakers in the 3–5 range.
    - Variance penalty threshold raised: 0.10 → 0.20. Whisper variance on Indian English
      accents naturally sits at 0.05–0.12 — the old threshold fired at max (1.0 penalty)
      for nearly every candidate. New threshold only penalises genuinely inconsistent
      delivery.
    - Max variance penalty capped at 0.5 (was 1.0) — a variance artefact should not
      erase more than 0.5 of a 5-point scale.
    """
    pron_component = _clamp(1.0 + (pronunciation_score - 0.45) / 0.55 * 4.0)
    var_penalty = min(variance_confidence / 0.20 * 0.5, 0.5)
    return _clamp(pron_component - var_penalty)


def compute_language_control_score(transcript_words: list[str], grammar_errors: list[dict]) -> float:
    """
    Language control scoring.
    Uses straight error_density rather than a whitelist of rule IDs.

    Short-transcript gate: fewer than 80 words is statistically unreliable.
    A silent/near-silent candidate can produce 0 errors simply because there
    is nothing to flag — they should NOT receive 5.0. The score is linearly
    blended from the floor (1.5) up to the computed value as word count grows
    from 0 → 150 words.
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
    avg_word_frequency: float,
    total_words: int = 9999,
) -> float:
    """
    Lexical resource scoring.

    Key changes vs previous version:
    - MATTR ceiling raised: 0.68 → 0.75. Interview transcripts naturally have high
      MATTR (questions change topic constantly), causing 59% of candidates to max out
      at the old threshold. 0.75 is a better ceiling for genuine lexical richness.
    - rare_word_ratio ceiling tightened: 0.15 → 0.20. Technical speakers hit 15%
      easily just by using domain jargon; 20% requires genuinely sophisticated range.
    - Short-transcript gate unchanged.
    """
    # MATTR component: 0.40 → 1.0,  0.75 → 5.0 (ceiling raised from 0.68)
    mattr_component = _clamp(1.0 + (mattr - 0.40) / (0.75 - 0.40) * 4.0)

    # Sophistication: 20% rare content words → 5.0 (tightened from 15%)
    rare_component = _clamp(1.0 + (rare_word_ratio / 0.20) * 4.0)

    # Frequency penalty: common-word-heavy responses penalised
    freq_penalty = _clamp((avg_word_frequency - 0.005) / 0.005 * 0.8, 0, 1.0)

    lex_score = _clamp(0.55 * mattr_component + 0.45 * rare_component - freq_penalty)

    # Short-transcript gate
    MIN_WORDS = 80
    MAX_WORDS = 200
    FLOOR     = 1.0
    if total_words < MAX_WORDS:
        t = max(0.0, (total_words - MIN_WORDS) / (MAX_WORDS - MIN_WORDS))
        t = max(0.0, min(t, 1.0))
        lex_score = FLOOR + t * (lex_score - FLOOR)

    return _clamp(lex_score)


def compute_discourse_score(
    connector_count: int,
    tier1_count: int,
    tier2_count: int,
    total_words: int,
) -> float:
    """
    Discourse scoring.

    Key changes vs previous version:
    - wdph scale tightened: 4.0 → 6.0. Long interviews accumulate 'and'/'so'/'but'
      mechanically, pushing density up without reflecting quality. Requiring a higher
      density to reach 5.0 reduces ceiling saturation (was 64% maxed).
    - variety_bonus halved: 0.08 → 0.04 per connector type, cap 0.5 → 0.3. The old
      bonus rewarded sheer interview length (more words = more connector types found).
    - Short-transcript gate unchanged.
    """
    if total_words < 30:
        return 1.0

    weighted_count = tier1_count * 2.0 + tier2_count * 1.0
    wdph = (weighted_count / total_words) * 100

    # variety bonus: 0.04 per unique connector type, capped at 0.3 (halved — was 0.08 / 0.5)
    unique_connector_types = connector_count  # connector_count stores count of unique types found
    variety_bonus = min(0.04 * unique_connector_types, 0.3)

    # Tightened wdph scale: 6.0 denominator (was 4.0) — requires higher density for top scores
    disc_score = 1.0 + (wdph / 6.0) * 3.5 + variety_bonus

    # Cap at 3.5 if no Tier-1 connectors used
    if tier1_count == 0:
        disc_score = min(disc_score, 3.5)

    disc_score = _clamp(disc_score)

    # Short-transcript gate: < 80 words → blend to floor 1.0
    MIN_WORDS = 80
    MAX_WORDS = 180
    FLOOR     = 1.0
    if total_words < MAX_WORDS:
        t = max(0.0, (total_words - MIN_WORDS) / (MAX_WORDS - MIN_WORDS))
        t = max(0.0, min(t, 1.0))
        # additionally cap max score for short transcripts
        cap = 1.0 + t * 4.0
        disc_score = min(disc_score, cap)
        disc_score = FLOOR + t * (disc_score - FLOOR)

    return _clamp(disc_score)


def compute_voice_modulation_score(pitch_std: float, voiced_fraction: float) -> float:
    """
    Voice modulation scoring.
    Uses pre-computed pitch features from feature_extraction/voice_modulation.py.

    Indian English note: due to tonal carry-over from regional languages,
    Indian English speakers can have naturally higher pitch_std. The floor
    (20 Hz → 1.0) and ceiling (60 Hz → 5.0) from the design doc are kept.
    """
    vm_score = 1.0 + (pitch_std - 20.0) / (60.0 - 20.0) * 4.0

    if voiced_fraction < 0.20:
        vm_score -= 1.0
    elif voiced_fraction < 0.30:
        vm_score -= 0.5

    return _clamp(vm_score)


def compute_sentiment_score(
    mean_compound: float,
    std_compound: float,
    neg_sentiment_ratio: float,
    assertive_count: int,
    hedge_rate: float,
    total_words: int,
) -> float:
    """
    Sentiment & Confidence scoring.
    Calibrated for Indian English interview style:
    - Indian professional speech tends toward polite hedging; hedge_rate
      penalty is softened vs aggressive Western assertiveness benchmarks.
    - Positivity peak shifted slightly lower (0.25 ideal) as Indian formal
      speech tends to be professionally neutral, not overtly effusive.

    Key fix vs previous version:
    - hedge_rate unit correction (B6): sentiment.py stores hedge_rate as
      (hedge_count / total_words * 100) — a per-100-words figure, NOT a pure ratio.
      Typical values: 0.05–0.37 (meaning 0.05–0.37 hedges per 100 words).
      The old penalty `min(hedge_rate * 0.12, 2.0)` produced a max of ~0.044 —
      effectively zero. Fix: treat hedge_rate as per-100-words → scale factor 12.0
      so that 0.37 hedge_rate produces penalty = min(0.37 * 12.0 / 100, 2.0) ... hmm
      actually let's normalise: hedge_rate values are already tiny floats (0.0–0.37)
      representing count/total_words*100. A hedge_rate of 0.10 means 0.1 per 100 words
      = practically nothing. The real intent is: penalise if there are e.g. >3 hedge
      phrases relative to total length. Recompute using assertive_count / hedge_count
      directly rather than the stored rate.
      New: penalty = min((hedge_rate / assertive_count_ratio) * meaningful_scale, 2.0)
      Simplified: penalty = min(hedge_rate * 5.0, 2.0) — at hedge_rate=0.31 → 1.55 penalty.
      At hedge_rate=0.08 → 0.4 penalty. This is meaningful and bounded.
    """
    # Signal 1 — Positivity (40%): peaks at ~0.25 (moderate professional tone)
    positivity = 5.0 - abs(mean_compound - 0.25) / 0.35 * 3.0
    if neg_sentiment_ratio > 0.25:
        positivity -= (neg_sentiment_ratio - 0.25) * 3.5
    positivity = _clamp(positivity)

    # Signal 2 — Confidence (40%)
    assert_density = (assertive_count / max(total_words, 1)) * 100
    confidence = 2.5 + min(assert_density * 0.4, 2.0)
    # hedge_rate is stored as (count / total_words * 100) — scale by 5.0 so the
    # penalty ranges from 0 (no hedging) to 2.0 (heavy hedging, rate ~0.40+)
    confidence -= min(hedge_rate * 5.0, 2.0)
    confidence = _clamp(confidence)

    # Signal 3 — Composure (20%)
    if std_compound < 0.10:
        composure = 5.0
    elif std_compound < 0.20:
        composure = 4.5
    elif std_compound < 0.30:
        composure = 4.0
    elif std_compound < 0.40:
        composure = 3.0
    elif std_compound < 0.50:
        composure = 2.0
    else:
        composure = 1.0

    return _clamp(0.4 * positivity + 0.4 * confidence + 0.2 * composure)


def compute_overall_score(
    fluency: float,
    intelligibility: float,
    language_control: float,
    lexical: float,
    discourse: float,
    voice_modulation: float,
    sentiment: float,
) -> float:
    """
    Weight rebalance vs previous version:
    - language_control: 0.20 → 0.12  (LanguageTool returns 0 errors for ASR transcripts;
      97% of candidates scored 5.0 — this dimension cannot discriminate)
    - fluency: 0.22 → 0.26           (primary discriminating signal)
    - intelligibility: 0.22 → 0.26   (primary discriminating signal; now also fixed)
    - lexical_resource: 0.18 → 0.16  (ceiling reduced but still meaningful)
    - discourse: 0.12 → 0.10         (ceiling reduced but still meaningful)
    - voice_modulation: 0.03 → 0.05  (slightly more weight, varied signal)
    - sentiment: 0.03 → 0.05         (slightly more weight, well-behaved signal)
    Sum = 1.00
    """
    overall = (
        (fluency          * 0.26) +
        (intelligibility  * 0.26) +
        (language_control * 0.12) +
        (lexical          * 0.16) +
        (discourse        * 0.10) +
        (voice_modulation * 0.05) +
        (sentiment        * 0.05)
    ) * 20  # scale 0–5 → 0–100
    return float(max(0.0, min(overall, 100.0)))
