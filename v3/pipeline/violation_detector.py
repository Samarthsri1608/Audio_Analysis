"""
pipeline/violation_detector.py — Academic Violation Detection Engine (v3).

Architecture:
  Track A — Self-relative deviation (leave-one-out robust z-score vs. candidate's
             own baseline within the interview). Requires >=4 evaluable answers.
  Track C — Mechanism-level naturalness (works from interview #1, no baseline needed).

Flag logic (soft, not hard-threshold):
  composite_score is the primary gate.
  response_latency contributes via a sigmoid transform into the composite rather
  than a boolean cutoff — eliminates coin-flip cliff behaviour.
  flag_for_review fires when composite_score > COMPOSITE_THRESHOLD AND at least
  one of the two high-confidence signals (latency or flatness) exceeds its
  provisional threshold by a meaningful margin (hysteresis band).

BUGS FIXED IN THIS VERSION (v3):
  BUG 1 — Zero used as valid feature value ("insufficient data" masquerade).
    - Per-feature minimum word-count / duration thresholds defined.
    - Features below their minimum return None, not 0.0.
    - None values excluded from baseline pools AND from composite score calculation.
    - feature_availability map added to every evaluable evidence payload.

  BUG 2 — Hard single-feature gate causing coin-flip cliff on response_latency.
    - response_latency_sec now enters composite via sigmoid transform (continuous).
    - flag_for_review requires signal to exceed threshold by HYSTERESIS_BAND,
      not epsilon-close.

  BUG 3 — Flagged questions (Q3/Q8) with all Track A features at 0.0.
    - Consequence of BUG 1; resolved by BUG 1 fix. Non-computable features now
      appear as null in track_a_z_scores, not 0.0.

  BUG 4 — cross_question_naturalness_flatness and latency_variance always 0.0.
    - Replaced non-contributing z-score approach with provisional threshold on
      raw value. Contributes fixed PROVISIONAL_FLATNESS_SCORE /
      PROVISIONAL_LATENCY_VAR_SCORE points when raw value exceeds threshold.
    - track_c_provisional_flag boolean added to each evaluable payload.
    - Clearly labelled UNCALIBRATED — replace when population baseline available.

  BUG 5 — discourse_organization pegged at clip ceiling (sparse-count MAD=0).
    - Replaced robust z-score for discourse_organization with percentile rank
      within the candidate's own answers (rank / n, Fraction 0-1, then scaled).
    - Percentile rank is appropriate for sparse zero-heavy count distributions.

  BUG 6 — sbert_coherence returning 0.0 (silent fallback from text_features.py).
    - Distinguishes SBERT-unavailable (0.65 neutral) from SBERT-failure (None).
    - compute_sbert_coherence_strict() added to text_features.py side via the
      extraction call; here we treat 0.65 as the neutral fallback, not as a
      genuine measurement, and track when sbert_coherence is actually computable.
    - Explicit per-record logging added if coherence falls back to 0.65/0.0.
"""
from __future__ import annotations

import logging
import math
import statistics
from typing import Any

logger = logging.getLogger("v3.violation_detector")

# ---------------------------------------------------------------------------
# UNCALIBRATED -- placeholder pending validation.
# Do NOT tune these values until labeled data from the validation phase exists.
# See academic_violation_framework.md Section 6 for the required phases.
# ---------------------------------------------------------------------------

# Primary composite score gate for flag_for_review.
COMPOSITE_THRESHOLD: float = 6.0              # UNCALIBRATED placeholder

# Hard-signal threshold for latency / flatness co-trigger.
# A signal must exceed this by at least HYSTERESIS_BAND before it can single-
# handedly influence the flag decision (prevents epsilon-cliff flips).
HARD_SIGNAL_THRESHOLD: float = 2.0            # UNCALIBRATED placeholder
HYSTERESIS_BAND: float = 0.25                 # UNCALIBRATED placeholder

# Sigmoid scale factor for latency contribution to composite.
# Transforms the latency z-score through 1/(1+exp(-k*z)) so that large z values
# contribute proportionally rather than as a boolean gate.
LATENCY_SIGMOID_SCALE: float = 1.2           # UNCALIBRATED placeholder

# Minimum evaluable answers for Track A.
TRACK_A_MIN_EVALUABLE_ANSWERS: int = 4

# Eligibility gate: answer-level thresholds.
ELIGIBILITY_MIN_DURATION_S: float = 3.0
ELIGIBILITY_MIN_WORDS: int = 10
ELIGIBILITY_MIN_CONFIDENCE: float = 0.40

# Feature-specific minimum word counts for reliable extraction.
# An evaluable answer below these counts has its feature set to None (unavailable).
FEATURE_MIN_WORDS: dict[str, int] = {
    "lexical_mattr":          50,   # MATTR window size; below this it degrades to plain TTR
    "sbert_coherence":        20,   # need >=2 sentences; short answers can't be meaningfully compared
    "discourse_organization": 15,   # too few words = connector count is always 0 or 1
}
# Minimum duration (seconds) for a feature to be reliably extracted.
FEATURE_MIN_DURATION_S: dict[str, float] = {
    "intra_answer_pace_variance": 30.0,   # need >=2 x 15s buckets
    "pause_regularity":           10.0,   # need >= 2 pauses > 300ms
}

# SBERT neutral fallback sentinel: when sbert returns exactly this value it means
# "model unavailable / text too short", not a real similarity score.
SBERT_NEUTRAL_FALLBACK: float = 0.65

# Z-score hard clip (blowup prevention).
Z_SCORE_CLIP: float = 10.0

# Feature-scaled MAD floors (prevent divide-near-zero).
_MAD_FLOORS: dict[str, float] = {
    "speech_rate_wpm":   1.0,
    "filler_word_ratio": 0.002,
    "lexical_mattr":     0.005,
    "sbert_coherence":   0.005,
}
_DEFAULT_MAD_FLOOR: float = 1e-3

# Track A feature weights.
TRACK_A_WEIGHTS: dict[str, float] = {
    "speech_rate_wpm":        0.60,
    "filler_word_ratio":      0.60,
    "lexical_mattr":          0.60,
    "discourse_organization": 0.50,
    "sbert_coherence":        0.50,
}

# Track C feature weights (excluding latency — it enters via sigmoid now).
TRACK_C_WEIGHTS: dict[str, float] = {
    "intra_answer_pace_variance": 1.0,
    "pause_regularity":           1.5,
    "pitch_variance_ratio":       1.5,
}

# Sigmoid weight for the latency contribution to composite.
LATENCY_COMPOSITE_WEIGHT: float = 2.0        # UNCALIBRATED placeholder

# Provisional thresholds for features that lack a population baseline yet.
# These contribute fixed bonus points to composite when raw value is suspicious.
# STOPGAP until Track B (population-level) data exists — see Bug 4 comment.
PROVISIONAL_FLATNESS_THRESHOLD: float = 0.05   # UNCALIBRATED — raw flatness below this = suspicious
PROVISIONAL_FLATNESS_SCORE: float = 3.0        # UNCALIBRATED bonus points added to composite
PROVISIONAL_LATENCY_VAR_THRESHOLD: float = 0.8  # UNCALIBRATED — raw latency std below this = suspicious (seconds)
PROVISIONAL_LATENCY_VAR_SCORE: float = 2.5     # UNCALIBRATED bonus points added to composite


# ---------------------------------------------------------------------------
# Eligibility gate
# ---------------------------------------------------------------------------

def classify_eligibility(q: dict[str, Any]) -> tuple[bool, str | None]:
    """
    Return (is_evaluable, reason_code_or_None).

    Reason codes: "skipped", "no_speech_detected", "too_short",
                  "low_word_count", "low_asr_confidence".
    """
    if q.get("skipped") or q.get("off_camera"):
        return False, "skipped"

    text: str = (q.get("text") or "").strip()
    if not text:
        return False, "no_speech_detected"

    duration_s: float = float(q.get("duration_s", 0.0))
    if duration_s < ELIGIBILITY_MIN_DURATION_S:
        return False, "too_short"

    word_timestamps = q.get("word_timestamps") or []
    word_count = len(word_timestamps)
    if word_count == 0:
        word_count = len(text.split())
    if word_count < ELIGIBILITY_MIN_WORDS:
        return False, "low_word_count"

    confidence: float = float(q.get("mean_confidence", 1.0))
    if confidence < ELIGIBILITY_MIN_CONFIDENCE:
        return False, "low_asr_confidence"

    return True, None


def _word_count(q: dict[str, Any]) -> int:
    """Approximate word count for a question result dict."""
    wts = q.get("word_timestamps") or []
    if wts:
        return len(wts)
    return len((q.get("text") or "").split())


# ---------------------------------------------------------------------------
# Feature availability: returns None for features that can't be reliably computed
# ---------------------------------------------------------------------------

def _feature_availability(q: dict[str, Any]) -> dict[str, bool]:
    """
    Return a map of feature_name -> bool indicating whether each Track A
    feature is reliably computable for this question.

    A feature is NOT available if the question's word count or duration falls
    below the feature-specific minimum defined in FEATURE_MIN_WORDS /
    FEATURE_MIN_DURATION_S. When unavailable, the feature value should be
    treated as None / not-computed, NOT as 0.0.
    """
    wc = _word_count(q)
    dur = float(q.get("duration_s", 0.0))
    return {
        "lexical_mattr":          wc >= FEATURE_MIN_WORDS["lexical_mattr"],
        "sbert_coherence":        wc >= FEATURE_MIN_WORDS["sbert_coherence"],
        "discourse_organization": wc >= FEATURE_MIN_WORDS["discourse_organization"],
        "speech_rate_wpm":        wc >= 5,        # very lenient; WPM is always meaningful
        "filler_word_ratio":      wc >= 10,        # matches eligibility min
        "intra_answer_pace_variance": dur >= FEATURE_MIN_DURATION_S["intra_answer_pace_variance"],
        "pause_regularity":        dur >= FEATURE_MIN_DURATION_S["pause_regularity"],
    }


# ---------------------------------------------------------------------------
# Nullable feature extraction helpers
# ---------------------------------------------------------------------------

def _get_mattr(q: dict[str, Any], avail: dict[str, bool]) -> float | None:
    """Return lexical_mattr or None if below minimum word count."""
    if not avail.get("lexical_mattr", False):
        return None
    val = q.get("text_feats", {}).get("lexical_mattr")
    if val is None:
        return None
    fv = float(val)
    return None if fv == 0.0 else fv


def _get_sbert(q: dict[str, Any], avail: dict[str, bool]) -> float | None:
    """
    Return sbert_coherence or None if:
      - Below minimum word count
      - Value equals the SBERT_NEUTRAL_FALLBACK sentinel (0.65 = model unavailable)
      - Value is exactly 0.0 (extraction failure)

    BUG 6 FIX: distinguishes the 0.65 neutral fallback (SBERT unavailable)
    from a genuine coherence score. When the model is unavailable, we cannot
    use this feature — return None rather than treating 0.65 as real data.
    """
    if not avail.get("sbert_coherence", False):
        return None
    raw = q.get("text_feats", {}).get("sbert_coherence")
    if raw is None:
        return None
    fv = float(raw)
    if fv == 0.0:
        logger.warning(
            "Q%s: sbert_coherence==0.0 — extraction failure (SBERT call returned 0). "
            "Treating as unavailable. Check text_features.compute_sbert_coherence() "
            "for silent exception / model load failure.",
            q.get("q_no"),
        )
        return None
    if abs(fv - SBERT_NEUTRAL_FALLBACK) < 1e-6:
        # This is the neutral fallback sentinel, not a real measurement.
        # Log once per question so broken SBERT paths are visible.
        logger.warning(
            "Q%s: sbert_coherence==%.4f (SBERT neutral sentinel) — SBERT model "
            "is likely unavailable or text had < 2 sentences. Treating as None. "
            "Verify sentence_transformers is installed and the model loaded correctly.",
            q.get("q_no"), fv,
        )
        return None
    return fv


def _get_discourse_org(q: dict[str, Any], avail: dict[str, bool]) -> float | None:
    """Return discourse_organization composite or None if below minimum."""
    if not avail.get("discourse_organization", False):
        return None
    tf = q.get("text_feats", {})
    return float(tf.get("discourse_connectors", 0.0)) + 1.5 * float(tf.get("discourse_tier1", 0.0))


# ---------------------------------------------------------------------------
# Robust z-score (signed, direction-clipped)
# ---------------------------------------------------------------------------

def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _mad(values: list[float], med: float) -> float:
    return _median([abs(v - med) for v in values])


def _robust_z(
    value: float,
    baseline: list[float],
    feature_name: str = "",
    suspicious_direction: str = "increase",
) -> float:
    """
    Directional robust z-score with MAD floor and hard clip.
    Returns 0.0 if baseline has fewer than 2 elements.
    Suspicious direction clipping: only the suspicious side contributes.
    """
    if len(baseline) < 2:
        return 0.0
    med = _median(baseline)
    mad = _mad(baseline, med)
    floor = _MAD_FLOORS.get(feature_name, _DEFAULT_MAD_FLOOR)
    if mad < floor:
        mad = floor
    raw_z = 0.6745 * (value - med) / mad
    raw_z = max(-Z_SCORE_CLIP, min(Z_SCORE_CLIP, raw_z))
    if suspicious_direction == "increase":
        return max(0.0, raw_z)
    else:
        return max(0.0, -raw_z)


def _percentile_rank(value: float, distribution: list[float]) -> float:
    """
    Percentile rank of value within distribution (0.0 = lowest, 1.0 = highest).
    Uses interpolation: rank = (number of values strictly less than value) / (n-1).
    Returns 0.5 for distributions with < 2 elements.

    Used for sparse-count features like discourse_organization (BUG 5 fix).
    """
    if len(distribution) < 2:
        return 0.5
    n = len(distribution)
    below = sum(1 for v in distribution if v < value)
    equal = sum(1 for v in distribution if v == value)
    # Midpoint convention for ties
    rank = (below + 0.5 * equal) / n
    return round(rank, 4)


def _sigmoid(x: float, k: float = 1.0) -> float:
    """Standard sigmoid 1 / (1 + exp(-k*x)). Clipped to avoid overflow."""
    x_clipped = max(-20.0, min(20.0, k * x))
    return 1.0 / (1.0 + math.exp(-x_clipped))


# ---------------------------------------------------------------------------
# Intra-answer pace variance (returns None when answer too short)
# ---------------------------------------------------------------------------

def _intra_answer_pace_variance(
    word_timestamps: list[dict[str, Any]],
    duration_s: float,
    q_no: Any = None,
) -> float | None:
    """
    CV (std/mean) of local WPM across 15-second buckets.
    Returns None if fewer than 2 buckets (answer too short to be meaningful).
    LOW value = suspiciously even pacing.
    """
    if not word_timestamps or duration_s < FEATURE_MIN_DURATION_S["intra_answer_pace_variance"]:
        return None

    buckets: dict[int, int] = {}
    for wt in word_timestamps:
        idx = int(float(wt["start"]) // 15.0)
        buckets[idx] = buckets.get(idx, 0) + 1

    if len(buckets) < 2:
        logger.warning(
            "Q%s: intra_answer_pace_variance: only %d 15s bucket(s) despite "
            "duration %.1fs -- returning None.",
            q_no, len(buckets), duration_s,
        )
        return None

    rates = [(count / 15.0) * 60.0 for count in buckets.values()]
    mean_rate = sum(rates) / len(rates)
    if mean_rate <= 0:
        return None
    variance = sum((r - mean_rate) ** 2 for r in rates) / len(rates)
    return round(math.sqrt(variance) / mean_rate, 4)


# ---------------------------------------------------------------------------
# Pause regularity (returns None when answer too short)
# ---------------------------------------------------------------------------

def _pause_regularity(
    word_timestamps: list[dict[str, Any]],
    duration_s: float = 0.0,
    q_no: Any = None,
) -> float | None:
    """
    CV of inter-pause gaps (>300ms only). Returns None if fewer than 2 qualifying
    pauses, or if answer is below FEATURE_MIN_DURATION_S for this feature.
    LOW value = mechanically even pausing.
    """
    if duration_s < FEATURE_MIN_DURATION_S["pause_regularity"] or len(word_timestamps) < 3:
        return None

    pauses = [
        float(word_timestamps[i]["start"]) - float(word_timestamps[i - 1]["end"])
        for i in range(1, len(word_timestamps))
        if float(word_timestamps[i]["start"]) - float(word_timestamps[i - 1]["end"]) > 0.3
    ]

    if len(pauses) < 2:
        logger.warning(
            "Q%s: pause_regularity: only %d qualifying pauses (need >=2) -- returning None.",
            q_no, len(pauses),
        )
        return None

    mean_pause = sum(pauses) / len(pauses)
    if mean_pause <= 0:
        return None
    variance = sum((p - mean_pause) ** 2 for p in pauses) / len(pauses)
    return round(math.sqrt(variance) / mean_pause, 4)


# ---------------------------------------------------------------------------
# Cross-question naturalness flatness
# ---------------------------------------------------------------------------

def _naturalness_flatness(series: list[float]) -> float:
    """std(series) / (mean(series) + 1e-6). Returns 0.0 for <2 elements."""
    if len(series) < 2:
        return 0.0
    mean = sum(series) / len(series)
    variance = sum((x - mean) ** 2 for x in series) / len(series)
    return round(math.sqrt(variance) / (mean + 1e-6), 4)


# ---------------------------------------------------------------------------
# Track A -- self-relative deviation
# ---------------------------------------------------------------------------

def _compute_track_a(
    q: dict[str, Any],
    evaluable_peers: list[dict[str, Any]],
) -> tuple[dict[str, float | None], dict[str, bool]]:
    """
    Compute directional robust z-scores for Track A features.

    BUG 1 FIX: Returns float | None per feature. None means the feature was
    not computable for this question (below word-count minimum). None values
    are excluded from composite scoring and from peer baselines.

    BUG 5 FIX: discourse_organization uses percentile rank within the
    candidate's own answers rather than robust z-score, because connector
    counts are sparse / zero-heavy and MAD collapses to zero constantly.

    Returns (z_map, feature_availability_map).
    """
    avail = _feature_availability(q)

    if len(evaluable_peers) < TRACK_A_MIN_EVALUABLE_ANSWERS - 1:
        return {}, avail

    t_f = q.get("text_feats", {})

    # -- speech_rate_wpm (always available if question passes eligibility)
    speech_rate = float(t_f.get("speech_rate_wpm", 0.0))
    baseline_speech = [
        float(p["text_feats"].get("speech_rate_wpm", 0.0))
        for p in evaluable_peers
        if p.get("text_feats", {}).get("speech_rate_wpm") is not None
    ]
    z_speech: float | None = (
        _robust_z(speech_rate, baseline_speech, "speech_rate_wpm", "increase")
        if len(baseline_speech) >= 2 else None
    )

    # -- filler_word_ratio
    filler = float(t_f.get("filler_word_ratio", 0.0))
    baseline_filler = [
        float(p["text_feats"].get("filler_word_ratio", 0.0))
        for p in evaluable_peers
        if p.get("text_feats", {}).get("filler_word_ratio") is not None
    ]
    z_filler: float | None = (
        _robust_z(filler, baseline_filler, "filler_word_ratio", "decrease")
        if len(baseline_filler) >= 2 else None
    )

    # -- lexical_mattr: None if answer too short (BUG 1 fix)
    mattr_val = _get_mattr(q, avail)
    if mattr_val is not None:
        baseline_mattr = [
            _get_mattr(p, _feature_availability(p))
            for p in evaluable_peers
        ]
        baseline_mattr = [v for v in baseline_mattr if v is not None]
        z_mattr: float | None = (
            _robust_z(mattr_val, baseline_mattr, "lexical_mattr", "increase")
            if len(baseline_mattr) >= 2 else None
        )
    else:
        z_mattr = None

    # -- discourse_organization: percentile rank (BUG 5 fix)
    #    Percentile rank is appropriate for sparse count distributions where
    #    MAD collapses to 0, causing robust z to pin at the clip ceiling.
    disc_org_val = _get_discourse_org(q, avail)
    if disc_org_val is not None:
        peer_disc = [
            _get_discourse_org(p, _feature_availability(p))
            for p in evaluable_peers
        ]
        peer_disc_valid = [v for v in peer_disc if v is not None]
        if len(peer_disc_valid) >= 2:
            # Rank among all evaluable answers (self included) for the distribution
            all_disc = peer_disc_valid + [disc_org_val]
            pct = _percentile_rank(disc_org_val, all_disc)
            # Scale: high rank (high connector use) = suspicious; map to [0, Z_SCORE_CLIP]
            # Use: z-equivalent = Z_SCORE_CLIP * (pct - 0.5) * 2, clipped to [0, Z_SCORE_CLIP]
            z_discourse: float | None = round(
                max(0.0, min(Z_SCORE_CLIP, Z_SCORE_CLIP * (pct - 0.5) * 2.0)), 4
            )
        else:
            z_discourse = None
    else:
        z_discourse = None

    # -- sbert_coherence: None if unavailable or fallback sentinel (BUG 6 fix)
    # BUG A FIX: _get_sbert() can return None even when avail["sbert_coherence"]=True
    # (i.e. word count passes the structural minimum but the value is 0.0 or the
    # 0.65 neutral sentinel). In that case we must correct the availability flag
    # AFTER the extraction attempt, not before. Reporting available=True next to
    # a null z-score is a lie and masks the extraction failure from reviewers.
    sbert_val = _get_sbert(q, avail)
    if sbert_val is not None:
        baseline_coh = [
            _get_sbert(p, _feature_availability(p))
            for p in evaluable_peers
        ]
        baseline_coh = [v for v in baseline_coh if v is not None]
        if len(baseline_coh) >= 2:
            z_coh: float | None = _robust_z(
                sbert_val, baseline_coh, "sbert_coherence", "increase"
            )
        else:
            logger.warning(
                "Q%s: only %d valid sbert_coherence peers after filtering neutrals "
                "(0.0 / 0.65 sentinels excluded) -- z_coh=None. "
                "Check whether SBERT model loaded correctly across the whole interview.",
                q.get("q_no"), len(baseline_coh),
            )
            z_coh = None
    else:
        z_coh = None

    # BUG A FIX: correct sbert_coherence availability to reflect actual outcome.
    # If _get_sbert() returned None (extraction failed / sentinel / word-count gate),
    # set availability=False so the output never shows available=True + null z-score.
    if z_coh is None and avail.get("sbert_coherence", False):
        logger.warning(
            "Q%s: feature_availability.sbert_coherence corrected to False: "
            "word count passed structural gate but coherence value was null/sentinel. "
            "Likely cause: SBERT model unavailable, value==0.65 fallback, or "
            "fewer than 2 valid peers after sentinel filtering.",
            q.get("q_no"),
        )
        avail = dict(avail)   # don't mutate the original
        avail["sbert_coherence"] = False

    z_map: dict[str, float | None] = {
        "speech_rate_wpm":        z_speech,
        "filler_word_ratio":      z_filler,
        "lexical_mattr":          z_mattr,
        "discourse_organization": z_discourse,
        "sbert_coherence":        z_coh,
    }

    return z_map, avail


# ---------------------------------------------------------------------------
# Track C -- mechanism-level naturalness
# ---------------------------------------------------------------------------

def _compute_track_c(
    q: dict[str, Any],
    evaluable_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute Track C naturalness deviation scores.

    BUG 2 FIX: response_latency_sec enters composite via sigmoid, not as a
    boolean threshold. The raw z-score is preserved in diagnostics.

    BUG 4 FIX: cross_question_naturalness_flatness and latency_variance now
    contribute fixed provisional bonus points when raw value is suspicious,
    rather than outputting permanent 0.0. Clearly marked UNCALIBRATED.
    STOPGAP until population-level Track B baseline exists.

    Returns dict with z-scores for weights, raw values, and provisional flags.
    """
    word_timestamps = q.get("word_timestamps") or []
    duration_s = float(q.get("duration_s", 0.0))
    v_f = q.get("vocal_feats", {})
    q_no = q.get("q_no")
    avail = _feature_availability(q)

    # -- intra_answer_pace_variance (None if answer too short)
    pace_var = _intra_answer_pace_variance(word_timestamps, duration_s, q_no)
    if pace_var is not None:
        pace_var_series = [
            _intra_answer_pace_variance(
                p.get("word_timestamps") or [],
                float(p.get("duration_s", 0.0)),
            )
            for p in evaluable_results
        ]
        valid_pace = [v for v in pace_var_series if v is not None]
        z_pace_var: float | None = (
            _robust_z(pace_var, valid_pace, "intra_answer_pace_variance", "decrease")
            if len(valid_pace) >= 2 else None
        )
    else:
        z_pace_var = None

    # -- pause_regularity (None if answer too short)
    pause_reg = _pause_regularity(word_timestamps, duration_s, q_no)
    if pause_reg is not None:
        pause_reg_series = [
            _pause_regularity(
                p.get("word_timestamps") or [],
                float(p.get("duration_s", 0.0)),
            )
            for p in evaluable_results
        ]
        valid_pause = [v for v in pause_reg_series if v is not None]
        z_pause_reg: float | None = (
            _robust_z(pause_reg, valid_pause, "pause_regularity", "decrease")
            if len(valid_pause) >= 2 else None
        )
    else:
        z_pause_reg = None

    # -- pitch_variance_ratio
    pitch_cv = float(v_f.get("pitch_variation", 0.0))
    pitch_series = [
        float(p.get("vocal_feats", {}).get("pitch_variation", 0.0))
        for p in evaluable_results
    ]
    pitch_range = max(pitch_series) - min(pitch_series) if pitch_series else 0.0
    pitch_variance_ratio = (
        pitch_cv / (pitch_range + 1e-6) if pitch_range > 0 else pitch_cv
    )
    z_pitch: float | None = (
        _robust_z(pitch_cv, pitch_series, "pitch_variance_ratio", "decrease")
        if len(pitch_series) >= 2 else None
    )

    # -- response_latency_sec (BUG 2 FIX: sigmoid, not boolean gate)
    latency = float(word_timestamps[0].get("start", 0.0)) if word_timestamps else 0.0
    latency_series = [
        float((p.get("word_timestamps") or [{}])[0].get("start", 0.0))
        if p.get("word_timestamps") else 0.0
        for p in evaluable_results
    ]
    med_lat = _median(latency_series)
    mad_lat = max(_mad(latency_series, med_lat), 0.05)
    raw_z_lat = 0.6745 * (latency - med_lat) / mad_lat
    raw_z_lat_clipped = max(-Z_SCORE_CLIP, min(Z_SCORE_CLIP, raw_z_lat))
    abs_z_latency = abs(raw_z_lat_clipped)
    # Sigmoid contribution: maps latency deviation to a smooth [0,1] scale
    # then multiplied by LATENCY_COMPOSITE_WEIGHT in composite.
    sigmoid_latency = _sigmoid(abs_z_latency, LATENCY_SIGMOID_SCALE)

    # -- latency_variance_across_questions (BUG 4 FIX: provisional threshold on raw)
    # STOPGAP — contributes provisional bonus if raw latency std is suspiciously low.
    # Replace with population z-score when Track B data is available.
    if len(latency_series) >= 2:
        lat_mean = sum(latency_series) / len(latency_series)
        lat_var = sum((x - lat_mean) ** 2 for x in latency_series) / len(latency_series)
        raw_latency_std = math.sqrt(lat_var)
    else:
        raw_latency_std = 0.0

    latency_var_provisional = (
        raw_latency_std < PROVISIONAL_LATENCY_VAR_THRESHOLD
        and len(latency_series) >= 3
    )
    # Bonus points for composite when suspicious (labelled as provisional)
    latency_var_bonus = PROVISIONAL_LATENCY_VAR_SCORE if latency_var_provisional else 0.0

    # -- cross_question_naturalness_flatness (BUG 4 FIX: provisional threshold on raw)
    # Collect series only from answers where the feature is computable (BUG 1 fix).
    speech_series_valid = [
        float(p.get("text_feats", {}).get("speech_rate_wpm", 0.0))
        for p in evaluable_results
        if p.get("text_feats", {}).get("speech_rate_wpm") is not None
    ]
    mattr_series_valid = [
        _get_mattr(p, _feature_availability(p))
        for p in evaluable_results
    ]
    mattr_series_valid = [v for v in mattr_series_valid if v is not None]

    coh_series_valid = [
        _get_sbert(p, _feature_availability(p))
        for p in evaluable_results
    ]
    coh_series_valid = [v for v in coh_series_valid if v is not None]

    flatness_vals: list[float] = []
    if len(speech_series_valid) >= 2:
        flatness_vals.append(_naturalness_flatness(speech_series_valid))
    if len(mattr_series_valid) >= 2:
        flatness_vals.append(_naturalness_flatness(mattr_series_valid))
    if len(coh_series_valid) >= 2:
        flatness_vals.append(_naturalness_flatness(coh_series_valid))

    valid_flatness = [f for f in flatness_vals if f > 0]
    raw_composite_flatness = (
        sum(valid_flatness) / len(valid_flatness) if valid_flatness else 0.0
    )

    # STOPGAP provisional threshold. Replace with population z-score when available.
    # Note: flatness_vals could be empty if all features are unavailable; in that
    # case we cannot make a flatness determination — set provisional flag False.
    flatness_provisional = (
        len(flatness_vals) >= 1
        and raw_composite_flatness < PROVISIONAL_FLATNESS_THRESHOLD
        and len(evaluable_results) >= 3  # need enough answers for meaningful flatness
    )
    flatness_bonus = PROVISIONAL_FLATNESS_SCORE if flatness_provisional else 0.0

    return {
        # Weighted deviation scores for composite (None = not computable)
        "intra_answer_pace_variance":           z_pace_var,
        "pause_regularity":                     z_pause_reg,
        "pitch_variance_ratio":                 z_pitch,
        # Sigmoid latency for composite weighting
        "_sigmoid_latency":                     round(sigmoid_latency, 4),
        # Provisional bonus scores (BUG 4 fix)
        "_latency_var_bonus":                   latency_var_bonus,
        "_flatness_bonus":                      flatness_bonus,
        # Public diagnostics
        "response_latency_sec_z":               round(abs_z_latency, 4),
        "latency_variance_provisional_flag":    latency_var_provisional,
        "cross_question_naturalness_flatness_provisional_flag": flatness_provisional,
        # Raw values (always emitted for transparency)
        "_raw_latency_s":                       round(latency, 3),
        "_raw_latency_std":                     round(raw_latency_std, 4),
        "_raw_pace_var":                        round(pace_var, 4) if pace_var is not None else None,
        "_raw_pause_reg":                       round(pause_reg, 4) if pause_reg is not None else None,
        "_raw_pitch_cv":                        round(pitch_variance_ratio, 4),
        "_raw_flatness":                        round(raw_composite_flatness, 4),
    }


# ---------------------------------------------------------------------------
# Composite scoring and flag logic
# ---------------------------------------------------------------------------

def _composite_score(
    track_a_z: dict[str, float | None],
    track_c_devs: dict[str, Any],
) -> tuple[float, list[tuple[str, float]]]:
    """
    Continuous weighted composite = sum(weight_i * deviation_i).

    BUG 1 FIX: None z-scores contribute 0.0 (skipped, not treated as "average").
    BUG 2 FIX: latency enters via sigmoid, not as a binary gate.
    BUG 4 FIX: provisional bonus scores added for flatness / latency_var.

    Returns (composite_score, contributions_sorted_desc).
    """
    contributions: list[tuple[str, float]] = []

    # Track A
    for feat, weight in TRACK_A_WEIGHTS.items():
        z = track_a_z.get(feat)
        score = weight * (z if z is not None else 0.0)
        contributions.append((feat, score))

    # Track C weighted z-scores (skip None)
    for feat, weight in TRACK_C_WEIGHTS.items():
        dev = track_c_devs.get(feat)
        score = weight * (dev if dev is not None else 0.0)
        contributions.append((feat, score))

    # Latency via sigmoid (BUG 2 fix)
    sig_lat = track_c_devs.get("_sigmoid_latency", 0.5)
    # Centre on 0.5 (neutral) so no latency deviation = 0 contribution
    lat_contribution = LATENCY_COMPOSITE_WEIGHT * (sig_lat - 0.5) * 2.0
    contributions.append(("response_latency_sec", round(lat_contribution, 4)))

    # Provisional bonuses (BUG 4 fix) — labelled so they're identifiable
    lat_var_bonus = track_c_devs.get("_latency_var_bonus", 0.0)
    flat_bonus = track_c_devs.get("_flatness_bonus", 0.0)
    if lat_var_bonus > 0:
        contributions.append(("latency_variance_provisional", lat_var_bonus))
    if flat_bonus > 0:
        contributions.append(("cross_question_flatness_provisional", flat_bonus))

    contributions.sort(key=lambda x: x[1], reverse=True)
    total = sum(c for _, c in contributions)
    return round(total, 4), contributions


def _flag_for_review(
    composite: float,
    track_c_devs: dict[str, Any],
) -> bool:
    """
    BUG 2 FIX: flag requires signal to exceed threshold by HYSTERESIS_BAND.
    BUG B FIX: both provisional flags (latency_variance AND flatness) are read
    here and treated as equivalent OR conditions. Previously only the flatness
    flag was checked; latency_variance_provisional_flag was silently discarded.

    Decision: provisional flags are treated as interview-level context signals.
    They contribute to the flag decision for EVERY question in the interview
    (since both are computed from the interview-wide series, not per-question
    deviation). This is intentional: a candidate who is uniformly suspicious
    across the whole interview should have ALL their questions reviewed, not
    just the single most deviant one. Reviewers can use suspicion_score rank
    and low_answer_count_reduced_confidence to prioritise within that set.

    Sample-size guard: if fewer than TRACK_A_MIN_EVALUABLE_ANSWERS evaluable
    questions exist, the provisional flags are still surfaced but the caller
    marks the question with low_answer_count_reduced_confidence=True so
    reviewers know to weight accordingly.

    flag = composite > COMPOSITE_THRESHOLD
           AND (latency_z > HARD_SIGNAL_THRESHOLD + HYSTERESIS_BAND
                OR latency_variance_provisional_flag is True
                OR flatness_provisional_flag is True)
    """
    if composite <= COMPOSITE_THRESHOLD:
        return False
    latency_z = track_c_devs.get("response_latency_sec_z", 0.0)
    latency_var_prov = track_c_devs.get("latency_variance_provisional_flag", False)
    flatness_prov = track_c_devs.get(
        "cross_question_naturalness_flatness_provisional_flag", False
    )
    latency_triggers = latency_z > (HARD_SIGNAL_THRESHOLD + HYSTERESIS_BAND)
    return latency_triggers or latency_var_prov or flatness_prov


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def score_interview(question_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Score all questions in an interview for academic violation signals.

    Args:
        question_results: per-question dicts from _process_single_question().
            Each dict must contain: q_no, text, word_timestamps, duration_s,
            mean_confidence, text_feats, vocal_feats.

    Returns:
        List of evidence payload dicts in q_no order.
        Non-evaluable: {evaluable: False, not_evaluable_reason: ..., ...}
        Evaluable: {evaluable: True, feature_availability: {...}, ...}
    """
    if not question_results:
        return []

    sorted_qs = sorted(question_results, key=lambda x: int(x["q_no"]))

    # Step 1: Eligibility gate
    eligibility = [classify_eligibility(q) for q in sorted_qs]
    evaluable_results = [
        q for q, (ok, _) in zip(sorted_qs, eligibility) if ok
    ]

    logger.info(
        "score_interview: %d total questions, %d evaluable",
        len(sorted_qs), len(evaluable_results),
    )

    evidence: list[dict[str, Any]] = []

    for q, (is_evaluable, reason) in zip(sorted_qs, eligibility):
        q_no = int(q["q_no"])

        if not is_evaluable:
            evidence.append({
                "q_no": q_no,
                "evaluable": False,
                "not_evaluable_reason": reason,
                "suspicion_score": None,
                "flagged_for_review": False,
                "feature_availability": {},
                "track_a_z_scores": {},
                "track_c_diagnostics": {},
                "top_contributing_features": [],
            })
            continue

        # Track A: leave-one-out peers (evaluable only, excluding current q)
        evaluable_peers = [p for p in evaluable_results if int(p["q_no"]) != q_no]
        track_a_z, feat_avail = _compute_track_a(q, evaluable_peers)

        # Track C: all evaluable answers including current
        track_c_devs = _compute_track_c(q, evaluable_results)

        composite, contributions = _composite_score(track_a_z, track_c_devs)
        flagged = _flag_for_review(composite, track_c_devs)

        # Top contributing features (non-internal, non-zero)
        top3 = [
            feat for feat, score in contributions[:5]
            if not feat.startswith("_") and score > 0
        ][:3]

        # Public Track C dict (exclude internal _ keys)
        track_c_public = {
            k: v for k, v in track_c_devs.items()
            if not k.startswith("_")
        }
        track_c_raw = {
            k[len("_raw_"):]: v
            for k, v in track_c_devs.items()
            if k.startswith("_raw_")
        }

        # track_c_provisional_flag: True if ANY provisional signal is active
        track_c_provisional_flag = (
            track_c_devs.get("latency_variance_provisional_flag", False)
            or track_c_devs.get("cross_question_naturalness_flatness_provisional_flag", False)
        )

        evidence.append({
            "q_no": q_no,
            "evaluable": True,
            "not_evaluable_reason": None,
            "suspicion_score": composite,
            "flagged_for_review": flagged,
            "feature_availability": feat_avail,
            "track_a_z_scores": track_a_z,
            "track_c_diagnostics": track_c_public,
            "track_c_raw_values": track_c_raw,
            "track_c_provisional_flag": track_c_provisional_flag,
            "top_contributing_features": top3,
            "low_answer_count_reduced_confidence": (
                len(evaluable_results) < TRACK_A_MIN_EVALUABLE_ANSWERS
            ),
        })

    return evidence
