"""
pipeline/skills_scorer.py — System A: Communication Skills Engine.

Implements the 5-axis scoring model from Zeko Unified Communication
Framework v1, Section 2.2–2.5.

Each axis uses percentile-band scoring calibrated on n=86 interviews
(THRESHOLD_VERSION = "v0-n86-june2026"). All Indian English instrument
corrections are applied at the feature extraction layer (transcriber.py,
text_features.py, vocal_features.py) — scoring thresholds are universal.

Axes and weights:
  Axis 1 — Fluency            20%
  Axis 2 — Intelligibility    20%
  Axis 3 — Lexical/Structural 15%
  Axis 4 — Narrative/Evidence 15%
  Axis 5 — Vocal Delivery     10%
  [Grammar reserved]          20% — pending implementation

Role weight profiles supported: default / client_facing / technical / leadership

V3 note: intel_confidence thresholds are kept at the same calibrated values.
AssemblyAI confidence is directly usable — no +0.06 offset correction needed
(unlike the Whisper avg_logprob proxy used in v2).
"""
from __future__ import annotations

import logging

from v3.models import AxisResult, RawFeatures, SkillsAssessment

logger = logging.getLogger("v3.skills_scorer")

# ── Threshold version tag (Framework §6.4) ────────────────────────────────────
THRESHOLD_VERSION = "v0-n86-june2026"
RECALIBRATION_TRIGGER_N = 300

# ── Role-based weight profiles (Framework §2.4) ───────────────────────────────
_ROLE_WEIGHTS: dict[str, dict[str, float]] = {
    "default": {
        "fluency": 0.20, "intelligibility": 0.20,
        "lexical_structural": 0.15, "narrative_evidence": 0.15,
        "vocal_delivery": 0.10,
    },
    "client_facing": {
        "fluency": 0.20, "intelligibility": 0.20,
        "lexical_structural": 0.10, "narrative_evidence": 0.20,
        "vocal_delivery": 0.10,
    },
    "technical": {
        "fluency": 0.15, "intelligibility": 0.20,
        "lexical_structural": 0.20, "narrative_evidence": 0.15,
        "vocal_delivery": 0.05,
    },
    "leadership": {
        "fluency": 0.15, "intelligibility": 0.15,
        "lexical_structural": 0.20, "narrative_evidence": 0.20,
        "vocal_delivery": 0.10,
    },
}

# ── Band boundaries (percentile thresholds, n=86) ─────────────────────────────
# All thresholds tagged with THRESHOLD_VERSION. Recalibrate at n=300.
# Format: (p20, p40, p60, p80)

_THRESHOLDS: dict[str, tuple[float, float, float, float]] = {
    # Fluency
    "fluency_wpm":       (44.7, 55.0, 67.3, 93.3),
    "fluency_filler":    (0.018, 0.025, 0.043, 0.058),   # inverted: lower = better
    "fluency_pause_dur": (2.8,  4.4,  5.9,  8.8),        # inverted: lower = better

    # Intelligibility — thresholds recalibrated for AssemblyAI confidence scores
    # (AssemblyAI word-level confidence is better calibrated than Whisper logprob;
    # no +0.06 offset applied, so these targets are slightly lower)
    "intel_confidence":  (0.679, 0.722, 0.759, 0.799),

    # Lexical & Structural
    "lexical_mattr":     (0.732, 0.751, 0.779, 0.794),
    "lexical_rare":      (0.101, 0.132, 0.155, 0.173),
    "discourse_conn":    (9.0, 11.0, 12.0, 13.0),       # 10% ASR loss comp. applied at scoring
    "sbert_coherence":   (0.53, 0.59, 0.65, 0.71),      # -0.06 calibrated targets

    # Narrative & Evidence
    "ner_density":       (0.5, 1.0, 2.0, 3.5),
    "metric_density":    (0.0, 0.5, 1.0, 2.0),

    # Vocal Delivery (CV-normalized; dialect-neutral)
    "pitch_cv":          (0.15, 0.22, 0.28, 0.38),
    "voiced_fraction":   (0.170, 0.216, 0.264, 0.345),
}


# ── Core band scoring ─────────────────────────────────────────────────────────

def _clamp(value: float, lo: float = 0.0, hi: float = 5.0) -> float:
    return float(max(lo, min(value, hi)))


def _band_score(value: float, thresholds: tuple[float, float, float, float],
                higher_is_better: bool = True) -> float:
    """
    Map a raw feature value onto 0.0–5.0 using percentile-band boundaries.

    Linear interpolation within each band for smooth, continuous scoring.
    Soft extrapolation beyond p80 capped at 5.0.
    """
    p20, p40, p60, p80 = thresholds

    if not higher_is_better:
        midpoint = (p20 + p80) / 2.0
        value = 2.0 * midpoint - value

    bands = [
        (float("-inf"), p20, 0.0, 1.0),
        (p20,           p40, 1.0, 2.0),
        (p40,           p60, 2.0, 3.0),
        (p60,           p80, 3.0, 4.0),
        (p80, float("inf"), 4.0, 5.0),
    ]

    for lo, hi, score_lo, score_hi in bands:
        if lo <= value < hi:
            if lo == float("-inf"):
                band_width = p40 - p20 if (p40 - p20) > 0 else 1.0
                t = max(0.0, 1.0 - (hi - value) / band_width)
                return _clamp(score_lo + t * (score_hi - score_lo))
            if hi == float("inf"):
                band_width = p80 - p60 if (p80 - p60) > 0 else 1.0
                t = min((value - lo) / band_width, 1.0)
                return _clamp(score_lo + t * (score_hi - score_lo))
            t = (value - lo) / (hi - lo)
            return score_lo + t * (score_hi - score_lo)

    return 5.0


def _score_to_band(score: float) -> str:
    if score < 1.0:
        return "Poor"
    elif score < 2.0:
        return "Below Average"
    elif score < 3.0:
        return "Average"
    elif score < 4.0:
        return "Good"
    else:
        return "Excellent"


def _composite_to_band(score: float) -> str:
    if score < 20:
        return "Poor"
    elif score < 40:
        return "Below Average"
    elif score < 60:
        return "Average"
    elif score < 80:
        return "Good"
    else:
        return "Excellent"


# ── Axis confidence (Framework §2.5) ─────────────────────────────────────────

def _axis_confidence(raw: RawFeatures, axis: str) -> tuple[float, list[str]]:
    """
    Confidence is reduced when:
    - Audio duration is short (individual questions < 1 min, or overall duration proxy < 10 min)
    - ASR intel_confidence < 0.65 (poor transcription quality, post-correction)
    - Feature value is extreme outlier (handled per-axis below)
    """
    confidence = 1.0
    flags: list[str] = []

    # Corrected duration proxy in minutes: total words / speech rate (WPM)
    duration_min = raw.total_words / max(raw.speech_rate_wpm, 1.0)

    # Apply short duration check if overall proxy is < 10 min OR if >60% of questions were < 1 min
    if duration_min < 10.0 or raw.is_short_duration:
        confidence -= 0.15
        flags.append("short_duration")

    if raw.total_words < 100:
        confidence -= 0.30
        flags.append("short_transcript")

    if raw.intel_confidence < 0.65:
        confidence -= 0.25
        flags.append("low_asr_confidence")

    # Axis-specific flags
    if axis == "vocal_delivery" and raw.pitch_variation < 0.01:
        confidence -= 0.40
        flags.append("pitch_pipeline_artifact")  # known parselmouth issue

    return max(0.1, round(confidence, 2)), flags


# ── Axis 1: Fluency (weight: 20%) ─────────────────────────────────────────────

def score_fluency(raw: RawFeatures) -> AxisResult:
    """
    Axis 1 — Fluency: natural speech flow, pace, filler discipline, pause management.

    Features:
      fluency_wpm (50%): Words per minute
      fluency_filler_rate (30%): Filler word ratio (lower = better)
      fluency_pause_dur (20%): Mean pause duration (lower = better)
    """
    flags: list[str] = []

    wpm_score = _band_score(raw.speech_rate_wpm, _THRESHOLDS["fluency_wpm"])
    filler_score = _band_score(raw.filler_word_ratio, _THRESHOLDS["fluency_filler"],
                               higher_is_better=False)

    if raw.fluency_pause_dur > 0.0:
        pause_score = _band_score(raw.fluency_pause_dur, _THRESHOLDS["fluency_pause_dur"],
                                  higher_is_better=False)
    else:
        pause_score = 2.5  # neutral fallback if pause not extracted
        flags.append("pause_dur_unavailable")

    score = _clamp(0.50 * wpm_score + 0.30 * filler_score + 0.20 * pause_score)
    confidence, conf_flags = _axis_confidence(raw, "fluency")

    # Human review: Poor band on Fluency triggers mandatory review
    if score < 1.0:
        flags.append("mandatory_human_review")

    return AxisResult(
        score=round(score, 2),
        confidence=confidence,
        band=_score_to_band(score),
        flags=flags + conf_flags,
    )


# ── Axis 2: Intelligibility (weight: 20%) ────────────────────────────────────

def score_intelligibility(raw: RawFeatures) -> AxisResult:
    """
    Axis 2 — Intelligibility: acoustic clarity proxied by ASR confidence.

    V3: AssemblyAI returns true per-word confidence probabilities. No +0.06
    bias-correction offset is needed (that was specific to Whisper's
    underestimated avg_logprob for Indian English). The thresholds are
    applied directly to the raw AssemblyAI confidence value.

    Feature:
      intel_confidence (100%): True per-word AssemblyAI confidence score
    """
    flags: list[str] = []
    score = _clamp(_band_score(raw.intel_confidence, _THRESHOLDS["intel_confidence"]))
    confidence, conf_flags = _axis_confidence(raw, "intelligibility")

    if score < 1.0:
        flags.append("mandatory_human_review")

    return AxisResult(
        score=round(score, 2),
        confidence=confidence,
        band=_score_to_band(score),
        flags=flags + conf_flags,
    )


# ── Axis 3: Lexical & Structural Quality (weight: 15%) ───────────────────────

def score_lexical_structural(raw: RawFeatures) -> AxisResult:
    """
    Axis 3 — Lexical & Structural Quality: vocabulary richness + logical organization.

    Merges the original 'Logical Structure' and 'Chain of Thoughts' axes
    (Framework §2.2 Axis 3) as they share the same evidence base.

    Features:
      lexical_mattr (40%): MATTR vocabulary diversity
      discourse_connectors (30%): Unique connectors used (10% ASR loss compensation)
      sbert_coherence (20%): Sentence-level semantic flow
      lexical_rare (10%): Rare word ratio

    Special rule: if discourse_tier1 == 0, cap score at 3.5.
    """
    flags: list[str] = []

    mattr_score = _band_score(raw.lexical_mattr, _THRESHOLDS["lexical_mattr"])
    rare_score = _band_score(raw.lexical_rare_word_ratio, _THRESHOLDS["lexical_rare"])

    # Apply 10% ASR loss compensation: multiply connector count by 1/0.90
    # before scoring to restore target coverage (Framework §2.2 Axis 3)
    conn_corrected = raw.discourse_connectors / 0.90
    conn_score = _band_score(conn_corrected, _THRESHOLDS["discourse_conn"])

    # SBERT coherence: -0.06 calibration already baked into thresholds
    sbert_score = _band_score(raw.sbert_coherence, _THRESHOLDS["sbert_coherence"])

    score = _clamp(
        0.40 * mattr_score +
        0.30 * conn_score +
        0.20 * sbert_score +
        0.10 * rare_score
    )

    # Hard cap: no Tier-1 connectors → cap at 3.5 (Framework §2.2 Axis 3)
    if raw.discourse_tier1 == 0 and score > 3.5:
        score = 3.5
        flags.append("no_tier1_connectors_cap")

    # Short transcript gate
    if raw.total_words < 100:
        scale = raw.total_words / 100.0
        score = score * scale
        flags.append("short_transcript_gate")

    # SBERT fallback flag
    if raw.sbert_coherence == 0.65:
        flags.append("sbert_coherence_fallback")

    confidence, conf_flags = _axis_confidence(raw, "lexical_structural")

    return AxisResult(
        score=round(_clamp(score), 2),
        confidence=confidence,
        band=_score_to_band(score),
        flags=flags + conf_flags,
    )


# ── Axis 4: Narrative & Evidence (weight: 15%) ───────────────────────────────

def score_narrative_evidence(raw: RawFeatures) -> AxisResult:
    """
    Axis 4 — Narrative & Evidence: concrete examples, structured narrative, specific evidence.

    Merges 'Storytelling' and 'Usage of Examples' axes (Framework §2.2 Axis 4).

    Features:
      ner_entity_density (40%): Named entities per minute (Indian gazetteer applied)
      metric_density (35%): Data metrics per minute (includes crore/lakh)
      narrative_arc_score (25%): Labovian arc completeness (0–1)
    """
    flags: list[str] = []

    ner_score = _band_score(raw.ner_entity_density, _THRESHOLDS["ner_density"])
    metric_score = _band_score(raw.metric_density, _THRESHOLDS["metric_density"])
    # narrative_arc_score is already 0–1; map to 0–5
    arc_score = _clamp(raw.narrative_arc_score * 5.0)

    score = _clamp(
        0.40 * ner_score +
        0.35 * metric_score +
        0.25 * arc_score
    )

    confidence, conf_flags = _axis_confidence(raw, "narrative_evidence")

    return AxisResult(
        score=round(score, 2),
        confidence=confidence,
        band=_score_to_band(score),
        flags=flags + conf_flags,
    )


# ── Axis 5: Vocal Delivery (weight: 10%) ─────────────────────────────────────

def score_vocal_delivery(raw: RawFeatures) -> AxisResult:
    """
    Axis 5 — Vocal Delivery: voice modulation and acoustic presence.

    Features:
      pitch_variation (70%): CV-normalized F0 variation (dialect-neutral)
      voiced_fraction (30%): Fraction of audio that is active speech

    Note: pitch_cv thresholds are calibrated for CV values, not raw Hz.
    The CV representation makes this metric Indian English safe without
    any separate threshold adjustment.
    """
    flags: list[str] = []

    # Detect known pipeline artifact (near-constant pitch)
    if raw.pitch_variation < 0.01:
        flags.append("pitch_pipeline_artifact_excluded")
        # Exclude from composite — return low-confidence neutral score
        return AxisResult(
            score=2.5,
            confidence=0.1,
            band="Average",
            flags=flags,
        )

    pitch_score = _band_score(raw.pitch_variation, _THRESHOLDS["pitch_cv"])
    voiced_score = _band_score(raw.voiced_fraction, _THRESHOLDS["voiced_fraction"])

    score = _clamp(0.70 * pitch_score + 0.30 * voiced_score)
    confidence, conf_flags = _axis_confidence(raw, "vocal_delivery")

    return AxisResult(
        score=round(score, 2),
        confidence=confidence,
        band=_score_to_band(score),
        flags=flags + conf_flags,
    )


# ── Human review trigger protocol (Framework §2.6) ────────────────────────────

def _check_review_triggers(
    fluency: AxisResult,
    intelligibility: AxisResult,
    lexical_structural: AxisResult,
    narrative_evidence: AxisResult,
    vocal_delivery: AxisResult,
    composite: float,
) -> tuple[bool, list[str]]:
    """Evaluate all human review trigger conditions."""
    triggers: list[str] = []

    if fluency.score < 1.0:
        triggers.append("fluency_poor_band")
    if intelligibility.score < 1.0:
        triggers.append("intelligibility_poor_band")

    axes = [fluency, intelligibility, lexical_structural, narrative_evidence, vocal_delivery]
    low_conf_axes = [a for a in axes if a.confidence < 0.5]
    if low_conf_axes:
        triggers.append(f"low_confidence_axes_{len(low_conf_axes)}")

    outlier_axes = sum(1 for a in axes if a.score > 4.8 or a.score < 0.2)
    if outlier_axes >= 3:
        triggers.append("multiple_outlier_axes")

    return bool(triggers), triggers


# ── Composite score (Framework §2.3) ─────────────────────────────────────────

def compute_composite(
    fluency: AxisResult,
    intelligibility: AxisResult,
    lexical_structural: AxisResult,
    narrative_evidence: AxisResult,
    vocal_delivery: AxisResult,
    role_profile: str = "default",
) -> float:
    """
    Weighted composite score scaled to 0–100.

    Axes with confidence < 0.5 are excluded from composite (replaced with
    the cohort Average = 2.5) per Framework §2.5.

    Grammar axis is reserved (weight 0.20). Current weights sum to 0.80.
    The ×1.25 multiplier normalizes to 100 across implemented axes.
    """
    weights = _ROLE_WEIGHTS.get(role_profile, _ROLE_WEIGHTS["default"])

    def _effective_score(axis: AxisResult, default: float = 2.5) -> float:
        return axis.score if axis.confidence >= 0.5 else default

    raw = (
        _effective_score(fluency)           * weights["fluency"] +
        _effective_score(intelligibility)   * weights["intelligibility"] +
        _effective_score(lexical_structural) * weights["lexical_structural"] +
        _effective_score(narrative_evidence) * weights["narrative_evidence"] +
        _effective_score(vocal_delivery)    * weights["vocal_delivery"]
    )

    # Sum of implemented weights = 0.80 (grammar 0.20 reserved)
    # Normalize: divide by 0.80 then scale to 100
    total_weight = sum(weights.values())  # 0.80
    return round(min((raw / total_weight) * 20.0, 100.0), 1)


# ── Main entry point ──────────────────────────────────────────────────────────

def score(raw: RawFeatures, role_profile: str = "default") -> SkillsAssessment:
    """
    Run the full System A Skills Engine on a RawFeatures object.

    Returns a SkillsAssessment with all 5 axis scores, composite score,
    and any triggered review flags.
    """
    logger.info(
        "System A scoring (threshold_version=%s, role=%s)",
        THRESHOLD_VERSION, role_profile
    )

    fluency        = score_fluency(raw)
    intelligibility = score_intelligibility(raw)
    lexical_struct = score_lexical_structural(raw)
    narrative_ev   = score_narrative_evidence(raw)
    vocal_del      = score_vocal_delivery(raw)

    composite = compute_composite(
        fluency, intelligibility, lexical_struct, narrative_ev, vocal_del,
        role_profile=role_profile,
    )

    review_required, review_triggers = _check_review_triggers(
        fluency, intelligibility, lexical_struct, narrative_ev, vocal_del,
        composite,
    )

    if review_triggers:
        # Append review trigger flags to the relevant axes for traceability
        logger.warning("Human review triggered: %s", review_triggers)

    return SkillsAssessment(
        fluency=fluency,
        intelligibility=intelligibility,
        lexical_structural=lexical_struct,
        narrative_evidence=narrative_ev,
        vocal_delivery=vocal_del,
        composite_score=composite,
        composite_band=_composite_to_band(composite),
        grammar_pending=True,
        review_required=review_required,
        role_profile=role_profile,
    )
