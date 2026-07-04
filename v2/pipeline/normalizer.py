"""
pipeline/normalizer.py — Normalize raw feature values to 0-100 scale and
aggregate them into 5 Communication Style signals (System B).

Updated for Zeko Unified Communication Framework v1, Part 3:

  Change 1: vocabulary_density (TTR) → lexical_mattr (MATTR)
            Source field: RawFeatures.lexical_mattr
  Change 2: pitch_variation now uses CV values (not Hz) — FEATURE_BOUNDS
            already updated in config.py to reflect CV scale (0.0–0.70).
  Change 3: Signal formulas are UNCHANGED from the original Style Framework
            (Section 3.2 of Communication_Style_Evaluation_Framework.md).
            Only input field names are updated.

  The normalizer is System B ONLY. System A (skills_scorer.py) performs its
  own scoring directly on RawFeatures — it does not pass through this module.
"""
from __future__ import annotations

from v2.config import FEATURE_BOUNDS
from v2.models import CommunicationSignals, RawFeatures


def _norm(value: float, feature_key: str) -> float:
    """Min-max normalize a raw value to 0–100 using configured bounds."""
    lo, hi = FEATURE_BOUNDS.get(feature_key, (0.0, 1.0))
    if hi <= lo:
        return 0.0
    clamped = max(lo, min(value, hi))
    return ((clamped - lo) / (hi - lo)) * 100.0


def normalize(raw: RawFeatures) -> dict[str, float]:
    """
    Convert all raw features to 0-100 scale for use in signal aggregation.

    Inverse features (lower raw = higher score):
      - filler_word_ratio → preparation_confidence (100 - norm)

    Change 1 applied: vocabulary_density input now reads from lexical_mattr
    (MATTR, length-independent). The normalized name 'vocabulary_precision'
    is preserved for signal weight compatibility.

    Change 3 applied: pitch_variation bounds now cover CV range (0.0–0.70)
    rather than raw Hz (0–150). Bound change is in config.py.
    """
    n: dict[str, float] = {}

    # ── Structural signals ─────────────────────────────────────────────────────
    # F08 connector density (ratio form) — for logical_connectors normalized score
    n["logical_connectors"]        = _norm(raw.logical_connector_density,    "logical_connector_density")
    n["sentence_complexity"]       = _norm(raw.avg_sentence_length,          "avg_sentence_length")
    # Inverse: fewer fillers → higher preparation confidence
    n["preparation_confidence"]    = 100.0 - _norm(raw.filler_word_ratio,    "filler_word_ratio")

    # ── Interpersonal signals ──────────────────────────────────────────────────
    n["collaboration_orientation"] = _norm(raw.collaborative_language_ratio, "collaborative_language_ratio")
    n["listener_engagement"]       = _norm(raw.question_density,             "question_density")
    n["emotional_expressiveness"]  = _norm(raw.empathetic_language_score,    "empathetic_language_score")

    # ── Precision & detail ─────────────────────────────────────────────────────
    # Change 1: reads lexical_mattr (MATTR) instead of legacy vocabulary_density (TTR)
    n["vocabulary_precision"]      = _norm(raw.lexical_mattr,                "lexical_mattr")
    n["results_orientation"]       = _norm(raw.metric_density,               "metric_density")

    # ── Pace & baseline energy ─────────────────────────────────────────────────
    n["energy_level"]              = _norm(raw.speech_rate_wpm,              "speech_rate_wpm")
    n["pace_adaptability"]         = _norm(raw.speech_rate_variability,      "speech_rate_variability")

    # ── Vocal / prosodic ──────────────────────────────────────────────────────
    # Change 3: pitch_variation is now CV-normalized (dialect-neutral for Indian English).
    # Config bounds set to (0.0, 0.70) CV range — no threshold adjustment needed here.
    n["pitch_expressiveness"]      = _norm(raw.pitch_variation,              "pitch_variation")
    n["vocal_presence"]            = _norm(raw.vocal_confidence,             "vocal_confidence")
    n["fluency"]                   = _norm(raw.speech_fluency,               "speech_fluency")
    n["emotional_stability"]       = _norm(raw.stress_markers,               "stress_markers")

    return n


def aggregate_signals(n: dict[str, float]) -> CommunicationSignals:
    """
    Aggregate normalized features into the 5 Communication Style signals.

    Formulas are UNCHANGED from the original Communication Style Evaluation
    Framework v2.0, Section 3.2. Only input field names are updated to reflect
    the new feature set (MATTR for vocabulary_precision, CV for pitch_expressiveness).

    Signal 1 — Systematic Thinking
      Logical structure + sentence complexity + fluency + vocal presence
      Architect ≈ 87.5, Analyst ≈ 75, Synthesizer ≈ 67.5, Connector ≈ 52.5, Pragmatist ≈ 60

    Signal 2 — Collaborative Orientation
      Team pronouns + question frequency + pitch variation + empathy
      Connector ≈ 82.5, Synthesizer ≈ 65, Pragmatist ≈ 60, Architect ≈ 42.5, Analyst ≈ 37.5

    Signal 3 — Analytical Precision
      Vocabulary richness + metric density + vocal stability + composure
      Analyst ≈ 85, Architect ≈ 72.5, Synthesizer ≈ 62.5, Pragmatist ≈ 52.5, Connector ≈ 50

    Signal 4 — Expressive Engagement
      Empathy + pitch variation + pace adaptability + listener engagement
      Connector ≈ 72.5, Synthesizer ≈ 62.5, Pragmatist ≈ 55, Architect ≈ 47.5, Analyst ≈ 42.5

    Signal 5 — Action Orientation
      Speech energy + metric density + vocal stability + composure
      Pragmatist ≈ 85, Connector ≈ 60, Synthesizer ≈ 62.5, Analyst ≈ 60, Architect ≈ 55
    """

    # Signal 1: Systematic Thinking
    systematic_thinking = (
        n["logical_connectors"]  * 0.35 +
        n["sentence_complexity"] * 0.25 +
        n["fluency"]             * 0.25 +
        n["vocal_presence"]      * 0.15
    )

    # Signal 2: Collaborative Orientation
    collaborative_orientation = (
        n["collaboration_orientation"] * 0.35 +
        n["listener_engagement"]       * 0.25 +
        n["pitch_expressiveness"]      * 0.25 +
        n["emotional_expressiveness"]  * 0.15
    )

    # Signal 3: Analytical Precision
    analytical_precision = (
        n["vocabulary_precision"]  * 0.35 +
        n["results_orientation"]   * 0.30 +
        n["vocal_presence"]        * 0.20 +
        n["emotional_stability"]   * 0.15
    )

    # Signal 4: Expressive Engagement
    expressive_engagement = (
        n["emotional_expressiveness"] * 0.30 +
        n["pitch_expressiveness"]     * 0.30 +
        n["pace_adaptability"]        * 0.20 +
        n["listener_engagement"]      * 0.20
    )

    # Signal 5: Action Orientation
    action_orientation = (
        n["energy_level"]        * 0.35 +
        n["results_orientation"] * 0.30 +
        n["vocal_presence"]      * 0.20 +
        n["emotional_stability"] * 0.15
    )

    def _clamp(v: float) -> float:
        return round(max(0.0, min(100.0, v)), 2)

    return CommunicationSignals(
        systematic_thinking       = _clamp(systematic_thinking),
        collaborative_orientation = _clamp(collaborative_orientation),
        analytical_precision      = _clamp(analytical_precision),
        expressive_engagement     = _clamp(expressive_engagement),
        action_orientation        = _clamp(action_orientation),
    )
