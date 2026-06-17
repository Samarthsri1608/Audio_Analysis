"""
pipeline/archetype.py — Rule-based archetype blending using cosine similarity.

Updated for Zeko Unified Communication Framework v1, §3.2 Change 2:
  - Archetype centroids recalibrated to empirical WPM dataset (median 61 WPM)
  - Action Orientation / energy_level dimension recalibrated accordingly
  - Softmax temperature documented and tunable via config
  - GMM readiness flag and role-fit matching added

Each archetype is represented by a centroid vector in the 5-signal space.
Centroids are derived from the archetype typical signal ranges in the
original Communication Style Evaluation Framework (Section 5.1), updated
to use WPM-recalibrated ranges per Section 3.2 Change 2 of the Unified
Framework.

Candidate signal vector is compared to each centroid via cosine similarity.
Similarities are passed through softmax to produce percentages summing to 100.

The GMM from the original framework (Section 6.1) will replace this
once 50+ manually labeled interviews are available (Framework §3.4).
Until then, this deterministic approach is:
  ✅ Fully deterministic and idempotent
  ✅ Zero training data needed
  ✅ Explainable to regulators and candidates
  ✅ Fast (<1 ms)
  ✅ API-compatible with the GMM that will replace it
"""
from __future__ import annotations

import math
import logging

from v2.models import ArchetypeBlend, CommunicationSignals, StyleProfile

logger = logging.getLogger("v2.archetype")

# ── GMM readiness flag (Framework §3.4) ──────────────────────────────────────
GMM_TRAINED = False
GMM_LABEL_COUNT = 0              # Update when human annotation is complete
GMM_REQUIRED_LABELS = 50        # Minimum before GMM deployment

# ── Archetype signal centroids (recalibrated, Framework §3.2 Change 2) ───────
# Columns: [systematic, collaborative, analytical, expressive, action_orientation]
#
# Original ranges (Section 5.1 of Communication_Style_Evaluation_Framework.md):
#   Architect:   systematic 75–100, collab 30–55, analytical 60–85,
#                expressive 35–60,  action 45–65
#   Connector:   systematic 40–65, collab 70–95, analytical 40–60,
#                expressive 60–85,  action 50–70
#   Synthesizer: all signals 55–75
#   Analyst:     systematic 65–85, collab 25–50, analytical 75–95,
#                expressive 30–55,  action 50–70
#   Pragmatist:  systematic 50–70, collab 50–70, analytical 40–65,
#                expressive 50–70,  action 75–95
#
# Change 2 impact: Action Orientation centroids shifted downward because
# empirical WPM median is 61 (not 130–180 as assumed in original framework).
# p80 threshold is ~93 WPM — Pragmatist centroid action_orientation recalibrated
# from 85 → 72 (reflecting top quartile in actual interview dataset, not
# assumed native speaker rates).
#
# Recalibration note: These centroids will be replaced by GMM component means
# once 50+ labeled interviews exist. Tag: CENTROID_VERSION = "v0-n86-june2026"

CENTROID_VERSION = "v0-n86-june2026"

_ARCHETYPES: dict[str, list[float]] = {
    #                   [systematic, collaborative, analytical, expressive, action]
    "architect":   [87.5,  42.5,  72.5,  47.5,  55.0],
    "connector":   [52.5,  82.5,  50.0,  72.5,  60.0],
    "synthesizer": [67.5,  65.0,  62.5,  62.5,  62.5],
    "analyst":     [75.0,  37.5,  85.0,  42.5,  60.0],
    # Change 2: action_orientation centroid recalibrated from 85 → 72
    # (top quartile WPM in empirical dataset ≈ 93 WPM, which normalizes to ~72
    # on the 80–200 WPM scale; the original 85 assumed 160–180 WPM)
    "pragmatist":  [60.0,  60.0,  52.5,  55.0,  72.0],
}

_ARCHETYPE_ORDER = ["architect", "connector", "synthesizer", "analyst", "pragmatist"]

# ── Role-fit signal targets (Framework §4.3) ──────────────────────────────────
# Each entry: {signal_name: (min_required, label)}
# "required" = must meet minimum; "nice_to_have" = labeled but not hard-fail

_ROLE_FIT_PROFILES: dict[str, dict[str, tuple[float, str]]] = {
    "software_engineer": {
        "systematic_thinking":       (75.0, "required"),
        "analytical_precision":      (70.0, "required"),
        "collaborative_orientation": (50.0, "nice_to_have"),
        "action_orientation":        (55.0, "nice_to_have"),
        "expressive_engagement":     (40.0, "not_required"),
    },
    "sales": {
        "collaborative_orientation": (75.0, "required"),
        "expressive_engagement":     (70.0, "required"),
        "action_orientation":        (65.0, "required"),
        "systematic_thinking":       (45.0, "nice_to_have"),
        "analytical_precision":      (50.0, "nice_to_have"),
    },
    "product_manager": {
        "systematic_thinking":       (65.0, "required"),
        "collaborative_orientation": (70.0, "required"),
        "analytical_precision":      (65.0, "required"),
        "action_orientation":        (60.0, "required"),
        "expressive_engagement":     (55.0, "nice_to_have"),
    },
    "team_lead": {
        "collaborative_orientation": (75.0, "required"),
        "systematic_thinking":       (65.0, "required"),
        "expressive_engagement":     (65.0, "required"),
        "action_orientation":        (60.0, "required"),
        "analytical_precision":      (55.0, "nice_to_have"),
    },
    "data_scientist": {
        "analytical_precision":      (75.0, "required"),
        "systematic_thinking":       (70.0, "required"),
        "action_orientation":        (50.0, "nice_to_have"),
        "collaborative_orientation": (45.0, "nice_to_have"),
        "expressive_engagement":     (40.0, "not_required"),
    },
    # default — no role-specific requirements
    "default": {},
}

_SIGNAL_NAMES = [
    "systematic_thinking",
    "collaborative_orientation",
    "analytical_precision",
    "expressive_engagement",
    "action_orientation",
]


# ── Math helpers ──────────────────────────────────────────────────────────────

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _magnitude(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    denom = _magnitude(a) * _magnitude(b)
    if denom == 0:
        return 0.0
    return _dot(a, b) / denom


def _softmax(values: list[float], temperature: float = 5.0) -> list[float]:
    """
    Softmax with temperature scaling.

    Temperature = 5.0 (default):
      - Sharpens the distribution — a clearly dominant style gets a higher
        percentage. Matches hiring manager expectation of seeing a primary
        and secondary archetype rather than a flat 20/20/20/20/20 blend.
      - Lower values (< 3) → flatter (better for genuinely balanced candidates)
      - Higher values (> 8) → too spiky; small signal differences dominate

    NOTE: Temperature is NOT used to inflate or deflate any one archetype.
    It only controls the sharpness of the overall distribution. This is
    tunable but must be changed consistently across all candidates.
    """
    scaled = [v * temperature for v in values]
    max_v = max(scaled)
    exps = [math.exp(s - max_v) for s in scaled]
    total = sum(exps)
    return [e / total for e in exps]


# ── Role-fit assessment ───────────────────────────────────────────────────────

def compute_role_fit(
    signals: CommunicationSignals,
    role: str = "default",
) -> dict:
    """
    Compare candidate signal scores against role-profile targets.

    Returns:
        {
          "role": str,
          "match": "strong" | "moderate" | "partial" | "low",
          "required_gaps": list[str],   # required signals below threshold
          "met_requirements": list[str],
        }
    """
    profile = _ROLE_FIT_PROFILES.get(role, {})
    if not profile:
        return {"role": role, "match": "n/a", "required_gaps": [], "met_requirements": []}

    signal_values = {
        "systematic_thinking":       signals.systematic_thinking,
        "collaborative_orientation": signals.collaborative_orientation,
        "analytical_precision":      signals.analytical_precision,
        "expressive_engagement":     signals.expressive_engagement,
        "action_orientation":        signals.action_orientation,
    }

    required_gaps: list[str] = []
    met_required: list[str] = []
    nice_gaps: list[str] = []

    for signal, (threshold, label) in profile.items():
        value = signal_values.get(signal, 0.0)
        if label == "required":
            if value >= threshold:
                met_required.append(signal)
            else:
                required_gaps.append(f"{signal} ({value:.0f} < {threshold:.0f})")
        elif label == "nice_to_have" and value < threshold:
            nice_gaps.append(signal)

    # Match level
    if not required_gaps and len(nice_gaps) == 0:
        match = "strong"
    elif not required_gaps and len(nice_gaps) <= 1:
        match = "moderate"
    elif len(required_gaps) <= 1:
        match = "partial"
    else:
        match = "low"

    return {
        "role": role,
        "match": match,
        "required_gaps": required_gaps,
        "met_requirements": met_required,
    }


# ── Archetype classification ──────────────────────────────────────────────────

def classify(signals: CommunicationSignals, role: str = "default") -> ArchetypeBlend:
    """
    Map 5 communication signals to archetype blend percentages.

    Args:
        signals: CommunicationSignals with values 0–100
        role: role profile key for role-fit annotation (default = no role)

    Returns:
        ArchetypeBlend with percentages summing to 100.

    When GMM_TRAINED = True (future state after 50+ labeled interviews),
    this function should be replaced with:
        blend = gmm.predict_proba(signal_vector)[0]
    keeping the same ArchetypeBlend return signature.
    """
    if GMM_TRAINED:
        logger.warning(
            "GMM_TRAINED=True but GMM model not loaded — falling back to cosine similarity. "
            "Set GMM_TRAINED=False until the model is wired."
        )

    candidate_vec: list[float] = [
        signals.systematic_thinking,
        signals.collaborative_orientation,
        signals.analytical_precision,
        signals.expressive_engagement,
        signals.action_orientation,
    ]

    similarities: list[float] = [
        _cosine_similarity(candidate_vec, _ARCHETYPES[name])
        for name in _ARCHETYPE_ORDER
    ]

    probabilities = _softmax(similarities, temperature=5.0)
    percentages = [round(p * 100, 1) for p in probabilities]

    # Minor rounding correction so they sum to exactly 100
    diff = 100.0 - sum(percentages)
    max_idx = percentages.index(max(percentages))
    percentages[max_idx] = round(percentages[max_idx] + diff, 1)

    return ArchetypeBlend(
        architect   = percentages[0],
        connector   = percentages[1],
        synthesizer = percentages[2],
        analyst     = percentages[3],
        pragmatist  = percentages[4],
    )


def dominant(blend: ArchetypeBlend) -> str:
    """Return the name of the archetype with the highest percentage."""
    values = {
        "architect":   blend.architect,
        "connector":   blend.connector,
        "synthesizer": blend.synthesizer,
        "analyst":     blend.analyst,
        "pragmatist":  blend.pragmatist,
    }
    return max(values, key=values.__getitem__)


def build_style_profile(
    signals: CommunicationSignals,
    blend: ArchetypeBlend,
    role: str = "default",
) -> StyleProfile:
    """
    Assemble the full System B StyleProfile from signals, blend, and role-fit.

    Includes GMM readiness metadata and role-fit match assessment.
    """
    role_fit = compute_role_fit(signals, role)
    dom = dominant(blend)

    return StyleProfile(
        systematic_thinking       = signals.systematic_thinking,
        collaborative_orientation = signals.collaborative_orientation,
        analytical_precision      = signals.analytical_precision,
        expressive_engagement     = signals.expressive_engagement,
        action_orientation        = signals.action_orientation,
        archetype_blend           = {
            "architect":   blend.architect,
            "connector":   blend.connector,
            "synthesizer": blend.synthesizer,
            "analyst":     blend.analyst,
            "pragmatist":  blend.pragmatist,
        },
        dominant_archetype        = dom,
        gmm_trained               = GMM_TRAINED,
        centroid_version          = CENTROID_VERSION,
        role_fit                  = role_fit,
    )
