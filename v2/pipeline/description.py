"""
pipeline/description.py — Generate a plain-language candidate style summary
using the OpenAI chat API.

This is the ONLY place in V2 where an LLM is used. Everything else is
rule-based. The description is optional (include_description=False skips it).

Updated for Zeko Unified Communication Framework v1:
  - Prompt now includes role-fit match result when a role is specified
  - Prompt distinguishes System A (skills score) from System B (style) clearly
  - Fallback description (no API key) uses StyleProfile instead of just a label
  - Candidate-facing phrasing rules enforced: no raw scores, no accent mentions
"""
from __future__ import annotations

import logging

from openai import AsyncOpenAI

from v2.config import (
    DESCRIPTION_MAX_TOKENS,
    DESCRIPTION_MODEL,
    OPENAI_API_KEY,
)
from v2.models import ArchetypeBlend, CommunicationSignals, StyleProfile

logger = logging.getLogger("v2.description")

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _client


# ── Archetype narrative summaries ─────────────────────────────────────────────
_ARCHETYPE_SUMMARY = {
    "architect":   "systematic, logical, and process-oriented",
    "connector":   "collaborative, people-focused, and relationship-oriented",
    "synthesizer": "balanced, adaptable, and able to bridge structure and collaboration",
    "analyst":     "data-driven, precision-focused, and evidence-based",
    "pragmatist":  "action-oriented, results-focused, and efficient",
}

# ── Signal human-readable labels ──────────────────────────────────────────────
_SIGNAL_LABELS = {
    "systematic_thinking":       "Systematic Thinking",
    "collaborative_orientation": "Collaborative Orientation",
    "analytical_precision":      "Analytical Precision",
    "expressive_engagement":     "Expressive Engagement",
    "action_orientation":        "Action Orientation",
}

# ── Role-fit phrasing ─────────────────────────────────────────────────────────
_MATCH_PHRASES = {
    "strong":   "✓ Strong match",
    "moderate": "✓ Good match",
    "partial":  "~ Partial match",
    "low":      "✗ Low match",
    "n/a":      "",
}


def _build_prompt(
    signals: CommunicationSignals,
    blend: ArchetypeBlend,
    dominant_archetype: str,
    role_fit: dict | None = None,
) -> str:
    """
    Build the LLM prompt for generating a hiring manager style summary.

    Rules enforced (per Framework §5.2 and §5.3):
    - Describe communication PATTERNS, not grades or scores
    - Mention the dominant archetype naturally
    - Highlight top 2 signals as observable strengths
    - Include role-fit indication when a role is specified
    - Do NOT mention accent, ASR confidence, or any System A score
    - Professional, positive, factual tone
    """
    # Sort signals high → low to identify top 2
    signal_dict = {
        "systematic_thinking":       signals.systematic_thinking,
        "collaborative_orientation": signals.collaborative_orientation,
        "analytical_precision":      signals.analytical_precision,
        "expressive_engagement":     signals.expressive_engagement,
        "action_orientation":        signals.action_orientation,
    }
    sorted_signals = sorted(signal_dict.items(), key=lambda x: x[1], reverse=True)

    signal_lines = "\n".join([
        f"  - {_SIGNAL_LABELS[k]}: {v:.0f}/100"
        for k, v in sorted_signals
    ])

    blend_lines = "\n".join([
        f"  - Architect: {blend.architect:.0f}%",
        f"  - Connector: {blend.connector:.0f}%",
        f"  - Synthesizer: {blend.synthesizer:.0f}%",
        f"  - Analyst: {blend.analyst:.0f}%",
        f"  - Pragmatist: {blend.pragmatist:.0f}%",
    ])

    top_two = [_SIGNAL_LABELS[k] for k, _ in sorted_signals[:2]]

    role_fit_block = ""
    if role_fit and role_fit.get("match") and role_fit["match"] != "n/a":
        match_phrase = _MATCH_PHRASES.get(role_fit["match"], "")
        role = role_fit.get("role", "the target role").replace("_", " ")
        gaps = role_fit.get("required_gaps", [])
        gap_note = f" Gap areas: {', '.join(gaps)}." if gaps else ""
        role_fit_block = f"\nRole Fit ({role}): {match_phrase}.{gap_note}"

    return f"""You are a talent intelligence assistant helping a hiring manager understand a candidate's communication style.

Write a concise 2–3 sentence communication style profile based ONLY on the data below.

STRICT RULES:
- Describe communication PATTERNS, not grades or percentages
- Mention the dominant archetype naturally in the first sentence
- Reference the top 2 signals ({top_two[0]}, {top_two[1]}) as observable communication patterns
- If role-fit data is provided, add a brief role-fit note at the end
- Do NOT say "scored X out of 100" or "X percent"
- Do NOT mention accent, pronunciation, or ASR quality
- Be factual, professional, and constructive

Communication Signals (ranked high to low):
{signal_lines}

Archetype Blend:
{blend_lines}

Dominant Archetype: {dominant_archetype.capitalize()} ({_ARCHETYPE_SUMMARY.get(dominant_archetype, '')})
{role_fit_block}

Write the style profile now (2–3 sentences):"""


# ── Public API ────────────────────────────────────────────────────────────────

async def generate_description(
    signals: CommunicationSignals,
    blend: ArchetypeBlend,
    dominant_archetype: str,
    role_fit: dict | None = None,
) -> str:
    """
    Generate a 2–3 sentence plain-language communication style profile.

    Returns an empty string on failure (description is optional).

    Args:
        signals: System B CommunicationSignals (0–100 each)
        blend: Archetype blend percentages
        dominant_archetype: Name of the dominant archetype
        role_fit: Optional role-fit assessment dict from archetype.compute_role_fit()
    """
    if not OPENAI_API_KEY:
        logger.info("OPENAI_API_KEY not set — returning fallback description.")
        summary = _ARCHETYPE_SUMMARY.get(dominant_archetype, "adaptable and professional")

        # Identify top signal for fallback
        signal_dict = {
            "Systematic Thinking":       signals.systematic_thinking,
            "Collaborative Orientation": signals.collaborative_orientation,
            "Analytical Precision":      signals.analytical_precision,
            "Expressive Engagement":     signals.expressive_engagement,
            "Action Orientation":        signals.action_orientation,
        }
        top_signal = max(signal_dict, key=signal_dict.__getitem__)

        fallback = (
            f"The candidate's dominant communication style is {dominant_archetype.capitalize()}, "
            f"which is typically {summary}. "
            f"Their communication patterns show particularly strong {top_signal.lower()}. "
            f"Detailed signal and archetype breakdown is available in the structured report."
        )
        return fallback

    client = _get_client()
    prompt = _build_prompt(signals, blend, dominant_archetype, role_fit=role_fit)

    try:
        response = await client.chat.completions.create(
            model=DESCRIPTION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=DESCRIPTION_MAX_TOKENS,
            temperature=0.4,
        )
        text = response.choices[0].message.content or ""
        return text.strip()
    except Exception as exc:
        logger.warning("Description generation failed: %s", exc)
        return ""


async def generate_description_from_style_profile(style_profile: StyleProfile) -> str:
    """
    Convenience wrapper: generate description directly from a StyleProfile object.
    Extracts signals, blend, dominant_archetype, and role_fit automatically.
    """
    signals = CommunicationSignals(
        systematic_thinking       = style_profile.systematic_thinking,
        collaborative_orientation = style_profile.collaborative_orientation,
        analytical_precision      = style_profile.analytical_precision,
        expressive_engagement     = style_profile.expressive_engagement,
        action_orientation        = style_profile.action_orientation,
    )
    blend = ArchetypeBlend(**style_profile.archetype_blend)
    return await generate_description(
        signals,
        blend,
        style_profile.dominant_archetype,
        role_fit=style_profile.role_fit,
    )
