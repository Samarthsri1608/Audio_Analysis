"""
report_generator.py
Converts raw evaluation output (scores + features) into a clean,
client-facing report with scores, labels, reasoning, strengths,
areas for improvement, and an executive summary.

Uses Gemini for natural-language reasoning.
Falls back to rule-based text if the API is unavailable.
"""
import logging
import json

from app.settings import settings

logger = logging.getLogger(__name__)

from app.shared_models import get_gemini_client

_GEMINI_MODEL  = "gemini-3.1-flash-lite"


def _label(score: float) -> str:
    if score >= 4.5: return "Excellent"
    if score >= 3.5: return "Good"
    if score >= 2.5: return "Average"
    if score >= 1.5: return "Below Average"
    return "Poor"


def _grade(overall_100: float) -> str:
    """Grade based on a 0–100 overall score."""
    if overall_100 >= 85: return "A+"
    if overall_100 >= 75: return "A"
    if overall_100 >= 65: return "B+"
    if overall_100 >= 55: return "B"
    if overall_100 >= 45: return "C"
    if overall_100 >= 35: return "D"
    return "F"
_WEIGHTS = {
    "logical_cohesion":      1.0 / 6.0,
    "delivery_fluency":      1.0 / 6.0,
    "pronunciation_clarity": 1.0 / 6.0,
    "vocal_dynamism":        1.0 / 6.0,
    "collaborative_tone":    1.0 / 6.0,
    "lexical_precision":     1.0 / 6.0,
}

_RULE_TEMPLATES = {
    "logical_cohesion": {
        "Excellent":     "Responses are exceptionally well-structured with varied, high-quality discourse markers and cohesive transitions.",
        "Good":          "Ideas are logically connected using good discourse connectors and transitions.",
        "Average":       "Some discourse connectors are used, but responses could be better structured.",
        "Below Average": "Very few transitions are used; responses feel somewhat disjointed.",
        "Poor":          "Disjointed response structure with minimal or no transitions.",
    },
    "delivery_fluency": {
        "Excellent":     "Candidate maintains a highly natural, fluent pace with minimal hesitation or filler words.",
        "Good":          "Pacing is mostly fluent with comfortable speech flow and few filler words.",
        "Average":       "Pacing is adequate but minor pauses and fillers slow down the flow.",
        "Below Average": "Frequent pauses and filler words disrupt the overall fluency.",
        "Poor":          "Very hesitant pacing with high density of filler words and long pauses.",
    },
    "pronunciation_clarity": {
        "Excellent":     "Articulation is exceptionally clear; every word is recognized with high confidence.",
        "Good":          "Speech is clearly audible and easily understood with few low-confidence words.",
        "Average":       "Speech is generally intelligible, though some words are unclear or mumbled.",
        "Below Average": "Several parts are difficult to understand, affecting articulation quality.",
        "Poor":          "Pronunciation is highly unclear, making the speech difficult to comprehend.",
    },
    "vocal_dynamism": {
        "Excellent":     "Highly dynamic delivery with excellent pitch variation and voice modulation.",
        "Good":          "Good voice modulation; pitch changes appropriately to highlight points.",
        "Average":       "Moderate vocal variety; pacing and volume are acceptable but could be more engaging.",
        "Below Average": "Somewhat flat delivery with limited pitch standard deviation.",
        "Poor":          "Monotone vocal quality with little to no expressiveness.",
    },
    "collaborative_tone": {
        "Excellent":     "Tone is highly collaborative and warm, combining positive phrasing with team-oriented pronouns.",
        "Good":          "Friendly, collaborative tone with good team pronoun usage.",
        "Average":       "Neutral tone with moderate cooperative signaling; overall professional.",
        "Below Average": "Slightly distant or highly self-centered phrasing with few collaborative signals.",
        "Poor":          "Almost zero collaborative phrasing or team-oriented pronoun usage.",
    },
    "lexical_precision": {
        "Excellent":     "Diverse, precise vocabulary with sophisticated and domain-appropriate terms.",
        "Good":          "Wide range of vocabulary with good lexical variety and minimal repetition.",
        "Average":       "Adequate vocabulary but repetitive and using mostly simple terms.",
        "Below Average": "Limited vocabulary range with significant repetition of words.",
        "Poor":          "Extremely restricted and repetitive vocabulary.",
    },
}


def _rule_based_reasoning(dim: str, label: str) -> str:
    return _RULE_TEMPLATES.get(dim, {}).get(label, f"{label} performance in {dim}.")


def _gemini_reasoning(scores: dict, features: dict) -> dict | None:
    gemini_client = get_gemini_client()
    if gemini_client is None:
        return None

    fluency   = features.get("fluency", {})
    intel     = features.get("intelligibility", {})
    grammar   = features.get("language_control", {})
    lexical   = features.get("lexical_resource", {})
    discourse = features.get("discourse", {})
    vm        = features.get("voice_modulation", {})
    sentiment = features.get("sentiment", {})

    data_summary = {
        "scores": {k: round(v, 2) for k, v in scores.items()},
        "fluency": {
            "wpm":          fluency.get("wpm"),
            "filler_count": fluency.get("filler_count"),
            "filler_words": fluency.get("filler_words_found", [])[:5],
            "pause_freq":   fluency.get("pause_frequency"),
        },
        "intelligibility": {
            "mean_confidence":       intel.get("mean_confidence"),
            "pronunciation_score":   intel.get("pronunciation_score"),
            "mispronounced_sample":  [w["word"] for w in intel.get("mispronounced_words", [])[:3]],
        },
        "language_control": {
            "grammar_errors": grammar.get("grammar_error_count"),
        },
        "lexical_resource": {
            "mattr":                lexical.get("mattr"),
            "rare_word_ratio":      lexical.get("rare_word_ratio"),
            "sophisticated_sample": lexical.get("sophisticated_words_sample", [])[:5],
        },
        "discourse": {
            "connectors_used": [str(c) for c in discourse.get("connectors_used", [])[:8]],
            "tier1_count":     discourse.get("tier1_count", 0),
        },
        "voice_modulation": {
            "pitch_std":   vm.get("pitch_std"),
            "monotone":    vm.get("monotone_flag"),
            "voiced_frac": vm.get("voiced_fraction"),
        },
        "sentiment": {
            "mean_compound":   sentiment.get("mean_compound"),
            "assertive_count": sentiment.get("assertive_count"),
            "hedge_rate":      sentiment.get("hedge_rate"),
        },
    }

    prompt = f"""You are an expert English language and communication coach evaluating a job interview candidate's spoken English and communication style in a professional context.

Below is the raw assessment data from an automated speech analysis system mapping onto a 6-Axis Communication Fingerprint model. Generate a concise, professional, and constructive evaluation report.

RAW DATA:
{json.dumps(data_summary, indent=2)}

Return ONLY a valid JSON object (no markdown, no code fences) with this exact structure:
{{
  "dimension_reasoning": {{
    "logical_cohesion": "1-2 sentence professional reasoning for the logical_cohesion score",
    "delivery_fluency": "1-2 sentence professional reasoning for the delivery_fluency score",
    "pronunciation_clarity": "1-2 sentence professional reasoning for the pronunciation_clarity score",
    "vocal_dynamism": "1-2 sentence professional reasoning for the vocal_dynamism score",
    "collaborative_tone": "1-2 sentence professional reasoning for the collaborative_tone score",
    "lexical_precision": "1-2 sentence professional reasoning for the lexical_precision score"
  }},
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "areas_for_improvement": ["area 1", "area 2", "area 3"],
  "executive_summary": "3-4 sentence overall professional summary suitable for an HR report."
}}

Rules:
- Be constructive and specific, referencing the actual numbers where appropriate.
- Use professional HR/language assessment terminology.
- Strengths and areas_for_improvement must each have exactly 3 items.
- Do not include any text outside the JSON object.
"""

    max_retries = 3
    backoff = 2
    response = None
    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=prompt,
            )
            break
        except Exception as e:
            import time
            if attempt == max_retries - 1:
                logger.error(f"Gemini reasoning failed after {max_retries} attempts: {e}")
                return None
            logger.warning(f"Gemini reasoning attempt {attempt+1} failed: {e}. Retrying in {backoff}s...")
            time.sleep(backoff)
            backoff *= 2

    try:
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        logger.error(f"Gemini reasoning failed: {e}")
        return None


def generate_report(scores: dict, features: dict, filename: str = "") -> dict:
    scorable_dims = {k: v for k, v in scores.items() if k != "overall_score"}

    total_weight = sum(_WEIGHTS.get(d, 0) for d in scorable_dims)
    weighted_avg = (
        sum(scorable_dims[d] * _WEIGHTS.get(d, 0) for d in scorable_dims) / total_weight
        if total_weight > 0 else 0.0
    )
    overall = round(weighted_avg * 20.0, 1)

    dimension_labels = {d: _label(s) for d, s in scorable_dims.items()}

    gemini_result = _gemini_reasoning(scorable_dims, features)

    if gemini_result:
        reasoning    = gemini_result.get("dimension_reasoning", {})
        strengths    = gemini_result.get("strengths", [])
        improvements = gemini_result.get("areas_for_improvement", [])
        summary      = gemini_result.get("executive_summary", "")
    else:
        reasoning    = {d: _rule_based_reasoning(d, dimension_labels[d]) for d in scorable_dims}
        strengths    = [
            d.replace("_", " ").title()
            for d, _ in sorted(scorable_dims.items(), key=lambda x: -x[1])[:3]
        ]
        improvements = [
            d.replace("_", " ").title()
            for d, _ in sorted(scorable_dims.items(), key=lambda x: x[1])[:3]
        ]
        best  = max(scorable_dims, key=scorable_dims.get)
        worst = min(scorable_dims, key=scorable_dims.get)
        summary = (
            f"The candidate scored {overall}/100 overall ({_grade(overall)}). "
            f"Their strongest area is {best.replace('_', ' ')} "
            f"({_label(scorable_dims[best])}, {scorable_dims[best]:.1f}/5) and their "
            f"primary area for development is {worst.replace('_', ' ')} "
            f"({_label(scorable_dims[worst])}, {scorable_dims[worst]:.1f}/5)."
        )

    dimensions_report = {
        dim: {
            "score":     round(score, 2),
            "out_of":    5.0,
            "label":     dimension_labels[dim],
            "reasoning": reasoning.get(dim, ""),
        }
        for dim, score in scorable_dims.items()
    }

    return {
        "file":                  filename,
        "overall_score":         overall,
        "out_of":                100.0,
        "grade":                 _grade(overall),
        "label":                 _label(weighted_avg),
        "dimensions":            dimensions_report,
        "strengths":             strengths,
        "areas_for_improvement": improvements,
        "executive_summary":     summary,
        "reasoning_source":      "gemini" if gemini_result else "rule-based",
        "features":              features,
        "scores":               scores,
    }
