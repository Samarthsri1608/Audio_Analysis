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
    "fluency":          0.26,
    "intelligibility":  0.26,
    "language_control": 0.12,
    "lexical_resource": 0.16,
    "discourse":        0.10,
    "sentiment":        0.05,
    "voice_modulation": 0.05,
}

_RULE_TEMPLATES = {
    "fluency": {
        "Excellent":     "Speaker maintains a natural, confident pace with minimal hesitation or filler words.",
        "Good":          "Speech is mostly fluent with a comfortable pace. Minor use of filler words detected.",
        "Average":       "Pacing is adequate but noticeable pauses or filler words slow the delivery.",
        "Below Average": "Frequent pauses and filler words significantly disrupt the flow of speech.",
        "Poor":          "Very halting delivery with excessive fillers and long pauses throughout.",
    },
    "intelligibility": {
        "Excellent":     "Pronunciation is exceptionally clear — every word was recognised with high confidence.",
        "Good":          "Speech is clear and easily understood. A small number of words were less distinct.",
        "Average":       "Generally understandable, though some words were unclear or mispronounced.",
        "Below Average": "Several words were difficult to understand, affecting overall clarity.",
        "Poor":          "Pronunciation is significantly unclear, making the speech hard to follow.",
    },
    "language_control": {
        "Excellent":     "Grammatically precise throughout with no notable errors detected.",
        "Good":          "Minor grammatical slips only — grammar is largely accurate and natural.",
        "Average":       "Some grammatical errors present but they do not impede understanding.",
        "Below Average": "Frequent grammatical errors that occasionally make the meaning unclear.",
        "Poor":          "Grammar errors are pervasive and significantly hinder comprehension.",
    },
    "lexical_resource": {
        "Excellent":     "Diverse, sophisticated vocabulary with strong use of domain-appropriate and rare words.",
        "Good":          "Wide vocabulary range with effective use of varied and sometimes advanced words.",
        "Average":       "Adequate vocabulary but with some repetition and limited use of advanced words.",
        "Below Average": "Limited vocabulary range with noticeable repetition of the same words.",
        "Poor":          "Very restricted vocabulary — heavily repetitive with little lexical diversity.",
    },
    "discourse": {
        "Excellent":     "Responses are exceptionally well-structured with varied, high-quality discourse markers.",
        "Good":          "Ideas are logically connected using good discourse connectors.",
        "Average":       "Some discourse connectors used, but responses could be better structured.",
        "Below Average": "Very few connectors — responses feel disjointed and lack clear logical flow.",
        "Poor":          "No meaningful discourse structure — ideas are presented without organisation.",
    },
    "voice_modulation": {
        "Excellent":     "Dynamic, expressive delivery with excellent pitch variation — very engaging to listen to.",
        "Good":          "Good vocal variety. Pitch changes appropriately to emphasise key points.",
        "Average":       "Moderate vocal variation. Delivery is acceptable but could be more engaging.",
        "Below Average": "Somewhat monotone delivery — limited pitch variation makes speech less engaging.",
        "Poor":          "Completely flat, monotone delivery with almost no vocal expressiveness.",
    },
    "sentiment": {
        "Excellent":     "Highly confident, assertive, and professionally positive tone throughout.",
        "Good":          "Professional and largely confident delivery with minimal hedging.",
        "Average":       "Neutral-to-positive tone with some hedging language; overall acceptable.",
        "Below Average": "Noticeable lack of confidence — frequent hedging and uncertain language.",
        "Poor":          "Speech dominated by hedging, self-doubt, or negative tone.",
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

    prompt = f"""You are an expert English language and communication coach evaluating a job interview candidate's spoken English in an Indian professional context.

Below is the raw assessment data from an automated speech analysis system. Generate a concise, professional, and constructive evaluation report.

RAW DATA:
{json.dumps(data_summary, indent=2)}

Return ONLY a valid JSON object (no markdown, no code fences) with this exact structure:
{{
  "dimension_reasoning": {{
    "fluency": "1-2 sentence professional reasoning for the fluency score",
    "intelligibility": "1-2 sentence professional reasoning for the intelligibility score",
    "language_control": "1-2 sentence professional reasoning for the language_control score",
    "lexical_resource": "1-2 sentence professional reasoning for the lexical_resource score",
    "discourse": "1-2 sentence professional reasoning for the discourse score",
    "voice_modulation": "1-2 sentence professional reasoning for the voice_modulation score",
    "sentiment": "1-2 sentence professional reasoning for the sentiment score"
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
