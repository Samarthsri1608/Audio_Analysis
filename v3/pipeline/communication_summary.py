"""
communication_summary.py — Generate a compact two-line communication summary
using Gemini.

This is the V3 communication endpoint summary path. It consumes the extracted
raw features plus the System A score object, then returns exactly two lines of
candidate-facing summary text in a structured model.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re

from google import genai
from google.genai import types

from v3.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    COMMUNICATION_SUMMARY_MAX_TOKENS,
)
from v3.models import CommunicationSummary, RawFeatures, SkillsAssessment

logger = logging.getLogger("v3.communication_summary")

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise ValueError("Gemini API key is not configured.")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _feature_payload(raw: RawFeatures, skills: SkillsAssessment) -> dict:
    return {
        "communication_axes": {
            "fluency": skills.fluency.model_dump(),
            "intelligibility": skills.intelligibility.model_dump(),
            "lexical_structural": skills.lexical_structural.model_dump(),
            "narrative_evidence": skills.narrative_evidence.model_dump(),
            "vocal_delivery": skills.vocal_delivery.model_dump(),
            "composite_score": skills.composite_score,
            "composite_band": skills.composite_band,
            "review_required": skills.review_required,
            "role_profile": skills.role_profile,
        },
        "raw_communication_features": {
            "speech_rate_wpm": raw.speech_rate_wpm,
            "speech_rate_variability": raw.speech_rate_variability,
            "filler_word_ratio": raw.filler_word_ratio,
            "lexical_mattr": raw.lexical_mattr,
            "lexical_rare_word_ratio": raw.lexical_rare_word_ratio,
            "total_words": raw.total_words,
            "discourse_connectors": raw.discourse_connectors,
            "discourse_tier1": raw.discourse_tier1,
            "sbert_coherence": raw.sbert_coherence,
            "ner_entity_density": raw.ner_entity_density,
            "metric_density": raw.metric_density,
            "narrative_arc_score": raw.narrative_arc_score,
            "collaborative_language_ratio": raw.collaborative_language_ratio,
            "question_density": raw.question_density,
            "empathetic_language_score": raw.empathetic_language_score,
            "avg_sentence_length": raw.avg_sentence_length,
            "pitch_variation": raw.pitch_variation,
            "vocal_confidence": raw.vocal_confidence,
            "speech_fluency": raw.speech_fluency,
            "stress_markers": raw.stress_markers,
            "fluency_pause_dur": raw.fluency_pause_dur,
            "fluency_pause_freq": raw.fluency_pause_freq,
            "voiced_fraction": raw.voiced_fraction,
            "intel_confidence": raw.intel_confidence,
            "is_short_duration": raw.is_short_duration,
        },
    }


def _build_prompt(raw: RawFeatures, skills: SkillsAssessment) -> str:
    payload = json.dumps(_feature_payload(raw, skills), ensure_ascii=False, indent=2)
    return f"""
        You are a senior communication analyst writing a concise candidate report.

        Task:
        - Analyze the candidate's metrics and write exactly 2 specific, candidate-facing sentences.
        - Line 1 MUST describe the candidate's overall speech flow, pacing, and delivery style, referencing concrete patterns (e.g. structured delivery, steady pacing, or lexical range).
        - Line 2 MUST highlight a specific observable strength or area for development (e.g. minimal filler word usage, strong structural connectivity, or pacing variation/pauses in extended responses).
        - Each line must be a single, complete, well-formed sentence.
        - The summary must be highly specific, professional, and directly reflect this candidate's raw data.
        - The "summary" array must contain exactly two items: index 0 is the first sentence and index 1 is the second sentence.
        - Do not use vague or generic statements like "The candidate has good communication skills" or "The candidate is clear."
        - Do not mention exact numeric values, raw feature names, scores, bands, transcripts, or accents.
        - Do not use bullet points, list markers, or labels.
        - Write in a neutral, professional, candidate-facing tone.

        Candidate data:
        {payload}
        """.strip()


def _fallback_summary(raw: RawFeatures, skills: SkillsAssessment) -> CommunicationSummary:
    if skills.composite_score >= 80:
        line1 = "The candidate communicates with strong control and clear intent, making the core message easy to follow."
    elif skills.composite_score >= 65:
        line1 = "The candidate communicates with generally steady control, keeping the message readable and purposeful."
    elif skills.composite_score >= 50:
        line1 = "The candidate communicates adequately, but the delivery loses polish when the response becomes longer."
    else:
        line1 = "The candidate’s delivery is uneven enough that the core message does not always land cleanly."

    if raw.filler_word_ratio > 0.07 or skills.fluency.score < 3.0:
        line2 = "The response softens in longer stretches, where pacing, sentence shaping, and repetition become more noticeable."
    elif raw.lexical_mattr >= 0.75 and skills.lexical_structural.score >= 3.5:
        line2 = "The response stays strongest when it remains concrete and organized, with good vocabulary range supporting the explanation."
    else:
        line2 = "The response is strongest when it stays concise and structured, with only mild drift when ideas become more extended."

    return CommunicationSummary(summary=[line1, line2])


def _parse_summary_content(content: str) -> CommunicationSummary | None:
    text = content.strip()
    if not text:
        return None

    # Strip common markdown fences if the model adds them.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    # First try strict JSON.
    try:
        payload = json.loads(text)
        summary = payload.get("summary")
        if isinstance(summary, list):
            return CommunicationSummary(summary=summary)
        if isinstance(payload.get("line_1"), str) and isinstance(payload.get("line_2"), str):
            return CommunicationSummary(summary=[payload["line_1"], payload["line_2"]])
    except Exception:
        pass

    # Then try to recover a JSON object embedded inside text.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            summary = payload.get("summary")
            if isinstance(summary, list):
                return CommunicationSummary(summary=summary)
            if isinstance(payload.get("line_1"), str) and isinstance(payload.get("line_2"), str):
                return CommunicationSummary(summary=[payload["line_1"], payload["line_2"]])
        except Exception:
            pass

    # Finally, treat the first two non-empty lines as the summary.
    lines = [line.strip("-• \t") for line in text.splitlines() if line.strip()]
    if len(lines) >= 2:
        return CommunicationSummary(summary=lines[:2])

    return None


async def generate_communication_summary(raw: RawFeatures, skills: SkillsAssessment) -> CommunicationSummary:
    """
    Generate the final two-line communication summary for the V3 endpoint.
    """
    try:
        client = _get_client()
    except Exception as exc:
        logger.warning("Gemini config missing — using fallback summary: %s", exc)
        return _fallback_summary(raw, skills)

    prompt = _build_prompt(raw, skills)

    try:
        response = None
        max_retries = 3
        retry_delay = 1.0
        for attempt in range(max_retries + 1):
            try:
                response = await client.aio.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=(
                            "You are a talent intelligence engine. Your output must be a single, valid JSON object "
                            "containing exactly one 'summary' key mapping to a list of two strings. "
                            "Do NOT include any markdown formatting, code block fences, preambles, or conversational commentary. "
                            "The summary list must contain exactly two specific and professional sentences."
                        ),
                        temperature=0.5,
                        max_output_tokens=COMMUNICATION_SUMMARY_MAX_TOKENS,
                        response_mime_type="application/json",
                        response_schema=CommunicationSummary,
                    )
                )
                break
            except Exception as exc:
                exc_str = str(exc)
                is_temporary = any(k in exc_str for k in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "demand"])
                if is_temporary and attempt < max_retries:
                    sleep_time = retry_delay * (2 ** attempt)
                    logger.warning(
                        "Gemini API returned temporary error (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt + 1, max_retries + 1, exc, sleep_time
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    raise exc

        content = response.text or "{}"
        parsed = _parse_summary_content(content)
        if parsed is not None:
            return parsed
        raise ValueError(f"Gemini response did not contain a valid summary array: {content[:200]!r}")
    except Exception as exc:
        logger.warning("Gemini summary generation failed — using fallback summary: %s", exc)
        return _fallback_summary(raw, skills)
