"""
pipeline/corroboration.py — OR-gate corroboration logic and evidence payload builder.

Implements spec §7:

  flag_A = Track_A.deviation_score > threshold_A   (only if baseline exists)
  flag_C = Track_C.any_signal_fires

  final_flag = flag_A OR flag_C

  confidence levels:
    "high"               — both Track A AND Track C fired
    "medium"             — Track C alone (with baseline available)
    "medium"             — Track A alone (with baseline, no Track C signal)
    "medium_provisional" — Track C alone, cold-start (no Track A baseline yet)
    "low"                — neither track fired

The OR-gate is intentional: either track alone is sufficient to flag.
This preserves recall, which was the biggest weakness of the previous system
(~77% of confirmed cases missed). Corroboration sets *confidence*, not the flag.

Also runs the full interview-level pipeline:
  - Iterates questions in order
  - Updates the InterviewBaseline after each evaluable answer
  - Returns the list of QuestionEvidencePayload objects
"""
from __future__ import annotations

import logging
from typing import Optional

from v4_proctoring.models import (
    AudioFeatures,
    EvaluabilityResult,
    QuestionEvidencePayload,
    TrackAResult,
    TrackCResult,
)
from v4_proctoring.pipeline.track_a import score_all_answers
from v4_proctoring.pipeline.track_c import InterviewBaseline, analyze as track_c_analyze

logger = logging.getLogger("v4_proctoring.corroboration")


def _determine_confidence(
    flag_a: bool,
    flag_c: bool,
    track_a_available: bool,
    is_cold_start: bool,
) -> str:
    """Compute the confidence level for a flagged or unflagged answer."""
    if flag_a and flag_c:
        return "high"
    if flag_c and is_cold_start:
        return "medium_provisional"
    if flag_c:
        return "medium"
    if flag_a:
        return "medium"
    return "low"


def _build_contributing_features(
    features: AudioFeatures,
    track_c: TrackCResult,
) -> dict[str, Optional[float]]:
    """
    Build the contributing_features dict for the reviewer UI.

    Only includes features that are directly interpretable by a human reviewer.
    Does not dump the full feature vector — that would be overwhelming.
    """
    contrib: dict[str, Optional[float]] = {
        "response_latency_s": features.response_latency,
        "spectral_flatness_mean": features.spectral_flatness_mean,
        "pause_ratio": features.pause_ratio,
        "f0_mean_hz": features.f0_mean,
        "f0_std_hz": features.f0_std,
        "speech_rate_proxy": features.speech_rate_proxy,
        "energy_mean": features.energy_mean,
        "speech_duration_s": features.speech_duration_s,
        "total_duration_s": features.total_duration_s,
    }
    # Add room/noise details if any Track C environment signal fired
    if "acoustic_environment_shift" in track_c.signals or "background_noise_shift" in track_c.signals:
        contrib["room_fingerprint"] = features.room_fingerprint
        contrib["noise_floor_centroid"] = features.noise_floor_centroid
    return contrib


def build_interview_evidence(
    question_data: list[dict],
) -> list[QuestionEvidencePayload]:
    """
    Run the full corroboration pipeline for one interview.

    Args:
        question_data: List of dicts, each containing:
            {
                "q_no": int,
                "evaluability": EvaluabilityResult,
                "features": Optional[AudioFeatures],   # None if not evaluable
            }
            Must be sorted by q_no ascending.

    Returns:
        List of QuestionEvidencePayload objects (one per question).
    """
    # ── Step 1: Run Track A across all questions ───────────────────────────────
    # Track A needs all feature vectors at once (requires the full candidate history).
    features_list: list[Optional[AudioFeatures]] = [
        d["features"] for d in question_data
    ]
    track_a_results: list[TrackAResult] = score_all_answers(features_list)

    # ── Step 2: Run Track C question by question, updating the baseline ────────
    baseline = InterviewBaseline()
    track_c_results: list[TrackCResult] = []

    for d in question_data:
        features = d["features"]
        if features is None:
            # Non-evaluable: still record an empty Track C result
            track_c_results.append(TrackCResult(flagged=False, signals=[]))
        else:
            # Run Track C BEFORE updating the baseline so the current answer
            # is compared against the history of *prior* answers only.
            tc_result = track_c_analyze(features, baseline)
            track_c_results.append(tc_result)
            # Update baseline AFTER scoring
            baseline.update(features)

    # ── Step 3: Build evidence payloads ───────────────────────────────────────
    payloads: list[QuestionEvidencePayload] = []

    for i, d in enumerate(question_data):
        q_no = d["q_no"]
        evaluability: EvaluabilityResult = d["evaluability"]
        features: Optional[AudioFeatures] = d["features"]
        ta: TrackAResult = track_a_results[i]
        tc: TrackCResult = track_c_results[i]

        if not evaluability.evaluable:
            # Non-evaluable: produce an explicit reasoned payload (spec §8).
            payloads.append(
                QuestionEvidencePayload(
                    q_no=q_no,
                    evaluable=False,
                    not_evaluable_reason=evaluability.not_evaluable_reason,
                    flagged_for_review=False,
                    confidence="low",
                    track_a=None,
                    track_c=None,
                    is_cold_start=not ta.available,
                )
            )
            continue

        # Evaluable answer: apply OR-gate
        flag_a: bool = ta.available and ta.flagged
        flag_c: bool = tc.flagged
        final_flag = flag_a or flag_c
        is_cold_start = not ta.available

        confidence = _determine_confidence(
            flag_a=flag_a,
            flag_c=flag_c,
            track_a_available=ta.available,
            is_cold_start=is_cold_start,
        )

        if final_flag:
            logger.info(
                "Q%d FLAGGED — flag_a=%s, flag_c=%s, confidence=%s, cold_start=%s",
                q_no, flag_a, flag_c, confidence, is_cold_start,
            )

        contributing = _build_contributing_features(features, tc) if features else {}

        payloads.append(
            QuestionEvidencePayload(
                q_no=q_no,
                evaluable=True,
                not_evaluable_reason=None,
                flagged_for_review=final_flag,
                confidence=confidence if final_flag else "low",
                track_a=ta,
                track_c=tc,
                is_cold_start=is_cold_start,
                contributing_features=contributing,
            )
        )

    return payloads
