"""
pipeline/track_a.py — Self-baseline deviation detection (Track A).

Goal: flag an individual question whose response_latency is anomalously high
compared to this candidate's own baseline latency across the interview.

Method:
  1. Collect response_latency values from all evaluable answers.
  2. Compute robust baseline: median + MAD (not mean/std — outlier-resistant).
  3. Per-question robust z-score: z = clip(0.6745 * (x - median) / MAD, -8, 8)
     - The 0.6745 constant makes MAD-based z-scores comparable to std-based ones
       for normally distributed data.
     - Clipping to [-8, 8] prevents blow-up when MAD approaches zero
       (previously produced scores in the hundreds/thousands — now capped).
     - MAD == 0 guard: if all baseline latencies are identical (MAD = 0),
       the z-score is set to 0 (no deviation detectable).
  4. Flag the question if z >= TRACK_A_THRESHOLD (default 1.75).
     Threshold is configurable via env var — see config.py.

Scoring is per-question, not per-interview:
  Each question gets its own flagged=True/False. The interview-level
  flagged_questions list is the union of all per-question flags.
  This gives proctors the specific question number(s) to verify.

Cold-start:
  If the candidate has fewer than TRACK_A_MIN_ANSWERS evaluable answers,
  Track A does not fire — all questions return available=False.
  By design: the baseline is too sparse to be reliable.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from v4_proctoring.config import (
    TRACK_A_THRESHOLD,
    TRACK_A_FEATURE_KEY,
    TRACK_A_MIN_ANSWERS,
    TRACK_A_Z_CLIP,
)
from v4_proctoring.models import AudioFeatures, TrackAResult

logger = logging.getLogger("v4_proctoring.track_a")


def score_all_answers(
    features_list: list[Optional[AudioFeatures]],
    min_history: int = TRACK_A_MIN_ANSWERS,
) -> list[TrackAResult]:
    """
    Compute Track A latency z-scores for every evaluable answer in one interview.

    Args:
        features_list: Per-question AudioFeatures in question order. None entries
                       represent non-evaluable questions — they are excluded from
                       the baseline pool AND receive available=False in output.
        min_history:   Minimum evaluable answers needed before Track A fires.

    Returns:
        List of TrackAResult objects aligned 1-to-1 with features_list.
        Non-evaluable positions (None) always return TrackAResult(available=False).
    """
    n = len(features_list)

    # ── Identify evaluable indices ────────────────────────────────────────────
    evaluable_indices = [i for i, f in enumerate(features_list) if f is not None]

    # ── Cold-start guard ──────────────────────────────────────────────────────
    if len(evaluable_indices) < min_history:
        logger.info(
            "Track A: cold-start — only %d evaluable answers (need %d)",
            len(evaluable_indices), min_history,
        )
        return [TrackAResult(available=False) for _ in range(n)]

    # ── Extract single feature: response_latency ──────────────────────────────
    latencies: list[Optional[float]] = []
    for i in evaluable_indices:
        feat = features_list[i]
        val = getattr(feat, TRACK_A_FEATURE_KEY, None)
        latencies.append(float(val) if val is not None else None)

    # Filter to non-None values for baseline computation
    valid_latencies = np.array([v for v in latencies if v is not None], dtype=float)

    # ── Baseline: median + MAD ────────────────────────────────────────────────
    median_lat = float(np.median(valid_latencies))
    mad = float(np.median(np.abs(valid_latencies - median_lat)))

    # ── MAD floor ───────────────────────────────────────────────
    # MAD can be zero when many answers share the same latency (e.g. all near 0s
    # when the API returns pre-recorded responses). Using a hard zero collapses the
    # z-score to 0 for all answers, hiding genuine outliers.
    # Floor at 0.1s = conservative minimum spread; well below typical human
    # answer-to-answer variation (~0.3–1.5s), so it won't over-flag on clean data.
    MAD_FLOOR = 0.1  # seconds
    mad_effective = max(mad, MAD_FLOOR)

    # ── Score each evaluable answer ───────────────────────────────────────────
    evaluable_results: list[TrackAResult] = []
    for latency in latencies:
        if latency is None:
            # Feature not computable for this question
            z = 0.0
        else:
            raw_z = 0.6745 * (latency - median_lat) / mad_effective
            z = float(np.clip(raw_z, -TRACK_A_Z_CLIP, TRACK_A_Z_CLIP))

        flagged = z >= TRACK_A_THRESHOLD

        evaluable_results.append(
            TrackAResult(
                available=True,
                deviation_score=round(z, 4),
                flagged=flagged,
                z_scores={TRACK_A_FEATURE_KEY: round(z, 4)},
            )
        )

    # ── Log flagged questions ─────────────────────────────────────────────────
    for seq_i, (q_idx, result) in enumerate(zip(evaluable_indices, evaluable_results)):
        if result.flagged:
            logger.info(
                "Track A: Q%d flagged — latency_z=%.3f (threshold=%.2f, "
                "latency=%.2fs, median_baseline=%.2fs, mad=%.3f)",
                q_idx + 1,
                result.deviation_score,
                TRACK_A_THRESHOLD,
                latencies[seq_i] if latencies[seq_i] is not None else -1.0,
                median_lat,
                mad,
            )

    # ── Map results back to the full question list ────────────────────────────
    output: list[TrackAResult] = [TrackAResult(available=False) for _ in range(n)]
    for q_idx, result in zip(evaluable_indices, evaluable_results):
        output[q_idx] = result

    return output
