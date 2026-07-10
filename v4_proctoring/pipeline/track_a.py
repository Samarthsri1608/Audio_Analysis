"""
pipeline/track_a.py — Self-baseline deviation detection (Track A).

Goal: flag an answer that behaves very differently from how this candidate
delivers their *other* answers in the same interview.

Method (per spec §4):
  1. Collect feature vectors from all evaluable answers.
  2. Compute robust baseline per feature: median + MAD (not mean/std).
  3. Compute robust z-score per feature: z = 0.6745 * (x - median) / MAD
  4. Per-answer deviation score = max |z-score| across features.
  5. Flag if deviation_score > TRACK_A_THRESHOLD.

Cold-start:
  If a candidate has fewer than TRACK_A_MIN_ANSWERS evaluable answers, Track A
  does not fire — returns available=False for all answers in that batch.
  This is by design: there's no reliable baseline with too few data points.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from v4_proctoring.config import TRACK_A_THRESHOLD, TRACK_A_FEATURE_KEYS, TRACK_A_MIN_ANSWERS
from v4_proctoring.models import AudioFeatures, TrackAResult

logger = logging.getLogger("v4_proctoring.track_a")


def score_all_answers(
    features_list: list[Optional[AudioFeatures]],
    min_history: int = TRACK_A_MIN_ANSWERS,
) -> list[TrackAResult]:
    """
    Compute Track A deviation scores for all evaluable answers of one candidate.

    Args:
        features_list: Per-question AudioFeatures in question order. None entries
                       represent non-evaluable questions — they are excluded from
                       the baseline pool AND receive available=False in output.
        min_history:   Minimum evaluable answers needed before Track A fires.

    Returns:
        List of TrackAResult objects aligned 1-to-1 with features_list.
        Non-evaluable positions (None in features_list) always return
        TrackAResult(available=False).
    """
    n = len(features_list)

    # Identify evaluable indices (non-None features)
    evaluable_indices = [i for i, f in enumerate(features_list) if f is not None]

    # Cold-start: not enough evaluable answers for a reliable baseline
    if len(evaluable_indices) < min_history:
        logger.info(
            "Track A: cold-start — only %d evaluable answers (need %d)",
            len(evaluable_indices), min_history,
        )
        return [TrackAResult(available=False) for _ in range(n)]

    # Build feature matrix: shape (n_evaluable, n_features)
    # Any individual feature that is None is treated as NaN and excluded
    # from that feature's baseline computation — avoids zero-masking bug.
    def _get_value(feat: AudioFeatures, key: str) -> float:
        val = getattr(feat, key, None)
        return float(val) if val is not None else np.nan

    evaluable_features = [features_list[i] for i in evaluable_indices]
    matrix = np.array(
        [[_get_value(f, k) for k in TRACK_A_FEATURE_KEYS] for f in evaluable_features],
        dtype=float,
    )  # shape: (n_evaluable, n_features)

    # Compute per-feature median and MAD using only non-NaN values
    medians = np.nanmedian(matrix, axis=0)  # (n_features,)
    abs_deviations = np.abs(matrix - medians)
    mads = np.nanmedian(abs_deviations, axis=0)  # (n_features,)
    # Avoid division by zero: if MAD=0 for a feature, substitute a small epsilon
    mads_safe = np.where(mads == 0, 1e-6, mads)

    # Robust z-scores: 0.6745 * (x - median) / MAD
    robust_z = 0.6745 * (matrix - medians) / mads_safe

    # Build results: per evaluable answer, max |z| across all features
    # (ignore NaN features when computing max)
    evaluable_results: list[TrackAResult] = []
    for i in range(len(evaluable_indices)):
        row = robust_z[i]
        abs_row = np.abs(row)

        # Build per-feature z-score dict (None for features that were NaN)
        z_scores: dict[str, Optional[float]] = {}
        for j, key in enumerate(TRACK_A_FEATURE_KEYS):
            raw_val = matrix[i, j]
            z_scores[key] = None if np.isnan(raw_val) else round(float(row[j]), 4)

        # Max absolute z, ignoring NaN
        valid_abs = abs_row[~np.isnan(abs_row)]
        if len(valid_abs) == 0:
            deviation_score = None
            flagged = False
        else:
            deviation_score = float(np.max(valid_abs))
            flagged = deviation_score > TRACK_A_THRESHOLD

        evaluable_results.append(
            TrackAResult(
                available=True,
                deviation_score=round(deviation_score, 4) if deviation_score is not None else None,
                flagged=flagged,
                z_scores=z_scores,
            )
        )
        if flagged:
            logger.info(
                "Track A: flagged evaluable answer #%d — deviation_score=%.3f",
                evaluable_indices[i] + 1,
                deviation_score,
            )

    # Map evaluable results back to the full question list
    output: list[TrackAResult] = [TrackAResult(available=False) for _ in range(n)]
    for idx, result in zip(evaluable_indices, evaluable_results):
        output[idx] = result

    return output
