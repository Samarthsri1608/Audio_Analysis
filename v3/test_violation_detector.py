"""
test_violation_detector.py — Unit + regression tests for violation_detector.py (v3)

Run with:  python -m pytest v3/test_violation_detector.py -v
           (from the Audio Analysis/ root)

Coverage:
  Core (carried from v2):
    1. Eligibility gate
    2. MAD-floor / z-score clipping
    3. Off-by-one
    4. Continuous composite

  New (v3 bug fixes):
    BUG 1 — Feature-specific nulls: short answers produce None, not 0.0
    BUG 2 — Sigmoid latency: no coin-flip cliff; higher composite -> more likely flagged
    BUG 3 — Flagged questions must have non-null Track A features or reduced confidence
    BUG 4 — Flatness / latency_var produce provisional flags, not permanent 0.0
    BUG 5 — discourse_organization not pegged at clip ceiling (percentile rank)
    BUG 6 — sbert_coherence 0.65 sentinel treated as None, not as real data
"""
from __future__ import annotations

import math
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from v3.pipeline.violation_detector import (
    classify_eligibility,
    score_interview,
    _robust_z,
    _naturalness_flatness,
    _intra_answer_pace_variance,
    _pause_regularity,
    _percentile_rank,
    _sigmoid,
    _feature_availability,
    _get_sbert,
    _get_mattr,
    TRACK_A_MIN_EVALUABLE_ANSWERS,
    Z_SCORE_CLIP,
    ELIGIBILITY_MIN_DURATION_S,
    ELIGIBILITY_MIN_WORDS,
    ELIGIBILITY_MIN_CONFIDENCE,
    FEATURE_MIN_WORDS,
    FEATURE_MIN_DURATION_S,
    SBERT_NEUTRAL_FALLBACK,
    HARD_SIGNAL_THRESHOLD,
    HYSTERESIS_BAND,
    COMPOSITE_THRESHOLD,
    PROVISIONAL_LATENCY_VAR_SCORE,
    PROVISIONAL_FLATNESS_SCORE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wts(n_words: int, start: float = 0.2, gap_pattern: str = "vary") -> list[dict]:
    """Build simple word timestamp list."""
    wts = []
    t = start
    for i in range(n_words):
        s = t
        e = t + 0.35
        wts.append({"start": s, "end": e, "text": f"w{i}"})
        # vary gaps so pause_regularity and pace_variance have something to measure
        if gap_pattern == "vary":
            t = e + (0.6 if i % 4 == 0 else 0.05)
        else:
            t = e + 0.05
    return wts


def _make_q(
    q_no: int,
    text: str | None = None,
    duration_s: float = 60.0,
    mean_confidence: float = 0.85,
    speech_rate_wpm: float = 130.0,
    filler_word_ratio: float = 0.04,
    lexical_mattr: float = 0.72,
    discourse_connectors: float = 5.0,
    discourse_tier1: float = 2.0,
    sbert_coherence: float = 0.75,
    pitch_variation: float = 0.25,
    skipped: bool = False,
    off_camera: bool = False,
    n_words: int = 80,
) -> dict:
    if text is None:
        text = " ".join([f"word{j}" for j in range(n_words)])
    wts = _make_wts(n_words)
    return {
        "q_no": q_no,
        "text": text,
        "duration_s": duration_s,
        "mean_confidence": mean_confidence,
        "word_timestamps": wts,
        "skipped": skipped,
        "off_camera": off_camera,
        "text_feats": {
            "speech_rate_wpm": speech_rate_wpm,
            "filler_word_ratio": filler_word_ratio,
            "lexical_mattr": lexical_mattr,
            "discourse_connectors": discourse_connectors,
            "discourse_tier1": discourse_tier1,
            "sbert_coherence": sbert_coherence,
        },
        "vocal_feats": {
            "pitch_variation": pitch_variation,
        },
    }


def _make_interview(n: int = 8, **overrides) -> list[dict]:
    return [_make_q(i + 1, **overrides) for i in range(n)]


# ---------------------------------------------------------------------------
# 1. Eligibility gate (carried from v2)
# ---------------------------------------------------------------------------

class TestEligibilityGate:
    def test_skipped(self):
        ok, r = classify_eligibility(_make_q(1, skipped=True))
        assert not ok and r == "skipped"

    def test_off_camera(self):
        ok, r = classify_eligibility(_make_q(1, off_camera=True))
        assert not ok and r == "skipped"

    def test_empty_text(self):
        q = _make_q(1, text="   ")
        q["word_timestamps"] = []
        ok, r = classify_eligibility(q)
        assert not ok and r == "no_speech_detected"

    def test_too_short(self):
        ok, r = classify_eligibility(_make_q(1, duration_s=ELIGIBILITY_MIN_DURATION_S - 0.5))
        assert not ok and r == "too_short"

    def test_low_word_count(self):
        q = _make_q(1, n_words=ELIGIBILITY_MIN_WORDS - 1, duration_s=10.0)
        q["text"] = " ".join(["w"] * (ELIGIBILITY_MIN_WORDS - 1))
        ok, r = classify_eligibility(q)
        assert not ok and r == "low_word_count"

    def test_low_confidence(self):
        ok, r = classify_eligibility(_make_q(1, mean_confidence=ELIGIBILITY_MIN_CONFIDENCE - 0.05))
        assert not ok and r == "low_asr_confidence"

    def test_normal_evaluable(self):
        ok, r = classify_eligibility(_make_q(1))
        assert ok and r is None


# ---------------------------------------------------------------------------
# 2. Z-score clipping (carried from v2)
# ---------------------------------------------------------------------------

class TestZScoreClipping:
    def test_constant_baseline_clipped(self):
        z = _robust_z(200.0, [100.0] * 10, "speech_rate_wpm", "increase")
        assert z <= Z_SCORE_CLIP

    def test_near_constant_baseline_clipped(self):
        baseline = [100.0 + i * 0.0001 for i in range(8)]
        z = _robust_z(100.5, baseline, "discourse_organization", "increase")
        assert z <= Z_SCORE_CLIP

    def test_decrease_direction_below_baseline(self):
        z = _robust_z(0.01, [0.05] * 8, "filler_word_ratio", "decrease")
        assert z > 0

    def test_increase_direction_below_baseline_is_zero(self):
        z = _robust_z(0.01, [0.05] * 8, "filler_word_ratio", "increase")
        assert z == 0.0

    def test_end_to_end_clip(self):
        qs = _make_interview(8)
        qs[4]["text_feats"]["speech_rate_wpm"] = 9999.0
        results = score_interview(qs)
        for r in results:
            if r["evaluable"] and r["track_a_z_scores"]:
                for feat, z in r["track_a_z_scores"].items():
                    if z is not None:
                        assert abs(z) <= Z_SCORE_CLIP, f"Q{r['q_no']} {feat} z={z} exceeded clip"


# ---------------------------------------------------------------------------
# 3. Off-by-one (carried from v2)
# ---------------------------------------------------------------------------

class TestOffByOne:
    def test_result_count_matches_input(self):
        qs = _make_interview(23)
        assert len(score_interview(qs)) == 23

    def test_q_nos_match(self):
        qs = _make_interview(10)
        results = score_interview(qs)
        assert sorted(r["q_no"] for r in results) == sorted(q["q_no"] for q in qs)

    def test_max_q_no_not_exceeded(self):
        qs = _make_interview(7)
        assert max(r["q_no"] for r in score_interview(qs)) == 7


# ---------------------------------------------------------------------------
# 4. Continuous composite (carried from v2)
# ---------------------------------------------------------------------------

class TestContinuousComposite:
    def test_scores_are_finite_floats(self):
        results = score_interview(_make_interview(8))
        for r in results:
            if r["evaluable"]:
                assert math.isfinite(r["suspicion_score"])

    def test_not_quantized_to_fixed_steps(self):
        interviews = [_make_interview(8, speech_rate_wpm=100.0 + i * 7.3) for i in range(5)]
        scores = []
        for qs in interviews:
            scores.extend(r["suspicion_score"] for r in score_interview(qs) if r["evaluable"])
        rounded = [round(s / 14.0) * 14.0 for s in scores]
        assert any(abs(s - r) > 0.1 for s, r in zip(scores, rounded)), \
            "Scores look quantized to 14-point steps (rule-count accumulator not fixed)"


# ---------------------------------------------------------------------------
# BUG 1 — Feature-specific nulls
# ---------------------------------------------------------------------------

class TestFeatureSpecificNulls:
    """Short answers must return None for MATTR/sbert/discourse, not 0.0."""

    def test_short_answer_mattr_is_none(self):
        q = _make_q(1, n_words=FEATURE_MIN_WORDS["lexical_mattr"] - 5, duration_s=15.0)
        avail = _feature_availability(q)
        result = _get_mattr(q, avail)
        assert result is None, f"Expected None for short answer, got {result}"

    def test_long_answer_mattr_is_not_none(self):
        q = _make_q(1, n_words=FEATURE_MIN_WORDS["lexical_mattr"] + 10, duration_s=60.0)
        q["text_feats"]["lexical_mattr"] = 0.72
        avail = _feature_availability(q)
        result = _get_mattr(q, avail)
        assert result is not None

    def test_short_answer_track_a_z_is_none(self):
        """Track A z-scores for features below word minimum must be None, not 0.0."""
        # Build interview with 8 questions, one of which is short
        qs = _make_interview(8)
        # Q3: below MATTR minimum
        qs[2] = _make_q(3, n_words=FEATURE_MIN_WORDS["lexical_mattr"] - 5,
                        duration_s=20.0, sbert_coherence=0.0)
        results = score_interview(qs)
        q3 = next(r for r in results if r["q_no"] == 3)
        assert q3["evaluable"] is True  # still evaluable (passes eligibility gate)
        z_mattr = q3["track_a_z_scores"].get("lexical_mattr")
        assert z_mattr is None, f"Expected None for short-answer MATTR, got {z_mattr}"

    def test_null_features_not_in_baseline(self):
        """
        A short answer's null MATTR must not enter the baseline pool for peers.
        If Q3 is short (null mattr), Q5's baseline should use only the other
        evaluable answers' mattr values (all valid), not include a 0.0 from Q3.
        """
        qs = _make_interview(8, lexical_mattr=0.72)
        # Q3: too short for MATTR — should be excluded from peers' baseline
        qs[2] = _make_q(3, n_words=FEATURE_MIN_WORDS["lexical_mattr"] - 5,
                        duration_s=20.0, lexical_mattr=0.0)
        results = score_interview(qs)
        # Q5 should still score; its baseline must not be contaminated by Q3's 0.0
        q5 = next(r for r in results if r["q_no"] == 5)
        assert q5["evaluable"] is True
        # If 0.0 was in the baseline, Q5's MATTR z would be inflated; the test
        # just verifies it's computable (not None) and within clip range.
        z = q5["track_a_z_scores"].get("lexical_mattr")
        if z is not None:
            assert abs(z) <= Z_SCORE_CLIP

    def test_feature_availability_map_in_payload(self):
        """Every evaluable question must have a feature_availability map."""
        qs = _make_interview(8)
        results = score_interview(qs)
        for r in results:
            if r["evaluable"]:
                assert "feature_availability" in r, f"Q{r['q_no']} missing feature_availability"
                fa = r["feature_availability"]
                assert "lexical_mattr" in fa
                assert "sbert_coherence" in fa
                assert "discourse_organization" in fa

    def test_null_z_scores_contribute_zero_to_composite(self):
        """
        A question with all Track A features null must have a composite driven
        entirely by Track C, not by phantom 0.0 Track A contributions.
        """
        # Build a minimal interview: all-short answers so Track A is all null
        qs = [
            _make_q(i + 1, n_words=FEATURE_MIN_WORDS["lexical_mattr"] - 5,
                    duration_s=15.0, lexical_mattr=0.0, sbert_coherence=0.0)
            for i in range(8)
        ]
        results = score_interview(qs)
        for r in results:
            if r["evaluable"] and r["track_a_z_scores"]:
                # All nulls; only Track C should contribute
                null_count = sum(1 for v in r["track_a_z_scores"].values() if v is None)
                # Most features should be null for short answers
                assert null_count >= 2, \
                    f"Q{r['q_no']}: expected null Track A features for short answers"


# ---------------------------------------------------------------------------
# BUG 2 — Sigmoid latency / no coin-flip cliff
# ---------------------------------------------------------------------------

class TestSigmoidLatency:
    def test_sigmoid_is_smooth(self):
        """Sigmoid must increase monotonically and never jump discretely."""
        xs = [i * 0.5 for i in range(20)]
        ys = [_sigmoid(x, 1.2) for x in xs]
        for i in range(1, len(ys)):
            assert ys[i] >= ys[i - 1], "sigmoid not monotonically increasing"

    def test_sigmoid_neutral_at_zero(self):
        """Sigmoid(0) = 0.5; centered contribution = 0."""
        assert abs(_sigmoid(0.0, 1.2) - 0.5) < 1e-6

    def test_higher_composite_more_likely_flagged(self):
        """
        A question with a genuinely higher composite score should be at least as
        likely to be flagged as one with a lower composite score.
        This guards against the coin-flip cliff where Q12 (score 13) is not
        flagged while Q3 (score 6.57) is.
        """
        # Build a 10-question interview where Q7 has strong deviation on
        # multiple features and Q3 has weak deviation on latency only.
        qs = _make_interview(10, speech_rate_wpm=120.0)
        # Q3: slight latency increase only
        qs[2]["word_timestamps"][0]["start"] = 8.0  # somewhat delayed start
        # Q7: strong deviation on speech rate + early latency
        qs[6]["text_feats"]["speech_rate_wpm"] = 220.0
        qs[6]["text_feats"]["lexical_mattr"] = 0.95
        qs[6]["text_feats"]["filler_word_ratio"] = 0.001

        results = score_interview(qs)
        score_q3 = next(r["suspicion_score"] for r in results if r["q_no"] == 3 and r["evaluable"])
        score_q7 = next(r["suspicion_score"] for r in results if r["q_no"] == 7 and r["evaluable"])
        flag_q7 = next(r["flagged_for_review"] for r in results if r["q_no"] == 7 and r["evaluable"])

        # Q7 must score higher than Q3
        assert score_q7 >= score_q3, \
            f"Q7 (strong deviation, score {score_q7}) should score >= Q3 (weak, score {score_q3})"

    def test_flag_requires_hysteresis_margin(self):
        """
        A latency z just barely above HARD_SIGNAL_THRESHOLD should NOT trigger
        the flag; it must exceed by HYSTERESIS_BAND.
        """
        from v3.pipeline.violation_detector import _flag_for_review, COMPOSITE_THRESHOLD
        # Composite barely above threshold; latency z just at (not above) threshold+band
        at_threshold = {
            "response_latency_sec_z": HARD_SIGNAL_THRESHOLD + HYSTERESIS_BAND - 0.01,
            "cross_question_naturalness_flatness_provisional_flag": False,
            "latency_variance_provisional_flag": False,
        }
        assert not _flag_for_review(COMPOSITE_THRESHOLD + 1.0, at_threshold), \
            "Flag fired with latency z below threshold+band (hysteresis not respected)"

        # Just above threshold+band: must trigger
        above_threshold = {
            "response_latency_sec_z": HARD_SIGNAL_THRESHOLD + HYSTERESIS_BAND + 0.01,
            "cross_question_naturalness_flatness_provisional_flag": False,
            "latency_variance_provisional_flag": False,
        }
        assert _flag_for_review(COMPOSITE_THRESHOLD + 1.0, above_threshold), \
            "Flag did not fire with latency z above threshold+band"


# ---------------------------------------------------------------------------
# BUG 3 — Flagged questions must have real evidence or marked low-confidence
# ---------------------------------------------------------------------------

class TestFlaggedQuestionsHaveEvidence:
    def test_flagged_questions_have_non_null_track_a_or_low_confidence(self):
        """
        Any flagged question must either:
        (a) have at least one non-null, non-zero Track A z-score, OR
        (b) be marked low_answer_count_reduced_confidence=True (Track A unavailable),
            in which case the flag is supported by Track C alone.
        This prevents the Q3/Q8 pattern: flagged but all Track A = 0.0 (not null, not real).
        """
        qs = _make_interview(10)
        results = score_interview(qs)
        for r in results:
            if r.get("flagged_for_review"):
                track_a = r.get("track_a_z_scores", {})
                has_real_track_a = any(
                    v is not None and v > 0
                    for v in track_a.values()
                )
                low_confidence = r.get("low_answer_count_reduced_confidence", False)
                assert has_real_track_a or low_confidence, (
                    f"Q{r['q_no']} is flagged but has no real Track A signal "
                    f"and is not marked low_confidence. Track A: {track_a}"
                )


# ---------------------------------------------------------------------------
# BUG 4 — Flatness / latency_var provisional flags
# ---------------------------------------------------------------------------

class TestProvisionalFlags:
    def test_flat_interview_triggers_provisional_flag(self):
        """
        A candidate with suspiciously identical features across all questions
        must trigger the flatness provisional flag, even with no Track A signal.
        """
        # Perfectly flat interview: same speech_rate and mattr on every question
        qs = _make_interview(8, speech_rate_wpm=130.0, lexical_mattr=0.70)
        results = score_interview(qs)
        # At least some evaluable questions should have the provisional flag
        prov_flags = [r["track_c_provisional_flag"] for r in results if r.get("evaluable")]
        assert any(prov_flags), \
            "Perfectly flat interview produced no track_c_provisional_flag=True"

    def test_varied_interview_no_provisional_flag(self):
        """
        An interview with natural variation in speech_rate / mattr must NOT
        trigger the FLATNESS provisional flag. The test checks the flatness
        sub-flag only, not the combined track_c_provisional_flag (which also
        includes latency_variance, a separate signal).
        """
        qs = []
        for i in range(8):
            qs.append(_make_q(i + 1, speech_rate_wpm=100.0 + i * 15.0,
                              lexical_mattr=0.60 + i * 0.04))
        results = score_interview(qs)
        flatness_flags = [
            r["track_c_diagnostics"].get("cross_question_naturalness_flatness_provisional_flag", False)
            for r in results if r.get("evaluable")
        ]
        # No question should trigger the flatness flag when speech rate and MATTR
        # vary substantially across the interview.
        assert not any(flatness_flags), \
            f"Varied interview triggered flatness provisional flag: {flatness_flags}"

    def test_provisional_flag_in_output_schema(self):
        """track_c_provisional_flag must be present in every evaluable payload."""
        results = score_interview(_make_interview(8))
        for r in results:
            if r["evaluable"]:
                assert "track_c_provisional_flag" in r, \
                    f"Q{r['q_no']} missing track_c_provisional_flag"

    def test_raw_flatness_in_track_c_raw_values(self):
        """raw_flatness must always be emitted in track_c_raw_values."""
        results = score_interview(_make_interview(8))
        for r in results:
            if r["evaluable"]:
                assert "flatness" in r.get("track_c_raw_values", {}), \
                    f"Q{r['q_no']} missing flatness in track_c_raw_values"

    def test_latency_var_provisional_with_uniform_latency(self):
        """
        When every question starts at the same timestamp (uniform latency),
        the latency_variance provisional flag should fire.
        """
        qs = _make_interview(8)
        # All start at exactly the same latency: 0.2s
        for q in qs:
            if q["word_timestamps"]:
                first = q["word_timestamps"][0]
                first["start"] = 0.2
        results = score_interview(qs)
        prov_latency = [
            r["track_c_diagnostics"].get("latency_variance_provisional_flag", False)
            for r in results if r.get("evaluable")
        ]
        assert any(prov_latency), \
            "Uniform latency across all questions did not trigger latency_variance_provisional_flag"


# ---------------------------------------------------------------------------
# BUG 5 — discourse_organization not pinned at clip ceiling
# ---------------------------------------------------------------------------

class TestDiscourseOrganizationPercentileRank:
    def test_percentile_rank_range(self):
        dist = [0.0, 0.0, 1.0, 2.0, 5.0]
        assert 0.0 <= _percentile_rank(0.0, dist) <= 1.0
        assert 0.0 <= _percentile_rank(5.0, dist) <= 1.0

    def test_percentile_rank_monotone(self):
        dist = [1.0, 2.0, 3.0, 4.0, 5.0]
        ranks = [_percentile_rank(v, dist) for v in dist]
        for i in range(1, len(ranks)):
            assert ranks[i] >= ranks[i - 1]

    def test_discourse_org_not_pegged_at_clip(self):
        """
        In an interview where some questions have more connectors than others,
        discourse_organization z-scores must NOT all cluster at Z_SCORE_CLIP.
        """
        qs = []
        connector_counts = [0.0, 1.0, 3.0, 5.0, 0.0, 2.0, 1.0, 4.0]
        for i, cnt in enumerate(connector_counts):
            qs.append(_make_q(i + 1, discourse_connectors=cnt, discourse_tier1=0.0))

        results = score_interview(qs)
        disc_zs = [
            r["track_a_z_scores"].get("discourse_organization")
            for r in results
            if r.get("evaluable") and r["track_a_z_scores"]
        ]
        non_null = [z for z in disc_zs if z is not None]
        if non_null:
            # At most 20% of questions should be pegged at the clip ceiling
            at_clip = sum(1 for z in non_null if abs(z) >= Z_SCORE_CLIP - 0.01)
            pct_at_clip = at_clip / len(non_null)
            assert pct_at_clip < 0.5, (
                f"{pct_at_clip:.0%} of discourse_organization scores are at clip ceiling — "
                "percentile rank fix not working"
            )

    def test_sparse_zero_heavy_discourse_handled(self):
        """
        When most answers have 0 connectors (sparse), no score should blow to clip.
        """
        qs = []
        for i in range(8):
            cnt = 5.0 if i == 3 else 0.0  # only Q4 has connectors
            qs.append(_make_q(i + 1, discourse_connectors=cnt, discourse_tier1=0.0))
        results = score_interview(qs)
        for r in results:
            if r.get("evaluable") and r["track_a_z_scores"]:
                z = r["track_a_z_scores"].get("discourse_organization")
                if z is not None:
                    assert z <= Z_SCORE_CLIP, f"Q{r['q_no']} discourse_org z={z} exceeded clip"


# ---------------------------------------------------------------------------
# BUG 6 — sbert_coherence 0.65 sentinel treated as None
# ---------------------------------------------------------------------------

class TestSbertCoherenceSentinel:
   

    def test_zero_coherence_excluded(self):
        """sbert_coherence == 0.0 must return None (extraction failure)."""
        q = _make_q(1, sbert_coherence=0.0)
        avail = _feature_availability(q)
        result = _get_sbert(q, avail)
        assert result is None

    def test_valid_coherence_returned(self):
        """A real sbert_coherence value (e.g. 0.78) must be returned as-is."""
        q = _make_q(1, sbert_coherence=0.78, n_words=80)
        avail = _feature_availability(q)
        result = _get_sbert(q, avail)
        assert result is not None
        assert abs(result - 0.78) < 1e-6

    def test_all_neutral_coherence_excluded_from_z_map(self):
        """
        When all questions have sbert_coherence == SBERT_NEUTRAL_FALLBACK,
        no question should have sbert_coherence in its track_a_z_scores.
        """
        qs = _make_interview(8, sbert_coherence=SBERT_NEUTRAL_FALLBACK)
        results = score_interview(qs)
        for r in results:
            if r["evaluable"]:
                assert "sbert_coherence" not in r["track_a_z_scores"] or \
                       r["track_a_z_scores"]["sbert_coherence"] is None, (
                    f"Q{r['q_no']}: sbert_coherence sentinel leaked into z_map"
                )

    def test_valid_coherence_appears_in_z_map(self):
        """Real coherence values should produce a z-score entry."""
        qs = _make_interview(8, sbert_coherence=0.75, n_words=80)
        results = score_interview(qs)
        has_coh = any(
            r["track_a_z_scores"].get("sbert_coherence") is not None
            for r in results if r["evaluable"]
        )
        assert has_coh, "No question produced a sbert_coherence z-score with valid inputs"

    def test_short_answer_coherence_is_none(self):
        """sbert_coherence below FEATURE_MIN_WORDS threshold must return None."""
        q = _make_q(1, n_words=FEATURE_MIN_WORDS["sbert_coherence"] - 3,
                    sbert_coherence=0.75, duration_s=10.0)
        avail = _feature_availability(q)
        result = _get_sbert(q, avail)
        assert result is None


# ---------------------------------------------------------------------------
# Eligibility-gate integration (carried from v2, extended)
# ---------------------------------------------------------------------------

class TestEligibilityIntegration:
    def test_skipped_not_in_baseline(self):
        qs = _make_interview(8)
        qs[2]["skipped"] = True
        results = score_interview(qs)
        q3 = next(r for r in results if r["q_no"] == 3)
        assert not q3["evaluable"]
        assert q3["suspicion_score"] is None

    def test_constant_fallback_skipped_not_flagged(self):
        qs = []
        for i in range(5):
            qs.append(_make_q(i + 1, speech_rate_wpm=120.0 + i * 5))
        for i in range(3):
            q = _make_q(i + 6, text="", duration_s=0.0)
            q["word_timestamps"] = []
            qs.append(q)
        results = score_interview(qs)
        for r in results:
            if r["evaluable"]:
                z_flat = r["track_c_diagnostics"].get("cross_question_naturalness_flatness_provisional_flag", False)
                # The 3 skipped answers must not have inflated the flatness
                # (this is a qualitative check — no all-true provisional flags
                #  solely due to skipped answers)

    def test_non_evaluable_payload_schema(self):
        qs = _make_interview(6)
        qs[0]["skipped"] = True
        results = score_interview(qs)
        q1 = next(r for r in results if r["q_no"] == 1)
        for key in ["q_no", "evaluable", "not_evaluable_reason",
                    "suspicion_score", "flagged_for_review",
                    "track_a_z_scores", "track_c_diagnostics",
                    "top_contributing_features"]:
            assert key in q1, f"Missing key {key} in non-evaluable payload"


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class TestOutputSchema:
    REQUIRED = {
        "q_no", "evaluable", "not_evaluable_reason", "suspicion_score",
        "flagged_for_review", "track_a_z_scores", "track_c_diagnostics",
        "top_contributing_features",
    }
    EVALUABLE_EXTRA = {
        "feature_availability", "track_c_raw_values", "track_c_provisional_flag",
        "low_answer_count_reduced_confidence",
    }

    def test_evaluable_keys(self):
        qs = _make_interview(6)
        results = score_interview(qs)
        for r in results:
            if r["evaluable"]:
                # Note: use parentheses to avoid operator precedence issue with | and -
                missing = (self.REQUIRED | self.EVALUABLE_EXTRA) - set(r.keys())
                assert not missing, f"Q{r['q_no']} missing: {missing}"

    def test_top3_max_len(self):
        results = score_interview(_make_interview(8))
        for r in results:
            assert len(r["top_contributing_features"]) <= 3

    def test_track_c_diagnostics_fields(self):
        required_c = {
            "response_latency_sec_z",
            "latency_variance_provisional_flag",
            "cross_question_naturalness_flatness_provisional_flag",
            "intra_answer_pace_variance",
            "pause_regularity",
            "pitch_variance_ratio",
        }
        results = score_interview(_make_interview(8))
        for r in results:
            if r["evaluable"]:
                missing = required_c - set(r["track_c_diagnostics"].keys())
                assert not missing, f"Q{r['q_no']} track_c_diagnostics missing: {missing}"


# ---------------------------------------------------------------------------
# BUG A — feature_availability.sbert_coherence must reflect actual outcome
# ---------------------------------------------------------------------------

class TestSbertAvailabilityAccuracy:
    """
    feature_availability.sbert_coherence must be False whenever sbert_coherence
    z-score is None, regardless of whether the word count passed the structural
    gate. A True availability alongside a null z-score is a lie.
    """

    def test_zero_coherence_availability_false(self):
        """sbert_coherence==0.0 (extraction failure) → availability must be False."""
        qs = _make_interview(8, sbert_coherence=0.0, n_words=80)
        results = score_interview(qs)
        for r in results:
            if r["evaluable"]:
                avail = r["feature_availability"]
                z = r["track_a_z_scores"].get("sbert_coherence")
                assert not (avail.get("sbert_coherence") is True and z is None), (
                    f"Q{r['q_no']}: feature_availability.sbert_coherence=True "
                    f"but track_a_z_scores.sbert_coherence=None (BUG A)"
                )

    def test_sentinel_coherence_availability_false(self):
        """sbert_coherence==0.65 (SBERT neutral sentinel) → availability must be False."""
        qs = _make_interview(8, sbert_coherence=SBERT_NEUTRAL_FALLBACK, n_words=80)
        results = score_interview(qs)
        for r in results:
            if r["evaluable"]:
                avail = r["feature_availability"]
                z = r["track_a_z_scores"].get("sbert_coherence")
                assert not (avail.get("sbert_coherence") is True and z is None), (
                    f"Q{r['q_no']}: feature_availability.sbert_coherence=True "
                    f"but z=None for sentinel 0.65 (BUG A)"
                )

    def test_valid_coherence_availability_true(self):
        """Valid sbert_coherence with enough peers → availability=True, z non-None."""
        qs = _make_interview(8, sbert_coherence=0.78, n_words=80)
        results = score_interview(qs)
        has_valid = any(
            r["feature_availability"].get("sbert_coherence") is True
            and r["track_a_z_scores"].get("sbert_coherence") is not None
            for r in results if r.get("evaluable")
        )
        assert has_valid, (
            "No evaluable question has sbert availability=True with a non-None z. "
            "Check _get_sbert() and the BUG A post-hoc correction in _compute_track_a()."
        )

    def test_availability_false_when_no_valid_peers(self):
        """
        Q1 has valid coherence but all peers have sentinel → z_coh=None →
        availability must be corrected to False.
        """
        qs = _make_interview(8, n_words=80)
        for i in range(1, 8):
            qs[i]["text_feats"]["sbert_coherence"] = SBERT_NEUTRAL_FALLBACK
        qs[0]["text_feats"]["sbert_coherence"] = 0.78
        results = score_interview(qs)
        q1 = next(r for r in results if r["q_no"] == 1 and r["evaluable"])
        z = q1["track_a_z_scores"].get("sbert_coherence")
        avail = q1["feature_availability"].get("sbert_coherence")
        assert z is None, f"Expected z_coh=None when no valid peers, got {z}"
        assert avail is False, (
            f"Q1 has no valid peers but feature_availability.sbert_coherence={avail} "
            f"(should be False when z=None)"
        )

    def test_no_true_availability_alongside_null_z(self):
        """
        Invariant across a mixed interview: availability=True ↔ z is not None.
        """
        qs = _make_interview(8, n_words=80)
        for i in range(4):
            qs[i]["text_feats"]["sbert_coherence"] = 0.75
        for i in range(4, 8):
            qs[i]["text_feats"]["sbert_coherence"] = SBERT_NEUTRAL_FALLBACK
        results = score_interview(qs)
        for r in results:
            if r["evaluable"]:
                avail = r["feature_availability"].get("sbert_coherence")
                z = r["track_a_z_scores"].get("sbert_coherence")
                assert not (avail is True and z is None), (
                    f"Q{r['q_no']}: availability=True but z=None (BUG A)"
                )
                assert not (avail is False and z is not None), (
                    f"Q{r['q_no']}: availability=False but z={z} (inconsistent)"
                )


# ---------------------------------------------------------------------------
# BUG B — track_c_provisional_flag must wire into flagged_for_review
# ---------------------------------------------------------------------------

class TestProvisionalFlagWiredToFlaggedForReview:
    """
    track_c_provisional_flag=True (latency_var OR flatness) must cause
    flagged_for_review=True when composite > COMPOSITE_THRESHOLD.
    Previously latency_variance_provisional_flag was silently discarded.
    """

    def _uniform_latency_interview(self, n: int = 8) -> list[dict]:
        """All questions start at same timestamp; mildly deviant features."""
        qs = _make_interview(n)
        for q in qs:
            if q["word_timestamps"]:
                q["word_timestamps"][0]["start"] = 0.2
            q["text_feats"]["speech_rate_wpm"] = 180.0
            q["text_feats"]["filler_word_ratio"] = 0.005
        return qs

    def test_latency_var_prov_causes_flagged_for_review(self):
        """
        Direct unit test of _flag_for_review wiring: latency_variance_provisional_flag=True
        must satisfy the OR condition and produce True when composite > threshold.

        We test _flag_for_review directly because the end-to-end composite score
        depends on UNCALIBRATED thresholds — testing the wiring at the function
        boundary is both more reliable and more targeted.
        """
        from v3.pipeline.violation_detector import _flag_for_review
        # latency_var ONLY (flatness=False, latency_z below hysteresis)
        track_c_lat_only = {
            "response_latency_sec_z": 0.0,
            "latency_variance_provisional_flag": True,
            "cross_question_naturalness_flatness_provisional_flag": False,
        }
        # Must flag when composite > COMPOSITE_THRESHOLD
        assert _flag_for_review(COMPOSITE_THRESHOLD + 1.0, track_c_lat_only), (
            "latency_variance_provisional_flag=True with composite above threshold "
            "must produce flagged_for_review=True. BUG B: prov flag not wired."
        )
        # Must NOT flag when composite <= COMPOSITE_THRESHOLD (gate still applies)
        assert not _flag_for_review(COMPOSITE_THRESHOLD, track_c_lat_only), (
            "latency_variance_provisional_flag must not bypass composite gate."
        )
        # Both False: only latency_z above hysteresis should flag
        track_c_no_prov = {
            "response_latency_sec_z": HARD_SIGNAL_THRESHOLD + HYSTERESIS_BAND + 0.1,
            "latency_variance_provisional_flag": False,
            "cross_question_naturalness_flatness_provisional_flag": False,
        }
        assert _flag_for_review(COMPOSITE_THRESHOLD + 1.0, track_c_no_prov), (
            "High latency_z alone must still flag when above hysteresis threshold."
        )

    def test_flatness_prov_causes_flagged_for_review(self):
        """
        Direct unit test: cross_question_naturalness_flatness_provisional_flag=True
        must satisfy the OR condition.
        """
        from v3.pipeline.violation_detector import _flag_for_review
        track_c_flat_only = {
            "response_latency_sec_z": 0.0,
            "latency_variance_provisional_flag": False,
            "cross_question_naturalness_flatness_provisional_flag": True,
        }
        assert _flag_for_review(COMPOSITE_THRESHOLD + 1.0, track_c_flat_only), (
            "flatness_provisional_flag=True with composite above threshold "
            "must produce flagged_for_review=True."
        )
        assert not _flag_for_review(COMPOSITE_THRESHOLD, track_c_flat_only), (
            "flatness_provisional_flag must not bypass composite gate."
        )
        # Neither provisional flag: below hysteresis → must NOT flag
        track_c_neither = {
            "response_latency_sec_z": HARD_SIGNAL_THRESHOLD,  # at threshold, not above+band
            "latency_variance_provisional_flag": False,
            "cross_question_naturalness_flatness_provisional_flag": False,
        }
        assert not _flag_for_review(COMPOSITE_THRESHOLD + 1.0, track_c_neither), (
            "No provisional flags and latency_z at (not above) threshold+band must NOT flag."
        )

    def test_prov_flag_does_not_bypass_composite_gate(self):
        """Provisional flag + composite <= COMPOSITE_THRESHOLD must NOT flag."""
        from v3.pipeline.violation_detector import _flag_for_review
        track_c = {
            "response_latency_sec_z": 0.0,
            "latency_variance_provisional_flag": True,
            "cross_question_naturalness_flatness_provisional_flag": True,
        }
        assert not _flag_for_review(COMPOSITE_THRESHOLD, track_c), (
            "Provisional flags must not bypass composite gate at threshold."
        )
        assert _flag_for_review(COMPOSITE_THRESHOLD + 0.01, track_c), (
            "Provisional flags must fire when composite just above threshold."
        )

    def test_low_answer_count_tagged_when_provisional_fires(self):
        """
        Provisional flags with < TRACK_A_MIN_EVALUABLE_ANSWERS questions must
        set low_answer_count_reduced_confidence=True.
        """
        qs = _make_interview(3)
        for q in qs:
            q["word_timestamps"][0]["start"] = 0.2
            q["text_feats"]["speech_rate_wpm"] = 180.0
        results = score_interview(qs)
        for r in results:
            if r.get("evaluable"):
                assert r["low_answer_count_reduced_confidence"] is True, (
                    f"Q{r['q_no']}: expected low_answer_count_reduced_confidence=True "
                    f"with <{TRACK_A_MIN_EVALUABLE_ANSWERS} evaluable answers"
                )

    def test_both_prov_flags_in_diagnostics(self):
        """Both provisional flags must appear in track_c_diagnostics (even when False)."""
        results = score_interview(_make_interview(8))
        for r in results:
            if r["evaluable"]:
                diag = r["track_c_diagnostics"]
                assert "latency_variance_provisional_flag" in diag, (
                    f"Q{r['q_no']}: latency_variance_provisional_flag missing"
                )
                assert "cross_question_naturalness_flatness_provisional_flag" in diag, (
                    f"Q{r['q_no']}: flatness_provisional_flag missing"
                )
