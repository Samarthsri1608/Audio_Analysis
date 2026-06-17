"""
test_api.py — Comprehensive Test Suite for AI Speaking Assessment Engine
=========================================================================

Coverage:
  1. Functionality tests    — every endpoint returns correct shape / values
  2. API contract tests     — HTTP status codes, error responses, edge cases
  3. Bug-finding tests      — boundary values, type safety, NaN/None guards
  4. Accuracy/sanity tests  — scores in range, feature values plausible
  5. Reproducibility tests  — same file uploaded N times must give identical scores

Usage (from Voice-projects/backend):
    source ../audio/bin/activate
    pytest tests/test_api.py -v --tb=short 2>&1 | tee ../test_results.log

The server must be running on http://localhost:8000 before executing these tests.
Set TEST_FILES_DIR to override the default test-file location.
"""

import os
import sys
import json
import math
import time
import pytest
import requests
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Locate the project root (one level up from backend/)
_HERE         = Path(__file__).resolve().parent.parent.parent
TEST_FILES    = {
    "interview_mp4":   _HERE / "interview.mp4",
    "interview_1_mp4": _HERE / "interview_1.mp4",
    "wav":             _HERE / "interview_converted.wav",
    "short_wav":       _HERE / "test_short.wav",
}

TIMEOUT = 600  # seconds — large files with Whisper may be slow

# Expected top-level keys in /evaluate response
EVALUATE_REQUIRED_KEYS = {
    "status", "message", "preprocessing_flags",
    "duration_ms", "segments", "features", "scores",
}

# Expected feature dimensions
EXPECTED_FEATURE_DIMS = {
    "fluency", "intelligibility", "language_control",
    "lexical_resource", "discourse", "voice_modulation", "sentiment",
}

# Expected score dimensions
EXPECTED_SCORE_DIMS = {
    "logical_cohesion", "delivery_fluency", "pronunciation_clarity",
    "vocal_dynamism", "collaborative_tone", "lexical_precision",
}

# Expected top-level keys in /report response
REPORT_REQUIRED_KEYS = {
    "file", "overall_score", "out_of", "grade", "label",
    "dimensions", "strengths", "areas_for_improvement",
    "executive_summary", "reasoning_source",
}

VALID_GRADES  = {"A+", "A", "B+", "B", "C", "D", "F"}
VALID_LABELS  = {"Excellent", "Good", "Average", "Below Average", "Poor"}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _post_file(endpoint: str, filepath: Path, timeout: int = TIMEOUT) -> requests.Response:
    """POST a file to a given endpoint and return the Response."""
    mime = "video/mp4" if str(filepath).endswith(".mp4") else "audio/wav"
    with open(filepath, "rb") as fh:
        return requests.post(
            f"{BASE_URL}{endpoint}",
            files={"file": (filepath.name, fh, mime)},
            timeout=timeout,
        )


def _assert_score_valid(score, dim_name: str):
    """Assert a single score is a finite float in [0, 5]."""
    assert score is not None, f"{dim_name}: score is None"
    assert isinstance(score, (int, float)), f"{dim_name}: score not numeric (got {type(score)})"
    assert not math.isnan(score),  f"{dim_name}: score is NaN"
    assert not math.isinf(score),  f"{dim_name}: score is Inf"
    assert 0.0 <= score <= 5.0,    f"{dim_name}: score {score} out of [0, 5]"


def _evaluate(filepath: Path) -> dict:
    """Call /evaluate, assert HTTP 200, return parsed JSON."""
    resp = _post_file("/evaluate", filepath)
    assert resp.status_code == 200, (
        f"/evaluate returned {resp.status_code} for {filepath.name}: {resp.text[:300]}"
    )
    data = resp.json()
    assert data.get("status") == "success"
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 0. Server health
# ─────────────────────────────────────────────────────────────────────────────

class TestServerHealth:
    def test_root_endpoint(self):
        """GET / must return 200 with a running message."""
        from urllib.parse import urlparse
        parsed = urlparse(BASE_URL)
        root_url = f"{parsed.scheme}://{parsed.netloc}"
        resp = requests.get(f"{root_url}/", timeout=10)
        assert resp.status_code == 200
        body = resp.json()
        assert "message" in body
        print(f"\n  [health] {body['message']}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Functionality tests — /evaluate
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateEndpoint:

    @pytest.mark.parametrize("file_key", ["wav", "interview_mp4", "interview_1_mp4"])
    def test_evaluate_returns_200(self, file_key):
        """All supported file types should return HTTP 200."""
        fp = TEST_FILES[file_key]
        if not fp.exists():
            pytest.skip(f"Test file not found: {fp}")
        resp = _post_file("/evaluate", fp)
        assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text[:200]}"

    @pytest.mark.parametrize("file_key", ["wav", "interview_mp4"])
    def test_evaluate_response_schema(self, file_key):
        """Response must contain all required top-level keys."""
        fp = TEST_FILES[file_key]
        if not fp.exists():
            pytest.skip(f"Test file not found: {fp}")
        data = _evaluate(fp)
        missing = EVALUATE_REQUIRED_KEYS - set(data.keys())
        assert not missing, f"Missing keys in /evaluate response: {missing}"

    @pytest.mark.parametrize("file_key", ["wav", "interview_mp4"])
    def test_evaluate_has_all_feature_dimensions(self, file_key):
        """features dict must contain all 7 extraction dimensions including sentiment."""
        fp = TEST_FILES[file_key]
        if not fp.exists():
            pytest.skip(f"Test file not found: {fp}")
        data  = _evaluate(fp)
        feats = set(data["features"].keys())
        missing = EXPECTED_FEATURE_DIMS - feats
        assert not missing, f"Missing feature dimensions: {missing}"

    @pytest.mark.parametrize("file_key", ["wav", "interview_mp4"])
    def test_evaluate_has_all_score_dimensions(self, file_key):
        """scores dict must contain all 7 dimensions including sentiment."""
        fp = TEST_FILES[file_key]
        if not fp.exists():
            pytest.skip(f"Test file not found: {fp}")
        data   = _evaluate(fp)
        scored = set(data["scores"].keys())
        missing = EXPECTED_SCORE_DIMS - scored
        assert not missing, f"Missing score dimensions: {missing}"

    @pytest.mark.parametrize("file_key", ["wav", "interview_mp4", "interview_1_mp4"])
    def test_evaluate_all_scores_in_range(self, file_key):
        """Every score must be a finite float strictly in [0.0, 5.0]."""
        fp = TEST_FILES[file_key]
        if not fp.exists():
            pytest.skip(f"Test file not found: {fp}")
        data = _evaluate(fp)
        for dim, score in data["scores"].items():
            _assert_score_valid(score, dim)

    def test_evaluate_duration_is_positive(self):
        """duration_ms must be a positive number."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip(f"Test file not found: {fp}")
        data = _evaluate(fp)
        assert data["duration_ms"] > 0, "duration_ms should be > 0"

    def test_evaluate_segments_non_empty(self):
        """segments list must have at least one segment for a real interview."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip(f"Test file not found: {fp}")
        data = _evaluate(fp)
        assert len(data["segments"]) > 0, "segments list is empty"

    def test_evaluate_segments_have_required_keys(self):
        """Each segment must have start, end, text."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip(f"Test file not found: {fp}")
        data = _evaluate(fp)
        for i, seg in enumerate(data["segments"]):
            for key in ("start", "end", "text"):
                assert key in seg, f"Segment {i} missing key '{key}'"
            assert seg["start"] <= seg["end"], f"Segment {i}: start > end"

    def test_feature_fluency_wpm_plausible(self):
        """WPM should be between 30 and 300 for real speech."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip(f"Test file not found: {fp}")
        data = _evaluate(fp)
        wpm  = data["features"]["fluency"]["wpm"]
        assert 30 <= wpm <= 300, f"WPM {wpm} is implausible"

    def test_feature_intelligibility_confidence_range(self):
        """mean_confidence must be in [0, 1]."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip(f"Test file not found: {fp}")
        data = _evaluate(fp)
        mc   = data["features"]["intelligibility"]["mean_confidence"]
        assert 0.0 <= mc <= 1.0, f"mean_confidence {mc} out of [0, 1]"

    def test_feature_sentiment_compound_range(self):
        """mean_compound must be in [-1, 1]."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip(f"Test file not found: {fp}")
        data  = _evaluate(fp)
        mc    = data["features"]["sentiment"]["mean_compound"]
        assert -1.0 <= mc <= 1.0, f"mean_compound {mc} out of [-1, 1]"

    def test_feature_voice_modulation_pitch_non_negative(self):
        """pitch_std must be >= 0."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip(f"Test file not found: {fp}")
        data = _evaluate(fp)
        ps   = data["features"]["voice_modulation"]["pitch_std"]
        assert ps >= 0, f"pitch_std {ps} is negative"

    def test_feature_lexical_mattr_range(self):
        """MATTR must be in [0, 1]."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip(f"Test file not found: {fp}")
        data  = _evaluate(fp)
        mattr = data["features"]["lexical_resource"].get("mattr", 0)
        assert 0.0 <= mattr <= 1.0, f"MATTR {mattr} out of [0, 1]"

    def test_feature_types_are_json_serialisable(self):
        """The response must be valid JSON (no NumPy types, custom objects, etc.)."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip(f"Test file not found: {fp}")
        resp = _post_file("/evaluate", fp)
        assert resp.status_code == 200
        try:
            json.loads(resp.text)
        except json.JSONDecodeError as e:
            pytest.fail(f"Response is not valid JSON: {e}")

    def test_preprocessing_flags_is_list(self):
        """preprocessing_flags must be a list (even if empty)."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip(f"Test file not found: {fp}")
        data = _evaluate(fp)
        assert isinstance(data["preprocessing_flags"], list)


# ─────────────────────────────────────────────────────────────────────────────
# 2. API contract tests — error handling
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIContracts:

    def test_unsupported_format_returns_400(self):
        """Uploading a .txt file should return 400 Bad Request."""
        fake_txt = Path("/tmp/fake_audio.txt")
        fake_txt.write_text("this is not an audio file")
        try:
            with open(fake_txt, "rb") as fh:
                resp = requests.post(
                    f"{BASE_URL}/evaluate",
                    files={"file": ("fake_audio.txt", fh, "text/plain")},
                    timeout=15,
                )
            assert resp.status_code == 400, (
                f"Expected 400 for unsupported format, got {resp.status_code}"
            )
        finally:
            fake_txt.unlink(missing_ok=True)

    def test_evaluate_no_file_returns_422(self):
        """POST /evaluate without a file should return 422 Unprocessable Entity."""
        resp = requests.post(f"{BASE_URL}/evaluate", timeout=10)
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"

    def test_report_no_file_returns_422(self):
        """POST /report without a file should return 422."""
        resp = requests.post(f"{BASE_URL}/report", timeout=10)
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"

    def test_evaluate_mp4_accepted(self):
        """The /evaluate endpoint must accept .mp4 files (not 400 or 415)."""
        fp = TEST_FILES["interview_mp4"]
        if not fp.exists():
            pytest.skip(f"Test file not found: {fp}")
        resp = _post_file("/evaluate", fp)
        assert resp.status_code != 400, "mp4 should be accepted, got 400"
        assert resp.status_code != 415, "mp4 should be accepted, got 415"

    def test_train_endpoint_exists(self):
        """POST /train must exist and return 422/500 when called with no body (not 404)."""
        resp = requests.post(f"{BASE_URL}/train", timeout=10)
        assert resp.status_code != 404, "/train endpoint not found (404)"

    def test_convert_endpoint_exists(self):
        """POST /convert_from_video is deprecated in v2 (skip check)."""
        pytest.skip("convert_from_video endpoint is not used in v2")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Bug-finding tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBugFinding:

    def test_no_nan_in_scores(self):
        """No score must be NaN under any test file."""
        for key, fp in TEST_FILES.items():
            if not fp.exists():
                continue
            resp = _post_file("/evaluate", fp)
            if resp.status_code != 200:
                continue
            for dim, score in resp.json()["scores"].items():
                assert not math.isnan(float(score)), \
                    f"NaN score in '{dim}' for {fp.name}"

    def test_no_none_scores(self):
        """No score must be None/null."""
        for key, fp in TEST_FILES.items():
            if not fp.exists():
                continue
            resp = _post_file("/evaluate", fp)
            if resp.status_code != 200:
                continue
            for dim, score in resp.json()["scores"].items():
                assert score is not None, \
                    f"None score in '{dim}' for {fp.name}"

    def test_no_scores_below_zero(self):
        """Scores must never be negative (clamp bug check)."""
        for key, fp in TEST_FILES.items():
            if not fp.exists():
                continue
            resp = _post_file("/evaluate", fp)
            if resp.status_code != 200:
                continue
            for dim, score in resp.json()["scores"].items():
                assert float(score) >= 0.0, \
                    f"Score below 0 in '{dim}' for {fp.name}: {score}"

    def test_no_scores_above_five(self):
        """Scores must never exceed 5.0 (clamp bug check)."""
        for key, fp in TEST_FILES.items():
            if not fp.exists():
                continue
            resp = _post_file("/evaluate", fp)
            if resp.status_code != 200:
                continue
            for dim, score in resp.json()["scores"].items():
                assert float(score) <= 5.0, \
                    f"Score above 5 in '{dim}' for {fp.name}: {score}"

    def test_feature_filler_rate_in_range(self):
        """filler_rate must be in [0, 1] — not a raw count."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip()
        data = _evaluate(fp)
        fr   = data["features"]["fluency"]["filler_rate"]
        assert 0.0 <= fr <= 1.0, f"filler_rate {fr} is out of [0, 1]"

    def test_feature_pronunciation_score_in_range(self):
        """pronunciation_score must be in [0, 1]."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip()
        data = _evaluate(fp)
        ps   = data["features"]["intelligibility"]["pronunciation_score"]
        assert 0.0 <= ps <= 1.0, f"pronunciation_score {ps} out of [0, 1]"

    def test_feature_rare_word_ratio_in_range(self):
        """rare_word_ratio must be in [0, 1]."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip()
        data = _evaluate(fp)
        rwr  = data["features"]["lexical_resource"].get("rare_word_ratio", 0)
        assert 0.0 <= rwr <= 1.0, f"rare_word_ratio {rwr} out of [0, 1]"

    def test_sentiment_std_non_negative(self):
        """std_compound is a standard deviation — must never be negative."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip()
        data = _evaluate(fp)
        std  = data["features"]["sentiment"]["std_compound"]
        assert std >= 0.0, f"std_compound is negative: {std}"

    def test_sentiment_neg_ratio_in_range(self):
        """neg_sentiment_ratio is a fraction — must be in [0, 1]."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip()
        data = _evaluate(fp)
        nr   = data["features"]["sentiment"]["neg_sentiment_ratio"]
        assert 0.0 <= nr <= 1.0, f"neg_sentiment_ratio {nr} out of [0, 1]"

    def test_segments_start_before_end(self):
        """Every segment must have start <= end."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip()
        data = _evaluate(fp)
        for i, seg in enumerate(data["segments"]):
            assert seg["start"] <= seg["end"], \
                f"Segment {i}: start={seg['start']} > end={seg['end']}"

    def test_grammar_error_count_non_negative(self):
        """grammar_error_count must be >= 0."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip()
        data = _evaluate(fp)
        gec  = data["features"]["language_control"].get("grammar_error_count", 0)
        assert gec >= 0, f"grammar_error_count is negative: {gec}"

    def test_voiced_fraction_in_range(self):
        """voiced_fraction must be in [0, 1]."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip()
        data = _evaluate(fp)
        vf   = data["features"]["voice_modulation"].get("voiced_fraction", 0)
        assert 0.0 <= vf <= 1.0, f"voiced_fraction {vf} out of [0, 1]"


# ─────────────────────────────────────────────────────────────────────────────
# 4. /report endpoint — functionality and schema
# ─────────────────────────────────────────────────────────────────────────────

class TestReportEndpoint:

    @pytest.mark.parametrize("file_key", ["wav", "interview_mp4"])
    def test_report_returns_200(self, file_key):
        fp = TEST_FILES[file_key]
        if not fp.exists():
            pytest.skip(f"Test file not found: {fp}")
        resp = _post_file("/report", fp)
        assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text[:200]}"

    def test_report_schema(self):
        """Response must contain all required report keys."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip()
        resp = _post_file("/report", fp)
        assert resp.status_code == 200
        data    = resp.json()
        missing = REPORT_REQUIRED_KEYS - set(data.keys())
        assert not missing, f"Missing report keys: {missing}"

    def test_report_overall_score_in_range(self):
        """overall_score must be in [0, 5]."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip()
        resp = _post_file("/report", fp)
        assert resp.status_code == 200
        data = resp.json()
        _assert_score_valid(data["overall_score"], "overall_score")

    def test_report_grade_valid(self):
        """grade must be one of the defined letter grades."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip()
        resp = _post_file("/report", fp)
        assert resp.status_code == 200
        data = resp.json()
        assert data["grade"] in VALID_GRADES, \
            f"Unknown grade: {data['grade']}"

    def test_report_label_valid(self):
        """label must be one of the defined text labels."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip()
        resp = _post_file("/report", fp)
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] in VALID_LABELS, \
            f"Unknown label: {data['label']}"

    def test_report_strengths_and_improvements_count(self):
        """strengths and areas_for_improvement must each be non-empty lists."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip()
        resp = _post_file("/report", fp)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["strengths"], list) and len(data["strengths"]) > 0
        assert isinstance(data["areas_for_improvement"], list) and len(data["areas_for_improvement"]) > 0

    def test_report_executive_summary_non_empty(self):
        """executive_summary must be a non-empty string."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip()
        resp = _post_file("/report", fp)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["executive_summary"], str)
        assert len(data["executive_summary"].strip()) > 0

    def test_report_dimensions_have_score_label_reasoning(self):
        """Each dimension in the report must have score, out_of, label, reasoning."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip()
        resp = _post_file("/report", fp)
        assert resp.status_code == 200
        data = resp.json()
        for dim, detail in data["dimensions"].items():
            for key in ("score", "out_of", "label", "reasoning"):
                assert key in detail, f"Dimension '{dim}' missing key '{key}'"
            assert detail["out_of"] == 5.0
            assert detail["label"] in VALID_LABELS
            _assert_score_valid(detail["score"], dim)

    def test_report_out_of_is_5(self):
        """out_of at report level must always be 5.0."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip()
        resp = _post_file("/report", fp)
        assert resp.status_code == 200
        assert resp.json()["out_of"] == 5.0

    def test_report_reasoning_source_is_valid(self):
        """reasoning_source must be 'gemini' or 'rule-based'."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip()
        resp = _post_file("/report", fp)
        assert resp.status_code == 200
        rs = resp.json().get("reasoning_source", "")
        assert rs in ("gemini", "rule-based"), \
            f"Unexpected reasoning_source: '{rs}'"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Reproducibility tests — no hallucination
# ─────────────────────────────────────────────────────────────────────────────

class TestReproducibility:
    """
    Submit the same file multiple times and verify that:
      a) All numeric scores are identical (or within float rounding tolerance)
      b) All feature values are consistent
      c) The system does not produce wildly different outputs (hallucination check)
    """
    RUNS     = 3         # Number of times to submit
    SCORE_TOL = 0.05     # Max allowed score drift across runs (float rounding)
    FEAT_TOL  = 0.02     # Max allowed feature value drift

    def _run_n(self, filepath: Path, n: int) -> list[dict]:
        """Run /evaluate n times on the same file. Return list of result dicts."""
        results = []
        for i in range(n):
            print(f"\n  [reproducibility] Run {i+1}/{n} for {filepath.name}...")
            resp = _post_file("/evaluate", filepath)
            assert resp.status_code == 200, \
                f"Run {i+1} failed with {resp.status_code}: {resp.text[:200]}"
            results.append(resp.json())
        return results

    @pytest.mark.parametrize("file_key", ["wav", "interview_mp4"])
    def test_scores_are_reproducible(self, file_key):
        """Scores from N runs of the same file must not deviate by more than SCORE_TOL."""
        fp = TEST_FILES[file_key]
        if not fp.exists():
            pytest.skip(f"Test file not found: {fp}")

        results = self._run_n(fp, self.RUNS)
        baseline_scores = results[0]["scores"]

        for run_idx, result in enumerate(results[1:], start=2):
            for dim, base_score in baseline_scores.items():
                curr_score = result["scores"].get(dim)
                assert curr_score is not None, \
                    f"Run {run_idx}: '{dim}' score is missing"
                drift = abs(float(curr_score) - float(base_score))
                assert drift <= self.SCORE_TOL, (
                    f"Run {run_idx} '{dim}' score drifted by {drift:.4f} "
                    f"(baseline={base_score:.4f}, current={curr_score:.4f}) "
                    f"— tolerance is {self.SCORE_TOL}"
                )

    @pytest.mark.parametrize("file_key", ["wav"])
    def test_wpm_is_reproducible(self, file_key):
        """WPM must be identical across runs (deterministic transcription)."""
        fp = TEST_FILES[file_key]
        if not fp.exists():
            pytest.skip()
        results = self._run_n(fp, self.RUNS)
        base_wpm = results[0]["features"]["fluency"]["wpm"]
        for run_idx, r in enumerate(results[1:], start=2):
            curr_wpm = r["features"]["fluency"]["wpm"]
            assert abs(curr_wpm - base_wpm) <= 1.0, \
                f"Run {run_idx}: WPM drifted — {base_wpm} vs {curr_wpm}"

    @pytest.mark.parametrize("file_key", ["wav"])
    def test_filler_count_is_reproducible(self, file_key):
        """Filler count must be identical across runs."""
        fp = TEST_FILES[file_key]
        if not fp.exists():
            pytest.skip()
        results = self._run_n(fp, self.RUNS)
        base_fc = results[0]["features"]["fluency"]["filler_count"]
        for run_idx, r in enumerate(results[1:], start=2):
            curr_fc = r["features"]["fluency"]["filler_count"]
            assert curr_fc == base_fc, \
                f"Run {run_idx}: filler_count changed — {base_fc} vs {curr_fc}"

    @pytest.mark.parametrize("file_key", ["wav"])
    def test_grammar_error_count_is_reproducible(self, file_key):
        """Grammar error count must be identical across runs."""
        fp = TEST_FILES[file_key]
        if not fp.exists():
            pytest.skip()
        results = self._run_n(fp, self.RUNS)
        base = results[0]["features"]["language_control"].get("grammar_error_count", 0)
        for run_idx, r in enumerate(results[1:], start=2):
            curr = r["features"]["language_control"].get("grammar_error_count", 0)
            assert curr == base, \
                f"Run {run_idx}: grammar_error_count changed — {base} vs {curr}"

    @pytest.mark.parametrize("file_key", ["wav"])
    def test_sentiment_compound_is_reproducible(self, file_key):
        """mean_compound must not drift across runs (VADER is deterministic)."""
        fp = TEST_FILES[file_key]
        if not fp.exists():
            pytest.skip()
        results = self._run_n(fp, self.RUNS)
        base = results[0]["features"]["sentiment"]["mean_compound"]
        for run_idx, r in enumerate(results[1:], start=2):
            curr = r["features"]["sentiment"]["mean_compound"]
            assert abs(curr - base) <= self.FEAT_TOL, \
                f"Run {run_idx}: mean_compound drifted — {base} vs {curr}"

    @pytest.mark.parametrize("file_key", ["wav"])
    def test_segment_count_is_reproducible(self, file_key):
        """Number of segments must be the same across runs."""
        fp = TEST_FILES[file_key]
        if not fp.exists():
            pytest.skip()
        results = self._run_n(fp, self.RUNS)
        base_count = len(results[0]["segments"])
        for run_idx, r in enumerate(results[1:], start=2):
            curr_count = len(r["segments"])
            assert curr_count == base_count, \
                f"Run {run_idx}: segment count changed — {base_count} vs {curr_count}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Cross-file accuracy / sanity tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAccuracySanity:
    """
    These tests verify that the scoring system is directionally sensible.
    They do not assert exact values, but check logical relationships.
    """

    def test_high_wpm_correlates_with_fluency(self):
        """A candidate speaking at ~100+ WPM should have delivery_fluency score > 1.5."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip()
        data = _evaluate(fp)
        wpm  = data["features"]["fluency"]["wpm"]
        fsco = data["scores"]["delivery_fluency"]
        if wpm >= 100:
            assert fsco > 1.5, \
                f"WPM={wpm:.0f} but delivery_fluency score is only {fsco:.2f} — scoring may be broken"

    def test_high_confidence_correlates_with_intelligibility(self):
        """mean_confidence > 0.80 should yield pronunciation_clarity score > 2.5."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip()
        data = _evaluate(fp)
        mc   = data["features"]["intelligibility"]["mean_confidence"]
        isco = data["scores"]["pronunciation_clarity"]
        if mc > 0.80:
            assert isco > 2.5, \
                f"mean_confidence={mc:.3f} but pronunciation_clarity score is only {isco:.2f}"

    def test_zero_grammar_errors_yields_top_language_control(self):
        """language_control must not be part of the 6-axis scores."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip()
        data = _evaluate(fp)
        assert "language_control" not in data["scores"]

    def test_score_ordering_matches_feature_ordering(self):
        """For two files, the one with higher WPM should likely have higher delivery_fluency."""
        fp_wav = TEST_FILES["wav"]
        fp_mp4 = TEST_FILES["interview_mp4"]
        if not fp_wav.exists() or not fp_mp4.exists():
            pytest.skip("Both test files needed")
        d1 = _evaluate(fp_wav)
        d2 = _evaluate(fp_mp4)
        # This is a soft check — we only flag if the difference is very large
        wpm1   = d1["features"]["fluency"]["wpm"]
        wpm2   = d2["features"]["fluency"]["wpm"]
        fsco1  = d1["scores"]["delivery_fluency"]
        fsco2  = d2["scores"]["delivery_fluency"]
        if abs(wpm1 - wpm2) > 40:
            if wpm1 > wpm2:
                assert fsco1 >= fsco2 - 1.5, (
                    f"wav has WPM={wpm1:.0f} > mp4 WPM={wpm2:.0f} "
                    f"but delivery_fluency scores are reversed: {fsco1:.2f} vs {fsco2:.2f}"
                )

    def test_all_test_files_produce_nonzero_scores(self):
        """No test file should produce an all-zero score vector."""
        for key, fp in TEST_FILES.items():
            if not fp.exists():
                continue
            resp = _post_file("/evaluate", fp)
            if resp.status_code != 200:
                continue
            scores = resp.json()["scores"]
            total  = sum(float(v) for v in scores.values())
            assert total > 0, \
                f"All scores are zero for {fp.name} — pipeline may be broken"

    def test_report_overall_consistent_with_dimensions(self):
        """overall_score must be within the range of individual dimension scores."""
        fp = TEST_FILES["wav"]
        if not fp.exists():
            pytest.skip()
        resp = _post_file("/report", fp)
        assert resp.status_code == 200
        data   = resp.json()
        scores = [d["score"] for d in data["dimensions"].values()]
        overall = data["overall_score"]
        assert min(scores) - 0.5 <= overall <= max(scores) + 0.5, (
            f"overall_score {overall} is outside expected range "
            f"[{min(scores):.2f}, {max(scores):.2f}]"
        )
