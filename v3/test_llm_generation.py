"""
test_llm_generation.py — Unit tests for Gemini summary and description generation.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from google import genai
from google.genai import types

from v3.models import RawFeatures, SkillsAssessment, AxisResult, StyleProfile
from v3.pipeline.communication_summary import generate_communication_summary
from v3.pipeline.description import generate_description_from_style_profile
from v3.config import GEMINI_MODEL


@pytest.mark.anyio
async def test_communication_summary_success():
    raw = RawFeatures()
    skills = SkillsAssessment(
        fluency=AxisResult(score=4.0, confidence=1.0, band="Good"),
        intelligibility=AxisResult(score=5.0, confidence=1.0, band="Excellent"),
        lexical_structural=AxisResult(score=3.5, confidence=1.0, band="Good"),
        narrative_evidence=AxisResult(score=3.0, confidence=1.0, band="Average"),
        vocal_delivery=AxisResult(score=4.2, confidence=1.0, band="Good"),
        composite_score=82.0,
        composite_band="Good",
        review_required=False,
    )

    mock_response = MagicMock()
    mock_response.text = '{"summary": ["Line 1 text.", "Line 2 text."]}'

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch("v3.pipeline.communication_summary.GEMINI_API_KEY", "dummy_key"), \
         patch("v3.pipeline.communication_summary._get_client", return_value=mock_client):
        res = await generate_communication_summary(raw, skills)
        assert res.summary == ["Line 1 text.", "Line 2 text."]

        # Verify call arguments
        mock_client.aio.models.generate_content.assert_called_once()
        kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        assert kwargs["model"] == GEMINI_MODEL
        assert isinstance(kwargs["config"], types.GenerateContentConfig)
        assert kwargs["config"].response_mime_type == "application/json"


@pytest.mark.anyio
async def test_communication_summary_fallback():
    raw = RawFeatures(filler_word_ratio=0.08)
    skills = SkillsAssessment(
        fluency=AxisResult(score=2.5, confidence=1.0, band="Below Average"),
        intelligibility=AxisResult(score=5.0, confidence=1.0, band="Excellent"),
        lexical_structural=AxisResult(score=3.5, confidence=1.0, band="Good"),
        narrative_evidence=AxisResult(score=3.0, confidence=1.0, band="Average"),
        vocal_delivery=AxisResult(score=4.2, confidence=1.0, band="Good"),
        composite_score=55.0,
        composite_band="Average",
        review_required=False,
    )

    # Force config missing fallback
    with patch("v3.pipeline.communication_summary.GEMINI_API_KEY", ""):
        res = await generate_communication_summary(raw, skills)
        # Check that we fall back to rule-based summary
        assert len(res.summary) == 2
        assert "adequate" in res.summary[0]
        assert "softens" in res.summary[1] or "Strongest" in res.summary[1] or "response" in res.summary[1]


@pytest.mark.anyio
async def test_description_generation_success():
    style = StyleProfile(
        systematic_thinking=80.0,
        collaborative_orientation=60.0,
        analytical_precision=70.0,
        expressive_engagement=50.0,
        action_orientation=40.0,
        dominant_archetype="architect",
        archetype_blend={"architect": 50.0, "connector": 20.0, "synthesizer": 10.0, "analyst": 10.0, "pragmatist": 10.0},
    )

    mock_response = MagicMock()
    mock_response.text = "This is a mocked style description of the candidate."

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch("v3.pipeline.description.GEMINI_API_KEY", "dummy_key"), \
         patch("v3.pipeline.description._get_client", return_value=mock_client):
        res = await generate_description_from_style_profile(style)
        assert res == "This is a mocked style description of the candidate."

        mock_client.aio.models.generate_content.assert_called_once()
        kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        assert kwargs["model"] == GEMINI_MODEL
        assert isinstance(kwargs["config"], types.GenerateContentConfig)


@pytest.mark.anyio
async def test_description_generation_fallback():
    style = StyleProfile(
        systematic_thinking=80.0,
        collaborative_orientation=60.0,
        analytical_precision=70.0,
        expressive_engagement=50.0,
        action_orientation=40.0,
        dominant_archetype="architect",
        archetype_blend={"architect": 50.0, "connector": 20.0, "synthesizer": 10.0, "analyst": 10.0, "pragmatist": 10.0},
    )

    with patch("v3.pipeline.description.GEMINI_API_KEY", ""):
        res = await generate_description_from_style_profile(style)
        assert "Architect" in res


@pytest.mark.anyio
async def test_communication_summary_retry_on_503():
    raw = RawFeatures()
    skills = SkillsAssessment(
        fluency=AxisResult(score=4.0, confidence=1.0, band="Good"),
        intelligibility=AxisResult(score=5.0, confidence=1.0, band="Excellent"),
        lexical_structural=AxisResult(score=3.5, confidence=1.0, band="Good"),
        narrative_evidence=AxisResult(score=3.0, confidence=1.0, band="Average"),
        vocal_delivery=AxisResult(score=4.2, confidence=1.0, band="Good"),
        composite_score=82.0,
        composite_band="Good",
        review_required=False,
    )

    mock_response = MagicMock()
    mock_response.text = '{"summary": ["Retry line 1.", "Retry line 2."]}'

    # Side effect: raise 503 twice, then succeed
    mock_call = AsyncMock()
    mock_call.side_effect = [
        Exception("503 Service Unavailable"),
        Exception("503 Service Unavailable"),
        mock_response
    ]

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = mock_call

    with patch("v3.pipeline.communication_summary.GEMINI_API_KEY", "dummy_key"), \
         patch("v3.pipeline.communication_summary._get_client", return_value=mock_client), \
         patch("asyncio.sleep", AsyncMock()) as mock_sleep:  # mock sleep to keep tests fast!
        res = await generate_communication_summary(raw, skills)
        assert res.summary == ["Retry line 1.", "Retry line 2."]

        # Verify it was called exactly 3 times (2 failures + 1 success)
        assert mock_call.call_count == 3
        # Verify sleep was called twice with exponential backoff (1s, 2s)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)

