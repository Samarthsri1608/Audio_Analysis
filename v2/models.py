"""
models.py — Pydantic schemas for V2 pipeline inputs and outputs.

Updated for Zeko Unified Communication Framework v1:
- RawFeatures extended with System A scoring fields
- SkillsAssessment added for System A output
- PersonalityResult: System B (style/archetype) endpoint response
- CommunicationResult: System A (skills scoring) endpoint response
- FeatureCacheEntry: lightweight cache object storing only raw features
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ── Raw features (20 features per framework §1.1) ─────────────────────────────

class RawFeatures(BaseModel):
    # ── Text features ──────────────────────────────────────────────────────────
    # Fluency
    speech_rate_wpm: float = Field(0.0, ge=0)
    speech_rate_variability: float = Field(0.0, ge=0, le=1)
    filler_word_ratio: float = Field(0.0, ge=0, le=1)           # F02

    # Lexical (MATTR replaces TTR per framework §3.2 Change 1)
    lexical_mattr: float = Field(0.0, ge=0, le=1)               # F06 — MATTR
    lexical_rare_word_ratio: float = Field(0.0, ge=0, le=1)     # F07
    total_words: float = Field(0.0, ge=0)
    vocabulary_density: float = Field(0.0, ge=0, le=1)          # legacy alias → MATTR

    # Discourse
    discourse_connectors: float = Field(0.0, ge=0)              # F08 — unique connector count
    discourse_tier1: float = Field(0.0, ge=0)                   # F09 — Tier-1 connectors
    sbert_coherence: float = Field(0.65, ge=0, le=1)            # F13 — sentence cosine similarity

    # Narrative & Evidence
    ner_entity_density: float = Field(0.0, ge=0)                # F11 — entities/min
    metric_density: float = Field(0.0, ge=0)                    # F12 — metrics/min
    narrative_arc_score: float = Field(0.0, ge=0, le=1)         # narrative completeness 0–1

    # Style signals (used by System B)
    collaborative_language_ratio: float = Field(0.0, ge=0, le=1)  # F14
    question_density: float = Field(0.0, ge=0)                    # F15
    empathetic_language_score: float = Field(0.0, ge=0, le=1)     # F16
    avg_sentence_length: float = Field(0.0, ge=0)                 # F20
    logical_connector_density: float = Field(0.0, ge=0, le=1)     # legacy alias

    # ── Vocal features ─────────────────────────────────────────────────────────
    pitch_variation: float = Field(0.0, ge=0)    # F10 — CV-normalized (dialect-neutral)
    vocal_confidence: float = Field(0.0, ge=0, le=1)   # F17
    speech_fluency: float = Field(0.0, ge=0, le=1)     # F18
    stress_markers: float = Field(0.0, ge=0, le=1)     # F19 — composure (higher = calmer)
    fluency_pause_dur: float = Field(0.0, ge=0)        # F03 — mean pause duration (s)
    fluency_pause_freq: float = Field(0.0, ge=0)       # F04 — pause count
    voiced_fraction: float = Field(0.0, ge=0, le=1)   # fraction of audio that is speech

    # ── ASR quality ────────────────────────────────────────────────────────────
    intel_confidence: float = Field(0.75, ge=0, le=1)  # F05 — bias-corrected Whisper confidence
    is_short_duration: bool = False


# ── Feature cache entry ───────────────────────────────────────────────────────

class FeatureCacheEntry(BaseModel):
    """
    Lightweight cache object stored after feature extraction.
    Only raw features + minimal metadata — no evaluation results.
    Evaluation (System A / System B) is computed on demand per endpoint.
    """
    response_id: str
    raw_features: RawFeatures
    transcript: str = ""
    duration_ms: float = 0.0


# ── System A — Skills Assessment (5 axes, 0–5 each) ──────────────────────────

class AxisResult(BaseModel):
    """Single axis score with reliability metadata."""
    score: float = Field(0.0, ge=0, le=5)          # 0–5 band score
    confidence: float = Field(1.0, ge=0, le=1)     # how reliable is this measurement
    band: str = "Average"                           # Poor / Below Average / Average / Good / Excellent
    flags: list[str] = Field(default_factory=list)  # edge case flags


class SkillsAssessment(BaseModel):
    """
    System A — Communication Skills Engine output.
    Scores candidate on measurable communication quality dimensions.
    """
    # 5-axis scores
    fluency: AxisResult
    intelligibility: AxisResult
    lexical_structural: AxisResult
    narrative_evidence: AxisResult
    vocal_delivery: AxisResult

    # Composite score (0–100)
    composite_score: float = Field(0.0, ge=0, le=100)
    composite_band: str = "Average"

    # Flags
    grammar_pending: bool = True    # True until grammar axis is implemented
    review_required: bool = False   # triggered by human review protocol
    role_profile: str = "default"   # default / client_facing / technical / leadership


# ── System B — Communication Style signals (0-100 each) ──────────────────────

class CommunicationSignals(BaseModel):
    systematic_thinking: float = Field(0.0, ge=0, le=100)
    collaborative_orientation: float = Field(0.0, ge=0, le=100)
    analytical_precision: float = Field(0.0, ge=0, le=100)
    expressive_engagement: float = Field(0.0, ge=0, le=100)
    action_orientation: float = Field(0.0, ge=0, le=100)


# ── Archetype blend ───────────────────────────────────────────────────────────

class ArchetypeBlend(BaseModel):
    architect: float = Field(0.0, ge=0, le=100)
    connector: float = Field(0.0, ge=0, le=100)
    synthesizer: float = Field(0.0, ge=0, le=100)
    analyst: float = Field(0.0, ge=0, le=100)
    pragmatist: float = Field(0.0, ge=0, le=100)


# ── System B — Style Profile ──────────────────────────────────────────────────

class StyleProfile(BaseModel):
    """
    System B — Communication Style Engine output.

    Describes HOW the candidate communicates (style/personality).
    This is non-evaluative — there is no good or bad archetype.
    Never merge or display this alongside System A scores in the same view.
    """
    # 5 communication signals (0–100)
    systematic_thinking: float = Field(0.0, ge=0, le=100)
    collaborative_orientation: float = Field(0.0, ge=0, le=100)
    analytical_precision: float = Field(0.0, ge=0, le=100)
    expressive_engagement: float = Field(0.0, ge=0, le=100)
    action_orientation: float = Field(0.0, ge=0, le=100)

    # Archetype blend (percentages summing to 100)
    archetype_blend: dict = Field(
        default_factory=lambda: {
            "architect": 0.0, "connector": 0.0, "synthesizer": 0.0,
            "analyst": 0.0, "pragmatist": 0.0,
        }
    )
    dominant_archetype: str = "synthesizer"

    # GMM readiness metadata (Framework §3.4)
    gmm_trained: bool = False       # False until 50+ labeled interviews exist
    centroid_version: str = "v0-n86-june2026"   # replace when GMM is trained

    # Role-fit assessment (Framework §4.3)
    role_fit: dict = Field(
        default_factory=lambda: {"role": "default", "match": "n/a",
                                 "required_gaps": [], "met_requirements": []}
    )


# ── Endpoint response models ──────────────────────────────────────────────────

class PersonalityResult(BaseModel):
    """
    Response model for GET /v2/analyse/{response_id}/personality.

    System B output: HOW the candidate communicates — style, archetype blend,
    and role-fit signals. Non-evaluative (no good/bad archetype).
    """
    response_id: str
    status: str                          # "success" | "error"
    error: Optional[str] = None
    duration_ms: float = 0.0
    transcript: Optional[str] = None

    # System B outputs
    style_profile: Optional[StyleProfile] = None
    signals: Optional[CommunicationSignals] = None
    archetype_blend: Optional[ArchetypeBlend] = None
    dominant_archetype: Optional[str] = None

    # Optional LLM description
    description: Optional[str] = None


class CommunicationResult(BaseModel):
    """
    Response model for GET /v2/analyse/{response_id}/communication.

    V2 public contract: compact two-line summary, tags, and schema version only.
    """
    response_id: str
    status: str                          # "success" | "error"
    duration: float = Field(0.0, ge=0)
    result: "CommunicationSummary"
    tags: list[str] = Field(default_factory=list)
    schema_version: str = "v2"


class CommunicationSummary(BaseModel):
    """Two-line communication summary without raw feature disclosure."""
    summary: list[str] = Field(default_factory=list)

    @field_validator("summary")
    @classmethod
    def _validate_summary(cls, value: list[str]) -> list[str]:
        lines = [str(line).strip() for line in value if str(line).strip()]
        if len(lines) != 2:
            raise ValueError("summary must contain exactly two non-empty lines")
        return lines


# ── Request models ────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    response_id: str
    include_description: bool = True
    role_profile: str = "default"   # System A weight profile: default / client_facing / technical / leadership
    style_role: str = "default"     # System B role-fit target: default / software_engineer / sales / product_manager / team_lead / data_scientist
