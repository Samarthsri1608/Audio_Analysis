"""
models.py — Pydantic schemas for the V4 audio-only proctoring pipeline.

No text/transcription fields — all signals are derived purely from the audio
waveform.  A transcript is never a model input; it may only be generated
downstream by a human reviewer tool after a flag is raised.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ── Per-question audio features ───────────────────────────────────────────────

class AudioFeatures(BaseModel):
    """
    Audio-only feature vector for a single answer.

    All fields are Optional[float] rather than float because:
    - Some features are undefined for very short answers (e.g. f0 on silence).
    - None explicitly signals "not computable" — never silently zero-filled.
    """
    # Pitch (fundamental frequency)
    f0_mean: Optional[float] = None          # Mean F0 over voiced frames (Hz)
    f0_std: Optional[float] = None           # Std dev of F0 over voiced frames (Hz)

    # Speaking pace proxy (no ASR needed)
    speech_rate_proxy: Optional[float] = None  # Voiced-frame fraction / duration (proxy for pace)

    # Pausing behaviour
    pause_ratio: Optional[float] = None        # Silence fraction over total answer duration
    pause_duration_mean: Optional[float] = None  # Mean length of individual silence segments (s)
    pause_duration_max: Optional[float] = None   # Max silence segment length (s)

    # Response latency: time from audio start to first detected speech
    response_latency: Optional[float] = None   # Seconds

    # Energy / vocal effort
    energy_mean: Optional[float] = None        # RMS energy mean over voiced frames
    energy_std: Optional[float] = None         # RMS energy std dev over voiced frames

    # Timbre (vocal-tract consistency)
    mfcc_mean: Optional[list[float]] = None    # Mean of MFCC coefficients 1–13
    mfcc_std: Optional[list[float]] = None     # Std dev of MFCC coefficients 1–13

    # Monotone / scripted-delivery proxy
    spectral_flatness_mean: Optional[float] = None  # Mean spectral flatness over voiced frames

    # Room / acoustic fingerprint for Track C environment-shift detection
    room_fingerprint: Optional[float] = None   # Spectral rolloff mean (Hz) — room proxy
    noise_floor_centroid: Optional[float] = None  # Spectral centroid of noise segments (Hz)

    # Speech duration metrics
    total_duration_s: float = 0.0              # Total file duration (seconds)
    speech_duration_s: float = 0.0            # Duration of voiced speech (seconds)


# ── Evaluability result ───────────────────────────────────────────────────────

class EvaluabilityResult(BaseModel):
    """Gate result: is this answer usable for proctoring analysis?"""
    evaluable: bool
    not_evaluable_reason: Optional[str] = None  # None when evaluable=True
    # Valid reasons (per spec §8):
    #   "file_not_found" | "corrupt_audio" | "insufficient_speech_duration"
    #   "no_speech_detected" | "low_signal_quality" | "audio_quality_issue"


# ── Track A result ────────────────────────────────────────────────────────────

class TrackAResult(BaseModel):
    """Output of the self-baseline deviation track for one answer."""
    available: bool = False              # False when cold-start (not enough history)
    deviation_score: Optional[float] = None  # Max robust |z-score| across features
    flagged: bool = False
    z_scores: dict[str, Optional[float]] = Field(default_factory=dict)
    # Per-feature z-scores (None = feature not computable for this answer)


# ── Track C result ────────────────────────────────────────────────────────────

class TrackCResult(BaseModel):
    """Output of the naturalness/mechanism track for one answer."""
    flagged: bool = False
    signals: list[str] = Field(default_factory=list)
    # Possible values:
    #   "latency_fluency_mismatch"
    #   "acoustic_environment_shift"
    #   "background_noise_shift"
    signal_details: dict[str, float] = Field(default_factory=dict)
    # Raw values that triggered each signal (for reviewer context)


# ── Evidence payload (per question) ──────────────────────────────────────────

class QuestionEvidencePayload(BaseModel):
    """
    Complete output for a single question's audio analysis.
    This is the unit of output attached to the interview report.
    """
    q_no: int
    evaluable: bool
    not_evaluable_reason: Optional[str] = None

    # Flags
    flagged_for_review: bool = False
    confidence: str = "low"
    # confidence levels (per spec §7):
    #   "high"               — both Track A and Track C fired
    #   "medium"             — Track C alone
    #   "medium"             — Track A alone (with baseline)
    #   "medium_provisional" — Track C alone, cold-start (no Track A baseline yet)
    #   "low"                — neither track fired

    # Track outputs
    track_a: Optional[TrackAResult] = None
    track_c: Optional[TrackCResult] = None

    # Cold-start metadata
    is_cold_start: bool = False

    # Key contributing features (for reviewer UI — not raw feature dump)
    contributing_features: dict[str, Optional[float]] = Field(default_factory=dict)


# ── Interview-level proctoring response ───────────────────────────────────────

class ProctoringResponse(BaseModel):
    """Top-level response for the /v4/internal/analyse/{response_id}/proctoring endpoint."""
    response_id: str
    status: str                          # "success" | "fail"
    error: Optional[dict] = None         # populated when status="fail"

    flagged_questions: list[int] = Field(default_factory=list)
    question_evidence: list[QuestionEvidencePayload] = Field(default_factory=list)

    # Summary stats
    total_questions_evaluated: int = 0
    total_questions_flagged: int = 0

    schema_version: str = "v4-audio-only"
