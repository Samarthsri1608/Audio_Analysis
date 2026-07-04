# Zeko AI — Unified Communication Analysis Framework
## Combined Candidate Evaluation & Communication Style Profiling

**Version:** 1.0  
**Prepared for:** Zeko AI — Avya Evaluation Engine  
**Date:** June 2026  
**Status:** Agent-Ready Implementation Spec  
**Supersedes:** `communication_report.md` (5-axis skills model) + `Communication_Style_Evaluation_Framework.md` (style profiling model v2.0)

---

## Overview

This document defines two parallel but distinct analytical systems that run on the same audio input:

| System | Name | Purpose | Output | Audience |
|---|---|---|---|---|
| **System A** | Communication Skills Engine | Score communication quality | 5-axis band score + composite (0–100) | Recruiters, ATS pass/fail |
| **System B** | Communication Style Engine | Describe communication personality | 5-archetype blend % | Hiring managers, role-fit matching |

**Critical design principle:** These are not the same system. System A grades. System B describes. They share a feature extraction pipeline but produce independent outputs. Never merge or conflate the two outputs in the UI.

---

## Part 1: Shared Feature Extraction Pipeline

Both systems draw from a single unified feature extraction run. This avoids double-processing audio and ensures both systems operate on identical inputs.

### 1.1 Complete Feature Inventory (17 features)

| # | Feature | Extraction Source | Used By |
|---|---|---|---|
| F01 | `fluency_wpm` | Word timestamps (Whisper) | Skills: Fluency axis |
| F02 | `fluency_filler_rate` | Transcript NLP | Skills: Fluency axis + Style: Structural |
| F03 | `fluency_pause_dur` | Word timestamps | Skills: Fluency axis |
| F04 | `fluency_pause_freq` | Word timestamps | Skills: Intelligibility axis |
| F05 | `intel_confidence` | Whisper token confidence | Skills: Intelligibility axis |
| F06 | `lexical_mattr` | Transcript NLP (MATTR) | Skills: Lexical axis |
| F07 | `lexical_rare` | Transcript NLP | Skills: Lexical axis |
| F08 | `discourse_connectors` | Transcript NLP | Skills: Discourse axis |
| F09 | `discourse_tier1` | Transcript NLP | Skills: Discourse axis |
| F10 | `pitch_std` | librosa / parselmouth | Skills: Vocal Delivery axis + Style: Vocal |
| F11 | `ner_entity_density` | SpaCy + Indian Gazetteer | Skills: Narrative & Evidence axis |
| F12 | `metric_density` | Regex (numbers + % patterns) | Skills: Narrative & Evidence axis + Style: Precision |
| F13 | `sbert_coherence` | Sentence-BERT (multilingual) | Skills: Discourse axis |
| F14 | `collaborative_ratio` | Pronoun NLP | Style: Interpersonal |
| F15 | `question_density` | Sentence tokenizer | Style: Interpersonal |
| F16 | `empathetic_markers` | Keyword NLP | Style: Interpersonal |
| F17 | `vocal_confidence` | librosa (spectral stability) | Style: Vocal |
| F18 | `speech_fluency_score` | Word timestamp gaps | Style: Vocal |
| F19 | `stress_markers` | librosa (F0 + spectral) | Style: Vocal |
| F20 | `avg_sentence_length` | Sentence tokenizer | Style: Structural |

> **Note on `sentiment_compound` (VADER):** Removed from both systems. The Style Framework (v2.0) explicitly excludes sentiment detection for accuracy and fairness reasons. The Skills model's Sentiment axis has been replaced by a strengthened Discourse + Narrative axis, which captures assertiveness and confidence more reliably.

### 1.2 Key Implementation Notes for the Pipeline

**ASR — Whisper Accent Priming (Indian English)**

All Whisper transcription calls must include accent priming to prevent score deflation for Indian English speakers. This is an instrument correction, not a threshold adjustment — the goal is to make ASR equally accurate for all speakers.

```python
result = whisper_model.transcribe(
    audio_path,
    initial_prompt="The speaker has an Indian accent. Transcription of technical interview response in Indian English.",
    language="en",
    temperature=0.0
)
```

**NER — Indian Gazetteer (Required for F11)**

```python
import spacy
from spacy.pipeline import EntityRuler

nlp = spacy.load("en_core_web_sm")
ruler = nlp.add_pipe("entity_ruler", before="ner")

patterns = [
    # Educational institutions
    {"label": "ORG", "pattern": "IIT"}, {"label": "ORG", "pattern": "NIT"},
    {"label": "ORG", "pattern": "IIM"}, {"label": "ORG", "pattern": "BITS Pilani"},
    {"label": "ORG", "pattern": "IIT Madras"}, {"label": "ORG", "pattern": "IIT Bombay"},
    {"label": "ORG", "pattern": "IIT Delhi"},
    # Companies
    {"label": "ORG", "pattern": "Infosys"}, {"label": "ORG", "pattern": "TCS"},
    {"label": "ORG", "pattern": "Wipro"}, {"label": "ORG", "pattern": "HCL"},
    {"label": "ORG", "pattern": "Razorpay"}, {"label": "ORG", "pattern": "Zerodha"},
    {"label": "ORG", "pattern": "Flipkart"}, {"label": "ORG", "pattern": "Swiggy"},
    {"label": "ORG", "pattern": "Zomato"}, {"label": "ORG", "pattern": "BYJU"},
    # Geographies
    {"label": "GPE", "pattern": "Bengaluru"}, {"label": "GPE", "pattern": "Pune"},
    {"label": "GPE", "pattern": "Hyderabad"}, {"label": "GPE", "pattern": "Chennai"},
    {"label": "GPE", "pattern": "Noida"}, {"label": "GPE", "pattern": "Gurugram"},
]
ruler.add_patterns(patterns)
```

> **Recalibration needed:** Expand this list systematically before production. Tier 2/3 cities, state universities, and startup ecosystem terms (crore, lakh, bootstrapped, Series A) should all be included.

**Sentence Embeddings — Multilingual Model (Required for F13)**

Replace any English-only sentence transformer with:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
# OR: 'distiluse-base-multilingual-cased-v1'
```

This maps Indian English spelling variations and ASR phonetic errors to the same semantic space, preserving coherence scores.

**Homophone Correction Dictionary (Required for Discourse Axis)**

Apply pre-processing to fix common ASR mismatches for Indian English discourse connectors before scoring:

```python
HOMOPHONE_CORRECTIONS = {
    "hens": "hence", "more over": "moreover",
    "consequent league": "consequently",
    "there for": "therefore", "how ever": "however",
    "as such": "as such",  # preserve — valid Indian English formal register
    "itself": "itself",    # preserve — used for emphasis in Indian English
}
```

---

## Part 2: System A — Communication Skills Engine

### 2.1 Purpose and Principles

System A measures **communication quality** — how well a candidate communicates, independent of style. Output is a scored, bandable result suitable for recruiter review and ATS integration.

**Calibration philosophy:** All calibration targets make the measurement instrument accurate for all speakers. Thresholds are universal — calibration happens at the feature extraction layer, not the scoring layer.

### 2.2 Five-Axis Model (Revised)

#### Axis 1 — Fluency

**What it measures:** Natural speech flow — pace, filler discipline, pause management.

| Feature | Role | Scoring Method |
|---|---|---|
| `fluency_wpm` | Primary | Percentile (non-normal, skew +1.40) |
| `fluency_filler_rate` | Secondary (inverted) | Percentile (non-normal, skew +3.65) |
| `fluency_pause_dur` | Penalty flag | Percentile — flag if > p80 |

**Empirical Thresholds (n=86, v0 — recalibrate at n=300):**

| Band | WPM | Filler Rate | Score |
|---|---|---|---|
| Excellent | > 93.3 | < 0.018 | 4–5 |
| Good | 67.3–93.3 | 0.018–0.025 | 3–4 |
| Average | 55.0–67.3 | 0.025–0.043 | 2–3 |
| Below Average | 44.7–55.0 | 0.043–0.058 | 1–2 |
| Poor | < 44.7 | > 0.058 | 0–1 |

> **Recalibration note:** Current WPM median is 61 — well below native professional rate (130–160 WPM) because candidates think while speaking. WPM bands should be re-evaluated against human quality ratings at n=300. High WPM alone does not confirm fluency — validate against human-annotated ground truth.

**Axis weight in composite:** 20%

---

#### Axis 2 — Intelligibility

**What it measures:** How clearly the speech is understood — ASR confidence as a proxy for acoustic clarity.

| Feature | Role | Scoring Method |
|---|---|---|
| `intel_confidence` | Primary | Z-score (normal, Shapiro p=0.270) |
| `fluency_pause_freq` | Secondary | Percentile |

> **Drop `pronunciation_score`:** r = 0.979 with `intel_confidence` — redundant. Use `intel_confidence` only.

**Empirical Thresholds:**

| Band | `intel_confidence` | Z-score | Score |
|---|---|---|---|
| Excellent | > 0.799 | z > +0.7 | 4–5 |
| Good | 0.759–0.799 | +0.2 to +0.7 | 3–4 |
| Average | 0.722–0.759 | ±0.2 | 2–3 |
| Below Average | 0.679–0.722 | −0.8 to −0.2 | 1–2 |
| Poor | < 0.679 | z < −0.8 | 0–1 |

**Indian English instrument correction:** Apply `+0.06` linear offset to `intel_confidence` before scoring, and use accent-primed Whisper (see Section 1.2). This corrects for ASR bias at the extraction layer — the threshold above is applied uniformly after correction.

**Axis weight in composite:** 20%

---

#### Axis 3 — Lexical & Structural Quality

**What it measures:** Vocabulary richness and logical organization of response. This axis **merges** the original "Logical Structure" and "Chain of Thoughts" axes from the 5-axis model — they measured overlapping constructs in practice (both ASR-transcript dependent, both driven by connector density and coherence).

| Feature | Role | Scoring Method |
|---|---|---|
| `lexical_mattr` | Primary | Z-score (normal, Shapiro p=0.288) |
| `discourse_connectors` | Secondary | Z-score (normal, Shapiro p=0.061) |
| `discourse_tier1` | Secondary | Percentile (non-normal) |
| `sbert_coherence` | Supporting | Calibrated cosine similarity |
| `lexical_rare` | Supporting | Z-score |

**Empirical Thresholds:**

| Band | MATTR | Connectors | Score |
|---|---|---|---|
| Excellent | > 0.794 | > 13 (or ≥ 5 Tier-1) | 4–5 |
| Good | 0.779–0.794 | 12–13 | 3–4 |
| Average | 0.751–0.779 | 11–12 | 2–3 |
| Below Average | 0.732–0.751 | 9–11 | 1–2 |
| Poor | < 0.732 | < 9 | 0–1 |

**SBERT coherence calibration (Indian English):**

```
Sim_calibrated = Sim_raw - 0.06
Adjusted Average threshold: Sim_calibrated ≥ 0.59 (from baseline 0.65)
```

Apply `−0.06` offset to cosine similarity targets — this corrects for embedding drift caused by ASR transcription errors in Indian English phonetics, not for the scoring threshold itself.

> **Recalibration note:** Connector count has very narrow std (2.4 on mean 11.4) — weak differentiator alone. Tier-1 connectors provide better signal. Once SBERT coherence is added to the dataset, re-weight this axis.

**ASR Loss Compensation for Connectors:** Scale down connector count target by 10% to account for missed connector words in ASR transcripts before ASR priming is confirmed accurate.

**Axis weight in composite:** 15%

---

#### Axis 4 — Narrative & Evidence

**What it measures:** Ability to communicate through concrete examples, structured narrative, and specific evidence. This axis **replaces and merges** the original "Storytelling" and "Usage of Examples" axes — they share the same evidence: concrete specificity.

| Feature | Role | Scoring Method |
|---|---|---|
| `ner_entity_density` | Primary (with Indian gazetteer) | Percentile |
| `metric_density` | Primary | Percentile |
| Narrative arc markers | Secondary | Rule-based detection |

**Narrative Arc Detection (text-based, accent-neutral):**

```python
ORIENTATION_MARKERS = ['when i', 'at the time', 'the situation was', 'we were']
COMPLICATION_MARKERS = ['however', 'but then', 'the challenge was', 'the problem']
ACTION_MARKERS = ['so i', 'what i did', 'my approach', 'i decided', 'we implemented']
RESOLUTION_MARKERS = ['as a result', 'ultimately', 'the outcome', 'we achieved', 'this led to']

def score_narrative_arc(transcript: str) -> float:
    """Score 0-1 based on presence of narrative structure stages."""
    t = transcript.lower()
    stages_present = sum([
        any(m in t for m in ORIENTATION_MARKERS),
        any(m in t for m in COMPLICATION_MARKERS),
        any(m in t for m in ACTION_MARKERS),
        any(m in t for m in RESOLUTION_MARKERS)
    ])
    return stages_present / 4.0
```

> **Important:** `pitch_std` is removed from this axis. Prosodic expressiveness is not a reliable measure of storytelling quality — it conflates delivery style with narrative skill, and is systematically biased against syllable-timed Indian English speech. `pitch_std` moves entirely to Axis 5 (Vocal Delivery).

**NER Calibration:**
- Apply 1.3× multiplier to entity density when Indian gazetteer entities are matched
- OR lower the target entity density threshold by 20% — not to lower the bar, but to correct for NER model blind spots on Indian proper nouns

**Axis weight in composite:** 15%

---

#### Axis 5 — Vocal Delivery

**What it measures:** Voice modulation and acoustic presence.

| Feature | Role | Scoring Method |
|---|---|---|
| `pitch_std` | Primary | Z-score (normal, Shapiro p=0.250) |
| `voiced_fraction` | Secondary | Percentile |

> **Drop `pitch_mean`:** Gender-dependent (women naturally higher absolute Hz). `pitch_std` (variation) is gender-neutral.

**Empirical Thresholds:**

| Band | Pitch Std (Hz) | Score |
|---|---|---|
| Excellent (Highly expressive) | > 59.1 | 4–5 |
| Good (Expressive) | 51.6–59.1 | 3–4 |
| Average | 47.3–51.6 | 2–3 |
| Below Average (Low variation) | 42.3–47.3 | 1–2 |
| Poor (Monotone) | < 42.3 | 0–1 |

**Indian English prosodic calibration:**

Indian English is syllable-timed (equal duration per syllable), unlike stress-timed British/American English. This produces naturally narrower pitch swings in formal speech, not monotone delivery.

```
PitchThreshold_en-IN = PitchThreshold_en-US × 0.825
Adjusted Average band: ≥ 41.7 Hz (from baseline 50.6 Hz)
```

Apply this correction at the feature extraction normalization stage, not as a separate scoring track.

**Axis weight in composite:** 10%

> **Redesign note:** Vocal Delivery is weighted lower than other axes because `pitch_std` from the empirical dataset shows potential pipeline artifact (`pitch_range` near-constant issue noted in analysis report). Investigate `parselmouth` ceiling issue before increasing this weight.

---

### 2.3 Composite Score Formula

```
communication_skills_score = (
    fluency_score         × 0.20 +
    intelligibility_score × 0.20 +
    lexical_structural    × 0.15 +
    narrative_evidence    × 0.15 +
    vocal_delivery        × 0.10
) × 20

# Note: weights sum to 0.80 intentionally.
# Remaining 0.20 reserved for grammar dimension (currently mocked — all = 0 in dataset).
# Once grammar scoring is implemented, add:
# + grammar_score × 0.20
# and remove the reserved multiplier adjustment.
```

> **Current grammar gap:** The empirical analysis dataset shows grammar check is mocked (all = 0). Grammar / language control is a meaningful communication dimension. Until this is implemented, scores are computed on 5 axes with adjusted weighting. Flag this gap clearly in any recruiter-facing score display.

---

### 2.4 Score Aggregation Logic

**Band-to-score mapping (0–5 per axis):**

| Band | Score | Percentile Equivalent |
|---|---|---|
| Poor | 0–1 | < p20 |
| Below Average | 1–2 | p20–p40 |
| Average | 2–3 | p40–p60 |
| Good | 3–4 | p60–p80 |
| Excellent | 4–5 | > p80 |

**Minimum floor rules:**
- A score of Poor (0–1) on Fluency or Intelligibility triggers a mandatory human review flag regardless of composite score.
- Composite score in borderline band (within ±5 points of pass/fail threshold) triggers human review flag.

**Role-based weight profiles:** The default weights above are for general roles. Adjust per role type:

| Role Type | Fluency | Intelligibility | Lexical & Structural | Narrative & Evidence | Vocal |
|---|---|---|---|---|---|
| Default | 20% | 20% | 15% | 15% | 10% |
| Client-facing / Sales | 20% | 20% | 10% | 20% | 10% |
| Backend / Technical | 15% | 20% | 20% | 15% | 5% |
| Leadership / Managerial | 15% | 15% | 20% | 20% | 10% |

---

### 2.5 Axis Confidence Scores

Each axis must output a confidence value alongside its score. Low confidence triggers downweighting or human review — not a garbage score in the aggregate.

```python
@dataclass
class AxisResult:
    score: float          # 0-5 band score
    confidence: float     # 0-1, how reliable is this axis measurement
    flags: list           # any edge case flags

def assess_axis_confidence(features: dict, axis: str) -> float:
    """
    Confidence is reduced when:
    - Audio duration < 10 min (too little signal)
    - ASR intel_confidence < 0.65 (poor transcription quality)
    - Feature value is extreme outlier (|z| > 3)
    """
    confidence = 1.0
    if features.get('duration_min', 999) < 10:
        confidence -= 0.30
    if features.get('intel_confidence', 1.0) < 0.65:
        confidence -= 0.25
    return max(0.1, confidence)
```

---

### 2.6 Human Review Trigger Protocol

| Condition | Trigger | Action |
|---|---|---|
| Any axis confidence < 0.5 | Auto-flag | Mark axis as "Low Signal" — exclude from composite |
| Composite score within ±5 pts of pass/fail threshold | Auto-flag | Route to human reviewer |
| Fluency or Intelligibility = Poor band | Auto-flag | Human review required before decision |
| > 3 features at \|z\| > 3 | Auto-flag | Outlier candidate — human review |
| `fluency_scripted_score` > 0 | Auto-flag | Possible scripted response — flag for review |
| `sentiment_hedge_rate` > p80 (> 0.303) | Score reduction | Reduce Lexical & Structural score — signals excessive uncertainty |
| `pitch_range` near-constant | Exclude | Known pipeline issue — exclude Vocal Delivery axis from composite |
| Duration < 10 min | Warning | Flag on profile — lexical scores are less reliable |

---

## Part 3: System B — Communication Style Engine

### 3.1 Purpose and Principles

System B describes **how** a candidate communicates — their natural style, tendencies, and communication personality. It is **non-evaluative**. There is no good or bad archetype. Output informs hiring managers about communication fit with a team or role, not about quality.

**The original Communication Style Evaluation Framework (v2.0) is preserved intact in this system.** The sections below note only the changes and additions made to that framework.

### 3.2 Changes to the Original Style Framework

**Change 1 — Replace TTR with MATTR (Feature 7)**

The original framework uses Type-Token Ratio (TTR) for vocabulary density. Replace with MATTR (Moving Average Type-Token Ratio) as Feature 7. MATTR is length-independent — it gives accurate vocabulary diversity scores regardless of response length, which varies significantly in interview contexts (range: 2.2–43.5 min in empirical dataset). TTR artificially deflates for longer responses.

```python
def calculate_mattr(transcript: str, window_size: int = 50) -> float:
    """MATTR: sliding window TTR. Length-independent vocabulary diversity."""
    words = word_tokenize(transcript.lower())
    words = [w for w in words if w.isalnum()]
    if len(words) < window_size:
        return len(set(words)) / len(words) if words else 0.0
    ttrs = []
    for i in range(len(words) - window_size + 1):
        window = words[i:i + window_size]
        ttrs.append(len(set(window)) / window_size)
    return float(np.mean(ttrs))
```

**Change 2 — Recalibrate WPM Archetype Ranges**

The original framework's WPM ranges for archetypes (Pragmatists: 160–180 WPM, Architects: 120–140 WPM) are assumed, not data-derived. The empirical dataset shows a median of 61 WPM — all archetype ranges need recalibration once archetype labels are applied to real interviews.

Interim recalibrated WPM ranges (derived from empirical dataset percentiles):

| Archetype | Original WPM Range | Recalibrated Range (v0) | Notes |
|---|---|---|---|
| Pragmatist | 160–180 | > 93 (p80+) | Top quartile in interview context |
| Storyteller | 140–160 | 67–93 (p60–p80) | Moderate pace with variation |
| Connector | 130–150 | 55–80 | Conversational pace |
| Architect | 120–140 | 50–70 (p40–p60) | Measured, deliberate |
| Analyst | 110–130 | < 67 (p40 and below) | Careful, precise |

> These ranges are indicative. Recalibrate fully once 50+ interviews are manually labeled with dominant archetype.

**Change 3 — Pitch Variation Normalization (Indian English)**

Feature 10 (Pitch Variation) must use normalized F0, not raw Hz values. Indian English is syllable-timed — narrower pitch swings in formal speech are a dialect characteristic, not low expressiveness.

Apply speaker-relative normalization:

```python
def normalized_pitch_std(audio_path: str) -> float:
    """
    Normalize F0 std dev relative to speaker's own mean F0.
    Returns coefficient of variation (CV) — dialect-neutral.
    """
    y, sr = librosa.load(audio_path)
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    voiced_f0 = f0[f0 > 0]
    if len(voiced_f0) < 10:
        return 0.0
    cv = np.std(voiced_f0) / np.mean(voiced_f0)  # coefficient of variation
    return float(cv)
```

This makes the expressiveness signal comparable across different speakers (male/female/different dialect) without demographic bias.

**Change 4 — Sentiment Exclusion Confirmed**

The original framework explicitly excluded sentiment detection (VADER or similar) for the following reasons, which also apply to System A:
- Accuracy across demographics: ~60–70%
- Higher false positive rate for non-native English speakers
- Interview language is neutral-professional by default — VADER trained on social media, not formal speech

`sentiment_compound` and `sentiment_assertive` are excluded from both systems. Assertiveness signal is captured more reliably by `discourse_tier1` connectors (declarative, forward-moving language) in System A.

### 3.3 Five Archetypes — Preserved

The five archetypes (Architect, Connector, Synthesizer, Analyst, Pragmatist) from the original framework are preserved exactly. Refer to Section 5 of the original `Communication_Style_Evaluation_Framework.md` for full archetype definitions, signal profiles, vocal profiles, and role mappings.

### 3.4 GMM Training Requirements

The GMM cannot be trained or deployed until the following is complete:

- [ ] Manually label 50+ interviews with dominant archetype (human annotators)
- [ ] Annotators should be blind to candidate demographics
- [ ] Use holdout validation on 20 interviews
- [ ] Recalibrate WPM and pitch ranges from labeled data
- [ ] Run bias audit (Phase 3 of original framework) before deployment

### 3.5 Bias Audit Framework — Preserved

The complete four-phase bias audit process from Section 8 of the original framework is preserved intact:

- Phase 1: Data Audit (`audit_training_data`)
- Phase 2: Feature Distribution Audit (`audit_feature_distributions`)
- Phase 3: Archetype Distribution Audit (`audit_archetype_distributions`)
- Phase 4: Hiring Impact Audit (`audit_hiring_impact`)

**Additional audit requirement (Indian English):**

Add `native_language` as a demographic variable in all four audit phases. Flag if any feature shows statistically significant differences (p < 0.05) between Indian English and other speaker groups after accent corrections are applied.

---

## Part 4: Combined Output Model

### 4.1 Full Candidate Profile Structure

```python
@dataclass
class CandidateCommunicationProfile:
    
    candidate_id: str
    interview_duration_min: float
    
    # System A: Skills Assessment
    skills: SkillsAssessment
    
    # System B: Style Profile
    style: StyleProfile
    
    # Meta
    flags: list[str]
    review_required: bool
    confidence_overall: float   # average axis confidence across System A

@dataclass
class SkillsAssessment:
    # Axis scores (0-5 each)
    fluency: AxisResult
    intelligibility: AxisResult
    lexical_structural: AxisResult
    narrative_evidence: AxisResult
    vocal_delivery: AxisResult
    
    # Composite
    composite_score: float       # 0-100
    composite_band: str          # Poor / Below Average / Average / Good / Excellent
    grammar_pending: bool        # True until grammar axis is implemented
    
    # Role-adjusted score (optional)
    role_adjusted_score: float   # 0-100, weighted per role profile

@dataclass
class StyleProfile:
    # 5 communication signals (0-100)
    systematic_thinking: float
    collaborative_orientation: float
    analytical_precision: float
    expressive_engagement: float
    action_orientation: float
    
    # Archetype blend (percentages, sum to 100)
    archetype_blend: dict        # {'architect': 50, 'connector': 20, ...}
    dominant_archetype: str
    
    # GMM readiness
    gmm_trained: bool            # False until 50+ labeled interviews exist
```

### 4.2 Pipeline Flow

```
Audio Input
    │
    ▼
Feature Extraction Layer (20 features, single pass)
    │   ├─ Whisper ASR (accent-primed)
    │   ├─ librosa audio analysis
    │   ├─ NLP pipeline (spaCy + Indian Gazetteer)
    │   ├─ Sentence-BERT (multilingual)
    │   └─ Whisper word timestamps
    │
    ├──────────────────────┬─────────────────────────┐
    ▼                      ▼                         ▼
System A                System B              Shared Flags
Skills Engine           Style Engine          & Confidence
    │                      │                         │
    ▼                      ▼                         ▼
5-Axis Scores       5-Signal Scores          Review Triggers
+ Composite         + Archetype Blend
    │                      │
    └──────────┬────────────┘
               ▼
    CandidateCommunicationProfile
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
Recruiter   Hiring    Candidate
  View      Manager   Feedback
            View       View
```

---

## Part 5: UI / Output Recommendations

### 5.1 Recruiter View — Skills Score

Show: Composite score, axis breakdown, flags, review triggers.

Do not show: Archetype blend (it's non-evaluative and will confuse pass/fail workflow).

```
┌─────────────────────────────────────────────────────┐
│  Priya Sharma — Communication Skills Assessment     │
│                                                     │
│  Overall Score: 74/100  ●  Band: Good               │
│  ⚠ Grammar axis pending (score reflects 5 axes)    │
│                                                     │
│  Fluency:          ████████░░  Good (3.8)           │
│  Intelligibility:  ███████░░░  Good (3.5)           │
│  Lexical/Structure:███████░░░  Good (3.4)           │
│  Narrative/Evidence:██████░░░  Average (2.9)        │
│  Vocal Delivery:   ████████░░  Good (3.7)           │
│                                                     │
│  ✓ No review flags triggered                        │
└─────────────────────────────────────────────────────┘
```

### 5.2 Hiring Manager View — Style Profile

Show: Archetype blend, signal spider chart (click to expand), role-fit indication.

Do not show: Raw axis scores (they are grading metrics, not role-fit descriptors).

```
┌─────────────────────────────────────────────────────┐
│  Priya Sharma — Communication Style                 │
│                                                     │
│  Primary: Synthesizer (42%)                         │
│  Secondary: Architect (28%) · Analyst (18%)         │
│  Tertiary: Connector (8%) · Pragmatist (4%)         │
│                                                     │
│  "Balanced, adaptable communicator — structured     │
│   thinking with collaborative instincts."           │
│                                                     │
│  [View Signal Breakdown ↓]                          │
│  Systematic Thinking:      78                       │
│  Collaborative Orientation: 55                      │
│  Analytical Precision:     68                       │
│  Expressive Engagement:    52                       │
│  Action Orientation:       61                       │
│                                                     │
│  Role fit: ✓ Good match for Tech Lead, PM roles     │
└─────────────────────────────────────────────────────┘
```

### 5.3 Candidate-Facing Feedback

Show: Axis-level qualitative feedback (not raw scores), style description (framed positively).

Do not show: Composite score, archetype percentages as numbers (reframe as description), any mention of accent scoring.

```
"Your responses showed strong logical organization and clear vocabulary.
Your narrative style demonstrated good use of specific examples. 
Consider using more concrete metrics and outcomes in your answers.

Your communication style shows systematic thinking combined with 
collaborative instincts — a style that works well in cross-functional roles."
```

---

## Part 6: Recalibration & Redesign Roadmap

### 6.1 Immediate — Before Any Production Use

| Item | Action | Priority |
|---|---|---|
| Ground truth validation | Get 30–40 human-annotated scores; validate feature→quality correlation per axis. Any axis with r < 0.5 against human judgment needs feature revision. | 🔴 Blocker |
| `pitch_range` pipeline bug | Investigate parselmouth ceiling issue — `pitch_range` near-constant in current dataset. Until fixed, exclude Vocal Delivery axis from composite and flag affected records. | 🔴 Blocker |
| Grammar axis | Implement grammar/language control scoring. Currently mocked (all = 0). Reserve 20% weight in composite until live. | 🔴 Blocker |
| SBERT coherence | Add `sbert_coherence` (F13) to feature extraction pipeline — currently absent from empirical dataset. | 🔴 Blocker |

### 6.2 Before First Enterprise Deployment

| Item | Action | Priority |
|---|---|---|
| Composite weight validation | Validate 20/20/15/15/10 weights against human ratings. Until validated, consider equal weights (16.7% each) as safer default. | 🟡 High |
| Bias audit (System A) | Run accent/gender bias check — compare score distributions across demographic groups for equivalent response quality. | 🟡 High |
| Archetype GMM training | Collect 50+ manually labeled interviews; train GMM; validate on 20 holdout. Do not deploy System B until this is complete. | 🟡 High |
| WPM recalibration | Recalibrate archetype WPM ranges using labeled data. Current ranges in style framework are not data-derived. | 🟡 High |
| Role weight profiles | Implement at least 3 role profiles (default / client-facing / technical) in System A composite. | 🟡 High |

### 6.3 After n=300 Recordings

| Item | Action | Priority |
|---|---|---|
| Full threshold recalibration | Recalibrate all percentile-based thresholds. Current n=86 gives ~17 data points per band — statistically thin. | 🟢 Planned |
| Discourse connector range | `discourse_connectors` has very narrow std (2.4) — weak differentiator. Reassess whether this feature adds signal or just noise at larger n. | 🟢 Planned |
| System B bias audit | Run all 4 phases of bias audit from original framework. Gate deployment on passing all phases. | 🟢 Planned |
| Indian English lexicon expansion | Expand SpaCy gazetteer to include Tier 2/3 cities, state universities, startup ecosystem vocabulary. | 🟢 Planned |

### 6.4 Threshold Versioning (Required from Day One)

All thresholds must be version-controlled in the codebase. Tag every threshold set with the n-count at which it was derived.

```python
THRESHOLD_VERSION = "v0-n86-june2026"
RECALIBRATION_TRIGGERS = {
    "n_threshold": 300,       # Recalibrate percentiles at n=300
    "n_full_recalibration": 500,
    "demographic_shift": True, # Recalibrate if new demographic group enters funnel
}
```

---

## Part 7: Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| n=86 recordings | Thresholds are indicative; band boundaries are unstable | Version-tag all thresholds; recalibrate at n=300 |
| No ground truth scores | Cannot validate feature→quality mapping | Human annotation sprint required (30–40 interviews) |
| Grammar dimension mocked | Composite score reflects 5 axes, not 6 | Reserve 20% weight; display caveat in UI |
| `pitch_range` pipeline artifact | Vocal Delivery axis unreliable for some records | Exclude affected records; fix parselmouth pipeline |
| GMM untrained | System B cannot generate archetype blends yet | Label 50 interviews; train GMM before System B goes live |
| Indian English calibration | Corrections applied at extraction layer — not yet validated against ground truth | Include accent as variable in bias audit |
| WPM distribution compressed | All candidates speak slower than native rate in interview context | Reframe WPM as "interview pacing" not "speech speed"; validate against quality ratings |

---

## Appendix A: Feature-to-System Mapping (Quick Reference)

| Feature | System A Axis | System B Signal | Notes |
|---|---|---|---|
| `fluency_wpm` | Fluency (primary) | Action Orientation (energy_level) | Same raw signal, different interpretation |
| `fluency_filler_rate` | Fluency (secondary, inverted) | Structural (prep/confidence, inverted) | Same raw signal |
| `fluency_pause_dur` | Fluency (penalty flag) | — | Skills only |
| `fluency_pause_freq` | Intelligibility (secondary) | — | Skills only |
| `intel_confidence` | Intelligibility (primary) | — | Skills only |
| `lexical_mattr` | Lexical & Structural (primary) | Vocabulary Precision (replaces TTR) | MATTR preferred over TTR |
| `lexical_rare` | Lexical & Structural (supporting) | — | Skills only |
| `discourse_connectors` | Lexical & Structural (secondary) | Logical Connector Density | Same signal |
| `discourse_tier1` | Lexical & Structural (secondary) | — | Skills only |
| `sbert_coherence` | Lexical & Structural (supporting) | — | Skills only; not yet in dataset |
| `pitch_std` | Vocal Delivery (primary) | Pitch Expressiveness | Same signal; normalized differently |
| `voiced_fraction` | Vocal Delivery (secondary) | — | Skills only |
| `ner_entity_density` | Narrative & Evidence (primary) | — | Skills only |
| `metric_density` | Narrative & Evidence (primary) | Results Orientation | Same signal |
| `collaborative_ratio` | — | Collaborative Orientation | Style only |
| `question_density` | — | Listener Engagement | Style only |
| `empathetic_markers` | — | Emotional Expressiveness | Style only |
| `vocal_confidence` | — | Vocal Presence | Style only |
| `speech_fluency_score` | — | Fluency (Style) | Style only |
| `stress_markers` | — | Emotional Stability | Style only |
| `avg_sentence_length` | — | Sentence Complexity | Style only |
| `sentiment_compound` | ❌ Excluded | ❌ Excluded | Removed from both systems |

---

## Appendix B: Implementation Checklist for AI Agent

### Phase 1: Shared Pipeline (Weeks 1–2)
- [ ] Set up Whisper ASR with Indian English accent priming
- [ ] Implement all 20 feature extraction functions
- [ ] Implement Indian NER gazetteer in SpaCy
- [ ] Implement MATTR (replace TTR)
- [ ] Implement multilingual SBERT for coherence scoring
- [ ] Implement narrative arc marker detection
- [ ] Implement homophone correction dictionary
- [ ] Implement normalized pitch CV (replace raw F0 std)
- [ ] Test pipeline on 10 sample interviews
- [ ] Investigate and fix `pitch_range` parselmouth ceiling bug

### Phase 2: System A — Skills Engine (Weeks 3–4)
- [ ] Implement 5-axis scoring with empirical thresholds (v0-n86)
- [ ] Implement Indian English instrument corrections at extraction layer
- [ ] Implement composite score formula (5 axes, grammar reserved)
- [ ] Implement axis confidence scores
- [ ] Implement human review trigger protocol
- [ ] Implement role-based weight profiles (3 profiles minimum)
- [ ] Tag all thresholds with `THRESHOLD_VERSION`
- [ ] Implement grammar axis stub (returns 0, flags as pending)

### Phase 3: Ground Truth Validation (Weeks 5–6)
- [ ] Human annotation sprint: 30–40 interviews scored per axis (1–5)
- [ ] Compute correlation between feature scores and human scores per axis
- [ ] Flag any axis with r < 0.5 for feature revision
- [ ] Validate composite weight assumptions; adjust if needed
- [ ] Run accent/demographic bias audit on System A

### Phase 4: System B — Style Engine (Weeks 7–8)
- [ ] Implement all 13 feature extractions (9 text + 4 vocal) per original framework
- [ ] Implement 5-signal aggregation (per original Section 3.2)
- [ ] Manually label 50 interviews with dominant archetype
- [ ] Train GMM on labeled data
- [ ] Validate GMM on holdout (20 interviews)
- [ ] Recalibrate WPM archetype ranges from labeled data

### Phase 5: Bias Audit (Weeks 9–10)
- [ ] Run all 4 phases of bias audit (per original Section 8)
- [ ] Add `native_language` as demographic variable
- [ ] **Gate: only proceed if bias audit passes all phases**

### Phase 6: UI Integration (Weeks 11–12)
- [ ] Build recruiter view (skills score + axis breakdown + flags)
- [ ] Build hiring manager view (archetype blend + signal spider chart)
- [ ] Build candidate feedback view (qualitative, no raw scores)
- [ ] Integration with Avya evaluation pipeline

### Phase 7: Pilot & Recalibration (Ongoing)
- [ ] Pilot with 1 role, 100 candidates
- [ ] Collect hiring manager feedback on style profile accuracy
- [ ] Recalibrate all thresholds at n=300
- [ ] Full recalibration at n=500
- [ ] Monthly bias audit (random sample of 50 candidates)

---

*Framework version: 1.0 | June 2026 | Zeko AI*  
*Prepared as implementation spec for AI agent pipeline development*  
*Next review: After Phase 3 (ground truth validation)*
