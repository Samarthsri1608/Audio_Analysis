# Bias Audit

This is a structural bias audit of the `v3/` pipeline, not a statistical fairness benchmark.
I checked the code paths for direct use of protected attributes and for obvious proxy risks.
I did **not** run a full disparate-impact test because the repository does not contain demographic labels or a fairness evaluation dataset.

## What I Could Verify

### No direct protected-attribute features in `v3/`

I did not find any explicit use of:
- caste
- creed
- gender
- race
- religion
- disability

in the `v3/` scoring or feature-extraction code.

Relevant code paths inspected:
- transcription and ASR setup: `v3/pipeline/transcriber.py`
- text feature extraction: `v3/pipeline/text_features.py`
- vocal feature extraction: `v3/pipeline/vocal_features.py`
- System A scoring: `v3/pipeline/skills_scorer.py`
- System B normalization and archetype logic: `v3/pipeline/normalizer.py`, `v3/pipeline/archetype.py`

## Main Bias Risks

### 1. Accent and dialect bias

The pipeline is heavily tuned for Indian English:
- `v3/pipeline/transcriber.py` boosts Indian and backend/HR-specific vocabulary.
- `v3/pipeline/text_features.py` applies Indian-English homophone corrections.
- `v3/pipeline/text_features.py` adds an Indian-specific gazetteer for named entities.
- `v3/pipeline/vocal_features.py` uses dialect-neutral pitch CV, which is good, but still evaluates speech delivery acoustically.
- `v3/pipeline/skills_scorer.py` uses thresholds calibrated on n=86 interviews and assumes those distributions are broadly valid.

This means:
- candidates speaking Indian English may be helped relative to a generic English ASR baseline
- candidates with other accents or code-mixed speech may be under-measured if the calibration set is not representative
- people whose speech patterns differ from the calibration population may be scored differently even if their communication quality is similar

### 2. Fluency and speech-impairment proxy bias

The pipeline rewards or penalizes:
- filler-word rate
- pause duration
- speech rate
- vocal steadiness
- speech fluency
- semantic coherence of the transcript

These are legitimate communication features, but they are also proxy risks for:
- stuttering
- speech disorders
- anxiety
- neurodivergent speaking styles
- limited interview time
- non-native English speakers

This is the biggest disability-adjacent risk in the current design.

### 3. Socioeconomic and education proxy bias

The pipeline rewards:
- rare-word usage
- named entities
- metric density
- structured narrative
- specific company/institute references

These signals can correlate with:
- elite educational access
- certain job histories
- interview coaching
- socio-economic background

So even if the system does not use caste/gender/race directly, it can still correlate with social advantage.

### 4. Short-response penalty

`v3/pipeline/skills_scorer.py` reduces confidence for short transcripts and short interview duration. That is reasonable from a measurement perspective, but it can also disadvantage:
- candidates given less speaking time
- concise communicators
- candidates with speech limitations
- candidates whose answers were interrupted

### 5. Recruiter misuse risk

The final response is qualitative now, which helps.
However, the output still contains structured evidence signals and raw features, so a recruiter can still use it as an implicit ranking surrogate unless the UI and policy explicitly prevent that.

### 6. Personality / archetype outputs can encode workplace norms

If the personality endpoint is used, the archetype and role-fit machinery in `v3/pipeline/archetype.py` can encode assumptions about what “good” communication looks like for a given role.
That is not a direct protected-class bias, but it can still disadvantage candidates whose communication style is culturally different from the calibration baseline.

## What Is Safer About the Current Version

- There is no explicit caste/gender/race/creed/disability field in the model.
- The final communication response is descriptive rather than numeric.
- Vocal variation uses CV-normalized pitch rather than raw Hz, which reduces one accent-related bias source.
- Indian-English ASR corrections reduce known transcription loss for the target population.

## What Is Still Not Safe Enough

The pipeline is **not** bias-free just because it omits protected attributes.
The current design still uses many proxy features that can create unequal outcomes across:
- dialects and accents
- non-native speakers
- people with speech impairments
- neurodivergent speakers
- candidates with short answers or interrupted interviews

## What Should Change

### 1. Add empirical fairness testing

Create a labeled evaluation set with coverage across:
- gender
- caste
- creed / religion
- race / ethnicity
- dialect / accent
- disability / speech impairment
- native vs non-native speakers

Measure:
- score distribution shifts
- false positive / false negative differences
- calibration error by group
- disagreement rates between human raters and the pipeline

### 2. Add a low-bias operating mode

For accessibility-sensitive use cases, consider a mode that:
- suppresses vocal features
- reduces filler/pause penalties
- uses transcript content only
- returns “insufficient speech evidence” instead of low-confidence judgments

### 3. Separate dialect adaptation from evaluation

If the tool is used outside the current Indian-English target population, do not reuse the same thresholds blindly.
Calibrate per locale or per deployment population.

### 4. Limit recruiter exposure

Keep the tool assistive only:
- no auto-ranking
- no auto-reject
- no hidden score displayed as an objective truth
- show evidence notes and uncertainty
- force human sign-off

### 5. Redact sensitive self-disclosures

The transcript can contain personal details that are not needed for the communication summary.
Do not surface caste, religion, gender, or other sensitive self-disclosures in the recruiter UI unless there is a legitimate reason and policy basis.

## Alternatives

If you want a lower-bias design, the safer alternatives are:
- transcript-only qualitative notes
- recruiter rubric with manual scoring
- post-interview summary that avoids vocal and fluency penalties
- accessibility-aware mode for speech-impaired candidates
- locale-specific calibration with documented fairness checks

## Bottom Line

I did **not** find direct caste/creed/gender/race/disability logic in the code.
I **did** find multiple strong proxy channels that can create disparate impact, especially for accent, dialect, speech patterns, and disability-related communication differences.

So the correct conclusion is:
- **not explicitly discriminatory by code**
- **not proven fair**
- **needs empirical bias testing before broad recruiter use**

