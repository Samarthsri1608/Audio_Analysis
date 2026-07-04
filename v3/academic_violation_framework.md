# Academic Violation Detection Framework — Final (v1: Track A + Track C)

## 1. Design philosophy

Single-candidate scoring, no population comparison required. Every feature is either **relative to the candidate's own interview** (Track A) or **structurally diagnostic on its own** (Track C) — nothing here depends on absolute cutoffs or other candidates' data, so it works from interview #1 with zero historical volume.

Output is a **suspicion score with evidence**, routed to human review. Never an autonomous flag or auto-action.

---

## 2. Track A — Self-relative deviation

**Baseline construction:** leave-one-out. For question *i*, baseline = median and MAD of each feature across all *other* answers in the same interview. Requires ≥4 answers total to be meaningful (below that, Track A is skipped and Track C carries the score alone — flag this as a known limitation, see Section 6).

**Features (all computed per-answer, then compared to per-candidate baseline):**

| Feature | Definition | Why relative, not absolute |
|---|---|---|
| `speech_rate_wpm` | words / duration | fluent-by-nature candidates shouldn't trip a fixed floor |
| `filler_word_ratio` | filler count / total words | some candidates naturally use almost none |
| `lexical_mattr` | moving-average TTR, window=50 | correlates with education/native fluency, not honesty |
| `discourse_organization` | z-scored composite of (connector count, tier-1 connector count) | collapses two redundant v1 features into one |
| `sbert_coherence` | cosine similarity of sentence embeddings within answer | some people are just more coherent speakers |
| `self_correction_rate` | (false starts + repairs) / 100 words | drop in self-correction is more diagnostic than absolute filler rate |

**Deviation formula** (robust z-score, resistant to outlier answers skewing the baseline):

```python
def robust_z(feature_value, baseline_values):
    median = np.median(baseline_values)
    mad = np.median(np.abs(baseline_values - median))
    if mad == 0:
        mad = 1e-6  # guard against constant baseline
    return 0.6745 * (feature_value - median) / mad
```

Applied per feature per question, using leave-one-out baseline for that specific question.

---

## 3. Track C — Mechanism-level naturalness (no baseline needed)

These detect the *artifact* of reading/assistance directly, computed within a single answer or across the interview's own internal variance — never against another candidate or a "normal" reference.

| Feature | Definition | What it catches |
|---|---|---|
| `response_latency_sec` | silence between question-end and first word | mechanically consistent latency across easy/hard questions regardless of difficulty |
| `latency_variance_across_questions` | std dev of latency across all Q's in this interview | should naturally be high (hard Qs → longer pause); abnormally low variance is the tell |
| `intra_answer_pace_variance` | coefficient of variation of local WPM within one answer | reading = evenly paced; live thought = uneven |
| `pause_regularity` | variance of inter-pause intervals within an answer | recitation produces mechanically even pause spacing |
| `pitch_variance_ratio` | std dev of F0 (pitch) within answer, normalized to candidate's own vocal range | flattened intonation vs. this candidate's own other answers |
| `cross_question_naturalness_flatness` | std dev of (speech_rate, coherence, MATTR) **across all questions in interview** | real answers fluctuate with question difficulty; recited/assisted answers stay abnormally flat across every question — this is your day-1-cheater catch |

`cross_question_naturalness_flatness` is the single most important feature in the whole framework — it's the only one that doesn't care whether cheating started at question 1 or question 8, because it measures internal consistency of the *whole interview*, not a before/after comparison.

```python
def naturalness_flatness(feature_series):
    # feature_series = e.g. list of sbert_coherence across all questions
    return np.std(feature_series) / (np.mean(feature_series) + 1e-6)
    # LOW value = abnormally flat = suspicious
```

---

## 4. Composite scoring

```python
def suspicion_score(track_a_devs: dict, track_c_devs: dict, weights: dict):
    a_component = sum(weights[f] * abs(z) for f, z in track_a_devs.items())
    c_component = sum(weights[f] * abs(z) for f, z in track_c_devs.items())
    return a_component + c_component
```

**Weights (initial, pre-calibration):**
- `response_latency_sec`, `latency_variance_across_questions`, `pause_regularity`, `cross_question_naturalness_flatness` → weight 1.5–2.0 (hardest to fake)
- `pitch_variance_ratio`, `intra_answer_pace_variance` → weight 1.0
- Track A lexical/discourse/coherence features → weight 0.5–0.75 (most confoundable with personality — same caution flagged for accent-adjacent scoring elsewhere in this pipeline)

**Flag logic — OR, not AND:**
```python
flag_for_review = (
    suspicion_score > COMPOSITE_THRESHOLD
    and (
        abs(track_c_devs['response_latency_sec']) > HARD_SIGNAL_THRESHOLD
        or abs(track_c_devs['cross_question_naturalness_flatness']) > HARD_SIGNAL_THRESHOLD
    )
)
```
Composite alone isn't enough — requires corroboration from at least one hard-to-fake Track C signal before surfacing to a human. Prevents lexical/discourse noise from single-handedly triggering review.

---

## 5. Implementation pipeline (Python)

```
audio_input
   │
   ▼
[1] feature_extraction.py
   - ASR transcript + word timestamps (existing pipeline)
   - prosody extraction (F0, pause timing) — new module
   - per-answer feature vector: {speech_rate, filler_ratio, mattr, 
     discourse_score, coherence, self_correction, latency, 
     pace_variance, pause_regularity, pitch_variance}
   │
   ▼
[2] interview_aggregator.py
   - collects all answer-level feature vectors for one interview
   - computes Track A leave-one-out baselines (per question)
   - computes Track C cross-question series (latency_variance, 
     naturalness_flatness) — needs full interview, runs after 
     last question is answered
   │
   ▼
[3] scoring_engine.py
   - robust_z() per Track A feature
   - naturalness_flatness() per Track C feature
   - composite suspicion_score per question
   - flag_for_review boolean per question
   │
   ▼
[4] evidence_payload.py
   - NOT a verdict — outputs per-question deviation profile:
     { question_id, suspicion_score, top_3_contributing_features,
       z_scores: {...}, flagged: bool }
   │
   ▼
proctor_dashboard (existing surface)
```

**Integration point:** Stage [2] and [3] can only run after the full interview is complete (Track C needs all questions for cross-question variance) — so this is a post-interview batch job, not real-time per-question scoring. Flag this clearly to whoever consumes it: proctors get a post-interview report, not a live alert.

---

## 6. Validation & testing methodology

**Phase 1 — Synthetic/red-team validation (do this first, before any real candidate data)**
- Record the same ~10 people answering the same question set twice: once normally, once reading a prepared script or answering with assistance
- Confirms the detector separates the two conditions using *relative* features, since absolute fluency will be similar in both conditions for a fluent person reading naturally
- This directly tests the exact failure mode of v1 (fluent-but-honest getting flagged) — a fluent person's honest answers should score low; the same person's scripted answers should score high

**Phase 2 — Historical replay**
- Run the pipeline against past interviews with no known ground truth, just to check score distribution isn't pathological (e.g., not flagging 40% of candidates, not flagging 0%)
- Manually review the top-scoring 10-15 answers — do the top contributing features make sense on listen-back?

**Phase 3 — Disparate impact check (mandatory before any production use)**
- Split flagged rate by candidate language background / accent group if that metadata exists
- If flag rate is meaningfully higher for non-native or ESL candidates, the framework isn't ready — this is the same exposure category already flagged for accent-based scoring elsewhere in the pipeline (Title VII/EEOC, EU AI Act high-risk), and here the consequence is an integrity accusation rather than a lower score, so the bar for clearing this check should be stricter, not looser

**Phase 4 — Threshold calibration**
- Gate `COMPOSITE_THRESHOLD` and `HARD_SIGNAL_THRESHOLD` on labeled data only — don't ship guessed numbers. Same gating discipline as the GMM archetype model (50+ labeled interviews before trusting output)
- Track precision at the top decile of suspicion_score specifically — this is a "better to under-flag than falsely accuse" system, so tune for precision over recall

**Phase 5 — Shadow mode**
- Run in production computing scores but not surfacing them to proctors for a period, compare against any independently-reported integrity concerns from that window, before turning on the review flag

---

## 7. Known outliers / edge cases

| Case | Problem | Handling |
|---|---|---|
| Interview with <4 answered questions | Track A baseline unreliable (leave-one-out on 2-3 points is noise) | Track A weight auto-set to 0; Track C carries score alone; flag payload notes `"low_answer_count_reduced_confidence": true` |
| Candidate cheats from question 1 (the case you raised) | Track A baseline itself is corrupted | Caught only by Track C's `cross_question_naturalness_flatness` — this is why Track C is weighted highest and required for the OR-gate |
| Technical/coding-round answers | Speech patterns differ fundamentally from conversational answers (long pauses to think through code are *normal*, not suspicious) | Segment scoring separately for coding-round vs. conversational questions; do not pool them into the same baseline or cross-question series |
| Genuinely nervous candidate (elevated filler words, disfluency) early in interview, settles down later | Legitimate settling-in effect could look like a "drop in disfluency = suspicious" pattern under Track A | Use first 1-2 questions as warm-up, excluded from baseline computation entirely |
| Audio quality issues (poor mic, background noise) inflating/deflating prosody features | False signal on pitch/pause features, not a violation | QC gate: if ASR confidence or audio SNR below threshold for a question, exclude that question's Track C prosody features from scoring, flag payload as `"low_audio_quality": true` |
| Candidate with genuinely flat affect / atypical speech patterns (some neurodivergent speech presentations) | Cross-question flatness feature could misfire on someone whose natural speaking style doesn't vary much regardless of question difficulty | This is the most serious latent equity risk in the whole framework — needs explicit coverage in Phase 3 validation, not just language background. No fully reliable pipeline-level fix; recommend this be flagged prominently to proctors as "unusually low variance can have several explanations" language in the evidence payload, never as a standalone violation flag |
| Single outlier answer with extreme values (e.g., ASR misfire produces garbage transcript) | Corrupts the leave-one-out baseline for every other question | Winsorize baseline inputs (clip to 5th-95th percentile of that candidate's own values) before computing median/MAD |

---

## 8. Rollout gate

Do not enable `flag_for_review` in the live proctor dashboard until Phases 1–4 above are complete and Phase 3 (disparate impact) shows no meaningful skew. Until then, run in evidence-logging-only mode — score computed and stored, not shown.