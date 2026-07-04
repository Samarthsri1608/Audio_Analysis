# Compliance Review

This note is a practical engineering review, not legal advice. It is based on the current code in `v3/` and the intended use case: recruiter-facing interview analysis with a human recruiter making the final decision.

## Scope

The pipeline:
- fetches interview audio
- transcribes speech
- extracts raw features
- produces a qualitative analysis payload for a recruiter

Relevant laws and references:
- EU AI Act: https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- GDPR: https://eur-lex.europa.eu/eli/reg/2016/679/oj

Key provisions to review:
- EU AI Act Annex III, employment, workers and access to self-employment
- EU AI Act Article 9, risk management system
- EU AI Act Article 14, human oversight
- GDPR Article 22, automated individual decision-making, including profiling

## What Is Relatively Aligned

- The pipeline is not currently set up to auto-rank or auto-reject candidates.
- The final output is qualitative, not numeric, and the recruiter is intended to make the final decision.
- The new response shape is descriptive and avoids direct scoring or ranking language.
- The code keeps raw feature extraction separate from the final analysis payload, which is a good base for auditability.

## What Is Still Risky Or Not Yet Compliant Enough

### 1. Recruitment use can still be high-risk under the EU AI Act

If this system is used to analyze candidates for hiring, it likely falls into the EU AI Act high-risk category for employment-related AI systems under Annex III. The fact that a human recruiter makes the final decision does not automatically remove it from scope.

### 2. Automated decision safeguards may still apply

If the output materially influences recruitment outcomes, GDPR Article 22 may become relevant. Even when the final decision is human-made, the system still needs clear safeguards if it is used in a significant employment context.

### 3. The code does not yet show compliance governance

The repository does not currently show:
- a documented risk-management process
- a human oversight workflow
- a retention/deletion policy
- a candidate notice/privacy flow
- logging and traceability for analysis versions
- bias/fairness testing or monitoring
- a formal appeal or challenge path for the candidate

### 4. External processors are involved

The pipeline depends on external services for transcription and possibly text generation. That means processor agreements, transfer controls, and data-processing disclosures matter.

## What Should Be Changed

### Product behavior

- Keep the output as recruiter-assistive only.
- Do not present the result as a score, rank, or hidden numerical surrogate.
- Do not auto-reject, auto-shortlist, or auto-prioritize candidates based solely on the pipeline.
- Show evidence-backed observations rather than personality labels or archetypes.

### UI and workflow

- Add a clear disclaimer that the output is an assistive summary.
- Require a recruiter confirmation step before any action is taken.
- Show evidence notes alongside each observation.
- Allow the recruiter to override or ignore the analysis.
- Provide a way to record why the recruiter accepted or rejected the system’s suggestion.

### Data governance

- Add retention rules for audio, transcripts, raw features, and final reports.
- Add deletion controls for candidate data.
- Log model/version metadata for each analysis.
- Document all external processors and data transfers.
- Minimize what is stored in the final response if the downstream user does not need raw features.

### Risk controls

- Create a human oversight policy.
- Run bias and drift checks on outputs across candidate groups.
- Test whether the descriptive fields correlate with protected characteristics.
- Review whether any field functions as a proxy for protected traits.

## Practical Compliance Position

### Safer use case

The safer framing is:
- recruiter-assistive interview notes
- summary of communication behavior
- no automated decision
- no score-based ranking
- human reviews the output before acting on it

### Riskier use case

The risk increases sharply if:
- the output is used to rank candidates
- the output is used to filter candidates out
- the output is shown as an objective or quantitative measure
- the output is used without notice, retention rules, or oversight

## Alternatives By Legal Constraint

### If you want to stay inside a lower-risk mode

- Use the pipeline only for note generation after the interview.
- Do not use it as a selection filter.
- Keep the recruiter as the only decision-maker.

### If you need stronger EU compliance posture

- Treat the system as a high-risk recruitment-support tool.
- Implement risk management, documentation, human oversight, logging, and post-deployment monitoring.
- Add candidate-facing transparency and data-rights handling.

### If you cannot support those controls

- Restrict the tool to internal summarization only.
- Do not let it influence hiring decisions.
- Do not expose it as an evaluation system to recruiters.

## Code Areas That Matter

- Feature extraction and caching: `v3/main.py`
- Final qualitative response schema: `v3/models.py`
- Audio transcription: `v3/pipeline/transcriber.py`
- Text and vocal feature extraction: `v3/pipeline/text_features.py`, `v3/pipeline/vocal_features.py`
- System A scoring engine: `v3/pipeline/skills_scorer.py`
- System B style engine: `v3/pipeline/normalizer.py`, `v3/pipeline/archetype.py`

## Bottom Line

As a recruiter-assistive summary tool, this is much safer than an automated ranking or rejection system.

As a recruitment AI system, it is not automatically compliant yet. To get closer, you need governance, transparency, retention controls, human oversight, and bias testing in addition to the current code changes.
