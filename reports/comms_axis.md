# Proposal: Zeko AI Version 2 Communication Fingerprint Model

This document outlines the proposed communication profiling model for Version 2 of the Zeko AI Assessment Engine. It details the transition from traditional, high-risk "personality grading" to a legally compliant, shape-matching **Communication Fingerprint** (inspired by the GitHub Activity Overview Radar Chart).

---

## 1. Executive Summary: The Visual Paradigm Shift

Hiring managers need a fast, intuitive way to understand a candidate's communication style. However, assigning vertical grades (e.g., *Empathy: 1.5/5.0* or *Confidence: 2.0/5.0*) based on voice patterns is statistically unreliable, prone to gender and cultural bias, and legally high-risk (violating the EU AI Act's ban on emotion recognition and triggering expensive bias audits under NYC LL144).

### The Solution: The GitHub-Style Activity Radar Chart
We propose mapping the candidate's speech dynamics onto a multi-axis **radar (spider) chart**.
*   **No Grades, Just Shape:** There is no "good" or "bad" shape. Each profile represents a different, valid communication style (e.g. *The Structured Planner* vs. *The Dynamic Presenter*).
*   **Role Match Overlay:** The candidate's shape is overlaid against a **Job Template Baseline Shape** (e.g., Software Engineer, Sales, Support) defined by the recruiter. The system calculates a **Role Compatibility Match %** based on the overlap, giving hiring managers a clear screening metric while remaining legally safe.

```
                      1. LOGICAL COHESION
                             100%
                             / \
                            /   \
  6. LEXICAL PRECISION     /  *  \     2. DELIVERY FLUENCY
                          /  ***  \
                         +----+----+
                          \  * *  /
   5. COLLABORATIVE TONE   \  *  /     3. PRONUNCIATION CLARITY
                            \ * /
                             \ /
                            100%
                     4. VOCAL DYNAMISM
```

---

## 2. Core Proposal: The 6-Axis Model

This configuration maps Zeko AI's complete acoustic (vocal) and linguistic (transcript) feature pipeline onto a balanced hexagonal shape:

| Axis | What it Measures | Underlying Technical Signals |
| :--- | :--- | :--- |
| **1. Logical Cohesion** | How structured and reasoned the candidate's arguments are. | • Density of logical transition connectors (*"consequently"*, *"however"*).<br>• Average sentence lengths. |
| **2. Delivery Fluency** | Pacing and smoothness of delivery. | • Words Per Minute (`wpm`).<br>• Filler word rate (*"um"*, *"uh"*, *"like"*). |
| **3. Pronunciation Clarity** | Phonetic clarity and articulation stability. | • Whisper ASR word-level confidence (`mean_confidence`).<br>• Percentage of high-confidence words ($\ge 0.75$). |
| **4. Vocal Dynamism** | Vocal presence, energy, and voice modulation. | • Pitch standard deviation (`pitch_std_hz`) from Praat.<br>• Volume/Intensity standard deviation (`intensity_std_db`). |
| **5. Collaborative Tone** | The warmth and cooperative nature of their phrasing style. | • Ratio of team pronouns (*"we"*, *"our"*, *"us"*) to singular pronouns.<br>• VADER positive sentiment compound score.<br>• Spectral centroid (average voice pitch depth/warmth). |
| **6. Lexical Precision** | Vocabulary variety and domain-specific vocabulary density. | • Moving Average Type-Token Ratio (`mattr`) over a sliding 50-word window.<br>• Rare content-word ratio (via the `wordfreq` database). |

---

## 3. Alternative Axis Configurations

To provide options for the product roadmap, here are three alternative configurations for Zeko AI's dashboard:

### Option A: The 4-Axis Core Diamond Model
*Focuses strictly on the four fundamental communication styles. Extremely clean and fast to read.*
*   **Axes:**
    1.  **Structure (Planner):** Density of logical transition connectors.
    2.  **Expressiveness (Presenter):** Pitch variation and speaking tempo (WPM).
    3.  **Assertiveness (Driver):** Low hedging rate and high assertive phrasing.
    4.  **Collaboration (Coordinator):** Positive sentiment compound and team-oriented pronouns.
*   **Best for:** General candidate behavior profiling where detailed language mechanics (pronunciation, grammar) are less important than style.

### Option B: The 5-Axis IELTS-Aligned Model
*Maps directly to global English proficiency standards (IELTS/CEFR), making it highly familiar to international recruiters.*
*   **Axes:**
    1.  **Fluency:** Pause ratios, filler rates, and tempo.
    2.  **Lexical Resource:** Vocabulary variety (MATTR) and sophisticated words.
    3.  **Grammatical Range:** Grammar density and sentence structures.
    4.  **Pronunciation:** ASR phoneme and word confidence.
    5.  **Coherence (Discourse):** Structural transitions and topic development.
*   **Best for:** High-stakes English language screening (e.g. offshore BPO/call center hiring).

### Option C: The 8-Axis Full Spectrum Model
*The most comprehensive possible model, separating delivery mechanics completely from cognitive/linguistic depth.*
*   **Axes:**
    1.  **Pacing (Fluency):** WPM and voiced fraction.
    2.  **Vocal Variety:** Pitch standard deviation.
    3.  **Vocal Presence:** Intensity stability and volume dynamics.
    4.  **Clarity:** Whisper ASR word-level confidence.
    5.  **Precision:** Vocabulary diversity (MATTR).
    6.  **Logic:** Discourse connector density.
    7.  **Assertiveness:** Assertive count vs. hedging rate.
    8.  **Warmth:** Sentiment positivity and collaborative pronouns.
*   **Best for:** High-volume executive search where recruiters want a highly detailed "Communication Fingerprint."

---

## 4. Legal Compliance Justification

By presenting this radar chart model, Zeko AI resolves the major legal and compliance issues of the AI recruitment market:

1.  **Bypasses the EU AI Act Emotion Recognition Ban:** The Act prohibits using AI to infer psychological/emotional states (such as "empathy" or "stress") in the workplace. By measuring **Collaborative Tone** (which is a linguistic behavior of using team pronouns and warm spectral centroid) rather than "Empathy," we evaluate **objective skill, not personal psychology**.
2.  **Bypasses NYC LL144 Bias Audits:** Because this is a descriptive profile (describing their style shape) rather than a vertical scoring metric (declaring someone "Good" or "Bad"), it does not act as an automated gatekeeper. The final screening is always left to a human manager utilizing the Role Match Overlay.
3.  **Prevents Neurodiversity and Accent Discrimination:** A neurodiverse candidate or non-native speaker who speaks slowly (low Fluency) but has deep technical clarity (high Lexical and Logical Cohesion) is not labeled as "Poor." They are labeled as **The Architect**, which is a highly positive and desired style for technical roles.
