# Scientific and Technical Report: Accent-Calibrated Communication Assessment Model

This report details the theoretical foundations, academic validation, and accent-calibration strategies for the 5-axis communication profiling framework designed for the **Zeko AI Assessment Engine**. It specifically addresses algorithmic bias related to the **Indian English (en-IN)** accent and details the required refactoring, rescaling, and pipeline adjustments.

---

## The 5-Axis Communication Model at a Glance

```mermaid
radar
    title Accent-Calibrated Communication Model
    axes
        Pronunciation: 85
        Logical Structure: 78
        Chain of Thoughts: 70
        Storytelling: 68
        Usage of Examples: 82
```

---

## 1. Pronunciation (Speech Intelligibility)

### Conceptual Definition
Evaluates the acoustic clarity of the speaker's phonetics, vowel/consonant articulation stability, and overall vocal intelligibility. Rather than grading proximity to standard native accents (US/UK), it measures whether the speaker's phonemes are clearly distinguishable.

### Academic Citations
*   **Goodness of Pronunciation (GOP):** Witt, S. M., & Young, S. J. (2000). *Phone-level pronunciation scoring and assessment for an interactive English language learning system.* Speech Communication, 30(2-3), 95-108.
*   **ASR-Based Proficiency Scoring:** *“One Whisper to Grade Them All: A Multimodal Manifold Learning Approach to Automated Speaking Assessment”* (Interspeech, 2023).

### The Indian Accent Challenge
Standard Whisper ASR models are trained heavily on native US/UK English corpora. When processing Indian speakers, the ASR confidence scores drop due to non-native phonetic features:
1.  **Retroflexion of Alveolar Consonants:** Pronouncing `/d/` and `/t/` with the tongue curled back (retroflex) causes acoustic mismatch in standard phonetic recognizers.
2.  **Monophthongization:** Pronouncing diphthongs like `/eɪ/` (in *gate*) and `/oʊ/` (in *goat*) as pure monophthongs `/e:/` and `/o:/`.
3.  **Lack of Aspiration:** Voiceless stops (`/p/`, `/t/`, `/k/`) are typically not aspirated in Indian English, which acoustic models misclassify as voiced stops (`/b/`, `/d/`, `/g/`).

### Calibration & Refactoring Strategy
To prevent artificial score deflation, the confidence thresholds must be shifted, or Whisper must be primed using phonetic context prompting.

> [!TIP]
> **Whisper Accent Priming:** Prime the decoding process by passing an explicit accent signature in the `initial_prompt`.

```python
# Refactored ASR call with accent prompting
result = whisper_model.transcribe(
    audio_path,
    initial_prompt="The speaker has an Indian accent. Transcription of technical interview response in Indian English.",
    temperature=0.0
)
```

#### Rescaling Factor
*   **Confidence Boost Offset:** Apply a linear shift of **$+0.05$ to $+0.08$** to the Whisper token confidence level $C_{mean}$ before scoring.
*   **Threshold Rescaling:** Reduce the band thresholds in `evaluation.py` by **7.5%**:
    $$\text{Threshold}_{en-IN} = \text{Threshold}_{en-US} \times 0.925$$
    *   *Example:* Downward-scale the *Excellent* threshold from `0.799` to `0.739`.

---

## 2. Logical Structure of Answer (Macro-structure)

### Conceptual Definition
Measures structural organization—evaluating whether the candidate organizes their thoughts linearly (Introduction $\rightarrow$ Body $\rightarrow$ Conclusion) or uses structural templates like the STAR framework (Situation, Task, Action, Result).

### Academic Citations
*   **Discourse Structure Evaluation:** Miltsakaki, E., & Kukich, K. (2004). *Evaluation of text coherence for document organization.* LREC.
*   **Text Readability and Structure:** Barzilay, R., & Lapata, M. (2008). *Modeling local coherence: An entity-based approach.* Computational Linguistics, 34(1), 1-34.

### The Indian Accent Challenge
While logical structure is a cognitive metric rather than an acoustic one, accent mismatch causes the ASR to mistranscribe key transitions (e.g., transcribing *"hence"* as *"hens"*, *"moreover"* as *"more over"*, or *"consequently"* as *"consequent league"*). This reduces the computed connector density.

### Calibration & Refactoring Strategy
1.  **Homophone Mapping Dictionary:** Implement a pre-processing phonetic spellchecker on the output transcript for common discourse connectors.
2.  **Indian English Dialect Lexicon:** Add common Indian formal transitions to the discourse whitelist (e.g., *"as such"*, *"itself"* used for emphasis, *"doubt"* used as a synonym for *"question"*).

#### Rescaling Factor
*   **ASR Loss Compensation:** Scale down the target unique connector count threshold by **10%** to account for missed words in noisy ASR transcripts.
*   *Example:* If the threshold for *Excellent* is $\ge 13$ connectors, adjust to $\ge 11.7$ (rounded to 12) for Indian speakers.

---

## 3. Chain of Thoughts — Placement of Thoughts (Micro-coherence)

### Conceptual Definition
Evaluates local, sentence-to-sentence logical progression. It measures whether statement $B$ flows naturally from statement $A$, checking for logical coherence and the avoidance of erratic topic shifts.

### Academic Citations
*   **LSA Coherence Measurement:** Foltz, P. W., Kintsch, W., & Landauer, T. K. (1998). *The measurement of textual coherence with Latent Semantic Analysis.* Discourse Processes, 25(2-3), 285-307.
*   **Sentence Embeddings for Coherence:** Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP.

### The Indian Accent Challenge
If the ASR introduces transcription errors at the word level, the vector representation (embedding) of the sentence changes. The cosine similarity between adjacent sentence vectors drops, creating the false impression of an incoherent chain of thought.

### Calibration & Refactoring Strategy
*   **Multilingual Embeddings:** Replace standard English-only sentence transformers with models trained on Indian English and code-mixed corpora.
*   **Model Recommendation:** Use `distiluse-base-multilingual-cased-v1` or `paraphrase-multilingual-MiniLM-L12-v2`. These models map spelling variations and ASR phonetic errors to the same semantic space, preserving similarity scores.

#### Rescaling Factor
*   **Cosine Similarity Floor Compression:** Subtract **$0.06$** from the target cosine similarity boundaries:
    $$\text{Sim}_{calibrated} = \text{Sim}_{raw} - 0.06$$
    *   *Example:* Adjust the target average coherence boundary from `0.65` down to `0.59`.

---

## 4. Storytelling Skills (Narrative Engagement)

### Conceptual Definition
Evaluates the speaker's capacity to recount events using a narrative arc (Orientation $\rightarrow$ Complication $\rightarrow$ Action $\rightarrow$ Resolution) combined with engaging vocal delivery.

### Academic Citations
*   **Narrative Analysis:** Labov, W., & Waletzky, J. (1967). *Narrative analysis: Oral versions of personal experience.* Journal of Narrative and Life History.
*   **Speech Prosody & Assessment:** Sassenhagen, J., et al. *Evaluating Spontaneous Narrative Storytelling in Speaking Assessments.* ISCA.

### The Indian Accent Challenge
Indian English is **syllable-timed** (equal duration given to each syllable), unlike British/American English which is **stress-timed** (variable duration based on syllable stress). Additionally, Indian English speakers naturally utilize narrower pitch swings during formal speech, which Western-tuned prosodic features misclassify as flat or monotone (deflating the `pitch_std_hz` score).

### Calibration & Refactoring Strategy
*   **Prosodic Normalization:** Rescale the raw standard deviation of fundamental frequency (F0).
*   **Temporal Connector Adjustments:** Account for localized pacing markers (e.g., using *"firstly"*, *"then"*, *"afterward"* in different rhythmic sequences).

#### Rescaling Factor
*   **Pitch Standard Deviation Compression:** Reduce the target pitch variation thresholds by **15% to 20%**:
    $$\text{PitchThreshold}_{en-IN} = \text{PitchThreshold}_{en-US} \times 0.825$$
    *   *Example:* Reduce the median pitch variation target from `50.6 Hz` to `41.7 Hz` to align with syllable-timed speech delivery.

---

## 5. Usage of Examples (Concreteness & Evidence)

### Conceptual Definition
Measures whether the candidate backs up abstract, generalized claims with concrete, specific examples (citing numbers, metrics, tools, databases, or events).

### Academic Citations
*   **Concreteness Index:** Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014). *Concreteness ratings for 40 thousand generally known English word lemmas.* Behavior Research Methods, 46(3), 904-911.
*   **Named Entities in Grading:** Yannakoudakis, H., Briscoe, T., & Medlock, B. (2011). *A new dataset and method for automatically grading ESOL texts.* ACL.

### The Indian Accent Challenge
Standard Named Entity Recognition (NER) models (such as SpaCy's default English models) fail to classify Indian proper nouns, universities, geographic regions, and local companies (e.g., *"IIT Madras"*, *"Bengaluru"*, *"TCS"*, *"Infosys"*) as entities, leading to a deflated Example/Evidence score.

### Calibration & Refactoring Strategy
*   **Custom Indian Gazetteer:** Integrate a gazetteer (custom entity lookup list) containing Indian technological institutes, cities, and businesses into the SpaCy pipeline.

```python
import spacy
from spacy.pipeline import EntityRuler

nlp = spacy.load("en_core_web_sm")
ruler = nlp.add_pipe("entity_ruler", before="ner")

# Inject Indian technological entities and companies
patterns = [
    {"label": "ORG", "pattern": "IIT"},
    {"label": "ORG", "pattern": "Infosys"},
    {"label": "ORG", "pattern": "TCS"},
    {"label": "GPE", "pattern": "Bengaluru"},
    {"label": "GPE", "pattern": "Pune"}
]
ruler.add_patterns(patterns)
```

#### Rescaling Factor
*   **NER Target Density Calibration:** Apply a **$1.3\times$** scaling multiplier to the entity density calculation when localized entities are matched, or lower the overall target entity density threshold by **20%** to avoid false negatives.

---

## Summary of Accent Calibration Settings

| Evaluation Axis | Primary Metric | Western Baseline Threshold | Indian Accent (en-IN) Target | Calibration Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **1. Pronunciation** | Whisper Token Confidence ($C_{mean}$) | $\ge 0.799$ (Excellent) | $\ge 0.739$ | $+0.06$ Linear Offset / Accent Prompting |
| **2. Logical Structure** | Discourse Connectors | $\ge 13$ connectors | $\ge 12$ connectors | Homophone mapping + Indian Register dictionary |
| **3. Chain of Thoughts** | Semantic similarity of sentences | $S_{cosine} \ge 0.65$ | $S_{cosine} \ge 0.59$ | Multilingual Sentence-BERT embeddings |
| **4. Storytelling** | Pitch variance ($F0_{std}$) | $\ge 50.6\text{ Hz}$ | $\ge 41.7\text{ Hz}$ | Prosodic scale compression ($17.5\%$ drop) |
| **5. Usage of Examples** | NER Entity Density | standard NER counts | custom Indian entities | Indian tech & business gazetteer expansion |
