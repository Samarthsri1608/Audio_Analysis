import logging
import os

try:
    import joblib
    _has_joblib = True
except ImportError:
    _has_joblib = False

from app.services.scoring_engine.feature_assembler import flatten_features
from app.services.scoring_engine.evaluation import (
    compute_fluency_score,
    compute_intelligibility_score,
    compute_language_control_score,
    compute_lexical_resource_score,
    compute_discourse_score,
    compute_voice_modulation_score,
    compute_sentiment_score,
    compute_overall_score,
)

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

DIMENSION_WEIGHTS = {
    "fluency":          0.22,
    "intelligibility":  0.22,
    "language_control": 0.20,
    "lexical_resource": 0.18,
    "discourse":        0.12,
    "sentiment":        0.03,
    "voice_modulation": 0.03,
}


def _compute_overall(scores: dict) -> float:
    return compute_overall_score(
        fluency=scores.get("fluency", 0),
        intelligibility=scores.get("intelligibility", 0),
        language_control=scores.get("language_control", 0),
        lexical=scores.get("lexical_resource", 0),
        discourse=scores.get("discourse", 0),
        voice_modulation=scores.get("voice_modulation", 0),
        sentiment=scores.get("sentiment", 0),
    )


def heuristic_fallback(features: dict) -> dict:
    scores = {}
    raw_text = features.get("raw_text", "")
    total_words = max(len(raw_text.split()), 1)

    # 1. Fluency
    fluency_feats  = features.get("fluency", {})
    wpm            = float(fluency_feats.get("wpm", 0.0))
    pause_freq     = float(fluency_feats.get("pause_frequency", 0.0))
    filler_rate    = float(fluency_feats.get("filler_rate", 0.0))
    duration_s     = features.get("duration_ms", 0) / 1000.0
    scores["fluency"] = compute_fluency_score(wpm, pause_freq, duration_s, filler_rate)

    # 2. Intelligibility
    intel_feats  = features.get("intelligibility", {})
    pron_score   = float(intel_feats.get("pronunciation_score", 0.0))
    var_conf     = float(intel_feats.get("variance_confidence", 0.0))
    scores["intelligibility"] = compute_intelligibility_score(pron_score, var_conf)

    # 3. Language Control
    grammar_data   = features.get("language_control", {})
    grammar_errors = grammar_data.get("errors", [])
    transcript_words = raw_text.split()
    scores["language_control"] = compute_language_control_score(transcript_words, grammar_errors)

    # 4. Lexical Resource
    lexical_feats = features.get("lexical_resource", {})
    mattr         = float(lexical_feats.get("mattr", lexical_feats.get("type_token_ratio", 0.0)))
    rare_ratio    = float(lexical_feats.get("rare_word_ratio", 0.0))
    avg_freq      = float(lexical_feats.get("avg_word_frequency", 0.0))
    scores["lexical_resource"] = compute_lexical_resource_score(mattr, rare_ratio, avg_freq)

    # 5. Discourse
    disc_feats    = features.get("discourse", {})
    conn_count    = int(disc_feats.get("connector_count", 0))
    tier1_count   = int(disc_feats.get("tier1_count", 0))
    tier2_count   = int(disc_feats.get("tier2_count", 0))
    scores["discourse"] = compute_discourse_score(conn_count, tier1_count, tier2_count, total_words)

    # 6. Voice Modulation
    vm_feats       = features.get("voice_modulation", {})
    pitch_std      = float(vm_feats.get("pitch_std", 0.0))
    voiced_frac    = float(vm_feats.get("voiced_fraction", 0.0))
    scores["voice_modulation"] = compute_voice_modulation_score(pitch_std, voiced_frac)

    # 7. Sentiment & Confidence
    sent_feats     = features.get("sentiment", {})
    mean_compound  = float(sent_feats.get("mean_compound", 0.0))
    std_compound   = float(sent_feats.get("std_compound", 0.0))
    neg_ratio      = float(sent_feats.get("neg_sentiment_ratio", 0.0))
    assert_count   = int(sent_feats.get("assertive_count", 0))
    hedge_rate     = float(sent_feats.get("hedge_rate", 0.0))
    scores["sentiment"] = compute_sentiment_score(
        mean_compound, std_compound, neg_ratio, assert_count, hedge_rate, total_words
    )

    scores["overall_score"] = _compute_overall(scores)
    return scores


def predict_scores(features: dict) -> dict:
    """
    Predict 0–5 scores for each dimension.
    Attempts ML models; falls back to heuristics when models are absent.
    """
    if not _has_joblib:
        return heuristic_fallback(features)

    dimensions = [
        "fluency", "intelligibility", "language_control",
        "lexical_resource", "discourse", "sentiment",
    ]

    try:
        X = [flatten_features(features)]
        scores = {}
        for dim in dimensions:
            model_path = os.path.join(MODELS_DIR, f"{dim}_model.pkl")
            if not os.path.exists(model_path):
                logger.info("One or more ML models missing — using heuristic fallback.")
                return heuristic_fallback(features)
            model = joblib.load(model_path)
            pred = model.predict(X)[0]
            scores[dim] = max(0.0, min(5.0, round(float(pred), 2)))

        scores["overall_score"] = _compute_overall(scores)
        return scores

    except Exception as e:
        logger.error(f"ML prediction failed, using heuristic fallback: {e}")
        return heuristic_fallback(features)
