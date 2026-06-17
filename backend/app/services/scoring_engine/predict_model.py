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
    compute_collaborative_tone_score,
    compute_overall_score,
)

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

DIMENSION_WEIGHTS = {
    "logical_cohesion":      1.0 / 6.0,
    "delivery_fluency":      1.0 / 6.0,
    "pronunciation_clarity": 1.0 / 6.0,
    "vocal_dynamism":        1.0 / 6.0,
    "collaborative_tone":    1.0 / 6.0,
    "lexical_precision":     1.0 / 6.0,
}


def _compute_overall(scores: dict) -> float:
    return compute_overall_score(
        logical_cohesion=scores.get("logical_cohesion", 0),
        delivery_fluency=scores.get("delivery_fluency", 0),
        pronunciation_clarity=scores.get("pronunciation_clarity", 0),
        vocal_dynamism=scores.get("vocal_dynamism", 0),
        collaborative_tone=scores.get("collaborative_tone", 0),
        lexical_precision=scores.get("lexical_precision", 0),
    )


def heuristic_fallback(features: dict) -> dict:
    scores = {}
    raw_text = features.get("raw_text", "")
    total_words = max(len(raw_text.split()), 1)

    # 1. Logical Cohesion
    disc_feats    = features.get("discourse", {})
    conn_count    = int(disc_feats.get("connector_count", 0))
    tier1_count   = int(disc_feats.get("tier1_count", 0))
    tier2_count   = int(disc_feats.get("tier2_count", 0))
    disc_words    = len(raw_text.split())
    scores["logical_cohesion"] = compute_discourse_score(
        conn_count, tier1_count, tier2_count, disc_words
    )

    # 2. Delivery Fluency
    fluency_feats  = features.get("fluency", {})
    wpm            = float(fluency_feats.get("wpm", 0.0))
    pause_freq     = float(fluency_feats.get("pause_frequency", 0.0))
    filler_rate    = float(fluency_feats.get("filler_rate", 0.0))
    duration_s     = features.get("duration_ms", 0) / 1000.0
    scores["delivery_fluency"] = compute_fluency_score(wpm, pause_freq, duration_s, filler_rate)

    # 3. Pronunciation Clarity
    intel_feats      = features.get("intelligibility", {})
    mean_confidence  = float(intel_feats.get("mean_confidence",
                             intel_feats.get("pronunciation_score", 0.0)))
    scores["pronunciation_clarity"] = compute_intelligibility_score(
        mean_confidence=mean_confidence
    )

    # 4. Vocal Dynamism
    vm_feats       = features.get("voice_modulation", {})
    pitch_std      = float(vm_feats.get("pitch_std", 0.0))
    voiced_frac    = float(vm_feats.get("voiced_fraction", 0.0))
    scores["vocal_dynamism"] = compute_voice_modulation_score(pitch_std, voiced_frac)

    # 5. Collaborative Tone
    sent_feats     = features.get("sentiment", {})
    mean_compound  = float(sent_feats.get("mean_compound", 0.0))
    scores["collaborative_tone"] = compute_collaborative_tone_score(mean_compound, raw_text)

    # 6. Lexical Precision
    lexical_feats = features.get("lexical_resource", {})
    mattr         = float(lexical_feats.get("mattr", lexical_feats.get("type_token_ratio", 0.0)))
    rare_ratio    = float(lexical_feats.get("rare_word_ratio", 0.0))
    lex_words     = len(raw_text.split())  # use actual word count for gate
    scores["lexical_precision"] = compute_lexical_resource_score(
        mattr, rare_ratio, total_words=lex_words
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
        "logical_cohesion", "delivery_fluency", "pronunciation_clarity",
        "vocal_dynamism", "collaborative_tone", "lexical_precision",
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
