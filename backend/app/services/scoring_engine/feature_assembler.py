def flatten_features(features: dict) -> list:
    """
    Takes the nested JSON `features` dict from Phase 3 extraction and flattens
    it into a 1D numerical array for Scikit-Learn models.

    Expected input keys:
        fluency, intelligibility, language_control, lexical_resource,
        discourse, sentiment

    Current vector dimensionality: 16
    """

    # 1. Fluency
    fluency        = features.get("fluency", {})
    wpm            = float(fluency.get("wpm", 0.0))
    pause_freq     = float(fluency.get("pause_frequency", 0.0))
    mean_pause_dur = float(fluency.get("mean_pause_duration", 0.0))
    total_words    = float(fluency.get("total_words", 0.0))
    filler_rate    = float(fluency.get("filler_rate", 0.0))

    # 2. Intelligibility
    intelligibility = features.get("intelligibility", {})
    mean_conf       = float(intelligibility.get("mean_confidence", 0.0))
    var_conf        = float(intelligibility.get("variance_confidence", 0.0))
    pron_score      = float(intelligibility.get("pronunciation_score", 0.0))

    # 3. Language Control
    lang_ctrl     = features.get("language_control", {})
    error_count   = float(lang_ctrl.get("grammar_error_count",
                          lang_ctrl.get("error_count", 0.0)))
    # Derived: error density per word
    error_density = error_count / total_words if total_words > 0 else 0.0

    # 4. Lexical Resource
    lexical      = features.get("lexical_resource", {})
    mattr        = float(lexical.get("mattr", lexical.get("type_token_ratio", 0.0)))
    unique_words = float(lexical.get("unique_words", 0.0))
    rare_ratio   = float(lexical.get("rare_word_ratio", 0.0))

    # 5. Discourse
    discourse       = features.get("discourse", {})
    connector_count = float(discourse.get("connector_count", 0.0))
    # Derived: connectors per 100 words
    cph = (connector_count / total_words * 100) if total_words > 0 else 0.0

    # 6. Sentiment & Confidence
    sentiment    = features.get("sentiment", {})
    mean_cmpd    = float(sentiment.get("mean_compound",       0.0))
    std_cmpd     = float(sentiment.get("std_compound",        0.0))
    neg_ratio    = float(sentiment.get("neg_sentiment_ratio", 0.0))
    assert_count = float(sentiment.get("assertive_count",     0))
    hedge_rate   = float(sentiment.get("hedge_rate",          0.0))

    # 16-dimensional feature vector
    return [
        wpm,             # 1
        pause_freq,      # 2
        mean_pause_dur,  # 3
        total_words,     # 4
        filler_rate,     # 5
        mean_conf,       # 6
        var_conf,        # 7
        pron_score,      # 8
        error_count,     # 9
        error_density,   # 10
        mattr,           # 11
        unique_words,    # 12
        rare_ratio,      # 13
        cph,             # 14
        mean_cmpd,       # 15
        std_cmpd,        # 16
        neg_ratio,       # 17
        assert_count,    # 18
        hedge_rate,      # 19
    ]
