import logging

logger = logging.getLogger(__name__)

TIER1_CONNECTORS = {
    "moreover", "furthermore", "therefore", "consequently",
    "nonetheless", "however", "in contrast", "on the other hand",
    "as a result", "hence", "thus", "accordingly", "nevertheless",
    "for instance", "for example", "to illustrate", "in conclusion",
    "to summarise", "to summarize", "in particular", "specifically",
    "subsequently", "although", "whereas",
}

TIER2_CONNECTORS = {
    "and", "but", "or", "so", "because", "while", "since",
    "then", "meanwhile", "next", "finally", "also", "additionally",
    "though", "yet", "still", "first", "second",
}


def extract_discourse_features(text: str) -> dict:
    if not text.strip():
        return {
            "connector_count": 0,
            "tier1_count": 0,
            "tier2_count": 0,
            "connectors_used": [],
        }

    text_lower = text.lower()
    import re

    # Tokenize words, stripping punctuation to avoid missing connectors next to commas/periods
    words = [w.strip(".,!?;:\"'()") for w in text_lower.split()]
    words = [w for w in words if w]

    tier1_found: list[str] = []
    tier2_found: list[str] = []

    # TIER 1: Use regex with word boundaries to count all occurrences of each connector
    for conn in TIER1_CONNECTORS:
        pattern = re.compile(rf"\b{re.escape(conn)}\b")
        matches = pattern.findall(text_lower)
        if matches:
            tier1_found.extend([conn] * len(matches))

    # TIER 2: Match against cleaned word list
    for conn in TIER2_CONNECTORS:
        count = words.count(conn)
        if count > 0:
            tier2_found.extend([conn] * count)

    all_found = tier1_found + tier2_found

    return {
        # connector_count needs to represent unique types for variety_bonus in evaluation.py
        "connector_count": len(set(all_found)),
        "tier1_count": len(tier1_found),
        "tier2_count": len(tier2_found),
        "connectors_used": sorted(set(all_found)),
    }
