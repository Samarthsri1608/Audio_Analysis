import logging
import json
from google.genai import types
from pydantic import BaseModel, Field

from app.shared_models import get_gemini_client

logger = logging.getLogger(__name__)

class GrammarError(BaseModel):
    message: str = Field(description="Clear explanation of the grammar mistake")
    context: str = Field(description="The exact text snippet around the mistake")
    ruleId: str = Field(description="A descriptive ID for the rule, e.g., SUBJECT_VERB_AGREEMENT")
    category: str = Field(description="Must be 'GRAMMAR'")
    replacements: list[str] = Field(description="Up to 3 suggested replacements for the error")

class GrammarResult(BaseModel):
    errors: list[GrammarError] = Field(description="List of all grammatical errors found")


def extract_language_control_features(text: str) -> dict:
    """
    Grammar checker using Gemini API.
    Identifies grammatical errors excluding punctuation/casing/style slips.
    """
    if not text.strip():
        return {
            "error_count": 0,
            "grammar_error_count": 0,
            "errors": [],
        }

    client = get_gemini_client()
    if client is None:
        logger.warning("Gemini client not initialized. Skipping grammar check.")
        return {
            "error_count": 0,
            "grammar_error_count": 0,
            "errors": [],
        }

    prompt = (
        "You are an expert English grammar checker. "
        "Analyze the following transcript of spoken English in an Indian professional context. "
        "Identify grammatical errors (excluding punctuation, casing, or styling slips, as this is a spoken transcript). "
        "Focus on verb-subject agreement, word choice, tenses, word order, and clear grammatical violations. "
        "Return the errors in the structured JSON format."
    )

    try:
        max_retries = 3
        backoff = 2
        response = None
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=[prompt, f"Transcript: {text}"],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=GrammarResult,
                    ),
                )
                break
            except Exception as e:
                import time
                if attempt == max_retries - 1:
                    logger.error(f"Gemini grammar check failed after {max_retries} attempts: {e}")
                    raise
                logger.warning(f"Gemini grammar check attempt {attempt+1} failed: {e}. Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2

        result_data = json.loads(response.text)
        errors = result_data.get("errors", [])
        return {
            "error_count": len(errors),
            "grammar_error_count": len(errors),
            "errors": errors,
        }
    except Exception as e:
        logger.error(f"Gemini grammar check failed: {e}")
        return {
            "error_count": 0,
            "grammar_error_count": 0,
            "errors": [],
        }
