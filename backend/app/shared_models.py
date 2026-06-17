"""
Shared heavy-model singletons (Stubs for backward compatibility) and API clients.
"""
import logging
from google import genai
from app.settings import settings

logger = logging.getLogger(__name__)

import re
import os

_genai_client = None
_diarization_cache = {}

def get_gemini_client():
    global _genai_client
    if _genai_client is None:
        api_key = settings.GOOGLE_API_KEY
        if api_key:
            try:
                _genai_client = genai.Client(api_key=api_key)
                logger.info("Shared Gemini API Client initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
    return _genai_client

def get_file_id(file_path: str) -> str:
    basename = os.path.basename(file_path)
    match = re.search(r"([a-f0-9\-]{36})", basename)
    if match:
        return match.group(1)
    return basename.split("_")[0].split(".")[0]

def get_diarization_cache() -> dict:
    global _diarization_cache
    return _diarization_cache

def get_spacy_nlp():
    return None

def get_language_tool():
    return None


