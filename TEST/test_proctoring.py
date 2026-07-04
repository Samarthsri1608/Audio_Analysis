from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)
TEST_FILE = ROOT / "proctoring_test.json"
OUTPUT_FILE = ROOT / "academic_violation_test_result.json"
API_URL = os.getenv(
    "PROCTORING_API_URL",
    "http://127.0.0.1:8002/v3/internal/analyse/{response_id}/proctoring",
)
INTERNAL_TOKEN = os.getenv("INTERNAL_PROCTORING_TOKEN", "")
REQUEST_TIMEOUT = int(os.getenv("PROCTORING_TIMEOUT_SEC", "1800"))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_proctoring")


def load_response_ids() -> list[str]:
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Missing test fixture: {TEST_FILE}")

    text = TEST_FILE.read_text(encoding="utf-8")
    ids = re.findall(r"\b[a-f0-9]{24}\b", text)

    seen: set[str] = set()
    ordered: list[str] = []
    for rid in ids:
        if rid not in seen:
            seen.add(rid)
            ordered.append(rid)
    return ordered


def normalize_error(response_id: str, status_code: int | None, detail: str) -> dict:
    lowered = detail.lower()
    if "403 forbidden" in lowered or "404 not found" in lowered or "no recordings found" in lowered:
        error_type = "not_found"
    elif status_code == 404:
        error_type = "not_found"
    else:
        error_type = "request_error"
    return {
        "response_id": response_id,
        "status": "failure",
        "error": {
            "type": error_type,
            "status_code": status_code,
            "detail": detail,
        },
        "question_review": [],
        "flagged_quuestions": [],
        "schema_version": "v2",
    }


def call_api(response_id: str) -> dict:
    if not INTERNAL_TOKEN:
        raise RuntimeError("INTERNAL_PROCTORING_TOKEN is not set in the environment.")

    url = API_URL.format(response_id=response_id)
    headers = {"X-Internal-Token": INTERNAL_TOKEN}
    start = time.time()
    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.Timeout:
        return normalize_error(response_id, None, "timeout")
    except Exception as exc:
        return normalize_error(response_id, None, str(exc))

    elapsed = round(time.time() - start, 3)

    if resp.status_code == 200:
        payload = resp.json()
        payload["_diagnostic"] = {"elapsed_sec": elapsed, "http_status": 200}
        return payload

    detail = resp.text[:500]
    return normalize_error(response_id, resp.status_code, detail)


def main() -> None:
    response_ids = load_response_ids()
    logger.info("Loaded %d response IDs from %s", len(response_ids), TEST_FILE)

    results: list[dict] = []
    for idx, response_id in enumerate(response_ids, 1):
        logger.info("[%d/%d] %s", idx, len(response_ids), response_id)
        result = call_api(response_id)
        results.append(result)

    OUTPUT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote %d proctoring results to %s", len(results), OUTPUT_FILE)


if __name__ == "__main__":
    main()
