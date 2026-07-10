#!/usr/bin/env python3
"""
run_audio_batch.py — Batch testing script for the V4 Audio-Only Proctoring pipeline.

Usage:
    python TEST_Proctoring/run_audio_batch.py \\
        --auth "Bearer <token>" \\
        --cookie "<cookie_string>" \\
        [--header "X-Custom: value"] \\
        [--concurrency 3] \\
        [--api-url http://127.0.0.1:8003]

Input:
    TEST_Proctoring/report_links.json      — candidate URLs (pre-formatted)

Outputs:
    TEST_Proctoring/output_audio.json      — raw per-candidate evidence payloads
    TEST_Proctoring/output_audio.csv       — flat CSV (one row per flagged question +
                                             one summary row per unflagged candidate)

CSV columns (designed to match output.csv for side-by-side v3/v4 comparison):
    candidate_id, job_id, response_id, status, report_link,
    Audited_cheat_probability, Remarks, Match,
    error_type, error_detail,
    flagged_questions,
    q_no, evaluable, not_evaluable_reason,
    flagged_for_review, confidence, is_cold_start,
    track_a_score, track_a_flagged,
    track_c_signals, track_c_details,
    response_latency_s, spectral_flatness_mean, pause_ratio,
    f0_mean_hz, f0_std_hz, speech_rate_proxy, energy_mean,
    speech_duration_s, total_duration_s,
    total_questions_evaluated, total_questions_flagged

Ground-truth join:
    Merges in Audited_cheat_probability and Remarks from
    "Persistent BFSI - Completed.csv" by matching the report URL.
    Match column is True when our flag decision agrees with ground truth.
"""
from __future__ import annotations

import asyncio
import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import httpx
from dotenv import load_dotenv

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=False)

# ── Inlined get_response_id (reconstructed from bytecode; extract_response_id.py missing) ──
# Original module lived in TEST_Proctoring/extract_response_id.py (only .pyc present).
# Logic deduced from bytecode constants and variable names.

def get_response_id(
    candidate_id: str,
    job_id: str,
    auth_token: str | None,
    cookie: str | None,
    custom_headers: list[str] | None,
) -> str | None:
    """
    Fetch the responseId for a (candidateId, jobId) pair from the Zeko API.

    Checks response headers first (X-Response-Id / responseId), then falls back
    to parsing the JSON body's `data` dict for a `responseId` key.
    """
    url = (
        f"https://api.zeko.ai/mygurukul/ait/interview-report"
        f"?candidateId={candidate_id}&jobId={job_id}"
    )
    headers_to_send: dict[str, str] = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    if auth_token:
        if auth_token.lower().startswith("bearer "):
            headers_to_send["Authorization"] = auth_token
        else:
            headers_to_send["Authorization"] = f"Bearer {auth_token}"
    if cookie:
        # Strip all newlines/extra whitespace — terminal word-wrap can embed \n in pasted values
        clean_cookie = " ".join(cookie.split())
        headers_to_send["Cookie"] = clean_cookie
    if custom_headers:
        for h in custom_headers:
            if ":" in h:
                k, v = h.split(":", 1)
                headers_to_send[k.strip()] = v.strip()

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(url, headers=headers_to_send)
        logger.debug("get_response_id status=%d for candidate=%s job=%s", resp.status_code, candidate_id, job_id)

        # Check response headers first
        for key in resp.headers:
            if "responseid" in key.lower():
                logger.debug("Found responseId in header '%s': %s", key, resp.headers[key])
                return resp.headers[key]

        # Fall back to JSON body
        try:
            body = resp.json()
            if isinstance(body, dict):
                data = body.get("data", body)
                if isinstance(data, dict) and "responseId" in data:
                    return data["responseId"]
        except Exception:
            pass

        logger.warning("responseId not found in headers or body for candidate=%s job=%s", candidate_id, job_id)
        return None

    except Exception as e:
        logger.error("get_response_id exception: %s", e, exc_info=True)
        return None


# ── config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
LINKS_FILE = SCRIPT_DIR / "report_links.json"
OUTPUT_JSON = SCRIPT_DIR / "output_audio.json"
OUTPUT_CSV = SCRIPT_DIR / "output_audio.csv"
GROUND_TRUTH_CSV = SCRIPT_DIR / "Persistent BFSI - Completed.csv"
INTERNAL_TOKEN = os.getenv("INTERNAL_PROCTORING_TOKEN", "internal_testing_token")
LOG_FILE = SCRIPT_DIR / "batch_audio.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
logger = logging.getLogger("batch_audio")


# ── CSV columns ───────────────────────────────────────────────────────────────
CSV_FIELDNAMES = [
    # identity
    "candidate_id", "job_id", "response_id", "status", "report_link",
    # ground truth (joined from audit CSV)
    "Audited_cheat_probability", "Remarks",
    # decision comparison
    "Match",
    # error info
    "error_type", "error_detail",
    # interview-level summary
    "flagged_questions", "total_questions_evaluated", "total_questions_flagged",
    # per-question fields (one row per flagged question; summary row if no flags)
    "q_no", "evaluable", "not_evaluable_reason",
    "flagged_for_review", "confidence", "is_cold_start",
    # Track A
    "track_a_available", "track_a_score", "track_a_flagged",
    # Track C
    "track_c_signals", "track_c_details",
    # Contributing features (key ones for human reviewers)
    "response_latency_s", "spectral_flatness_mean", "pause_ratio",
    "f0_mean_hz", "f0_std_hz", "speech_rate_proxy", "energy_mean",
    "speech_duration_s", "total_duration_s",
]


# ── Ground-truth loader ───────────────────────────────────────────────────────

def load_ground_truth() -> dict[str, dict]:
    """
    Load the auditor ground-truth CSV and build a lookup dict keyed by
    report URL → {Audited_cheat_probability, Remarks}.

    The CSV has two identically-named "Report" columns (index 7 = URL,
    index 8 = reviewer name), which breaks csv.DictReader. We use raw
    csv.reader with explicit column indices instead.

    Column layout (0-indexed):
      0  Candidate Name
      1  Email
      2  Phone
      3  Cheat Probability
      4  Status
      5  Interview score
      6  Coding Score
      7  Report (URL)           ← this is the join key
      8  Report (reviewer name) ← ignored
      9  Audited Cheat Probability
      10 Remarks (Time-stamps)
      11 Job Role
      12 Recruiter

    Returns empty dict if the file doesn't exist.
    """
    if not GROUND_TRUTH_CSV.exists():
        logger.warning("Ground-truth CSV not found at %s — skipping join", GROUND_TRUTH_CSV)
        return {}

    URL_COL    = 7
    AUDITED_COL = 9
    REMARKS_COL = 10

    gt: dict[str, dict] = {}
    with open(GROUND_TRUTH_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header row
        for row in reader:
            if len(row) <= AUDITED_COL:
                continue
            url = row[URL_COL].strip()
            if not url or not url.startswith("http"):
                continue
            gt[url] = {
                "Audited_cheat_probability": row[AUDITED_COL].strip(),
                "Remarks": row[REMARKS_COL].strip() if len(row) > REMARKS_COL else "",
            }

    logger.info("Loaded %d ground-truth entries", len(gt))
    return gt


def get_ground_truth(gt: dict[str, dict], job_id: str, candidate_id: str) -> dict:
    """
    Look up audited cheat probability by reconstructing the report URL.
    Tries both known URL formats.
    """
    urls = [
        f"https://app.zeko.ai/app/new-report?jobId={job_id}&candidateId={candidate_id}",
        f"https://app.zeko.ai/app/report?jobId={job_id}&candidateId={candidate_id}",
    ]
    for url in urls:
        if url in gt:
            return gt[url]
    return {"Audited_cheat_probability": "", "Remarks": ""}


def compute_match(audited: str, flagged_questions: list) -> str:
    """
    Compute whether v4 decision agrees with auditor ground truth.
    'High' audited = expected to be flagged. 'Low' = expected clean.
    Returns 'True' / 'False' / '' (if no ground truth).
    """
    if not audited:
        return ""
    v4_flagged = len(flagged_questions) > 0
    audited_high = audited.strip().lower() == "high"
    return str(v4_flagged == audited_high)


# ── JSON → CSV conversion ─────────────────────────────────────────────────────

def _safe(val) -> str:
    """Convert a value to a CSV-safe string."""
    if val is None:
        return ""
    if isinstance(val, list):
        return "|".join(str(v) for v in val)
    if isinstance(val, dict):
        # Compact one-liner for nested dicts (Track C details)
        return "; ".join(f"{k}={v}" for k, v in val.items())
    return str(val)


def json_to_csv(results: list[dict], gt: dict[str, dict]) -> None:
    """
    Flatten output_audio.json to output_audio.csv.

    Row-expansion strategy (mirrors output.csv from v3):
    - If an interview has flagged questions: one row per flagged question.
    - If an interview has NO flagged questions: one summary row (q_no blank).
    - If the interview failed entirely: one error row.
    - Non-flagged evaluable questions are NOT expanded into their own rows
      (keeps the CSV manageable), but the counts appear in the summary columns.
    """
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        for rec in results:
            cid = rec.get("candidate_id", "")
            jid = rec.get("job_id", "")
            rid = rec.get("response_id", "")
            url = rec.get("report_link", "")
            status = rec.get("status", "")
            flagged_qs = rec.get("flagged_questions", [])
            total_eval = rec.get("total_questions_evaluated", "")
            total_flagged = rec.get("total_questions_flagged", "")

            gt_info = get_ground_truth(gt, jid, cid)
            match = compute_match(gt_info["Audited_cheat_probability"], flagged_qs)

            base = {
                "candidate_id": cid,
                "job_id": jid,
                "response_id": rid,
                "status": status,
                "report_link": url,
                "Audited_cheat_probability": gt_info["Audited_cheat_probability"],
                "Remarks": gt_info["Remarks"],
                "Match": match,
                "error_type": _safe(rec.get("error", {}).get("type") if rec.get("error") else None),
                "error_detail": _safe(rec.get("error", {}).get("detail") if rec.get("error") else None),
                "flagged_questions": _safe(flagged_qs),
                "total_questions_evaluated": _safe(total_eval),
                "total_questions_flagged": _safe(total_flagged),
            }

            # Error row
            if status == "fail":
                writer.writerow(base | {k: "" for k in CSV_FIELDNAMES if k not in base})
                continue

            # Build a lookup of flagged question evidence payloads
            evidence_by_qno = {
                p["q_no"]: p
                for p in rec.get("question_evidence", [])
                if p.get("flagged_for_review")
            }

            if not flagged_qs:
                # Clean interview — one summary row with all per-question fields blank
                writer.writerow(base | {k: "" for k in CSV_FIELDNAMES if k not in base})
                continue

            # One row per flagged question
            for q_no in flagged_qs:
                p = evidence_by_qno.get(q_no, {})
                cf = p.get("contributing_features", {})
                ta = p.get("track_a") or {}
                tc = p.get("track_c") or {}

                row = base.copy()
                row.update({
                    "q_no": _safe(q_no),
                    "evaluable": _safe(p.get("evaluable")),
                    "not_evaluable_reason": _safe(p.get("not_evaluable_reason")),
                    "flagged_for_review": _safe(p.get("flagged_for_review")),
                    "confidence": _safe(p.get("confidence")),
                    "is_cold_start": _safe(p.get("is_cold_start")),
                    # Track A
                    "track_a_available": _safe(ta.get("available")),
                    "track_a_score": _safe(ta.get("deviation_score")),
                    "track_a_flagged": _safe(ta.get("flagged")),
                    # Track C
                    "track_c_signals": _safe(tc.get("signals", [])),
                    "track_c_details": _safe(tc.get("signal_details", {})),
                    # Contributing features
                    "response_latency_s": _safe(cf.get("response_latency_s")),
                    "spectral_flatness_mean": _safe(cf.get("spectral_flatness_mean")),
                    "pause_ratio": _safe(cf.get("pause_ratio")),
                    "f0_mean_hz": _safe(cf.get("f0_mean_hz")),
                    "f0_std_hz": _safe(cf.get("f0_std_hz")),
                    "speech_rate_proxy": _safe(cf.get("speech_rate_proxy")),
                    "energy_mean": _safe(cf.get("energy_mean")),
                    "speech_duration_s": _safe(cf.get("speech_duration_s")),
                    "total_duration_s": _safe(cf.get("total_duration_s")),
                })
                writer.writerow(row)

    logger.info("CSV written → %s", OUTPUT_CSV)


# ── API call ──────────────────────────────────────────────────────────────────

async def process_single_candidate(
    client: httpx.AsyncClient,
    item: dict,
    auth_token: str | None,
    cookie: str | None,
    custom_headers: list[str] | None,
    api_base_url: str,
) -> dict:
    """Fetch responseId then hit the V4 proctoring endpoint."""
    url = item["url"]
    job_id = item["job_id"]
    candidate_id = item["candidate_id"]

    # ── Step 1: resolve responseId ────────────────────────────────────────────
    loop = asyncio.get_running_loop()
    response_id = await loop.run_in_executor(
        None,
        lambda: get_response_id(candidate_id, job_id, auth_token, cookie, custom_headers),
    )

    if not response_id:
        logger.error("Could not get responseId for candidate=%s job=%s", candidate_id, job_id)
        return {
            "candidate_id": candidate_id,
            "job_id": job_id,
            "response_id": None,
            "report_link": url,
            "status": "fail",
            "error": {"type": "no_response_id", "detail": "Failed to retrieve responseId from Zeko API"},
            "flagged_questions": [],
            "question_evidence": [],
        }

    logger.info("Processing responseId=%s (candidate=%s)", response_id, candidate_id)

    # ── Step 2: call V4 proctoring endpoint ───────────────────────────────────
    proctor_url = f"{api_base_url}/v4/internal/analyse/{response_id}/proctoring"
    try:
        resp = await client.get(
            proctor_url,
            headers={"X-Internal-Token": INTERNAL_TOKEN},
            timeout=1800.0,
        )
        if resp.status_code == 200:
            result = resp.json()
            result["candidate_id"] = candidate_id
            result["job_id"] = job_id
            result["report_link"] = url
            n_flagged = result.get("total_questions_flagged", len(result.get("flagged_questions", [])))
            logger.info(
                "SUCCESS responseId=%s — %d question(s) flagged",
                response_id, n_flagged,
            )
            return result
        else:
            logger.error(
                "API error for responseId=%s (HTTP %d): %s",
                response_id, resp.status_code, resp.text[:300],
            )
            return {
                "candidate_id": candidate_id,
                "job_id": job_id,
                "response_id": response_id,
                "report_link": url,
                "status": "fail",
                "error": {
                    "type": "api_error",
                    "detail": f"HTTP {resp.status_code}: {resp.text[:300]}",
                },
                "flagged_questions": [],
                "question_evidence": [],
            }
    except Exception as exc:
        logger.error("Exception for responseId=%s: %s", response_id, exc)
        return {
            "candidate_id": candidate_id,
            "job_id": job_id,
            "response_id": response_id,
            "report_link": url,
            "status": "fail",
            "error": {"type": "exception", "detail": str(exc)},
            "flagged_questions": [],
            "question_evidence": [],
        }


# ── Main ──────────────────────────────────────────────────────────────────────

async def main_async(args: argparse.Namespace) -> None:
    # ── Load candidates ───────────────────────────────────────────────────────
    if not LINKS_FILE.exists():
        logger.error("report_links.json not found at %s", LINKS_FILE)
        sys.exit(1)

    with open(LINKS_FILE, encoding="utf-8") as f:
        candidates: list[dict] = json.load(f)

    if not candidates:
        logger.error("No candidates found in report_links.json")
        sys.exit(1)

    logger.info(
        "Starting V4 audio-only batch for %d candidates (concurrency=%d, api=%s)",
        len(candidates), args.concurrency, args.api_url,
    )

    # ── Load ground truth for CSV join ────────────────────────────────────────
    gt = load_ground_truth()

    # ── Run batch ─────────────────────────────────────────────────────────────
    results: list[dict] = []
    sem = asyncio.Semaphore(args.concurrency)

    custom_headers = args.header or []

    async with httpx.AsyncClient(timeout=1800.0) as client:

        async def worker(item: dict) -> None:
            async with sem:
                res = await process_single_candidate(
                    client, item,
                    auth_token=args.auth,
                    cookie=args.cookie,
                    custom_headers=custom_headers,
                    api_base_url=args.api_url,
                )
                results.append(res)
                # Write intermediate JSON after every candidate — safe against crashes
                OUTPUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

        tasks = [worker(item) for item in candidates]
        await asyncio.gather(*tasks)

    # ── Final JSON ────────────────────────────────────────────────────────────
    OUTPUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("JSON written → %s (%d records)", OUTPUT_JSON, len(results))

    # ── Convert to CSV ────────────────────────────────────────────────────────
    json_to_csv(results, gt)

    # ── Summary stats ─────────────────────────────────────────────────────────
    successes = [r for r in results if r.get("status") == "success"]
    failures  = [r for r in results if r.get("status") != "success"]
    flagged   = [r for r in successes if r.get("flagged_questions")]

    logger.info("─" * 60)
    logger.info("Batch complete")
    logger.info("  Total       : %d", len(results))
    logger.info("  Success     : %d", len(successes))
    logger.info("  Failed      : %d", len(failures))
    logger.info("  Flagged     : %d / %d", len(flagged), len(successes))

    if gt:
        # Accuracy against ground truth (where available)
        matched = [
            r for r in successes
            if compute_match(
                get_ground_truth(gt, r.get("job_id", ""), r.get("candidate_id", ""))
                ["Audited_cheat_probability"],
                r.get("flagged_questions", [])
            ) == "True"
        ]
        gt_available = [
            r for r in successes
            if get_ground_truth(gt, r.get("job_id", ""), r.get("candidate_id", ""))
            ["Audited_cheat_probability"]
        ]
        if gt_available:
            acc = len(matched) / len(gt_available) * 100
            logger.info("  Accuracy    : %.1f%% (%d / %d with ground truth)", acc, len(matched), len(gt_available))

    logger.info("─" * 60)
    logger.info("Outputs:")
    logger.info("  JSON: %s", OUTPUT_JSON)
    logger.info("  CSV : %s", OUTPUT_CSV)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V4 Audio-Only Proctoring — Batch Testing Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With auth token and cookie (same as v3 batch runner):
  python TEST_Proctoring/run_audio_batch.py \\
      --auth "Bearer eyJ..." \\
      --cookie "session=abc123" \\
      --concurrency 3

  # With custom headers:
  python TEST_Proctoring/run_audio_batch.py \\
      --auth "Bearer eyJ..." \\
      --header "X-Tenant-Id: acme" \\
      --concurrency 2

  # Against a different API host:
  python TEST_Proctoring/run_audio_batch.py \\
      --auth "Bearer eyJ..." \\
      --api-url http://localhost:8003
        """,
    )
    parser.add_argument("--auth", help="Authorization / Bearer token for Zeko API")
    parser.add_argument("--cookie", help="Cookie string for Zeko API")
    parser.add_argument(
        "--header",
        action="append",
        metavar="Name:Value",
        help="Custom header in 'Name: Value' format (can be repeated)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Max concurrent proctoring calls (default: 3)",
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8003",
        help="Base URL of the V4 proctoring service (default: http://127.0.0.1:8003)",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
