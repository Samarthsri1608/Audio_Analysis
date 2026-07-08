#!/usr/bin/env python3
import os
import sys
import json
import asyncio
import argparse
import logging
from urllib.parse import urlparse, parse_qs
from pathlib import Path
import httpx
from dotenv import load_dotenv

# Set up paths and load .env
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

# Import get_response_id from extract_response_id.py in the same folder
sys.path.insert(0, str(ROOT))
try:
    from extract_response_id import get_response_id
except ImportError:
    # Fallback to local import if needed
    from TEST_Proctoring.extract_response_id import get_response_id

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "batch_proctoring.log", mode="w")
    ]
)
logger = logging.getLogger("batch_proctoring")

# Configuration from env/arguments
INTERNAL_TOKEN = os.getenv("INTERNAL_PROCTORING_TOKEN", "internal_testing_token")
API_BASE_URL = os.getenv("PROCTORING_API_BASE_URL", "http://127.0.0.1:8002")

def format_links_file(file_path: Path) -> list[dict]:
    """Reads raw report_links.json, parses URLs, formats it to clean JSON list of dicts, and saves it."""
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return []
    
    content = file_path.read_text(encoding="utf-8").strip()
    
    # Try parsing as JSON first in case it's already formatted
    try:
        data = json.loads(content)
        if isinstance(data, list) and all(isinstance(x, dict) and "url" in x for x in data):
            logger.info("report_links.json is already formatted as clean JSON.")
            return data
    except json.JSONDecodeError:
        pass
    
    # Not JSON, parse line by line
    logger.info("Formatting raw report_links.json to clean JSON...")
    lines = content.splitlines()
    parsed_items = []
    
    for line_no, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.lower() == "report":
            continue
        
        # Parse URL parameters
        try:
            parsed_url = urlparse(line)
            qs = parse_qs(parsed_url.query)
            job_id = qs.get("jobId", [None])[0]
            candidate_id = qs.get("candidateId", [None])[0]
            
            if job_id and candidate_id:
                parsed_items.append({
                    "url": line,
                    "job_id": job_id,
                    "candidate_id": candidate_id
                })
            else:
                logger.warning(f"Line {line_no}: Missing jobId or candidateId in URL: {line}")
        except Exception as e:
            logger.error(f"Line {line_no}: Failed to parse URL '{line}': {e}")
            
    # Write back clean JSON
    file_path.write_text(json.dumps(parsed_items, indent=2), encoding="utf-8")
    logger.info(f"Formatted {len(parsed_items)} links and wrote to {file_path}")
    return parsed_items

async def process_single_candidate(
    client: httpx.AsyncClient, 
    item: dict, 
    auth_token: str, 
    cookie: str, 
    custom_headers: list[str]
) -> dict:
    url = item["url"]
    job_id = item["job_id"]
    candidate_id = item["candidate_id"]
    
    # 1. Fetch responseId
    # Note: get_response_id is synchronous, so we run it in a thread executor to keep loop unblocked
    loop = asyncio.get_running_loop()
    response_id = await loop.run_in_executor(
        None, 
        lambda: get_response_id(candidate_id, job_id, auth_token, cookie, custom_headers)
    )
    
    if not response_id:
        logger.error(f"Could not retrieve responseId for candidate {candidate_id}, job {job_id}")
        return {
            "candidate_id": candidate_id,
            "job_id": job_id,
            "status": "fail",
            "error": "Failed to retrieve responseId from Zeko API"
        }
        
    # 2. Call local proctoring API
    proctor_url = f"{API_BASE_URL}/v4/internal/analyse/{response_id}/proctoring"
    logger.info(f"Triggering proctoring for responseId {response_id}...")
    
    try:
        resp = await client.get(
            proctor_url,
            headers={"X-Internal-Token": INTERNAL_TOKEN},
            timeout=1800.0  # Allow long timeout for feature extraction / transcriptions
        )
        if resp.status_code == 200:
            result = resp.json()
            # Inject candidate metadata into result
            result["candidate_id"] = candidate_id
            result["job_id"] = job_id
            logger.info(f"Proctoring SUCCESS for responseId {response_id}: status={result.get('status')}")
            return result
        else:
            logger.error(f"Proctoring API error for {response_id} (HTTP {resp.status_code}): {resp.text[:200]}")
            return {
                "candidate_id": candidate_id,
                "job_id": job_id,
                "response_id": response_id,
                "status": "fail",
                "error": f"Proctoring API returned status {resp.status_code}: {resp.text[:200]}"
            }
    except Exception as e:
        logger.error(f"Exception calling proctoring API for {response_id}: {e}")
        return {
            "candidate_id": candidate_id,
            "job_id": job_id,
            "response_id": response_id,
            "status": "fail",
            "error": f"Exception calling proctoring API: {str(e)}"
        }

async def main_async():
    parser = argparse.ArgumentParser(description="Batch proctoring process runner")
    parser.add_argument("--auth", help="Authorization Token / Bearer Token for Zeko API")
    parser.add_argument("--cookie", help="Cookie string to send with the request to Zeko API")
    parser.add_argument("--header", action="append", help="Custom headers in 'Name:Value' format (can be specified multiple times)")
    parser.add_argument("--concurrency", type=int, default=3, help="Max concurrent proctoring tasks")
    args = parser.parse_args()
    
    links_file = ROOT / "report_links.json"
    output_file = ROOT / "output.json"
    
    # Step 1: Format report_links.json to clean JSON
    candidates = format_links_file(links_file)
    if not candidates:
        logger.error("No valid candidate links to process.")
        return
        
    logger.info(f"Starting batch proctoring for {len(candidates)} candidates (concurrency: {args.concurrency})...")
    
    results = []
    sem = asyncio.Semaphore(args.concurrency)
    
    async with httpx.AsyncClient(timeout=1800.0) as client:
        async def worker(item):
            async with sem:
                res = await process_single_candidate(client, item, args.auth, args.cookie, args.header)
                results.append(res)
                # Periodically save intermediate results in case of interruption
                output_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        
        tasks = [worker(item) for item in candidates]
        await asyncio.gather(*tasks)
        
    logger.info(f"Batch proctoring completed. Wrote {len(results)} results to {output_file}")

if __name__ == "__main__":
    asyncio.run(main_async())
