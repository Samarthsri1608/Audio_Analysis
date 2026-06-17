import os
import sys
import json
import time
import requests
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

API_URL = "http://127.0.0.1:8001/v2/analyze"
TEST_FILE = "testing_interviews.json"
OUTPUT_FILE = "new_framework_results.jsonl"

def load_processed_ids():
    processed = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if "response_id" in data:
                        processed.add(data["response_id"])
                    elif "error" in data and "response_id" in data.get("request", {}):
                        processed.add(data["request"]["response_id"])
                except json.JSONDecodeError:
                    pass
    return processed

def main():
    if not os.path.exists(TEST_FILE):
        logging.error(f"Test file {TEST_FILE} not found!")
        return

    with open(TEST_FILE, "r") as f:
        test_cases = json.load(f)
    
    # test_cases is a list of dicts like [{"_id": "..."}]
    total_cases = len(test_cases)
    logging.info(f"Loaded {total_cases} test cases from {TEST_FILE}")

    processed_ids = load_processed_ids()
    logging.info(f"Found {len(processed_ids)} already processed cases. Resuming...")

    # Open output file in append mode
    with open(OUTPUT_FILE, "a") as out_f:
        for idx, case in enumerate(test_cases, 1):
            response_id = case.get("_id")
            if not response_id:
                logging.warning(f"Skipping case at index {idx}: missing '_id'")
                continue
            
            if response_id in processed_ids:
                logging.info(f"[{idx}/{total_cases}] Skipping {response_id} (already processed)")
                continue

            logging.info(f"[{idx}/{total_cases}] Processing {response_id}...")
            start_time = time.time()
            
            payload = {
                "response_id": response_id,
                "include_description": True,
                "role_profile": "default",
                "style_role": "default"
            }
            
            try:
                # 30-minute timeout to allow for heavy local model downloading and execution
                response = requests.post(API_URL, json=payload, timeout=1800)
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    out_f.write(json.dumps(result) + "\n")
                    out_f.flush()
                    logging.info(f"  → Success in {elapsed:.2f}s")
                else:
                    err_text = response.text[:300]
                    logging.error(f"  → HTTP {response.status_code}: {err_text}")
                    # Write failure record so we know it was processed but failed
                    fail_record = {
                        "response_id": response_id,
                        "status": "error",
                        "error": f"HTTP {response.status_code}: {err_text}"
                    }
                    out_f.write(json.dumps(fail_record) + "\n")
                    out_f.flush()
            except requests.exceptions.Timeout:
                logging.error(f"  → Request timed out for {response_id}")
                fail_record = {
                    "response_id": response_id,
                    "status": "error",
                    "error": "Timeout"
                }
                out_f.write(json.dumps(fail_record) + "\n")
                out_f.flush()
            except Exception as e:
                logging.error(f"  → Exception for {response_id}: {e}")
                fail_record = {
                    "response_id": response_id,
                    "status": "error",
                    "error": str(e)
                }
                out_f.write(json.dumps(fail_record) + "\n")
                out_f.flush()

    logging.info("🎉 Done processing all test cases!")

if __name__ == "__main__":
    main()
