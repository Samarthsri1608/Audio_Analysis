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

API_URL = "http://127.0.0.1:8001/v2/analyze/{response_id}/communication"
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
            
            try:
                # Call only the Communication (System A) endpoint
                comm_url = f"http://127.0.0.1:8001/v2/analyse/{response_id}/communication"
                comm_params = {"role_profile": "default"}

                # 30-minute timeout to allow for heavy local model downloading and execution
                response_comm = requests.get(comm_url, params=comm_params, timeout=1800)
                
                if response_comm.status_code != 200:
                    err_text = response_comm.text[:300]
                    logging.error(f"  → Communication HTTP {response_comm.status_code}: {err_text}")
                    fail_record = {
                        "response_id": response_id,
                        "status": "error",
                        "error": f"Communication HTTP {response_comm.status_code}: {err_text}"
                    }
                    out_f.write(json.dumps(fail_record) + "\n")
                    out_f.flush()
                    continue

                elapsed = time.time() - start_time
                
                # Store only the communication results
                comm_data = response_comm.json()

                out_f.write(json.dumps(comm_data) + "\n")
                out_f.flush()
                logging.info(f"  → Success in {elapsed:.2f}s")

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
