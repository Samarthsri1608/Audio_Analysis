import os
import requests
import json

API_LOC = "http://127.0.0.1:8000/api/v1"
TEST_FILE = "testing_interviews.json"

Results = {}

import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(filename)s] %(message)s")

# Resume from existing results if available
if os.path.exists("test_results.json"):
    try:
        with open("test_results.json", "r") as f:
            Results = json.load(f)
    except json.JSONDecodeError:
        pass

with open(TEST_FILE, "r") as f:
    data = json.load(f)
    
for file, info in data.items():
    if file in Results and "error" not in Results[file]:
        logging.info(f"Skipping '{file}': already successfully evaluated.")
        continue
        
    logging.info(f"Starting evaluation for: {file} (ID: {file})")
    start_time = time.time()
    
    try:
        # Stream=True is critical because /evaluate_by_id returns NDJSON
        response = requests.post(
            f"{API_LOC}/evaluate_by_id", 
            params={"response_id": file},
            stream=True,
            timeout=1800  # 30 minutes; heavy local models need time!
        )
        
        if response.status_code != 200:
            err_msg = f"HTTP {response.status_code}: {response.text[:200]}"
            logging.error(f"Failed '{file}': {err_msg}")
            Results[file] = {"error": err_msg}
        else:
            final_report = None
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    if chunk.get("type") == "status":
                        logging.info(f"  → Status: {chunk.get('message')}")
                    elif chunk.get("type") == "result":
                        final_report = chunk.get("payload")
                        logging.info(f"  → 🟢 Evaluation completed successfully.")
                    elif chunk.get("type") == "error":
                        logging.error(f"  → 🔴 Pipeline Error: {chunk.get('message')}")
                        Results[file] = {"error": chunk.get("message")}
                except json.JSONDecodeError:
                    logging.warning(f"  → Ignored unparseable chunk: {line}")
            
            if final_report:
                Results[file] = final_report
            elif "error" not in Results[file]:
                Results[file] = {"error": "Stream closed without returning a result payload."}
                
    except requests.exceptions.Timeout:
        logging.error(f"Request timed out for '{file}'.")
        Results[file] = {"error": "Timeout"}
    except Exception as e:
        logging.error(f"Exception for '{file}': {e}")
        Results[file] = {"error": str(e)}

    elapsed = time.time() - start_time
    logging.info(f"Finished '{file}' in {elapsed:.1f}s\n")

    # Incrementally save results after each interview
    with open("test_results.json", "w") as f:
        json.dump(Results, f, indent=4)

logging.info("🎉 All testing completed!")
