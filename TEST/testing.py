import os
import requests
import json

API_LOC = "http://127.0.0.1:8000/"
TEST_FILE = r"/Users/samarthsrivastava/Voice-projects/Test_Interviews/test.json"

Results = {}

with open(TEST_FILE, "r") as f:
    data = json.load(f)
    for file, info in data.items():
        print(f"fetching {info['id']}...")
        try:
            response = requests.post(API_LOC + "evaluate_by_id", params={"response_id": info['id']})
            if response.status_code == 200:
                Results[file] = response.json()
                print(f"success: {file}")
            else:
                Results[file] = {"error": response.text}
                print(f"error: {file} → {response.status_code}: {response.text[:300]}")
        except Exception as e:
            Results[file] = {"error": str(e)}
            print(f"error: {file} → {str(e)[:300]}")

results = json.dumps(Results, indent=4)
with open("test_results.json", "w") as f:
    f.write(results)


