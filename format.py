import json
import os
from collections import defaultdict

# Load JSON data from file
input_file = "output.json"  # Change this to your actual file name
output_folder = "comp_data"
os.makedirs(output_folder, exist_ok=True)

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Group data by FORMULA_TYPE_NAME
grouped_data = defaultdict(list)
for obj in data:
    formula_type = obj.get("FORMULA_TYPE_NAME", "Unknown")
    grouped_data[formula_type].append(obj)

# Save grouped data into separate files
for formula_type, formulas in grouped_data.items():
    safe_filename = formula_type.replace(" ", "_").replace("/", "_")  # Ensure valid filename
    output_file = os.path.join(output_folder, f"{safe_filename}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formulas, f, indent=4)
    print(f"Saved {len(formulas)} records to {output_file}")
