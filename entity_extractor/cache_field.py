import json
import os

def save_field_to_single_json(field_name: str, data, json_file="fields.json"):
    # If JSON file exists → load it
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    # Add or update field data
    existing_data[field_name] = data

    # Save back to the same file
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)

    print(f"Updated {json_file} with field '{field_name}'")
