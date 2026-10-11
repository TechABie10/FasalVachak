import json
from pathlib import Path

# Automatically find the project root directory
# (assuming kb/database.py is one level down from the root)
BASE_DIR = Path(__file__).resolve().parent.parent


def build_master_kb() -> dict:
    """
    Loads and merges all crop JSON files into a single O(1) lookup dictionary.
    Raises an error if duplicate disease IDs exist across files.
    """
    master_kb = {}

    # Safely construct absolute paths to your JSON files
    files_to_load = [BASE_DIR / "kb" / "cotton.json", BASE_DIR / "kb" / "rice.json"]

    for file_path in files_to_load:
        with open(file_path, "r", encoding="utf-8") as f:
            crop_data = json.load(f)

            # Check for overlapping disease_ids
            overlap = set(master_kb) & set(crop_data)
            if overlap:
                raise ValueError(
                    f"Duplicate disease_id(s) found across files: {overlap}")

            # Merge into the master dictionary
            master_kb.update(crop_data)

    return master_kb


# This executes exactly ONCE when you do `from kb.database import FULL_KB_DICT`
# The dictionary stays cached in RAM for the lifetime of your server.
FULL_KB_DICT = build_master_kb()
