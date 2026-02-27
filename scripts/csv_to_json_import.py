"""
Convert CSV files in data/csv_exports/ to JSON in data/.
Run after updating Google Sheets and downloading CSVs, or after sync_from_sheets.py fetches.
Handles JSON columns (format, drills, focus, program_day_type), numbers, booleans, nulls.
"""
import csv
import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_DIR = DATA_DIR / "csv_exports"

# Columns that contain JSON (list or dict)
JSON_COLUMNS = {"format", "drills", "focus", "program_day_type"}

# Columns that should be integers
INT_COLUMNS = {
    "age_min", "age_max", "default_duration_sec", "default_sets",
    "coach_preference", "rounds", "work_sec", "rest_sec",
}

# CSV stem -> output path (default: data/{stem}.json)
CSV_TO_JSON = {
    "movement": "movement.json",
    "speed_agility": "speed_agility.json",
    "skating_mechanics": "skating_mechanics.json",
    "warmup": "warmup.json",
    "energy_systems": "energy_systems.json",
    "conditioning": "energy_systems.json",  # fallback name
    "stickhandling": "stickhandling.json",
    "shooting": "shooting.json",
    "mobility": "mobility.json",
    "performance": "performance.json",
    "strength": "performance.json",  # fallback
    "circuits": "circuits.json",
    "nhl_combine_results": "nhl_combine/nhl_combine_results.json",
}


def _parse_value(key: str, val: str):
    """Convert CSV cell string to appropriate Python type."""
    if val is None or (isinstance(val, str) and val.strip() in ("", "None", "null")):
        return None
    s = str(val).strip()

    # JSON columns
    if key in JSON_COLUMNS and s:
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                pass
        # Comma-separated list as fallback
        if "," in s and not s.startswith("["):
            return [x.strip() for x in s.split(",") if x.strip()]

    # Boolean
    if s.upper() == "TRUE":
        return True
    if s.upper() == "FALSE":
        return False

    # Integer
    if key in INT_COLUMNS:
        try:
            return int(re.sub(r"[^\d-]", "", s)) if s else None
        except (ValueError, TypeError):
            pass

    # Numeric (age_min, age_max, default_duration_sec, etc.)
    if key in INT_COLUMNS or "sec" in key or "min" in key or "age_" in key or key in ("default_sets", "coach_preference"):
        try:
            return int(float(s)) if s and s.replace(".", "").replace("-", "").isdigit() else None
        except (ValueError, TypeError):
            pass

    return s


def import_csv_to_json(csv_path: Path, json_path: Path) -> int:
    """Read CSV, parse rows, write JSON. Returns row count."""
    if not csv_path.exists():
        return 0
    rows = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out = {}
            for k, v in row.items():
                if not k:
                    continue
                out[k] = _parse_value(k, v)
            rows.append(out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return len(rows)


def main():
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_DIR.exists():
        print(f"  CSV directory not found: {CSV_DIR}")
        return

    for csv_stem, json_rel in CSV_TO_JSON.items():
        csv_path = CSV_DIR / f"{csv_stem}.csv"
        json_path = DATA_DIR / json_rel
        if csv_path.exists():
            n = import_csv_to_json(csv_path, json_path)
            print(f"  {csv_stem}.csv -> {json_rel} ({n} rows)")
        elif csv_stem not in ("conditioning", "strength"):  # skip fallback names
            pass  # optional, don't warn for every possible file

    print(f"\nJSON files written to: {DATA_DIR}")


if __name__ == "__main__":
    main()
