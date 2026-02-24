"""
Convert all active Bender JSON data files to CSV for Google Sheets import.
Outputs to data/csv_exports/ — one CSV per JSON.
"""
import csv
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = DATA_DIR / "csv_exports"

# Active JSONs used by the engine (canonical names; fallbacks resolved at load)
JSON_FILES = [
    "movement.json",
    "speed_agility.json",
    "skating_mechanics.json",
    "warmup.json",
    "energy_systems.json",   # fallback: conditioning.json
    "stickhandling.json",
    "shooting.json",
    "mobility.json",
    "performance.json",     # fallback: strength.json
    "circuits.json",
]
# NHL combine (in subdir)
NHL_COMBINE = "nhl_combine/nhl_combine_results.json"


def _flatten_value(val):
    """Convert nested dict/list to string for CSV cell."""
    if val is None:
        return ""
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    return str(val)


def _flatten_row(obj, columns=None):
    """Flatten one object to CSV-safe dict."""
    if columns is None:
        columns = list(obj.keys())
    return {c: _flatten_value(obj.get(c)) for c in columns}


def _all_keys(items):
    """Collect all unique keys across items."""
    keys = []
    seen = set()
    for item in items:
        for k in item.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


def export_array_to_csv(items, out_path, columns=None):
    """Export array of objects to CSV."""
    if not items:
        return
    if columns is None:
        columns = _all_keys(items)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for item in items:
            row = _flatten_row(item, columns)
            w.writerow(row)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Export each JSON
    for name in JSON_FILES:
        path = DATA_DIR / name
        if not path.exists():
            # Try fallbacks
            if name == "energy_systems.json":
                path = DATA_DIR / "conditioning.json"
            elif name == "performance.json":
                path = DATA_DIR / "strength.json"
        if not path.exists():
            print(f"  Skip (not found): {name}")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"  Skip (not array): {name}")
            continue
        out_name = path.stem + ".csv"
        export_array_to_csv(data, OUT_DIR / out_name)
        print(f"  Wrote {out_name} ({len(data)} rows)")

    # NHL combine (sparse keys per player)
    nhl_path = DATA_DIR / NHL_COMBINE
    if nhl_path.exists():
        with open(nhl_path, encoding="utf-8") as f:
            players = json.load(f)
        if isinstance(players, list):
            columns = _all_keys(players)
            export_array_to_csv(players, OUT_DIR / "nhl_combine_results.csv", columns)
            print(f"  Wrote nhl_combine_results.csv ({len(players)} rows)")

    print(f"\nCSV files saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
