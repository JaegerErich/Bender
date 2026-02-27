"""
One-click sync: Google Sheets → CSV → JSON.

Usage:
  1. Fetch from Google Sheets (requires sheet published or "Anyone with link can view"):
     python scripts/sync_from_sheets.py --fetch

  2. Use local CSVs only (after you manually download from Sheets):
     python scripts/sync_from_sheets.py --local

  Set SHEET_ID and SHEET_MAP below to match your Google Sheet.
  Your sheet: https://docs.google.com/spreadsheets/d/1zrEcAnd-Yhc6lhEse0f1Q4ZXSHANxZuwmmw68lmLHUE/
"""
import argparse
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# ----- CONFIG: Edit to match your Google Sheet -----
SHEET_ID = "1zrEcAnd-Yhc6lhEse0f1Q4ZXSHANxZuwmmw68lmLHUE"

# Map: sheet tab name (as shown in Google Sheets) -> output CSV filename (without .csv)
# Get tab names from your sheet. Use exact names (case-sensitive).
SHEET_MAP = {
    "performance": "performance",
    "shooting": "shooting",
    "stickhandling": "stickhandling",
    "skating_mechanics": "skating_mechanics",
    "energy_systems": "energy_systems",
    "mobility": "mobility",
    "movement": "movement",
    "speed_agility": "speed_agility",
    "warmup": "warmup",
    "circuits": "circuits",
    "nhl_combine_results": "nhl_combine_results",
}

# ----- Paths -----
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CSV_DIR = REPO_ROOT / "data" / "csv_exports"


def fetch_sheet_as_csv(sheet_id: str, sheet_name: str) -> str | None:
    """Fetch one sheet tab as CSV. Returns content or None on failure."""
    # Use gviz API with sheet= name (works for published sheets)
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={urllib.parse.quote(sheet_name)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Bender/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"    Sheet tab '{sheet_name}' not found (404)")
        else:
            print(f"    HTTP {e.code} for '{sheet_name}'")
        return None
    except urllib.error.URLError as e:
        print(f"    Network error for '{sheet_name}': {e.reason}")
        return None
    except Exception as e:
        print(f"    Error fetching '{sheet_name}': {e}")
        return None


def fetch_all_and_save(sheet_id: str, sheet_map: dict) -> bool:
    """Fetch each sheet tab, save to csv_exports. Returns True if any succeeded."""
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    ok = 0
    for sheet_name, csv_stem in sheet_map.items():
        content = fetch_sheet_as_csv(sheet_id, sheet_name)
        if content:
            path = CSV_DIR / f"{csv_stem}.csv"
            path.write_text(content, encoding="utf-8")
            rows = len(content.strip().split("\n")) - 1 if "\n" in content else 0
            print(f"  Fetched {sheet_name} -> {csv_stem}.csv ({rows} rows)")
            ok += 1
    return ok > 0


def main():
    parser = argparse.ArgumentParser(
        description="Sync Bender data: Google Sheets → CSV → JSON"
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch from Google Sheets first (sheet must be shared/published)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Skip fetch; use existing CSVs in data/csv_exports/",
    )
    args = parser.parse_args()

    # Default: fetch if --fetch, else local
    do_fetch = args.fetch or (not args.local and not args.fetch)
    # If neither flag: default to fetch
    if not args.fetch and not args.local:
        do_fetch = True

    if do_fetch:
        print("Fetching from Google Sheets...")
        if not fetch_all_and_save(SHEET_ID, SHEET_MAP):
            print("\n  No sheets fetched. Is the sheet shared as 'Anyone with the link can view'?")
            print("  Or use --local after manually downloading CSVs to data/csv_exports/")
            return 1

    print("\nConverting CSV -> JSON...")
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "csv_to_json_import.py")],
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        return result.returncode

    print("\nDone. Bender data is up to date.")
    return 0


if __name__ == "__main__":
    exit(main())
