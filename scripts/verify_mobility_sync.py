"""
Verify that mobility.csv and mobility.json match after syncing from Google Sheets.

Checks:
- Same set of IDs
- Matching name
- Matching equipment
- Matching status
- Matching step_by_step + coaching_cues
- video_url present + looks like a URL (starts with http)

Run from repo root:
  python scripts/verify_mobility_sync.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "data" / "csv_exports" / "mobility.csv"
JSON_PATH = REPO_ROOT / "data" / "mobility.json"


def norm(s: object) -> str:
    if s is None:
        return ""
    return str(s).strip()


def noneish(s: object) -> str:
    """Normalize CSV/JSON 'None'/'null'/'empty' variants to '' for comparison."""
    v = norm(s)
    if not v:
        return ""
    if v.lower() in ("none", "null", "—", "-"):
        return ""
    return v

def canon_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", norm(s))


def normalize_url(u: object) -> str:
    u = noneish(u)
    if not u:
        return ""
    return u


def main() -> int:
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing CSV: {CSV_PATH}")
    if not JSON_PATH.exists():
        raise SystemExit(f"Missing JSON: {JSON_PATH}")

    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        csv_rows = list(csv.DictReader(f))

    with JSON_PATH.open("r", encoding="utf-8") as f:
        json_rows = json.load(f)

    if not isinstance(json_rows, list):
        raise SystemExit("mobility.json is not a list")

    by_id_csv = {norm(r.get("id")): r for r in csv_rows if norm(r.get("id"))}
    by_id_js = {norm(d.get("id")): d for d in json_rows if isinstance(d, dict) and norm(d.get("id"))}

    csv_ids = set(by_id_csv.keys())
    js_ids = set(by_id_js.keys())

    missing_in_json = sorted(csv_ids - js_ids)
    missing_in_csv = sorted(js_ids - csv_ids)

    fields_to_check = ["name", "equipment", "status", "step_by_step", "coaching_cues", "video_url"]
    mismatches: list[tuple[str, str, str, str]] = []  # (id, field, csv, json)

    for drill_id in sorted(csv_ids & js_ids):
        c = by_id_csv[drill_id]
        j = by_id_js[drill_id]

        for field in fields_to_check:
            cv = c.get(field)
            jv = j.get(field)

            if field == "name":
                cvn = canon_spaces(noneish(cv))
                jvn = canon_spaces(noneish(jv))
            elif field in ("step_by_step", "coaching_cues"):
                cvn = canon_spaces(noneish(cv))
                jvn = canon_spaces(noneish(jv))
            else:
                cvn = noneish(cv)
                jvn = noneish(jv)

            if cvn != jvn:
                mismatches.append((drill_id, field, norm(cv), norm(jv)))

    # Video expectations: every remaining mobility drill should have a valid URL
    no_video_ids: list[str] = []
    bad_url_ids: list[tuple[str, str]] = []
    for drill_id, d in by_id_js.items():
        u = normalize_url(d.get("video_url"))
        if not u:
            no_video_ids.append(drill_id)
        elif not u.lower().startswith("http"):
            bad_url_ids.append((drill_id, u))

    # Reporting
    print(f"mobility.csv rows: {len(csv_rows)}")
    print(f"mobility.json rows: {len(json_rows)}")
    print(f"missing_in_json: {len(missing_in_json)}")
    if missing_in_json:
        print("  sample:", missing_in_json[:10])
    print(f"missing_in_csv: {len(missing_in_csv)}")
    if missing_in_csv:
        print("  sample:", missing_in_csv[:10])

    print(f"field mismatches (csv vs json): {len(mismatches)}")
    if mismatches:
        for row in mismatches[:10]:
            drill_id, field, cv, jv = row
            print(f"  mismatch id={drill_id} field={field}\n    csv={cv}\n    json={jv}")

    print(f"no video_url in json: {len(no_video_ids)}")
    if no_video_ids:
        print("  sample:", no_video_ids[:10])
    print(f"bad (non-http) video_url in json: {len(bad_url_ids)}")
    if bad_url_ids:
        for drill_id, u in bad_url_ids[:10]:
            print(f"  bad url id={drill_id}: {u}")

    # Exit code: non-zero if mismatches or video missing
    if missing_in_json or missing_in_csv or mismatches or no_video_ids or bad_url_ids:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

