#!/usr/bin/env python3
"""
Add draft info (team, round, overall pick) to data/nhl_combine/nhl_combine_results.json
by matching (year, player name) to hockey-reference draft data (GitHub CSV).
"""
from __future__ import annotations

import csv
import json
import math
import re
import urllib.request
from pathlib import Path

DRAFT_CSV_URL = "https://raw.githubusercontent.com/octonion/hockey/master/href/csv/draft_picks.csv"
COMBINE_YEARS = (2015, 2016, 2017, 2018, 2019, 2022, 2023, 2024, 2025)


def _normalize_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _name_variants(name: str) -> list[str]:
    """Return list of name variants for matching (normalized)."""
    n = _normalize_name(name)
    out = [n]
    # First Last -> try without middle initial/name
    parts = n.split()
    if len(parts) > 2:
        out.append(parts[0] + " " + parts[-1])
    if len(parts) == 2:
        out.append(parts[0][0] + " " + parts[1])  # F Last
    # Common substitutions (draft vs combine spelling)
    subs = [
        ("mitchell", "mitch"),
        ("zachary", "zach"),
        ("alexander", "alex"),
        ("nicholas", "nick"),
        ("nicholas", "nic"),
        ("michael", "mike"),
        ("matthew", "matt"),
        ("christopher", "chris"),
        ("daniel", "danny"),
        ("william", "will"),
        ("joseph", "joe"),
        ("robert", "rob"),
        ("d'artagnan", "dartagnan"),
    ]
    for old, new in subs:
        if old in n and n not in out:
            out.append(n.replace(old, new))
    return out


def _picks_per_round(year: int) -> int:
    if year <= 2016:
        return 30
    if year <= 2017:
        return 31
    return 32


def build_draft_lookup(csv_path: Path | None = None) -> dict[tuple[int, str], dict]:
    """
    Build (year, normalized_name) -> {draft_team, draft_round, draft_overall}.
    If csv_path is None, fetch from URL.
    """
    if csv_path and csv_path.exists():
        with open(csv_path, encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
    else:
        with urllib.request.urlopen(DRAFT_CSV_URL, timeout=30) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        rows = list(csv.reader(text.splitlines()))
    lookup: dict[tuple[int, str], dict] = {}
    by_year: dict[int, list[tuple[str, str]]] = {y: [] for y in COMBINE_YEARS}
    for row in rows:
        if len(row) < 3:
            continue
        try:
            year = int(row[0])
        except ValueError:
            continue
        if year not in COMBINE_YEARS:
            continue
        team = (row[1] or "").strip()
        player = (row[2] or "").strip()
        if not player or "invalid" in player.lower():
            continue
        by_year[year].append((team, player))
    for year in COMBINE_YEARS:
        picks = by_year.get(year, [])
        ppr = _picks_per_round(year)
        for i, (team, player) in enumerate(picks):
            overall = i + 1
            round_num = (overall - 1) // ppr + 1
            key = (year, _normalize_name(player))
            lookup[key] = {
                "draft_team": team,
                "draft_round": round_num,
                "draft_overall": overall,
            }
            # Also store with "First Last" only (no middle) for matching
            parts = _normalize_name(player).split()
            if len(parts) > 2:
                short = parts[0] + " " + parts[-1]
                if (year, short) not in lookup:
                    lookup[(year, short)] = lookup[key]
    return lookup


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    data_dir = repo / "data" / "nhl_combine"
    json_path = data_dir / "nhl_combine_results.json"
    csv_path = data_dir / "draft_picks.csv"

    print("Building draft lookup...")
    lookup = build_draft_lookup(csv_path)
    print(f"  Loaded {len(lookup)} draft pick keys")

    with open(json_path, encoding="utf-8") as f:
        players = json.load(f)

    matched = 0
    for p in players:
        year = p.get("year")
        name = (p.get("name") or "").strip()
        if not year or not name:
            continue
        info = None
        for variant in _name_variants(name):
            info = lookup.get((year, variant))
            if info:
                break
        if info:
            p["draft_team"] = info["draft_team"]
            p["draft_round"] = info["draft_round"]
            p["draft_overall"] = info["draft_overall"]
            matched += 1

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(players, f, indent=2, ensure_ascii=False)
    print(f"Updated {json_path}: added draft info for {matched} of {len(players)} players.")


if __name__ == "__main__":
    main()
