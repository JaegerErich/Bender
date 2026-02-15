#!/usr/bin/env python3
"""
Fetch/merge NHL Scouting Combine fitness data into data/nhl_combine/nhl_combine_results.json.
Used by Bender and MotionApp (shared dataset).

Fields: name, year, height_in, wingspan_in, standing_long_jump_in, vertical_jump_in,
       vertical_jump_force_plate_in, pull_ups, bench_press, bench_power_watts_per_kg,
       pro_agility_left_sec, pro_agility_right_sec, y_balance_composite_cm, source.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# 2025 NHL Combine – top 10 per test (Bleacher Report / NHL PDF summary)
# https://bleacherreport.com/articles/25203255-nhl-combine-2025-full-results-measurements-highlights-and-top-prospects
# ---------------------------------------------------------------------------
TOP10_2025 = {
    "standing_long_jump_in": [
        ("William Horcoff", 124.8),
        ("William Belle", 118.3),
        ("Sean Barnhill", 118.0),
        ("Lynden Lakovic", 117.0),
        ("William Moore", 116.0),
        ("Carter Klippenstein", 115.5),
        ("Maceo Phillips", 114.3),
        ("Roger McQueen", 112.8),
        ("Max Psenicka", 112.5),
        ("Porter Martone", 112.0),
    ],
    "vertical_jump_in": [
        ("Ryker Lee", 25.67),
        ("Kieren Dervin", 25.1),
        ("Malcolm Spence", 24.51),
        ("William Belle", 24.45),
        ("Bill Zonnon", 24.36),
        ("Milton Gastrin", 24.01),
        ("Haoxi (Simon) Wang", 24.01),
        ("Maceo Phillips", 23.34),
        ("Sean Barnhill", 23.07),
        ("Asher Barnett", 23.01),
    ],
    "vertical_jump_force_plate_in": [
        ("William Horcoff", 22.84),
        ("William Belle", 21.92),
        ("Haoxi (Simon) Wang", 21.25),
        ("Maceo Phillips", 21.25),
        ("Kieren Dervin", 21.03),
        ("Carlos Handel", 20.87),
        ("Milton Gastrin", 20.56),
        ("Sean Barnhill", 20.48),
        ("Braeden Cootes", 20.08),
        ("Sascha Boumedienne", 20.05),
    ],
    "pull_ups": [
        ("Bill Zonnon", 16),
        ("Shane Vansaghi", 15),
        ("Braeden Cootes", 15),
        ("Eric Nilson", 15),
        ("Sascha Boumedienne", 14),
        ("Cameron Reid", 14),
        ("Cole McKinney", 14),
        ("Kieren Dervin", 14),
        ("David Lewandowski", 14),
        ("Carter Klippenstein", 14),
    ],
    "bench_power_watts_per_kg": [
        ("Cameron Schmidt", 7.82),
        ("Adam Benak", 7.75),
        ("Maceo Phillips", 7.64),
        ("Cole McKinney", 7.62),
        ("Shane Vansaghi", 7.59),
        ("Kieren Dervin", 7.56),
        ("Daniil Prokhorov", 7.52),
        ("David Bedkovski", 7.41),
        ("Braeden Cootes", 7.23),
        ("Carter Klippenstein", 7.19),
    ],
    "pro_agility_left_sec": [
        ("Milton Gastrin", 4.12),
        ("William Moore", 4.19),
        ("Sean Barnhill", 4.23),
        ("William Horcoff", 4.23),
        ("Charlie Trethewey", 4.24),
        ("Jack Nesbitt", 4.25),
        ("Jakob Ihs-Wozniak", 4.26),
        ("Benjamin Kindel", 4.27),
        ("Carter Klippenstein", 4.32),
        ("Arvid Drott", 4.33),
    ],
    "pro_agility_right_sec": [
        ("Carter Klippenstein", 4.2),
        ("Charlie Trethewey", 4.21),
        ("Cameron Schmidt", 4.21),
        ("Sean Barnhill", 4.23),
        ("Milton Gastrin", 4.24),
        ("Cole Reschny", 4.28),
        ("William Horcoff", 4.29),
        ("Peyton Kettles", 4.29),
        ("Arvid Drott", 4.33),
        ("Jacob Rombach", 4.35),
    ],
    "wingspan_in": [
        ("Haoxi (Simon) Wang", 82.25),
        ("Carter Amico", 82.00),
        ("Maceo Phillips", 82.00),
        ("Jacob Rombach", 81.00),
        ("Jack Nesbitt", 80.00),
        ("Blake Fiddler", 79.75),
        ("William Moore", 79.75),
        ("Aleksei Medvedev", 79.50),
        ("Hayden Paupanekis", 79.50),
        ("Vaclav Nestrasil", 79.25),
    ],
}

# Height from combine notes (optional)
HEIGHT_2025 = [
    ("Roger McQueen", 77.25),   # 6'5.25"
    ("James Hagens", 70.5),     # 5'10.5"
]


def _merge_top10_into_players(year: int, top10: dict, height_list: list, source: str) -> list[dict]:
    """Merge per-test top-10 lists into one record per player."""
    players: dict[str, dict] = defaultdict(lambda: {"name": "", "year": year, "source": source})
    for field, pairs in top10.items():
        for name, value in pairs:
            players[name]["name"] = name
            players[name][field] = value
    for name, height_in in height_list:
        players[name]["name"] = name
        players[name]["height_in"] = height_in
    out = []
    for name, rec in sorted(players.items(), key=lambda x: x[0]):
        if any(k not in ("name", "year", "source") and rec.get(k) is not None for k in rec):
            out.append(rec)
    return out


def build_2025() -> list[dict]:
    return _merge_top10_into_players(2025, TOP10_2025, HEIGHT_2025, "nhl_combine_2025")


def build_all_years() -> list[dict]:
    """Build records for 2015–2019, 2022–2025 (2020–2021 combine cancelled)."""
    _script_dir = Path(__file__).resolve().parent
    if str(_script_dir) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(_script_dir))
    import nhl_combine_data_sources  # noqa: PLC0415

    out: list[dict] = []
    for year, top10, height_list, source in nhl_combine_data_sources.ALL_YEARS_DATA:
        out.extend(_merge_top10_into_players(year, top10, height_list, source))
    out.extend(build_2025())
    return out


def load_nhl_combine_results(data_dir: Path | None = None) -> list[dict]:
    """Load nhl_combine_results.json from data/nhl_combine (Bender) or given dir."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data" / "nhl_combine"
    path = data_dir / "nhl_combine_results.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    out_dir = repo / "data" / "nhl_combine"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "nhl_combine_results.json"

    all_records = build_all_years()
    by_key: dict[tuple[int, str], dict] = {}
    for rec in all_records:
        key = (rec["year"], rec["name"])
        if key not in by_key:
            by_key[key] = rec
        else:
            for k, v in rec.items():
                if v is not None and by_key[key].get(k) is None:
                    by_key[key][k] = v

    result = sorted(by_key.values(), key=lambda p: (-p["year"], p["name"]))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    years = sorted({p["year"] for p in result}, reverse=True)
    print(f"Wrote {len(result)} player records ({min(years)}–{max(years)}) to {out_path}")


if __name__ == "__main__":
    main()
