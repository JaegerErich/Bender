"""Add random category ranks (1-5) and XP to the 10 U16 Bears profiles."""
import json
import random
from pathlib import Path

PROFILES_DIR = Path(__file__).resolve().parent.parent / "data" / "profiles"

# bender_leveling.RANK_THRESHOLDS = [0, 150, 400, 800, 1400, 2200, 3200, 4500]
# Rank 1: 0-149, Rank 2: 150-399, Rank 3: 400-799, Rank 4: 800-1399, Rank 5: 1400-2199
RANK_THRESHOLDS = [0, 150, 400, 800, 1400, 2200]

PLAYERS = [
    "bears_marcus_chen",
    "bears_jake_thompson",
    "bears_tyler_williams",
    "bears_ryan_obrien",
    "bears_noah_martinez",
    "bears_cole_anderson",
    "bears_liam_sullivan",
    "bears_ethan_park",
    "bears_dylan_foster",
    "bears_carter_brooks",
]

CATEGORY_KEYS = [
    ("puck_mastery_xp", "puck_mastery_rank"),
    ("skating_xp", "skating_rank"),
    ("performance_xp", "performance_rank"),
    ("conditioning_xp", "conditioning_rank"),
    ("mobility_xp", "mobility_rank"),
]


def xp_for_rank(rank: int) -> int:
    """Return a random XP value in the range for this rank (1-5)."""
    lo = RANK_THRESHOLDS[rank - 1]
    hi = RANK_THRESHOLDS[rank] - 1
    return random.randint(lo, hi)


def main():
    random.seed(42)
    for uid in PLAYERS:
        path = PROFILES_DIR / f"{uid}.json"
        if not path.exists():
            print(f"Skip {path}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for xp_key, rank_key in CATEGORY_KEYS:
            rank = random.randint(1, 5)
            xp = xp_for_rank(rank)
            data[xp_key] = xp
            data[rank_key] = rank
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Updated {path.name}")


if __name__ == "__main__":
    main()
