"""Seed 10 U16 Bears players with random demo data. Run from Bender repo root."""
import json
import random
from datetime import date, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROFILES_DIR = DATA_DIR / "profiles"
TEAMS_PATH = DATA_DIR / "teams.json"

# 10 U16 Bears players: (user_id, display_name, target_level, position)
PLAYERS = [
    ("bears_marcus_chen", "Marcus Chen", 2, "C"),
    ("bears_jake_thompson", "Jake Thompson", 1, "RW"),
    ("bears_tyler_williams", "Tyler Williams", 4, "D"),
    ("bears_ryan_obrien", "Ryan O'Brien", 5, "C"),
    ("bears_noah_martinez", "Noah Martinez", 2, "LW"),
    ("bears_cole_anderson", "Cole Anderson", 3, "RW"),
    ("bears_liam_sullivan", "Liam Sullivan", 4, "D"),
    ("bears_ethan_park", "Ethan Park", 1, "LW"),
    ("bears_dylan_foster", "Dylan Foster", 3, "C"),
    ("bears_carter_brooks", "Carter Brooks", 5, "D"),
]

# XP thresholds for levels 1-5
LEVEL_XP = [0, 100, 250, 450, 700]

MODES = [
    "performance",
    "skating_mechanics",
    "stickhandling",
    "shooting",
    "energy_systems",
    "mobility",
]

# Same auth as erich_jaeger (demo only)
AUTH = {
    "password_salt": "a4f44bea8bdac1de7089930365aaeda5",
    "password_hash": "c6669de43bff8c1715cdc70c80c5b0155ccd278878609f2d4b1c7de209b8903c",
}


def xp_for_workout(mode: str, minutes: int) -> int:
    """Rough XP per workout (~15-40 depending on mode/duration)."""
    base = {"performance": 25, "skating_mechanics": 22, "stickhandling": 20, "shooting": 20, "energy_systems": 12, "mobility": 12}.get(mode, 15)
    return max(5, int(base * (minutes / 45)))


def generate_completion_history(target_xp: int) -> list:
    """Generate completion_history entries that sum to ~target_xp."""
    hist = []
    xp = 0
    today = date.today()
    for _ in range(50):
        if xp >= target_xp:
            break
        days_ago = random.randint(0, 28)
        d = today - timedelta(days=days_ago)
        mode = random.choice(MODES)
        mins = random.choice([20, 25, 30, 35, 40, 45])
        x = xp_for_workout(mode, mins)
        hist.append({
            "date": d.isoformat(),
            "completed_at": f"{d}T18:{random.randint(0, 59):02d}:00",
            "mode": mode,
            "minutes": mins,
        })
        xp += x
    return hist


def compute_stats(hist: list) -> dict:
    """Compute private_victory_stats from completion_history."""
    stats = {
        "stickhandling_hours": 0, "shots": 0, "gym_hours": 0,
        "skating_hours": 0, "conditioning_hours": 0, "mobility_hours": 0,
        "completions_count": len(hist),
    }
    for e in hist:
        mins = e.get("minutes", 0) / 60
        mode = (e.get("mode") or "").lower()
        if mode in ("stickhandling", "shooting", "skills_only"):
            stats["stickhandling_hours"] += mins * 0.5
            stats["shots"] += int(mins * 10)
        elif mode == "performance":
            stats["gym_hours"] += mins
        elif mode == "skating_mechanics":
            stats["skating_hours"] += mins
        elif mode == "energy_systems":
            stats["conditioning_hours"] += mins
        elif mode == "mobility":
            stats["mobility_hours"] += mins
    return stats


def main():
    random.seed(42)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    team_id = "t_bears_u16_001"
    members = []

    for user_id, display_name, level, position in PLAYERS:
        target_xp = random.randint(
            LEVEL_XP[level - 1],
            (LEVEL_XP[level] - 1) if level < 5 else 999
        )
        hist = generate_completion_history(target_xp)
        stats = compute_stats(hist)

        prof = {
            "user_id": user_id,
            "display_name": display_name,
            "age": random.randint(15, 16),
            "position": position,
            "equipment": ["None"],
            "equipment_setup_done": True,
            **AUTH,
            "created_at": "2026-02-15T10:00:00",
            "updated_at": "2026-03-09T12:00:00",
            "completion_history": hist,
            "private_victory_stats": stats,
            "total_xp": sum(xp_for_workout(e.get("mode", ""), e.get("minutes", 0)) for e in hist),
            "level": level,
            "level_title": ["Initiate", "Rookie", "Prospect", "Practice Player", "Grinder"][level - 1],
            "total_workouts": len(hist),
            "workout_streak": random.randint(0, min(5, len(hist))),
            "longest_streak": random.randint(1, min(10, len(hist))),
            "player_teams_cache": [{"team_id": team_id, "team_name": "U16 Bears", "role": "player"}],
        }
        # Cap total_xp to keep level <= 5
        if prof["total_xp"] >= 1000:
            prof["total_xp"] = random.randint(700, 999)
            prof["level"] = 5
            prof["level_title"] = "Grinder"

        path = PROFILES_DIR / f"{user_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(prof, f, indent=2)
        print(f"Created {path}")

        members.append({
            "user_id": user_id,
            "role": "player",
            "joined_at": "2026-02-15T10:00:00.000000",
        })

    # Update teams.json
    teams = json.loads(TEAMS_PATH.read_text(encoding="utf-8"))
    for t in teams:
        if t.get("team_id") == team_id:
            existing = {m["user_id"] for m in t.get("members", [])}
            for m in members:
                if m["user_id"] not in existing:
                    t["members"].append(m)
            break
    with open(TEAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(teams, f, indent=2)
    print(f"Updated {TEAMS_PATH}")


if __name__ == "__main__":
    main()
