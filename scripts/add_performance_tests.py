"""Add random performance_tests to the 10 U16 Bears profiles."""
import json
import random
from pathlib import Path

PROFILES_DIR = Path(__file__).resolve().parent.parent / "data" / "profiles"

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


def main():
    random.seed(99)
    for uid in PLAYERS:
        path = PROFILES_DIR / f"{uid}.json"
        if not path.exists():
            print(f"Skip {path}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        # Vertical jump: inches, U16 typical 18–26
        vj = round(random.uniform(18.5, 26.0), 1)
        # 5-10-5: seconds, lower is better, typical 4.3–5.4
        agility = round(random.uniform(4.25, 5.45), 2)
        # Shooting / stickhandling / conditioning: 0–100
        shooting = random.randint(35, 98)
        stick = random.randint(40, 95)
        cond = random.randint(38, 97)
        data["performance_tests"] = {
            "vertical_jump": f"{vj} in",
            "agility_5_10_5": f"{agility} s",
            "shooting_tests": str(shooting),
            "stickhandling_tests": str(stick),
            "conditioning_test": str(cond),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Updated {path.name}")


if __name__ == "__main__":
    main()
