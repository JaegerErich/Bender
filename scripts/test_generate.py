#!/usr/bin/env python3
"""
Test that workout generation outputs strength content.
Run: python scripts/test_generate.py
"""
import os
import sys

# Ensure Bender is on path (run from repo root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bender_generate_v8_1 import load_all_data, generate_session

def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    if not os.path.exists(os.path.join(data_dir, "performance.json")):
        print("ERROR: data/performance.json not found")
        print("  CWD:", os.getcwd())
        print("  data_dir:", data_dir)
        sys.exit(1)

    data = load_all_data("data")
    perf = data.get("performance", [])
    if not perf:
        print("ERROR: performance data is empty")
        sys.exit(1)

    out = generate_session(
        data=data,
        age=16,
        seed=42,
        focus=None,
        session_mode="performance",
        session_len_min=45,
        strength_day_type="heavy_leg",
        strength_full_gym=True,
        user_equipment=None,
        use_memory=False,
    )

    has_strength = any(x in out for x in ["PRIMARY", "POSTERIOR", "FRONTAL", "STRENGTH", "ISO", "CIRCUIT"])
    if not has_strength:
        print("ERROR: Output has no strength sections")
        print("Output:", out[:500])
        sys.exit(1)

    print("OK: Workout generated with strength content")
    print("Length:", len(out))
    print()
    print(out[:1500])

if __name__ == "__main__":
    main()
