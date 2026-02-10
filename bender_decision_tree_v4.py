#!/usr/bin/env python3
"""BENDER: Decision Tree (CLI) — V4 (layers 1–5 compatible)

This CLI gathers user intent and calls generate_session() from bender_generate_v8_1.py.

Key additions:
- strength emphasis (power/strength/hypertrophy/recovery)
- skate_within_24h flag (Layer 4 fatigue guardrails)
- post-lift conditioning y/n + modality (Layer 2/5)
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple, List, Dict, Any

# Import generator from same folder
from bender_generate_v8_1 import load_all_data, generate_session

SESSION_MODES = [
    "skills_only",
    "shooting",
    "stickhandling",
    "performance",
    "energy_systems",
    "skating_mechanics",
    "mobility",
]

def _ask_int(prompt: str, default: int, min_v: int = 1, max_v: int = 999) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            v = int(raw)
            if v < min_v or v > max_v:
                raise ValueError()
            return v
        except Exception:
            print(f"Please enter a number between {min_v} and {max_v}.")


def _ask_choice(prompt: str, options: List[str], default_idx: int = 1) -> int:
    # returns 1-based selection index
    print(prompt)
    for i, opt in enumerate(options, start=1):
        print(f"  {i}) {opt}")
    while True:
        raw = input(f"Choose 1-{len(options)} [{default_idx}]: ").strip()
        if raw == "":
            return default_idx
        try:
            v = int(raw)
            if 1 <= v <= len(options):
                return v
        except Exception:
            pass
        print("Invalid choice.")


def main():
    print("\n=== BENDER: Decision Tree (V4) ===\n")

    age = _ask_int("Player age", 14, 6, 60)
    session_len = _ask_int("Session length in minutes", 45, 15, 180)
    athlete_id = input("Athlete ID (optional, Enter=default): ").strip() or "default"

    focus_idx = _ask_choice(
        "What should today focus on?",
        [
            "Puck Mastery",
            "Performance",
            "Skating Mechanics",
            "Energy Systems",
            "Mobility & Recovery",
        ],
        default_idx=1,
    )

    focus = None
    session_mode = "skills_only"  # default; overridden per branch
    strength_emphasis = "strength"
    strength_day_type = None
    strength_full_gym = False
    include_post_lift_conditioning = None
    post_lift_conditioning_type = None
    skate_within_24h = False

    shooting_shots = None
    stickhandling_min = None
    shooting_min = None

    # 1) Puck Mastery
    if focus_idx == 1:
        sub = _ask_choice(
            "Puck Mastery — focus?",
            ["Shooting", "Stickhandling", "Both"],
            default_idx=3,
        )
        # Route to the correct engine mode (prevents non-skill choices from defaulting to skills output)
        if sub == 1:
            session_mode = "shooting"
            focus = None
        elif sub == 2:
            session_mode = "stickhandling"
            focus = None
        else:
            # "skills_only" uses a stickhandling+shooting split (shot-volume based shooting)
            session_mode = "skills_only"
            focus = None

    # 2) Performance
    elif focus_idx == 2:
        gym_idx = _ask_choice(
            "Performance — are you at a gym?",
            ["no gym", "gym"],
            default_idx=1,
        )
        strength_full_gym = (gym_idx == 2)
        session_mode = "performance"
        focus = "performance"

        # No-gym: circuits-only (no day-split / skating / conditioning prompts)
        if not strength_full_gym:
            strength_day_type = "leg"                  # safe default; circuits drive selection
            strength_emphasis = "strength"             # safe default
            skate_within_24h = False                   # default (no prompt)
            include_post_lift_conditioning = False     # default (no prompt)
            post_lift_conditioning_type = None

        # Full gym: ask normal performance prompts
        else:
            dt = _ask_choice(
                "Performance — what type of day?",
                ["lower", "upper", "full"],
                default_idx=1,
            )
            strength_day_type = {1: "leg", 2: "upper", 3: "full"}[dt]

            em = _ask_choice(
                "Performance — what are we training today?",
                [
                    "power (explosive speed)",
                    "strength (game strength)",
                    "hypertrophy (strength capacity)",
                    "recovery (less stress)",
                ],
                default_idx=2,
            )
            strength_emphasis = {1: "power", 2: "strength", 3: "hypertrophy", 4: "recovery"}[em]

            sk = _ask_choice(
                "Skating today or within 24 hours?",
                ["no", "yes"],
                default_idx=1,
            )
            skate_within_24h = (sk == 2)

            cond = _ask_choice(
                "Add energy systems work after lifting?",
                ["no", "yes"],
                default_idx=1,
            )
            include_post_lift_conditioning = (cond == 2)

            if include_post_lift_conditioning:
                mod = _ask_choice(
                    "Post-lift energy systems — choose modality:",
                    ["cones/sprints (no equipment)", "bike", "treadmill", "surprise"],
                    default_idx=4,
                )
                # Map to generator expectations
                if mod == 1:
                    post_lift_conditioning_type = "cones"
                elif mod == 2:
                    post_lift_conditioning_type = "bike"
                elif mod == 3:
                    post_lift_conditioning_type = "treadmill"
                else:
                    post_lift_conditioning_type = "surprise"
            else:
                post_lift_conditioning_type = None


    # 3) Skating Mechanics (off-ice: movement pool = speed, agility, skating mechanics)
    elif focus_idx == 3:
        session_mode = "skating_mechanics"
        focus = None

    # 4) Energy Systems
    elif focus_idx == 4:
        session_mode = "energy_systems"
        focus = "energy_systems"
        gym_idx = _ask_choice(
            "Energy Systems — are you at a gym?",
            ["no gym", "gym"],
            default_idx=2,
        )
        if gym_idx == 2:
            mod = _ask_choice(
                "Energy Systems (gym) — choose modality:",
                ["bike", "treadmill", "surprise"],
                default_idx=3,
            )
            focus = {1: "conditioning_bike", 2: "conditioning_treadmill", 3: "conditioning"}[mod]
        else:
            # no-gym: cones/no-equipment only
            focus = "conditioning_cones"

    # 5) Mobility
    else:
        session_mode = "mobility"
        focus = "mobility"

    # Load data from ./data by default
    data = load_all_data("data")

    plan = generate_session(
        data=data,
        age=age,
        seed=0,
        focus=focus,
        session_mode=session_mode,
        session_len_min=session_len,
        athlete_id=athlete_id,
        use_memory=True,
        memory_sessions=7,
        recent_penalty=0.1,
        strength_emphasis=strength_emphasis,
        shooting_shots=shooting_shots,
        stickhandling_min=stickhandling_min,
        shooting_min=shooting_min,
        strength_day_type=strength_day_type,
        strength_full_gym=strength_full_gym,
        include_post_lift_conditioning=include_post_lift_conditioning,
        post_lift_conditioning_type=post_lift_conditioning_type,
        skate_within_24h=skate_within_24h,
    )

    print(plan)


if __name__ == "__main__":
    main()
