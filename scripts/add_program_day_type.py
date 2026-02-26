#!/usr/bin/env python3
"""
Add program_day_type to every entry in data/performance.json.
Values: "heavy_leg", "upper_core_stability", "heavy_explosive" (or list for multi-day).
"""
import json
import os
import re

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PERF_PATH = os.path.join(DATA_DIR, "performance.json")


def norm(s):
    return (s or "").strip().lower()


def get(d, key, default=None):
    v = d.get(key, default)
    return v if v is not None else default


def tags_contain(d, *keywords):
    t = norm(get(d, "tags", ""))
    return any(k in t for k in keywords)


def assign_program_day_type(drill):
    """Return a list of program_day_type values for this drill."""
    out = set()
    pr = norm(get(drill, "primary_region", ""))
    sf = norm(get(drill, "strength_focus", ""))
    lr = norm(get(drill, "lift_role", ""))
    mp = norm(get(drill, "movement_pattern", ""))
    uni = norm(get(drill, "unilateral", "")) in ("true", "1", "yes")
    load = norm(get(drill, "load_type", ""))
    cns = norm(get(drill, "cns_load", ""))
    fc = norm(get(drill, "fatigue_cost", ""))

    # ---- HEAVY LEG ----
    if pr == "lower":
        if lr == "primary" and sf == "max_strength" and not uni:
            out.add("heavy_leg")
        if uni and lr in ("primary", "secondary") and sf == "max_strength":
            out.add("heavy_leg")
        if mp == "hinge" and sf in ("hypertrophy", "strength"):
            out.add("heavy_leg")
        if tags_contain(drill, "adductor", "frontal_plane", "lateral_strength"):
            out.add("heavy_leg")
        if tags_contain(drill, "isometric", "deceleration", "hard_stop"):
            out.add("heavy_leg")
        if tags_contain(drill, "lateral_lunge", "goblet", "cossack"):
            out.add("heavy_leg")
        if tags_contain(drill, "posterior_chain", "RDL", "hip_thrust", "glutes", "hamstrings"):
            out.add("heavy_leg")
        if tags_contain(drill, "barbell_squat", "front_squat", "trap_bar", "bulgarian_split", "step_up", "single_leg"):
            out.add("heavy_leg")
        if tags_contain(drill, "wall sit", "knee_stability") or "single_leg" in norm(get(drill, "name", "")) and "wall" in norm(get(drill, "name", "")):
            out.add("heavy_leg")

    # ---- UPPER CORE STABILITY ----
    if tags_contain(drill, "scap_control", "rotator_cuff", "serratus") and fc == "low":
        out.add("upper_core_stability")
    if pr == "upper" and mp == "push" and sf in ("stability", "hypertrophy"):
        out.add("upper_core_stability")
    if pr == "upper" and mp == "pull" and tags_contain(drill, "posture", "scap_control"):
        out.add("upper_core_stability")
    if mp == "anti_rotation" or tags_contain(drill, "anti_rotation", "anti_extension"):
        out.add("upper_core_stability")
    if tags_contain(drill, "controlled_rotation", "core_control") and sf != "power":
        out.add("upper_core_stability")
    if mp == "rotation" and "chop" in norm(get(drill, "name", "")) or tags_contain(drill, "diagonal_lift", "rotational_strength"):
        out.add("upper_core_stability")
    if mp == "carry" or tags_contain(drill, "carry"):
        out.add("upper_core_stability")
    if pr == "core" and sf == "stability":
        out.add("upper_core_stability")
    if tags_contain(drill, "landmine_press", "landmine_row", "pullup", "inverted_row", "Pallof"):
        out.add("upper_core_stability")

    # ---- HEAVY EXPLOSIVE (Power day) ----
    if sf == "power":
        out.add("heavy_explosive")
    if tags_contain(drill, "plyometric", "vertical_jump", "lateral_power", "horizontal_jump", "split_stance"):
        out.add("heavy_explosive")
    if tags_contain(drill, "elastic", "reactive") and load == "bodyweight" and fc != "high":
        out.add("heavy_explosive")
    if tags_contain(drill, "box_jump", "trap_bar_jump", "push_press", "triple_extension"):
        out.add("heavy_explosive")
    if tags_contain(drill, "rotational", "med_ball", "scoop", "slam") and sf == "power":
        out.add("heavy_explosive")
    if tags_contain(drill, "skating_specific", "lateral_strength", "unilateral_power") and (uni or "lateral" in norm(get(drill, "tags", ""))):
        out.add("heavy_explosive")
    if lr == "primary" and cns == "high" and sf == "power":
        out.add("heavy_explosive")

    # If nothing matched, allow all three so templates can still pick by structure
    if not out:
        out = {"heavy_leg", "upper_core_stability", "heavy_explosive"}

    return list(sorted(out))


def main():
    with open(PERF_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    for d in data:
        d["program_day_type"] = assign_program_day_type(d)
    with open(PERF_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Updated {len(data)} drills with program_day_type.")


if __name__ == "__main__":
    main()
