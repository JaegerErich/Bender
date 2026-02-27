"""
Admin Plan Builder: multi-week workout plans for Bender.
Only exposed to admin users (see ADMIN_USERS) via UI gating.
Templates for 3–7 days/week; progression notes; optional drill generation with variety.
Age-based: foundation (<=12) uses templates that always include skating + puck mastery.
"""
from datetime import date, timedelta
from typing import Any, Callable, Optional

# Display label for each mode (Bible App style: Devotional = Performance, etc.)
MODE_DISPLAY_LABELS: dict[str, str] = {
    "performance": "Performance",
    "skating_mechanics": "Skating Mechanics",
    "skills_only": "Puck Mastery",
    "energy_systems": "Conditioning",
    "mobility": "Mobility/Recovery",
}


def _determine_stage(age: int) -> str:
    """Foundation <=12, development 13-15, performance 16-18, advanced 19+."""
    if age <= 12:
        return "foundation"
    if age <= 15:
        return "development"
    if age <= 18:
        return "performance"
    return "advanced"

# Week templates: day_index (0-based) -> list of category focus strings
# Each day has PRIMARY + optional SECONDARY + Mobility/Recovery as warmup/cooldown
WEEK_TEMPLATES: dict[int, list[list[str]]] = {
    3: [
        ["Performance (Strength Lower/Full)", "Skating Mechanics (accel/plyo)", "light Puck Mastery"],
        ["Puck Mastery (primary)", "Conditioning (intervals)", "Mobility/Recovery"],
        ["Performance (Strength Upper/Full)", "Skating Mechanics (agility/footwork)", "Puck Mastery (hands/shooting)"],
    ],
    4: [
        ["Performance (Lower Strength)", "Skating Mechanics (accel/plyo)", "light Puck Mastery"],
        ["Puck Mastery (technical)", "Conditioning (intervals)", "Mobility/Recovery"],
        ["Performance (Upper Strength)", "Skating Mechanics (agility/footwork)", "Puck Mastery (hands)"],
        ["Puck Mastery (reactive/game-like)", "Conditioning (repeat sprints)", "Mobility/Recovery"],
    ],
    5: [
        ["Performance (Lower Strength)", "Skating Mechanics (accel/plyo)", "light Puck Mastery"],
        ["Puck Mastery (technical)", "Conditioning (intervals)", "Mobility/Recovery"],
        ["Performance (Upper Strength)", "Skating Mechanics (agility/footwork)", "Puck Mastery"],
        ["Puck Mastery (reactive/game-like)", "Conditioning (repeat sprints)", "Mobility/Recovery"],
        ["Performance (Power Day)", "Skating Mechanics (bounds/starts)", "light Puck Mastery"],
    ],
    6: [
        ["Performance (Lower Strength)", "Skating Mechanics (accel/plyo)", "light Puck Mastery"],
        ["Puck Mastery (technical)", "Conditioning (intervals)", "Mobility/Recovery"],
        ["Performance (Upper Strength)", "Skating Mechanics (agility/footwork)", "Puck Mastery"],
        ["Puck Mastery (reactive/game-like)", "Conditioning (repeat sprints)", "Mobility/Recovery"],
        ["Performance (Power Day)", "Skating Mechanics (bounds/starts)", "light Puck Mastery"],
        ["Puck Mastery (small-area/competitive)", "optional Conditioning (short finisher)", "Mobility/Recovery"],
    ],
    7: [
        ["Performance (Lower Strength)", "Skating Mechanics (accel/plyo)", "light Puck Mastery"],
        ["Puck Mastery (technical)", "Conditioning (intervals)", "Mobility/Recovery"],
        ["Performance (Upper Strength)", "Skating Mechanics (agility/footwork)", "Puck Mastery"],
        ["Puck Mastery (reactive/game-like)", "Conditioning (repeat sprints)", "Mobility/Recovery"],
        ["Performance (Power Day)", "Skating Mechanics (bounds/starts)", "light Puck Mastery"],
        ["Puck Mastery (competitive)", "optional Conditioning finisher", "Mobility/Recovery"],
        ["Mobility/Recovery ONLY (longer recovery session)"],
    ],
}

# Foundation (age <=12): every week includes skating mechanics + puck mastery; no Heavy Leg/Explosive.
# Performance days become "Foundation (Skating + Puck + Strength)" so generator returns youth template.
FOUNDATION_WEEK_TEMPLATES: dict[int, list[list[str]]] = {
    3: [
        ["Skating Mechanics (accel/plyo)", "Foundation (Skating + Puck + Strength)", "Puck Mastery"],
        ["Puck Mastery (technical)", "Conditioning (intervals)", "Mobility/Recovery"],
        ["Skating Mechanics (agility/footwork)", "Foundation (Skating + Puck + Strength)", "Puck Mastery (hands)"],
    ],
    4: [
        ["Skating Mechanics (accel/plyo)", "Foundation (Skating + Puck + Strength)", "light Puck Mastery"],
        ["Puck Mastery (technical)", "Conditioning (intervals)", "Mobility/Recovery"],
        ["Skating Mechanics (agility/footwork)", "Foundation (Skating + Puck + Strength)", "Puck Mastery"],
        ["Puck Mastery (reactive/game-like)", "Conditioning (repeat sprints)", "Mobility/Recovery"],
    ],
    5: [
        ["Skating Mechanics (accel/plyo)", "Foundation (Skating + Puck + Strength)", "light Puck Mastery"],
        ["Puck Mastery (technical)", "Conditioning (intervals)", "Mobility/Recovery"],
        ["Skating Mechanics (agility/footwork)", "Foundation (Skating + Puck + Strength)", "Puck Mastery"],
        ["Puck Mastery (reactive/game-like)", "Conditioning (repeat sprints)", "Mobility/Recovery"],
        ["Skating Mechanics (bounds/starts)", "Foundation (Skating + Puck + Strength)", "light Puck Mastery"],
    ],
    6: [
        ["Skating Mechanics (accel/plyo)", "Foundation (Skating + Puck + Strength)", "light Puck Mastery"],
        ["Puck Mastery (technical)", "Conditioning (intervals)", "Mobility/Recovery"],
        ["Skating Mechanics (agility/footwork)", "Foundation (Skating + Puck + Strength)", "Puck Mastery"],
        ["Puck Mastery (reactive/game-like)", "Conditioning (repeat sprints)", "Mobility/Recovery"],
        ["Skating Mechanics (bounds/starts)", "Foundation (Skating + Puck + Strength)", "light Puck Mastery"],
        ["Puck Mastery (small-area/competitive)", "optional Conditioning (short finisher)", "Mobility/Recovery"],
    ],
    7: [
        ["Skating Mechanics (accel/plyo)", "Foundation (Skating + Puck + Strength)", "light Puck Mastery"],
        ["Puck Mastery (technical)", "Conditioning (intervals)", "Mobility/Recovery"],
        ["Skating Mechanics (agility/footwork)", "Foundation (Skating + Puck + Strength)", "Puck Mastery"],
        ["Puck Mastery (reactive/game-like)", "Conditioning (repeat sprints)", "Mobility/Recovery"],
        ["Skating Mechanics (bounds/starts)", "Foundation (Skating + Puck + Strength)", "light Puck Mastery"],
        ["Puck Mastery (competitive)", "optional Conditioning finisher", "Mobility/Recovery"],
        ["Mobility/Recovery ONLY (longer recovery session)"],
    ],
}


ADMIN_USERS = {"Erich Jaeger", "Austin Azzinnaro"}


def is_admin_user(display_name: str | None) -> bool:
    """Exact match, case-sensitive. Admin users see Plan Builder tab."""
    return display_name in ADMIN_USERS


def get_progression_note(week_index: int) -> str:
    """Week 0-based. Weeks 1–2: base; 3–4: +1 set; every 4th: deload."""
    if (week_index + 1) % 4 == 0:
        return "Deload: reduce volume ~20–30%"
    if week_index >= 2:
        return "+1 set on main strength lifts or +1 round on circuits"
    return "Base volume (normal)"


def get_template_for_days(days_per_week: int, age: Optional[int] = None) -> list[list[str]]:
    """Return day templates for given days/week. If age <= 12 use foundation templates (skating + puck every week)."""
    d = max(3, min(7, int(days_per_week)))
    if age is not None:
        try:
            age_int = int(age)
            if _determine_stage(age_int) == "foundation":
                return FOUNDATION_WEEK_TEMPLATES.get(d, FOUNDATION_WEEK_TEMPLATES[3])
        except (TypeError, ValueError):
            pass
    return WEEK_TEMPLATES.get(d, WEEK_TEMPLATES[3])


def generate_plan(
    weeks: int,
    days_per_week: int,
    start_date_opt: date | None = None,
    age: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Generate a multi-week plan. Returns list of week dicts:
    [{"week": 1, "days": [...], "progression_note": "..."}, ...]
    If age <= 12, uses foundation templates (skating + puck mastery every week; no heavy leg/explosive).
    """
    weeks = max(1, min(16, int(weeks)))
    template = get_template_for_days(days_per_week, age)
    plan: list[dict[str, Any]] = []
    start = start_date_opt or date.today()

    for w in range(weeks):
        week_start = start + timedelta(days=7 * w)
        days_out: list[dict[str, Any]] = []
        for d_idx, focus_list in enumerate(template):
            day_date = week_start + timedelta(days=d_idx)
            days_out.append({
                "day_num": d_idx + 1,
                "date": day_date,
                "focus": focus_list,
            })
        plan.append({
            "week": w + 1,
            "week_start": week_start,
            "days": days_out,
            "progression_note": get_progression_note(w),
        })
    return plan


def parse_focus_to_engine_params(focus_str: str) -> dict[str, Any] | None:
    """
    Map plan focus string to engine params (mode, focus, strength_day_type, session_len_min, etc.).
    Returns None if focus should be skipped (e.g. optional Conditioning).
    """
    s = (focus_str or "").strip().lower()
    if not s:
        return None
    # Skip optional conditioning
    if "optional" in s and "conditioning" in s:
        return None

    out: dict[str, Any] = {"location": "no_gym", "strength_day_type": "full"}

    # Foundation (age <=12): same as performance mode; generator routes to foundation template by age
    if "foundation" in s and ("skating" in s or "puck" in s or "strength" in s):
        out["mode"] = "performance"
        out["location"] = "gym"
        out["strength_day_type"] = "leg"
        out["session_len_min"] = 60
        out["focus"] = None
        return out

    # Performance (plan builder: lifts at least 60 min)
    if "performance" in s:
        out["mode"] = "performance"
        out["location"] = "gym"
        if "lower" in s or "leg" in s:
            out["strength_day_type"] = "leg"
        elif "upper" in s:
            out["strength_day_type"] = "upper"
        elif "power" in s:
            out["strength_day_type"] = "full"
            out["strength_emphasis"] = "power"
        else:
            out["strength_day_type"] = "full"
        out["session_len_min"] = 60
        out["focus"] = None
        return out

    # Skating Mechanics
    if "skating" in s or "mechanics" in s:
        out["mode"] = "skating_mechanics"
        out["session_len_min"] = 12
        if "accel" in s or "plyo" in s:
            out["focus"] = "skating_accel_plyo"
        elif "agility" in s or "footwork" in s:
            out["focus"] = "skating_agility"
        elif "bounds" in s or "starts" in s:
            out["focus"] = "skating_bounds_starts"
        else:
            out["focus"] = None
        return out

    # Puck Mastery
    if "puck" in s or "skills" in s:
        out["mode"] = "skills_only"
        out["session_len_min"] = 25 if "light" in s else 20
        out["focus"] = None
        return out

    # Conditioning
    if "conditioning" in s:
        out["mode"] = "energy_systems"
        out["session_len_min"] = 12 if "short" in s or "finisher" in s else 15
        if "intervals" in s:
            out["focus"] = "conditioning_intervals"
        elif "repeat" in s or "sprint" in s:
            out["focus"] = "conditioning_repeat_sprints"
        elif "cones" in s:
            out["focus"] = "conditioning_cones"
        else:
            out["focus"] = "conditioning"
        return out

    # Mobility/Recovery
    if "mobility" in s or "recovery" in s:
        out["mode"] = "mobility"
        out["session_len_min"] = 25 if "only" in s else 12
        out["focus"] = "mobility"
        return out

    return None


def generate_plan_with_workouts(
    plan: list[dict[str, Any]],
    generate_callback: Callable[[int, str, dict[str, Any]], str],
    base_seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Hydrate plan with full workouts for each focus slot. Uses different seeds per slot
    to reduce duplicates. generate_callback(day_idx, focus_str, params) -> workout_text.
    """
    flat_day_idx = 0
    for w in plan:
        for d in w["days"]:
            focus_items: list[dict[str, Any]] = []
            for f_str in d.get("focus", []):
                params = parse_focus_to_engine_params(f_str)
                if params is None:
                    continue
                seed = base_seed + flat_day_idx * 100 + len(focus_items)
                params["seed"] = seed
                try:
                    workout = generate_callback(flat_day_idx, f_str, params)
                except Exception:
                    workout = "(Workout generation failed)"
                mode_key = params.get("mode", "unknown")
                label = MODE_DISPLAY_LABELS.get(mode_key, mode_key.replace("_", " ").title())
                focus_items.append({
                    "label": label,
                    "mode_key": mode_key,
                    "original": f_str,
                    "workout": workout,
                    "params": {k: v for k, v in params.items() if k != "seed"},
                })
            d["focus_items"] = focus_items
            d["focus"] = [fi["label"] for fi in focus_items]  # keep for backward compat
            flat_day_idx += 1
    return plan


def format_plan_as_text(plan: list[dict[str, Any]]) -> str:
    """Render plan as readable text."""
    lines: list[str] = []
    for w in plan:
        lines.append(f"\n{'='*50}")
        lines.append(f"WEEK {w['week']} ({w['week_start'].strftime('%b %d')} – {w['week_start'] + timedelta(days=6):%b %d})")
        lines.append(f"Progression: {w['progression_note']}")
        lines.append("")
        for d in w["days"]:
            dt_str = d["date"].strftime("%a %b %d")
            lines.append(f"  Day {d['day_num']} ({dt_str})")
            for f in d["focus"]:
                lines.append(f"    • {f}")
            lines.append("")
    return "\n".join(lines).strip()
