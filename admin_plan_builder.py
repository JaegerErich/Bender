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

# Plan builder: mode keys for frequency/length config (order for UI)
PLAN_MODES: list[str] = [
    "performance",
    "skating_mechanics",
    "skills_only",
    "energy_systems",
    "mobility",
]

# Default session length (min) per mode for plan builder
MODE_SESSION_LEN_DEFAULTS: dict[str, int] = {
    "performance": 60,
    "skating_mechanics": 12,
    "skills_only": 25,
    "energy_systems": 15,
    "mobility": 12,
}

FREQUENCY_OPTIONS: list[str] = [
    "As in plan",
    "1x/week",
    "2x/week",
    "3x/week",
    "4x/week",
    "5x/week",
    "6x/week",
    "7x/week",
]

# Weekday names for day-of-week checkboxes (Mon=0, Sun=6)
WEEKDAY_NAMES: list[str] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Variants per mode for building focus strings (cycled by day index)
PERFORMANCE_VARIANTS: list[str] = [
    "Performance (Lower Strength)",
    "Performance (Upper Strength)",
    "Performance (Power Day)",
]
SKATING_VARIANTS: list[str] = [
    "Skating Mechanics (accel/plyo)",
    "Skating Mechanics (agility/footwork)",
    "Skating Mechanics (bounds/starts)",
]
PUCK_PRIMARY: str = "Puck Mastery (technical)"
PUCK_LIGHT: str = "light Puck Mastery"
CONDITIONING_VARIANTS: list[str] = [
    "Conditioning (intervals)",
    "Conditioning (repeat sprints)",
]
MOBILITY: str = "Mobility/Recovery"
MOBILITY_ONLY: str = "Mobility/Recovery ONLY"


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


def get_template_from_mode_days(
    mode_days: dict[str, set[int]],
    age: Optional[int] = None,
) -> list[list[str]]:
    """
    Build a 7-day week template from day-of-week mode selections.
    mode_days: {mode_key: {0, 2, 4}} for Mon, Wed, Fri (weekday 0=Mon, 6=Sun).
    Returns [focus_list_mon, focus_list_tue, ...] for each weekday 0-6.
    """
    use_foundation = age is not None and _determine_stage(int(age)) == "foundation"

    perf_idx = 0
    skate_idx = 0
    cond_idx = 0
    template: list[list[str]] = []

    for wd in range(7):
        focus_list: list[str] = []
        has_perf = wd in mode_days.get("performance", set())
        has_skate = wd in mode_days.get("skating_mechanics", set())
        has_puck = wd in mode_days.get("skills_only", set())
        has_cond = wd in mode_days.get("energy_systems", set())
        has_mob = wd in mode_days.get("mobility", set())

        if use_foundation:
            if has_perf:
                focus_list.append("Foundation (Skating + Puck + Strength)")
            if has_skate:
                focus_list.append(SKATING_VARIANTS[skate_idx % len(SKATING_VARIANTS)])
                skate_idx += 1
            if has_puck:
                focus_list.append(PUCK_LIGHT if has_perf else "Puck Mastery")
        else:
            if has_perf:
                focus_list.append(PERFORMANCE_VARIANTS[perf_idx % len(PERFORMANCE_VARIANTS)])
                perf_idx += 1
            if has_skate:
                focus_list.append(SKATING_VARIANTS[skate_idx % len(SKATING_VARIANTS)])
                skate_idx += 1
            if has_puck:
                focus_list.append(PUCK_LIGHT if has_perf else PUCK_PRIMARY)
        if has_cond:
            focus_list.append(CONDITIONING_VARIANTS[cond_idx % len(CONDITIONING_VARIANTS)])
            cond_idx += 1
        if has_mob:
            focus_list.append(MOBILITY_ONLY if not focus_list else MOBILITY)
        if not focus_list:
            focus_list.append(MOBILITY_ONLY)
        template.append(focus_list)

    return template


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
    mode_days: dict[str, set[int]] | None = None,
) -> list[dict[str, Any]]:
    """
    Generate a multi-week plan. Returns list of week dicts:
    [{"week": 1, "days": [...], "progression_note": "..."}, ...]
    If mode_days is provided, uses day-of-week selections (Mon=0..Sun=6) and aligns
    each week to Monday. Otherwise uses get_template_for_days(days_per_week, age).
    """
    weeks = max(1, min(16, int(weeks)))
    start = start_date_opt or date.today()

    if mode_days:
        template = get_template_from_mode_days(mode_days, age)
        # Align week_start to Monday
        week_start_base = start - timedelta(days=start.weekday())
        num_days = 7
    else:
        template = get_template_for_days(days_per_week, age)
        week_start_base = start
        num_days = len(template)

    plan: list[dict[str, Any]] = []
    for w in range(weeks):
        week_start = week_start_base + timedelta(days=7 * w)
        days_out: list[dict[str, Any]] = []
        for d_idx in range(num_days):
            day_date = week_start + timedelta(days=d_idx)
            focus_list = template[d_idx]
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


def _apply_mode_config(out: dict[str, Any], mode_config: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    """Override session_len_min from mode_config if provided."""
    if mode_config and out.get("mode") and out["mode"] in mode_config:
        cfg = mode_config[out["mode"]]
        if "session_len_min" in cfg:
            out["session_len_min"] = cfg["session_len_min"]
    return out


def parse_focus_to_engine_params(
    focus_str: str,
    mode_config: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """
    Map plan focus string to engine params (mode, focus, strength_day_type, session_len_min, etc.).
    Returns None if focus should be skipped (e.g. optional Conditioning).
    If mode_config is provided, overrides session_len_min per mode.
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
        return _apply_mode_config(out, mode_config)

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
        return _apply_mode_config(out, mode_config)

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
        return _apply_mode_config(out, mode_config)

    # Puck Mastery
    if "puck" in s or "skills" in s:
        out["mode"] = "skills_only"
        out["session_len_min"] = 25 if "light" in s else 20
        out["focus"] = None
        return _apply_mode_config(out, mode_config)

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
        return _apply_mode_config(out, mode_config)

    # Mobility/Recovery
    if "mobility" in s or "recovery" in s:
        out["mode"] = "mobility"
        out["session_len_min"] = 25 if "only" in s else 12
        out["focus"] = "mobility"
        return _apply_mode_config(out, mode_config)

    return None


def generate_plan_with_workouts(
    plan: list[dict[str, Any]],
    generate_callback: Callable[[int, str, dict[str, Any]], str],
    base_seed: int = 42,
    mode_config: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Hydrate plan with full workouts for each focus slot. Uses different seeds per slot
    to reduce duplicates. generate_callback(day_idx, focus_str, params) -> workout_text.
    mode_config: per-mode overrides (session_len_min, frequency) from plan builder.
    """
    flat_day_idx = 0
    for w in plan:
        for d in w["days"]:
            focus_items: list[dict[str, Any]] = []
            for f_str in d.get("focus", []):
                params = parse_focus_to_engine_params(f_str, mode_config=mode_config)
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
