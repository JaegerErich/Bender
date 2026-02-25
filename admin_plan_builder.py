"""
Admin Plan Builder: multi-week workout plans for Bender.
Only exposed to admin user "Erich Jaeger" via UI gating.
Templates for 3–7 days/week; progression notes; optional drill generation with variety.
"""
from datetime import date, timedelta
from typing import Any, Optional

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


def is_admin_user(display_name: str | None) -> bool:
    """Only Erich Jaeger is admin. Exact match, case-sensitive."""
    return display_name == "Erich Jaeger"


def get_progression_note(week_index: int) -> str:
    """Week 0-based. Weeks 1–2: base; 3–4: +1 set; every 4th: deload."""
    if (week_index + 1) % 4 == 0:
        return "Deload: reduce volume ~20–30%"
    if week_index >= 2:
        return "+1 set on main strength lifts or +1 round on circuits"
    return "Base volume (normal)"


def get_template_for_days(days_per_week: int) -> list[list[str]]:
    """Return day templates for given days/week. Falls back to 3 if invalid."""
    d = max(3, min(7, int(days_per_week)))
    return WEEK_TEMPLATES.get(d, WEEK_TEMPLATES[3])


def generate_plan(
    weeks: int,
    days_per_week: int,
    start_date_opt: date | None = None,
) -> list[dict[str, Any]]:
    """
    Generate a multi-week plan. Returns list of week dicts:
    [{"week": 1, "days": [...], "progression_note": "..."}, ...]
    """
    weeks = max(1, min(16, int(weeks)))
    template = get_template_for_days(days_per_week)
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
