"""
Bender Leveling System: XP, overall level (1-25), category ranks (8 per category), achievements.
All XP is earned by completing existing workouts. Do not modify workout generation.
Includes XP caps, workout-length scaling, daily category caps, and same-workout cooldown.
"""
from datetime import date, datetime, timedelta
from typing import Any

# --- Category XP weights (base XP at average duration) ---
CATEGORY_XP = {
    "puck_mastery": 30,
    "skating_mechanics": 25,
    "performance": 25,
    "conditioning": 10,
    "mobility": 10,
}

# --- Average workout duration by category (minutes) — full XP baseline ---
CATEGORY_AVERAGE_MINUTES = {
    "puck_mastery": 45,
    "skating_mechanics": 45,
    "performance": 60,
    "conditioning": 15,
    "mobility": 30,
    "mobility_recovery": 30,
}

# Map session mode (from metadata) to our category key
MODE_TO_CATEGORY = {
    "stickhandling": "puck_mastery",
    "shooting": "puck_mastery",
    "puck_mastery": "puck_mastery",
    "skills_only": "puck_mastery",
    "skating_mechanics": "skating_mechanics",
    "performance": "performance",
    "energy_systems": "conditioning",
    "mobility": "mobility",
}

# --- Overall Bender Level (1-25) ---
LEVEL_THRESHOLDS = [
    0, 100, 250, 450, 700, 1000, 1350, 1750, 2200, 2700,
    3250, 3850, 4500, 5200, 6000, 6900, 7900, 9000, 10200,
    11500, 12900, 14400, 16000, 17700, 19500,
]

LEVEL_TITLES = [
    "Initiate", "Rookie", "Prospect", "Practice Player", "Grinder",
    "Competitor", "Playmaker", "Sniper", "Two Way", "Power Forward",
    "Enforcer", "Veteran", "Alternate Captain", "Captain", "All Star",
    "Game Breaker", "Franchise", "Superstar", "Hall of Famer", "Legend",
    "Icon", "Dynasty", "Immortal", "Mythic", "Bender",
]

# --- Category rank thresholds (same for all 5 categories): 8 ranks ---
RANK_THRESHOLDS = [0, 150, 400, 800, 1400, 2200, 3200, 4500]

CATEGORY_RANK_TITLES = {
    "puck_mastery": ["Stone Hands", "Dusty Mitts", "Toe Drag Trainee", "Dangler", "Puck Magician", "Silky Mitts", "Pond Wizard", "Ankle Breaker"],
    "skating_mechanics": ["Baby Giraffe", "C Cut Rookie", "Stride Apprentice", "Glide Master", "Ice Burner", "Jet Skates", "Edge Savant", "Wheels"],
    "performance": ["Twig", "Practice Squad", "Grinder", "Form Master", "Workhorse", "Rep Monster", "Net Front Titan", "Crease King"],
    "conditioning": ["Hustler", "Grinder", "Workhorse", "Relentless", "Iron Lung", "Endless Motor", "Storm Engine", "Never Tired"],
    "mobility": ["Mover", "Limber", "Fluid", "Elastic", "Durable", "Resilient", "Unbreakable", "Indestructible"],
}

# Rank 8 badge names (for profile card)
RANK_8_BADGES = {
    "puck_mastery": "Ankle Breaker",
    "skating_mechanics": "Wheels",
    "performance": "Crease King",
    "conditioning": "Never Tired",
    "mobility": "Indestructible",
}

# Full XP workout milestone badges (lifetime count of workouts that earned 100% XP)
FULL_XP_WORKOUT_BADGES = {
    10: "Showing Up",
    25: "Practice Regular",
    50: "Dialed In",
    100: "Training Consistent",
    200: "Daily Driver",
    400: "Built Different",
    700: "Work Ethic",
    1000: "Iron Habit",
}

# --- Achievement bonuses (add to total_xp only, not category XP) ---
ACHIEVEMENTS = [
    {"id": "streak_7", "name": "7 Day Workout Streak", "bonus_xp": 50, "check": "streak", "value": 7},
    {"id": "streak_30", "name": "30 Day Workout Streak", "bonus_xp": 200, "check": "streak", "value": 30},
    {"id": "workouts_50", "name": "First 50 Workouts", "bonus_xp": 100, "check": "total_workouts", "value": 50},
    {"id": "workouts_100", "name": "First 100 Workouts", "bonus_xp": 200, "check": "total_workouts", "value": 100},
    {"id": "workouts_250", "name": "First 250 Workouts", "bonus_xp": 500, "check": "total_workouts", "value": 250},
]


def category_from_mode(mode: str) -> str | None:
    """Map session mode to category key, or None if not tracked."""
    if not mode:
        return None
    return MODE_TO_CATEGORY.get((mode or "").strip().lower())


def xp_for_category(category: str) -> int:
    """Base XP for this category (at average duration)."""
    return CATEGORY_XP.get(category, 0)


def get_category_average_minutes(category: str) -> int:
    """Average workout duration for category (minutes). Full XP at this duration."""
    return CATEGORY_AVERAGE_MINUTES.get(category, 45)


def get_length_multiplier(category: str, completed_duration_minutes: float) -> float:
    """Length multiplier: 0.5 to 1.5 based on duration vs category average."""
    avg = get_category_average_minutes(category)
    if avg <= 0:
        return 1.0
    ratio = completed_duration_minutes / avg
    return max(0.5, min(1.5, ratio))


def get_category_multiplier(category: str, workouts_today_in_category: int) -> float:
    """Daily category cap: 1.0, 0.5, or 0.0 based on how many in this category today (including this one)."""
    n = workouts_today_in_category
    if category == "puck_mastery":
        if n <= 3:
            return 1.0
        if n <= 5:
            return 0.5
        return 0.0
    if category == "skating_mechanics":
        if n <= 2:
            return 1.0
        if n == 3:
            return 0.5
        return 0.0
    if category == "performance":
        if n == 1:
            return 1.0
        if n == 2:
            return 0.25
        return 0.0
    if category == "conditioning":
        if n == 1:
            return 1.0
        if n == 2:
            return 0.5
        return 0.0
    if category == "mobility":
        if n <= 2:
            return 1.0
        if n == 3:
            return 0.5
        return 0.0
    return 1.0


def get_repeat_multiplier(same_exact_workout_completed_within_24_hours: bool) -> float:
    """0.25 if same workout repeated within 24h, else 1.0."""
    return 0.25 if same_exact_workout_completed_within_24_hours else 1.0


def is_full_xp_workout(length_multiplier: float, category_multiplier: float, repeat_multiplier: float) -> bool:
    """True only if the workout earned full XP (no length, daily, or repeat penalty)."""
    return (
        length_multiplier >= 1.0
        and category_multiplier == 1.0
        and repeat_multiplier == 1.0
    )


def calculate_workout_xp(
    base_xp: float,
    category: str,
    completed_duration_minutes: float,
    workouts_today_in_category: int,
    same_exact_workout_completed_within_24_hours: bool,
) -> tuple[float, str]:
    """
    Returns (final_xp, feedback_message). Uses consistent rounding (int for XP).
    """
    length_mult = get_length_multiplier(category, completed_duration_minutes)
    category_mult = get_category_multiplier(category, workouts_today_in_category)
    repeat_mult = get_repeat_multiplier(same_exact_workout_completed_within_24_hours)
    raw = base_xp * length_mult * category_mult * repeat_mult
    final_xp = max(0, round(raw))

    if final_xp == 0:
        if category_mult == 0.0:
            return (0.0, "0 points — Daily points limit reached for this category.")
        return (0.0, "0 points")

    reasons = []
    if length_mult < 0.99:
        reasons.append("Shorter workout duration adjusted points")
    if category_mult < 1.0 and category_mult > 0:
        reasons.append("Reduced points after optimal daily training volume")
    if repeat_mult < 1.0:
        reasons.append("Repeated workout cooldown applied")

    if not reasons:
        return (float(final_xp), f"+{final_xp} points")
    return (float(final_xp), f"+{final_xp} points — {'; '.join(reasons)}")


def level_from_total_xp(total_xp: int) -> int:
    """Bender level 1-25 from total XP."""
    level = 1
    for i, thresh in enumerate(LEVEL_THRESHOLDS):
        if total_xp >= thresh:
            level = i + 1
    return min(level, 25)


def title_for_level(level: int) -> str:
    """Level title for display."""
    if level < 1 or level > 25:
        return LEVEL_TITLES[0]
    return LEVEL_TITLES[level - 1]


def xp_threshold_for_level(level: int) -> int:
    """XP required to reach this level (threshold at index level-1)."""
    if level < 1:
        return 0
    if level > 25:
        return LEVEL_THRESHOLDS[-1]
    return LEVEL_THRESHOLDS[level - 1]


def rank_from_category_xp(category_xp: int) -> int:
    """Rank 1-8 from category XP."""
    rank = 1
    for i, thresh in enumerate(RANK_THRESHOLDS):
        if category_xp >= thresh:
            rank = i + 1
    return min(rank, 8)


def rank_title_for_category(category: str, rank: int) -> str:
    """Display title for category rank."""
    titles = CATEGORY_RANK_TITLES.get(category)
    if not titles or rank < 1 or rank > 8:
        return "—"
    return titles[rank - 1]


def category_xp_threshold_for_rank(rank: int) -> int:
    """XP required in category to reach this rank."""
    if rank < 1 or rank > 8:
        return RANK_THRESHOLDS[0]
    return RANK_THRESHOLDS[rank - 1]


def _compute_streak(completion_history: list) -> int:
    """Consecutive calendar days with at least one completion (ending today or yesterday)."""
    if not completion_history:
        return 0
    dates = set()
    for e in completion_history:
        d = e.get("date") or ""
        try:
            dt = date.fromisoformat(d[:10]) if d else None
            if dt:
                dates.add(dt)
        except (ValueError, TypeError):
            continue
    if not dates:
        return 0
    today = date.today()
    # Streak: today or yesterday must have a completion to count
    if today not in dates and (today - timedelta(days=1)) not in dates:
        return 0
    # Count back consecutive days
    streak = 0
    d = today
    for _ in range(400):  # cap
        if d in dates:
            streak += 1
            d -= timedelta(days=1)
        else:
            break
    return streak


def get_longest_streak(completion_history: list) -> int:
    """Longest consecutive calendar days with at least one completion (any time period)."""
    if not completion_history:
        return 0
    dates = []
    for e in completion_history:
        d = e.get("date") or ""
        try:
            dt = date.fromisoformat(d[:10]) if d else None
            if dt:
                dates.append(dt)
        except (ValueError, TypeError):
            continue
    dates = sorted(set(dates))
    if not dates:
        return 0
    best = 1
    curr = 1
    for i in range(1, len(dates)):
        if (dates[i] - dates[i - 1]).days == 1:
            curr += 1
        else:
            best = max(best, curr)
            curr = 1
    return max(best, curr)


def _check_achievements(profile: dict) -> tuple[list[str], int]:
    """Return (list of newly unlocked achievement ids, total bonus XP to add to total_xp)."""
    unlocked = set(profile.get("achievements_unlocked") or [])
    completion_history = profile.get("completion_history") or []
    total_workouts = int(profile.get("total_workouts") or 0)
    streak = _compute_streak(completion_history)

    new_ids = []
    bonus_xp = 0
    for ach in ACHIEVEMENTS:
        if ach["id"] in unlocked:
            continue
        if ach["check"] == "streak" and streak >= ach["value"]:
            new_ids.append(ach["id"])
            bonus_xp += ach["bonus_xp"]
        elif ach["check"] == "total_workouts" and total_workouts >= ach["value"]:
            new_ids.append(ach["id"])
            bonus_xp += ach["bonus_xp"]
    return (new_ids, bonus_xp)


def ensure_leveling_defaults(profile: dict) -> dict:
    """Ensure profile has all leveling fields; set defaults for missing. Returns new dict."""
    p = dict(profile)
    defaults = {
        "total_xp": 0,
        "level": 1,
        "level_title": "Initiate",
        "puck_mastery_xp": 0,
        "puck_mastery_rank": 1,
        "skating_xp": 0,
        "skating_rank": 1,
        "performance_xp": 0,
        "performance_rank": 1,
        "conditioning_xp": 0,
        "conditioning_rank": 1,
        "mobility_xp": 0,
        "mobility_rank": 1,
        "workout_streak": 0,
        "longest_streak": 0,
        "total_workouts": 0,
        "achievements_unlocked": [],
        "full_xp_workouts_total": 0,
        "full_xp_workout_badges_unlocked": [],
    }
    for k, v in defaults.items():
        if k not in p or p[k] is None:
            p[k] = v
    if "total_workouts" not in p or p["total_workouts"] is None:
        stats = p.get("private_victory_stats") or {}
        p["total_workouts"] = int(stats.get("completions_count", 0) or 0)
    # Keep longest_streak in sync from completion_history
    p["longest_streak"] = get_longest_streak(p.get("completion_history") or [])
    return p


def _category_profile_key(category: str) -> tuple[str, str]:
    """(xp_key, rank_key) for profile."""
    key = category.replace("_mechanics", "").replace("_mastery", "_mastery")
    if category == "skating_mechanics":
        return ("skating_xp", "skating_rank")
    if category == "puck_mastery":
        return ("puck_mastery_xp", "puck_mastery_rank")
    if category == "performance":
        return ("performance_xp", "performance_rank")
    if category == "conditioning":
        return ("conditioning_xp", "conditioning_rank")
    if category == "mobility":
        return ("mobility_xp", "mobility_rank")
    return (f"{category}_xp", f"{category}_rank")


def _count_workouts_today_in_category(completion_history: list, category: str) -> int:
    """Count completions today in this category (including the one just appended)."""
    today_str = date.today().isoformat()
    count = 0
    for e in completion_history or []:
        d = (e.get("date") or "")[:10]
        if d != today_str:
            continue
        mode = (e.get("mode") or "").strip().lower()
        cat = category_from_mode(mode)
        if cat == category:
            count += 1
    return count


def _same_workout_completed_within_24_hours(completion_history: list, workout_id: str, completed_at_iso: str) -> bool:
    """True if another completion with same workout_id exists within the last 24 hours."""
    if not workout_id or not isinstance(workout_id, str) or not workout_id.strip():
        return False
    try:
        now = datetime.fromisoformat(completed_at_iso.replace("Z", "+00:00")[:26])
        if now.tzinfo:
            now = now.replace(tzinfo=None)
        cutoff = now - timedelta(hours=24)
    except (ValueError, TypeError):
        return False
    count = 0
    for e in completion_history or []:
        if (e.get("workout_id") or "").strip() != workout_id.strip():
            continue
        ts = e.get("completed_at") or ""
        try:
            t = datetime.fromisoformat(ts.replace("Z", "+00:00")[:26])
            if t.tzinfo:
                t = t.replace(tzinfo=None)
            if t >= cutoff:
                count += 1
        except (ValueError, TypeError):
            continue
    return count >= 2


def apply_xp_and_leveling(profile: dict, metadata: dict) -> tuple[dict, int, str]:
    """
    Apply workout completion: add scaled XP, update level, ranks, streak, achievements.
    Call this after the caller has already appended to completion_history and incremented completions_count.
    metadata: mode, minutes (or len_min), optional workout_id.
    Returns (updated_profile, xp_awarded, feedback_message).
    """
    p = ensure_leveling_defaults(profile)
    mode = (metadata.get("mode") or "").strip().lower()
    category = category_from_mode(mode)
    feedback = ""

    if not category:
        return (p, 0, feedback)

    base_xp = xp_for_category(category)
    if base_xp <= 0:
        return (p, 0, feedback)

    completed_min = float(metadata.get("minutes") or metadata.get("len_min") or 0)
    hist = p.get("completion_history") or []
    workouts_today = _count_workouts_today_in_category(hist, category)
    workout_id = (metadata.get("workout_id") or "").strip()
    last_at = hist[-1].get("completed_at") if hist else ""
    same_in_24h = _same_workout_completed_within_24_hours(hist, workout_id, last_at or datetime.now().isoformat())

    length_mult = get_length_multiplier(category, completed_min)
    category_mult = get_category_multiplier(category, workouts_today)
    repeat_mult = get_repeat_multiplier(same_in_24h)

    final_xp, feedback = calculate_workout_xp(
        base_xp, category, completed_min, workouts_today, same_in_24h
    )
    xp_int = max(0, int(final_xp))

    # Full XP workout track: count and unlock milestone badges
    if is_full_xp_workout(length_mult, category_mult, repeat_mult):
        p["full_xp_workouts_total"] = int(p.get("full_xp_workouts_total") or 0) + 1
        total_full = p["full_xp_workouts_total"]
        unlocked = list(p.get("full_xp_workout_badges_unlocked") or [])
        for threshold, badge_name in FULL_XP_WORKOUT_BADGES.items():
            if total_full >= threshold and badge_name not in unlocked:
                unlocked.append(badge_name)
        p["full_xp_workout_badges_unlocked"] = unlocked

    # Category XP and rank
    xp_key, rank_key = _category_profile_key(category)
    p[xp_key] = int(p.get(xp_key) or 0) + xp_int
    p[rank_key] = rank_from_category_xp(p[xp_key])

    # Total XP
    p["total_xp"] = int(p.get("total_xp") or 0) + xp_int
    p["level"] = level_from_total_xp(p["total_xp"])
    p["level_title"] = title_for_level(p["level"])

    # total_workouts = completions_count (caller already incremented it)
    stats = p.get("private_victory_stats") or {}
    p["total_workouts"] = int(stats.get("completions_count", 0) or 0)
    p["workout_streak"] = _compute_streak(p.get("completion_history") or [])
    p["longest_streak"] = get_longest_streak(p.get("completion_history") or [])

    # Achievements
    new_ach, bonus_xp = _check_achievements(p)
    if new_ach:
        p["achievements_unlocked"] = list(set(p.get("achievements_unlocked") or []) | set(new_ach))
        p["total_xp"] = int(p.get("total_xp") or 0) + bonus_xp
        p["level"] = level_from_total_xp(p["total_xp"])
        p["level_title"] = title_for_level(p["level"])

    return (p, xp_int, feedback)


def _streak_from_dates(dates: set) -> int:
    """Consecutive calendar days with at least one completion, ending today or yesterday."""
    if not dates:
        return 0
    today = date.today()
    if today not in dates and (today - timedelta(days=1)) not in dates:
        return 0
    streak = 0
    d = today
    for _ in range(400):
        if d in dates:
            streak += 1
            d -= timedelta(days=1)
        else:
            break
    return streak


def get_level_progress(profile: dict) -> dict:
    """Current level, title, current XP, next threshold, progress 0-100."""
    p = ensure_leveling_defaults(profile)
    total = int(p.get("total_xp") or 0)
    level = level_from_total_xp(total)
    current_thresh = xp_threshold_for_level(level)
    next_thresh = xp_threshold_for_level(level + 1) if level < 25 else current_thresh
    if level >= 25:
        pct = 100
        next_thresh = current_thresh
    else:
        span = next_thresh - current_thresh
        pct = round(100 * (total - current_thresh) / span, 1) if span else 0
    next_title = title_for_level(level + 1) if level < 25 else title_for_level(level)
    return {
        "level": level,
        "title": title_for_level(level),
        "next_title": next_title,
        "current_xp": total,
        "next_xp": next_thresh,
        "progress_pct": pct,
    }


def get_category_progress(profile: dict, category: str) -> dict:
    """Current rank, title, current XP, next threshold, progress 0-100."""
    p = ensure_leveling_defaults(profile)
    xp_key, rank_key = _category_profile_key(category)
    xp = int(p.get(xp_key) or 0)
    rank = rank_from_category_xp(xp)
    current_thresh = category_xp_threshold_for_rank(rank)
    next_thresh = category_xp_threshold_for_rank(rank + 1) if rank < 8 else current_thresh
    if rank >= 8:
        pct = 100
        next_thresh = current_thresh
        next_title = rank_title_for_category(category, rank)
    else:
        span = next_thresh - current_thresh
        pct = round(100 * (xp - current_thresh) / span, 1) if span else 0
        next_title = rank_title_for_category(category, rank + 1)
    return {
        "rank": rank,
        "title": rank_title_for_category(category, rank),
        "next_title": next_title,
        "current_xp": xp,
        "next_xp": next_thresh,
        "progress_pct": pct,
    }


def get_unlocked_badges(profile: dict) -> list[str]:
    """All badge names: rank 8 category badges + full XP workout milestone badges."""
    p = ensure_leveling_defaults(profile)
    badges = []
    for cat, badge_name in RANK_8_BADGES.items():
        xp_key, _ = _category_profile_key(cat)
        xp = int(p.get(xp_key) or 0)
        if rank_from_category_xp(xp) >= 8:
            badges.append(badge_name)
    badges.extend(p.get("full_xp_workout_badges_unlocked") or [])
    return badges


def get_full_xp_workouts_total(profile: dict) -> int:
    """Lifetime count of workouts that earned 100% XP."""
    p = ensure_leveling_defaults(profile)
    return int(p.get("full_xp_workouts_total") or 0)
