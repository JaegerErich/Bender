"""
Bender Leveling System: XP, overall level (1-25), category ranks (8 per category), achievements.
All XP is earned by completing existing workouts. Do not modify workout generation.
"""
from datetime import date, timedelta
from typing import Any

# --- Category XP weights (per completed workout) ---
CATEGORY_XP = {
    "puck_mastery": 30,
    "skating_mechanics": 25,
    "performance": 25,
    "conditioning": 10,
    "mobility": 10,
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
    """XP awarded for completing a workout in this category."""
    return CATEGORY_XP.get(category, 0)


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
        "total_workouts": 0,
        "achievements_unlocked": [],
    }
    for k, v in defaults.items():
        if k not in p or p[k] is None:
            p[k] = v
    if "total_workouts" not in p or p["total_workouts"] is None:
        stats = p.get("private_victory_stats") or {}
        p["total_workouts"] = int(stats.get("completions_count", 0) or 0)
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


def apply_xp_and_leveling(profile: dict, metadata: dict) -> dict:
    """
    Apply workout completion: add category XP and total XP, update level, ranks, streak, achievements.
    Call this after the caller has already appended to completion_history and incremented completions_count.
    metadata must have 'mode' (session mode). Returns updated profile (new dict).
    """
    p = ensure_leveling_defaults(profile)
    mode = (metadata.get("mode") or "").strip().lower()
    category = category_from_mode(mode)
    if not category:
        return p

    xp = xp_for_category(category)
    if xp <= 0:
        return p

    # Category XP and rank
    xp_key, rank_key = _category_profile_key(category)
    p[xp_key] = int(p.get(xp_key) or 0) + xp
    p[rank_key] = rank_from_category_xp(p[xp_key])

    # Total XP
    p["total_xp"] = int(p.get("total_xp") or 0) + xp
    p["level"] = level_from_total_xp(p["total_xp"])
    p["level_title"] = title_for_level(p["level"])

    # total_workouts = completions_count (caller already incremented it)
    stats = p.get("private_victory_stats") or {}
    p["total_workouts"] = int(stats.get("completions_count", 0) or 0)
    # Streak from completion_history (caller already appended today's completion)
    p["workout_streak"] = _compute_streak(p.get("completion_history") or [])

    # Achievements (check after total_workouts and streak are updated)
    new_ach, bonus_xp = _check_achievements(p)
    if new_ach:
        p["achievements_unlocked"] = list(set(p.get("achievements_unlocked") or []) | set(new_ach))
        p["total_xp"] = int(p.get("total_xp") or 0) + bonus_xp
        p["level"] = level_from_total_xp(p["total_xp"])
        p["level_title"] = title_for_level(p["level"])

    return p


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
    return {
        "level": level,
        "title": title_for_level(level),
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
    else:
        span = next_thresh - current_thresh
        pct = round(100 * (xp - current_thresh) / span, 1) if span else 0
    return {
        "rank": rank,
        "title": rank_title_for_category(category, rank),
        "current_xp": xp,
        "next_xp": next_thresh,
        "progress_pct": pct,
    }


def get_unlocked_badges(profile: dict) -> list[str]:
    """Badge names for categories where player has reached rank 8."""
    p = ensure_leveling_defaults(profile)
    badges = []
    for cat, badge_name in RANK_8_BADGES.items():
        xp_key, _ = _category_profile_key(cat)
        xp = int(p.get(xp_key) or 0)
        if rank_from_category_xp(xp) >= 8:
            badges.append(badge_name)
    return badges
