# bender_generate_v8_1_FIXED.py
# NOTE: This is a cleaned + error-fixed single-file version based on what you pasted.
# It compiles and runs as a standalone engine (CLI main included).
# It preserves your non-negotiables:
# - Canonical session modes
# - Strength full gym = fixed template with RX table driven sets/reps
# - Strength no-gym = preset bodyweight circuits from circuits.json (no hierarchy)
# - Optional post-lift conditioning rules (gym: bike/treadmill; no-gym: cones/sprints/no hill)
# - Skate within 24h constraints
# - Memory soft-avoid for drills + circuits

import json
import os
import random
import time
import re
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Default dirs (absolute, so Streamlit Cloud cwd doesn't matter)
DATA_DIR = os.path.join(BASE_DIR, "data")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

# ------------------------------
# Runtime memory globals (set inside generate_session)
# ------------------------------
_CURRENT_RECENT_IDS: set = set()
_CURRENT_RECENT_CIRCUIT_IDS: set = set()
_CURRENT_RECENT_PENALTY: float = 1.0
_CURRENT_LAST_CIRCUIT_SIGNATURE: Tuple[str, ...] = tuple()

# ------------------------------
# Presets (off-ice only)
# ------------------------------
PRESETS: Dict[str, Dict[str, Any]] = {
    "30_min_skills_off_ice": {"session_mode": "skills_only", "session_len_min": 30, "shooting_shots": 150},
    "45_min_skills_off_ice": {"session_mode": "skills_only", "session_len_min": 45, "shooting_shots": 220},
    "60_min_skills_off_ice": {"session_mode": "skills_only", "session_len_min": 60, "shooting_shots": 300},
    "45_min_off_ice": {"session_mode": "movement", "session_len_min": 45},
    "in_season_lift": {
        "session_mode": "strength",
        "session_len_min": 45,
        "strength_emphasis": "strength",
        "strength_day_type": "full",
        "strength_full_gym": True,
        "include_post_lift_conditioning": False,
    },
    "off_season_lift": {
        "session_mode": "strength",
        "session_len_min": 60,
        "strength_emphasis": "hypertrophy",
        "strength_day_type": "lower",
        "strength_full_gym": True,
        "include_post_lift_conditioning": True,
    },
}


def apply_preset(config: Dict[str, Any], preset_name: Optional[str]) -> Dict[str, Any]:
    if not preset_name:
        return config
    key = preset_name.strip()
    if key not in PRESETS:
        valid = ", ".join(sorted(PRESETS.keys()))
        raise SystemExit(f"Unknown preset '{preset_name}'. Valid presets: {valid}")
    preset = PRESETS[key]
    for k, v in preset.items():
        if config.get(k) is None:
            config[k] = v
    return config


# ------------------------------
# Utilities
# ------------------------------
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def norm(s: Any) -> str:
    return str(s).strip() if s is not None else ""


def to_int(x: Any, default: int) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def to_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def truthy(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = norm(x).lower()
    return s in ("true", "1", "yes", "y")


def get(d: Dict[str, Any], *keys: str, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def is_active(d: Dict[str, Any]) -> bool:
    return norm(get(d, "status", default="active")).lower() == "active"


def age_ok(d: Dict[str, Any], age: int) -> bool:
    amin = to_int(get(d, "age_min", default=0), 0)
    amax = to_int(get(d, "age_max", default=99), 99)
    return amin <= age <= amax


def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


def format_seconds_short(sec: int) -> str:
    m = sec // 60
    s = sec % 60
    return f"{m}:{s:02d}"


def _parse_tags(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [norm(x).lower() for x in val if norm(x)]
    s = norm(val).lower()
    if not s:
        return []
    return [t.strip() for t in s.split(",") if t.strip()]


# ------------------------------
# Equipment filtering
# ------------------------------
EQUIP_LEVELS = ("none", "basic", "full")
BASIC_EQUIP_KEYWORDS = (
    "dumbbell",
    "kettlebell",
    "band",
    "medicine ball",
    "med ball",
    "box",
    "bench",
    "pull-up",
    "pull up",
    "rings",
    "bar",
)


def _equip_norm(s: str) -> str:
    return norm(s).lower().replace("&", "and")


def drill_equipment_list(d: Dict[str, Any]) -> List[str]:
    raw = norm(get(d, "equipment", default=""))
    if not raw:
        return []
    r = _equip_norm(raw)
    if r in ("none", "no", "bodyweight"):
        return []
    parts = [p.strip() for p in r.split(",") if p.strip()]
    out: List[str] = []
    for p in parts:
        out.extend([x.strip() for x in p.split("and") if x.strip()])
    return out


def equipment_ok(d: Dict[str, Any], equip_level: str) -> bool:
    lvl = _equip_norm(equip_level)
    if lvl not in EQUIP_LEVELS:
        lvl = "full"

    eq = drill_equipment_list(d)

    if lvl == "none":
        return len(eq) == 0
    if lvl == "full":
        return True
    if not eq:
        return True

    joined = " ".join(eq)
    return any(k in joined for k in BASIC_EQUIP_KEYWORDS)


def preference_score(d: Dict[str, Any]) -> int:
    cp = get(d, "coach_preference", default="")
    if norm(cp) == "":
        return 99
    return to_int(cp, 99)


def coach_pref_multiplier(d: Dict[str, Any]) -> float:
    p = preference_score(d)
    if p == 1:
        return 1.35
    if p == 2:
        return 1.20
    if p == 3:
        return 1.10
    return 1.00


# ------------------------------
# Load data
# ------------------------------
def load_category(filename: str) -> List[Dict[str, Any]]:
    path = os.path.join(DATA_DIR, filename)
    return load_json(path)


def load_all_data(data_dir: str = "data", **kwargs) -> Dict[str, List[Dict[str, Any]]]:
    global DATA_DIR

    # Allow passing "data" or an absolute path
    if os.path.isabs(data_dir):
        DATA_DIR = data_dir
    else:
        DATA_DIR = os.path.join(BASE_DIR, data_dir)

    return {
        "warmup": load_category("warmup.json"),
        "movement": load_category("movement.json"),
        "conditioning": load_category("conditioning.json"),
        "stickhandling": load_category("stickhandling.json"),
        "shooting": load_category("shooting.json"),
        "mobility": load_category("mobility.json"),
        "strength": load_category("strength.json"),
        "circuits": load_category("circuits.json"),
    }


# ------------------------------
# Focus logic
# ------------------------------
FOCUS_MAP: Dict[str, Dict[str, Dict[str, Any]]] = {
    "quick_release": {
        "stickhandling": {"tags_any": ["quick_hands", "deception"], "weight": 1.8},
        "shooting": {"shooting_bucket_any": ["quick_release"], "weight": 2.8},
    },
    "backhand": {
        "stickhandling": {"tags_any": ["backhand"], "weight": 2.0},
        "shooting": {"shot_type_any": ["backhand"], "weight": 3.0},
    },
    "toe_drag": {
        "stickhandling": {"tags_any": ["toe_drags", "toe_drag"], "weight": 2.2},
        "shooting": {"tags_any": ["toe_drag"], "weight": 2.0},
    },
    "puck_protection": {
        "stickhandling": {"tags_any": ["puck_protection", "protection"], "weight": 2.2},
        "shooting": {"tags_any": ["puck_protection"], "weight": 1.6},
    },
    "conditioning": {"conditioning": {"weight": 2.0}},
    "recovery": {"mobility": {"weight": 2.5}},
}


def get_focus_rules(focus: Optional[str], category: str) -> Optional[Dict[str, Any]]:
    if not focus:
        return None
    focus = focus.lower().strip()
    return FOCUS_MAP.get(focus, {}).get(category)


def matches_any(value: Any, wanted: List[str]) -> bool:
    v = norm(value).lower()
    return any(v == norm(w).lower() for w in wanted)


def drill_matches_rule(d: Dict[str, Any], rule: Dict[str, Any]) -> bool:
    if not rule:
        return False

    if "tags_any" in rule:
        tags = set(_parse_tags(get(d, "tags", default="")))
        wanted = [norm(x).lower() for x in rule["tags_any"]]
        if any(w in tags for w in wanted):
            return True

    for k, wanted in rule.items():
        if not k.endswith("_any"):
            continue
        field = k[:-4]
        have = get(d, field, default=None)
        if have is None:
            continue
        if matches_any(have, wanted):
            return True

    return False


def focus_multiplier_for_drill(d: Dict[str, Any], rule: Optional[Dict[str, Any]]) -> float:
    if not rule:
        return 1.0
    w = float(rule.get("weight", 1.0))
    non_weight_keys = [k for k in rule.keys() if k != "weight"]
    if not non_weight_keys:
        return w
    return w if drill_matches_rule(d, rule) else 1.0


# ------------------------------
# Filtering
# ------------------------------
def filter_drills(drills: List[Dict[str, Any]], age: int, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for d in drills:
        if not is_active(d):
            continue
        if not age_ok(d, age):
            continue

        ok = True
        for key, want in filters.items():
            if want is None:
                continue

            if key == "plyo":
                have = get(d, "Plyo", "plyo", default=False)
                if truthy(have) != truthy(want):
                    ok = False
                    break
                continue

            have = get(d, key, default=None)
            if isinstance(want, str):
                if norm(have).lower() != want.lower():
                    ok = False
                    break
            else:
                if have != want:
                    ok = False
                    break

        if ok:
            out.append(d)
    return out


# ------------------------------
# Weighted picking
# ------------------------------
def weighted_choice(items: List[Dict[str, Any]], weights: List[float], rnd: random.Random) -> Dict[str, Any]:
    total = sum(max(w, 0.0) for w in weights)
    if total <= 0:
        return rnd.choice(items)
    r = rnd.random() * total
    upto = 0.0
    for item, w in zip(items, weights):
        w = max(w, 0.0)
        upto += w
        if upto >= r:
            return item
    return items[-1]


def pick_n(
    drills: List[Dict[str, Any]],
    n: int,
    rnd: random.Random,
    focus_rule: Optional[Dict[str, Any]] = None,
    avoid_ids: Optional[set] = None,
    recent_ids: Optional[set] = None,
    recent_penalty: float = 0.25,
) -> List[Dict[str, Any]]:
    if not drills or n <= 0:
        return []

    pool = drills[:]
    picked: List[Dict[str, Any]] = []
    seen_ids = {norm(x) for x in (avoid_ids or set()) if norm(x)}

    if recent_ids is None:
        recent_ids = _CURRENT_RECENT_IDS
    recent_ids_norm = {norm(x) for x in (recent_ids or set()) if norm(x)}

    for _ in range(n):
        candidates = [d for d in pool if norm(get(d, "id", "")) not in seen_ids]
        if not candidates:
            candidates = pool

        weights: List[float] = []
        for d in candidates:
            w = 1.0
            w *= coach_pref_multiplier(d)
            w *= focus_multiplier_for_drill(d, focus_rule)
            did = norm(get(d, "id", default=""))
            if did and did in recent_ids_norm:
                w *= float(recent_penalty)
            w *= (0.95 + 0.10 * rnd.random())
            weights.append(w)

        chosen = weighted_choice(candidates, weights, rnd)
        picked.append(chosen)

        did = norm(get(chosen, "id", default=""))
        if did:
            seen_ids.add(did)

    return picked


# ------------------------------
# Strength helpers (needed early by format_drill)
# ------------------------------
def _std_level(x: Any) -> str:
    s = norm(x).lower()
    if s in ("med", "medium"):
        return "medium"
    if s in ("mod", "moderate"):
        return "moderate"
    if s in ("low-med", "low_moderate", "lowmoderate"):
        return "moderate"
    if s in ("hi", "high"):
        return "high"
    if s in ("very-low", "verylow", "very_low"):
        return "low"
    return s

def _strength_time_profile(session_len_min: int, skate_within_24h: bool) -> Dict[str, int]:
    m = int(session_len_min)

    prof = {
        "speed": 1,         # number of speed/power drills
        "blocks": 2,        # number of (secondary+resilience) blocks
        "accessory": 0,     # extra lifting after blocks
        "mobility_n": 3,    # mobility drills
        "finisher_min": 0,  # optional post-lift conditioning minutes
        "warmup_cap": 10,   # max warmup drills to print
    }

    # FAST strength: 20–29 min
    if m <= 29:
        prof.update({
            "speed": 0,          # NO power block
            "blocks": 1,         # ONLY Block A
            "accessory": 0,
            "mobility_n": 2,
            "finisher_min": 0,
            "warmup_cap": 6,     # quick warmup
        })
    elif m <= 40:
        prof.update({"speed": 1, "blocks": 1, "accessory": 0, "mobility_n": 2, "finisher_min": 0, "warmup_cap": 8})
    elif m <= 55:
        prof.update({"speed": 1, "blocks": 2, "accessory": 0, "mobility_n": 3, "finisher_min": 0, "warmup_cap": 10})
    elif m <= 70:
        prof.update({"speed": 2, "blocks": 2, "accessory": 1, "mobility_n": 3, "finisher_min": 6, "warmup_cap": 10})
    else:  # 75–90
        prof.update({"speed": 2, "blocks": 2, "accessory": 2, "mobility_n": 4, "finisher_min": 8, "warmup_cap": 10})

    if skate_within_24h:
        prof["speed"] = min(prof["speed"], 1)
        prof["accessory"] = 0
        prof["finisher_min"] = 0

    return prof

def _is_bodyweightish(d: Dict[str, Any]) -> bool:
    did = norm(get(d, "id", "")).upper()
    if did.startswith("BW_"):
        return True
    eq = norm(get(d, "equipment", "")).lower()
    return eq in ("", "none", "bodyweight", "no equipment")

def _upper_subpattern(d: Dict[str, Any]) -> str:
    """
    Classifies upper push/pull variety buckets.
    Uses tags/name keywords if present; safe fallbacks.
    """
    mp = movement_pattern(d)
    name = norm(get(d, "name", default="")).lower()
    tags = norm(get(d, "tags", default="")).lower()

    # Push variety
    if mp == "push":
        if "incline" in name or "incline" in tags:
            return "push_incline"
        if "overhead" in name or "ohp" in name or "shoulder press" in name or "vertical" in tags:
            return "push_vertical"
        if "dip" in name or "dip" in tags:
            return "push_dip"
        return "push_horizontal"

    # Pull variety
    if mp == "pull":
        if "pullup" in name or "chin" in name or "lat pulldown" in name or "pulldown" in name or "vertical" in tags:
            return "pull_vertical"
        if "row" in name or "row" in tags:
            return "pull_horizontal"
        return "pull_other"

    return mp or "other"

def _upper_direction(d: Dict[str, Any]) -> Optional[str]:
    """
    Classifies upper lifts as horizontal or vertical.
    Used ONLY for upper-day balancing.
    """
    mp = movement_pattern(d)
    name = norm(get(d, "name", "")).lower()
    tags = norm(get(d, "tags", "")).lower()

    if mp == "push":
        if "bench" in name or "chest" in tags:
            return "horizontal"
        return "vertical"

    if mp == "pull":
        if "row" in name:
            return "horizontal"
        return "vertical"

    return None

def _is_heavy_vertical(d: Dict[str, Any]) -> bool:
    return (
        _upper_direction(d) == "vertical"
        and fatigue_cost_level(d) == "high"
    )

def strength_focus(d: Dict[str, Any]) -> str:
    sf = norm(get(d, "strength_focus", default="")).lower()
    if sf:
        return sf
    b = norm(get(d, "strength_bucket", default="")).lower()
    mp = norm(get(d, "movement_pattern", default="")).lower()
    name = norm(get(d, "name", default="")).lower()
    tags = set(_parse_tags(get(d, "tags", default="")))

    if b == "power" or mp == "power" or any(t in tags for t in ("plyometric", "vertical_jump", "lateral_power", "rotational_power")):
        return "power"
    if b in ("core_anti", "core_rotation") or "plank" in name or "pallof" in name:
        return "stability"
    return "hypertrophy"


def movement_pattern(d: Dict[str, Any]) -> str:
    mp = norm(get(d, "movement_pattern", default="")).lower()
    return mp or "unknown"

def lift_role(d: Dict[str, Any]) -> str:
    """
    Returns the drill's lift_role (e.g., primary/auxiliary/accessory),
    normalized to lowercase. Safe if key is missing.
    """
    return norm(get(d, "lift_role", default="")).lower()


def primary_region(d: Dict[str, Any]) -> str:
    """
    Returns the drill's primary_region (e.g., upper/lower/core/full),
    normalized to lowercase. Safe if key is missing.
    """
    return norm(get(d, "primary_region", default="")).lower()

def cns_load_level(d: Dict[str, Any]) -> str:
    return _std_level(get(d, "cns_load", default=get(d, "CNS_load", default="low")))


def fatigue_cost_level(d: Dict[str, Any]) -> str:
    fc = _std_level(get(d, "fatigue_cost", default=""))
    if fc in ("low", "medium", "high"):
        return fc
    cns = cns_load_level(d)
    inten = _std_level(get(d, "intensity", default=""))
    if cns == "high" or inten == "high":
        return "high"
    if cns in ("medium", "moderate") or inten in ("medium", "moderate"):
        return "medium"
    return "low"

def is_scap_accessory(d: Dict[str, Any]) -> bool:
    return (
        strength_focus(d) == "stability"
        and lift_role(d) == "accessory"
        and primary_region(d) == "upper"
    )

def is_stability_candidate(d: Dict[str, Any]) -> bool:
    """
    Canonical rule:
      - If strength_focus is explicitly 'stability' -> True
    Compatibility fallback:
      - For older/dirty data, allow limited token + name-based detection
        so nothing breaks during migration.
    """
    # 1) Canonical (preferred)
    sf = norm(get(d, "strength_focus", default="")).lower()
    if sf == "stability":
        return True

    # 2) Compatibility fallback (keep SMALL and predictable)
    # Only check a couple legacy fields that you might still have in older JSON
    for key in ("lift_role", "movement_pattern", "tags"):
        v = norm(get(d, key, default="")).lower()
        if any(tok in v for tok in ("anti-rotation", "anti rotation", "core", "groin", "copenhagen", "carry")):
            return True

    # 3) Last-resort name fallback (minimal keywords)
    name = norm(get(d, "name", "")).lower()
    keywords = (
        "pallof",
        "dead bug",
        "deadbug",
        "bird dog",
        "birddog",
        "side plank",
        "copenhagen",
        "suitcase carry",
        "farmer carry",
        "overhead carry",
        "carry",
        "ab wheel",
        "rollout",
        "stir the pot",
    )
    return any(k in name for k in keywords)


def _is_lower_day(day_type: str) -> bool:
    dt = (day_type or "").strip().lower()
    return ("lower" in dt) or ("leg" in dt)

def _is_upper_day(day_type: str) -> bool:
    dt = (day_type or "").strip().lower()
    return "upper" in dt

def _region_ok_for_day(d: Dict[str, Any], day_type: str) -> bool:
    """
    Locks non-resilience strength picks to the day type.
    Allows 'full' and 'core' on both days.
    """
    region = norm(get(d, "primary_region", "")).lower()

    # Always allow these on either day
    if region in ("full", "core"):
        return True

    if _is_lower_day(day_type):
        return region == "lower"

    if _is_upper_day(day_type):
        return region == "upper"

    # If day_type is unknown/other, don't block selection
    return True

def _is_push_pull(mp: str) -> bool:
    return (mp or "").strip().lower() in ("push", "pull")

def _count_push_pull(drills: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"push": 0, "pull": 0}
    for d in drills:
        mp = movement_pattern(d)
        if mp == "push":
            counts["push"] += 1
        elif mp == "pull":
            counts["pull"] += 1
    return counts

def _opposing_push_pull(mp: str) -> Optional[str]:
    mp = (mp or "").strip().lower()
    if mp == "push":
        return "pull"
    if mp == "pull":
        return "push"
    return None

# ------------------------------
# Formatting
# ------------------------------
def format_drill(d: Dict[str, Any]) -> str:
    did = norm(get(d, "id", default=""))
    name = norm(get(d, "name", default="(unnamed)"))
    cues = norm(get(d, "coaching_cues", default=""))
    steps = norm(get(d, "step_by_step", default=""))
    line = f"- {did} {name}".strip()
    if cues:
        line += f"\n  Cues: {cues}"
    if steps:
        line += f"\n  Steps: {steps}"
    if strength_focus(d) == "power":
        line += "\n  Intent: Max intent each rep. Full reset between reps."
    return line


# ------------------------------
# Warmup (Strength-only rules)
# ------------------------------
LEG_WARMUP_SEQUENCE = [f"WU_{i:03d}" for i in range(1, 16)]


def build_strength_warmup(
    warmups: List[Dict[str, Any]],
    age: int,
    rnd: random.Random,
    day_type: str,
) -> List[Dict[str, Any]]:
    active_age_ok = [d for d in warmups if is_active(d) and age_ok(d, age)]

    if day_type in ("leg", "full", "lower"):
        by_id = {norm(get(d, "id", "")): d for d in active_age_ok}
        chosen = []
        missing = []
        for wid in LEG_WARMUP_SEQUENCE:
            if wid in by_id:
                chosen.append(by_id[wid])
            else:
                missing.append(wid)
        if missing:
            chosen.append(
                {"id": "WARN", "name": f"Missing warmups: {', '.join(missing)}", "coaching_cues": "", "step_by_step": ""}
            )
        return chosen

    uppers = [d for d in active_age_ok if norm(get(d, "warmup_bucket", "")).lower() == "upper_body"]
    if not uppers:
        return [{"id": "WARN", "name": "No warmups found in warmup_bucket=upper_body", "coaching_cues": "", "step_by_step": ""}]
    return pick_n(uppers, n=min(4, len(uppers)), rnd=rnd)


# ------------------------------
# Stickhandling
# ------------------------------
def stickhandling_skill_mix_counts(block_minutes: int) -> Dict[str, int]:
    if block_minutes <= 15:
        return {"beginner": 1, "intermediate": 1, "advanced": 1}
    return {"beginner": 2, "intermediate": 2, "advanced": 2}


def pick_stickhandling_mixed(
    drills: List[Dict[str, Any]],
    age: int,
    rnd: random.Random,
    block_minutes: int,
    focus_rule: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    pool = [d for d in drills if is_active(d) and age_ok(d, age)]
    if not pool:
        return []

    counts = stickhandling_skill_mix_counts(block_minutes)

    def pool_for(level: str) -> List[Dict[str, Any]]:
        return [d for d in pool if norm(get(d, "skill_level", "")).lower() == level]

    chosen: List[Dict[str, Any]] = []
    for level in ("beginner", "intermediate", "advanced"):
        need = counts[level]
        sub = pool_for(level)
        if sub:
            chosen.extend(pick_n(sub, n=min(need, len(sub)), rnd=rnd, focus_rule=focus_rule))

    if len(chosen) < sum(counts.values()):
        chosen_ids = {norm(get(d, "id", "")) for d in chosen}
        remaining = [d for d in pool if norm(get(d, "id", "")) not in chosen_ids]
        if remaining:
            chosen.extend(pick_n(remaining, n=(sum(counts.values()) - len(chosen)), rnd=rnd, focus_rule=focus_rule))

    rnd.shuffle(chosen)
    return chosen


def build_stickhandling_circuit(drills: List[Dict[str, Any]], block_seconds: int) -> List[str]:
    work, rest = 45, 15
    per_drill = work + rest
    if not drills:
        return ["- [No matching drills found]"]

    round_time = per_drill * len(drills)
    rounds = clamp(block_seconds // max(1, round_time), 2, 6)
    total_est = rounds * round_time

    lines: List[str] = []
    lines.append(f"Format: {work}s work / {rest}s rest | {rounds} rounds (~{format_seconds_short(total_est)})")
    lines.append("Run as a loop — repeat the same drills each round.")
    for d in drills:
        lines.append(format_drill(d))
    return lines


# ------------------------------
# Shooting
# ------------------------------
def infer_shot_type(d: Dict[str, Any]) -> str:
    st = norm(get(d, "shot_type", "")).lower()
    if st in ("forehand", "backhand", "slapshot"):
        return st

    bucket = norm(get(d, "shooting_bucket", "")).lower()
    name = norm(get(d, "name", "")).lower()

    if "backhand" in bucket or "backhand" in name:
        return "backhand"
    if "slap" in bucket or "slap" in name:
        return "slapshot"
    return "forehand"


def choose_shooting_drills(
    drills: List[Dict[str, Any]],
    age: int,
    rnd: random.Random,
    total_shots: int,
    focus: Optional[str],
    focus_rule: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    pool = [d for d in drills if is_active(d) and age_ok(d, age)]
    if not pool:
        return []

    min_shots_per_drill = 20
    max_drills = 5
    min_drills = 3

    n = clamp(total_shots // min_shots_per_drill, min_drills, max_drills)

    explicit_type_focus = focus in ("forehand", "backhand", "slapshot")

    forehand_pool = [d for d in pool if infer_shot_type(d) == "forehand"]
    backhand_pool = [d for d in pool if infer_shot_type(d) == "backhand"]

    chosen: List[Dict[str, Any]] = []
    if not explicit_type_focus:
        if forehand_pool:
            chosen.extend(pick_n(forehand_pool, 1, rnd, focus_rule=focus_rule))
        if backhand_pool:
            chosen.extend(pick_n(backhand_pool, 1, rnd, focus_rule=focus_rule))

    chosen_ids = {norm(get(d, "id", "")) for d in chosen}
    remaining = [d for d in pool if norm(get(d, "id", "")) not in chosen_ids]

    if len(chosen) < n and remaining:
        chosen.extend(pick_n(remaining, n - len(chosen), rnd, focus_rule=focus_rule))

    if not explicit_type_focus:
        have_fh = any(infer_shot_type(d) == "forehand" for d in chosen)
        have_bh = any(infer_shot_type(d) == "backhand" for d in chosen)
        if not have_fh and forehand_pool:
            chosen[-1:] = pick_n(forehand_pool, 1, rnd, focus_rule=focus_rule)
        if not have_bh and backhand_pool:
            chosen[-1:] = pick_n(backhand_pool, 1, rnd, focus_rule=focus_rule)

    rnd.shuffle(chosen)
    return chosen[:n]


def build_shooting_by_shots(drills: List[Dict[str, Any]], target_shots: int) -> List[str]:
    if not drills:
        return ["- [No matching drills found]"]

    min_shots_per_drill = 20
    n = len(drills)

    required_min_total = n * min_shots_per_drill
    if target_shots < required_min_total:
        target_shots = required_min_total

    base = target_shots // n
    rem = target_shots % n
    per_drill = [base + (1 if i < rem else 0) for i in range(n)]
    for i in range(n):
        if per_drill[i] < min_shots_per_drill:
            per_drill[i] = min_shots_per_drill
    target_shots = sum(per_drill)

    def split_sets(shots: int) -> str:
        sets = max(1, min(6, round(shots / 10)))
        per_set = max(10, shots // sets)
        planned = sets * per_set
        last_set = per_set + (shots - planned)
        if sets == 1:
            return f"{shots} shots"
        if last_set == per_set:
            return f"{sets} x {per_set} shots"
        return f"{sets-1} x {per_set} + 1 x {last_set} shots"

    lines: List[str] = []
    lines.append(f"Target volume: {target_shots} total shots | Min {min_shots_per_drill}/drill")
    lines.append("Guidelines: stay on one drill long enough to feel it. Full intent, clean mechanics.")

    for d, shots in zip(drills, per_drill):
        did = norm(get(d, "id", ""))
        name = norm(get(d, "name", "(unnamed)"))
        cues = norm(get(d, "coaching_cues", default=""))
        steps = norm(get(d, "step_by_step", default=""))
        stype = infer_shot_type(d)
        lines.append(f"- {did} {name} ({stype}) | {split_sets(shots)}")
        if cues:
            lines.append(f"  Cues: {cues}")
        if steps:
            lines.append(f"  Steps: {steps}")

    return lines


def build_shooting_from_defaults(drills: List[Dict[str, Any]]) -> List[str]:
    if not drills:
        return ["- [No matching drills found]"]
    lines: List[str] = []
    for d in drills:
        did = norm(get(d, "id", ""))
        name = norm(get(d, "name", "(unnamed)"))
        reps = norm(get(d, "default_reps", default=""))
        if not reps:
            reps = "20"
        cues = norm(get(d, "coaching_cues", default=""))
        steps = norm(get(d, "step_by_step", default=""))
        lines.append(f"- {did} {name} — {reps} shots")
        if cues:
            lines.append(f"  Cues: {cues}")
        if steps:
            lines.append(f"  Steps: {steps}")
    return lines


# ------------------------------
# Conditioning
# ------------------------------
def conditioning_modality(d: Dict[str, Any]) -> str:
    eq = norm(get(d, "equipment", "")).lower()
    name = norm(get(d, "name", "")).lower()
    text = f"{eq} {name}"

    if "bike" in text:
        return "bike"
    if "curved treadmill" in text or "treadmill" in text:
        return "treadmill"
    if "hill" in text:
        return "hill"
    if "stair" in text:
        return "stairs"
    if "cone" in text or "shuttle" in text:
        return "cones"
    return "bodyweight"


def conditioning_energy_system(d: Dict[str, Any]) -> str:
    b = norm(get(d, "conditioning_bucket", "")).lower()
    if b in ("repeat_sprint",):
        return "repeat_sprint"
    if b in ("tempo_base",):
        return "tempo"
    if b in ("mixed_direction",):
        return "mixed_direction"
    return "aerobic"


def pick_conditioning_drills(
    drills: List[Dict[str, Any]],
    age: int,
    rnd: random.Random,
    session_len_min: int,
    focus_rule: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    pool = [d for d in drills if is_active(d) and age_ok(d, age)]
    if not pool:
        return []

    n = 1 if session_len_min <= 30 else 2
    if n == 1:
        return pick_n(pool, 1, rnd, focus_rule=focus_rule)

    first = pick_n(pool, 1, rnd, focus_rule=focus_rule)[0]
    first_mod = conditioning_modality(first)
    first_es = conditioning_energy_system(first)

    cand = [d for d in pool if conditioning_modality(d) != first_mod] or pool[:]
    cand_mix = [d for d in cand if conditioning_energy_system(d) != first_es]
    second_pool = cand_mix if cand_mix else cand

    second = pick_n(second_pool, 1, rnd, focus_rule=focus_rule, avoid_ids={norm(get(first, "id", ""))})[0]
    return [first, second]


def conditioning_default_work_rest(d: Dict[str, Any]) -> Tuple[int, int]:
    work = to_int(get(d, "default_duration_sec", default=60), 60)
    es = conditioning_energy_system(d)
    work = clamp(work, 20, 600)

    if es == "repeat_sprint":
        rest = clamp(int(work * 2), 45, 180)
        return work, rest
    if es == "tempo":
        rest = clamp(int(work * 0.75), 20, 90)
        return work, rest
    if es == "mixed_direction":
        rest = clamp(int(work * 1.0), 30, 120)
        return work, rest
    rest = clamp(int(work * 1.0), 30, 120)
    return work, rest


def build_conditioning_block(drills: List[Dict[str, Any]], block_seconds: int) -> List[str]:
    if not drills:
        return ["- [No matching drills found]"]

    lines: List[str] = []

    def describe_one(d: Dict[str, Any], seconds: int) -> List[str]:
        did = norm(get(d, "id", ""))
        name = norm(get(d, "name", "(unnamed)"))
        cues = norm(get(d, "coaching_cues", default=""))
        steps = norm(get(d, "step_by_step", default=""))
        mod = conditioning_modality(d)
        es = conditioning_energy_system(d)
        work, rest = conditioning_default_work_rest(d)

        ramp = clamp(int(seconds * 0.10), 60, 180) if seconds >= 8 * 60 else clamp(int(seconds * 0.12), 45, 120)
        main = max(0, seconds - ramp)
        interval = work + rest
        rounds = max(1, main // max(1, interval))
        est = ramp + rounds * interval

        out: List[str] = []
        out.append(f"- {did} {name} [{mod} | {es}]")
        out.append(
            f"  Time plan: ~{format_seconds_short(ramp)} ramp + {rounds} rounds of ({work}s work / {rest}s easy) (~{format_seconds_short(est)})"
        )
        if cues:
            out.append(f"  Cues: {cues}")
        if steps:
            out.append(f"  Steps: {steps}")
        return out

    if len(drills) == 1:
        lines.append("Structure: 1 conditioning focus (repeat to fill time).")
        lines.extend(describe_one(drills[0], block_seconds))
        return lines

    a_time = block_seconds // 2
    b_time = block_seconds - a_time
    lines.append("Structure: 2 conditioning blocks (different types when possible).")
    lines.append(f"Block A (~{format_seconds_short(a_time)})")
    lines.extend(describe_one(drills[0], a_time))
    lines.append(f"Block B (~{format_seconds_short(b_time)})")
    lines.extend(describe_one(drills[1], b_time))
    return lines


def filter_post_lift_conditioning_pool(
    conditioning_drills: List[Dict[str, Any]],
    *,
    full_gym: bool,
    post_lift_conditioning_type: Optional[str],
) -> List[Dict[str, Any]]:
    if full_gym:
        if post_lift_conditioning_type in ("bike", "treadmill"):
            return [d for d in conditioning_drills if conditioning_modality(d) == post_lift_conditioning_type]
        return [d for d in conditioning_drills if conditioning_modality(d) in ("bike", "treadmill")]

    allowed = ("cones", "bodyweight", "sprints", "none")
    out: List[Dict[str, Any]] = []
    for d in conditioning_drills:
        mod = conditioning_modality(d)
        if mod == "hill":
            continue
        eq = norm(get(d, "equipment", default="")).lower()
        noeq = eq in ("", "none", "no equipment", "n/a")
        cone_ok = "cone" in eq
        if (mod in allowed or mod == "cones") and (noeq or cone_ok):
            out.append(d)
    return out


# ------------------------------
# Mobility
# ------------------------------
def mobility_intensity_ok(d: Dict[str, Any]) -> bool:
    inten = norm(get(d, "intensity", "")).lower()
    return inten in ("very-low", "low", "")


def pick_mobility_drills(
    drills: List[Dict[str, Any]],
    age: int,
    rnd: random.Random,
    n: int,
    focus_rule: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    pool = [d for d in drills if is_active(d) and age_ok(d, age) and mobility_intensity_ok(d)]
    return pick_n(pool, n=n, rnd=rnd, focus_rule=focus_rule)


def build_mobility_cooldown_circuit(drills: List[Dict[str, Any]], block_seconds: int) -> List[str]:
    if not drills:
        return ["- [No matching drills found]"]

    if block_seconds <= 8 * 60:
        rounds = 2
    elif block_seconds <= 12 * 60:
        rounds = 3
    else:
        rounds = 3

    per_drill = max(30, min(60, (block_seconds // max(1, (len(drills) * rounds))) if rounds > 0 else 45))

    lines: List[str] = []
    lines.append(f"Format: {len(drills)} drills | {per_drill}s each | {rounds} rounds (~{format_seconds_short(per_drill * len(drills) * rounds)})")
    lines.append("Move slow. Nasal breathing. No forcing.")
    for d in drills:
        lines.append(format_drill(d))
    return lines


def build_mobility_timed_session(drills: List[Dict[str, Any]], total_seconds: int) -> List[str]:
    if not drills:
        return ["- [No matching drills found]"]

    n = len(drills)
    per = max(120, total_seconds // max(1, n))
    lines: List[str] = []
    for d in drills:
        did = norm(get(d, "id", ""))
        name = norm(get(d, "name", "(unnamed)"))
        cues = norm(get(d, "coaching_cues", default=""))
        steps = norm(get(d, "step_by_step", default=""))
        lines.append(f"- {did} {name} ({per // 60} min)")
        if cues:
            lines.append(f"  Cues: {cues}")
        if steps:
            lines.append(f"  Steps: {steps}")
    return lines


# ------------------------------
# Strength (RX table + fixed template)
# ------------------------------
STRENGTH_EMPHASIS_UI_NAME = {
    "power": "power (explosive speed)",
    "strength": "strength (game strength)",
    "hypertrophy": "hypertrophy (strength capacity)",
    "recovery": "recovery (less stress)",
}

FATIGUE_ROLE_HIGH = "high_fatigue"
FATIGUE_ROLE_SECONDARY = "secondary"
FATIGUE_ROLE_RESILIENCE = "resilience"

STRENGTH_RX_TABLE: Dict[str, Dict[str, Any]] = {
    "power": {
        FATIGUE_ROLE_HIGH: {"sets": "4–6", "reps": "3–6", "intent": "Max speed and intent, full reset between reps"},
        FATIGUE_ROLE_SECONDARY: {"sets": "3–4", "reps": "4–6", "intent": "Fast but controlled explosive work"},
        FATIGUE_ROLE_RESILIENCE: {"sets": "2–3", "reps": "6–10", "intent": "Elastic, reactive, low fatigue"},
    },
    "strength": {
        FATIGUE_ROLE_HIGH: {"sets": "4–6", "reps": "4–7", "intent": "Heavy, clean reps with full control"},
        FATIGUE_ROLE_SECONDARY: {"sets": "3–4", "reps": "6–10", "intent": "Support strength, stability focus"},
        FATIGUE_ROLE_RESILIENCE: {"sets": "2–3", "reps": "8–12", "intent": "Trunk, carries, unilateral control"},
    },
    "hypertrophy": {
        FATIGUE_ROLE_HIGH: {"sets": "3–5", "reps": "6–10", "intent": "Muscle + strength overlap"},
        FATIGUE_ROLE_SECONDARY: {"sets": "3–4", "reps": "8–14", "intent": "Volume, joint-friendly loading"},
        FATIGUE_ROLE_RESILIENCE: {"sets": "2–3", "reps": "12–20", "intent": "Tendons, balance, symmetry"},
    },
    "recovery": {
        FATIGUE_ROLE_HIGH: None,
        FATIGUE_ROLE_SECONDARY: {"sets": "2–3", "reps": "10–15", "intent": "Blood flow, light movement"},
        FATIGUE_ROLE_RESILIENCE: {"sets": "2–3", "reps": "20–40s", "intent": "Calm, controlled, restore"},
    },
}


def _normalize_strength_emphasis(emphasis: Any) -> str:
    e = norm(emphasis).lower()
    if e in ("power", "strength", "hypertrophy", "recovery"):
        return e
    if e == "youth":
        return "hypertrophy"
    return "strength"


def _rx_for(emphasis: Any, fatigue_role: str) -> Optional[Dict[str, str]]:
    e = _normalize_strength_emphasis(emphasis)
    role = fatigue_role if fatigue_role in (FATIGUE_ROLE_HIGH, FATIGUE_ROLE_SECONDARY, FATIGUE_ROLE_RESILIENCE) else FATIGUE_ROLE_SECONDARY
    entry = STRENGTH_RX_TABLE.get(e, STRENGTH_RX_TABLE["strength"]).get(role)
    return entry  # may be None for recovery+high_fatigue


def rx_for(emphasis: Any, fatigue_role: str) -> Optional[Dict[str, str]]:
    return _rx_for(emphasis, fatigue_role)


def _fatigue_role_for_speed_drill(d: Dict[str, Any]) -> str:
    return FATIGUE_ROLE_HIGH if fatigue_cost_level(d) == "high" else FATIGUE_ROLE_SECONDARY


def fatigue_role_for_drill(d: Dict[str, Any], block: str) -> str:
    b = (block or "").lower().strip()
    if b == "speed":
        return _fatigue_role_for_speed_drill(d)
    if b == "high":
        return FATIGUE_ROLE_HIGH
    if b == "secondary":
        return FATIGUE_ROLE_SECONDARY
    if b == "resilience":
        return FATIGUE_ROLE_RESILIENCE
    return FATIGUE_ROLE_SECONDARY


def _cns_is_high(d: Dict[str, Any]) -> bool:
    return cns_load_level(d) == "high"


def _fatigue_rank(level: str) -> int:
    return {"low": 1, "medium": 2, "moderate": 2, "high": 3}.get(norm(level).lower(), 2)


def _pick_by_filter(
    pool: List[Dict[str, Any]],
    rnd: random.Random,
    n: int,
    focus_rule: Optional[Dict[str, Any]] = None,
    avoid_ids: Optional[set] = None,
    pred=None,
) -> List[Dict[str, Any]]:
    candidates = pool
    if pred is not None:
        candidates = [d for d in pool if pred(d)]
    if not candidates:
        return []
    return pick_n(candidates, n=min(n, len(candidates)), rnd=rnd, focus_rule=focus_rule, avoid_ids=avoid_ids)


def _avoid_movement_pattern(pool: List[Dict[str, Any]], mp_to_avoid: str) -> List[Dict[str, Any]]:
    if not mp_to_avoid or mp_to_avoid == "unknown":
        return pool
    out = [d for d in pool if movement_pattern(d) != mp_to_avoid]
    return out if out else pool


# ------------------------------
# Rep guardrails (light clamps)
# ------------------------------
REP_LIMITS: Dict[str, Tuple[int, int]] = {
    "power": (1, 5),
    "max_strength": (2, 6),
    "hypertrophy": (6, 15),
    "strength_endurance": (10, 25),
    "youth": (6, 12),
}


def _split_rep_suffix(rep_str: str) -> Tuple[str, str]:
    s = norm(rep_str)
    lower = s.lower()
    for suf in ("/side", " per side", " each side"):
        if suf in lower:
            i = lower.find(suf)
            return s[:i].strip(), s[i:].strip()
    return s, ""


def _parse_rep_numbers(rep_part: str) -> Optional[Tuple[int, int]]:
    t = norm(rep_part).strip()
    if not t:
        return None
    if any(x in t.lower() for x in ("s", "sec", "seconds", "min")):
        return None
    m = re.match(r"^\s*(\d+)\s*[-–]\s*(\d+)\s*$", t)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi
    m = re.match(r"^\s*(\d+)\s*$", t)
    if m:
        v = int(m.group(1))
        return v, v
    return None


def _format_rep_range(lo: int, hi: int) -> str:
    return f"{lo}" if lo == hi else f"{lo}-{hi}"


def _apply_rep_guardrails(d: Dict[str, Any], reps: str) -> str:
    sf = strength_focus(d)
    if sf not in REP_LIMITS:
        return reps

    rep_part, suffix = _split_rep_suffix(reps)
    parsed = _parse_rep_numbers(rep_part)
    if parsed is None:
        return reps

    lo, hi = parsed
    min_r, max_r = REP_LIMITS[sf]
    lo = clamp(lo, min_r, max_r)
    hi = clamp(hi, min_r, max_r)
    if lo > hi:
        lo, hi = hi, lo
    return (_format_rep_range(lo, hi) + (" " + suffix if suffix else "")).strip()


def _apply_strength_emphasis_guardrails(emphasis: str, fatigue_role: str, reps: str) -> str:
    rep_part, suffix = _split_rep_suffix(reps)
    parsed = _parse_rep_numbers(rep_part)
    if parsed is None:
        return reps

    e = _normalize_strength_emphasis(emphasis)
    role = fatigue_role

    LIMITS = {
        "power": {FATIGUE_ROLE_HIGH: (2, 6), FATIGUE_ROLE_SECONDARY: (3, 6), FATIGUE_ROLE_RESILIENCE: (6, 10)},
        "strength": {FATIGUE_ROLE_HIGH: (3, 7), FATIGUE_ROLE_SECONDARY: (5, 10), FATIGUE_ROLE_RESILIENCE: (8, 12)},
        "hypertrophy": {FATIGUE_ROLE_HIGH: (6, 10), FATIGUE_ROLE_SECONDARY: (8, 14), FATIGUE_ROLE_RESILIENCE: (12, 20)},
        "recovery": {FATIGUE_ROLE_HIGH: (0, 0), FATIGUE_ROLE_SECONDARY: (10, 15), FATIGUE_ROLE_RESILIENCE: (0, 0)},
    }

    lo, hi = parsed
    band = LIMITS.get(e, LIMITS["strength"]).get(role)
    if not band or band == (0, 0):
        return reps

    min_r, max_r = band
    lo = clamp(lo, min_r, max_r)
    hi = clamp(hi, min_r, max_r)
    if lo > hi:
        lo, hi = hi, lo

    return (_format_rep_range(lo, hi) + (" " + suffix if suffix else "")).strip()


def format_strength_drill_with_prescription(d: Dict[str, Any], sets: Any, reps: str, rest_sec: Optional[int] = None) -> str:
    did = norm(get(d, "id", default=""))
    name = norm(get(d, "name", default="(unnamed)"))
    cues = norm(get(d, "coaching_cues", default=""))
    steps = norm(get(d, "step_by_step", default=""))
    rx = f"{sets} x {reps}"
    line = f"- {did} {name} | {rx}".strip()
    if rest_sec:
        line += f" | Rest {rest_sec}s"
    if cues:
        line += f"\n  Cues: {cues}"
    if steps:
        line += f"\n  Steps: {steps}"
    return line


# ------------------------------
# Bodyweight circuits (no-gym strength)
# ------------------------------
def _blob_for_fields(d: Dict[str, Any], fields: List[str]) -> str:
    parts: List[str] = []
    for f in fields:
        v = d.get(f)
        if v is None:
            continue
        if isinstance(v, list):
            parts.extend([str(x) for x in v if norm(x)])
        else:
            parts.append(str(v))
    return " ".join(parts).lower()


def is_bodyweight_strength_drill(d: Dict[str, Any]) -> bool:
    did = norm(get(d, "id", default="")).upper()
    if did.startswith("BW_"):
        return True
    eq = norm(get(d, "equipment", default="")).lower()
    if eq in ("", "none", "bodyweight", "no equipment"):
        return True
    blob = _blob_for_fields(d, ["name", "equipment", "tags", "bucket", "strength_bucket", "movement_pattern", "step_by_step", "coaching_cues"])
    return any(tok in blob for tok in ("bodyweight", "no equipment", "push-up", "push up", "plank", "isometric"))


def _bw_category(d: Dict[str, Any]) -> str:
    mp = movement_pattern(d)
    if mp in ("squat", "hinge", "lunge", "carry"):
        return "lower"
    if mp in ("push", "pull"):
        return "upper"
    if is_stability_candidate(d):
        return "core"
    name = norm(get(d, "name", "")).lower()
    if any(k in name for k in ("squat", "lunge", "hinge", "glute", "hamstring")):
        return "lower"
    if any(k in name for k in ("push", "press", "row", "pull")):
        return "upper"
    return "misc"


def _unique_by_name(drills: List[Dict[str, Any]], avoid_names: set) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in drills:
        nm = norm(get(d, "name", "")).strip().lower()
        if not nm or nm in avoid_names:
            continue
        out.append(d)
    return out


def pick_bw_circuit(
    pool: List[Dict[str, Any]],
    rnd: random.Random,
    target_len: int,
    prefer: List[str],
    avoid_names: Optional[set] = None,
    focus_rule: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    avoid_names = set(avoid_names or set())
    picked: List[Dict[str, Any]] = []

    by_cat: Dict[str, List[Dict[str, Any]]] = {"lower": [], "upper": [], "core": [], "misc": []}
    for d in pool:
        by_cat[_bw_category(d)].append(d)

    def pick_one(cat: str) -> Optional[Dict[str, Any]]:
        candidates = _unique_by_name(by_cat.get(cat, []), avoid_names)
        if not candidates:
            return None
        chosen = pick_n(candidates, n=1, rnd=rnd, focus_rule=focus_rule)[0]
        avoid_names.add(norm(get(chosen, "name", "")).lower())
        return chosen

    for cat in prefer:
        if len(picked) >= target_len:
            break
        p = pick_one(cat)
        if p:
            picked.append(p)

    all_candidates = _unique_by_name(pool, avoid_names)
    while len(picked) < target_len and all_candidates:
        nxt = pick_n(all_candidates, n=1, rnd=rnd, focus_rule=focus_rule)[0]
        picked.append(nxt)
        avoid_names.add(norm(get(nxt, "name", "")).lower())
        all_candidates = _unique_by_name(pool, avoid_names)

    return picked


def pick_preset_bw_circuits(
    circuits: List[Dict[str, Any]],
    age: int,
    rnd: random.Random,
    k: int,
    prefer_session_types: Optional[List[str]] = None,
    recent_circuit_ids: Optional[set] = None,
    recent_penalty: float = 0.10,
) -> List[Dict[str, Any]]:
    active = [c for c in (circuits or []) if is_active(c) and age_ok(c, age)]
    if not active:
        return []

    pool = active
    if prefer_session_types:
        preferred = [c for c in active if norm(get(c, "session_type", "")).lower() in prefer_session_types]
        if preferred:
            pool = preferred

    recent = set(recent_circuit_ids or [])
    penalty = max(0.0, min(1.0, float(recent_penalty) if recent_penalty is not None else 0.10))

    chosen: List[Dict[str, Any]] = []
    candidates = list(pool)
    for _ in range(min(int(k or 0), len(candidates))):
        weights = []
        for c in candidates:
            cid = norm(get(c, "id", ""))
            w = penalty if (cid and cid in recent) else 1.0
            weights.append(max(0.0, float(w)))

        total = sum(weights)
        if total <= 0:
            rnd.shuffle(candidates)
            chosen.append(candidates.pop(0))
            continue

        r = rnd.random() * total
        acc = 0.0
        pick_i = 0
        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                pick_i = i
                break
        chosen.append(candidates.pop(pick_i))

    return chosen


def render_preset_circuit(circuit: Dict[str, Any], strength_by_id: Dict[str, Dict[str, Any]]) -> List[str]:
    fmt = circuit.get("format", {}) or {}
    rounds = int(fmt.get("rounds", 3) or 3)
    work = fmt.get("work_sec")
    rest = fmt.get("rest_sec")

    lines: List[str] = []
    title = norm(get(circuit, "name", "Circuit"))
    cid = norm(get(circuit, "id", ""))
    lines.append(f"{title} ({cid})" if cid else title)

    if work is not None and rest is not None:
        lines.append(f"Format: {rounds} rounds | {int(work)}s work / {int(rest)}s rest")
    else:
        lines.append(f"Format: {rounds} rounds")

    for item in (circuit.get("drills", []) or []):
        did = item.get("id") if isinstance(item, dict) else item
        did = norm(did)
        d = strength_by_id.get(did)
        if not d:
            lines.append(f"- [Missing drill in strength.json: {did}]")
        else:
            lines.append(format_drill(d))

    return lines


def build_bw_strength_circuits(
    strength_drills: List[Dict[str, Any]],
    warmups: List[Dict[str, Any]],
    mobility_drills: List[Dict[str, Any]],
    conditioning_drills: List[Dict[str, Any]],
    circuits: List[Dict[str, Any]],
    age: int,
    rnd: random.Random,
    day_type: str,
    session_len_min: int,
    focus_rule: Optional[Dict[str, Any]] = None,
    include_finisher: Optional[bool] = None,
    skate_within_24h: bool = False,
) -> List[str]:
    lines: List[str] = []

    wu = build_strength_warmup(warmups, age, rnd, day_type=day_type)
    lines.append(f"\nWARMUP (Strength Circuits - {day_type})")
    if wu:
        for d in wu[:10]:
            lines.append(format_drill(d))
    else:
        lines.append("- [No warmups found]")

    strength_by_id = {norm(get(d, "id", "")): d for d in strength_drills}

    want_k = 1 if session_len_min <= 45 else 2
    presets = pick_preset_bw_circuits(
        circuits=circuits,
        age=age,
        rnd=rnd,
        k=want_k,
        prefer_session_types=["bodyweight_circuit"],
        recent_circuit_ids=_CURRENT_RECENT_CIRCUIT_IDS,
        recent_penalty=_CURRENT_RECENT_PENALTY,
    )

    # prevent exact back-to-back repeat signature
    last_sig = _CURRENT_LAST_CIRCUIT_SIGNATURE
    if presets and last_sig:
        cur_sig = tuple(sorted([norm(get(c, "id", "")) for c in presets if norm(get(c, "id", ""))]))
        if cur_sig == last_sig:
            pool_active = [c for c in (circuits or []) if is_active(c) and age_ok(c, age)]
            if want_k == 1 and len(pool_active) > 1:
                alt_pool = [c for c in pool_active if norm(get(c, "id", "")) not in set(last_sig)]
                if alt_pool:
                    presets = pick_preset_bw_circuits(
                        circuits=alt_pool,
                        age=age,
                        rnd=rnd,
                        k=want_k,
                        prefer_session_types=["bodyweight_circuit"],
                        recent_circuit_ids=_CURRENT_RECENT_CIRCUIT_IDS,
                        recent_penalty=_CURRENT_RECENT_PENALTY,
                    )

    if presets:
        lines.append("\nSTRENGTH CIRCUITS (Preset)")
        for i, c in enumerate(presets, start=1):
            lines.append("")
            lines.append("CIRCUIT " + ("A" if i == 1 else "B"))

            cc = dict(c)
            if skate_within_24h:
                fmt = cc.get("format", {}) or {}
                if "rounds" in fmt and isinstance(fmt.get("rounds"), int):
                    fmt["rounds"] = min(int(fmt.get("rounds", 2)), 2)
                cc["format"] = fmt

            lines.extend(render_preset_circuit(cc, strength_by_id))
    else:
        bw_pool = [d for d in strength_drills if is_active(d) and age_ok(d, age) and is_bodyweight_strength_drill(d)]
        lines.append("\nSTRENGTH CIRCUITS")
        if not bw_pool:
            lines.append("- [No matching bodyweight strength drills found — make sure BW drills are in strength.json]")
            return lines

        if session_len_min <= 30:
            a_len, b_len, rounds = 4, 0, 3
        elif session_len_min <= 45:
            a_len, b_len, rounds = 5, 0, 3
        else:
            a_len, b_len, rounds = 5, 5, 4

        if day_type == "leg":
            prefer_a = ["lower", "upper", "core", "lower"]
            prefer_b = ["lower", "upper", "core", "misc"]
        elif day_type == "upper":
            prefer_a = ["upper", "lower", "core", "upper"]
            prefer_b = ["upper", "lower", "core", "misc"]
        else:
            prefer_a = ["lower", "upper", "core", "misc"]
            prefer_b = ["lower", "upper", "core", "misc"]

        avoid_names: set = set()
        circuit_a = pick_bw_circuit(bw_pool, rnd, a_len, prefer_a, avoid_names=avoid_names, focus_rule=focus_rule)

        if skate_within_24h:
            rounds = min(rounds, 2)

        work = 40 if age >= 13 else 30
        rest = 20 if age >= 13 else 30
        lines.append(f"\nCIRCUIT A (repeat {rounds} rounds)")
        lines.append(f"Format: {work}s work / {rest}s rest")
        for d in circuit_a:
            lines.append(format_drill(d))

        if b_len > 0:
            avoid_names |= {norm(get(d, "name", "")).lower() for d in circuit_a}
            circuit_b = pick_bw_circuit(bw_pool, rnd, b_len, prefer_b, avoid_names=avoid_names, focus_rule=focus_rule)
            lines.append(f"\nCIRCUIT B (repeat {rounds} rounds)")
            lines.append(f"Format: {work}s work / {rest}s rest")
            for d in circuit_b:
                lines.append(format_drill(d))

    # optional post-lift conditioning (no-gym rules)
    if include_finisher:
        fin_min = prof["finisher_min"]
        if include_finisher and fin_min > 0 and not skate_within_24h:
        cond_pool = filter_post_lift_conditioning_pool(conditioning_drills, full_gym=False, post_lift_conditioning_type=None)
        fin_drills = pick_conditioning_drills(cond_pool, age, rnd, fin_min, focus_rule=get_focus_rules(None, "conditioning"))
        lines.append(f"\nPOST-LIFT CONDITIONING (optional, ~{fin_min} min)")
        if not fin_drills:
            lines.append("- [No matching no-equipment conditioning drills found]")
        else:
            lines.extend(build_conditioning_block(fin_drills, fin_min * 60))

    # mobility cooldown (required)
    m = pick_mobility_drills(mobility_drills, age, rnd, n=3, focus_rule=get_focus_rules(None, "mobility"))
    lines.append("\nMOBILITY COOLDOWN CIRCUIT")
    if not m:
        lines.append("- [No mobility drills found]")
    else:
        lines.append("- Perform 2 rounds")
        for d in m:
            did = norm(get(d, "id", ""))
            name = norm(get(d, "name", "(unnamed)"))
            lines.append(f"- {did} {name} (30–45s)")

    return lines


# ------------------------------
# Strength session builder (full gym template)
# ------------------------------
def build_hockey_strength_session(
    strength_drills: List[Dict[str, Any]],
    warmups: List[Dict[str, Any]],
    mobility_drills: List[Dict[str, Any]],
    conditioning_drills: List[Dict[str, Any]],
    circuits: List[Dict[str, Any]],
    age: int,
    rnd: random.Random,
    day_type: str,
    session_len_min: int,
    emphasis: str = "strength",
    focus_rule: Optional[Dict[str, Any]] = None,
    include_finisher: Optional[bool] = None,
    full_gym: bool = True,
    post_lift_conditioning_type: Optional[str] = None,
    skate_within_24h: bool = False,
) -> List[str]:
    # No gym: circuits only
    if not full_gym:
        return build_bw_strength_circuits(
            strength_drills=strength_drills,
            warmups=warmups,
            mobility_drills=mobility_drills,
            conditioning_drills=conditioning_drills,
            circuits=circuits,
            age=age,
            rnd=rnd,
            day_type=day_type,
            session_len_min=session_len_min,
            focus_rule=focus_rule,
            include_finisher=include_finisher,
            skate_within_24h=skate_within_24h,
        )

    emphasis = _normalize_strength_emphasis(emphasis)
    lines: List[str] = []
    prof = _strength_time_profile(session_len_min, skate_within_24h)

    # Warm-up
    warmup_drills_picked = build_strength_warmup(warmups, age, rnd, day_type=day_type)
    lines.append(f"\nWARMUP (strength - {day_type} day)")
    for d in warmup_drills_picked[:prof["warmup_cap"]]:
        lines.append(format_drill(d))


    pool = [d for d in strength_drills if is_active(d) and age_ok(d, age)]
    day_pool = [d for d in pool if _region_ok_for_day(d, day_type)]

    # Gym rule: avoid bodyweight lifts in main strength selections
    day_pool = [d for d in day_pool if not _is_bodyweightish(d)]

    if not pool:
        lines.append("\nSTRENGTH")
        lines.append("- [No matching strength drills found — check strength.json]")
        return lines

    used_ids: set = set()
    upper_strength_picks: List[Dict[str, Any]] = []
    used_upper_subpatterns: set = set()

    # SPEED / POWER (time-aware)
    speed_picks: List[Dict[str, Any]] = []

    if prof["speed"] > 0:
        speed_pool = [
            d for d in day_pool
            if strength_focus(d) == "power"
            and norm(get(d, "id", "")) not in used_ids
        ]

        if skate_within_24h:
            speed_pool = [
                d for d in speed_pool
                if not _cns_is_high(d)
            and _fatigue_rank(fatigue_cost_level(d)) <= 2
            ]

        speed_count = prof["speed"]

        if speed_pool and speed_count > 0:
            first = _pick_by_filter(speed_pool, rnd, 1, focus_rule=focus_rule, avoid_ids=used_ids)
            if first:
                speed_picks += first
                used_ids.add(norm(get(first[0], "id", "")))

            if speed_count > 1:
                second_pool = [d for d in speed_pool if norm(get(d, "id", "")) not in used_ids]
                second = _pick_by_filter(second_pool, rnd, 1, focus_rule=focus_rule, avoid_ids=used_ids)
                if second:
                    speed_picks += second
                    used_ids.add(norm(get(second[0], "id", "")))

        lines.append("\nSPEED / POWER (1–2 drills)")
        if not speed_picks:
            lines.append("- [No speed/power drills found — continuing]")
        else:
            for d in speed_picks:
                role = _fatigue_role_for_speed_drill(d)
               rx = _rx_for(emphasis, role)
                if rx:
                    reps = _apply_strength_emphasis_guardrails(emphasis, role, rx["reps"])
                    lines.append(
                        format_strength_drill_with_prescription(
                            d, sets=rx["sets"], reps=reps, rest_sec=120
                        )
                    )


    # HIGH FATIGUE (1)
    hf_pool = [d for d in day_pool if norm(get(d, "id", "")) not in used_ids]
    if skate_within_24h:
        hf_pool = [d for d in hf_pool if _fatigue_rank(fatigue_cost_level(d)) <= 2 and not _cns_is_high(d)]

    # prefer fatigue_cost high, else compound-ish
    high_fatigue = [d for d in hf_pool if fatigue_cost_level(d) == "high"]
    compound = [d for d in hf_pool if movement_pattern(d) in ("squat", "hinge", "push", "pull", "lunge", "carry")]

    # Upper day: prefer real upper push/pull strength/power for the high-fatigue lift when possible
    if _is_upper_day(day_type):
        upper_hf = [
            d for d in hf_pool  
            if _is_push_pull(movement_pattern(d))
            and strength_focus(d) in ("max_strength", "power", "strength")
        ]
        upper_hf_high = [d for d in upper_hf if fatigue_cost_level(d) == "high"]
        upper_hf_comp = [d for d in upper_hf if movement_pattern(d) in ("push", "pull")]

        hf_candidates = upper_hf_high or upper_hf or high_fatigue or compound or hf_pool
    else:
        hf_candidates = high_fatigue or compound or hf_pool



    mp_avoid = movement_pattern(speed_picks[-1]) if speed_picks else ""
    hf_candidates = _avoid_movement_pattern(hf_candidates, mp_avoid)

    hf_pick = _pick_by_filter(hf_candidates, rnd, 1, focus_rule=focus_rule, avoid_ids=used_ids)
    if hf_pick and _is_upper_day(day_type):
        upper_strength_picks.append(hf_pick[0])
    if hf_pick and _is_upper_day(day_type):
        used_upper_subpatterns.add(_upper_subpattern(hf_pick[0]))

    hf_dir = _upper_direction(hf_pick[0]) if hf_pick and _is_upper_day(day_type) else None

    lines.append("\nHIGH FATIGUE (1 exercise)")
    if not hf_pick:
        lines.append("- [No high-fatigue lift found]")
    else:
        d = hf_pick[0]
        used_ids.add(norm(get(d, "id", "")))
        rx = _rx_for(emphasis, FATIGUE_ROLE_HIGH)
        if rx is None:
            lines.append("- [Recovery emphasis: skipping high-fatigue lift]")
        else:
            reps = _apply_strength_emphasis_guardrails(emphasis, FATIGUE_ROLE_HIGH, rx["reps"])
            lines.append(format_strength_drill_with_prescription(d, sets=rx["sets"], reps=reps, rest_sec=180))

    # SCAP / SHOULDER HEALTH (upper days only)
    scap_pool: List[Dict[str, Any]] = []
    if _is_upper_day(day_type):
        scap_pool = [
            d for d in pool
            if is_scap_accessory(d)
            and norm(get(d, "id", "")) not in used_ids
        ]
        rnd.shuffle(scap_pool)


    # SECONDARY pool
    sec_pool = [d for d in day_pool if norm(get(d, "id", "")) not in used_ids]
    # prefer non-power, non-stability “strength-ish”
    sec_pool = [d for d in sec_pool if strength_focus(d) in ("hypertrophy", "strength_endurance", "strength", "max_strength") and not is_stability_candidate(d)]

    if hf_pick:
        sec_pool = _avoid_movement_pattern(sec_pool, movement_pattern(hf_pick[0]))

    # Upper day: prefer the opposite of the high-fatigue push/pull (to guarantee balance)
    hf_mp = movement_pattern(hf_pick[0]) if hf_pick else ""
    needed_mp = _opposing_push_pull(hf_mp) if _is_upper_day(day_type) else None

    # RESILIENCE pool (not region-limited)
    res_pool = [
        d for d in pool
        if norm(get(d, "id", "")) not in used_ids
        and is_stability_candidate(d)
    ]

    # Upper day: keep scap/shoulder-health accessories OUT of resilience pool
    # (scap accessory is guaranteed later in its own section)
    if _is_upper_day(day_type):
        res_pool = [d for d in res_pool if not is_scap_accessory(d)]

    rnd.shuffle(res_pool)

    # Pick resilience A + B (and mark used so we don't repeat)
    res_a: List[Dict[str, Any]] = _pick_by_filter(res_pool, rnd, 1, focus_rule=focus_rule, avoid_ids=used_ids)
    if res_a:
        used_ids.add(norm(get(res_a[0], "id", "")))

    res_b: List[Dict[str, Any]] = []

    if prof["blocks"] >= 2:
        res_pool_b = [d for d in res_pool if norm(get(d, "id", "")) not in used_ids]
        res_b = _pick_by_filter(res_pool_b, rnd, 1, focus_rule=focus_rule, avoid_ids=used_ids)
        if res_b:
            used_ids.add(norm(get(res_b[0], "id", "")))

    sec_a: List[Dict[str, Any]] = []
    sec_b: List[Dict[str, Any]] = []

    # --- SEC A ---
    if _is_upper_day(day_type) and needed_mp:
        sec_pool_needed = [
            d for d in sec_pool
            if movement_pattern(d) == needed_mp
            and _upper_direction(d) != hf_dir
            and _upper_subpattern(d) not in used_upper_subpatterns
        ]
        sec_pool_needed_fallback = [
            d for d in sec_pool
            if movement_pattern(d) == needed_mp
            and _upper_direction(d) != hf_dir
        ]

        sec_a = (
            _pick_by_filter(sec_pool_needed, rnd, 1, focus_rule=focus_rule, avoid_ids=used_ids)
            or _pick_by_filter(sec_pool_needed_fallback, rnd, 1, focus_rule=focus_rule, avoid_ids=used_ids)
            or _pick_by_filter(sec_pool, rnd, 1, focus_rule=focus_rule, avoid_ids=used_ids)
        )
    else:
        sec_a = _pick_by_filter(sec_pool, rnd, 1, focus_rule=focus_rule, avoid_ids=used_ids)

    # After SEC A is chosen, always mark used + track upper variety
    if sec_a:
        used_ids.add(norm(get(sec_a[0], "id", "")))
    if sec_a and _is_upper_day(day_type):
        upper_strength_picks.append(sec_a[0])
        used_upper_subpatterns.add(_upper_subpattern(sec_a[0]))

    # Upper day: check push/pull coverage so far (HF + SEC A)
    have_push = False
    have_pull = False

    if _is_upper_day(day_type):
        chosen_mps = []
        if hf_pick:
            chosen_mps.append(movement_pattern(hf_pick[0]))
        if sec_a:
            chosen_mps.append(movement_pattern(sec_a[0]))

        have_push = any(mp == "push" for mp in chosen_mps)
        have_pull = any(mp == "pull" for mp in chosen_mps)

    sec_b: List[Dict[str, Any]] = []

    if prof["blocks"] >= 2:
        # --- SEC B ---
        sec_pool_b = [d for d in sec_pool if norm(get(d, "id", "")) not in used_ids]

        if _is_upper_day(day_type) and hf_pick:
            if _is_heavy_vertical(hf_pick[0]):
                sec_pool_b = [d for d in sec_pool_b if not _is_heavy_vertical(d)]

        if _is_upper_day(day_type):
            missing_mp = None
            if not have_push:
                missing_mp = "push"
            elif not have_pull:
                missing_mp = "pull"

            if missing_mp:
                sec_pool_missing = [
                    d for d in sec_pool_b
                    if movement_pattern(d) == missing_mp
                    and _upper_subpattern(d) not in used_upper_subpatterns
                ]
                sec_pool_missing_fallback = [
                    d for d in sec_pool_b
                    if movement_pattern(d) == missing_mp
                ]

                sec_b = (
                    _pick_by_filter(sec_pool_missing, rnd, 1, focus_rule=focus_rule, avoid_ids=used_ids)
                    or _pick_by_filter(sec_pool_missing_fallback, rnd, 1, focus_rule=focus_rule, avoid_ids=used_ids)
                    or _pick_by_filter(sec_pool_b, rnd, 1, focus_rule=focus_rule, avoid_ids=used_ids)
                )
            else:
                mp_a = movement_pattern(sec_a[0]) if sec_a else ""
                sec_pool_diff = _avoid_movement_pattern(sec_pool_b, mp_a) if mp_a else sec_pool_b
                sec_pool_diff_var = [
                    d for d in sec_pool_diff
                    if _upper_subpattern(d) not in used_upper_subpatterns
                ]

                sec_b = (
                    _pick_by_filter(sec_pool_diff_var, rnd, 1, focus_rule=focus_rule, avoid_ids=used_ids)
                    or _pick_by_filter(sec_pool_diff, rnd, 1, focus_rule=focus_rule, avoid_ids=used_ids)
                    or _pick_by_filter(sec_pool_b, rnd, 1, focus_rule=focus_rule, avoid_ids=used_ids)
                )

        else:
            sec_b = _pick_by_filter(sec_pool_b, rnd, 1, focus_rule=focus_rule, avoid_ids=used_ids) or sec_a

        if sec_b:
            used_ids.add(norm(get(sec_b[0], "id", "")))
        if sec_b and _is_upper_day(day_type):
            upper_strength_picks.append(sec_b[0])
            used_upper_subpatterns.add(_upper_subpattern(sec_b[0]))


    # Render Blocks
    lines.append("\nBLOCK A (Secondary + Resilience)")
    if not sec_a:
        lines.append("- [No secondary strength lift found]")
    else:
        d = sec_a[0]
        rx = _rx_for(emphasis, FATIGUE_ROLE_SECONDARY) or _rx_for("strength", FATIGUE_ROLE_SECONDARY)
        reps = _apply_strength_emphasis_guardrails(emphasis, FATIGUE_ROLE_SECONDARY, rx["reps"])
        lines.append(format_strength_drill_with_prescription(d, sets=rx["sets"], reps=reps, rest_sec=90))

    if not res_a:
        lines.append("- [No resilience drill found]")
    else:
        d = res_a[0]
        rx = _rx_for(emphasis, FATIGUE_ROLE_RESILIENCE) or _rx_for("strength", FATIGUE_ROLE_RESILIENCE)
        reps = _apply_strength_emphasis_guardrails(emphasis, FATIGUE_ROLE_RESILIENCE, rx["reps"])
        lines.append(format_strength_drill_with_prescription(d, sets=rx["sets"], reps=reps, rest_sec=45))

    if prof["blocks"] >= 2:
        lines.append("\nBLOCK B (Secondary + Resilience)")

        if not sec_b:
            lines.append("- [No secondary strength lift found]")
        else:
            d = sec_b[0]
            rx = _rx_for(emphasis, FATIGUE_ROLE_SECONDARY) or _rx_for("strength", FATIGUE_ROLE_SECONDARY)
            reps = _apply_strength_emphasis_guardrails(emphasis, FATIGUE_ROLE_SECONDARY, rx["reps"])
            lines.append(format_strength_drill_with_prescription(d, sets=rx["sets"], reps=reps, rest_sec=90))

        if not res_b:
            lines.append("- [No resilience drill found]")
        else:
            d = res_b[0]
            rx = _rx_for(emphasis, FATIGUE_ROLE_RESILIENCE) or _rx_for("strength", FATIGUE_ROLE_RESILIENCE)
            reps = _apply_strength_emphasis_guardrails(emphasis, FATIGUE_ROLE_RESILIENCE, rx["reps"])
            lines.append(format_strength_drill_with_prescription(d, sets=rx["sets"], reps=reps, rest_sec=45))

    if not res_b:
        lines.append("- [No resilience drill found]")
    else:
        d = res_b[0]
        rx = _rx_for(emphasis, FATIGUE_ROLE_RESILIENCE) or _rx_for("strength", FATIGUE_ROLE_RESILIENCE)
        reps = _apply_strength_emphasis_guardrails(emphasis, FATIGUE_ROLE_RESILIENCE, rx["reps"])
        lines.append(format_strength_drill_with_prescription(d, sets=rx["sets"], reps=reps, rest_sec=45))

    # SCAP / SHOULDER HEALTH ACCESSORY (guaranteed 1 on upper days)
    if _is_upper_day(day_type):
        lines.append("\nSCAP / SHOULDER HEALTH")
        if not scap_pool:
            lines.append("- [No scap / shoulder-health accessory found]")
        else:
            d = scap_pool[0]
            used_ids.add(norm(get(d, "id", "")))

            # Use resilience-style RX (low stress, controlled)
            rx = _rx_for(emphasis, FATIGUE_ROLE_RESILIENCE) or _rx_for("strength", FATIGUE_ROLE_RESILIENCE)
            reps = _apply_strength_emphasis_guardrails(emphasis, FATIGUE_ROLE_RESILIENCE, rx["reps"])

            lines.append(
                format_strength_drill_with_prescription(
                    d,
                    sets=rx["sets"],
                    reps=reps,
                    rest_sec=45
                )
            )
 
    # PUSH / PULL SAFETY NET (upper days only)
    if _is_upper_day(day_type):
        counts = _count_push_pull(upper_strength_picks)

        if counts["push"] == 0 or counts["pull"] == 0:
            missing_mp = "pull" if counts["push"] > 0 else "push"

            fallback_pool = [
                d for d in pool
                if movement_pattern(d) == missing_mp
                and not is_stability_candidate(d)
                and norm(get(d, "id", "")) not in used_ids
            ]

            rnd.shuffle(fallback_pool)

            if fallback_pool:
                d = fallback_pool[0]
                used_ids.add(norm(get(d, "id", "")))
                upper_strength_picks.append(d)
 

                rx = _rx_for(emphasis, FATIGUE_ROLE_SECONDARY) or _rx_for("strength", FATIGUE_ROLE_SECONDARY)
                reps = _apply_strength_emphasis_guardrails(emphasis, FATIGUE_ROLE_SECONDARY, rx["reps"])

                lines.append("\nPUSH / PULL BALANCE (auto-added)")
                lines.append(
                    format_strength_drill_with_prescription(
                        d,
                        sets=rx["sets"],
                        reps=reps,
                        rest_sec=75
                    )
                )

    # Optional Post-Lift Conditioning Finisher (Strength sessions only)
    if include_finisher and not skate_within_24h:
        fin_min = 8 if session_len_min >= 60 else 6
        cond_pool = filter_post_lift_conditioning_pool(
            conditioning_drills,
            full_gym=True,
            post_lift_conditioning_type=post_lift_conditioning_type,
        )
        fin_drills = pick_conditioning_drills(cond_pool, age, rnd, fin_min, focus_rule=get_focus_rules(None, "conditioning"))
        lines.append(f"\nPOST-LIFT CONDITIONING (optional, ~{fin_min} min)")
        if not fin_drills:
            lines.append("- [No conditioning drills found]")
        else:
            lines.extend(build_conditioning_block(fin_drills, fin_min * 60))

    # Mobility Cooldown Circuit (Required)
    m = pick_mobility_drills(mobility_drills, age, rnd, n=prof["mobility_n"], focus_rule=get_focus_rules(None, "mobility"))
    lines.append("\nMOBILITY COOLDOWN CIRCUIT")
    if not m:
        lines.append("- [No mobility drills found]")
    else:
        lines.append("- Perform 2 rounds")
        for d in m:
            did = norm(get(d, "id", ""))
            name = norm(get(d, "name", "(unnamed)"))
            cues = norm(get(d, "coaching_cues", default=""))
            steps = norm(get(d, "step_by_step", default=""))
            lines.append(f"- {did} {name} (30–45s)")
            if cues:
                lines.append(f"  Cues: {cues}")
            if steps:
                lines.append(f"  Steps: {steps}")

    return lines


# ------------------------------
# Allocate blocks
# ------------------------------
def allocate_blocks(session_mode: str, total_sec: int) -> List[Tuple[str, int]]:
    session_mode = (session_mode or "").strip().lower()

    if session_mode in ("skills_mixed", "skills_only"):
        a = int(total_sec * 0.50)
        b = total_sec - a
        return [("stickhandling", a), ("shooting", b)]

    if session_mode in ("stickhandling", "shooting", "mobility", "recovery"):
        return [(session_mode, total_sec)]

    if session_mode == "conditioning":
        cond = int(total_sec * 0.80)
        mob = total_sec - cond
        return [("conditioning", cond), ("mobility", mob)]

    if session_mode == "movement":
        wu = int(total_sec * 0.20)
        mv = int(total_sec * 0.65)
        mob = total_sec - wu - mv
        return [("warmup", wu), ("movement", mv), ("mobility", mob)]

    if session_mode == "strength":
        return [("strength", total_sec)]

    a = int(total_sec * 0.50)
    b = total_sec - a
    return [("stickhandling", a), ("shooting", b)]


# ------------------------------
# History / memory
# ------------------------------
def _history_dir() -> str:
    path = os.path.join(DATA_DIR, "history")
    os.makedirs(path, exist_ok=True)
    return path


def _history_path(athlete_id: str) -> str:
    aid = norm(athlete_id) or "default"
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", aid)
    return os.path.join(_history_dir(), f"{safe}.json")


def load_history(athlete_id: str) -> Dict[str, Any]:
    path = _history_path(athlete_id)
    if not os.path.exists(path):
        return {"athlete_id": norm(athlete_id) or "default", "sessions": []}
    try:
        return load_json(path)
    except Exception:
        return {"athlete_id": norm(athlete_id) or "default", "sessions": []}


def save_history(athlete_id: str, history: Dict[str, Any]) -> None:
    path = _history_path(athlete_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def recent_ids_from_history(history: Dict[str, Any], max_sessions: int = 6, mode: Optional[str] = None) -> set:
    sessions = history.get("sessions") or []
    if mode:
        sessions = [s for s in sessions if norm(s.get("mode", "")) == norm(mode)]
    take = sessions[-max_sessions:] if max_sessions and max_sessions > 0 else sessions
    out = set()
    for s in take:
        for did in (s.get("drill_ids") or []):
            didn = norm(did)
            if didn:
                out.add(didn)
    return out


def recent_circuit_ids_from_history(history: Dict[str, Any], max_sessions: int = 6, mode: Optional[str] = None) -> set:
    sessions = history.get("sessions") or []
    if mode:
        sessions = [s for s in sessions if norm(s.get("mode", "")) == norm(mode)]
    take = sessions[-max_sessions:] if max_sessions and max_sessions > 0 else sessions
    out = set()
    for s in take:
        for cid in (s.get("circuit_ids") or []):
            cidn = norm(cid)
            if cidn:
                out.add(cidn)
    return out


def last_circuit_signature_from_history(history: Dict[str, Any], mode: Optional[str] = None) -> Tuple[str, ...]:
    sessions = history.get("sessions") or []
    if mode:
        sessions = [s for s in sessions if norm(s.get("mode", "")) == norm(mode)]
    if not sessions:
        return tuple()
    last = sessions[-1]
    ids = [norm(x) for x in (last.get("circuit_ids") or []) if norm(x)]
    return tuple(sorted(ids))


_DRILL_ID_RE = re.compile(r"^\s*-\s*([A-Z]{2,10}_[0-9]{3})\b", re.M)
_CIRCUIT_ID_RE = re.compile(r"\(([A-Z]{2,10}_[0-9]{3})\)")


def extract_circuit_ids_from_plan(plan_text: str) -> List[str]:
    ids = _CIRCUIT_ID_RE.findall(plan_text or "")
    seen = set()
    out: List[str] = []
    for cid in ids:
        cidn = norm(cid)
        if cidn and cidn not in seen:
            out.append(cidn)
            seen.add(cidn)
    return out


def extract_ids_from_plan(plan_text: str) -> List[str]:
    ids = _DRILL_ID_RE.findall(plan_text or "")
    seen = set()
    out: List[str] = []
    for did in ids:
        didn = norm(did)
        if didn and didn not in seen:
            seen.add(didn)
            out.append(didn)
    return out


# ------------------------------
# Session generation
# ------------------------------
def generate_session(
    data: Dict[str, List[Dict[str, Any]]],
    age: int,
    seed: int,
    focus: Optional[str],
    session_mode: str,
    session_len_min: int,
    athlete_id: str = "default",
    use_memory: bool = True,
    memory_sessions: int = 6,
    recent_penalty: float = 0.25,
    strength_emphasis: str = "strength",
    shooting_shots: Optional[int] = None,
    stickhandling_min: Optional[int] = None,
    shooting_min: Optional[int] = None,
    strength_day_type: Optional[str] = None,
    strength_full_gym: bool = False,
    include_post_lift_conditioning: Optional[bool] = None,
    post_lift_conditioning_type: Optional[str] = None,
    skate_within_24h: bool = False,
    primary_signal: Optional[str] = None,
    **kwargs,
) -> str:
    rnd = random.Random(seed)

    global _CURRENT_RECENT_IDS, _CURRENT_RECENT_CIRCUIT_IDS, _CURRENT_RECENT_PENALTY, _CURRENT_LAST_CIRCUIT_SIGNATURE
    history = None
    if use_memory:
        history = load_history(athlete_id)
        _CURRENT_RECENT_IDS = recent_ids_from_history(history, max_sessions=memory_sessions, mode=session_mode)
        _CURRENT_RECENT_CIRCUIT_IDS = recent_circuit_ids_from_history(history, max_sessions=memory_sessions, mode=session_mode)
        _CURRENT_LAST_CIRCUIT_SIGNATURE = last_circuit_signature_from_history(history, mode=session_mode)
        _CURRENT_RECENT_PENALTY = float(recent_penalty)
    else:
        _CURRENT_RECENT_IDS = set()
        _CURRENT_RECENT_CIRCUIT_IDS = set()
        _CURRENT_RECENT_PENALTY = 1.0
        _CURRENT_LAST_CIRCUIT_SIGNATURE = tuple()

    def _finalize_and_return(plan_text: str) -> str:
        if use_memory:
            try:
                ids = extract_ids_from_plan(plan_text)
                circuit_ids = extract_circuit_ids_from_plan(plan_text)
                entry = {"ts": time.strftime("%Y-%m-%d"), "mode": session_mode, "len_min": session_len_min, "drill_ids": ids, "circuit_ids": circuit_ids}
                sessions = list((history or {}).get("sessions") or [])
                sessions.append(entry)
                sessions = sessions[-30:]
                hist_out = history or {"athlete_id": norm(athlete_id) or "default", "sessions": []}
                hist_out["athlete_id"] = norm(athlete_id) or "default"
                hist_out["sessions"] = sessions
                save_history(athlete_id, hist_out)
            except Exception:
                pass
        return plan_text

    lines: List[str] = []
    lines.append(f"\nBENDER SINGLE WORKOUT | mode={session_mode} | len={session_len_min} min | age={age} | focus={focus or 'none'}\n")

    total_sec = session_len_min * 60
    blocks = allocate_blocks(session_mode, total_sec)

    # skills_only defaults
    if session_mode == "skills_only":
        if stickhandling_min is None and shooting_min is None:
            stickhandling_min = session_len_min // 2
            shooting_min = session_len_min - stickhandling_min
        elif stickhandling_min is None:
            stickhandling_min = max(0, session_len_min - to_int(shooting_min, 0))
        elif shooting_min is None:
            shooting_min = max(0, session_len_min - to_int(stickhandling_min, 0))

        if shooting_shots is None or shooting_shots <= 0:
            shooting_shots = max(120, int(max(1, shooting_min or 1) * 12))

    # Strength: decide optional post-lift conditioning (unless user sets it)
    if include_post_lift_conditioning is None:
        include_post_lift_conditioning = (session_mode == "strength" and session_len_min >= 45)

    for category, seconds in blocks:
        if category not in data:
            continue

        focus_rule = get_focus_rules(focus, category)

        # Strength
        if session_mode == "strength" and category == "strength":
            dt = (strength_day_type or "leg").lower().strip()
            if dt in ("lower", "lower_body", "legs", "leg_day"):
                dt = "leg"
            elif dt in ("upper_body", "upper_day"):
                dt = "upper"
            elif dt not in ("leg", "upper", "full"):
                dt = "leg"


            include_finisher = include_post_lift_conditioning
            if skate_within_24h:
                include_finisher = False
                include_post_lift_conditioning = False
            # Enforce minimum strength session length
            if session_mode == "strength":
                session_len_min = max(20, session_len_min)
            strength_lines = build_hockey_strength_session(
                strength_drills=data["strength"],
                warmups=data["warmup"],
                mobility_drills=data["mobility"],
                conditioning_drills=data["conditioning"],
                circuits=data.get("circuits", []),
                age=age,
                rnd=rnd,
                day_type=dt,
                session_len_min=session_len_min,
                emphasis=strength_emphasis,
                focus_rule=focus_rule,
                include_finisher=include_finisher,
                full_gym=strength_full_gym,
                post_lift_conditioning_type=post_lift_conditioning_type,
                skate_within_24h=skate_within_24h,
            )
            return _finalize_and_return("\n".join(lines + strength_lines))

        # Shooting
        if category == "shooting":
            if (session_mode == "skills_only") or (shooting_shots is not None and shooting_shots > 0):
                minutes = max(1, seconds // 60)
                if session_mode == "skills_only" and shooting_min is not None:
                    minutes = max(1, int(shooting_min))
                chosen = choose_shooting_drills(data["shooting"], age, rnd, int(shooting_shots or 0), focus, focus_rule)
                lines.append("\nSHOOTING (shot volume)")
                if not chosen:
                    lines.append("- [No matching drills found]")
                else:
                    lines.extend(build_shooting_by_shots(chosen, int(shooting_shots)))
                continue

            # legacy default volumes
            est_total = 60 if session_len_min <= 30 else (80 if session_len_min <= 45 else 100)
            chosen = choose_shooting_drills(data["shooting"], age, rnd, est_total, focus, focus_rule)
            lines.append("\nSHOOTING (default volumes)")
            if not chosen:
                lines.append("- [No matching drills found]")
            else:
                lines.extend(build_shooting_from_defaults(chosen))
            continue

        # Conditioning
        if category == "conditioning":
            minutes = max(1, seconds // 60)
            c_drills = pick_conditioning_drills(data["conditioning"], age, rnd, minutes, focus_rule)
            lines.append(f"\nCONDITIONING (~{minutes} min)")
            lines.extend(build_conditioning_block(c_drills, seconds))
            continue

        # Mobility
        if category == "mobility":
            minutes = max(1, seconds // 60)
            if session_mode in ("mobility", "recovery"):
                n = clamp(minutes // 6, 4, 6)
                m = pick_mobility_drills(data["mobility"], age, rnd, n=n, focus_rule=focus_rule)
                lines.append(f"\nMOBILITY RESET (~{minutes} min)")
                lines.extend(build_mobility_timed_session(m, seconds))
            else:
                m = pick_mobility_drills(data["mobility"], age, rnd, n=2, focus_rule=focus_rule)
                lines.append(f"\nMOBILITY (Cooldown Circuit ~{minutes} min)")
                lines.extend(build_mobility_cooldown_circuit(m, seconds))
            continue

        # Stickhandling
        if category == "stickhandling":
            minutes = max(1, seconds // 60)
            if session_mode == "skills_only" and stickhandling_min is not None:
                minutes = max(1, int(stickhandling_min))
            drills = [d for d in data["stickhandling"] if is_active(d) and age_ok(d, age)]
            picked = pick_stickhandling_mixed(drills, age, rnd, block_minutes=minutes, focus_rule=focus_rule)
            lines.append(f"\nSTICKHANDLING (~{minutes} min)")
            if not picked:
                lines.append("- [No matching drills found]")
            else:
                lines.extend(build_stickhandling_circuit(picked, minutes * 60))
            continue

        # Generic category
        if category == "strength" and (not strength_full_gym):
            drills = [d for d in data[category] if is_active(d) and age_ok(d, age) and equipment_ok(d, "none")]
        else:
            drills = [d for d in data[category] if is_active(d) and age_ok(d, age)]

        minutes = max(1, seconds // 60)
        count = 2 if minutes <= 8 else (3 if minutes <= 15 else 4)
        chosen = pick_n(drills, n=min(count, len(drills)) if drills else count, rnd=rnd, focus_rule=focus_rule)

        lines.append(f"\n{category.upper()} (~{minutes} min)")
        if not chosen:
            lines.append("- [No matching drills found]")
        else:
            if category in ("movement",):
                per = max(30, min(90, seconds // max(1, len(chosen) * 3)))
                rounds = clamp(seconds // max(1, (per * len(chosen))), 2, 4)
                lines.append(f"Time plan: {len(chosen)} drills | ~{per}s each | {rounds} rounds (~{format_seconds_short(per*len(chosen)*rounds)})")
            for d in chosen:
                lines.append(format_drill(d))

    return _finalize_and_return("\n".join(lines))


# ------------------------------
# CLI main
# ------------------------------
def main():
    print("\nBENDER v8.1 (Fixed) - Single Workout Generator\n")

    age = to_int(input("Player age (e.g., 15): ").strip(), 15)

    preset = norm(
        input("Preset (Enter to skip). Options: " + ", ".join(sorted(PRESETS.keys())) + "\n> ")
    ).strip() or None

    config = {
        "session_mode": None,
        "session_len_min": None,
        "shooting_shots": None,
        "stickhandling_min": None,
        "shooting_min": None,
        "strength_day_type": None,
        "strength_full_gym": None,
        "include_post_lift_conditioning": None,
        "strength_emphasis": "strength",
    }
    config = apply_preset(config, preset)

    session_mode = config.get("session_mode")
    session_len_min = config.get("session_len_min")
    shooting_shots = config.get("shooting_shots")
    stickhandling_min = config.get("stickhandling_min")
    shooting_min = config.get("shooting_min")
    strength_day_type = config.get("strength_day_type")
    include_post_lift_conditioning = config.get("include_post_lift_conditioning")
    strength_emphasis = config.get("strength_emphasis") or "strength"
    strength_full_gym = config.get("strength_full_gym")

    focus: Optional[str] = None  # keep token-based focus for UI/API use later

    if session_mode is None:
        session_mode = norm(
            input("\nSession mode (strength/movement/conditioning/stickhandling/shooting/skills_mixed/skills_only/recovery/mobility): ")
        ).lower()
    else:
        print(f"Using preset session_mode: {session_mode}")

    valid_modes = ("skills_only", "skills_mixed", "shooting", "stickhandling", "strength", "conditioning", "mobility", "movement", "recovery")
    if session_mode not in valid_modes:
        print("Unknown session mode. Defaulting to skills_mixed.")
        session_mode = "skills_mixed"

    if session_len_min is None:
        session_len_min = to_int(input("Session length (minutes, e.g., 30): ").strip(), 30)
    else:
        print(f"Using preset session length: {session_len_min} min")
    session_len_min = max(10, min(int(session_len_min), 90))

    # Random seed every run
    seed = random.randint(1, 2_000_000_000)

    post_lift_conditioning_type: Optional[str] = None
    skate_within_24h = False

    if session_mode == "strength":
        if strength_full_gym is None:
            fg = norm(input("Full gym available? (y/n): ")).lower()
            strength_full_gym = fg in ("y", "yes", "true", "1")
        else:
            print(f"Using preset strength_full_gym: {strength_full_gym}")

        if strength_day_type is None:
            strength_day_type = norm(input("Strength day type (leg/upper/full): ")).lower()
        if strength_day_type not in ("leg", "upper", "full"):
            strength_day_type = "leg"

        sw = norm(input("Skate within 24h? (y/n): ")).lower()
        skate_within_24h = sw in ("y", "yes", "true", "1")

        if include_post_lift_conditioning is None:
            plc = norm(input("Include optional post-lift conditioning? (y/n/Enter=auto): ")).lower()
            if plc in ("y", "yes"):
                include_post_lift_conditioning = True
            elif plc in ("n", "no"):
                include_post_lift_conditioning = False
            else:
                include_post_lift_conditioning = None

        strength_emphasis_in = norm(input("Strength emphasis (power/strength/hypertrophy/recovery): ")).lower()
        if strength_emphasis_in in ("power", "strength", "hypertrophy", "recovery"):
            strength_emphasis = strength_emphasis_in

        if include_post_lift_conditioning:
            if strength_full_gym:
                plc_type = norm(input("Post-lift conditioning type (bike/treadmill/surprise): ")).lower()
                if plc_type in ("bike", "treadmill", "surprise"):
                    post_lift_conditioning_type = plc_type
            else:
                post_lift_conditioning_type = None  # no-gym constraints enforced automatically

    if session_mode == "skills_only":
        if shooting_shots is None:
            ss = norm(input("Target shooting shots (Enter=auto): ")).strip()
            shooting_shots = to_int(ss, 0) if ss else None

    data = load_all_data()
    plan = generate_session(
        data=data,
        age=age,
        seed=seed,
        focus=focus,
        session_mode=session_mode,
        session_len_min=session_len_min,
        strength_full_gym=bool(strength_full_gym) if isinstance(strength_full_gym, bool) else bool(strength_full_gym),
        strength_emphasis=strength_emphasis,
        shooting_shots=shooting_shots,
        stickhandling_min=stickhandling_min,
        shooting_min=shooting_min,
        strength_day_type=strength_day_type,
        include_post_lift_conditioning=include_post_lift_conditioning,
        post_lift_conditioning_type=post_lift_conditioning_type,
        skate_within_24h=skate_within_24h,
    )
    print(plan)


if __name__ == "__main__":
    main()
