# ui_streamlit.py
import hashlib
import json
import os
import random
import re
import secrets
from contextlib import nullcontext
from datetime import date, datetime, timedelta
from pathlib import Path

import streamlit as st
import urllib.parse

# Profile storage (data/profiles/*.json)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_DIR = Path(BASE_DIR) / "data" / "profiles"
CUSTOM_PLAN_REQUESTS_PATH = Path(BASE_DIR) / "data" / "custom_plan_requests.json"

PBKDF2_ITERATIONS = 100_000


def load_custom_plan_requests() -> list[dict]:
    """Load custom plan requests (list of questionnaire responses)."""
    if not CUSTOM_PLAN_REQUESTS_PATH.exists():
        return []
    try:
        with open(CUSTOM_PLAN_REQUESTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _render_equipment_dropdowns(equipment_by_mode: dict, current_equip: set, key_prefix: str) -> None:
    """Render equipment per mode as expanders (dropdowns) with Select All + individual checkboxes."""
    _equip_tooltips = {"Line/Tape": "Floor line (tape or painted) for agility.", "Reaction ball": "Small rebound ball for reaction drills."}
    for mode_name, opts in equipment_by_mode.items():
        opts_real = [o for o in opts if o != "None"]
        if not opts_real:
            continue
        with st.expander(mode_name, expanded=False):
            sel_all_key = f"{key_prefix}_{mode_name}_Select All"
            select_all_default = all(opt in current_equip for opt in opts_real)
            sel_all = st.checkbox("Select All", value=select_all_default, key=sel_all_key)
            # When Select All is checked, sync all individual checkboxes to checked
            if sel_all:
                for opt in opts_real:
                    st.session_state[f"{key_prefix}_{mode_name}_{opt}"] = True
            for opt in opts_real:
                if opt in _equip_tooltips and not sel_all:
                    st.caption(_equip_tooltips[opt])
                opt_val = st.session_state.get(f"{key_prefix}_{mode_name}_{opt}", opt in current_equip)
                st.checkbox(opt, value=opt_val, key=f"{key_prefix}_{mode_name}_{opt}")
            # If Select All is on but user unchecked one, sync Select All off
            if sel_all and not all(st.session_state.get(f"{key_prefix}_{mode_name}_{opt}", False) for opt in opts_real):
                st.session_state[sel_all_key] = False
                st.rerun()


def _collect_equipment_from_session(equipment_by_mode: dict, key_prefix: str) -> list:
    """Collect selected equipment from session state based on key_prefix."""
    out = []
    for mode_name, opts in equipment_by_mode.items():
        opts_real = [o for o in opts if o != "None"]
        if not opts_real:
            continue
        if st.session_state.get(f"{key_prefix}_{mode_name}_Select All", False):
            out.extend(opts_real)
        else:
            for opt in opts_real:
                if st.session_state.get(f"{key_prefix}_{mode_name}_{opt}", False):
                    out.append(opt)
    return out


def save_custom_plan_request(request: dict) -> None:
    """Append a new custom plan request."""
    requests = load_custom_plan_requests()
    request["id"] = str(len(requests) + 1)
    request["created_at"] = datetime.now().isoformat()
    requests.append(request)
    CUSTOM_PLAN_REQUESTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CUSTOM_PLAN_REQUESTS_PATH, "w", encoding="utf-8") as f:
        json.dump(requests, f, indent=2)


def mark_custom_plan_request_complete(req_id: str) -> None:
    """Mark a custom plan request as complete. Updates the request in place."""
    requests = load_custom_plan_requests()
    for req in requests:
        if str(req.get("id", "")) == str(req_id):
            req["completed"] = True
            break
    CUSTOM_PLAN_REQUESTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CUSTOM_PLAN_REQUESTS_PATH, "w", encoding="utf-8") as f:
        json.dump(requests, f, indent=2)


def _hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    """Return (salt_hex, hash_hex). If salt is None, generate a new one."""
    if salt is None:
        salt = secrets.token_bytes(16)
    else:
        salt = salt if isinstance(salt, bytes) else bytes.fromhex(salt)
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return salt.hex(), key.hex()


def _verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    """Return True if password matches stored salt+hash."""
    try:
        _, derived_hex = _hash_password(password, bytes.fromhex(salt_hex))
        return secrets.compare_digest(derived_hex, hash_hex)
    except Exception:
        return False


def _user_id_taken(user_id: str) -> bool:
    """True if a profile already exists with this user_id."""
    if not user_id:
        return False
    return _profile_path(user_id).exists()


def _sanitize_user_id(name: str) -> str:
    s = re.sub(r"[^\w\-]", "_", (name or "").strip().lower())
    return s[:80] if s else ""


def _profile_path(user_id: str) -> Path:
    return PROFILE_DIR / f"{user_id}.json"


def load_profile(user_id: str) -> dict | None:
    path = _profile_path(user_id)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_profile(profile: dict) -> None:
    user_id = (profile.get("user_id") or "").strip()
    if not user_id:
        return
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    path = _profile_path(user_id)
    profile["updated_at"] = datetime.now().isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)


def list_profiles() -> list[dict]:
    if not PROFILE_DIR.exists():
        return []
    out = []
    for p in PROFILE_DIR.glob("*.json"):
        try:
            with open(p, encoding="utf-8") as f:
                out.append(json.load(f))
        except Exception:
            continue
    return sorted(out, key=lambda x: (x.get("display_name") or x.get("user_id") or "").lower())


def _apply_custom_request_to_plan_builder(req: dict) -> None:
    """Pre-fill Admin Plan Builder form with data from a custom plan request."""
    weeks = int(req.get("weeks") or 4)
    days_per_week = int(req.get("days_per_week") or 4)
    session_len_str = str(req.get("session_length") or "45")
    try:
        session_min = int(re.search(r"\d+", session_len_str).group(0))
    except (AttributeError, ValueError):
        session_min = 45
    session_min = max(10, min(90, session_min))
    plan_name = f"Custom: {req.get('display_name', 'Unknown')} — {str(req.get('primary_goal', ''))[:40]}"
    st.session_state.admin_weeks = weeks
    st.session_state.admin_plan_name = plan_name
    all_profiles = list_profiles()
    builder_options = [(None, "Default (admin / no equipment filter)")] + [(p, (p.get("display_name") or p.get("user_id") or "Unknown")) for p in all_profiles]
    req_user = (req.get("user_id") or "").strip()
    target_idx = 0
    for idx, (prof, _) in enumerate(builder_options):
        if prof and (prof.get("user_id") or "").strip() == req_user:
            target_idx = idx
            break
    st.session_state.admin_plan_target = target_idx
    try:
        from admin_plan_builder import PLAN_MODES, MODE_SESSION_LEN_DEFAULTS
    except ImportError:
        return
    days_to_select = list(range(min(days_per_week, 7)))
    for mode_key in PLAN_MODES:
        default_len = MODE_SESSION_LEN_DEFAULTS.get(mode_key, 30)
        use_len = session_min if mode_key == "performance" else default_len
        st.session_state[f"admin_mode_len_{mode_key}"] = use_len
        for wd in range(7):
            st.session_state[f"admin_mode_day_{mode_key}_{wd}"] = wd in days_to_select
    st.session_state.admin_custom_request_integrated = True


POSITION_OPTIONS = ["Forward", "Defense", "Goalie"]
CURRENT_LEVEL_OPTIONS = ["Youth", "HS", "AA", "AAA", "Junior", "College", "Beer League"]


def _equipment_setup_done(profile: dict) -> bool:
    """True if profile has completed required equipment setup (can generate workouts)."""
    return bool(profile.get("equipment_setup_done"))


def _serialize_plan_for_storage(plan: list, plan_name: str = "") -> dict:
    """Convert plan to dict with dates as ISO strings for JSON storage. Returns {plan: [...], plan_name: str}."""
    out = []
    weeks = plan if isinstance(plan, list) else plan.get("plan", plan)
    for w in weeks:
        week_copy = dict(w)
        week_copy["days"] = []
        for d in w.get("days", []):
            day_copy = dict(d)
            dt = d.get("date")
            if hasattr(dt, "isoformat"):
                day_copy["date"] = dt.isoformat()
            elif isinstance(dt, str):
                day_copy["date"] = dt
            week_copy["days"].append(day_copy)
        if "week_start" in week_copy and hasattr(week_copy["week_start"], "isoformat"):
            week_copy["week_start"] = week_copy["week_start"].isoformat()
        out.append(week_copy)
    name = plan_name or (plan.get("plan_name", "") if isinstance(plan, dict) else "")
    return {"plan": out, "plan_name": name or ""}


def _deserialize_plan_for_display(plan: list | dict) -> tuple[list, str]:
    """Convert date strings back to date objects. Returns (plan_list, plan_name). Handles legacy list or {plan, plan_name}."""
    if isinstance(plan, dict):
        weeks = plan.get("plan", [])
        plan_name = plan.get("plan_name", "")
    else:
        weeks = plan or []
        plan_name = ""
    out = []
    for w in weeks:
        week_copy = dict(w)
        week_copy["days"] = []
        for d in w.get("days", []):
            day_copy = dict(d)
            dt = d.get("date")
            if isinstance(dt, str):
                try:
                    day_copy["date"] = date.fromisoformat(dt)
                except (ValueError, TypeError):
                    day_copy["date"] = dt
            week_copy["days"].append(day_copy)
        if isinstance(week_copy.get("week_start"), str):
            try:
                week_copy["week_start"] = date.fromisoformat(week_copy["week_start"])
            except (ValueError, TypeError):
                pass
        out.append(week_copy)
    return (out, plan_name or "")


def _render_plan_view(plan: list | dict, completed: dict, profile: dict, on_complete: callable) -> None:
    """Render Bible App–style plan view. completed = {day_idx: [mode_key, ...]}. on_complete(day_idx, mode_key, params_dict|None)."""
    plan, _ = _deserialize_plan_for_display(plan)
    total_days = sum(len(w["days"]) for w in plan)
    flat_days: list[tuple[int, dict]] = []
    for w in plan:
        for d in w["days"]:
            flat_days.append((w["week"], d))
    if total_days == 0:
        st.caption("No days in plan.")
        return

    today_date = date.today()
    total_workouts = sum(len(d.get("focus_items", d.get("focus", []))) for _, d in flat_days)
    workouts_done = 0
    days_complete = 0
    days_missed = 0
    for i, (_, d) in enumerate(flat_days):
        fi = d.get("focus_items", [])
        if not fi:
            fi = [{"mode_key": f} for f in d.get("focus", [])]
        comp = completed.get(i) or completed.get(str(i)) or []
        comp_set = set(comp) if isinstance(comp, list) else set(comp)
        done_count = sum(1 for x in fi if x.get("mode_key") in comp_set)
        workouts_done += done_count
        if fi and all(x.get("mode_key") in comp_set for x in fi):
            days_complete += 1
        day_date = d.get("date")
        past = day_date < today_date if hasattr(day_date, "__lt__") else False
        if past and fi and not all(x.get("mode_key") in comp_set for x in fi):
            days_missed += 1
    plan_start = flat_days[0][1].get("date")
    if hasattr(plan_start, "strftime"):
        days_elapsed = (today_date - plan_start).days
        current_day_index = min(max(0, days_elapsed), total_days - 1)
    else:
        current_day_index = 0

    if "plan_selected_day" not in st.session_state:
        st.session_state.plan_selected_day = min(current_day_index, total_days - 1)
    sel_idx = min(st.session_state.plan_selected_day, total_days - 1)
    st.session_state.plan_selected_day = sel_idx

    # Workout view (separate page): Back at top, Complete at end
    if "plan_workout_view" in st.session_state and st.session_state.plan_workout_view is not None:
        wv_day, wv_mode = st.session_state.plan_workout_view
        if wv_day < len(flat_days):
            _, wv_day_data = flat_days[wv_day]
            focus_items = wv_day_data.get("focus_items", [])
            fi = next((x for x in focus_items if x["mode_key"] == wv_mode), None)
            if fi:
                _wv_title = ({"leg": "Heavy legs", "upper": "Upper / Core stability", "full": "Explosive legs"}.get((fi.get("params") or {}).get("strength_day_type", "full") or "full", fi["label"]) if fi.get("mode_key") == "performance" else fi["label"])
                st.markdown(f"### {_wv_title} — Day {wv_day + 1}")
                if st.button("← Back", key="plan_workout_back"):
                    st.session_state.plan_workout_view = None
                    st.rerun()
                _workout_text = fi.get("workout") or "(No workout)"
                if _workout_text != "(No workout)":
                    render_workout_readable(_workout_text)
                else:
                    st.caption(_workout_text)
                st.divider()
                if st.button("✓ Complete", key="plan_workout_complete"):
                    st.session_state.plan_workout_view = None
                    _comp = completed.get(wv_day) or completed.get(str(wv_day)) or []
                    _comp_set = set(_comp) if isinstance(_comp, list) else set(_comp)
                    _comp_set.add(wv_mode)
                    _focus = wv_day_data.get("focus_items", [])
                    all_done = len(_focus) > 0 and all(x["mode_key"] in _comp_set for x in _focus)
                    if all_done and wv_day < total_days - 1:
                        st.session_state.plan_selected_day = wv_day + 1
                    _params = fi.get("params") or {}
                    _meta = {"mode": fi.get("mode_key") or _params.get("mode"), "minutes": _params.get("session_len_min", 25)}
                    if _params.get("shooting_min") is not None:
                        _meta["shooting_min"] = _params["shooting_min"]
                    if _params.get("stickhandling_min") is not None:
                        _meta["stickhandling_min"] = _params["stickhandling_min"]
                    on_complete(wv_day, wv_mode, _meta)
                return
        st.session_state.plan_workout_view = None

    # Plan progress summary
    st.caption(f"**Progress:** {days_complete} day{'s' if days_complete != 1 else ''} complete, {days_missed} missed • {workouts_done} of {total_workouts} workouts done")
    # Day squares (clean dark card design): single row with horizontal scroll bar
    st.markdown('<div id="plan-day-grid" aria-hidden="true"></div>', unsafe_allow_html=True)
    st.markdown(f"**Day {sel_idx + 1} of {total_days}**")
    row_cols = st.columns(total_days)
    for i in range(total_days):
        with row_cols[i]:
            day_data = flat_days[i][1]
            day_date = day_data.get("date")
            date_str = day_date.strftime("%b %d") if hasattr(day_date, "strftime") else str(day_date)[:6]
            _completed = completed.get(i) or completed.get(str(i)) or []
            _comp_set = set(_completed) if isinstance(_completed, list) else set(_completed)
            focus_items_i = day_data.get("focus_items", [])
            day_complete = len(focus_items_i) > 0 and all(x["mode_key"] in _comp_set for x in focus_items_i)
            past = day_date < today_date if hasattr(day_date, "__lt__") else False
            missed = past and not day_complete
            if day_complete:
                st.markdown('<div class="plan-day-complete" aria-hidden="true"></div>', unsafe_allow_html=True)
            elif missed:
                st.markdown('<div class="plan-day-missed-marker" aria-hidden="true"></div>', unsafe_allow_html=True)
            # Number (button) + date below; selected date in white pill; "Missed day" below date when missed
            is_sel = i == sel_idx
            btn_type = "primary" if is_sel else "secondary"
            if st.button(str(i + 1), key=f"plan_day_{i}", type=btn_type):
                st.session_state.plan_selected_day = i
                st.rerun()
            _pill_cls = " plan-day-date-pill" if is_sel else ""
            if missed:
                st.markdown(
                    f'<div class="plan-day-date-block"><p class="plan-day-date{_pill_cls}">{date_str}</p>'
                    '<p class="plan-day-missed">Missed day</p></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f'<p class="plan-day-date{_pill_cls}">{date_str}</p>', unsafe_allow_html=True)
    st.divider()

    _, day_data = flat_days[sel_idx]
    dt_display = day_data["date"].strftime("%A, %b %d") if hasattr(day_data["date"], "strftime") else str(day_data["date"])
    st.markdown(f"### Day {sel_idx + 1}: {dt_display}")
    _completed_for_day = completed.get(sel_idx) or completed.get(str(sel_idx)) or []
    _completed_set = set(_completed_for_day) if isinstance(_completed_for_day, list) else _completed_for_day
    past_sel = day_data.get("date") < today_date if hasattr(day_data.get("date"), "__lt__") else False
    missed_sel = past_sel and not (len(day_data.get("focus_items", [])) > 0 and all(
        x["mode_key"] in _completed_set for x in day_data.get("focus_items", [])
    ))
    if missed_sel:
        st.caption("Missed day")
    day_done_sel = len(day_data.get("focus_items", [])) > 0 and all(x["mode_key"] in _completed_set for x in day_data.get("focus_items", []))
    if day_done_sel:
        st.caption("✓ Day complete")

    # Mode buttons (same as admin): click to open workout view
    st.markdown('<div id="plan-modes" aria-hidden="true"></div>', unsafe_allow_html=True)
    focus_items = day_data.get("focus_items", [])
    if focus_items:
        for fi in focus_items:
            done = fi["mode_key"] in _completed_set
            label = f"{'✓ ' if done else ''}{fi['label']}"
            if st.button(label, key=f"plan_open_{sel_idx}_{fi['mode_key']}", type="primary" if done else "secondary"):
                st.session_state.plan_workout_view = (sel_idx, fi["mode_key"])
                st.rerun()
    else:
        for f in day_data.get("focus", []):
            st.markdown(f"- {f}")


# Optional API mode (off by default)
USE_API = os.getenv("BENDER_USE_API", "0").strip().lower() in ("1", "true", "yes", "y")
API_BASE = os.getenv("BENDER_API_BASE", "http://127.0.0.1:8000").strip()

if USE_API:
    import requests
# -----------------------------
# Pretty workout renderer (UI only)
# -----------------------------
def render_workout_pretty(text: str) -> None:
    """
    Light formatting for engine output:
    - SECTION HEADERS: uppercase lines rendered as headers
    - Drill lines kept monospace for easy copy
    """
    if not text:
        return

    lines = text.splitlines()

    for line in lines:
        s = line.strip()

        # Section headers (e.g. WARMUP, BLOCK A, SHOOTING)
        if s and s == s.upper() and not s.startswith("-") and len(s) <= 60:
            st.markdown(f"### {s}")
            continue

        # Blank spacing
        if not s:
            st.write("")
            continue

        # Everything else stays copy-friendly
        st.markdown(f"`{line}`")

# -----------------------------
# Import engine (direct mode)
# -----------------------------
import sys
import importlib

ENGINE_IMPORT_ERROR = None
ENGINE = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cloud-safe: ensure the folder containing ui_streamlit.py is importable
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    ENGINE = importlib.import_module("bender_generate_v8_1")
except Exception as e:
    ENGINE_IMPORT_ERROR = e

if ENGINE is None:
    st.error("Bender engine failed to import.")
    st.write("BASE_DIR:", BASE_DIR)
    st.write("Files in BASE_DIR:", os.listdir(BASE_DIR))
    st.write("sys.path (first 10):", sys.path[:10])
    st.exception(ENGINE_IMPORT_ERROR)   # THIS will show the real error message
    st.stop()


def get_mode_options():
    # Use Streamlit top-level categories (Puck Mastery shows sub-choice; engine gets shooting/stickhandling/skills_only)
    return ["puck_mastery", "performance", "energy_systems", "skating_mechanics", "mobility"]


RAW_MODES = get_mode_options()

# Top-level display labels
MODE_LABELS = {
    "puck_mastery": "Puck Mastery",
    "performance": "Performance",
    "energy_systems": "Conditioning",
    "skating_mechanics": "Skating Mechanics",
    "mobility": "Mobility & Recovery",
}

# Puck Mastery sub-options (map to actual session_mode sent to engine)
SKILLS_SUB_LABELS = ["Shooting", "Stickhandling", "Both"]
SKILLS_SUB_TO_MODE = {"Shooting": "shooting", "Stickhandling": "stickhandling", "Both": "skills_only"}

DISPLAY_MODES = [MODE_LABELS.get(m, m) for m in RAW_MODES]
LABEL_TO_MODE = {MODE_LABELS.get(m, m): m for m in RAW_MODES}

# -----------------------------
# Strength emphasis labels
# -----------------------------
STRENGTH_EMPHASIS_LABELS = {
    "power": "power (explosive speed)",
    "strength": "strength (game strength)",
    "hypertrophy": "hypertrophy (strength capacity)",
    "recovery": "recovery (less stress)",
}
EMPHASIS_KEYS = list(STRENGTH_EMPHASIS_LABELS.keys())
EMPHASIS_DISPLAY = [STRENGTH_EMPHASIS_LABELS[k] for k in EMPHASIS_KEYS]
EMPHASIS_LABEL_TO_KEY = {STRENGTH_EMPHASIS_LABELS[k]: k for k in EMPHASIS_KEYS}

# -----------------------------
# Feedback (Google Form - auto-filled)
# -----------------------------
FORM_BASE = "https://docs.google.com/forms/d/e/1FAIpQLSd2bOUc6bJkZHzTglQ6KgOv8QkzJro-iFR9uLE_rpHw5G4I8g/viewform"

ENTRY_ATHLETE = "entry.1733182497"
ENTRY_MODE = "entry.379036262"
ENTRY_LOCATION = "entry.2120495587"
ENTRY_EMPHASIS = "entry.1294938744"
ENTRY_RATING = "entry.591938645"
ENTRY_NOTES = "entry.1295575438"


def build_prefilled_feedback_url(
    *,
    athlete: str,
    mode_label: str,
    location_label: str,
    emphasis_key: str,
    rating: int = 4,
    notes: str = ""
) -> str:
    params = {
        "usp": "pp_url",
        ENTRY_ATHLETE: athlete or "",
        ENTRY_MODE: mode_label or "",
        ENTRY_LOCATION: location_label or "",
        ENTRY_EMPHASIS: emphasis_key or "",
        ENTRY_RATING: str(int(rating)),
        ENTRY_NOTES: notes or "",
    }
    return FORM_BASE + "?" + urllib.parse.urlencode(params)


def clear_last_output():
    st.session_state.last_output_text = None
    st.session_state.last_session_id = None
    if "last_output_metadata" in st.session_state:
        st.session_state.last_output_metadata = None


def _parse_workout_header_for_metadata(text: str) -> dict:
    """Parse BENDER SINGLE WORKOUT header for mode and len. Returns metadata dict or {}."""
    import re as _re
    if not text or not isinstance(text, str):
        return {}
    m = _re.search(r"mode=(\w+)\s*\|\s*len=(\d+)\s*min", text, _re.IGNORECASE)
    if m:
        return {"mode": m.group(1).strip(), "minutes": int(m.group(2) or 0)}
    return {}


def _compute_volume_from_metadata(metadata: dict) -> dict:
    """Compute volume deltas from workout metadata for Your Work stats.
    Returns {stickhandling_hours, shots, gym_hours, skating_hours, conditioning_hours, mobility_hours}."""
    mode = (metadata.get("mode") or "").lower()
    minutes = max(0, int(metadata.get("minutes") or 0))
    hours = minutes / 60.0
    loc = (metadata.get("location") or "").lower()
    conditioning = bool(metadata.get("conditioning"))
    out = {
        "stickhandling_hours": 0.0,
        "shots": 0,
        "gym_hours": 0.0,
        "skating_hours": 0.0,
        "conditioning_hours": 0.0,
        "mobility_hours": 0.0,
    }
    if mode == "performance":
        out["gym_hours"] = hours
        if conditioning:
            # ~6–8 min typical post-lift conditioning
            out["conditioning_hours"] += 0.1  # ~6 min
    elif mode == "stickhandling":
        out["stickhandling_hours"] = hours
    elif mode == "shooting":
        out["shots"] = max(0, int(minutes * 8))  # ~8 shots/min
    elif mode in ("skills_only", "puck_mastery"):
        shoot_min = int(metadata.get("shooting_min") or minutes // 2)
        stick_min = int(metadata.get("stickhandling_min") or minutes - shoot_min)
        out["stickhandling_hours"] = stick_min / 60.0
        out["shots"] = max(0, int(shoot_min * 8))
    elif mode == "skating_mechanics":
        out["skating_hours"] = hours
    elif mode == "energy_systems":
        out["conditioning_hours"] = hours
    elif mode == "mobility":
        out["mobility_hours"] = hours
    return out


def _add_completion_to_profile(profile: dict, metadata: dict) -> dict:
    """Add workout completion volumes to profile's private_victory_stats (Your Work)."""
    prof = dict(profile)
    stats = dict(prof.get("private_victory_stats") or {})
    for k, default in [
        ("stickhandling_hours", 0.0),
        ("gym_hours", 0.0),
        ("skating_hours", 0.0),
        ("conditioning_hours", 0.0),
        ("mobility_hours", 0.0),
        ("shots", 0),
        ("completions_count", 0),
    ]:
        if k not in stats:
            stats[k] = default
    deltas = _compute_volume_from_metadata(metadata)
    stats["stickhandling_hours"] = stats["stickhandling_hours"] + deltas["stickhandling_hours"]
    stats["shots"] = stats["shots"] + deltas["shots"]
    stats["gym_hours"] = stats["gym_hours"] + deltas["gym_hours"]
    stats["skating_hours"] = stats["skating_hours"] + deltas["skating_hours"]
    stats["conditioning_hours"] = stats["conditioning_hours"] + deltas["conditioning_hours"]
    stats["mobility_hours"] = stats["mobility_hours"] + deltas["mobility_hours"]
    stats["completions_count"] = stats["completions_count"] + 1
    prof["private_victory_stats"] = stats
    return prof


def _render_your_work_stats():
    """Show all mode hours and shots, including 0 values."""
    prof = st.session_state.get("current_profile") or {}
    stats = prof.get("private_victory_stats") or {}
    gym_h = float(stats.get("gym_hours", 0) or 0)
    skating_h = float(stats.get("skating_hours", 0) or 0)
    cond_h = float(stats.get("conditioning_hours", 0) or 0)
    stick_h = float(stats.get("stickhandling_hours", 0) or 0)
    mob_h = float(stats.get("mobility_hours", 0) or 0)
    shots = int(stats.get("shots", 0) or 0)
    total_hours = gym_h + skating_h + cond_h + stick_h + mob_h

    st.markdown(
        '<div class="your-work-stats-card">'
        '<div class="your-work-section"><span class="your-work-label">Total Shots</span><span class="your-work-value">{:,}</span></div>'
        '<div class="your-work-divider"></div>'
        '<div class="your-work-row"><span class="your-work-cat">Gym</span><span class="your-work-num">{:.1f} h</span></div>'
        '<div class="your-work-row"><span class="your-work-cat">Skating mechanics</span><span class="your-work-num">{:.1f} h</span></div>'
        '<div class="your-work-row"><span class="your-work-cat">Conditioning</span><span class="your-work-num">{:.1f} h</span></div>'
        '<div class="your-work-row"><span class="your-work-cat">Stickhandling</span><span class="your-work-num">{:.1f} h</span></div>'
        '<div class="your-work-row"><span class="your-work-cat">Mobility / recovery</span><span class="your-work-num">{:.1f} h</span></div>'
        '<div class="your-work-divider"></div>'
        '<div class="your-work-section"><span class="your-work-label">Total Hours</span><span class="your-work-value">{:.1f} h</span></div>'
        '</div>'.format(shots, gym_h, skating_h, cond_h, stick_h, mob_h, total_hours),
        unsafe_allow_html=True,
    )


# -----------------------------
# Pretty workout renderer (UI only)
# -----------------------------
_SECTION_RE = re.compile(
    r"^(warmup|speed|power|high fatigue|block a|block b|strength circuits|circuit a|circuit b|shooting|stickhandling|conditioning|energy systems|speed agility|skating mechanics|mobility|post-lift|youth|"
    r"primary bilateral|primary press|primary pull|heavy unilateral|posterior chain|frontal-plane|^iso\b|scap\s|core anti-rotation|controlled rotation|carry finisher|elastic cns|contrast block|strength)\b",
    re.IGNORECASE,
)

def _is_section_header(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if s.startswith("-"):
        return False
    # Most of your headers start with one of these keywords
    return bool(_SECTION_RE.match(s))

def _header_style(title: str) -> str:
    t = title.lower()
    if "warmup" in t:
        return "Warm-up"
    if "speed" in t or "power" in t:
        return "Speed / Power"
    if "high fatigue" in t:
        return "High Fatigue"
    if "block a" in t:
        return "Block A"
    if "block b" in t:
        return "Block B"
    if "strength circuits" in t or "circuit a" in t or "circuit b" in t:
        return "Strength Circuits"
    if "shooting" in t:
        return "Shooting"
    if "stickhandling" in t:
        return "Stickhandling"
    if "post-lift" in t:
        return "Post-Lift Conditioning"
    if "youth" in t:
        return "Youth (13 & under) guidance"
    if "conditioning" in t or "energy systems" in t:
        return "Conditioning"
    if "speed agility" in t:
        return "Speed & Agility"
    if "skating mechanics" in t:
        return "Skating Mechanics"
    if "mobility" in t:
        return "Mobility"
    if "primary bilateral" in t or "heavy unilateral" in t:
        return "Strength"
    if "posterior chain" in t or "hinge" in t:
        return "Posterior Chain"
    if "frontal" in t or "adductor" in t:
        return "Frontal / Adductor"
    if "iso" in t or "decel" in t or "braking" in t:
        return "Iso / Decel"
    if "scap" in t or "rotator" in t:
        return "Scap / RC Prep"
    if "primary press" in t or "primary pull" in t:
        return "Strength"
    if "core anti-rotation" in t or "anti-extension" in t:
        return "Core"
    if "controlled rotation" in t or "carry" in t:
        return "Core / Carry"
    if "elastic" in t or "contrast" in t:
        return "Power"
    return "Section"

def render_workout_readable(text: str) -> None:
    """
    Renders engine text into clean sections.
    Only the warm-up section uses a dropdown (expander); all other sections are always visible.
    Hides the BENDER SINGLE WORKOUT | mode=... header line from display.
    """
    if not text:
        return

    lines = text.splitlines()
    # Skip BENDER SINGLE WORKOUT | mode=... header and any leading blank lines
    while lines:
        s = lines[0].strip()
        if s.startswith("BENDER SINGLE WORKOUT") or (not s and len(lines) > 1):
            lines.pop(0)
        else:
            break
    current_title = None
    buffer: list[str] = []

    def is_warmup_header(title: str) -> bool:
        t = (title or "").strip().upper()
        return t.startswith("WARMUP")

    def flush_section(title: str, body_lines: list[str]) -> None:
        if not title and not body_lines:
            return
        label = (title.strip() or "Section")
        tag = _header_style(title) if title else ""
        # Only warm-up gets a dropdown; rest of workout is always visible. Bold "WARMUP", same size.
        if is_warmup_header(title):
            warmup_display = "**WARMUP**" + (label[6:] if label.upper().startswith("WARMUP") else "")
            expander_label = f"{warmup_display} — {tag}" if (tag and tag != "Section") else warmup_display
            with st.expander(expander_label):
                for ln in body_lines:
                    s = ln.strip()
                    if not s:
                        continue
                    if s.startswith("-"):
                        st.markdown(s)
                    else:
                        st.caption(s)
        else:
            with st.container(border=True):
                st.subheader(label)
                if tag and tag != "Section":
                    st.caption(tag)
                for ln in body_lines:
                    s = ln.strip()
                    if not s:
                        continue
                    if s.startswith("-"):
                        st.markdown(s)
                    else:
                        st.caption(s)

    for ln in lines:
        s = ln.strip()

        if _is_section_header(s):
            flush_section(current_title or "", buffer)
            current_title = s
            buffer = []
            continue

        buffer.append(ln)

    flush_section(current_title or "", buffer)


def _section_title_to_key(title: str, ex_idx_in_section: int = 0) -> str | None:
    """Map section header text to engine section key for drill filtering."""
    t = (title or "").strip().lower()
    if not t:
        return None
    if "speed" in t and "high fatigue" in t:
        return "high_fatigue" if ex_idx_in_section == 1 else "speed_power"
    if "high fatigue" in t:
        return "high_fatigue"
    if "core anti" in t or "anti-rotation" in t or "anti-extension" in t:
        return "core_anti_rotation"
    if "primary press" in t:
        return "primary_press"
    if "primary pull" in t:
        return "primary_pull"
    if "primary bilateral" in t:
        return "primary_bilateral"
    if "posterior chain" in t or "hinge" in t:
        return "posterior_chain"
    if "heavy unilateral" in t or "frontal" in t or "frontal-plane" in t or "adductor" in t:
        return "heavy_unilateral"
    if "iso" in t or "decel" in t or "braking" in t:
        return "iso_decel"
    if "controlled rotation" in t:
        return "controlled_rotation"
    if "carry" in t:
        return "carry_finisher"
    if "scap" in t or "shoulder health" in t:
        return "scap"
    if "resilience" in t or "circuit a" in t or "circuit b" in t or "block a" in t or "block b" in t:
        return "resilience"
    if "secondary" in t or "strength circuit" in t:
        return "secondary"
    if "speed" in t or "power" in t or "elastic" in t:
        return "speed_power"
    if "warmup" in t or "warm-up" in t:
        return "warmup"
    if "shooting" in t:
        return "shooting"
    if "stickhandling" in t or "puck" in t:
        return "stickhandling"
    if "conditioning" in t or "energy" in t:
        return "conditioning"
    if "skating" in t or "movement" in t:
        return "skating_mechanics"
    if "mobility" in t:
        return "mobility"
    return None


def _parse_workout_for_editing(text: str) -> list[dict]:
    """Parse workout text into editable items. Each item: {type, ...} where type is section|strength|timed|simple|raw. Exercises include section_key for dropdown filtering."""
    if not text:
        return []
    items = []
    lines = text.splitlines()
    i = 0
    current_section = ""
    ex_idx_in_section = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if _is_section_header(s):
            items.append({"type": "section", "title": s})
            current_section = s
            ex_idx_in_section = 0
            i += 1
            continue
        if s.startswith("- "):
            body = s[2:].strip()
            cues = ""
            steps = ""
            i += 1
            while i < len(lines) and (lines[i].strip().startswith("Cues:") or lines[i].strip().startswith("Steps:")):
                sub = lines[i].strip()
                if sub.lower().startswith("cues:"):
                    cues = sub[5:].strip()
                elif sub.lower().startswith("steps:"):
                    steps = sub[6:].strip()
                i += 1
            section_key = _section_title_to_key(current_section, ex_idx_in_section)
            ex_idx_in_section += 1
            if " | " in body:
                parts = [p.strip() for p in body.split("|")]
                name = parts[0]
                sets, reps, rest = None, None, None
                for p in parts[1:]:
                    if " x " in p.lower():
                        sr = p.split(" x ", 1)
                        try:
                            sets = int(sr[0].strip())
                        except (ValueError, TypeError):
                            sets = 3
                        reps = sr[1].strip() if len(sr) > 1 else "8"
                    elif p.lower().startswith("rest "):
                        r = p[5:].replace("s", "").strip()
                        try:
                            rest = int(r)
                        except (ValueError, TypeError):
                            rest = 60
                items.append({"type": "strength", "name": name, "sets": sets or 3, "reps": reps or "8", "rest": rest or 60, "cues": cues, "steps": steps, "section_key": section_key})
            elif " — " in body:
                nm, _, dur = body.partition(" — ")
                name = nm.strip()
                dur_s = dur.replace("s", "").strip()
                try:
                    duration = int(dur_s)
                except (ValueError, TypeError):
                    duration = 30
                items.append({"type": "timed", "name": name.strip(), "duration": duration, "cues": cues, "steps": steps, "section_key": section_key})
            else:
                items.append({"type": "simple", "name": body, "cues": cues, "steps": steps, "section_key": section_key})
            continue
        if s:
            items.append({"type": "raw", "line": ln})
        i += 1
    return items


def _rebuild_workout_from_edits(items: list[dict], form_vals: dict) -> str:
    """Rebuild workout text from parsed items and form values (from st.session_state)."""
    out = []
    ex_idx = 0
    for it in items:
        if it["type"] == "section":
            out.append(it["title"])
        elif it["type"] == "strength":
            v = form_vals.get(ex_idx, {})
            name = v.get("name", it["name"])
            sets = v.get("sets", it.get("sets", 3))
            reps = v.get("reps", it.get("reps", "8"))
            rest = v.get("rest", it.get("rest", 60))
            line = f"- {name} | {sets} x {reps} | Rest {rest}s"
            out.append(line)
            if it.get("cues"):
                out.append(f"  Cues: {it['cues']}")
            if it.get("steps"):
                out.append(f"  Steps: {it['steps']}")
            ex_idx += 1
        elif it["type"] == "timed":
            v = form_vals.get(ex_idx, {})
            name = v.get("name", it["name"])
            duration = v.get("duration", it.get("duration", 30))
            line = f"- {name} — {duration}s"
            out.append(line)
            if it.get("cues"):
                out.append(f"  Cues: {it['cues']}")
            if it.get("steps"):
                out.append(f"  Steps: {it['steps']}")
            ex_idx += 1
        elif it["type"] == "simple":
            v = form_vals.get(ex_idx, {})
            name = v.get("name", it["name"])
            out.append(f"- {name}")
            if it.get("cues"):
                out.append(f"  Cues: {it['cues']}")
            if it.get("steps"):
                out.append(f"  Steps: {it['steps']}")
            ex_idx += 1
        elif it["type"] == "raw":
            out.append(it["line"])
    return "\n".join(out)


def render_workout_editable(
    text: str,
    params: dict,
    data: dict,
    age: int,
    user_equipment: list | None,
    key_prefix: str,
) -> str | None:
    """
    Render editable workout: exercise dropdowns + sets/reps. Returns modified workout text on Save, else None.
    Uses section-specific drill pools (e.g. high fatigue, core anti-rotation) for dropdown alternatives.
    """
    if not text or text == "(No workout)":
        return None
    items = _parse_workout_for_editing(text)
    if not items:
        return None
    get_pool = getattr(ENGINE, "get_drills_for_section", None) or getattr(ENGINE, "get_drills_pool_for_plan_slot", lambda *a: [])
    form_vals = {}
    ex_idx = 0
    for it in items:
        if it["type"] == "section":
            st.subheader(it["title"][:60])
        elif it["type"] in ("strength", "timed", "simple"):
            pk = f"{key_prefix}_ex_{ex_idx}"
            current_name = it["name"]
            section_key = it.get("section_key")
            pool = []
            if ENGINE:
                get_section = getattr(ENGINE, "get_drills_for_section", None)
                get_slot = getattr(ENGINE, "get_drills_pool_for_plan_slot", lambda *a: [])
                if get_section:
                    pool = get_section(data, age, params, user_equipment, section_key)
                else:
                    pool = get_slot(data, age, params, user_equipment)
                if not pool and user_equipment:
                    pool = get_section(data, age, params, None, section_key) if get_section else get_slot(data, age, params, None)
            display_names = []
            for d in pool:
                nm = getattr(ENGINE, "_display_name", lambda x: x.get("name", ""))(d) if ENGINE else d.get("name", "(unnamed)")
                display_names.append(nm)
            if not display_names:
                display_names = ["(No alternatives)"]
            options = list(dict.fromkeys([current_name] + [n for n in display_names if n != current_name and n != "(No alternatives)"]))
            if not options:
                options = [current_name]
            default_idx = options.index(current_name) if current_name in options else 0
            sel = st.selectbox("Exercise", options=options, index=default_idx, key=f"{pk}_sel")
            form_vals[ex_idx] = {"name": sel}
            if it["type"] == "strength":
                c1, c2, c3 = st.columns(3)
                with c1:
                    sets = st.number_input("Sets", 1, 10, value=it.get("sets", 3), key=f"{pk}_sets")
                with c2:
                    reps = st.text_input("Reps", value=str(it.get("reps", "8")), key=f"{pk}_reps")
                with c3:
                    rest = st.number_input("Rest (s)", 0, 300, value=it.get("rest", 60), key=f"{pk}_rest")
                form_vals[ex_idx]["sets"] = sets
                form_vals[ex_idx]["reps"] = reps
                form_vals[ex_idx]["rest"] = rest
            elif it["type"] == "timed":
                _dur_val = it.get("duration", 30)
                _dur_val = max(5, min(900, int(_dur_val) if _dur_val is not None else 30))
                dur = st.number_input("Duration (s)", 5, 900, value=_dur_val, key=f"{pk}_dur")
                form_vals[ex_idx]["duration"] = dur
            ex_idx += 1
        elif it["type"] == "raw":
            st.caption(it["line"])
    if st.button("Save workout", key=f"{key_prefix}_save", type="primary"):
        return _rebuild_workout_from_edits(items, form_vals)
    return None


# -----------------------------
# No-gym strength: circuits-only renderer
# (Warmup first, then Circuit A/B only once)
# -----------------------------
def render_no_gym_strength_circuits_only(text: str) -> None:
    """No-gym strength: only warm-up has a dropdown; Circuit A/B and Mobility are always visible."""
    if not text:
        return

    lines = text.splitlines()

    def find_first_exact(targets: tuple[str, ...], start: int = 0) -> int:
        for i in range(start, len(lines)):
            if lines[i].strip() in targets:
                return i
        return -1

    def grab_section(start_idx: int, stop_headers: set[str]) -> list[str]:
        out: list[str] = []
        for j in range(start_idx + 1, len(lines)):
            s = lines[j].strip()
            if s in stop_headers:
                break
            if s:
                out.append(lines[j])
        return out

    def render_section(label: str, tag: str, body: list[str], as_dropdown: bool = False) -> None:
        if as_dropdown:
            with st.expander(f"{label} — {tag}"):
                for ln in body:
                    s = ln.strip()
                    if s.startswith("-"):
                        st.markdown(s)
                    else:
                        st.caption(s)
        else:
            with st.container(border=True):
                st.subheader(label)
                st.caption(tag)
                for ln in body:
                    s = ln.strip()
                    if s.startswith("-"):
                        st.markdown(s)
                    else:
                        st.caption(s)

    # ---- headers we care about ----
    CIRCUIT_HEADERS = {"CIRCUIT A", "CIRCUIT B"}
    MOBILITY_HEADERS = {"MOBILITY COOLDOWN CIRCUIT", "MOBILITY"}
    STOP_HEADERS = CIRCUIT_HEADERS | MOBILITY_HEADERS | {
        "POST-LIFT CONDITIONING",
        "POST-LIFT ENERGY SYSTEMS",
        "SHOOTING",
        "STICKHANDLING",
        "CONDITIONING",
        "ENERGY SYSTEMS",
    }

    # ---- Warmup: render if present BEFORE circuits ----
    warmup_start = -1
    for i, ln in enumerate(lines):
        if ln.strip().startswith("WARMUP"):
            warmup_start = i
            break

    circuits_start = find_first_exact(("CIRCUIT A", "CIRCUIT B", "STRENGTH CIRCUITS (Preset)", "STRENGTH CIRCUITS"))

    if warmup_start != -1:
        warmup_end = circuits_start if circuits_start != -1 and circuits_start > warmup_start else len(lines)
        warmup_body = [ln for ln in lines[warmup_start + 1 : warmup_end] if ln.strip()]
        render_section("**WARMUP**", "Strength Circuits", warmup_body, as_dropdown=True)

    # ---- Find first Circuit A, then first Circuit B after that ----
    a_start = find_first_exact(("CIRCUIT A",))
    if a_start == -1:
        st.text(text)
        return

    b_start = find_first_exact(("CIRCUIT B",), start=a_start + 1)

    a_body = grab_section(a_start, STOP_HEADERS)
    render_section("CIRCUIT A", "Strength Circuits", a_body, as_dropdown=False)

    if b_start != -1:
        b_body = grab_section(b_start, STOP_HEADERS)
        render_section("CIRCUIT B", "Strength Circuits", b_body, as_dropdown=False)

    mob_start = -1
    for i, ln in enumerate(lines):
        if ln.strip().startswith("MOBILITY"):
            mob_start = i
            break

    if mob_start != -1:
        mob_body = grab_section(mob_start, STOP_HEADERS)
        render_section(lines[mob_start].strip(), "Mobility", mob_body, as_dropdown=False)

CACHE_VERSION = "2026-02-11"  # bump this when data files change

@st.cache_resource
def _load_engine_data(cache_version: str = CACHE_VERSION):
    _ = cache_version

    if ENGINE is None:
        raise RuntimeError(f"Engine import failed: {ENGINE_IMPORT_ERROR}")

    return ENGINE.load_all_data(data_dir="data")

def _generate_via_engine(payload: dict) -> dict:
    """
    Calls bender_generate_v8_1.generate_session directly.
    Returns a dict shaped similarly to API response.
    """
    data = _load_engine_data()

    # Stable seed per click (randomized each time user hits Generate)
    seed = random.randint(1, 2_000_000_000)

    mode = payload["mode"]
    minutes = int(payload["minutes"])
    age = int(payload["age"])
    athlete_id = payload["athlete_id"]

    focus = payload.get("focus", None)

    # Strength-specific tokens
    location = payload.get("location", "no_gym")
    strength_full_gym = (payload.get("mode") == "performance" and location == "gym")

    strength_day_type = payload.get("strength_day_type", None)  # "leg"/"upper"
    strength_emphasis = payload.get("strength_emphasis", "strength")
    skate_within_24h = bool(payload.get("skate_within_24h", False))

    include_post_lift_conditioning = bool(payload.get("conditioning", False)) if payload.get("mode") == "performance" else None
    post_lift_conditioning_type = payload.get("conditioning_type", None)

    # Skills-only extras (optional later: expose shot volume)
    shooting_shots = payload.get("shooting_shots", None)
    stickhandling_min = payload.get("stickhandling_min", None)
    shooting_min = payload.get("shooting_min", None)

    profile = st.session_state.get("current_profile") or {}
    equipment = profile.get("equipment")
    user_equipment = ENGINE.expand_user_equipment(equipment) if equipment else None
    available_space = payload.get("available_space")
    conditioning_mode = payload.get("conditioning_mode")
    conditioning_effort = payload.get("conditioning_effort")
    out_text = ENGINE.generate_session(
        data=data,
        age=age,
        seed=seed,
        focus=focus,
        session_mode=mode,
        session_len_min=minutes,
        athlete_id=athlete_id,
        use_memory=True,
        memory_sessions=6,
        recent_penalty=0.25,
        strength_emphasis=strength_emphasis,
        shooting_shots=shooting_shots,
        stickhandling_min=stickhandling_min,
        shooting_min=shooting_min,
        strength_day_type=strength_day_type,
        strength_full_gym=strength_full_gym,
        include_post_lift_conditioning=include_post_lift_conditioning,
        post_lift_conditioning_type=post_lift_conditioning_type,
        skate_within_24h=skate_within_24h,
        user_equipment=user_equipment,
        available_space=available_space,
        conditioning_mode=conditioning_mode,
        conditioning_effort=conditioning_effort,
    )

    # Lightweight "session_id" for display/share later (not persisted)
    session_id = f"{athlete_id.strip().lower()}-{seed}"

    return {"session_id": session_id, "output_text": out_text}


def _generate_via_api(payload: dict) -> dict:
    r = requests.post(f"{API_BASE}/api/generate", json=payload, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(r.text)
    return r.json()


# -----------------------------
# UI
# -----------------------------
# Collapse sidebar after Save equipment (one-time flag)
_sidebar_state = "collapsed" if st.session_state.get("collapse_sidebar_after_save") else "expanded"
if st.session_state.get("collapse_sidebar_after_save"):
    st.session_state.collapse_sidebar_after_save = False
_page_icon = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "b_logo.png")
if not os.path.isfile(_page_icon):
    _page_icon = None
st.set_page_config(page_title="Bender", layout="wide", initial_sidebar_state=_sidebar_state, page_icon=_page_icon if _page_icon else "🏒")

# Custom CSS: single-column main; sidebar for equipment
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');

    .stApp { background: #000000 !important; }
    .main { max-width: 100% !important; width: 100% !important; }
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; padding-left: 1rem; padding-right: 1rem; max-width: min(1800px, 98vw) !important; width: 100% !important; min-width: 0 !important; background: transparent; overflow-x: visible !important; }
    h1 { font-family: 'DM Sans', sans-serif !important; font-weight: 700 !important; color: #ffffff !important; letter-spacing: -0.02em; }
    .bender-tagline { font-family: 'DM Sans', sans-serif; color: #ffffff; font-size: 1.15rem; margin-bottom: 1.25rem; letter-spacing: 0.05em; }
    .bender-brand-sub { font-family: 'DM Sans', sans-serif; color: #ffffff; font-size: 1.05rem; letter-spacing: 0.15em; opacity: 0.9; margin-top: 0.25rem; }
    label { font-family: 'DM Sans', sans-serif !important; color: #ffffff !important; }

    /* Sidebar: black to match Bender branding */
    [data-testid="stSidebar"] { background-color: #000000 !important; }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .stCheckbox > label,
    [data-testid="stSidebar"] [data-testid="stCheckbox"] label,
    [data-testid="stSidebar"] div[data-testid="stCheckbox"] * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] .stCaption {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    /* Equipment checkboxes: white text for black theme; checked box checkmark BLACK for clear visibility */
    .stCheckbox label, [data-testid="stCheckbox"] label {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
    }
    .stCheckbox input:checked, [data-testid="stCheckbox"] input:checked,
    input[type="checkbox"]:checked {
        accent-color: #000000 !important;
        background-color: #ffffff !important;
    }
    /* Checkmark color: black for clear visibility when checked */
    [data-testid="stCheckbox"]:has(input:checked) svg path,
    [data-testid="stCheckbox"]:has(input:checked) svg,
    .stCheckbox:has(input:checked) svg path,
    .stCheckbox:has(input:checked) svg,
    /* Sidebar Save equipment + Sign out: dark background, white text for visibility */
    [data-testid="stSidebar"] .stButton button {
        background: #333333 !important; color: #ffffff !important; border: 1px solid #666666 !important;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: #555555 !important; color: #ffffff !important; border-color: #888888 !important;
    }

    /* Account Settings + Equipment: subtle shading so dropdowns/expanders are visible */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background-color: #1a1a1a !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
        margin-bottom: 0.5rem !important;
        border: 1px solid #333333 !important;
    }

    /* Plan day selector: exactly 5 cards visible, scroll for more (match Faith That Endures design) */
    #plan-day-grid ~ [data-testid="stHorizontalBlock"],
    #plan-day-grid ~ * [data-testid="stHorizontalBlock"],
    #admin-plan-day-grid ~ [data-testid="stHorizontalBlock"],
    #admin-plan-day-grid ~ * [data-testid="stHorizontalBlock"],
    #admin-edit-day-grid ~ [data-testid="stHorizontalBlock"],
    #admin-edit-day-grid ~ * [data-testid="stHorizontalBlock"],
    [data-testid="stMarkdown"]:has(#plan-day-grid) ~ [data-testid="stHorizontalBlock"],
    div:has(#plan-day-grid) ~ div [data-testid="stHorizontalBlock"],
    div:has(#admin-plan-day-grid) ~ div [data-testid="stHorizontalBlock"],
    div:has(#admin-edit-day-grid) ~ div [data-testid="stHorizontalBlock"] {
        overflow-x: auto !important; overflow-y: hidden !important;
        width: 24rem !important; max-width: 100% !important; min-width: 0 !important;
        padding-bottom: 0.75rem !important;
        -webkit-overflow-scrolling: touch !important; scrollbar-width: auto !important;
        scrollbar-color: #888888 #2a2a2a !important;
        flex-direction: row !important; flex-wrap: nowrap !important;
        box-sizing: border-box !important;
        display: flex !important;
    }
    /* Plan day scrollbar: visible so user knows they can scroll (5 days at a time) */
    #plan-day-grid ~ [data-testid="stHorizontalBlock"]::-webkit-scrollbar,
    #plan-day-grid ~ * [data-testid="stHorizontalBlock"]::-webkit-scrollbar,
    #admin-plan-day-grid ~ [data-testid="stHorizontalBlock"]::-webkit-scrollbar,
    #admin-plan-day-grid ~ * [data-testid="stHorizontalBlock"]::-webkit-scrollbar,
    #admin-edit-day-grid ~ [data-testid="stHorizontalBlock"]::-webkit-scrollbar,
    #admin-edit-day-grid ~ * [data-testid="stHorizontalBlock"]::-webkit-scrollbar,
    [data-testid="stMarkdown"]:has(#plan-day-grid) ~ [data-testid="stHorizontalBlock"]::-webkit-scrollbar,
    div:has(#plan-day-grid) ~ div [data-testid="stHorizontalBlock"]::-webkit-scrollbar,
    div:has(#admin-plan-day-grid) ~ div [data-testid="stHorizontalBlock"]::-webkit-scrollbar,
    div:has(#admin-edit-day-grid) ~ div [data-testid="stHorizontalBlock"]::-webkit-scrollbar {
        height: 8px !important;
    }
    #plan-day-grid ~ [data-testid="stHorizontalBlock"]::-webkit-scrollbar-track,
    #plan-day-grid ~ * [data-testid="stHorizontalBlock"]::-webkit-scrollbar-track,
    #admin-plan-day-grid ~ [data-testid="stHorizontalBlock"]::-webkit-scrollbar-track,
    #admin-plan-day-grid ~ * [data-testid="stHorizontalBlock"]::-webkit-scrollbar-track,
    #admin-edit-day-grid ~ [data-testid="stHorizontalBlock"]::-webkit-scrollbar-track,
    #admin-edit-day-grid ~ * [data-testid="stHorizontalBlock"]::-webkit-scrollbar-track,
    [data-testid="stMarkdown"]:has(#plan-day-grid) ~ [data-testid="stHorizontalBlock"]::-webkit-scrollbar-track,
    div:has(#plan-day-grid) ~ div [data-testid="stHorizontalBlock"]::-webkit-scrollbar-track,
    div:has(#admin-plan-day-grid) ~ div [data-testid="stHorizontalBlock"]::-webkit-scrollbar-track,
    div:has(#admin-edit-day-grid) ~ div [data-testid="stHorizontalBlock"]::-webkit-scrollbar-track {
        background: #2a2a2a !important; border-radius: 3px !important;
    }
    #plan-day-grid ~ [data-testid="stHorizontalBlock"]::-webkit-scrollbar-thumb,
    #plan-day-grid ~ * [data-testid="stHorizontalBlock"]::-webkit-scrollbar-thumb,
    #admin-plan-day-grid ~ [data-testid="stHorizontalBlock"]::-webkit-scrollbar-thumb,
    #admin-plan-day-grid ~ * [data-testid="stHorizontalBlock"]::-webkit-scrollbar-thumb,
    #admin-edit-day-grid ~ [data-testid="stHorizontalBlock"]::-webkit-scrollbar-thumb,
    #admin-edit-day-grid ~ * [data-testid="stHorizontalBlock"]::-webkit-scrollbar-thumb,
    [data-testid="stMarkdown"]:has(#plan-day-grid) ~ [data-testid="stHorizontalBlock"]::-webkit-scrollbar-thumb,
    div:has(#plan-day-grid) ~ div [data-testid="stHorizontalBlock"]::-webkit-scrollbar-thumb,
    div:has(#admin-plan-day-grid) ~ div [data-testid="stHorizontalBlock"]::-webkit-scrollbar-thumb,
    div:has(#admin-edit-day-grid) ~ div [data-testid="stHorizontalBlock"]::-webkit-scrollbar-thumb {
        background: #888888 !important; border-radius: 3px !important;
    }
    /* Card: dark grey unselected, white selected (match Faith That Endures), 5 visible + scroll */
    #plan-day-grid ~ [data-testid="stHorizontalBlock"] > *,
    #plan-day-grid ~ * [data-testid="stHorizontalBlock"] > *,
    #admin-plan-day-grid ~ [data-testid="stHorizontalBlock"] > *,
    #admin-plan-day-grid ~ * [data-testid="stHorizontalBlock"] > *,
    #admin-edit-day-grid ~ [data-testid="stHorizontalBlock"] > *,
    #admin-edit-day-grid ~ * [data-testid="stHorizontalBlock"] > *,
    [data-testid="stMarkdown"]:has(#plan-day-grid) ~ [data-testid="stHorizontalBlock"] > *,
    div:has(#plan-day-grid) ~ div [data-testid="stHorizontalBlock"] > *,
    div:has(#admin-plan-day-grid) ~ div [data-testid="stHorizontalBlock"] > *,
    div:has(#admin-edit-day-grid) ~ div [data-testid="stHorizontalBlock"] > * {
        min-width: 4.5rem !important; width: 4.5rem !important; max-width: 4.5rem !important;
        flex: 0 0 4.5rem !important; flex-shrink: 0 !important; flex-grow: 0 !important;
        gap: 0.25rem !important; box-sizing: border-box !important;
        padding: 0.55rem 0.35rem !important; border-radius: 12px !important;
        background: #4a4a4a !important; border: 2px solid #3a3a3a !important;
        overflow: visible !important; min-height: 4rem !important;
        display: flex !important; flex-direction: column !important; align-items: center !important; justify-content: center !important;
    }
    /* Selected day: dark gray card + white border; date in white pill */
    #plan-day-grid ~ [data-testid="stHorizontalBlock"] > *:has(button[kind="primary"]),
    #plan-day-grid ~ * [data-testid="stHorizontalBlock"] > *:has(button[kind="primary"]),
    div:has(#plan-day-grid) ~ div [data-testid="stHorizontalBlock"] > *:has(button[kind="primary"]) {
        background: #4a4a4a !important; border: 2px solid #ffffff !important;
    }
    #plan-day-grid ~ * [data-testid="stHorizontalBlock"],
    #admin-plan-day-grid ~ * [data-testid="stHorizontalBlock"],
    #admin-edit-day-grid ~ [data-testid="stHorizontalBlock"],
    #admin-edit-day-grid ~ * [data-testid="stHorizontalBlock"],
    [data-testid="stMarkdown"]:has(#plan-day-grid) ~ [data-testid="stHorizontalBlock"],
    div:has(#plan-day-grid) ~ div [data-testid="stHorizontalBlock"],
    div:has(#admin-plan-day-grid) ~ div [data-testid="stHorizontalBlock"],
    div:has(#admin-edit-day-grid) ~ div [data-testid="stHorizontalBlock"] {
        gap: 0.4rem !important;
    }
    /* Day column content: allow date/missed to display fully (no clipping) */
    #plan-day-grid ~ [data-testid="stHorizontalBlock"] > * > *,
    #plan-day-grid ~ * [data-testid="stHorizontalBlock"] > * > *,
    #admin-plan-day-grid ~ [data-testid="stHorizontalBlock"] > * > *,
    #admin-plan-day-grid ~ * [data-testid="stHorizontalBlock"] > * > *,
    #admin-edit-day-grid ~ [data-testid="stHorizontalBlock"] > * > *,
    #admin-edit-day-grid ~ * [data-testid="stHorizontalBlock"] > * > *,
    div:has(#plan-day-grid) ~ div [data-testid="stHorizontalBlock"] > * > *,
    div:has(#admin-plan-day-grid) ~ div [data-testid="stHorizontalBlock"] > * > *,
    div:has(#admin-edit-day-grid) ~ div [data-testid="stHorizontalBlock"] > * > * {
        max-width: 100% !important; min-width: 0 !important; overflow: visible !important;
    }
    /* Mobile + Safari: keep day cards in one horizontal row; inside each card show number + date side-by-side */
    @media (max-width: 768px) {
        #plan-day-grid ~ [data-testid="stHorizontalBlock"],
        #plan-day-grid ~ * [data-testid="stHorizontalBlock"],
        #admin-plan-day-grid ~ [data-testid="stHorizontalBlock"],
        #admin-plan-day-grid ~ * [data-testid="stHorizontalBlock"],
        #admin-edit-day-grid ~ [data-testid="stHorizontalBlock"],
        #admin-edit-day-grid ~ * [data-testid="stHorizontalBlock"],
        [data-testid="stMarkdown"]:has(#plan-day-grid) ~ [data-testid="stHorizontalBlock"],
        div:has(#plan-day-grid) ~ div [data-testid="stHorizontalBlock"],
        div:has(#admin-plan-day-grid) ~ div [data-testid="stHorizontalBlock"],
        div:has(#admin-edit-day-grid) ~ div [data-testid="stHorizontalBlock"] {
            display: -webkit-flex !important; display: flex !important;
            -webkit-flex-direction: row !important; flex-direction: row !important;
            -webkit-flex-wrap: nowrap !important; flex-wrap: nowrap !important;
            overflow-x: auto !important; overflow-y: hidden !important;
            -webkit-overflow-scrolling: touch !important;
            width: 24rem !important; max-width: 100% !important; min-width: 0 !important;
        }
        /* Each day card: 5 visible, scroll for more, match reference design */
        #plan-day-grid ~ [data-testid="stHorizontalBlock"] > *,
        #plan-day-grid ~ * [data-testid="stHorizontalBlock"] > *,
        #admin-plan-day-grid ~ [data-testid="stHorizontalBlock"] > *,
        #admin-plan-day-grid ~ * [data-testid="stHorizontalBlock"] > *,
        #admin-edit-day-grid ~ [data-testid="stHorizontalBlock"] > *,
        #admin-edit-day-grid ~ * [data-testid="stHorizontalBlock"] > *,
        [data-testid="stMarkdown"]:has(#plan-day-grid) ~ [data-testid="stHorizontalBlock"] > *,
        div:has(#plan-day-grid) ~ div [data-testid="stHorizontalBlock"] > *,
        div:has(#admin-plan-day-grid) ~ div [data-testid="stHorizontalBlock"] > *,
        div:has(#admin-edit-day-grid) ~ div [data-testid="stHorizontalBlock"] > * {
            -webkit-flex: 0 0 4.5rem !important; flex: 0 0 4.5rem !important;
            min-width: 4.5rem !important; width: 4.5rem !important; max-width: 4.5rem !important;
            min-height: 4.75rem !important;
            background: #4a4a4a !important; border: 2px solid #3a3a3a !important;
            display: -webkit-flex !important; display: flex !important;
            -webkit-flex-shrink: 0 !important; flex-shrink: 0 !important;
            -webkit-flex-direction: column !important; flex-direction: column !important;
            -webkit-align-items: center !important; align-items: center !important;
            -webkit-justify-content: center !important; justify-content: center !important;
            gap: 0.1rem !important;
            overflow: visible !important;
        }
        /* Inner block: column so number on top, date under (centered) */
        #plan-day-grid ~ [data-testid="stHorizontalBlock"] > * > [data-testid="stVerticalBlock"],
        #plan-day-grid ~ * [data-testid="stHorizontalBlock"] > * > [data-testid="stVerticalBlock"],
        #admin-plan-day-grid ~ [data-testid="stHorizontalBlock"] > * > [data-testid="stVerticalBlock"],
        #admin-plan-day-grid ~ * [data-testid="stHorizontalBlock"] > * > [data-testid="stVerticalBlock"],
        #admin-edit-day-grid ~ [data-testid="stHorizontalBlock"] > * > [data-testid="stVerticalBlock"],
        #admin-edit-day-grid ~ * [data-testid="stHorizontalBlock"] > * > [data-testid="stVerticalBlock"],
        [data-testid="stMarkdown"]:has(#plan-day-grid) ~ [data-testid="stHorizontalBlock"] > * > [data-testid="stVerticalBlock"],
        div:has(#plan-day-grid) ~ div [data-testid="stHorizontalBlock"] > * > [data-testid="stVerticalBlock"],
        div:has(#admin-plan-day-grid) ~ div [data-testid="stHorizontalBlock"] > * > [data-testid="stVerticalBlock"],
        div:has(#admin-edit-day-grid) ~ div [data-testid="stHorizontalBlock"] > * > [data-testid="stVerticalBlock"] {
            -webkit-flex-direction: column !important; flex-direction: column !important;
            -webkit-flex-wrap: nowrap !important; flex-wrap: nowrap !important;
            -webkit-align-items: center !important; align-items: center !important;
            width: 100% !important; gap: 0.1rem !important;
        }
        #plan-day-grid ~ [data-testid="stHorizontalBlock"] > * > *,
        #plan-day-grid ~ * [data-testid="stHorizontalBlock"] > * > *,
        #admin-plan-day-grid ~ [data-testid="stHorizontalBlock"] > * > *,
        #admin-plan-day-grid ~ * [data-testid="stHorizontalBlock"] > * > *,
        #admin-edit-day-grid ~ [data-testid="stHorizontalBlock"] > * > *,
        #admin-edit-day-grid ~ * [data-testid="stHorizontalBlock"] > * > *,
        [data-testid="stMarkdown"]:has(#plan-day-grid) ~ [data-testid="stHorizontalBlock"] > * > *,
        div:has(#plan-day-grid) ~ div [data-testid="stHorizontalBlock"] > * > *,
        div:has(#admin-plan-day-grid) ~ div [data-testid="stHorizontalBlock"] > * > *,
        div:has(#admin-edit-day-grid) ~ div [data-testid="stHorizontalBlock"] > * > * {
            -webkit-flex-direction: column !important; flex-direction: column !important;
            -webkit-flex-wrap: nowrap !important; flex-wrap: nowrap !important;
            -webkit-align-items: center !important; align-items: center !important;
            width: 100% !important;
        }
        /* Nested column wrappers: keep horizontal */
        #plan-day-grid ~ * [data-testid="stHorizontalBlock"] [data-testid="stVerticalBlock"],
        #admin-plan-day-grid ~ * [data-testid="stHorizontalBlock"] [data-testid="stVerticalBlock"],
        #admin-edit-day-grid ~ * [data-testid="stHorizontalBlock"] [data-testid="stVerticalBlock"] {
            min-width: 0 !important;
        }
        .plan-day-date { display: block !important; width: 100% !important; font-size: 0.65rem !important; color: #1a1a1a !important; }
    }
    @media (max-width: 380px) {
        #plan-day-grid ~ [data-testid="stHorizontalBlock"],
        #plan-day-grid ~ * [data-testid="stHorizontalBlock"],
        #admin-plan-day-grid ~ [data-testid="stHorizontalBlock"],
        #admin-plan-day-grid ~ * [data-testid="stHorizontalBlock"],
        #admin-edit-day-grid ~ [data-testid="stHorizontalBlock"],
        #admin-edit-day-grid ~ * [data-testid="stHorizontalBlock"],
        [data-testid="stMarkdown"]:has(#plan-day-grid) ~ [data-testid="stHorizontalBlock"],
        div:has(#plan-day-grid) ~ div [data-testid="stHorizontalBlock"],
        div:has(#admin-plan-day-grid) ~ div [data-testid="stHorizontalBlock"],
        div:has(#admin-edit-day-grid) ~ div [data-testid="stHorizontalBlock"] {
            max-width: 100% !important;
        }
        #plan-day-grid ~ [data-testid="stHorizontalBlock"] > *,
        #plan-day-grid ~ * [data-testid="stHorizontalBlock"] > *,
        #admin-plan-day-grid ~ [data-testid="stHorizontalBlock"] > *,
        #admin-plan-day-grid ~ * [data-testid="stHorizontalBlock"] > *,
        #admin-edit-day-grid ~ [data-testid="stHorizontalBlock"] > *,
        #admin-edit-day-grid ~ * [data-testid="stHorizontalBlock"] > *,
        [data-testid="stMarkdown"]:has(#plan-day-grid) ~ [data-testid="stHorizontalBlock"] > *,
        div:has(#plan-day-grid) ~ div [data-testid="stHorizontalBlock"] > *,
        div:has(#admin-plan-day-grid) ~ div [data-testid="stHorizontalBlock"] > *,
        div:has(#admin-edit-day-grid) ~ div [data-testid="stHorizontalBlock"] > * {
            flex: 0 0 4.5rem !important; min-width: 4.5rem !important; width: 4.5rem !important; max-width: 4.5rem !important;
        }
    }

    /* Plan day card: flex layout, number + date centered (admin keeps dark) */
    #admin-plan-day-grid ~ [data-testid="stHorizontalBlock"] > *,
    #admin-plan-day-grid ~ * [data-testid="stHorizontalBlock"] > *,
    #admin-edit-day-grid ~ [data-testid="stHorizontalBlock"] > *,
    #admin-edit-day-grid ~ * [data-testid="stHorizontalBlock"] > *,
    div:has(#admin-plan-day-grid) ~ div [data-testid="stHorizontalBlock"] > *,
    div:has(#admin-edit-day-grid) ~ div [data-testid="stHorizontalBlock"] > * {
        background: #1a1a1a !important; padding: 0.25rem 0.2rem !important; border-radius: 10px !important; margin: 0 0.04rem !important; border: 2px solid #333333 !important;
        display: flex !important; flex-direction: column !important; align-items: center !important; justify-content: center !important; gap: 0.1rem !important;
    }
    /* Inner stack: number then date, centered, minimal gap */
    #plan-day-grid ~ [data-testid="stHorizontalBlock"] > * > [data-testid="stVerticalBlock"],
    #plan-day-grid ~ * [data-testid="stHorizontalBlock"] > * > [data-testid="stVerticalBlock"],
    #admin-plan-day-grid ~ [data-testid="stHorizontalBlock"] > * > [data-testid="stVerticalBlock"],
    #admin-plan-day-grid ~ * [data-testid="stHorizontalBlock"] > * > [data-testid="stVerticalBlock"],
    #admin-edit-day-grid ~ [data-testid="stHorizontalBlock"] > * > [data-testid="stVerticalBlock"],
    #admin-edit-day-grid ~ * [data-testid="stHorizontalBlock"] > * > [data-testid="stVerticalBlock"],
    [data-testid="stMarkdown"]:has(#plan-day-grid) ~ [data-testid="stHorizontalBlock"] > * > [data-testid="stVerticalBlock"],
    div:has(#plan-day-grid) ~ div [data-testid="stHorizontalBlock"] > * > [data-testid="stVerticalBlock"],
    div:has(#admin-plan-day-grid) ~ div [data-testid="stHorizontalBlock"] > * > [data-testid="stVerticalBlock"],
    div:has(#admin-edit-day-grid) ~ div [data-testid="stHorizontalBlock"] > * > [data-testid="stVerticalBlock"] {
        display: flex !important; flex-direction: column !important; align-items: center !important; justify-content: center !important; width: 100% !important; gap: 0.1rem !important;
    }
    #plan-day-grid ~ [data-testid="stHorizontalBlock"] > * > *,
    #plan-day-grid ~ * [data-testid="stHorizontalBlock"] > * > *,
    #admin-plan-day-grid ~ [data-testid="stHorizontalBlock"] > * > *,
    #admin-plan-day-grid ~ * [data-testid="stHorizontalBlock"] > * > *,
    [data-testid="stMarkdown"]:has(#plan-day-grid) ~ [data-testid="stHorizontalBlock"] > * > *,
    div:has(#plan-day-grid) ~ div [data-testid="stHorizontalBlock"] > * > *,
    div:has(#admin-plan-day-grid) ~ div [data-testid="stHorizontalBlock"] > * > * {
        display: flex !important; flex-direction: column !important; align-items: center !important; width: 100% !important; box-sizing: border-box !important; gap: 0.05rem !important;
    }
    /* Workout number button container: center, no wrap */
    #plan-day-grid ~ [data-testid="stHorizontalBlock"] > * .stButton,
    #plan-day-grid ~ * [data-testid="stHorizontalBlock"] > * .stButton,
    #admin-plan-day-grid ~ [data-testid="stHorizontalBlock"] > * .stButton,
    #admin-plan-day-grid ~ * [data-testid="stHorizontalBlock"] > * .stButton,
    #admin-edit-day-grid ~ [data-testid="stHorizontalBlock"] > * .stButton,
    #admin-edit-day-grid ~ * [data-testid="stHorizontalBlock"] > * .stButton,
    [data-testid="stMarkdown"]:has(#plan-day-grid) ~ [data-testid="stHorizontalBlock"] > * .stButton,
    div:has(#plan-day-grid) ~ div [data-testid="stHorizontalBlock"] > * .stButton,
    div:has(#admin-plan-day-grid) ~ div [data-testid="stHorizontalBlock"] > * .stButton,
    div:has(#admin-edit-day-grid) ~ div [data-testid="stHorizontalBlock"] > * .stButton {
        display: flex !important; justify-content: center !important; align-items: center !important; width: 100% !important; flex-wrap: nowrap !important;
    }
    /* No white box for selected — all cards look the same */
    /* Day number: centered in grey box, black text, no white button/box */
    #plan-day-grid ~ [data-testid="stHorizontalBlock"] .stButton button,
    #plan-day-grid ~ * [data-testid="stHorizontalBlock"] .stButton button,
    [data-testid="stMarkdown"]:has(#plan-day-grid) ~ [data-testid="stHorizontalBlock"] .stButton button,
    #admin-plan-day-grid ~ [data-testid="stHorizontalBlock"] .stButton button,
    #admin-plan-day-grid ~ * [data-testid="stHorizontalBlock"] .stButton button,
    #admin-edit-day-grid ~ [data-testid="stHorizontalBlock"] .stButton button,
    #admin-edit-day-grid ~ * [data-testid="stHorizontalBlock"] .stButton button,
    div:has(#plan-day-grid) ~ div [data-testid="stHorizontalBlock"] .stButton button,
    div:has(#admin-plan-day-grid) ~ div [data-testid="stHorizontalBlock"] .stButton button,
    div:has(#admin-edit-day-grid) ~ div [data-testid="stHorizontalBlock"] .stButton button {
        min-width: 100% !important; width: 100% !important; max-width: 100% !important; height: auto !important; min-height: 1.5rem !important;
        border-radius: 0 !important; font-weight: 600 !important; font-size: 1.1rem !important;
        background: transparent !important; color: #111111 !important; border: none !important; box-shadow: none !important;
        white-space: nowrap !important; padding: 0.2rem 0 !important;
        display: -webkit-inline-flex !important; display: inline-flex !important;
        -webkit-align-items: center !important; align-items: center !important;
        -webkit-justify-content: center !important; justify-content: center !important;
        flex-wrap: nowrap !important; word-break: keep-all !important; overflow: visible !important; text-overflow: clip !important;
    }
    /* Plan day: number-only button — override nowrap above */
    #plan-day-grid ~ [data-testid="stHorizontalBlock"] .stButton button,
    #plan-day-grid ~ * [data-testid="stHorizontalBlock"] .stButton button,
    div:has(#plan-day-grid) ~ div [data-testid="stHorizontalBlock"] .stButton button {
        white-space: nowrap !important; min-height: 1.5rem !important;
    }
    /* Force button label one line for admin (10+ days); plan-day-grid uses multi-line for number+date+status */
    #admin-plan-day-grid ~ [data-testid="stHorizontalBlock"] .stButton button *,
    #admin-plan-day-grid ~ * [data-testid="stHorizontalBlock"] .stButton button *,
    #admin-edit-day-grid ~ [data-testid="stHorizontalBlock"] .stButton button *,
    #admin-edit-day-grid ~ * [data-testid="stHorizontalBlock"] .stButton button *,
    div:has(#admin-plan-day-grid) ~ div [data-testid="stHorizontalBlock"] .stButton button *,
    div:has(#admin-edit-day-grid) ~ div [data-testid="stHorizontalBlock"] .stButton button * {
        white-space: nowrap !important; display: inline !important; flex-shrink: 0 !important;
        font-size: inherit !important; line-height: inherit !important;
    }
    /* Day of week: below number, centered */
    .plan-day-date {
        background: transparent !important; color: #1a1a1a !important; font-size: 0.65rem !important; text-align: center !important;
        margin: 0.15rem auto 0 !important; padding: 0 !important; line-height: 1.2 !important; width: 100% !important;
        display: block !important; overflow: visible !important;
    }
    .plan-day-date-selected {
        background: transparent !important; color: #1a1a1a !important; padding: 0 !important;
    }
    /* Date + missed/complete block: date on top, label directly below */
    .plan-day-date-block {
        display: flex !important; flex-direction: column !important; align-items: center !important;
        width: 100% !important; max-width: 4.5rem !important; gap: 0.1rem !important;
        margin: 0.2rem 0 0 !important; padding: 0 !important; overflow: hidden !important;
    }
    .plan-day-date-block .plan-day-date { margin-bottom: 0 !important; }
    .plan-day-date-block .plan-day-missed { margin-top: 0 !important; }
    .plan-day-date-block .plan-day-complete-label { margin-top: 0 !important; }
    /* Green/red cards: white text for contrast */
    #plan-day-grid ~ * [data-testid="stHorizontalBlock"] > *:has(.plan-day-complete) .stButton button,
    #plan-day-grid ~ * [data-testid="stHorizontalBlock"] > *:has(.plan-day-complete) .plan-day-date,
    div:has(#plan-day-grid) ~ div [data-testid="stHorizontalBlock"] > *:has(.plan-day-complete) .stButton button,
    div:has(#plan-day-grid) ~ div [data-testid="stHorizontalBlock"] > *:has(.plan-day-complete) .plan-day-date {
        color: #ffffff !important;
    }
    #admin-plan-day-grid ~ * [data-testid="stHorizontalBlock"] > *:has(.admin-day-complete) .stButton button,
    #admin-plan-day-grid ~ * [data-testid="stHorizontalBlock"] > *:has(.admin-day-missed-marker) .stButton button,
    #admin-edit-day-grid ~ * [data-testid="stHorizontalBlock"] > *:has(.admin-day-complete) .stButton button,
    #admin-edit-day-grid ~ * [data-testid="stHorizontalBlock"] > *:has(.admin-day-missed-marker) .stButton button,
    div:has(#admin-plan-day-grid) ~ div [data-testid="stHorizontalBlock"] > *:has(.admin-day-complete) .stButton button,
    div:has(#admin-plan-day-grid) ~ div [data-testid="stHorizontalBlock"] > *:has(.admin-day-missed-marker) .stButton button,
    div:has(#admin-edit-day-grid) ~ div [data-testid="stHorizontalBlock"] > *:has(.admin-day-complete) .stButton button,
    div:has(#admin-edit-day-grid) ~ div [data-testid="stHorizontalBlock"] > *:has(.admin-day-missed-marker) .stButton button {
        color: #ffffff !important;
    }
    /* Day missed: no red — keep dark grey card, show "Missed day" text below date */
    .plan-day-missed-marker { display: none; }
    /* Missed day: text below date, outside the card content area, within column width */
    .plan-day-missed {
        font-size: 0.55rem !important; color: #b0b0b0 !important; margin: 0.15rem 0 0 !important;
        padding: 0 !important; line-height: 1.2 !important; text-align: center !important;
        display: block !important; font-weight: 500 !important;
        max-width: 4.5rem !important; overflow: hidden !important; text-overflow: ellipsis !important;
        white-space: nowrap !important;
    }
    /* Day complete: text under date (white on green card) */
    .plan-day-complete-label {
        font-size: 0.6rem !important; color: #ffffff !important; margin: 0 !important; padding: 0 !important;
        line-height: 1.2 !important; text-align: center !important; width: 100% !important; display: block !important;
        font-weight: 600 !important;
    }
    /* Player day complete: whole day card turns green */
    .plan-day-complete { display: none; }
    #plan-day-grid ~ * [data-testid="stHorizontalBlock"] > *:has(.plan-day-complete),
    [data-testid="stMarkdown"]:has(#plan-day-grid) ~ [data-testid="stHorizontalBlock"] > *:has(.plan-day-complete),
    div:has(#plan-day-grid) ~ div [data-testid="stHorizontalBlock"] > *:has(.plan-day-complete) {
        background: #16a34a !important;
    }
    /* Admin plan day selector: same as player plan — 5 visible, scroll, fixed card size, number then date */
    #admin-plan-day-grid ~ * [data-testid="stHorizontalBlock"] .plan-day-date,
    #admin-edit-day-grid ~ * [data-testid="stHorizontalBlock"] .plan-day-date {
        background: transparent !important; color: #1a1a1a !important; margin: 0 !important; padding: 0 !important; font-size: 0.65rem !important; line-height: 1.2 !important; text-align: center !important; width: 100% !important; display: block !important;
    }
    #admin-plan-day-grid ~ * [data-testid="stHorizontalBlock"] .plan-day-date-selected,
    #admin-edit-day-grid ~ * [data-testid="stHorizontalBlock"] .plan-day-date-selected {
        background: transparent !important; color: #1a1a1a !important; padding: 0 !important;
    }
    /* Admin day complete: card turns green, number crossed off */
    .admin-day-complete { display: none; }
    #admin-plan-day-grid ~ * [data-testid="stHorizontalBlock"] > *:has(.admin-day-complete),
    #admin-edit-day-grid ~ * [data-testid="stHorizontalBlock"] > *:has(.admin-day-complete),
    div:has(#admin-plan-day-grid) ~ div [data-testid="stHorizontalBlock"] > *:has(.admin-day-complete),
    div:has(#admin-edit-day-grid) ~ div [data-testid="stHorizontalBlock"] > *:has(.admin-day-complete) {
        background: #16a34a !important;
    }
    /* Admin day missed: card turns red */
    .admin-day-missed-marker { display: none; }
    #admin-plan-day-grid ~ * [data-testid="stHorizontalBlock"] > *:has(.admin-day-missed-marker),
    #admin-edit-day-grid ~ * [data-testid="stHorizontalBlock"] > *:has(.admin-day-missed-marker),
    div:has(#admin-plan-day-grid) ~ div [data-testid="stHorizontalBlock"] > *:has(.admin-day-missed-marker),
    div:has(#admin-edit-day-grid) ~ div [data-testid="stHorizontalBlock"] > *:has(.admin-day-missed-marker) {
        background: #dc2626 !important;
    }
    /* Admin mode buttons: white/gray theme (incomplete = outline, complete = filled white) */
    #admin-plan-modes ~ * .stButton button {
        background: transparent !important; color: #ffffff !important; border: 1px solid #666666 !important;
    }
    #admin-plan-modes ~ * .stButton button[kind="primary"] {
        background: #ffffff !important; color: #000000 !important; border: 1px solid #ffffff !important;
    }

    /* Player My Plan mode buttons: same black/white style */
    #plan-modes ~ * .stButton button {
        background: transparent !important; color: #ffffff !important; border: 1px solid #666666 !important;
    }
    #plan-modes ~ * .stButton button[kind="primary"] {
        background: #ffffff !important; color: #000000 !important; border: 1px solid #ffffff !important;
    }

    /* Classic tab style: sleek, simple — words fit in each tab, light grey line, even spacing */
    [data-testid="stMarkdown"]:has(#admin-tab-bar) + div [data-testid="stHorizontalBlock"],
    div:has(#admin-tab-bar) + div [data-testid="stHorizontalBlock"] {
        display: flex !important; gap: 8px !important; margin-bottom: 1rem !important;
        border-bottom: 1px solid #888888 !important; padding-bottom: 0 !important; flex-wrap: wrap !important;
        align-items: flex-end !important; justify-content: flex-start !important;
    }
    /* Player tabs: tight gap like admin — container wrapper ensures selector matches */
    div:has(#player-tab-bar) [data-testid="stHorizontalBlock"],
    [data-testid="stMarkdown"]:has(#player-tab-bar[data-tab-style="classic"]) + div [data-testid="stHorizontalBlock"],
    [data-testid="stMarkdown"]:has(#player-tab-bar[data-tab-style="classic"]) + * [data-testid="stHorizontalBlock"],
    [data-testid="stMarkdown"]:has(#player-tab-bar[data-tab-style="classic"]) + [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"],
    div:has(#player-tab-bar[data-tab-style="classic"]) + div [data-testid="stHorizontalBlock"],
    div:has(#player-tab-bar[data-tab-style="classic"]) + * [data-testid="stHorizontalBlock"],
    div:has(#player-tab-bar[data-tab-style="classic"]) + [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] {
        display: flex !important; gap: 8px !important; margin-bottom: 1rem !important;
        border-bottom: 1px solid #888888 !important; padding-bottom: 0 !important; flex-wrap: wrap !important;
        align-items: flex-end !important; justify-content: flex-start !important;
    }
    [data-testid="stMarkdown"]:has(#admin-tab-bar) + div [data-testid="stHorizontalBlock"] > div,
    div:has(#admin-tab-bar) + div [data-testid="stHorizontalBlock"] > div,
    div:has(#player-tab-bar) [data-testid="stHorizontalBlock"] > div,
    [data-testid="stMarkdown"]:has(#player-tab-bar[data-tab-style="classic"]) + div [data-testid="stHorizontalBlock"] > div,
    [data-testid="stMarkdown"]:has(#player-tab-bar[data-tab-style="classic"]) + * [data-testid="stHorizontalBlock"] > div,
    [data-testid="stMarkdown"]:has(#player-tab-bar[data-tab-style="classic"]) + [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > div,
    div:has(#player-tab-bar[data-tab-style="classic"]) + div [data-testid="stHorizontalBlock"] > div,
    div:has(#player-tab-bar[data-tab-style="classic"]) + * [data-testid="stHorizontalBlock"] > div,
    div:has(#player-tab-bar[data-tab-style="classic"]) + [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > div {
        flex: 0 0 auto !important; min-width: min-content !important; margin: 0 !important; padding: 0 !important;
    }
    [data-testid="stMarkdown"]:has(#admin-tab-bar) + div [data-testid="stHorizontalBlock"] .stButton button,
    div:has(#admin-tab-bar) + div [data-testid="stHorizontalBlock"] .stButton button,
    [data-testid="stMarkdown"]:has(#player-tab-bar[data-tab-style="classic"]) + div [data-testid="stHorizontalBlock"] .stButton button,
    div:has(#player-tab-bar[data-tab-style="classic"]) + div [data-testid="stHorizontalBlock"] .stButton button {
        background: #2e2e2e !important; border: 1px solid #3a3a3a !important; border-radius: 8px 8px 0 0 !important;
        border-bottom: 2px solid transparent !important; margin-bottom: -1px !important;
        color: #cccccc !important; font-weight: 500 !important; font-size: 0.75rem !important;
        padding: 0.25rem 0.5rem !important; box-shadow: none !important;
        min-height: auto !important; line-height: 1.2 !important; white-space: nowrap !important;
        min-width: min-content !important;
    }
    [data-testid="stMarkdown"]:has(#admin-tab-bar) + div [data-testid="stHorizontalBlock"] .stButton button:hover,
    div:has(#admin-tab-bar) + div [data-testid="stHorizontalBlock"] .stButton button:hover,
    [data-testid="stMarkdown"]:has(#player-tab-bar[data-tab-style="classic"]) + div [data-testid="stHorizontalBlock"] .stButton button:hover,
    div:has(#player-tab-bar[data-tab-style="classic"]) + div [data-testid="stHorizontalBlock"] .stButton button:hover {
        background: #383838 !important; color: #dddddd !important; border-color: #444444 !important;
    }
    [data-testid="stMarkdown"]:has(#admin-tab-bar) + div [data-testid="stHorizontalBlock"] .stButton button[kind="primary"],
    div:has(#admin-tab-bar) + div [data-testid="stHorizontalBlock"] .stButton button[kind="primary"],
    [data-testid="stMarkdown"]:has(#player-tab-bar[data-tab-style="classic"]) + div [data-testid="stHorizontalBlock"] .stButton button[kind="primary"],
    div:has(#player-tab-bar[data-tab-style="classic"]) + div [data-testid="stHorizontalBlock"] .stButton button[kind="primary"] {
        background: #e0e0e0 !important; color: #333333 !important; font-weight: 600 !important;
        border-color: #e0e0e0 !important; border-bottom: 3px solid #888888 !important; margin-bottom: -2px !important;
    }
    @media (max-width: 640px) {
        [data-testid="stMarkdown"]:has(#admin-tab-bar) + div [data-testid="stHorizontalBlock"],
        div:has(#admin-tab-bar) + div [data-testid="stHorizontalBlock"] {
            gap: 8px !important;
        }
        div:has(#player-tab-bar) [data-testid="stHorizontalBlock"],
        [data-testid="stMarkdown"]:has(#player-tab-bar[data-tab-style="classic"]) + div [data-testid="stHorizontalBlock"],
        div:has(#player-tab-bar[data-tab-style="classic"]) + div [data-testid="stHorizontalBlock"] {
            gap: 8px !important;
        }
        [data-testid="stMarkdown"]:has(#admin-tab-bar) + div [data-testid="stHorizontalBlock"] .stButton button,
        div:has(#admin-tab-bar) + div [data-testid="stHorizontalBlock"] .stButton button,
        [data-testid="stMarkdown"]:has(#player-tab-bar[data-tab-style="classic"]) + div [data-testid="stHorizontalBlock"] .stButton button,
        div:has(#player-tab-bar[data-tab-style="classic"]) + div [data-testid="stHorizontalBlock"] .stButton button {
            padding: 0.22rem 0.4rem !important; font-size: 0.75rem !important;
        }
    }

    /* Player tab bar: underline style (only when data-tab-style="underline" — otherwise Classic applies) */
    [data-testid="stMarkdown"]:has(#player-tab-bar[data-tab-style="underline"]) + div [data-testid="stHorizontalBlock"],
    div:has(#player-tab-bar[data-tab-style="underline"]) + div [data-testid="stHorizontalBlock"] {
        display: flex !important; gap: 0 !important; margin-bottom: 1.25rem !important;
        padding-bottom: 0 !important; flex-wrap: wrap !important;
        border-bottom: 1px solid #444444 !important;
    }
    [data-testid="stMarkdown"]:has(#player-tab-bar[data-tab-style="underline"]) + div [data-testid="stHorizontalBlock"] .stButton button,
    div:has(#player-tab-bar[data-tab-style="underline"]) + div [data-testid="stHorizontalBlock"] .stButton button {
        background: transparent !important; border: none !important; border-radius: 0 !important;
        border-bottom: 3px solid transparent !important; margin-bottom: -1px !important;
        color: #999999 !important; font-weight: 500 !important; font-size: 0.95rem !important;
        padding: 0.7rem 1rem !important; box-shadow: none !important;
    }
    [data-testid="stMarkdown"]:has(#player-tab-bar[data-tab-style="underline"]) + div [data-testid="stHorizontalBlock"] .stButton button:hover,
    div:has(#player-tab-bar[data-tab-style="underline"]) + div [data-testid="stHorizontalBlock"] .stButton button:hover {
        color: #cccccc !important; background: transparent !important;
    }
    [data-testid="stMarkdown"]:has(#player-tab-bar[data-tab-style="underline"]) + div [data-testid="stHorizontalBlock"] .stButton button[kind="primary"],
    div:has(#player-tab-bar[data-tab-style="underline"]) + div [data-testid="stHorizontalBlock"] .stButton button[kind="primary"] {
        color: #ffffff !important; font-weight: 600 !important; border-bottom-color: #ffffff !important;
    }
    @media (max-width: 640px) {
        [data-testid="stMarkdown"]:has(#player-tab-bar[data-tab-style="underline"]) + div [data-testid="stHorizontalBlock"] .stButton button,
        div:has(#player-tab-bar[data-tab-style="underline"]) + div [data-testid="stHorizontalBlock"] .stButton button {
            padding: 0.6rem 0.75rem !important; font-size: 0.9rem !important;
        }
    }
    /* Pill style */
    [data-testid="stMarkdown"]:has(#player-tab-bar[data-tab-style="pill"]) + div [data-testid="stHorizontalBlock"] .stButton button,
    div:has(#player-tab-bar[data-tab-style="pill"]) + div [data-testid="stHorizontalBlock"] .stButton button {
        background: #3a3a3a !important; border: 1px solid #4a4a4a !important; border-radius: 999px !important;
        border-bottom: 1px solid #4a4a4a !important; color: #b0b0b0 !important; padding: 0.5rem 1.25rem !important;
    }
    [data-testid="stMarkdown"]:has(#player-tab-bar[data-tab-style="pill"]) + div [data-testid="stHorizontalBlock"] .stButton button:hover,
    div:has(#player-tab-bar[data-tab-style="pill"]) + div [data-testid="stHorizontalBlock"] .stButton button:hover {
        background: #444444 !important; color: #e0e0e0 !important; border-color: #555555 !important;
    }
    [data-testid="stMarkdown"]:has(#player-tab-bar[data-tab-style="pill"]) + div [data-testid="stHorizontalBlock"] .stButton button[kind="primary"],
    div:has(#player-tab-bar[data-tab-style="pill"]) + div [data-testid="stHorizontalBlock"] .stButton button[kind="primary"] {
        background: #ffffff !important; color: #000000 !important; border-color: #ffffff !important; font-weight: 600 !important;
    }
    /* Bordered style */
    [data-testid="stMarkdown"]:has(#player-tab-bar[data-tab-style="bordered"]) + div [data-testid="stHorizontalBlock"] .stButton button,
    div:has(#player-tab-bar[data-tab-style="bordered"]) + div [data-testid="stHorizontalBlock"] .stButton button {
        background: transparent !important; border: 1px solid #444444 !important; border-radius: 0 !important;
        color: #999999 !important; padding: 0.5rem 1.25rem !important;
    }
    [data-testid="stMarkdown"]:has(#player-tab-bar[data-tab-style="bordered"]) + div [data-testid="stHorizontalBlock"] .stButton button[kind="primary"],
    div:has(#player-tab-bar[data-tab-style="bordered"]) + div [data-testid="stHorizontalBlock"] .stButton button[kind="primary"] {
        background: #1a1a1a !important; color: #ffffff !important; border-color: #ffffff !important; font-weight: 600 !important;
    }
    /* Segmented style */
    [data-testid="stMarkdown"]:has(#player-tab-bar[data-tab-style="segmented"]) + div [data-testid="stHorizontalBlock"],
    div:has(#player-tab-bar[data-tab-style="segmented"]) + div [data-testid="stHorizontalBlock"] {
        background: #2a2a2a !important; border-radius: 10px !important; overflow: hidden !important;
        padding: 0.25rem !important; gap: 0 !important;
    }
    [data-testid="stMarkdown"]:has(#player-tab-bar[data-tab-style="segmented"]) + div [data-testid="stHorizontalBlock"] .stButton button,
    div:has(#player-tab-bar[data-tab-style="segmented"]) + div [data-testid="stHorizontalBlock"] .stButton button {
        background: transparent !important; border: none !important; border-right: 1px solid #3a3a3a !important;
        border-radius: 0 !important; color: #888888 !important;
    }
    [data-testid="stMarkdown"]:has(#player-tab-bar[data-tab-style="segmented"]) + div [data-testid="stHorizontalBlock"] .stButton:last-child button,
    div:has(#player-tab-bar[data-tab-style="segmented"]) + div [data-testid="stHorizontalBlock"] .stButton:last-child button {
        border-right: none !important;
    }
    [data-testid="stMarkdown"]:has(#player-tab-bar[data-tab-style="segmented"]) + div [data-testid="stHorizontalBlock"] .stButton button[kind="primary"],
    div:has(#player-tab-bar[data-tab-style="segmented"]) + div [data-testid="stHorizontalBlock"] .stButton button[kind="primary"] {
        background: #4a4a4a !important; color: #ffffff !important;
    }
    /* Floating style — selected tab appears raised with shadow, like a card */
    [data-testid="stMarkdown"]:has(#player-tab-bar[data-tab-style="floating"]) + div [data-testid="stHorizontalBlock"] .stButton button,
    div:has(#player-tab-bar[data-tab-style="floating"]) + div [data-testid="stHorizontalBlock"] .stButton button {
        background: #2a2a2a !important; border: 1px solid #3a3a3a !important; border-radius: 8px !important;
        color: #999999 !important; padding: 0.55rem 1.2rem !important; box-shadow: none !important;
    }
    [data-testid="stMarkdown"]:has(#player-tab-bar[data-tab-style="floating"]) + div [data-testid="stHorizontalBlock"] .stButton button:hover,
    div:has(#player-tab-bar[data-tab-style="floating"]) + div [data-testid="stHorizontalBlock"] .stButton button:hover {
        background: #333333 !important; color: #cccccc !important; border-color: #555555 !important;
    }
    [data-testid="stMarkdown"]:has(#player-tab-bar[data-tab-style="floating"]) + div [data-testid="stHorizontalBlock"] .stButton button[kind="primary"],
    div:has(#player-tab-bar[data-tab-style="floating"]) + div [data-testid="stHorizontalBlock"] .stButton button[kind="primary"] {
        background: #ffffff !important; color: #000000 !important; border-color: #ffffff !important;
        font-weight: 600 !important; box-shadow: 0 4px 12px rgba(0,0,0,0.4) !important; transform: translateY(-1px) !important;
    }

    /* Plan day cards: number (button) large on top, date below */
    #plan-day-grid ~ [data-testid="stHorizontalBlock"] .stButton button,
    #plan-day-grid ~ * [data-testid="stHorizontalBlock"] .stButton button,
    div:has(#plan-day-grid) ~ div [data-testid="stHorizontalBlock"] .stButton button {
        background: transparent !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: none !important;
        line-height: 1.2 !important;
        text-align: center !important;
        width: 100% !important;
        min-height: auto !important;
        padding: 0.25rem 0.2rem !important;
        overflow: visible !important;
        font-size: 1.5rem !important; font-weight: 600 !important;
    }
    /* Selected day: white number (same as others), date styled via plan-day-date-pill */
    #plan-day-grid ~ [data-testid="stHorizontalBlock"] .stButton button[kind="primary"],
    #plan-day-grid ~ * [data-testid="stHorizontalBlock"] .stButton button[kind="primary"],
    div:has(#plan-day-grid) ~ div [data-testid="stHorizontalBlock"] .stButton button[kind="primary"] {
        background: transparent !important;
        color: #ffffff !important;
    }
    /* Plan day date: inactive = light gray, selected = white pill */
    #plan-day-grid ~ * .plan-day-date,
    div:has(#plan-day-grid) .plan-day-date {
        margin: 0.15rem 0 0 !important; padding: 0 !important; font-size: 0.75rem !important;
        color: #b0b0b0 !important; text-align: center !important; width: 100% !important;
        line-height: 1.3 !important;
    }
    #plan-day-grid ~ * .plan-day-date-pill,
    div:has(#plan-day-grid) .plan-day-date-pill,
    .stMarkdown .plan-day-date-pill {
        display: inline-block !important; background: #ffffff !important;
        color: #000000 !important; -webkit-text-fill-color: #000000 !important;
        padding: 0.2rem 0.5rem !important; border-radius: 999px !important; margin-top: 0.2rem !important;
        font-size: 0.7rem !important; font-weight: 500 !important;
    }

    /* Form card (black/white theme) */
    .form-card {
        background: #1a1a1a;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(255,255,255,0.05);
        border: 1px solid #333333;
    }

    /* Workout result cards (black/white theme) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #1a1a1a !important;
        border: 1px solid #333333 !important;
        border-radius: 12px !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 0.75rem !important;
        box-shadow: 0 1px 3px rgba(255,255,255,0.05) !important;
    }

    /* Tabs: black/white theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.75rem;
        border-bottom: 1px solid #333333;
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif;
        color: #cccccc;
        padding: 0.5rem 1rem;
        margin-right: 0.25rem;
        border: 1px solid #333333;
        border-bottom: none;
        border-radius: 8px 8px 0 0;
        background: #1a1a1a;
    }
    .stTabs [data-baseweb="tab"]:first-child { margin-left: 0; }
    .stTabs [aria-selected="true"] {
        color: #000000 !important;
        background: #e8e8e8 !important;
        border-color: #555555 !important;
        border-bottom: 1px solid transparent !important;
        margin-bottom: -1px;
    }
    .stTabs [data-baseweb="tab"]:hover { background: #333333 !important; color: #ffffff !important; }
    .stTabs [data-baseweb="tab"]:focus-visible {
        outline: 2px solid #ffffff; outline-offset: 2px;
    }

    /* Form card: session options container (black/white) */
    .main .block-container div[data-testid="stVerticalBlock"]:has(.form-card-marker) {
        background: #1a1a1a;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(255,255,255,0.05);
        border: 1px solid #333333;
    }

    .stButton button {
        font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
        background: #ffffff !important; color: #000000 !important; border: 1px solid #ffffff !important;
        border-radius: 8px !important; padding: 0.5rem 1.5rem !important;
    }
    .stButton button:hover { background: #e0e0e0 !important; color: #000000 !important; }

    .workout-display-wrapper, #workout-result-section, #workout-result { width: 100% !important; max-width: 100% !important; }
    .workout-result-header { font-family: 'DM Sans', sans-serif; font-weight: 600; color: #ffffff; font-size: 1.05rem; margin-bottom: 0.35rem; }
    .workout-result-badge {
        display: inline-block; background: #333333; color: #ffffff;
        padding: 0.2rem 0.5rem; border-radius: 6px; font-size: 0.8rem; margin-bottom: 0.75rem;
    }
    .stMarkdown p, .stMarkdown li, .stMarkdown ul { color: #e0e0e0 !important; }
    .stCaption { color: #cccccc !important; }

    /* Your Work stats card */
    .your-work-stats-card {
        background: #1a1a1a; border: 1px solid #333333; border-radius: 12px; padding: 1.25rem 1.5rem;
        max-width: 24rem; margin-top: 0.5rem; font-family: 'DM Sans', sans-serif;
    }
    .your-work-section {
        display: flex; justify-content: space-between; align-items: center; padding: 0.35rem 0;
    }
    .your-work-label { color: #ffffff; font-weight: 700; font-size: 1rem; }
    .your-work-value { color: #ffffff; font-weight: 600; font-size: 1.1rem; }
    .your-work-divider {
        height: 1px; background: #333333; margin: 0.5rem 0;
    }
    .your-work-row {
        display: flex; justify-content: space-between; align-items: center; padding: 0.25rem 0;
    }
    .your-work-cat { color: #cccccc; font-size: 0.9rem; }
    .your-work-num { color: #e0e0e0; font-size: 0.9rem; font-weight: 500; }
    .your-work-footer {
        margin-top: 0.75rem; color: #888888; font-size: 0.8rem; text-align: center;
    }

    /* Workout headers and content: bold headers, full width, wide layout (desktop app, browser, mobile) */
    *:has(#workout-result-section) .stSubheader,
    *:has(#workout-result-section) .workout-result-header,
    *:has(#workout-result-section) h3 {
        font-weight: 700 !important;
        width: 100% !important;
        max-width: 100% !important;
    }
    /* Workout section: full width, no horizontal compression (fixes browser layout) */
    .main .block-container:has(#workout-result-section),
    .main:has(#workout-result-section) .block-container,
    div:has(#workout-result-section) {
        max-width: min(1800px, 98vw) !important;
        width: 100% !important;
    }
    *:has(#workout-result-section) .stTabs [data-testid="stVerticalBlockBorderWrapper"],
    *:has(#workout-result-section) .stTabs [data-testid="stVerticalBlock"],
    *:has(#workout-result-section) .stTabs,
    *:has(#workout-result-section) [data-testid="stVerticalBlock"],
    *:has(#workout-result-section) .stMarkdown,
    *:has(#workout-result-section) [data-testid="stMarkdown"],
    *:has(#workout-result-section) .stMarkdown > div,
    *:has(#workout-result-section) p, *:has(#workout-result-section) pre, *:has(#workout-result-section) code {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
    }
    /* Workout display: use entire section below tabs+Clear, spread out text, fill main area */
    *:has(#workout-result-section) .stTabs [role="tabpanel"] {
        min-height: calc(100vh - 300px) !important;
        width: 100% !important;
        overflow: visible !important;
        display: block !important;
    }
    *:has(#workout-result-section) .stTabs [data-testid="stVerticalBlockBorderWrapper"] {
        padding: 1.25rem 1.5rem !important;
        width: 100% !important;
        max-width: 100% !important;
    }
    /* Ensure workout tabs column fills available width (fixes empty black area in browser) */
    *:has(#workout-tabs-clear-row) ~ [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child {
        flex: 1 1 auto !important;
        min-width: 0 !important;
        max-width: 100% !important;
    }
    *:has(#workout-result-section) .stTabs .stMarkdown p,
    *:has(#workout-result-section) .stTabs .stMarkdown li {
        line-height: 1.85 !important;
    }

    /* Workout tabs + Clear workout: tabs column fills width, Clear is compact */
    [data-testid="stMarkdown"]:has(#workout-tabs-clear-row) ~ [data-testid="stHorizontalBlock"]:first-of-type {
        gap: 0.5rem !important;
        width: 100% !important;
    }
    [data-testid="stMarkdown"]:has(#workout-tabs-clear-row) ~ [data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"]:first-child {
        flex: 1 1 auto !important;
        min-width: 0 !important;
    }
    [data-testid="stMarkdown"]:has(#workout-tabs-clear-row) ~ [data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"]:last-child {
        flex: 0 0 auto !important;
        min-width: auto !important;
    }
    [data-testid="stMarkdown"]:has(#workout-tabs-clear-row) ~ [data-testid="stHorizontalBlock"]:first-of-type .stButton button {
        white-space: nowrap !important;
        padding: 0.5rem 1rem !important;
    }

    /* Generate workout + Request Custom Plan — side by side */
    /* Buttons: full text visible, no truncation. Marker ensures Chrome targets correctly. */
    [data-testid="stMarkdown"]:has(#generate-request-buttons) ~ [data-testid="stHorizontalBlock"],
    div:has(#generate-request-buttons) ~ div [data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-wrap: nowrap !important;
        gap: 0.75rem !important;
        width: 100% !important;
        min-width: 0 !important;
    }
    [data-testid="stMarkdown"]:has(#generate-request-buttons) ~ [data-testid="stHorizontalBlock"] [data-testid="column"],
    div:has(#generate-request-buttons) ~ div [data-testid="stHorizontalBlock"] [data-testid="column"] {
        flex: 1 1 0 !important;
        min-width: 140px !important;
        max-width: 50% !important;
    }
    [data-testid="stMarkdown"]:has(#generate-request-buttons) ~ [data-testid="stHorizontalBlock"] .stButton,
    div:has(#generate-request-buttons) ~ div [data-testid="stHorizontalBlock"] .stButton {
        width: 100% !important;
        min-width: 0 !important;
        overflow: visible !important;
    }
    [data-testid="stMarkdown"]:has(#generate-request-buttons) ~ [data-testid="stHorizontalBlock"] .stButton button,
    div:has(#generate-request-buttons) ~ div [data-testid="stHorizontalBlock"] .stButton button {
        width: 100% !important;
        min-width: 10rem !important;
        max-width: none !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
        padding: 0.75rem 1.5rem !important;
        box-sizing: border-box !important;
        font-size: 1rem !important;
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #666666 !important;
    }
    [data-testid="stMarkdown"]:has(#generate-request-buttons) ~ [data-testid="stHorizontalBlock"] .stButton button[kind="primary"],
    div:has(#generate-request-buttons) ~ div [data-testid="stHorizontalBlock"] .stButton button[kind="primary"] {
        background: #ffffff !important;
        color: #000000 !important;
        font-weight: 600 !important;
    }

    /* Custom plan intake: Submit & Cancel — side-by-side, enough space for buttons */
    [data-testid="stMarkdown"]:has(#intake-submit-cancel-row) ~ [data-testid="stHorizontalBlock"]:first-of-type {
        gap: 1.5rem !important;
        flex-wrap: nowrap !important;
    }
    [data-testid="stMarkdown"]:has(#intake-submit-cancel-row) ~ [data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"] {
        min-width: 160px !important;
        flex: 1 1 auto !important;
        flex-shrink: 0 !important;
        padding: 0.4rem !important;
    }
    [data-testid="stMarkdown"]:has(#intake-submit-cancel-row) ~ [data-testid="stHorizontalBlock"]:first-of-type .stButton button,
    [data-testid="stMarkdown"]:has(#intake-submit-cancel-row) ~ [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stFormSubmitButton"] button,
    [data-testid="stMarkdown"]:has(#intake-submit-cancel-row) ~ [data-testid="stHorizontalBlock"]:first-of-type button {
        min-width: 140px !important;
        width: 100% !important;
        white-space: nowrap !important;
        padding: 0.75rem 2rem !important;
        flex-shrink: 0 !important;
    }

    /* Admin mode days: selectable rectangles, no checkmark, clear checked vs unchecked */
    [id="admin-mode-days-section"] ~ * [data-testid="stCheckbox"],
    .block-container:has(#admin-mode-days-section) [data-testid="stCheckbox"] {
        display: inline-block;
        margin: 0 0.08rem;
        position: relative;
    }
    [id="admin-mode-days-section"] ~ * [data-testid="stCheckbox"] input,
    .block-container:has(#admin-mode-days-section) [data-testid="stCheckbox"] input {
        position: absolute !important;
        opacity: 0 !important;
        width: 100% !important;
        height: 100% !important;
        top: 0 !important;
        left: 0 !important;
        margin: 0 !important;
        cursor: pointer !important;
    }
    [id="admin-mode-days-section"] ~ * [data-testid="stCheckbox"] label,
    .block-container:has(#admin-mode-days-section) [data-testid="stCheckbox"] label {
        display: inline-flex !important;
        align-items: center;
        justify-content: center;
        min-width: 2rem;
        padding: 0.2rem 0.35rem;
        position: relative !important;
        background: #e2b8bd !important;
        color: #5a1720 !important;
        border: 1px solid #d9a5ab;
        border-radius: 5px;
        cursor: pointer;
        margin: 0 !important;
        white-space: nowrap !important;
        font-size: 0.85rem !important;
    }
    /* Sat/Sun (columns 6–7 in content row): darker red when unchecked */
    .block-container:has(#admin-mode-days-section) [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(6) [data-testid="stCheckbox"] label:not(:has(input:checked)),
    .block-container:has(#admin-mode-days-section) [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(7) [data-testid="stCheckbox"] label:not(:has(input:checked)) {
        background: #d08e95 !important;
        border-color: #c87d85 !important;
    }
    [id="admin-mode-days-section"] ~ * [data-testid="stCheckbox"] label:has(input:checked),
    .block-container:has(#admin-mode-days-section) [data-testid="stCheckbox"] label:has(input:checked) {
        background: #28a745 !important;
        color: #ffffff !important;
        border-color: #1e7e34 !important;
    }
    [id="admin-mode-days-section"] ~ * [data-testid="stHorizontalBlock"],
    .block-container:has(#admin-mode-days-section) [data-testid="stHorizontalBlock"] {
        gap: 0.25rem !important;
    }

    /* Admin mode days: mobile — days side-by-side in one row, length below */
    @media (max-width: 768px) {
        .block-container:has(#admin-mode-days-section) [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            flex-direction: row !important;
        }
        /* Days (first 7 columns): horizontal row, spread across screen */
        .block-container:has(#admin-mode-days-section) [data-testid="stHorizontalBlock"] > [data-testid="column"]:not(:last-child) {
            flex: 1 1 0 !important;
            min-width: 2rem !important;
            max-width: none !important;
        }
        /* Length (last column): full width below days */
        .block-container:has(#admin-mode-days-section) [data-testid="stHorizontalBlock"] > [data-testid="column"]:last-child {
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
        .block-container:has(#admin-mode-days-section) [data-testid="stCheckbox"] label {
            min-width: 1.5rem !important;
            padding: 0.15rem 0.2rem !important;
            font-size: 0.7rem !important;
        }
    }

    /* Admin edit: Save plan / Delete plan buttons — normal size, bordered box, Chrome-friendly */
    [data-testid="stMarkdown"]:has(#admin-edit-plan-actions) ~ [data-testid="stHorizontalBlock"]:first-of-type,
    div:has(#admin-edit-plan-actions) ~ div [data-testid="stHorizontalBlock"]:first-of-type {
        display: flex !important;
        flex-wrap: nowrap !important;
        gap: 1rem !important;
        align-items: stretch !important;
        width: 100% !important;
        min-width: 0 !important;
    }
    [data-testid="stMarkdown"]:has(#admin-edit-plan-actions) ~ [data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"],
    div:has(#admin-edit-plan-actions) ~ div [data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"],
    [data-testid="stMarkdown"]:has(#admin-edit-plan-actions) ~ [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stHorizontalBlock"] [data-testid="column"],
    div:has(#admin-edit-plan-actions) ~ div [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stHorizontalBlock"] [data-testid="column"] {
        border: 1px solid #555555 !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        min-width: 140px !important;
        flex: 1 1 0 !important;
        max-width: 50% !important;
    }
    [data-testid="stMarkdown"]:has(#admin-edit-plan-actions) ~ [data-testid="stHorizontalBlock"]:first-of-type .stButton,
    div:has(#admin-edit-plan-actions) ~ div [data-testid="stHorizontalBlock"]:first-of-type .stButton {
        width: 100% !important;
    }
    [data-testid="stMarkdown"]:has(#admin-edit-plan-actions) ~ [data-testid="stHorizontalBlock"]:first-of-type .stButton button,
    div:has(#admin-edit-plan-actions) ~ div [data-testid="stHorizontalBlock"]:first-of-type .stButton button {
        min-height: 48px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.05rem !important;
    }
    [data-testid="stMarkdown"]:has(#admin-edit-plan-actions) ~ [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stHorizontalBlock"] {
        gap: 0.75rem !important;
        margin-top: 0.5rem !important;
    }
    @media (max-width: 640px) {
        [data-testid="stMarkdown"]:has(#admin-edit-plan-actions) ~ [data-testid="stHorizontalBlock"]:first-of-type {
            flex-direction: column !important;
            gap: 0.75rem !important;
        }
        [data-testid="stMarkdown"]:has(#admin-edit-plan-actions) ~ [data-testid="stHorizontalBlock"]:first-of-type > [data-testid="column"] {
            flex: 1 1 100% !important;
            width: 100% !important;
        }
        [data-testid="stMarkdown"]:has(#admin-edit-plan-actions) ~ [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
        }
        [data-testid="stMarkdown"]:has(#admin-edit-plan-actions) ~ [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stHorizontalBlock"] [data-testid="column"] {
            flex: 1 1 100% !important;
            width: 100% !important;
        }
        [data-testid="stMarkdown"]:has(#admin-edit-plan-actions) ~ [data-testid="stHorizontalBlock"]:first-of-type .stButton button {
            width: 100% !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Bender branding: logo + tagline (black/white)
_logo_dir = os.path.join(BASE_DIR, "assets")
_full_logo_path = os.path.join(_logo_dir, "bender_full_logo.png")
_b_logo_path = os.path.join(_logo_dir, "b_logo.png")
if os.path.isfile(_full_logo_path):
    st.image(_full_logo_path, use_container_width=True)
else:
    st.markdown(
        '<div style="text-align:center; margin-bottom:0.5rem;">'
        '<p class="bender-tagline" style="font-size:2.25rem; font-weight:700; letter-spacing:0.08em; margin-bottom:0;">BENDER</p>'
        '<p class="bender-brand-sub" style="font-size:1.15rem;">• HOCKEY TRAINING •</p>'
        '</div>',
        unsafe_allow_html=True,
    )

# Session state init
if "last_session_id" not in st.session_state:
    st.session_state.last_session_id = None
if "last_output_text" not in st.session_state:
    st.session_state.last_output_text = None
if "last_output_metadata" not in st.session_state:
    st.session_state.last_output_metadata = None
if "last_inputs_fingerprint" not in st.session_state:
    st.session_state.last_inputs_fingerprint = None
if "scroll_to_workout" not in st.session_state:
    st.session_state.scroll_to_workout = False
if "current_user_id" not in st.session_state:
    st.session_state.current_user_id = None
if "current_profile" not in st.session_state:
    st.session_state.current_profile = None
if "page" not in st.session_state:
    st.session_state.page = "main"
if "auth_page" not in st.session_state:
    st.session_state.auth_page = "login"  # "login" or "create_account" when not logged in
if "admin_plan" not in st.session_state:
    st.session_state.admin_plan = None
if "admin_pending_integration" not in st.session_state:
    st.session_state.admin_pending_integration = None

# Restore login from URL (e.g. after page refresh) — keep user logged in unless they sign out
if st.session_state.current_user_id is None:
    _uid = st.query_params.get("uid")
    if _uid:
        _prof = load_profile(_uid)
        if _prof:
            st.session_state.current_user_id = _uid
            st.session_state.current_profile = _prof
            if not _equipment_setup_done(_prof):
                st.session_state.page = "equipment_onboarding"
            else:
                st.session_state.page = "main"
            st.rerun()

# ---------- Not logged in: Log in (first) or Create account (separate page) ----------
if st.session_state.current_user_id is None:
    # ----- Log in page (default) -----
    if st.session_state.auth_page == "login":
        st.markdown("#### Log in")
        if st.session_state.get("creation_success"):
            st.success(st.session_state.creation_success)
            del st.session_state.creation_success
        login_username = st.text_input(
            "Username (first and last name)",
            key="login_username",
            placeholder="e.g. John Smith",
        )
        login_password = st.text_input("Password", key="login_password", type="password")
        if st.button("Log in", key="btn_login"):
            login_username = (login_username or "").strip()
            if not login_username:
                st.error("Please enter your username.")
            elif not login_password:
                st.error("Please enter your password.")
            else:
                uid = _sanitize_user_id(login_username)
                prof = load_profile(uid)
                if prof is None:
                    st.error("No account with that username. Create an account or check spelling.")
                elif not prof.get("password_hash"):
                    st.error("This account was created before we added passwords. Please create a new account with your name and choose a password.")
                elif not _verify_password(
                    login_password,
                    prof.get("password_salt") or "",
                    prof.get("password_hash") or "",
                ):
                    st.error("Incorrect password.")
                else:
                    st.session_state.current_user_id = uid
                    st.session_state.current_profile = prof
                    st.query_params["uid"] = uid  # persist in URL so refresh keeps user logged in
                    if not _equipment_setup_done(prof):
                        st.session_state.page = "equipment_onboarding"
                    else:
                        st.session_state.page = "main"
                    st.rerun()
        st.caption("Don’t have an account?")
        if st.button("Create an account", key="goto_create"):
            st.session_state.auth_page = "create_account"
            st.rerun()
        st.stop()

    # ----- Create account page (separate) -----
    st.markdown("#### Create an account")
    st.caption("You’ll choose your equipment below.")
    create_username = st.text_input(
        "Username (your first and last name)",
        key="create_username",
        placeholder="e.g. John Smith",
        autocomplete="name",
    )
    create_age = st.number_input("Age", min_value=6, max_value=99, value=16, step=1, key="create_age")
    create_position = st.selectbox("Position", options=POSITION_OPTIONS, key="create_position")
    create_level = st.selectbox("Current Level", options=CURRENT_LEVEL_OPTIONS, key="create_level")
    _ch_h, _ch_w = st.columns(2)
    with _ch_h:
        create_height = st.text_input("Height", placeholder="e.g. 5'10\" or 180 cm", key="create_height")
    with _ch_w:
        create_weight = st.text_input("Weight", placeholder="e.g. 175 lbs or 79 kg", key="create_weight")
    create_password = st.text_input("Password", key="create_password", type="password")
    create_confirm = st.text_input("Confirm password", key="create_confirm", type="password")
    st.markdown("**Equipment**")
    st.caption("Choose what you have. You can change this anytime in the sidebar.")
    try:
        _create_equip_by_mode = ENGINE.get_canonical_equipment_by_mode()
    except Exception:
        _create_equip_by_mode = {"Performance": ["None"], "Puck Mastery": [], "Conditioning": ["None"], "Skating Mechanics": ["None"], "Mobility": ["None"]}
    _render_equipment_dropdowns(_create_equip_by_mode, set(), "create_equip")
    if st.button("Create account", key="btn_create"):
        create_username = (create_username or "").strip()
        if not create_username:
            st.error("Please enter your first and last name.")
        elif not create_password:
            st.error("Please enter a password.")
        elif create_password != create_confirm:
            st.error("Passwords do not match.")
        else:
            uid = _sanitize_user_id(create_username)
            if not uid or uid == "default":
                st.error("Please enter a valid first and last name.")
            elif _user_id_taken(uid):
                st.error("That username is already taken. Please choose another or log in.")
            else:
                salt_hex, hash_hex = _hash_password(create_password)
                _create_equipment = _collect_equipment_from_session(_create_equip_by_mode, "create_equip")
                profile = {
                    "user_id": uid,
                    "display_name": create_username,
                    "age": int(create_age),
                    "position": create_position,
                    "current_level": create_level,
                    "height": (create_height or "").strip(),
                    "weight": (create_weight or "").strip(),
                    "equipment": _create_equipment,
                    "equipment_setup_done": True,
                    "password_salt": salt_hex,
                    "password_hash": hash_hex,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                }
                save_profile(profile)
                st.session_state.creation_success = "Account created. Please log in."
                st.session_state.auth_page = "login"
                st.rerun()
    st.markdown("---")
    st.caption("Already have an account?")
    if st.button("Back to log in", key="back_to_login"):
        st.session_state.auth_page = "login"
        st.rerun()
    st.stop()

# ---------- Required: Equipment onboarding (before first workout) ----------
if st.session_state.page == "equipment_onboarding":
    prof = st.session_state.current_profile or {}
    st.markdown("#### Set up your equipment")
    st.caption("Choose what you have so we can build the right workouts. You can change this anytime in the sidebar.")
    try:
        equipment_by_mode = ENGINE.get_canonical_equipment_by_mode()
    except Exception:
        equipment_by_mode = {"Performance": ["None"], "Puck Mastery": [], "Conditioning": ["None"], "Skating Mechanics": ["None"], "Mobility": ["None"]}
    _canonicalize = getattr(ENGINE, "canonicalize_equipment_list", None)
    current_equip = set(_canonicalize(prof.get("equipment") or []) if _canonicalize else (prof.get("equipment") or []))
    _render_equipment_dropdowns(equipment_by_mode, current_equip, "onb")
    if st.button("Save and continue", key="onb_save"):
        selected = _collect_equipment_from_session(equipment_by_mode, "onb")
        if not selected:
            st.warning("Select at least one option so we can build workouts.")
        else:
            prof["equipment"] = selected
            prof["equipment_setup_done"] = True
            st.session_state.current_profile = prof
            save_profile(prof)
            st.session_state.page = "main"
            st.success("Saved. Taking you to Bender.")
            st.rerun()
    st.stop()

# ---------- Main app: sidebar for equipment + main area ----------
# Require equipment setup before generating (e.g. legacy profile)
if not _equipment_setup_done(st.session_state.current_profile or {}):
    st.session_state.page = "equipment_onboarding"
    st.rerun()

display_name = (st.session_state.current_profile or {}).get("display_name") or st.session_state.current_user_id or ""

# Sidebar: Bender branding + equipment + Sign out
with st.sidebar:
    if os.path.isfile(_b_logo_path):
        st.image(_b_logo_path, width=44)
    else:
        st.markdown('<p style="font-size:2rem; font-weight:700; letter-spacing:0.05em; color:#ffffff; margin-bottom:0;">B</p>', unsafe_allow_html=True)
    st.markdown(f"**{display_name}**")
    with st.expander("Account Settings", expanded=False):
        st.caption("Position, level, height & weight")
        _prof = st.session_state.current_profile or {}
        _pos_val = _prof.get("position") or "Forward"
        _pos_idx = POSITION_OPTIONS.index(_pos_val) if _pos_val in POSITION_OPTIONS else 0
        st.selectbox("Position", options=POSITION_OPTIONS, index=_pos_idx, key="sidebar_position")
        _lvl_val = _prof.get("current_level") or "Youth"
        _lvl_idx = CURRENT_LEVEL_OPTIONS.index(_lvl_val) if _lvl_val in CURRENT_LEVEL_OPTIONS else 0
        st.selectbox("Current Level", options=CURRENT_LEVEL_OPTIONS, index=_lvl_idx, key="sidebar_level")
        _row_hw = st.columns(2)
        with _row_hw[0]:
            _h = st.text_input("Height", value=_prof.get("height") or "", placeholder="e.g. 5'10\"", key="sidebar_height")
        with _row_hw[1]:
            _w = st.text_input("Weight", value=_prof.get("weight") or "", placeholder="e.g. 175 lbs", key="sidebar_weight")
    _equip_just_saved = st.session_state.pop("equipment_expander_collapse_after_save", False)
    _equip_label = "Equipment" + ("\u200b" if _equip_just_saved else "")  # Change identity when just saved so expander resets to collapsed
    with st.expander(_equip_label, expanded=False):
        try:
            equipment_by_mode = ENGINE.get_canonical_equipment_by_mode()
        except Exception:
            equipment_by_mode = {"Performance": ["None"], "Puck Mastery": [], "Conditioning": ["None"], "Skating Mechanics": ["None"], "Mobility": ["None"]}
        prof = st.session_state.current_profile or {}
        _canonicalize = getattr(ENGINE, "canonicalize_equipment_list", None)
        current_equip = set(_canonicalize(prof.get("equipment") or []) if _canonicalize else (prof.get("equipment") or []))
        _render_equipment_dropdowns(equipment_by_mode, current_equip, "sidebar")
        if st.button("Save equipment", key="sidebar_save"):
            new_equip = _collect_equipment_from_session(equipment_by_mode, "sidebar")
            prof["equipment"] = new_equip
            prof["position"] = st.session_state.get("sidebar_position", prof.get("position") or "Forward")
            prof["current_level"] = st.session_state.get("sidebar_level", prof.get("current_level") or "Youth")
            prof["height"] = (st.session_state.get("sidebar_height") or "").strip()
            prof["weight"] = (st.session_state.get("sidebar_weight") or "").strip()
            st.session_state.current_profile = prof
            save_profile(prof)
            st.session_state.collapse_sidebar_after_save = True
            st.session_state.equipment_expander_collapse_after_save = True
            st.session_state.page = "main"
            st.success("Saved")
            st.rerun()
    st.divider()
    if st.button("Sign out", key="sidebar_logout"):
        st.session_state.current_user_id = None
        st.session_state.current_profile = None
        st.session_state.page = "main"
        if "uid" in st.query_params:
            del st.query_params["uid"]
        st.rerun()  # Shows landing (Log in page)
    _sidebar_athlete = (st.session_state.current_profile or {}).get("display_name") or (st.session_state.current_profile or {}).get("user_id") or ""
    _sidebar_meta = st.session_state.get("last_output_metadata") or {}
    _sidebar_prefill = build_prefilled_feedback_url(
        athlete=_sidebar_athlete.strip(),
        mode_label={"performance": "Performance", "energy_systems": "Conditioning", "skating_mechanics": "Skating Mechanics", "shooting": "Puck Mastery (Shooting)", "stickhandling": "Puck Mastery (Stickhandling)", "skills_only": "Puck Mastery (Both)", "mobility": "Mobility"}.get(_sidebar_meta.get("mode", ""), ""),
        location_label="Gym" if _sidebar_meta.get("location") == "gym" else "No Gym",
        emphasis_key="",
        rating=4,
        notes="",
    )
    st.link_button("Leave Feedback (auto-filled)", _sidebar_prefill)
    st.link_button("Open Feedback Form (blank)", FORM_BASE)

# ---------- Main area: form in card ----------
# Signed-in line (Sign out is only in the sidebar Equipment section)
st.caption(f"Signed in as **{display_name}**")

# Athlete = logged-in user (for history, download filename, feedback)
athlete_id = (st.session_state.current_profile or {}).get("display_name") or (st.session_state.current_profile or {}).get("user_id") or ""
athlete_id = athlete_id.strip() or "athlete"

# Admin: Plan Builder tab (only for admin users)
try:
    from admin_plan_builder import (
        is_admin_user,
        generate_plan,
        generate_plan_with_workouts,
        compute_plan_highlights,
        PLAN_MODES,
        MODE_DISPLAY_LABELS,
        MODE_SESSION_LEN_DEFAULTS,
        FREQUENCY_OPTIONS,
        WEEKDAY_NAMES,
    )
except (ImportError, KeyError, Exception):
    is_admin_user = lambda _: False
    generate_plan = lambda *a, **k: []
    generate_plan_with_workouts = lambda p, cb, seed=42: p
    PLAN_MODES = ["performance", "skating_mechanics", "shooting", "stickhandling", "energy_systems", "mobility"]
    MODE_DISPLAY_LABELS = {
        "performance": "Performance",
        "skating_mechanics": "Skating Mechanics",
        "shooting": "Shooting",
        "stickhandling": "Stickhandling",
        "energy_systems": "Conditioning",
        "mobility": "Mobility/Recovery",
    }
    MODE_SESSION_LEN_DEFAULTS = {"performance": 60, "skating_mechanics": 12, "shooting": 25, "stickhandling": 25, "energy_systems": 15, "mobility": 12}
    FREQUENCY_OPTIONS = ["As in plan", "1x/week", "2x/week", "3x/week", "4x/week", "5x/week", "6x/week", "7x/week"]
    WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

_admin = is_admin_user(display_name)
_assigned_plan = (st.session_state.current_profile or {}).get("assigned_plan")
# Only show My Plan tab if player has a real plan with content
_weeks = []
if _assigned_plan:
    _weeks = (_assigned_plan.get("plan", _assigned_plan) if isinstance(_assigned_plan, dict) else _assigned_plan) or []
_has_valid_plan = bool(_weeks and len(_weeks) > 0)
if _admin:
    _custom_req_count = len([r for r in load_custom_plan_requests() if not r.get("completed")])
    _custom_req_tab_label = f"Admin: Custom Plan Request ({_custom_req_count})" if _custom_req_count > 0 else "Admin: Custom Plan Request"
    _admin_tab_names = ["Workout Generator", "Admin: Plan Builder", "Admin: Highscores", "Your Work", _custom_req_tab_label]
    _admin_default_idx = 1 if st.session_state.get("admin_pending_integration") else 0
    if "admin_tab_idx" not in st.session_state or st.session_state.get("admin_pending_integration"):
        st.session_state.admin_tab_idx = _admin_default_idx
    _tab_plan = None
elif _has_valid_plan:
    _tab_admin = None
    _tab_custom_requests = None
    _tab_highscores = None
    _tab_plan = None
    _tab_silent_work = None
else:
    _tab_admin = None
    _tab_custom_requests = None
    _tab_plan = None
    _tab_highscores = None
    _tab_silent_work = None

# Age from profile (set at account creation)
age = int((st.session_state.current_profile or {}).get("age") or 16)
age = max(6, min(99, age))


def _render_training_session():
    # Custom Plan Intake questionnaire (shown when Request Custom Plan clicked)
    if st.session_state.get("custom_plan_intake_open"):
        st.markdown("### BENDER PLAN INTAKE")
        profile = st.session_state.get("current_profile") or {}
        with st.form("custom_plan_intake_form"):
            q1 = st.slider(
                "1. How many weeks do you want your plan to run?",
                4, 12, 6,
                key="intake_weeks",
            )
            q2 = st.slider(
                "2. How many days per week can you train?",
                3, 7, 4,
                key="intake_days",
            )
            q3 = st.radio(
                "3. What is your primary goal for this phase?",
                [
                    "Get stronger",
                    "Increase speed / acceleration",
                    "Improve conditioning",
                    "Add lean muscle",
                    "Reduce injury risk",
                    "In-season maintenance",
                    "Other",
                ],
                key="intake_goal",
            )
            q3_other = st.text_input(
                "If Other, describe your goal:",
                placeholder="e.g. Peak for playoffs, rehab from injury...",
                key="intake_goal_other",
            )
            q4 = st.radio(
                "4. How long have you been lifting seriously?",
                ["New (0–6 months)", "1–2 years", "3+ years"],
                key="intake_experience",
            )
            q5 = st.radio(
                "5. How long can each session realistically be?",
                ["30", "45", "60", "75+ minutes"],
                key="intake_session_len",
            )
            q6 = st.slider(
                "6. On a scale of 1–10, how locked in are you for this phase?",
                1, 10, 7,
                key="intake_commitment",
            )
            st.markdown('<div id="intake-submit-cancel-row" aria-hidden="true"></div>', unsafe_allow_html=True)
            col_submit, _intake_gap, col_cancel = st.columns([1, 0.2, 1])
            with col_submit:
                submitted = st.form_submit_button("Submit")
            with col_cancel:
                cancelled = st.form_submit_button("Cancel")
        if cancelled:
            st.session_state.custom_plan_intake_open = False
            st.rerun()
        if submitted:
            save_custom_plan_request({
                "user_id": profile.get("user_id"),
                "display_name": profile.get("display_name") or profile.get("user_id") or "Unknown",
                "weeks": int(q1),
                "days_per_week": int(q2),
                "primary_goal": (q3_other.strip() if (q3 == "Other" and q3_other) else q3),
                "lifting_experience": q4,
                "session_length": q5,
                "commitment_1_10": int(q6),
            })
            st.session_state.custom_plan_intake_open = False
            st.success("Your custom plan request has been submitted. An admin will review it.")
            st.rerun()
        st.stop()

    @st.fragment
    def _training_session_fragment():
        form_container = st.container()
        with form_container:
            st.markdown('<div class="form-card-marker"></div>', unsafe_allow_html=True)
            st.markdown("#### Session options")
            minutes = st.slider("Session length (minutes)", 10, 120, 45, step=5)
            minutes = int(minutes)
    
            mode_label = st.selectbox("Mode", DISPLAY_MODES)
            mode = LABEL_TO_MODE[mode_label]
    
            if mode == "puck_mastery":
                skills_sub = st.selectbox("Puck Mastery — focus", SKILLS_SUB_LABELS, index=2)
                effective_mode = SKILLS_SUB_TO_MODE[skills_sub]
            else:
                effective_mode = mode
    
            if effective_mode == "performance":
                location = st.selectbox("Location", ["gym", "no_gym"], help="Choose 'gym' for strength day, skate-within-24h, and post-lift conditioning options.")
            else:
                location = "no_gym"
    
            focus = None
            strength_day_type = None
            strength_emphasis = "strength"
            skate_within_24h = False
            conditioning_focus = None
            conditioning_mode = None
            conditioning_effort = None
    
            if effective_mode == "energy_systems":
                prof_equip = (st.session_state.current_profile or {}).get("equipment") or []
                _canonicalize = getattr(ENGINE, "canonicalize_equipment_list", None)
                prof_equip_canonical = _canonicalize(prof_equip) if _canonicalize else prof_equip
                cond_modes = getattr(ENGINE, "get_conditioning_modes_for_equipment", lambda x: [("field", "Field/No equipment"), ("cones", "Cones"), ("hill", "Hill"), ("bike", "Stationary Bike"), ("treadmill", "Treadmill"), ("surprise", "Surprise me")])(prof_equip_canonical)
                mode_options = [label for _, label in cond_modes]
                mode_values = [v for v, _ in cond_modes]
                mode_idx = st.selectbox("Conditioning Mode", range(len(mode_options)), format_func=lambda i: mode_options[i], key="cond_mode")
                conditioning_mode = mode_values[mode_idx]
                effort_options = ["Easy", "Hard", "Surprise me"]
                effort_idx = st.selectbox("Effort", range(len(effort_options)), format_func=lambda i: effort_options[i], key="cond_effort")
                conditioning_effort = ["easy", "hard", "surprise"][effort_idx]
                if minutes > 25:
                    st.caption("Conditioning capped at 25 min for quality.")
    
            elif effective_mode == "performance":
                if location == "gym":
                    # Lower → heavy_leg, Upper → upper_core_stability, Power → heavy_explosive
                    STRENGTH_DAY_OPTIONS = ["Lower", "Upper", "Power"]
                    STRENGTH_DAY_TO_TYPE = {"Lower": "heavy_leg", "Upper": "upper_core_stability", "Power": "heavy_explosive"}
                    day_label = st.selectbox("Strength day", STRENGTH_DAY_OPTIONS)
                    strength_day_type = STRENGTH_DAY_TO_TYPE[day_label]
                    em_label = st.selectbox("Strength emphasis", EMPHASIS_DISPLAY, index=EMPHASIS_KEYS.index("strength"))
                    strength_emphasis = EMPHASIS_LABEL_TO_KEY[em_label]
                else:
                    st.caption("No-gym: you'll get a premade circuit + mobility. For strength day and post-lift conditioning, set Location to **gym**.")
                    strength_day_type = "heavy_explosive"
                    strength_emphasis = "strength"
                    skate_within_24h = False
    
            elif effective_mode == "mobility":
                focus = "mobility"
    
            available_space = None  # Assume user has necessary space
    
            conditioning = False
            conditioning_type = None
            if effective_mode == "performance" and location == "gym":
                conditioning = st.checkbox("Post-lift conditioning?", value=False)
                if conditioning:
                    conditioning_type = st.selectbox("Post-lift type (gym)", ["bike", "treadmill", "surprise"])
                else:
                    conditioning_type = None
    
            # Auto-clear old output if key inputs change
            inputs_fingerprint = (
                athlete_id.strip().lower(),
                int(age),
                int(minutes),
                effective_mode,
                location,
                focus,
                strength_day_type,
                strength_emphasis,
                skate_within_24h,
                conditioning,
                conditioning_type,
                conditioning_mode,
                conditioning_effort,
            )
    
            if st.session_state.last_inputs_fingerprint is None:
                st.session_state.last_inputs_fingerprint = inputs_fingerprint
            else:
                if inputs_fingerprint != st.session_state.last_inputs_fingerprint:
                    if st.session_state.last_session_id or st.session_state.last_output_text:
                        clear_last_output()
                    st.session_state.last_inputs_fingerprint = inputs_fingerprint
    
            # Generate action + Request Custom Plan — side by side
            st.markdown('<div id="generate-request-buttons" aria-hidden="true"></div>', unsafe_allow_html=True)
            col_gen, col_req = st.columns(2)
            with col_gen:
                generate_clicked = st.button("Generate session", type="primary", use_container_width=True)
            with col_req:
                request_plan_clicked = st.button("Request Custom Plan", type="secondary", use_container_width=True)
            if request_plan_clicked:
                st.session_state.custom_plan_intake_open = True
                st.rerun()
            if generate_clicked:
                profile = st.session_state.get("current_profile") or {}
                user_equipment = ENGINE.expand_user_equipment(profile.get("equipment"))
                payload = {
                    "athlete_id": athlete_id,
                    "age": int(age),
                    "minutes": int(minutes),
                    "mode": effective_mode,
                    "focus": focus,
                    "location": location,
                    "strength_day_type": strength_day_type,
                    "strength_emphasis": strength_emphasis,
                    "skate_within_24h": skate_within_24h,
                    "conditioning": conditioning,
                    "conditioning_type": conditioning_type,
                    "user_equipment": user_equipment,
                    "available_space": available_space if effective_mode in ("stickhandling", "skills_only") else None,
                    "conditioning_mode": conditioning_mode if effective_mode == "energy_systems" else None,
                    "conditioning_effort": conditioning_effort if effective_mode == "energy_systems" else None,
                }
    
                try:
                    with st.spinner("Generating workout..."):
                        if USE_API:
                            resp = _generate_via_api(payload)
                        else:
                            resp = _generate_via_engine(payload)
    
                    st.session_state.last_session_id = resp.get("session_id")
                    out_text = resp.get("output_text")
                    if out_text and out_text.strip():
                        st.session_state.last_output_text = out_text
                        st.session_state.last_output_metadata = {
                            "mode": effective_mode,
                            "minutes": int(minutes),
                            "location": location,
                            "conditioning": conditioning,
                            "conditioning_type": conditioning_type,
                        }
                        st.session_state.scroll_to_workout = True
                        st.success("Generated")
                    else:
                        st.session_state.last_output_text = (
                            "BENDER SINGLE WORKOUT | mode=performance | len=45 min\n\n"
                            "Generation returned no content. This can happen if:\n"
                            "- Data files (performance.json) are missing or empty\n"
                            "- All drills were filtered out by equipment/age\n\n"
                            "Try: Clear equipment in sidebar (use full gym), or check that data/performance.json exists."
                        )
                        st.warning("Generated but no exercises were returned — see message below.")
                        st.error("Missing equipment: Workout could not be generated. Please update your Equipment settings in the sidebar for best functionality.")
                except Exception as e:
                    st.error(str(e))

        # Display last generated workout (Tabbed) — isolated so it doesn't affect My Plan / Your Work tabs
        if st.session_state.last_output_text:
            st.divider()
            st.markdown('<div id="workout-result-section" class="workout-display-wrapper"></div>', unsafe_allow_html=True)
            st.markdown('<div id="workout-result"></div>', unsafe_allow_html=True)
            if st.session_state.get("scroll_to_workout"):
                st.session_state.scroll_to_workout = False
                st.components.v1.html(
                    "<script>var el = (window.parent && window.parent.document) ? window.parent.document.getElementById('workout-result') : document.getElementById('workout-result'); if (el) el.scrollIntoView({behavior: 'smooth'});</script>",
                    height=0,
                )
            _clear_col, _spacer = st.columns([1, 4])
            with _clear_col:
                if st.button("Clear workout", key="clear_workout_top"):
                    clear_last_output()
                    st.rerun()
            st.markdown('<div id="workout-tabs-clear-row" aria-hidden="true"></div>', unsafe_allow_html=True)
            tab_workout, tab_download = st.tabs(["Workout", "Download / Copy"])

            with tab_workout:
                if effective_mode == "performance" and location == "no_gym":
                    render_no_gym_strength_circuits_only(st.session_state.last_output_text)
                else:
                    render_workout_readable(st.session_state.last_output_text)
                st.divider()
                st.caption("Finished? Log your completion to Your Work.")
                _meta = st.session_state.get("last_output_metadata") or _parse_workout_header_for_metadata(st.session_state.last_output_text or "")
                if st.button("Workout Complete", type="primary", key="workout_complete_bender"):
                    prof = st.session_state.get("current_profile") or {}
                    if prof and _meta:
                        prof = _add_completion_to_profile(prof, _meta)
                        st.session_state.current_profile = prof
                        save_profile(prof)
                    clear_last_output()
                    st.success("Workout logged to Your Work!")
                    st.rerun()
                if st.button("Clear workout", type="secondary", key="clear_workout_bottom"):
                    clear_last_output()
                    st.rerun()

            with tab_download:
                safe_name = re.sub(r"[^\w\-]", "_", athlete_id.strip())[:30] or "workout"
                date_str = datetime.now().strftime("%Y-%m-%d")
                download_filename = f"bender_workout_{safe_name}_{date_str}.txt"
                st.download_button(
                    label="Download workout (.txt)",
                    data=st.session_state.last_output_text or "",
                    file_name=download_filename,
                    mime="text/plain",
                )
                st.caption("Your browser will download the file when you click above.")
                st.write("")
                if not (effective_mode == "performance" and location == "no_gym"):
                    with st.expander("Copy workout (raw text)"):
                        st.code(st.session_state.last_output_text)
                        st.caption("Select the text above and copy (Ctrl+C / Cmd+C).")

    _training_session_fragment()


# Admin: button-based tabs (independent rendering). Player: same style, different tabs.
if _admin:
    st.markdown('<div id="admin-tab-bar" data-tab-style="classic" aria-hidden="true"></div>', unsafe_allow_html=True)
    _admin_cols = st.columns(len(_admin_tab_names))
    for _ai, _aname in enumerate(_admin_tab_names):
        with _admin_cols[_ai]:
            _a_selected = st.session_state.admin_tab_idx == _ai
            if st.button(_aname, key=f"admin_tab_{_ai}", type="primary" if _a_selected else "secondary"):
                st.session_state.admin_tab_idx = _ai
                st.rerun()
    if st.session_state.admin_tab_idx == 0:
        _render_training_session()
else:
    # Player: button-based tabs (no radio circles); only render selected tab's content
    if "player_tab" not in st.session_state:
        st.session_state.player_tab = "Training Session"
    _tab_opts = ["Training Session", "My Plan", "Your Work"] if _has_valid_plan else ["Training Session", "Your Work"]
    with st.container():
        st.markdown('<div id="player-tab-bar" data-tab-style="classic" aria-hidden="true"></div>', unsafe_allow_html=True)
        _tab_cols = st.columns(len(_tab_opts))
    _sel = st.session_state.player_tab
    for _i, _opt in enumerate(_tab_opts):
        with _tab_cols[_i]:
            _is_selected = _sel == _opt
            if st.button(_opt, key=f"player_tab_{_opt.replace(' ', '_')}", type="primary" if _is_selected else "secondary"):
                st.session_state.player_tab = _opt
                st.rerun()
    _sel = st.session_state.player_tab

    if _sel == "Training Session":
        _render_training_session()
    elif _sel == "My Plan" and _has_valid_plan and _assigned_plan:
        _plan_data, _plan_name = _deserialize_plan_for_display(_assigned_plan)
        _plan_completed = (st.session_state.current_profile or {}).get("assigned_plan_completed") or {}
        if isinstance(_plan_completed, dict):
            _plan_completed = {str(k): (v if isinstance(v, list) else list(v)) for k, v in _plan_completed.items()}

        def _plan_on_complete(day_idx: int, mode_key: str, params_or_meta: dict | None = None) -> None:
            prof = dict(st.session_state.current_profile or {})
            c = dict(prof.get("assigned_plan_completed") or {})
            key = str(day_idx)
            c[key] = list(set(c.get(key, [])) | {mode_key})
            prof["assigned_plan_completed"] = c
            if params_or_meta:
                prof = _add_completion_to_profile(prof, params_or_meta)
            st.session_state.current_profile = prof
            save_profile(prof)
            st.rerun()

        if _plan_name:
            st.markdown(f"### {_plan_name}")
        _render_plan_view(_plan_data, _plan_completed, st.session_state.current_profile or {}, _plan_on_complete)
    elif _sel == "Your Work":
        _render_your_work_stats()

# Admin tab: Plan Builder (only for Erich Jaeger)
if _admin and st.session_state.get("admin_tab_idx") == 1:
        # Process pending integration from Custom Plan Request tab (before any widgets that use these keys)
        if st.session_state.get("admin_pending_integration"):
            _req = st.session_state.admin_pending_integration
            st.session_state.admin_pending_integration = None
            _apply_custom_request_to_plan_builder(_req)
        # Edit mode: player view only, Back + Save. Hide Build plan, Generate, Assign, etc.
        _edit_target = st.session_state.get("admin_plan_edit_target")
        if _edit_target and st.session_state.get("admin_plan"):
            _plan = st.session_state.admin_plan
            _target_profile = next((p for p in list_profiles() if p.get("user_id") == _edit_target), None)
            if st.button("← Back", key="admin_edit_back"):
                st.session_state.admin_plan_edit_target = None
                if "admin_workout_edit_vals" in st.session_state:
                    del st.session_state.admin_workout_edit_vals
                st.rerun()
            total_days = sum(len(w["days"]) for w in _plan)
            flat_days_edit: list[tuple[int, dict]] = []
            for w in _plan:
                for d in w["days"]:
                    flat_days_edit.append((w["week"], d))
            if "admin_plan_selected_day" not in st.session_state:
                st.session_state.admin_plan_selected_day = 0
            if "admin_plan_completed" not in st.session_state:
                st.session_state.admin_plan_completed = {}
            sel_idx = st.session_state.admin_plan_selected_day
            wv = st.session_state.get("admin_plan_workout_view")
            if wv is not None:
                wv_day, wv_mode = wv
                if wv_day < len(flat_days_edit):
                    _, wv_day_data = flat_days_edit[wv_day]
                    fi_edit = next((x for x in wv_day_data.get("focus_items", []) if x["mode_key"] == wv_mode), None)
                    if fi_edit:
                        _edit_title = ({"leg": "Heavy legs", "upper": "Upper / Core stability", "full": "Explosive legs"}.get((fi_edit.get("params") or {}).get("strength_day_type", "full") or "full", fi_edit["label"]) if fi_edit.get("mode_key") == "performance" else fi_edit["label"])
                        st.markdown(f"### {_edit_title} — Day {wv_day + 1} (Edit)")
                        if st.button("← Back", key="admin_edit_workout_back"):
                            st.session_state.admin_plan_workout_view = None
                            st.rerun()
                        _workout_text = fi_edit.get("workout") or ""
                        _profile = _target_profile or {}
                        _age = max(6, min(99, int(_profile.get("age") or 16)))
                        _equip = getattr(ENGINE, "expand_user_equipment", lambda x: x or [])(_profile.get("equipment")) if ENGINE and _profile.get("equipment") else None
                        try:
                            _data = _load_engine_data()
                        except Exception:
                            _data = {}
                        _saved = render_workout_editable(_workout_text, fi_edit.get("params", {}), _data, _age, _equip, f"admin_edit_{wv_day}_{wv_mode}")
                        if _saved is not None:
                            fi_edit["workout"] = _saved
                            st.session_state.admin_plan_workout_view = None
                            st.success("Workout saved.")
                            st.rerun()
                        st.stop()
            st.markdown("---")
            st.markdown('<div id="admin-edit-day-grid" aria-hidden="true"></div>', unsafe_allow_html=True)
            st.markdown(f"**Select day** — Day {sel_idx + 1} of {total_days}")
            _row_cols = st.columns(total_days)
            _today_edit = date.today()
            for i in range(total_days):
                with _row_cols[i]:
                    _dd = flat_days_edit[i][1]
                    _ds = _dd["date"].strftime("%a") if hasattr(_dd["date"], "strftime") else str(_dd["date"])[:3]
                    _adm_comp = st.session_state.admin_plan_completed.get(i, set()) or set()
                    _focus_i = _dd.get("focus_items", [])
                    _day_done = len(_focus_i) > 0 and all(x["mode_key"] in _adm_comp for x in _focus_i)
                    _past_edit = _dd["date"] < _today_edit if hasattr(_dd["date"], "__lt__") else False
                    _missed_edit = _past_edit and not _day_done
                    if _day_done:
                        st.markdown('<div class="admin-day-complete" aria-hidden="true"></div>', unsafe_allow_html=True)
                    elif _missed_edit:
                        st.markdown('<div class="admin-day-missed-marker" aria-hidden="true"></div>', unsafe_allow_html=True)
                    _edit_label = f"{'✓ ' if _day_done else ''}{i + 1}"
                    if st.button(_edit_label, key=f"admin_edit_day_{i}", type="primary" if i == sel_idx else "secondary"):
                        st.session_state.admin_plan_selected_day = i
                        st.rerun()
                    _dc = "plan-day-date plan-day-date-selected" if i == sel_idx else "plan-day-date"
                    if _missed_edit:
                        st.markdown(f'<div class="plan-day-date-block"><p class="{_dc}">{_ds}</p><p class="plan-day-missed">Missed day</p></div>', unsafe_allow_html=True)
                    elif _day_done:
                        st.markdown(f'<div class="plan-day-date-block"><p class="{_dc}">{_ds}</p><p class="plan-day-complete-label">Day complete</p></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<p class="{_dc}">{_ds}</p>', unsafe_allow_html=True)
            _, _day_data = flat_days_edit[sel_idx]
            st.markdown(f"### Day {sel_idx + 1} of {total_days}")
            st.caption(_day_data["date"].strftime("%A, %b %d") if hasattr(_day_data["date"], "strftime") else str(_day_data["date"]))
            st.markdown("**Modes** — click to edit workout")
            for _fi in _day_data.get("focus_items", []):
                _done = _fi["mode_key"] in (st.session_state.admin_plan_completed.get(sel_idx, set()) or set())
                _lbl = f"{'✓ ' if _done else ''}{_fi['label']}"
                if st.button(_lbl, key=f"admin_edit_open_{sel_idx}_{_fi['mode_key']}", type="primary" if _done else "secondary"):
                    st.session_state.admin_plan_workout_view = (sel_idx, _fi["mode_key"])
                    st.rerun()
            st.divider()
            st.markdown('<div id="admin-edit-plan-actions" aria-hidden="true"></div>', unsafe_allow_html=True)
            _col_save, _col_del = st.columns(2)
            with _col_save:
                if st.button("Save plan", key="admin_edit_save_plan", type="primary", use_container_width=True):
                    _plan_to_save = _serialize_plan_for_storage(_plan, st.session_state.get("admin_plan_name", ""))
                    if _target_profile:
                        _target_profile["assigned_plan"] = _plan_to_save
                        _target_profile["assigned_plan_completed"] = _target_profile.get("assigned_plan_completed") or {}
                        save_profile(_target_profile)
                        st.success("Plan saved to player.")
                    st.session_state.admin_plan_edit_target = None
                    if "admin_plan_delete_confirm" in st.session_state:
                        del st.session_state.admin_plan_delete_confirm
                    st.rerun()
            with _col_del:
                if st.session_state.get("admin_plan_delete_confirm"):
                    st.warning("Are you sure you want to delete this player's plan? This cannot be undone.")
                    _cd1, _cd2 = st.columns(2)
                    with _cd1:
                        if st.button("Yes, delete plan", key="admin_edit_delete_confirm", use_container_width=True):
                            if _target_profile:
                                _target_profile["assigned_plan"] = None
                                _target_profile["assigned_plan_completed"] = {}
                                save_profile(_target_profile)
                            st.session_state.admin_plan = None
                            st.session_state.admin_plan_edit_target = None
                            st.session_state.admin_plan_completed = {}
                            del st.session_state.admin_plan_delete_confirm
                            st.success("Plan deleted.")
                            st.rerun()
                    with _cd2:
                        if st.button("Cancel", key="admin_edit_delete_cancel", use_container_width=True):
                            del st.session_state.admin_plan_delete_confirm
                            st.rerun()
                else:
                    if st.button("Delete plan", key="admin_edit_delete", use_container_width=True):
                        st.session_state.admin_plan_delete_confirm = True
                        st.rerun()
            st.stop()

        st.subheader("Admin: Plan Builder")
        st.caption("Multi-week workout plans. Generate with full workouts for each day (Bible App–style view).")
        if st.session_state.pop("admin_custom_request_integrated", False):
            st.info("Form pre-filled from a custom plan request. Review the settings below and click **Generate plan** when ready.")
        # Build plan for: dropdown to select target player (uses their equipment & age)
        _all_profiles_for_builder = list_profiles()
        _builder_options = [(None, "Default (admin / no equipment filter)")] + [(p, (p.get("display_name") or p.get("user_id") or "Unknown")) for p in _all_profiles_for_builder]
        _builder_idx = st.selectbox(
            "Build plan for",
            options=range(len(_builder_options)),
            format_func=lambda i: _builder_options[i][1],
            key="admin_plan_target",
            help="Workouts will be filtered by this player's equipment (e.g. no shooting if they don't have Shooting pad & net).",
        )
        if "admin_plan_name" not in st.session_state:
            st.session_state.admin_plan_name = ""
        _plan_name = st.text_input("Plan name", value=st.session_state.admin_plan_name, key="admin_plan_name_input", placeholder="e.g. 4-Week Pre-Season")
        st.session_state.admin_plan_name = _plan_name
        _target_profile_for_plan = _builder_options[_builder_idx][0]
        _target_has_plan = _target_profile_for_plan and (_target_profile_for_plan.get("assigned_plan") or [])
        if _target_has_plan:
            _existing_plan = _target_profile_for_plan.get("assigned_plan")
            _existing_weeks = _existing_plan.get("plan", _existing_plan) if isinstance(_existing_plan, dict) else (_existing_plan or [])
            _existing_name = _existing_plan.get("plan_name", "") if isinstance(_existing_plan, dict) else ""
            if st.button("Edit current plan", key="admin_edit_plan"):
                _plan_loaded, _ = _deserialize_plan_for_display(_existing_plan)
                st.session_state.admin_plan = _plan_loaded
                st.session_state.admin_plan_name = _existing_name or f"{len(_existing_weeks) if isinstance(_existing_weeks, list) else 0}-Week Plan"
                st.session_state.admin_plan_selected_day = 0
                st.session_state.admin_plan_edit_target = _target_profile_for_plan.get("user_id")
                _existing_completed = _target_profile_for_plan.get("assigned_plan_completed") or {}
                st.session_state.admin_plan_completed = {
                    int(k) if str(k).isdigit() else k: set(v) if isinstance(v, list) else (v if isinstance(v, set) else set())
                    for k, v in _existing_completed.items()
                }
                st.rerun()
        _col_start, _col_weeks = st.columns(2)
        with _col_start:
            _start = st.date_input("Start date", value=date.today(), key="admin_start")
        with _col_weeks:
            _w = st.number_input("Weeks", 1, 16, value=4, key="admin_weeks")

        # Mode days-of-week & session length (above Start date)
        st.markdown("**Session length & days by mode**")
        st.markdown('<div id="admin-mode-days-section" aria-hidden="true" style="height:0;overflow:hidden"></div>', unsafe_allow_html=True)
        st.caption("Click weekdays each mode appears. Length & days on same row.")
        _mode_config: dict = {}
        _mode_days: dict = {}
        for _mode_key in PLAN_MODES:
            _label = MODE_DISPLAY_LABELS.get(_mode_key, _mode_key.replace("_", " ").title())
            _default_len = MODE_SESSION_LEN_DEFAULTS.get(_mode_key, 30)
            _header_cols = st.columns([7, 2])
            with _header_cols[0]:
                st.markdown(f"**{_label}**")
            with _header_cols[1]:
                st.markdown('<p style="text-align: right; margin-bottom: 0;"><strong>Length (min)</strong></p>', unsafe_allow_html=True)
            _content_cols = st.columns([1, 1, 1, 1, 1, 1, 1, 2])
            _selected_days: set[int] = set()
            for _wd, _wd_name in enumerate(WEEKDAY_NAMES):
                with _content_cols[_wd]:
                    if st.checkbox(_wd_name, value=(_wd < 5), key=f"admin_mode_day_{_mode_key}_{_wd}"):
                        _selected_days.add(_wd)
            with _content_cols[7]:
                _len_min = st.slider(
                    "Length (min)",
                    min_value=10,
                    max_value=90,
                    value=_default_len,
                    step=5,
                    key=f"admin_mode_len_{_mode_key}",
                    label_visibility="collapsed",
                )
            _mode_config[_mode_key] = {
                "days": _selected_days,
                "session_len_min": _len_min,
            }
            _mode_days[_mode_key] = _selected_days

        # Plan Highlights (admin only): weekly totals when plan exists
        if st.session_state.get("admin_plan"):
            try:
                _highlights = compute_plan_highlights(st.session_state.admin_plan)
                if _highlights:
                    with st.expander("Plan Highlights (weekly totals)", expanded=True):
                        for h in _highlights:
                            wk = h.get("week", 0)
                            st.markdown(f"**Week {wk}** — Stickhandling: {h.get('stickhandling_hours', 0):.1f}h | Shots: {h.get('shots', 0):,} | Gym: {h.get('gym_hours', 0):.1f}h | Skating: {h.get('skating_hours', 0):.1f}h | Conditioning: {h.get('conditioning_hours', 0):.1f}h | Mobility: {h.get('mobility_hours', 0):.1f}h")
            except Exception:
                pass

        _col_gen, _col_clear = st.columns(2)
        with _col_gen:
            if st.button("Generate plan", type="primary", key="admin_gen_full"):
                data = _load_engine_data()
                profile = _target_profile_for_plan or (st.session_state.get("current_profile") or {})
                _expand = getattr(ENGINE, "expand_user_equipment", lambda x: x or [])
                _equip = profile.get("equipment")
                user_equipment = _expand(_equip) if (ENGINE and _equip) else None
                try:
                    _plan_age = max(6, min(99, int(profile.get("age") or 16)))
                except (TypeError, ValueError):
                    _plan_age = 16
                _plan_athlete = (profile.get("display_name") or profile.get("user_id") or "athlete").strip()
                try:
                    _plan = generate_plan(_w, 7, _start, age=_plan_age, mode_days=_mode_days)
                except TypeError:
                    _plan = generate_plan(_w, 7, _start, mode_days=_mode_days)

                _progress = st.progress(0.0, text="Generating workouts…")
                _total_slots = sum(
                    sum(1 for f in d.get("focus", []) if not ("optional" in (f or "").lower() and "conditioning" in (f or "").lower()))
                    for w in _plan for d in w["days"]
                )
                _slot = [0]  # mutable counter for closure

                def _gen_cb(day_idx: int, focus_str: str, params: dict) -> str:
                    _slot[0] += 1
                    if _total_slots:
                        _progress.progress(min(1.0, _slot[0] / _total_slots), text=f"Generating workout {_slot[0]} of {_total_slots}…")
                    seed = params.get("seed", day_idx * 100)
                    session_mode = params.get("mode", "performance")
                    # Shooting slot without shooting equipment → fallback to stickhandling
                    if session_mode == "shooting" and user_equipment:
                        has_shooting = any("shooting pad" in (e or "").lower() for e in user_equipment)
                        if not has_shooting:
                            session_mode = "stickhandling"
                            params["mode"] = "stickhandling"  # so label matches generated workout
                    resp = ENGINE.generate_session(
                        data=data,
                        age=_plan_age,
                        seed=seed,
                        focus=params.get("focus"),
                        session_mode=session_mode,
                        session_len_min=params.get("session_len_min", 25),
                        athlete_id=f"plan_{_plan_athlete}",
                        use_memory=False,
                        strength_day_type=params.get("strength_day_type"),
                        strength_full_gym=(params.get("mode") == "performance" and params.get("location") == "gym"),
                        strength_emphasis=params.get("strength_emphasis", "strength"),
                        user_equipment=user_equipment,
                    )
                    return resp or "(Empty)"

                _plan = generate_plan_with_workouts(_plan, _gen_cb, base_seed=random.randint(1, 999999), mode_config=_mode_config)
                _progress.empty()
                st.session_state.admin_plan = _plan
                st.session_state.admin_plan_name = st.session_state.get("admin_plan_name") or f"{_w}-Week Plan"
                st.session_state.admin_plan_selected_day = 0
                st.session_state.admin_plan_edit_target = None
                st.rerun()
        with _col_clear:
            if st.button("Clear plan", key="admin_plan_clear"):
                st.session_state.admin_plan = None
                st.session_state.admin_plan_selected_day = 0
                st.session_state.admin_plan_edit_target = None
                if "admin_plan_completed" in st.session_state:
                    st.session_state.admin_plan_completed = {}
                if "admin_plan_workout_view" in st.session_state:
                    st.session_state.admin_plan_workout_view = None
                st.rerun()

        if st.session_state.get("admin_plan"):
            _plan = st.session_state.admin_plan
            total_days = sum(len(w["days"]) for w in _plan)
            flat_days: list[tuple[int, dict]] = []
            for w in _plan:
                for d in w["days"]:
                    flat_days.append((w["week"], d))

            # Bible App style: horizontal day selector, Day X of Y, clickable modes
            if "admin_plan_selected_day" not in st.session_state:
                st.session_state.admin_plan_selected_day = 0
            if "admin_plan_completed" not in st.session_state:
                st.session_state.admin_plan_completed = {}

            # Workout view (separate page): Back / Workout Complete
            if "admin_plan_workout_view" in st.session_state and st.session_state.admin_plan_workout_view is not None:
                wv_day, wv_mode = st.session_state.admin_plan_workout_view
                if wv_day < len(flat_days):
                    _, wv_day_data = flat_days[wv_day]
                    focus_items_wv = wv_day_data.get("focus_items", [])
                    fi_wv = next((x for x in focus_items_wv if x["mode_key"] == wv_mode), None)
                    if fi_wv:
                        _wv_title = ({"leg": "Heavy legs", "upper": "Upper / Core stability", "full": "Explosive legs"}.get((fi_wv.get("params") or {}).get("strength_day_type", "full") or "full", fi_wv["label"]) if fi_wv.get("mode_key") == "performance" else fi_wv["label"])
                        st.markdown(f"### {_wv_title} — Day {wv_day + 1}")
                        if st.button("← Back", key="admin_workout_back"):
                            st.session_state.admin_plan_workout_view = None
                            st.rerun()
                        _workout_text = fi_wv.get("workout") or "(No workout)"
                        if _workout_text != "(No workout)":
                            render_workout_readable(_workout_text)
                        else:
                            st.caption(_workout_text)
                        st.divider()
                        if st.button("Workout Complete", key="admin_workout_complete"):
                            st.session_state.admin_plan_workout_view = None
                            if wv_day not in st.session_state.admin_plan_completed:
                                st.session_state.admin_plan_completed[wv_day] = set()
                            st.session_state.admin_plan_completed[wv_day].add(wv_mode)
                            st.rerun()
                        st.stop()
                st.session_state.admin_plan_workout_view = None

            st.markdown("---")
            st.markdown('<div id="admin-plan-day-grid" aria-hidden="true"></div>', unsafe_allow_html=True)
            st.markdown(f"**Select day** — Day {st.session_state.admin_plan_selected_day + 1} of {total_days}")
            sel_idx = st.session_state.admin_plan_selected_day
            row_cols = st.columns(total_days)
            for i in range(total_days):
                with row_cols[i]:
                    day_data_i = flat_days[i][1]
                    date_str = day_data_i["date"].strftime("%a") if hasattr(day_data_i["date"], "strftime") else str(day_data_i["date"])[:3]
                    _adm_comp = st.session_state.admin_plan_completed.get(i, set()) or set()
                    focus_i = day_data_i.get("focus_items", [])
                    day_done = len(focus_i) > 0 and all(x["mode_key"] in _adm_comp for x in focus_i)
                    if day_done:
                        st.markdown('<div class="admin-day-complete" aria-hidden="true"></div>', unsafe_allow_html=True)
                    if st.button(f"{i + 1}", key=f"admin_plan_day_{i}", type="primary" if i == sel_idx else "secondary"):
                        st.session_state.admin_plan_selected_day = i
                        st.rerun()
                    date_cls = "plan-day-date plan-day-date-selected" if i == sel_idx else "plan-day-date"
                    st.markdown(f'<p class="{date_cls}">{date_str}</p>', unsafe_allow_html=True)

            _, day_data = flat_days[sel_idx]
            st.markdown(f"### Day {sel_idx + 1} of {total_days}")
            st.caption(f"{day_data['date'].strftime('%A, %b %d')}")
            _adm_done = st.session_state.admin_plan_completed.get(sel_idx, set()) or set()
            if day_data.get("focus_items") and all(x["mode_key"] in _adm_done for x in day_data["focus_items"]):
                st.caption("✓ Day complete")

            # List of modes — click to open workout page (Back / Workout Complete there)
            st.markdown('<div id="admin-plan-modes" aria-hidden="true"></div>', unsafe_allow_html=True)
            focus_items = day_data.get("focus_items", [])
            if focus_items:
                for fi in focus_items:
                    completed = st.session_state.admin_plan_completed.get(sel_idx, set()) or set()
                    done = fi["mode_key"] in completed
                    label = f"{'✓ ' if done else ''}{fi['label']}"
                    if st.button(label, key=f"admin_open_{sel_idx}_{fi['mode_key']}", type="primary" if done else "secondary"):
                        st.session_state.admin_plan_workout_view = (sel_idx, fi["mode_key"])
                        st.rerun()
            else:
                # Legacy: plan without focus_items
                for f in day_data.get("focus", []):
                    st.markdown(f"- {f}")

            st.markdown("---")
            st.subheader("Assign plan to player")
            st.caption("Assign this full plan (with workouts) to a player. They will see it in Bible App format when they log in.")
            all_profiles = list_profiles()
            # Exclude admin; show display_name or user_id
            player_options = [(p, p.get("display_name") or p.get("user_id") or "Unknown") for p in all_profiles]
            if player_options:
                _selected_label = st.selectbox(
                    "Select player",
                    options=range(len(player_options)),
                    format_func=lambda i: player_options[i][1],
                    key="admin_assign_player",
                )
                _target_profile = player_options[_selected_label][0]
                if st.button("Assign plan to this player", key="admin_assign_btn"):
                    _plan_name = st.session_state.get("admin_plan_name", "")
                    _plan_to_save = _serialize_plan_for_storage(_plan, _plan_name)
                    _target_profile["assigned_plan"] = _plan_to_save
                    _edit_target = st.session_state.get("admin_plan_edit_target")
                    if _edit_target and _target_profile.get("user_id") == _edit_target:
                        existing_completed = _target_profile.get("assigned_plan_completed") or {}
                        _target_profile["assigned_plan_completed"] = existing_completed
                    else:
                        _target_profile["assigned_plan_completed"] = {}
                    save_profile(_target_profile)
                    st.success(f"Plan assigned to {_target_profile.get('display_name') or _target_profile.get('user_id')}. They will see it on next login.")
                    st.rerun()
            else:
                st.caption("No other profiles found. Create accounts for players first.")

# Custom Plan Requester tab (admin only)
if _admin and st.session_state.get("admin_tab_idx") == 4:
        st.subheader("Admin: Custom Plan Request")
        st.caption("Submitted custom plan intake requests from athletes.")
        requests_list = load_custom_plan_requests()
        if not requests_list:
            st.info("No custom plan requests yet. Athletes can submit requests via **Request Custom Plan** on the Bender tab.")
        else:
            for i, req in enumerate(reversed(requests_list)):
                _req_completed = req.get("completed", False)
                _req_title = f"✓ **{req.get('display_name', 'Unknown')}** — {req.get('created_at', '')[:10]}" if _req_completed else f"**{req.get('display_name', 'Unknown')}** — {req.get('created_at', '')[:10]}"
                with st.expander(_req_title, expanded=(i == 0)):
                    st.markdown(f"**User:** {req.get('display_name', '—')} ({req.get('user_id', '—')})")
                    st.markdown(f"**Submitted:** {req.get('created_at', '—')}")
                    st.markdown("---")
                    st.markdown(f"**1. Weeks:** {req.get('weeks', '—')}")
                    st.markdown(f"**2. Days/week:** {req.get('days_per_week', '—')}")
                    st.markdown(f"**3. Primary goal:** {req.get('primary_goal', '—')}")
                    st.markdown(f"**4. Lifting experience:** {req.get('lifting_experience', '—')}")
                    st.markdown(f"**5. Session length:** {req.get('session_length', '—')}")
                    st.markdown(f"**6. Commitment (1–10):** {req.get('commitment_1_10', '—')}")
                    _btn_col1, _btn_col2 = st.columns(2)
                    with _btn_col1:
                        if st.button("Integrate into Plan Builder", key=f"integrate_req_{req.get('id', i)}", type="primary", use_container_width=True):
                            st.session_state.admin_pending_integration = req
                            st.rerun()
                    with _btn_col2:
                        if st.button("Mark as complete", key=f"complete_req_{req.get('id', i)}", type="secondary", use_container_width=True):
                            mark_custom_plan_request_complete(req.get("id", ""))
                            st.rerun()


# Your Work tab — admin only (players get Your Work via player tabs)
if _admin and st.session_state.get("admin_tab_idx") == 3:
        _render_your_work_stats()


# Admin: Highscores tab (admin only)
if _admin and st.session_state.get("admin_tab_idx") == 2:
        st.subheader("Admin: Highscores")
        st.caption("Lifetime completions across all players. Data from **Workout Complete** (Training Session) and plan completions (My Plan).")
        all_profs = list_profiles()
        if not all_profs:
            st.info("No accounts created yet.")
        else:
            _highscore_options = [(p.get("user_id") or "unknown", p.get("display_name") or p.get("user_id") or "Unknown") for p in all_profs]
            _highscore_ids = [x[0] for x in _highscore_options]
            _highscore_labels = [x[1] for x in _highscore_options]
            _selected_idx = st.selectbox(
                "Select account to view high scores",
                options=range(len(_highscore_ids)),
                format_func=lambda i: _highscore_labels[i],
                key="admin_highscore_select",
            )
            _selected_id = _highscore_ids[_selected_idx]
            _selected_prof = next((p for p in all_profs if (p.get("user_id") or "unknown") == _selected_id), None)
            if _selected_prof:
                stats = _selected_prof.get("private_victory_stats") or {}
                comp = int(stats.get("completions_count", 0))
                name = _selected_prof.get("display_name") or _selected_prof.get("user_id") or "Unknown"
                gym_h = float(stats.get("gym_hours", 0) or 0)
                skating_h = float(stats.get("skating_hours", 0) or 0)
                cond_h = float(stats.get("conditioning_hours", 0) or 0)
                stick_h = float(stats.get("stickhandling_hours", 0) or 0)
                mob_h = float(stats.get("mobility_hours", 0) or 0)
                total_hours = gym_h + skating_h + cond_h + stick_h + mob_h
                shots = int(stats.get("shots", 0) or 0)
                st.markdown(
                    '<div class="your-work-stats-card">'
                    '<div class="your-work-section"><span class="your-work-label">Total Hours</span><span class="your-work-value">{:.1f} h</span></div>'
                    '<div class="your-work-divider"></div>'
                    '<div class="your-work-row"><span class="your-work-cat">Gym</span><span class="your-work-num">{:.1f} h</span></div>'
                    '<div class="your-work-row"><span class="your-work-cat">Skating mechanics</span><span class="your-work-num">{:.1f} h</span></div>'
                    '<div class="your-work-row"><span class="your-work-cat">Conditioning</span><span class="your-work-num">{:.1f} h</span></div>'
                    '<div class="your-work-row"><span class="your-work-cat">Stickhandling</span><span class="your-work-num">{:.1f} h</span></div>'
                    '<div class="your-work-row"><span class="your-work-cat">Mobility / recovery</span><span class="your-work-num">{:.1f} h</span></div>'
                    '<div class="your-work-divider"></div>'
                    '<div class="your-work-section"><span class="your-work-label">Total Shots</span><span class="your-work-value">{:,}</span></div>'
                    '<div class="your-work-footer">{} workout{} completed</div>'
                    '</div>'.format(total_hours, gym_h, skating_h, cond_h, stick_h, mob_h, shots, comp, "s" if comp != 1 else ""),
                    unsafe_allow_html=True,
                )
            st.divider()
            st.caption("Category leaders (who leads each category)")
            # For each category, find the account with the highest value
            _cat_defs = [
                ("Total Hours", "total_hours", lambda s: float(s.get("gym_hours", 0) or 0) + float(s.get("skating_hours", 0) or 0) + float(s.get("conditioning_hours", 0) or 0) + float(s.get("stickhandling_hours", 0) or 0) + float(s.get("mobility_hours", 0) or 0), "{:.1f} h"),
                ("Gym", "gym_hours", lambda s: float(s.get("gym_hours", 0) or 0), "{:.1f} h"),
                ("Skating mechanics", "skating_hours", lambda s: float(s.get("skating_hours", 0) or 0), "{:.1f} h"),
                ("Conditioning", "conditioning_hours", lambda s: float(s.get("conditioning_hours", 0) or 0), "{:.1f} h"),
                ("Stickhandling", "stickhandling_hours", lambda s: float(s.get("stickhandling_hours", 0) or 0), "{:.1f} h"),
                ("Mobility / recovery", "mobility_hours", lambda s: float(s.get("mobility_hours", 0) or 0), "{:.1f} h"),
                ("Total Shots", "shots", lambda s: int(s.get("shots", 0) or 0), "{:,}"),
            ]
            _leader_rows = []
            for cat_label, _key, get_val, fmt in _cat_defs:
                best_val = -1
                best_prof = None
                for p in all_profs:
                    stats = p.get("private_victory_stats") or {}
                    v = get_val(stats)
                    if v > best_val:
                        best_val = v
                        best_prof = p
                if best_prof is not None and best_val >= 0:
                    name = best_prof.get("display_name") or best_prof.get("user_id") or "Unknown"
                    _leader_rows.append((cat_label, name, fmt.format(best_val)))
            if not _leader_rows:
                st.info("No category data yet. Complete workouts to see leaders.")
            else:
                _card_lines = ['<div class="your-work-stats-card">']
                _card_lines.append('<div class="your-work-section"><span class="your-work-label">Category leaders</span><span class="your-work-value"></span></div>')
                _card_lines.append('<div class="your-work-divider"></div>')
                for cat_label, leader_name, value in _leader_rows:
                    _card_lines.append(f'<div class="your-work-row"><span class="your-work-cat">{cat_label}</span><span class="your-work-num">{leader_name} — {value}</span></div>')
                _card_lines.append('</div>')
                st.markdown("\n".join(_card_lines), unsafe_allow_html=True)
