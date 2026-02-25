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

PBKDF2_ITERATIONS = 100_000


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


def _equipment_setup_done(profile: dict) -> bool:
    """True if profile has completed required equipment setup (can generate workouts)."""
    return bool(profile.get("equipment_setup_done"))


def _serialize_plan_for_storage(plan: list) -> list:
    """Convert date objects to ISO strings for JSON storage."""
    out = []
    for w in plan:
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
    return out


def _deserialize_plan_for_display(plan: list) -> list:
    """Convert date strings back to date objects for display."""
    out = []
    for w in plan:
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
    return out


def _render_plan_view(plan: list, completed: dict, profile: dict, on_complete: callable) -> None:
    """Render Bible App–style plan view. completed = {day_idx: [mode_key, ...]}. on_complete(day_idx, mode_key)."""
    plan = _deserialize_plan_for_display(plan)
    total_days = sum(len(w["days"]) for w in plan)
    flat_days: list[tuple[int, dict]] = []
    for w in plan:
        for d in w["days"]:
            flat_days.append((w["week"], d))

    if "plan_selected_day" not in st.session_state:
        st.session_state.plan_selected_day = 0
    day_options = [f"Day {i+1} ({d['date'].strftime('%b %d') if hasattr(d['date'], 'strftime') else d['date']})" for i, (_, d) in enumerate(flat_days)]
    _default_day = st.session_state.get("plan_selected_day", 0)
    sel_idx = st.selectbox("Jump to day", range(total_days), index=min(_default_day, max(0, total_days - 1)), format_func=lambda i: day_options[i], key="plan_day_sel")
    st.session_state.plan_selected_day = sel_idx

    _, day_data = flat_days[sel_idx]
    dt_display = day_data["date"].strftime("%A, %b %d") if hasattr(day_data["date"], "strftime") else str(day_data["date"])
    st.markdown(f"### Day {sel_idx + 1} of {total_days}")
    st.caption(dt_display)

    focus_items = day_data.get("focus_items", [])
    if focus_items:
        _completed_for_day = completed.get(sel_idx) or completed.get(str(sel_idx)) or []
        _completed_set = set(_completed_for_day) if isinstance(_completed_for_day, list) else _completed_for_day
        for fi in focus_items:
            done = fi["mode_key"] in _completed_set
            with st.expander(f"{'✓ ' if done else ''}{fi['label']}", expanded=not done):
                _workout_text = fi.get("workout") or "(No workout)"
                if _workout_text != "(No workout)":
                    render_workout_readable(_workout_text)
                else:
                    st.caption(_workout_text)
                if st.button("Completed Workout", key=f"plan_done_{sel_idx}_{fi['mode_key']}"):
                    on_complete(sel_idx, fi["mode_key"])
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

# -----------------------------
# Pretty workout renderer (UI only)
# -----------------------------
_SECTION_RE = re.compile(
    r"^(warmup|speed|power|high fatigue|block a|block b|strength circuits|circuit a|circuit b|shooting|stickhandling|conditioning|energy systems|speed agility|skating mechanics|mobility|post-lift|youth)\b",
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
    return "Section"

def render_workout_readable(text: str) -> None:
    """
    Renders engine text into clean sections.
    Only the warm-up section uses a dropdown (expander); all other sections are always visible.
    """
    if not text:
        return

    lines = text.splitlines()
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
            expander_label = f"{warmup_display} — {tag}" if tag else warmup_display
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
                if tag:
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

CACHE_VERSION = "2026-02-06"  # bump this when data files change

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
    user_equipment = ENGINE.expand_user_equipment(profile.get("equipment"))
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
st.set_page_config(page_title="Bender", layout="centered", initial_sidebar_state="expanded")

# Custom CSS: single-column main; sidebar for equipment
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');

    .stApp { background: #f8fafc; }
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 720px; }
    h1 { font-family: 'DM Sans', sans-serif !important; font-weight: 700 !important; color: #0f172a !important; letter-spacing: -0.02em; }
    .bender-tagline { font-family: 'DM Sans', sans-serif; color: #64748b; font-size: 0.95rem; margin-bottom: 1.25rem; }
    label { font-family: 'DM Sans', sans-serif !important; color: #334155 !important; }

    /* Form card */
    .form-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
    }

    /* Workout result cards */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: white !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 0.75rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    }

    /* Tabs: clearly separate Workout / Download / Feedback */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.75rem;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif;
        color: #64748b;
        padding: 0.5rem 1rem;
        margin-right: 0.25rem;
        border: 1px solid #e2e8f0;
        border-bottom: none;
        border-radius: 8px 8px 0 0;
        background: #f1f5f9;
    }
    .stTabs [data-baseweb="tab"]:first-child { margin-left: 0; }
    .stTabs [aria-selected="true"] {
        color: #0ea5e9 !important;
        background: white !important;
        border-color: #e2e8f0;
        border-bottom: 1px solid white !important;
        margin-bottom: -1px;
    }
    .stTabs [data-baseweb="tab"]:hover { background: #e2e8f0 !important; }
    .stTabs [data-baseweb="tab"]:focus-visible {
        outline: 2px solid #0ea5e9; outline-offset: 2px;
    }

    /* Form card: session options container */
    .main .block-container div[data-testid="stVerticalBlock"]:has(.form-card-marker) {
        background: white;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
    }

    .stButton button {
        font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
        background: #0ea5e9 !important; color: white !important; border: none !important;
        border-radius: 8px !important; padding: 0.5rem 1.5rem !important;
    }
    .stButton button:hover { background: #0284c7 !important; }

    .workout-result-header { font-family: 'DM Sans', sans-serif; font-weight: 600; color: #0f172a; font-size: 1.05rem; margin-bottom: 0.35rem; }
    .workout-result-badge {
        display: inline-block; background: #e0f2fe; color: #0369a1;
        padding: 0.2rem 0.5rem; border-radius: 6px; font-size: 0.8rem; margin-bottom: 0.75rem;
    }
    .stMarkdown p, .stMarkdown li, .stMarkdown ul { color: #334155 !important; }
    .stCaption { color: #64748b !important; }
</style>
""", unsafe_allow_html=True)

st.title("Bender")
st.markdown('<p class="bender-tagline">Hockey workout generator — build 2026-02</p>', unsafe_allow_html=True)

# Session state init
if "last_session_id" not in st.session_state:
    st.session_state.last_session_id = None
if "last_output_text" not in st.session_state:
    st.session_state.last_output_text = None
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
    st.caption("You’ll set your equipment after you log in.")
    create_username = st.text_input(
        "Username (your first and last name)",
        key="create_username",
        placeholder="e.g. John Smith",
        autocomplete="name",
    )
    create_age = st.number_input("Age", min_value=6, max_value=99, value=16, step=1, key="create_age")
    create_password = st.text_input("Password", key="create_password", type="password")
    create_confirm = st.text_input("Confirm password", key="create_confirm", type="password")
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
                profile = {
                    "user_id": uid,
                    "display_name": create_username,
                    "age": int(create_age),
                    "equipment": [],
                    "equipment_setup_done": False,
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
    selected = []
    for mode_name, opts in equipment_by_mode.items():
        st.markdown(f"**{mode_name}**")
        for opt in opts:
            if st.checkbox(opt, value=opt in current_equip, key=f"onb_{mode_name}_{opt}"):
                if opt not in selected:
                    selected.append(opt)
        st.caption("")
    if st.button("Save and continue", key="onb_save"):
        if not selected:
            st.warning("Select at least one option (e.g. \"None\" for no equipment) so we can build workouts.")
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

# Sidebar: equipment by mode (canonical list) + Sign out
with st.sidebar:
    st.markdown(f"**{display_name}**")
    st.divider()
    st.subheader("Equipment")
    st.caption("Check what you have. Workouts only include exercises that use this equipment.")
    try:
        equipment_by_mode = ENGINE.get_canonical_equipment_by_mode()
    except Exception:
        equipment_by_mode = {"Performance": ["None"], "Puck Mastery": [], "Conditioning": ["None"], "Skating Mechanics": ["None"], "Mobility": ["None"]}
    prof = st.session_state.current_profile or {}
    _canonicalize = getattr(ENGINE, "canonicalize_equipment_list", None)
    current_equip = set(_canonicalize(prof.get("equipment") or []) if _canonicalize else (prof.get("equipment") or []))
    all_canonical = []
    _equip_tooltips = {"Line/Tape": "Floor line (tape or painted) for agility / change of direction.", "Reaction ball": "Small rebound ball for reaction drills."}
    for mode_name, opts in equipment_by_mode.items():
        st.markdown(f"**{mode_name}**")
        for opt in opts:
            all_canonical.append((mode_name, opt))
            if opt in _equip_tooltips:
                st.caption(_equip_tooltips[opt])
            st.checkbox(opt, value=opt in current_equip, key=f"sidebar_{mode_name}_{opt}")
    if st.button("Save equipment", key="sidebar_save"):
        new_equip = [
            opt for _mode, opt in all_canonical
            if st.session_state.get(f"sidebar_{_mode}_{opt}", opt in current_equip)
        ]
        prof["equipment"] = new_equip
        st.session_state.current_profile = prof
        save_profile(prof)
        st.success("Saved")
        st.rerun()
    if st.button("Sign out", key="sidebar_logout"):
        st.session_state.current_user_id = None
        st.session_state.current_profile = None
        st.session_state.page = "main"
        st.rerun()  # Shows landing (Log in page)

# ---------- Main area: form in card ----------
# Signed-in line + Sign out (visible in main area)
_col_user, _col_signout = st.columns([4, 1])
with _col_user:
    st.caption(f"Signed in as **{display_name}**")
with _col_signout:
    if st.button("Sign out", key="main_signout"):
        st.session_state.current_user_id = None
        st.session_state.current_profile = None
        st.session_state.page = "main"
        st.rerun()

# Athlete = logged-in user (for history, download filename, feedback)
athlete_id = (st.session_state.current_profile or {}).get("display_name") or (st.session_state.current_profile or {}).get("user_id") or ""
athlete_id = athlete_id.strip() or "athlete"

# Admin: Plan Builder tab (only for Erich Jaeger)
try:
    from admin_plan_builder import is_admin_user, generate_plan, generate_plan_with_workouts
except ImportError:
    is_admin_user = lambda _: False
    generate_plan = lambda *a, **k: []
    generate_plan_with_workouts = lambda p, cb, seed=42: p

_admin = is_admin_user(display_name)
_assigned_plan = (st.session_state.current_profile or {}).get("assigned_plan")
if _admin:
    _tab_bender, _tab_admin = st.tabs(["Bender", "Admin: Plan Builder"])
    _bender_ctx = _tab_bender
    _tab_plan = None
elif _assigned_plan:
    _tab_plan, _tab_generate = st.tabs(["My Plan", "Generate Workout"])
    _bender_ctx = _tab_generate
    _tab_admin = None
else:
    _bender_ctx = nullcontext()
    _tab_admin = None
    _tab_plan = None

# Age from profile (set at account creation)
age = int((st.session_state.current_profile or {}).get("age") or 16)
age = max(6, min(99, age))

# My Plan tab (for players with assigned plan)
if _tab_plan is not None and _assigned_plan:
    with _tab_plan:
        _plan_data = _deserialize_plan_for_display(_assigned_plan)
        _plan_completed = (st.session_state.current_profile or {}).get("assigned_plan_completed") or {}
        if isinstance(_plan_completed, dict):
            _plan_completed = {str(k): (v if isinstance(v, list) else list(v)) for k, v in _plan_completed.items()}

        def _plan_on_complete(day_idx: int, mode_key: str) -> None:
            prof = dict(st.session_state.current_profile or {})
            c = dict(prof.get("assigned_plan_completed") or {})
            key = str(day_idx)
            c[key] = list(set(c.get(key, [])) | {mode_key})
            prof["assigned_plan_completed"] = c
            st.session_state.current_profile = prof
            save_profile(prof)
            st.rerun()

        _render_plan_view(_plan_data, _plan_completed, st.session_state.current_profile or {}, _plan_on_complete)

with _bender_ctx:
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

        if effective_mode in ("performance", "energy_systems"):
            location = st.selectbox("Location", ["gym", "no_gym"], help="Choose 'gym' for strength day, skate-within-24h, and post-lift conditioning options.")
        else:
            location = "no_gym"

        focus = None
        strength_day_type = None
        strength_emphasis = "strength"
        skate_within_24h = False
        conditioning_focus = None

        if effective_mode == "performance":
            if location == "gym":
                day = st.selectbox("Strength day", ["lower", "upper", "full"])
                strength_day_type = "leg" if day == "lower" else ("upper" if day == "upper" else "full")
                em_label = st.selectbox("Strength emphasis", EMPHASIS_DISPLAY, index=EMPHASIS_KEYS.index("strength"))
                strength_emphasis = EMPHASIS_LABEL_TO_KEY[em_label]
            else:
                st.caption("No-gym: you'll get a premade circuit + mobility. For strength day and post-lift conditioning, set Location to **gym**.")
                strength_day_type = "full"
                strength_emphasis = "strength"
                skate_within_24h = False

        elif effective_mode == "energy_systems":
            if location == "gym":
                mod = st.selectbox("Conditioning modality (gym)", ["bike", "treadmill", "surprise"])
                conditioning_focus = {"bike": "conditioning_bike", "treadmill": "conditioning_treadmill"}.get(mod, "conditioning")
            else:
                st.caption("No-gym: cones / no equipment")
                conditioning_focus = "conditioning_cones"
            focus = conditioning_focus

        elif effective_mode == "mobility":
            focus = "mobility"

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
        )

        if st.session_state.last_inputs_fingerprint is None:
            st.session_state.last_inputs_fingerprint = inputs_fingerprint
        else:
            if inputs_fingerprint != st.session_state.last_inputs_fingerprint:
                if st.session_state.last_session_id or st.session_state.last_output_text:
                    clear_last_output()
                st.session_state.last_inputs_fingerprint = inputs_fingerprint

        # Generate action (prominent in main area)
        col_btn, _ = st.columns([1, 3])
        with col_btn:
            generate_clicked = st.button("Generate workout", type="primary", use_container_width=True)
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
            }

            try:
                with st.spinner("Generating workout..."):
                    if USE_API:
                        resp = _generate_via_api(payload)
                    else:
                        resp = _generate_via_engine(payload)

                st.session_state.last_session_id = resp.get("session_id")
                st.session_state.last_output_text = resp.get("output_text")
                st.session_state.scroll_to_workout = True
                st.success("Generated")
            except Exception as e:
                st.error(str(e))

    # Display last generated workout (Tabbed)
    if st.session_state.last_output_text:
        st.divider()
        st.markdown('<div id="workout-result"></div>', unsafe_allow_html=True)
        if st.session_state.get("scroll_to_workout"):
            st.session_state.scroll_to_workout = False
            st.components.v1.html(
                "<script>var el = (window.parent && window.parent.document) ? window.parent.document.getElementById('workout-result') : document.getElementById('workout-result'); if (el) el.scrollIntoView({behavior: 'smooth'});</script>",
                height=0,
            )
        _col_tabs, _col_clear = st.columns([5, 1])
        with _col_tabs:
            tab_workout, tab_download, tab_feedback = st.tabs(["Workout", "Download / Copy", "Feedback"])
        with _col_clear:
            if st.button("Clear workout", type="secondary", use_container_width=True):
                clear_last_output()
                st.rerun()

        with tab_workout:
            st.markdown('<p class="workout-result-header">Your workout</p>', unsafe_allow_html=True)
            badge_label = f"{MODE_LABELS.get(effective_mode, effective_mode)} · {minutes} min"
            st.markdown(f'<span class="workout-result-badge">{badge_label}</span>', unsafe_allow_html=True)
            if effective_mode == "performance" and location == "no_gym":
                render_no_gym_strength_circuits_only(st.session_state.last_output_text)
            else:
                render_workout_readable(st.session_state.last_output_text)

        with tab_download:
            safe_name = re.sub(r"[^\w\-]", "_", athlete_id.strip())[:30] or "workout"
            date_str = datetime.now().strftime("%Y-%m-%d")
            download_filename = f"bender_workout_{safe_name}_{date_str}.txt"
            st.download_button(
                label="Download workout (.txt)",
                data=st.session_state.last_output_text,
                file_name=download_filename,
                mime="text/plain",
            )
            st.caption("Your browser will download the file when you click above.")
            st.write("")
            if not (effective_mode == "performance" and location == "no_gym"):
                with st.expander("Copy workout (raw text)"):
                    st.code(st.session_state.last_output_text)
                    st.caption("Select the text above and copy (Ctrl+C / Cmd+C).")

        with tab_feedback:
            st.write("Leave feedback so I can improve workouts.")

        # Map your internal mode token to the form’s expected label
            form_mode_value = {
                "skills_only": "Puck Mastery (Both)",
                "shooting": "Puck Mastery (Shooting)",
                "stickhandling": "Puck Mastery (Stickhandling)",
                "performance": "Performance",
                "energy_systems": "Conditioning",
                "skating_mechanics": "Skating Mechanics",
                "mobility": "Mobility",
            }.get(effective_mode, mode_label)
            if effective_mode in ("performance", "energy_systems"):
                form_location_value = "Gym" if location == "gym" else "No Gym"
            else:
                form_location_value = "No Gym"
            form_emphasis_value = strength_emphasis if effective_mode == "performance" else ""
            prefill_url = build_prefilled_feedback_url(
                athlete=athlete_id.strip(),
                mode_label=form_mode_value,
                location_label=form_location_value,
                emphasis_key=form_emphasis_value,
                rating=4,
                notes="",
            )
            st.link_button("Leave Feedback (auto-filled)", prefill_url)
            st.link_button("Open Feedback Form (blank)", FORM_BASE)

# Admin tab: Plan Builder (only for Erich Jaeger)
if _tab_admin is not None:
    with _tab_admin:
        st.subheader("Admin: Plan Builder")
        st.caption("Multi-week workout plans. Generate with full workouts for each day (Bible App–style view).")
        _w = st.number_input("Weeks", 1, 16, value=4, key="admin_weeks")
        _d = st.number_input("Days per week", 3, 7, value=5, key="admin_days")
        _start = st.date_input("Start date", value=date.today(), key="admin_start")
        _col_gen, _col_full = st.columns(2)
        with _col_gen:
            if st.button("Generate plan (structure only)", key="admin_gen"):
                _plan = generate_plan(_w, _d, _start)
                st.session_state.admin_plan = _plan
                st.rerun()
        with _col_full:
            if st.button("Generate plan with workouts", type="primary", key="admin_gen_full"):
                _plan = generate_plan(_w, _d, _start)
                data = _load_engine_data()
                profile = st.session_state.get("current_profile") or {}
                _expand = getattr(ENGINE, "expand_user_equipment", lambda x: x or [])
                user_equipment = _expand(profile.get("equipment")) if ENGINE else []

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
                    resp = ENGINE.generate_session(
                        data=data,
                        age=age,
                        seed=seed,
                        focus=params.get("focus"),
                        session_mode=params.get("mode", "performance"),
                        session_len_min=params.get("session_len_min", 25),
                        athlete_id=f"plan_{athlete_id}",
                        use_memory=False,
                        strength_day_type=params.get("strength_day_type"),
                        strength_full_gym=(params.get("mode") == "performance" and params.get("location") == "gym"),
                        strength_emphasis=params.get("strength_emphasis", "strength"),
                        user_equipment=user_equipment,
                    )
                    return resp or "(Empty)"

                _plan = generate_plan_with_workouts(_plan, _gen_cb, base_seed=random.randint(1, 999999))
                _progress.empty()
                st.session_state.admin_plan = _plan
                st.session_state.admin_plan_selected_day = 0
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

            st.markdown("---")
            st.markdown("**Select day** — *Day X of Y*")
            # Day selector: show ~4 at a time, scroll via slider/select
            day_options = [f"Day {i+1} ({d['date'].strftime('%b %d')})" for i, (_, d) in enumerate(flat_days)]
            _default_day = st.session_state.get("admin_plan_selected_day", 0)
            sel_idx = st.selectbox("Jump to day", range(total_days), index=min(_default_day, total_days - 1), format_func=lambda i: day_options[i], key="admin_plan_day_sel")
            st.session_state.admin_plan_selected_day = sel_idx

            _, day_data = flat_days[sel_idx]
            st.markdown(f"### Day {sel_idx + 1} of {total_days}")
            st.caption(f"{day_data['date'].strftime('%A, %b %d')}")

            # List of modes (Performance, Skating Mechanics, etc.) — click to expand, see full workout
            focus_items = day_data.get("focus_items", [])
            if focus_items:
                for fi in focus_items:
                    completed = st.session_state.admin_plan_completed.get(sel_idx, set()) or set()
                    done = fi["mode_key"] in completed
                    with st.expander(f"{'✓ ' if done else ''}{fi['label']}", expanded=not done):
                        _workout_text = fi.get("workout") or "(No workout)"
                        if _workout_text != "(No workout)":
                            render_workout_readable(_workout_text)
                        else:
                            st.caption(_workout_text)
                        if st.button("Completed Workout", key=f"admin_done_{sel_idx}_{fi['mode_key']}"):
                            if sel_idx not in st.session_state.admin_plan_completed:
                                st.session_state.admin_plan_completed[sel_idx] = set()
                            st.session_state.admin_plan_completed[sel_idx].add(fi["mode_key"])
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
                    _plan_to_save = _serialize_plan_for_storage(_plan)
                    _target_profile["assigned_plan"] = _plan_to_save
                    _target_profile["assigned_plan_completed"] = {}
                    save_profile(_target_profile)
                    st.success(f"Plan assigned to {_target_profile.get('display_name') or _target_profile.get('user_id')}. They will see it on next login.")
                    st.rerun()
            else:
                st.caption("No other profiles found. Create accounts for players first.")
