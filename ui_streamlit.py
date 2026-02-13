# ui_streamlit.py
import os
import random
import re
from datetime import datetime

import streamlit as st
import urllib.parse


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
    r"^(warmup|speed|power|high fatigue|block a|block b|strength circuits|circuit a|circuit b|shooting|stickhandling|conditioning|energy systems|speed agility|skating mechanics|mobility|post-lift)\b",
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
    Renders engine text into clean sections:
    - Section headers become bold titles inside bordered cards
    - Drill lines stay as bullets
    - Other guidance lines become small text
    """
    if not text:
        return

    lines = text.splitlines()

    current_title = None
    buffer: list[str] = []

    def flush_section(title: str, body_lines: list[str]) -> None:
        if not title and not body_lines:
            return

        with st.container(border=True):
            if title:
                tag = _header_style(title)
                st.markdown(
                    f"**<span style='color:#0f172a'>{title}</span>**  \n<span style='opacity:.8; color:#64748b; font-size:0.9em'>{tag}</span>",
                    unsafe_allow_html=True,
                )

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

        with st.container(border=True):
            st.subheader("WARMUP")
            st.caption("Strength Circuits")
            for ln in warmup_body:
                s = ln.strip()
                if s.startswith("-"):
                    st.markdown(s)
                else:
                    st.caption(s)

    # ---- Find first Circuit A, then first Circuit B after that ----
    a_start = find_first_exact(("CIRCUIT A",))
    if a_start == -1:
        st.text(text)
        return

    b_start = find_first_exact(("CIRCUIT B",), start=a_start + 1)

    # Circuit A
    a_body = grab_section(a_start, STOP_HEADERS)
    with st.container(border=True):
        st.subheader("CIRCUIT A")
        st.caption("Strength Circuits")
        for ln in a_body:
            s = ln.strip()
            if s.startswith("-"):
                st.markdown(s)
            else:
                st.caption(s)

    # Circuit B (optional)
    if b_start != -1:
        b_body = grab_section(b_start, STOP_HEADERS)
        with st.container(border=True):
            st.subheader("CIRCUIT B")
            st.caption("Strength Circuits")
            for ln in b_body:
                s = ln.strip()
                if s.startswith("-"):
                    st.markdown(s)
                else:
                    st.caption(s)

    # ---- Mobility (optional): render if present ----
    mob_start = -1
    for i, ln in enumerate(lines):
        if ln.strip().startswith("MOBILITY"):
            mob_start = i
            break

    if mob_start != -1:
        mob_body = grab_section(mob_start, STOP_HEADERS)
        with st.container(border=True):
            st.subheader(lines[mob_start].strip())
            st.caption("Mobility")
            for ln in mob_body:
                s = ln.strip()
                if s.startswith("-"):
                    st.markdown(s)
                else:
                    st.caption(s)


    return

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
st.set_page_config(page_title="Bender", layout="centered", initial_sidebar_state="collapsed")

# Custom CSS: clean single-column layout, no sidebar
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

# ---------- Main area: form in card (no sidebar) ----------
form_container = st.container()
with form_container:
    st.markdown('<div class="form-card-marker"></div>', unsafe_allow_html=True)
    st.markdown("#### Session options")
    c1, c2 = st.columns(2)
    with c1:
        athlete_id = st.text_input("Athlete name", value="", placeholder="Enter name")
    with c2:
        age = st.number_input("Age", min_value=6, max_value=99, value=16, step=1)
        age = int(age)
    minutes = st.slider("Session length (minutes)", 10, 120, 45, step=5)

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
            # No-gym: premade circuit + mobility only; no circuit focus or post-lift options
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
        if not athlete_id.strip():
            st.error("Athlete Name is required.")
        elif athlete_id.strip().lower() == "default":
            st.error('Athlete Name "default" is not allowed.')
        else:
            payload = {
                "athlete_id": athlete_id.strip(),
                "age": int(age),
                "minutes": int(minutes),
                "mode": effective_mode,
                "focus": focus,  # controlled tokens only
                "location": location,
                # Strength tokens
                "strength_day_type": strength_day_type,
                "strength_emphasis": strength_emphasis,
                "skate_within_24h": skate_within_24h,
                # Post-lift conditioning
                "conditioning": conditioning,
                "conditioning_type": conditioning_type,
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

# -----------------------------
# Display last generated workout (Tabbed)
# -----------------------------
if st.session_state.last_output_text:
    st.divider()
    # Anchor for scroll-after-generate
    st.markdown('<div id="workout-result"></div>', unsafe_allow_html=True)
    if st.session_state.get("scroll_to_workout"):
        st.session_state.scroll_to_workout = False
        st.components.v1.html(
            "<script>var el = (window.parent && window.parent.document) ? window.parent.document.getElementById('workout-result') : document.getElementById('workout-result'); if (el) el.scrollIntoView({behavior: 'smooth'});</script>",
            height=0,
        )
    # Row: tabs + Clear workout
    _col_tabs, _col_clear = st.columns([5, 1])
    with _col_tabs:
        tab_workout, tab_download, tab_feedback = st.tabs(["Workout", "Download / Copy", "Feedback"])
    with _col_clear:
        if st.button("Clear workout", type="secondary", use_container_width=True):
            clear_last_output()
            st.rerun()

    # -------------------------
    # TAB 1: Workout
    # -------------------------
    with tab_workout:
        st.markdown('<p class="workout-result-header">Your workout</p>', unsafe_allow_html=True)
        # Small badge: mode + duration
        badge_label = f"{MODE_LABELS.get(effective_mode, effective_mode)} · {minutes} min"
        st.markdown(f'<span class="workout-result-badge">{badge_label}</span>', unsafe_allow_html=True)

        # No-gym performance: show circuits-only view ONLY
        if effective_mode == "performance" and location == "no_gym":
            render_no_gym_strength_circuits_only(st.session_state.last_output_text)
        else:
            render_workout_readable(st.session_state.last_output_text)


    # -------------------------
    # TAB 2: Download / Copy
    # -------------------------
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


    # -------------------------
    # TAB 3: Feedback (Google Form)
    # -------------------------
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

        # Location label: only meaningful for performance/energy_systems
        if effective_mode in ("performance", "energy_systems"):
            form_location_value = "Gym" if location == "gym" else "No Gym"
        else:
            form_location_value = "No Gym"  # or "N/A" if your form supports that

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

