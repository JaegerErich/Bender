# ui_streamlit.py
import os
import random
import re
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
    # Prefer decision tree list if available
    try:
        import bender_decision_tree_v4 as dt
        if hasattr(dt, "SESSION_MODES"):
            modes = list(dt.SESSION_MODES)
            if modes:
                return modes
    except Exception:
        pass

    # Canonical defaults
    return ["skills_only", "shooting", "stickhandling", "strength", "conditioning", "mobility", "movement"]


RAW_MODES = get_mode_options()

MODE_LABELS = {
    "skills_only": "Shooting & Stickhandling",
    "shooting": "Shooting Only",
    "stickhandling": "Stickhandling Only",
    "strength": "Strength",
    "conditioning": "Conditioning",
    "mobility": "Mobility & Recovery",
    "movement": "Movement",
}

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
    r"^(warmup|speed|power|high fatigue|block a|block b|strength circuits|circuit a|circuit b|shooting|stickhandling|conditioning|mobility)\b",
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
    if "conditioning" in t:
        return "Conditioning"
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
                    f"**{title}**  \n<span style='opacity:.75'>{tag}</span>",
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
        "SHOOTING",
        "STICKHANDLING",
        "CONDITIONING",
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
    strength_full_gym = (mode == "strength" and location == "gym")

    strength_day_type = payload.get("strength_day_type", None)  # "leg"/"upper"
    strength_emphasis = payload.get("strength_emphasis", "strength")
    skate_within_24h = bool(payload.get("skate_within_24h", False))

    include_post_lift_conditioning = bool(payload.get("conditioning", False)) if mode == "strength" else None
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
st.set_page_config(page_title="Bender MVP", layout="centered")
st.title("Bender – Web MVP")
st.caption("Bender beta — build 2026-02-05")

# Session state init
if "last_session_id" not in st.session_state:
    st.session_state.last_session_id = None
if "last_output_text" not in st.session_state:
    st.session_state.last_output_text = None
if "last_inputs_fingerprint" not in st.session_state:
    st.session_state.last_inputs_fingerprint = None

# Core inputs
athlete_id = st.text_input("Athlete Name", value="")
age = st.number_input("Age", min_value=6, max_value=99, value=16, step=1)
age = int(age)
minutes = st.slider("Session length (minutes)", 10, 120, 45, step=5)

mode_label = st.selectbox("Mode", DISPLAY_MODES)
mode = LABEL_TO_MODE[mode_label]

# Location only relevant for strength/conditioning
if mode in ("strength", "conditioning"):
    location = st.selectbox("Location", ["gym", "no_gym"])
else:
    location = "no_gym"

# Mode-specific controls (structured, no free-text)
focus = None

# Strength extras
strength_day_type = None
strength_emphasis = "strength"
skate_within_24h = False

# Conditioning extras
conditioning_focus = None

if mode == "strength":
    if location == "gym":
        day = st.selectbox("Strength day", ["lower", "upper", "full"])
        strength_day_type = "leg" if day == "lower" else ("upper" if day == "upper" else "full")

        em_label = st.selectbox(
            "Strength emphasis",
            EMPHASIS_DISPLAY,
            index=EMPHASIS_KEYS.index("strength"),
        )
        strength_emphasis = EMPHASIS_LABEL_TO_KEY[em_label]

        skate_within_24h = st.checkbox("Skate within 24h?", value=False)
    else:
        day = st.selectbox("Circuit focus (no gym)", ["lower", "upper", "full"])
        strength_day_type = "leg" if day == "lower" else ("upper" if day == "upper" else "full")
        strength_emphasis = "strength"
        skate_within_24h = False

elif mode == "conditioning":
    if location == "gym":
        mod = st.selectbox("Conditioning modality (gym)", ["bike", "treadmill", "surprise"])
        if mod == "bike":
            conditioning_focus = "conditioning_bike"
        elif mod == "treadmill":
            conditioning_focus = "conditioning_treadmill"
        else:
            conditioning_focus = "conditioning"
    else:
        st.info("No-gym conditioning assumes cones/no equipment.")
        conditioning_focus = "conditioning_cones"

    focus = conditioning_focus

elif mode == "mobility":
    focus = "mobility"

# Strength-only: post-lift conditioning (gym/no_gym restrictions)
conditioning = False
conditioning_type = None

if mode == "strength":
    conditioning = st.checkbox("Post-lift conditioning?", value=False)
    if conditioning:
        if location == "gym":
            conditioning_type = st.selectbox(
                "Post-lift conditioning type (gym)",
                ["bike", "treadmill", "surprise"],
            )
        else:
            st.info("No-gym conditioning = no equipment (no bike/treadmill).")
            conditioning_type = None


# Auto-clear old output if key inputs change
inputs_fingerprint = (
    athlete_id.strip().lower(),
    int(age),
    int(minutes),
    mode,
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

# Generate action
if st.button("Generate"):
    if not athlete_id.strip():
        st.error("Athlete Name is required.")
    elif athlete_id.strip().lower() == "default":
        st.error('Athlete Name "default" is not allowed.')
    else:
        payload = {
            "athlete_id": athlete_id.strip(),
            "age": int(age),
            "minutes": int(minutes),
            "mode": mode,
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
            st.success("Generated")
        except Exception as e:
            st.error(str(e))

# -----------------------------
# Display last generated workout (Tabbed)
# -----------------------------
if st.session_state.last_output_text:
    st.divider()

    tab_workout, tab_download, tab_feedback = st.tabs(["Workout", "Download / Copy", "Feedback"])

    # -------------------------
    # TAB 1: Workout
    # -------------------------
    with tab_workout:
        st.subheader("Your Workout")

        # No-gym strength: show circuits-only view ONLY
        if mode == "strength" and location == "no_gym":
            render_no_gym_strength_circuits_only(st.session_state.last_output_text)
        else:
            render_workout_readable(st.session_state.last_output_text)


    # -------------------------
    # TAB 2: Download / Copy
    # -------------------------
    with tab_download:
        st.download_button(
            label="Download workout (.txt)",
            data=st.session_state.last_output_text,
            file_name="bender_workout.txt",
            mime="text/plain",
        )

        st.write("")

        if not (mode == "strength" and location == "no_gym"):
            with st.expander("Copy workout (raw text)"):
                st.code(st.session_state.last_output_text)


    # -------------------------
    # TAB 3: Feedback (Google Form)
    # -------------------------
    with tab_feedback:
        st.write("Leave feedback so I can improve workouts.")

        # Map your internal mode token to the form’s expected label
        form_mode_value = {
            "skills_only": "Shooting & Stickhandling",
            "shooting": "Shooting",
            "stickhandling": "Stickhandling",
            "strength": "Strength",
            "conditioning": "Conditioning",
            "mobility": "Mobility",
            "movement": "Movement",
        }.get(mode, mode_label)

        # Location label: only meaningful for strength/conditioning
        if mode in ("strength", "conditioning"):
            form_location_value = "Gym" if location == "gym" else "No Gym"
        else:
            form_location_value = "No Gym"  # or "N/A" if your form supports that

        form_emphasis_value = strength_emphasis if mode == "strength" else ""

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

