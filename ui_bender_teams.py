"""
Bender Teams UI: Coach dashboard, roster, assignments, feedback.
Rendered as additive tabs/views. Does not replace existing player experience.
"""
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st

# Import bender_teams data layer (profile loading passed from ui_streamlit)
try:
    from bender_teams import (
        add_member_to_team,
        create_assignment,
        create_feedback,
        create_team,
        create_team_request,
        has_pending_team_request,
        get_assignments_for_player,
        get_assignments_for_team,
        get_feedback_for_player,
        get_feedback_for_team,
        get_team_by_id,
        get_team_by_invite_code,
        get_team_members,
        get_team_players,
        get_team_players_extended,
        get_teams_coached_by,
        get_teams_for_user,
        remove_member_from_team,
        get_player_activity_summary,
        get_team_activity_summary,
        get_recent_team_activity,
        get_team_weekly_targets,
        set_team_weekly_targets,
        get_team_mode_minutes,
        get_team_volume_totals,
        TEAM_CATEGORY_ORDER,
        TEAM_CATEGORY_LABELS,
        is_team_coach,
        mark_assignment_completed,
        FEEDBACK_TYPES,
        ASSIGNED_TO_TEAM,
        ASSIGNED_TO_PLAYER,
        ASSIGNED_TO_SUBGROUP,
        SUBGROUPS,
        update_assignment,
        get_assignment_completion_counts,
        get_assignment_by_id,
        get_team_focus,
        set_team_focus,
        ASSIGNMENT_TYPE_GENERATED,
        ASSIGNMENT_TYPE_SPECIFIC,
        ASSIGNMENT_TYPE_TEAM_FOCUS,
        ASSIGNMENT_TYPE_CHALLENGE,
    )
except ImportError:
    # Graceful fallback if module missing
    def _noop(*a, **k):
        return [] if "team" in str(a) + str(k) else None
    add_member_to_team = create_assignment = create_feedback = create_team = create_team_request = _noop
    has_pending_team_request = lambda *a: False
    get_assignments_for_player = get_assignments_for_team = get_feedback_for_player = get_feedback_for_team = lambda *a: []
    get_team_by_id = get_team_by_invite_code = lambda *a: None
    remove_member_from_team = lambda *a: False
    get_team_members = get_team_players = get_team_players_extended = get_teams_for_user = get_teams_coached_by = lambda *a: []
    get_player_activity_summary = get_team_activity_summary = get_recent_team_activity = lambda *a: {}
    get_team_weekly_targets = lambda *a: {}
    set_team_weekly_targets = lambda *a: False
    get_team_mode_minutes = get_team_volume_totals = lambda *a: {}
    TEAM_CATEGORY_ORDER = []
    TEAM_CATEGORY_LABELS = {}
    is_team_coach = lambda *a: False
    mark_assignment_completed = lambda *a: False
    FEEDBACK_TYPES = ()
    ASSIGNED_TO_TEAM = ASSIGNED_TO_PLAYER = ASSIGNED_TO_SUBGROUP = ""
    SUBGROUPS = ()
    update_assignment = lambda *a, **k: False
    get_assignment_by_id = lambda *a, **k: None
    get_assignment_completion_counts = lambda *a, **k: {}
    get_team_focus = set_team_focus = lambda *a, **k: None
    ASSIGNMENT_TYPE_GENERATED = ASSIGNMENT_TYPE_SPECIFIC = ASSIGNMENT_TYPE_TEAM_FOCUS = ASSIGNMENT_TYPE_CHALLENGE = ""


def _add_team_to_player_profile(prof: dict, team: dict) -> dict:
    """Update profile with team membership + player_teams_cache. Returns updated profile."""
    ids = list(prof.get("bender_team_ids") or [])
    tid = team.get("team_id")
    if tid and tid not in ids:
        ids.append(tid)
    prof["bender_team_ids"] = ids
    prof["team"] = (team.get("team_name") or "").strip()
    cache = [c for c in (prof.get("player_teams_cache") or []) if c.get("team_id") != tid]
    cache.append(dict(team))
    prof["player_teams_cache"] = cache[-20:]
    return prof


def _remove_team_from_player_profile(prof: dict, team_id: str, remaining_teams: list[dict] | None = None) -> dict:
    """Remove team from profile. Returns updated profile."""
    ids = [x for x in (prof.get("bender_team_ids") or []) if x != team_id]
    prof["bender_team_ids"] = ids
    cache = [c for c in (prof.get("player_teams_cache") or []) if c.get("team_id") != team_id]
    prof["player_teams_cache"] = cache
    prof["team"] = (remaining_teams[0].get("team_name", "").strip() if remaining_teams else "") or ""
    return prof


def _profile_loader(load_profile_fn: Callable) -> Callable:
    def load(uid):
        return load_profile_fn(uid) if uid else None
    return load


# --- Join team only (for players not on a team) ---
def render_team_join_only(load_profile_fn: Callable, save_callback: Callable[[dict], None]):
    """Show join UI when player has no team (player portal entry point)."""
    st.info("You are not currently on a team. Join a team with an invite code.")
    with st.expander("Join a team", expanded=True):
        join_code = st.text_input("Invite code", key="teams_join_code_portal", placeholder="e.g. ABC123").strip().upper()
        if st.button("Join team", key="teams_join_btn_portal"):
            if not join_code:
                st.error("Enter an invite code.")
            else:
                t = get_team_by_invite_code(join_code)
                if not t:
                    st.error("Invalid invite code.")
                else:
                    uid = st.session_state.current_user_id
                    if add_member_to_team(t["team_id"], uid, "player"):
                        prof = _add_team_to_player_profile(load_profile_fn(uid) or {}, t)
                        save_callback(prof)
                        if st.session_state.get("current_user_id") == uid and "current_profile" in st.session_state:
                            st.session_state.current_profile = prof
                        st.success(f"You joined **{t.get('team_name', 'team')}**.")
                        st.rerun()
                    else:
                        st.info("You're already on this team.")


# --- Team creation + Join ---
def render_team_creation(load_profile_fn: Callable, save_callback: Callable[[dict], None]):
    st.subheader("Bender Teams")
    st.caption("Create a team or join one with a code.")
    # Join team
    with st.expander("Join a team", expanded=True):
        join_code = st.text_input("Invite code", key="teams_join_code", placeholder="e.g. ABC123").strip().upper()
        if st.button("Join team", key="teams_join_btn"):
            if not join_code:
                st.error("Enter an invite code.")
            else:
                t = get_team_by_invite_code(join_code)
                if not t:
                    st.error("Invalid invite code.")
                else:
                    uid = st.session_state.current_user_id
                    if add_member_to_team(t["team_id"], uid, "player"):
                        prof = _add_team_to_player_profile(load_profile_fn(uid) or {}, t)
                        save_callback(prof)
                        if st.session_state.get("current_user_id") == uid and "current_profile" in st.session_state:
                            st.session_state.current_profile = prof
                        st.success(f"You joined **{t.get('team_name', 'team')}**.")
                        st.rerun()
                    else:
                        st.info("You're already on this team.")
    st.markdown("---")
    st.subheader("Request a new team")
    st.caption("Submit a request to create a team. An admin will review and approve or deny.")
    with st.form("create_team_form"):
        team_name = st.text_input("Team name", placeholder="e.g. Eagles U16")
        age_group = st.text_input("Age group (optional)", placeholder="e.g. U16")
        level = st.selectbox("Level (optional)", ["", "Youth", "HS", "AA", "AAA", "Junior", "College", "Beer League"])
        season = st.text_input("Season (optional)", placeholder="e.g. 2024-25")
        if st.form_submit_button("Submit request"):
            name = (team_name or "").strip()
            if not name:
                st.error("Enter a team name.")
            else:
                prof = load_profile_fn(st.session_state.current_user_id)
                uid = (prof or {}).get("user_id") or st.session_state.current_user_id
                disp = (prof or {}).get("display_name") or uid
                if has_pending_team_request(uid, name):
                    st.error("You already have a pending request for this team name. Wait for admin approval before submitting again.")
                elif create_team_request(uid, disp, name, age_group=age_group or "", level=level or "", season=season or ""):
                    st.success("The request has been submitted.")
                    st.rerun()


# --- Coach: Overview dashboard ---
def render_coach_overview(team_id: str, load_profile_fn: Callable):
    st.subheader("Overview")
    t = get_team_by_id(team_id)
    if not t:
        st.info("Share your invite code above so players can join. Activity and metrics will appear here once they start training.")
        return
    loader = _profile_loader(load_profile_fn)
    summary = get_team_activity_summary(team_id, loader)
    players = get_team_players_extended(team_id)
    if not players:
        st.info("Share your invite code above so players can join. Activity and metrics will appear here once they start training.")
        return
    st.markdown("")  # spacer
    # Highlight: total training time
    total_all = int(summary.get("total_training_minutes_all_time", 0))
    total_week = int(summary.get("total_training_minutes", 0))
    if total_all > 0 or total_week > 0:
        parts = []
        if total_all > 0:
            h_all, m_all = divmod(total_all, 60)
            parts.append(f"{h_all}h {m_all}m total" if h_all > 0 else f"{m_all}m total")
        if total_week > 0:
            h_week, m_week = divmod(total_week, 60)
            parts.append(f"{h_week}h {m_week}m this week" if h_week > 0 else f"{m_week}m this week")
        st.info(f"**Team training time:** {' · '.join(parts)}")
    st.markdown("")
    # Top metrics — Active players this week & Workouts this week
    _metric_style = "border: 1px solid rgba(255,255,255,0.2); border-radius: 8px; padding: 1rem 1.25rem; background: rgba(255,255,255,0.05);"
    _metric_label = "font-size: 0.85rem; color: rgba(255,255,255,0.7);"
    _metric_val = "font-size: 1.5rem; font-weight: 700; color: #fff;"
    active = summary.get("active_players_this_week", 0)
    total = summary.get("total_players", 0)
    workouts = summary.get("workouts_this_week", 0)
    st.markdown(
        '<div style="display: flex; gap: 1rem; flex-wrap: wrap;">'
        f'<div style="{_metric_style} flex: 1; min-width: 140px;"><div style="{_metric_label}">Active players this week</div><div style="{_metric_val}">{active} / {total}</div></div>'
        f'<div style="{_metric_style} flex: 1; min-width: 140px;"><div style="{_metric_label}">Workouts this week</div><div style="{_metric_val}">{workouts}</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Weekly targets, current week avg, season totals
    st.markdown("#### Weekly targets & progress")
    targets = get_team_weekly_targets(team_id)
    week_mins = get_team_mode_minutes(team_id, loader, period="week")
    n_players = max(1, len(players))
    rows = []
    for cat in TEAM_CATEGORY_ORDER:
        label = TEAM_CATEGORY_LABELS.get(cat, cat)
        avg_week = int(week_mins.get(cat, 0) / n_players)
        rows.append({"Category": label, "Target (min/week)": int(targets.get(cat, 0)), "Avg this week": f"{avg_week} min"})
    df = pd.DataFrame(rows)
    edited = st.data_editor(
        df,
        column_config={
            "Category": st.column_config.TextColumn("Category", disabled=True, width="medium"),
            "Target (min/week)": st.column_config.NumberColumn("Target (min/week)", min_value=0, max_value=600, step=15, width="small"),
            "Avg this week": st.column_config.TextColumn("Avg this week", disabled=True, width="small"),
        },
        hide_index=True,
        key=f"weekly_targets_{team_id}",
    )
    if st.button("Save targets", key=f"save_targets_{team_id}"):
        new_targets = {}
        for i, cat in enumerate(TEAM_CATEGORY_ORDER):
            if i < len(edited):
                new_targets[cat] = int(edited.iloc[i]["Target (min/week)"])
        set_team_weekly_targets(team_id, new_targets)
        st.rerun()

    # Team volume by mode (Bender Board style)
    st.markdown("#### Team volume by mode")
    week_vol = get_team_volume_totals(team_id, loader, period="week")
    season_vol = get_team_volume_totals(team_id, loader, period="season")
    vol_cats = [
        ("Shots", "shots", "{:,}"),
        ("Stickhandling time", "stickhandling_hours", "{:.1f} h"),
        ("Skating mechanics time", "skating_hours", "{:.1f} h"),
        ("Conditioning time", "conditioning_hours", "{:.1f} h"),
        ("Gym time", "gym_hours", "{:.1f} h"),
        ("Mobility / recovery time", "mobility_hours", "{:.1f} h"),
    ]
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.caption("**Category**")
    with c2:
        st.caption("**This week**")
    with c3:
        st.caption("**Season**")
    for label, key, fmt in vol_cats:
        with c1:
            st.write(label)
        with c2:
            v = week_vol.get(key, 0)
            st.write(fmt.format(v))
        with c3:
            v = season_vol.get(key, 0)
            st.write(fmt.format(v))

    st.divider()
    # Roster snapshot + attention
    col_roster, col_attention = st.columns([2, 1])
    with col_roster:
        st.markdown("#### Roster snapshot")
        players = get_team_players_extended(team_id)
        for m in players[:15]:
            uid = m.get("user_id")
            prof = load_profile_fn(uid)
            name = (prof or {}).get("display_name") or uid
            pos = (prof or {}).get("position") or "—"
            act = get_player_activity_summary(prof or {}, 7)
            streak = act.get("streak", 0)
            last = act.get("last_workout_date") or "Never"
            workouts = act.get("workouts_this_period", 0)
            st.caption(f"**{name}** ({pos}) · Streak: {streak} · Last: {last} · This week: {workouts}")
    with col_attention:
        st.markdown("#### Needs attention")
        inactive = summary.get("inactive_7_days", [])
        for uid in inactive[:5]:
            prof = load_profile_fn(uid)
            name = (prof or {}).get("display_name") or uid
            st.caption(f"⚠️ {name} — No workout in 7+ days")
    st.divider()
    st.markdown("#### Recent activity")
    feed = get_recent_team_activity(team_id, loader, 15)
    for item in feed:
        tpe = item.get("type", "")
        if tpe == "workout_completed":
            st.caption(f"✓ **{item.get('display_name', '')}** completed a workout ({item.get('mode', '')}) — {item.get('date', '')}")
        elif tpe == "assignment":
            st.caption(f"📋 Workout **{item.get('workout_title', '')}** assigned — {item.get('created_at', '')[:10]}")
        elif tpe == "feedback":
            st.caption(f"💬 Feedback left for player — {item.get('created_at', '')[:10]}")


# --- Coach: Roster page ---
def render_coach_roster(team_id: str, load_profile_fn: Callable, save_profile_fn: Callable | None, on_select_player: Callable[[str], None]):
    st.subheader("Roster")
    t = get_team_by_id(team_id)
    if not t:
        st.info("Share your invite code above so players can join. The roster will appear here once they join your team.")
        return
    players = get_team_players_extended(team_id)
    if not players:
        st.info("Share your invite code above so players can join. The roster will appear here once they join your team.")
        return
    for m in players:
        uid = m.get("user_id")
        prof = load_profile_fn(uid)
        act = get_player_activity_summary(prof or {}, 7)
        name = (prof or {}).get("display_name") or uid
        pos = (prof or {}).get("position") or "—"
        streak = act.get("streak", 0)
        last = act.get("last_workout_date") or "Never"
        workouts_7 = act.get("workouts_this_period", 0)
        if workouts_7 > 0:
            status = "Active"
        elif last == "Never":
            status = "Inactive"
        elif last:
            try:
                status = "Inactive" if (date.today() - date.fromisoformat(last[:10])).days >= 7 else "Due soon"
            except Exception:
                status = "—"
        else:
            status = "—"
        with st.container():
            cols = st.columns([3, 1, 1, 1, 0.9])
            with cols[0]:
                st.write(f"**{name}**")
                st.caption(f"{pos} · {status}")
            with cols[1]:
                st.write(str(streak))
                st.caption("Streak")
            with cols[2]:
                st.write(last[:10] if len(last or "") >= 10 else last)
                st.caption("Last workout")
            with cols[3]:
                st.write(str(workouts_7))
                st.caption("This week")
            with cols[4]:
                if st.button("View", key=f"roster_view_{uid}"):
                    on_select_player(uid)
                    st.rerun()
                if save_profile_fn and st.button("Remove", key=f"roster_remove_{uid}"):
                    remove_member_from_team(team_id, uid)
                    prof = load_profile_fn(uid) or {}
                    remaining = [c for c in (prof.get("player_teams_cache") or []) if c.get("team_id") != team_id]
                    prof = _remove_team_from_player_profile(prof, team_id, remaining)
                    save_profile_fn(prof)
                    st.rerun()
            if m != players[-1]:
                st.markdown("<hr style='margin: 0.25rem 0; opacity: 0.4;'>", unsafe_allow_html=True)


# --- Coach: Player profile view ---
def render_coach_player_profile(team_id: str, player_id: str, load_profile_fn: Callable, generate_callback=None):
    prof = load_profile_fn(player_id)
    if not prof:
        st.warning("Player not found.")
        return
    name = prof.get("display_name") or player_id
    st.subheader(name)
    st.caption(f"{prof.get('position', '')} · {prof.get('age', '')} · {prof.get('current_level', '')}")
    act_7 = get_player_activity_summary(prof, 7)
    act_30 = get_player_activity_summary(prof, 30)
    a1, a2, a3 = st.columns(3)
    with a1:
        st.metric("Workouts (7d)", act_7.get("workouts_this_period", 0))
    with a2:
        st.metric("Workouts (30d)", act_30.get("workouts_this_period", 0))
    with a3:
        st.metric("Current streak", act_7.get("streak", 0))
    hist = prof.get("completion_history") or []
    cats = {}
    for e in hist:
        m = (e.get("mode") or "").lower()
        cats[m] = cats.get(m, 0) + 1
    st.markdown("#### Category breakdown")
    st.caption(f"Performance: {cats.get('performance', 0)} · Skating: {cats.get('skating_mechanics', 0)} · Puck: {cats.get('stickhandling', 0) + cats.get('shooting', 0) + cats.get('skills_only', 0)} · Conditioning: {cats.get('energy_systems', 0)} · Mobility: {cats.get('mobility', 0)}")
    st.markdown("#### Assign workout")
    with st.expander("Assign workout"):
        wt = st.text_input("Workout title", key=f"assign_title_{player_id}", placeholder="e.g. Lower body strength")
        wtext = st.text_area("Workout details (optional)", key=f"assign_text_{player_id}", placeholder="Paste workout text or leave blank")
        due = st.date_input("Due date", key=f"assign_due_{player_id}", value=date.today() + timedelta(days=7))
        req = st.radio("Type", ["required", "suggested"], key=f"assign_req_{player_id}")
        note = st.text_input("Note to player", key=f"assign_note_{player_id}", placeholder="Optional note")
        if st.button("Assign", key=f"assign_btn_{player_id}"):
            create_assignment(
                team_id, st.session_state.current_user_id, ASSIGNED_TO_PLAYER, player_id,
                workout_title=wt or "Workout", workout_text=wtext, due_date=due.isoformat(),
                required_or_suggested=req, note_from_coach=note,
            )
            st.success("Assigned.")
            st.rerun()
    st.markdown("#### Leave feedback")
    with st.expander("Leave feedback"):
        msg = st.text_area("Message", key=f"feedback_msg_{player_id}", placeholder="Encouragement, correction, focus area...")
        ftype = st.selectbox("Type", FEEDBACK_TYPES, key=f"feedback_type_{player_id}")
        if st.button("Send", key=f"feedback_btn_{player_id}"):
            create_feedback(player_id, st.session_state.current_user_id, msg, feedback_type=ftype)
            st.success("Feedback sent.")
            st.rerun()
    st.markdown("#### Recent feedback")
    for f in get_feedback_for_player(player_id)[:5]:
        st.caption(f"**{f.get('feedback_type', '')}** — {f.get('created_at', '')[:10]}")
        st.write(f.get("message", ""))


# --- Coach: Assignments page ---
# Focus area labels and engine mode mapping for Generate Workout
FOCUS_OPTIONS = [
    ("performance", "Performance"),
    ("skating_mechanics", "Skating Mechanics"),
    ("skills_only", "Puck Mastery"),
    ("energy_systems", "Conditioning"),
    ("mobility", "Mobility Recovery"),
]
DURATION_OPTIONS = [10, 20, 30, 45, 60]
DIFFICULTY_OPTIONS = ["Easy", "Moderate", "Hard"]


def _get_generate_session_fn():
    """Lazy import of streamlit engine generator so Assignments can call it when available."""
    try:
        from ui_streamlit import _generate_via_engine
        return _generate_via_engine
    except Exception:
        return None


def _assignment_target_label(a: dict, load_profile_fn: Callable) -> str:
    to_type = a.get("assigned_to_type", "")
    to_id = (a.get("assigned_to_id") or "").strip()
    if to_type == ASSIGNED_TO_PLAYER and to_id:
        return (load_profile_fn(to_id) or {}).get("display_name") or to_id
    if to_type == ASSIGNED_TO_SUBGROUP and to_id:
        return to_id.capitalize()
    return "Team"


def _assignment_type_label(t: str) -> str:
    if t == ASSIGNMENT_TYPE_GENERATED:
        return "Generated"
    if t == ASSIGNMENT_TYPE_SPECIFIC:
        return "Specific"
    if t == ASSIGNMENT_TYPE_TEAM_FOCUS:
        return "Team focus"
    if t == ASSIGNMENT_TYPE_CHALLENGE:
        return "Challenge"
    return t or "Assignment"


def render_coach_assignments(team_id: str, load_profile_fn: Callable, generate_session_fn=None):
    st.subheader("Assignments")
    t = get_team_by_id(team_id)
    if not t:
        st.info("Share your invite code above so players can join. You can assign workouts once they join your team.")
        return
    gen_fn = generate_session_fn or _get_generate_session_fn()
    players_ext = get_team_players_extended(team_id)
    def _get_players( tid ):
        return players_ext if tid == team_id else []
    # ----- Section 1: Active Assignments -----
    st.markdown("#### Active Assignments")
    all_assignments = get_assignments_for_team(team_id)
    active = [a for a in all_assignments if a.get("is_active", True) and (a.get("status") or "active") == "active"]
    if not active:
        st.caption("No active assignments. Create one below.")
    else:
        for a in reversed(active):
            counts = get_assignment_completion_counts(
                a, team_id, _get_players, profile_loader=load_profile_fn
            )
            title = a.get("workout_title") or "Workout"
            a_type = _assignment_type_label(a.get("assignment_type") or ASSIGNMENT_TYPE_SPECIFIC)
            target = _assignment_target_label(a, load_profile_fn)
            due_str = (a.get("due_date") or "")[:10]
            req = a.get("required_or_suggested") or "required"
            completed = counts.get("completed", 0)
            pending = counts.get("pending", 0)
            overdue = counts.get("overdue", 0)
            total = counts.get("total", 0)
            pct = counts.get("completion_pct", 0)
            created = (a.get("created_at") or "")[:10]
            with st.container():
                row = st.columns([3, 1, 1, 1])
                with row[0]:
                    st.write(f"**{title}** · {a_type} · To: **{target}**")
                    st.caption(f"Due {due_str or '—'} · {req} · Created {created}")
                with row[1]:
                    st.metric("Done", f"{completed}/{total}" if total else "—")
                with row[2]:
                    st.metric("Pending", pending)
                with row[3]:
                    if overdue:
                        st.metric("Overdue", overdue)
                    else:
                        st.metric("Completion", f"{pct}%")
                aid = a.get("assignment_id", "")
                if st.button("View", key=f"assign_view_{aid}"):
                    st.session_state[f"assign_detail_{aid}"] = True
                    st.rerun()
                if st.session_state.get(f"assign_detail_{aid}"):
                    with st.expander("Details", expanded=True):
                        st.text_area("Workout", value=a.get("workout_text") or "(no details)", key=f"assign_text_{aid}", height=120, disabled=True)
                        if st.button("Close", key=f"assign_close_{aid}"):
                            st.session_state[f"assign_detail_{aid}"] = False
                            st.rerun()
    st.divider()
    # ----- Section 2: Create Assignment -----
    st.markdown("#### Create Assignment")
    create_choice = st.radio(
        "How do you want to create an assignment?",
        ["Generate Workout", "Assign Specific Workout", "Set Team Focus", "Create Team Challenge"],
        key="assign_create_choice",
        horizontal=True,
    )
    if create_choice == "Generate Workout":
        _render_create_generate_workout(team_id, load_profile_fn, players_ext, gen_fn)
    elif create_choice == "Assign Specific Workout":
        _render_create_assign_specific(team_id, load_profile_fn, players_ext)
    elif create_choice == "Set Team Focus":
        _render_create_team_focus(team_id, load_profile_fn, players_ext)
    else:
        st.caption("Team challenges (e.g. complete 3 puck mastery workouts this week) — coming soon.")
    st.divider()
    # ----- Section 3: Assignment History -----
    st.markdown("#### Assignment History")
    inactive = [a for a in all_assignments if not a.get("is_active", True) or (a.get("status") or "") not in ("", "active")]
    if not inactive:
        st.caption("No past assignments.")
    else:
        for a in reversed(inactive[-20:]):
            counts = get_assignment_completion_counts(a, team_id, _get_players, profile_loader=load_profile_fn)
            title = a.get("workout_title") or "Workout"
            target = _assignment_target_label(a, load_profile_fn)
            due_str = (a.get("due_date") or "")[:10]
            completed = counts.get("completed", 0)
            total = counts.get("total", 0)
            pct = counts.get("completion_pct", 0)
            st.write(f"**{title}** · To: {target} · Due {due_str} · Completed {completed}/{total} ({pct}%)")
            aid = a.get("assignment_id", "")
            if st.button("Duplicate", key=f"hist_dup_{aid}"):
                _duplicate_assignment(a, team_id)
                st.rerun()


def _duplicate_assignment(a: dict, team_id: str) -> None:
    uid = st.session_state.get("current_user_id", "")
    create_assignment(
        team_id, uid,
        a.get("assigned_to_type", ASSIGNED_TO_TEAM),
        a.get("assigned_to_id", team_id),
        workout_title=a.get("workout_title", "Workout"),
        workout_text=a.get("workout_text", ""),
        due_date=(date.today() + timedelta(days=7)).isoformat(),
        required_or_suggested=a.get("required_or_suggested", "required"),
        note_from_coach=a.get("note_from_coach", ""),
        assignment_type=a.get("assignment_type", ASSIGNMENT_TYPE_SPECIFIC),
        workout_id=a.get("workout_id"),
        generated_session_id=a.get("generated_session_id"),
        focus_primary=a.get("focus_primary") or None,
        focus_secondary=a.get("focus_secondary") or None,
    )


def _render_create_generate_workout(team_id: str, load_profile_fn: Callable, players_ext: list, gen_fn: Callable | None) -> None:
    st.caption("Use Bender to generate a workout, then assign it to the team, a subgroup, or a player.")
    if not gen_fn:
        st.warning("Workout generator is not available in this context. Use the main Bender app to generate, or assign a specific workout below.")
    target_type = st.radio("Assign to", ["Entire Team", "Forwards", "Defense", "Goalies", "Individual Player"], key="gen_assign_to")
    if target_type == "Individual Player":
        if not players_ext:
            st.caption("No players on team yet.")
            pid = None
        else:
            pid = st.selectbox("Player", [p.get("user_id") for p in players_ext], format_func=lambda x: (load_profile_fn(x) or {}).get("display_name") or x, key="gen_player")
    else:
        pid = None
    focus_label = st.selectbox("Focus area", [label for _, label in FOCUS_OPTIONS], key="gen_focus")
    focus_mode = next(m for m, label in FOCUS_OPTIONS if label == focus_label)
    duration = st.selectbox("Duration (min)", DURATION_OPTIONS, key="gen_duration")
    difficulty = st.selectbox("Difficulty", DIFFICULTY_OPTIONS, key="gen_difficulty")
    due = st.date_input("Due date", value=date.today() + timedelta(days=7), key="gen_due")
    required = st.radio("Required or suggested", ["required", "suggested"], key="gen_req")
    note = st.text_input("Coach note (optional)", key="gen_note")
    if st.button("Generate workout", key="gen_btn"):
        payload = {
            "mode": focus_mode,
            "minutes": duration,
            "age": 14,
            "athlete_id": f"team_{team_id}",
        }
        try:
            with st.spinner("Generating..."):
                resp = gen_fn(payload)
            out_text = (resp or {}).get("output_text", "")
            if not out_text or not out_text.strip():
                st.error("Generation returned no content.")
            else:
                st.session_state.assign_gen_preview = out_text
                st.session_state.assign_gen_params = {
                    "team_id": team_id,
                    "focus_mode": focus_mode,
                    "duration": duration,
                    "due": due,
                    "required": required,
                    "note": note,
                    "target_type": target_type,
                    "pid": pid,
                    "session_id": (resp or {}).get("session_id", ""),
                }
                st.rerun()
        except Exception as e:
            st.error(str(e))
    if st.session_state.get("assign_gen_preview"):
        st.markdown("**Preview**")
        st.text_area("Generated workout", value=st.session_state.assign_gen_preview, height=200, key="gen_preview_area", disabled=True)
        params = st.session_state.get("assign_gen_params") or {}
        if st.button("Assign this workout", key="gen_assign_btn"):
            uid = st.session_state.get("current_user_id", "")
            if params.get("target_type") == "Individual Player" and params.get("pid"):
                to_type, to_id = ASSIGNED_TO_PLAYER, params["pid"]
            elif params.get("target_type") == "Entire Team":
                to_type, to_id = ASSIGNED_TO_TEAM, team_id
            elif params.get("target_type") == "Forwards":
                to_type, to_id = ASSIGNED_TO_SUBGROUP, "forwards"
            elif params.get("target_type") == "Defense":
                to_type, to_id = ASSIGNED_TO_SUBGROUP, "defense"
            elif params.get("target_type") == "Goalies":
                to_type, to_id = ASSIGNED_TO_SUBGROUP, "goalies"
            else:
                to_type, to_id = ASSIGNED_TO_TEAM, team_id
            create_assignment(
                team_id, uid, to_type, to_id,
                workout_title=f"{next((label for m, label in FOCUS_OPTIONS if m == params.get('focus_mode')), params.get('focus_mode', ''))} — {params.get('duration', 30)} min",
                workout_text=st.session_state.assign_gen_preview,
                workout_params={"duration": params.get("duration"), "focus": params.get("focus_mode"), "session_id": params.get("session_id")},
                due_date=params.get("due", date.today()).isoformat() if hasattr(params.get("due"), "isoformat") else (params.get("due") or ""),
                required_or_suggested=params.get("required", "required"),
                note_from_coach=params.get("note", ""),
                assignment_type=ASSIGNMENT_TYPE_GENERATED,
                generated_session_id=params.get("session_id", ""),
            )
            st.session_state.assign_gen_preview = None
            st.session_state.assign_gen_params = None
            st.success("Assignment created.")
            st.rerun()
        if st.button("Regenerate", key="gen_regen_btn"):
            st.session_state.assign_gen_preview = None
            st.session_state.assign_gen_params = None
            st.rerun()


def _render_create_assign_specific(team_id: str, load_profile_fn: Callable, players_ext: list) -> None:
    st.caption("Choose an existing workout from the library and assign it.")
    target_type = st.radio("Assign to", ["Team", "Forwards", "Defense", "Goalies", "Individual Player"], key="spec_assign_to")
    if target_type == "Individual Player":
        pid = st.selectbox("Player", [p.get("user_id") for p in players_ext], format_func=lambda x: (load_profile_fn(x) or {}).get("display_name") or x, key="spec_player") if players_ext else None
    else:
        pid = None
    wt = st.text_input("Workout title", key="spec_title", placeholder="e.g. Puck Mastery Session")
    wtext = st.text_area("Workout details", key="spec_text", placeholder="Paste or describe the workout.")
    due = st.date_input("Due date", value=date.today() + timedelta(days=7), key="spec_due")
    required = st.radio("Required or suggested", ["required", "suggested"], key="spec_req")
    note = st.text_input("Coach note (optional)", key="spec_note")
    if st.button("Create assignment", key="spec_btn"):
        uid = st.session_state.get("current_user_id", "")
        if target_type == "Individual Player" and pid:
            to_type, to_id = ASSIGNED_TO_PLAYER, pid
        elif target_type == "Team":
            to_type, to_id = ASSIGNED_TO_TEAM, team_id
        elif target_type == "Forwards":
            to_type, to_id = ASSIGNED_TO_SUBGROUP, "forwards"
        elif target_type == "Defense":
            to_type, to_id = ASSIGNED_TO_SUBGROUP, "defense"
        elif target_type == "Goalies":
            to_type, to_id = ASSIGNED_TO_SUBGROUP, "goalies"
        else:
            to_type, to_id = ASSIGNED_TO_TEAM, team_id
        create_assignment(
            team_id, uid, to_type, to_id,
            workout_title=wt or "Workout", workout_text=wtext,
            due_date=due.isoformat(), required_or_suggested=required, note_from_coach=note,
            assignment_type=ASSIGNMENT_TYPE_SPECIFIC,
        )
        st.success("Assignment created.")
        st.rerun()


def _render_create_team_focus(team_id: str, load_profile_fn: Callable, players_ext: list) -> None:
    st.caption("Set a team or player focus so Bender can suggest relevant workouts and Signals can track priorities.")
    primary = st.selectbox("Primary focus", [label for _, label in FOCUS_OPTIONS], key="focus_primary")
    secondary = st.selectbox("Secondary focus (optional)", ["(none)"] + [label for _, label in FOCUS_OPTIONS], key="focus_secondary")
    target = st.radio("Apply to", ["Entire team", "Individual player"], key="focus_target")
    player_id = None
    if target == "Individual player" and players_ext:
        player_id = st.selectbox("Player", [p.get("user_id") for p in players_ext], format_func=lambda x: (load_profile_fn(x) or {}).get("display_name") or x, key="focus_player")
    start = st.date_input("Start", value=date.today(), key="focus_start")
    end = st.date_input("End (optional)", value=None, key="focus_end")
    if st.button("Set focus", key="focus_btn"):
        uid = st.session_state.get("current_user_id", "")
        primary_key = next(m for m, label in FOCUS_OPTIONS if label == primary)
        secondary_key = None
        if secondary and secondary != "(none)":
            secondary_key = next(m for m, label in FOCUS_OPTIONS if label == secondary)
        end_date = end.isoformat() if end and hasattr(end, "isoformat") else None
        set_team_focus(
            team_id, uid, primary_key,
            secondary_focus=secondary_key,
            player_id=player_id,
            start_date=start.isoformat() if hasattr(start, "isoformat") else None,
            end_date=end_date,
        )
        st.success("Team focus set. Players will see this priority and Bender can use it for suggestions.")
        st.rerun()


# --- Coach: Feedback page ---
def render_coach_feedback(team_id: str, load_profile_fn: Callable):
    st.subheader("Feedback")
    t = get_team_by_id(team_id)
    if not t:
        st.info("Share your invite code above so players can join. Feedback you leave will appear here.")
        return
    feed = get_feedback_for_team(team_id, 30)
    if not feed:
        st.info("No feedback yet. Visit the Roster tab, click View on a player, and leave feedback from their profile.")
    for f in feed:
        pname = (load_profile_fn(f.get("player_id")) or {}).get("display_name") or f.get("player_id")
        st.caption(f"To **{pname}** · {f.get('feedback_type', '')} · {f.get('created_at', '')[:10]}")
        st.write(f.get("message", ""))


# --- NHL combine bests (for Team Performance comparison) ---
def _load_nhl_combine_bests() -> dict:
    """Load NHL combine results and return best (or best-for-lower) per test key. Keys: vertical_jump, agility_5_10_5, pull_ups."""
    out = {}
    try:
        path = Path(__file__).resolve().parent / "data" / "nhl_combine" / "nhl_combine_results.json"
        if not path.exists():
            return out
        with open(path, encoding="utf-8") as f:
            import json
            data = json.load(f)
    except Exception:
        return out
    if not isinstance(data, list):
        return out

    def _num(val):
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    vj_vals = []
    agil_vals = []
    pull_vals = []
    for row in data:
        v = _num(row.get("vertical_jump_in"))
        if v is not None:
            vj_vals.append(v)
        al, ar = _num(row.get("pro_agility_left_sec")), _num(row.get("pro_agility_right_sec"))
        for t in (al, ar):
            if t is not None:
                agil_vals.append(t)
        p = _num(row.get("pull_ups"))
        if p is not None:
            pull_vals.append(p)
    if vj_vals:
        out["vertical_jump"] = max(vj_vals)
    if agil_vals:
        out["agility_5_10_5"] = min(agil_vals)
    if pull_vals:
        out["pull_ups"] = max(pull_vals)
    return out


# --- Coach: Team Performance page ---
def render_coach_team_performance(team_id: str, load_profile_fn: Callable):
    st.subheader("Team Performance")
    t = get_team_by_id(team_id)
    if not t:
        st.info("Share your invite code above so players can join. Performance data will appear here once players are training.")
        return
    players = get_team_players_extended(team_id)
    if not players:
        st.info("Share your invite code above so players can join. Performance data will appear here once players are training.")
        return

    try:
        from bender_leveling import ensure_leveling_defaults, get_category_progress
    except Exception:
        st.info("Leveling data is not available right now.")
        return

    # Load profiles and cache basic data
    roster: list[dict] = []
    for m in players:
        uid = m.get("user_id")
        prof = load_profile_fn(uid) or {}
        prof = ensure_leveling_defaults(prof)
        name = (prof.get("display_name") or uid) or uid
        tests = prof.get("performance_tests") or {}
        roster.append({"user_id": uid, "name": name, "profile": prof, "tests": tests})

    # --- Performance tests: team averages ---
    st.markdown("#### Performance tests (team averages)")

    def _parse_float(val) -> float | None:
        if val is None:
            return None
        s = str(val).strip()
        if not s:
            return None
        # Take first token to ignore units like \"in\" or \"s\"
        token = s.split()[0]
        try:
            return float(token)
        except ValueError:
            return None

    test_defs = [
        ("Vertical Jump (in)", "vertical_jump", "higher"),
        ("5-10-5 Agility (s)", "agility_5_10_5", "lower"),
        ("Pull-ups", "pull_ups", "higher"),
        ("Shooting Test", "shooting_tests", "higher"),
        ("Stickhandling Tests", "stickhandling_tests", "higher"),
        ("Conditioning Test", "conditioning_test", "lower"),
    ]
    nhl_bests = _load_nhl_combine_bests()
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.caption("**Test**")
    with c2:
        st.caption("**Team average**")
    with c3:
        st.caption("**NHL combine best**")
    for label, key, _dir in test_defs:
        vals: list[float] = []
        for row in roster:
            v = _parse_float(row["tests"].get(key))
            if v is not None:
                vals.append(v)
        with c1:
            st.write(label)
        with c2:
            if vals:
                avg = sum(vals) / len(vals)
                st.write(f"{avg:.2f}")
            else:
                st.write("—")
        with c3:
            nhl_val = nhl_bests.get(key)
            if nhl_val is not None:
                st.write(f"{nhl_val:.0f}" if key == "pull_ups" else f"{nhl_val:.2f}")
            else:
                st.write("—")

    # --- Separate section: Player lookup and rankings ---
    st.divider()
    st.markdown("---")
    st.markdown("#### Player lookup & rankings")
    st.caption("Select a player to see their category levels and where their performance test scores rank on the team.")
    st.markdown("")
    name_to_row = {row["name"]: row for row in roster}
    selected_name = st.selectbox(
        "Select a player",
        sorted(name_to_row.keys()),
        key="team_perf_player_select",
    )
    sel_row = name_to_row.get(selected_name)
    if not sel_row:
        return
    prof = sel_row["profile"]
    tests = sel_row["tests"]

    # Show this player's category levels (card format)
    st.markdown(f"**{selected_name} — Category levels**")
    categories = [
        ("puck_mastery", "Puck Mastery"),
        ("skating_mechanics", "Skating Mechanics"),
        ("performance", "Performance (Strength)"),
        ("conditioning", "Conditioning"),
        ("mobility", "Mobility & Recovery"),
    ]
    _card_style = "border: 1px solid rgba(255,255,255,0.2); border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 1rem; background: rgba(255,255,255,0.05);"
    _label_style = "font-weight: 500; color: rgba(255,255,255,0.9);"
    _val_style = "font-weight: 700; color: #fff; font-size: 1.05rem;"
    _row_style_p = "display: flex; justify-content: space-between; align-items: center; padding: 0.35rem 0; border-bottom: 1px solid rgba(255,255,255,0.08);"
    _row_last_p = "display: flex; justify-content: space-between; align-items: center; padding: 0.35rem 0;"
    player_rows = []
    for i, (key, label) in enumerate(categories):
        cp = get_category_progress(prof, key)
        rank = int(cp.get("rank") or 0)
        val = str(rank) if rank else "—"
        style = _row_last_p if i == len(categories) - 1 else _row_style_p
        player_rows.append(f'<div style="{style}"><span style="{_label_style}">{label}</span><span style="{_val_style}">{val}</span></div>')
    st.markdown(f'<div style="{_card_style}">' + "".join(player_rows) + "</div>", unsafe_allow_html=True)

    # Rank this player's performance tests within team
    st.markdown("")
    st.markdown("**Performance test ranking vs team**")
    h1, h2, h3, h4 = st.columns([2, 1, 1, 1])
    with h1:
        st.caption("**Test**")
    with h2:
        st.caption("**Player**")
    with h3:
        st.caption("**Rank**")
    with h4:
        st.caption("**Team best**")

    for label, key, direction in test_defs:
        player_val = _parse_float(tests.get(key))
        all_vals: list[tuple[str, float]] = []
        for row in roster:
            v = _parse_float(row["tests"].get(key))
            if v is not None:
                all_vals.append((row["name"], v))
        with h1:
            st.write(label)
        if player_val is None or not all_vals:
            with h2:
                st.write("—")
            with h3:
                st.write("—")
            with h4:
                st.write("—")
            continue
        # Sort according to better direction
        reverse = direction == "higher"
        all_vals_sorted = sorted(all_vals, key=lambda x: x[1], reverse=reverse)
        names_only = [n for n, _ in all_vals_sorted]
        try:
            idx = names_only.index(selected_name)
        except ValueError:
            idx = None
        with h2:
            st.write(f"{player_val:.2f}")
        with h3:
            if idx is not None:
                rank = idx + 1
                st.write(f"{rank} of {len(all_vals_sorted)}")
            else:
                st.write("—")
        with h4:
            best_name, best_val = all_vals_sorted[0]
            st.write(f"{best_val:.2f} ({best_name})")


# --- Coach: Signals (Smart Coaching Signals placeholder) ---
def _render_coach_signals_placeholder(team_id: str, load_profile_fn: Callable) -> None:
    st.subheader("Signals")
    st.caption("Smart Coaching Signals will appear here: inactive players, missed assignments, recovery neglect, and quick actions (Assign, Set Focus, Feedback).")
    st.info("Coming soon: actionable signals based on team and player activity.")

# --- Coach: Settings (Add Team, Join team, team setup) ---
def _render_coach_settings(team_id: str, load_profile_fn: Callable, save_profile_fn: Callable, uid: str) -> None:
    st.subheader("Settings")
    st.markdown("#### Create another team")
    st.caption("Submit a request to create a new team. An admin will review and approve.")
    _prof = load_profile_fn(uid)
    disp = (_prof or {}).get("display_name") or uid
    with st.form("create_team_form_settings"):
        c_name = st.text_input("Team name", placeholder="e.g. Eagles U14", key="add_team_name")
        c_age = st.text_input("Age group (optional)", placeholder="e.g. U14", key="add_team_age")
        c_level = st.selectbox("Level (optional)", ["", "Youth", "HS", "AA", "AAA", "Junior", "College"], key="add_team_level")
        c_season = st.text_input("Season (optional)", placeholder="e.g. 2024-25", key="add_team_season")
        if st.form_submit_button("Submit request"):
            name = (c_name or "").strip()
            if not name:
                st.error("Enter a team name.")
            elif has_pending_team_request(uid, name):
                st.error("You already have a pending request for this team name. Wait for admin approval before submitting again.")
            elif create_team_request(uid, disp, name, age_group=c_age or "", level=c_level or "", season=c_season or ""):
                st.success("The request has been submitted.")
                st.rerun()
    st.divider()
    st.markdown("#### Join a team")
    st.caption("Enter an invite code to join another team as a player.")
    join_code = st.text_input("Invite code", key="teams_join_code_coach", placeholder="e.g. ABC123").strip().upper()
    if st.button("Join team", key="teams_join_btn_coach"):
        if not join_code:
            st.error("Enter an invite code.")
        else:
            t = get_team_by_invite_code(join_code)
            if not t:
                st.error("Invalid invite code.")
            else:
                if add_member_to_team(t["team_id"], uid, "player"):
                    prof = _add_team_to_player_profile(load_profile_fn(uid) or {}, t)
                    save_profile_fn(prof)
                    if st.session_state.get("current_user_id") == uid and "current_profile" in st.session_state:
                        st.session_state.current_profile = prof
                    st.success(f"You joined **{t.get('team_name', 'team')}**.")
                    st.rerun()
                else:
                    st.info("You're already on this team.")


# --- Main coach dashboard router ---
def render_bender_teams_coach(
    load_profile_fn: Callable,
    save_profile_fn: Callable,
    generate_session_fn: Callable | None = None,
):
    """Main entry for Bender Teams coach view. Renders team selector + sub-pages."""
    uid = st.session_state.current_user_id
    coached = get_teams_coached_by(uid, load_profile_fn)
    if not coached:
        st.subheader("Bender Teams")
        st.caption("Create a team to get started.")
        render_team_creation(load_profile_fn, save_profile_fn)
        return
    team_ids = [t["team_id"] for t in coached]
    def _team_label(t):
        name = t.get("team_name", t["team_id"])
        code = (t.get("invite_code") or "").strip()
        return f"{name} - Invite code: {code}" if code else name
    team_names = [_team_label(t) for t in coached]

    if "bender_teams_team_idx" not in st.session_state:
        st.session_state.bender_teams_team_idx = 0
    n_teams = len(team_ids)
    sel_idx = st.selectbox(
        "Team",
        range(n_teams),
        index=min(st.session_state.bender_teams_team_idx, n_teams - 1) if n_teams else 0,
        format_func=lambda i: team_names[i],
        key="bender_teams_team_select",
    )
    st.session_state.bender_teams_team_idx = sel_idx
    team_id = team_ids[sel_idx]
    sub = st.session_state.get("bender_teams_sub", "Overview")
    opts = ["Overview", "Roster", "Team Performance", "Assignments", "Signals", "Feedback", "Settings"]
    st.markdown('<div id="bender-teams-sub-tab-bar" data-tab-style="classic" aria-hidden="true"></div>', unsafe_allow_html=True)
    _tab_cols = st.columns(len(opts))
    for _i, o in enumerate(opts):
        with _tab_cols[_i]:
            _is_sel = sub == o
            if st.button(o, key=f"teams_sub_{o.replace(' ', '_')}", type="primary" if _is_sel else "secondary"):
                st.session_state.bender_teams_sub = o
                st.rerun()
    st.session_state.bender_teams_sub = sub
    if sub == "Overview":
        render_coach_overview(team_id, load_profile_fn)
    elif sub == "Roster":
        player_view = st.session_state.get("bender_teams_roster_player")
        if player_view:
            if st.button("← Back to roster"):
                st.session_state.bender_teams_roster_player = None
                st.rerun()
            render_coach_player_profile(team_id, player_view, load_profile_fn)
        else:
            def on_select(pid):
                st.session_state.bender_teams_roster_player = pid
            render_coach_roster(team_id, load_profile_fn, save_profile_fn, on_select)
    elif sub == "Team Performance":
        render_coach_team_performance(team_id, load_profile_fn)
    elif sub == "Assignments":
        render_coach_assignments(team_id, load_profile_fn, generate_session_fn)
    elif sub == "Signals":
        _render_coach_signals_placeholder(team_id, load_profile_fn)
    elif sub == "Feedback":
        render_coach_feedback(team_id, load_profile_fn)
    elif sub == "Settings":
        _render_coach_settings(team_id, load_profile_fn, save_profile_fn, uid)


# --- Player: Join flow ---
def handle_join_flow(join_code: str, current_user_id: str, load_profile_fn: Callable, save_profile_fn: Callable[[dict], None] | None = None) -> tuple[bool, str]:
    """Handle ?join=CODE. Returns (success, message). Updates profile with bender_team_ids and team when join succeeds."""
    if not join_code or not current_user_id:
        return False, ""
    t = get_team_by_invite_code(join_code)
    if not t:
        return False, "Invalid invite code."
    if add_member_to_team(t["team_id"], current_user_id, "player"):
        if save_profile_fn:
            prof = _add_team_to_player_profile(load_profile_fn(current_user_id) or {}, t)
            save_profile_fn(prof)
        return True, f"You joined **{t.get('team_name', 'team')}**."
    return False, "You're already on this team."


# --- Assignment category display (from workout_params or title) ---
_ASSIGNMENT_CATEGORIES = {
    "performance": "Performance",
    "skating_mechanics": "Skating Mechanics",
    "stickhandling": "Puck Mastery",
    "shooting": "Puck Mastery",
    "skills_only": "Puck Mastery",
    "energy_systems": "Conditioning",
    "mobility": "Mobility Recovery",
}


def _assignment_category(a: dict) -> str:
    params = a.get("workout_params") or {}
    mode = (params.get("mode") or "").strip().lower()
    if mode:
        return _ASSIGNMENT_CATEGORIES.get(mode, mode.replace("_", " ").title())
    title = (a.get("workout_title") or "").lower()
    for key, label in _ASSIGNMENT_CATEGORIES.items():
        if key in title:
            return label
    return "Workout"


# --- Player Portal: full team locker room / development board ---
def render_bender_teams_player_portal(
    user_id: str,
    team: dict,
    load_profile_fn: Callable,
    save_profile_fn: Callable,
):
    """Player portal: tabs for Assignments, Team Progress, Coach Feedback."""
    team_id = team.get("team_id", "")
    loader = _profile_loader(load_profile_fn)
    profile = load_profile_fn(user_id) or {}
    team_name = team.get("team_name", "Team")

    # Header
    st.markdown(f"### {team_name}")
    st.caption("Your team locker room · Stay on track with your coach")
    # Coach priority (team focus) when set
    focus = get_team_focus(team_id)
    if focus:
        primary = focus.get("primary_focus", "")
        secondary = focus.get("secondary_focus", "")
        def _focus_label(key):
            return next((label for m, label in FOCUS_OPTIONS if m == key), (key or "").replace("_", " ").title())
        parts = [_focus_label(primary)] if primary else []
        if secondary:
            parts.append(_focus_label(secondary))
        if parts:
            st.info(f"**Coach priority this week:** {' · '.join(parts)}")

    # Sub-tabs
    opts = ["Assignments", "Team Progress", "Coach Feedback"]
    sub = st.session_state.get("bender_teams_player_sub", "Assignments")
    st.markdown('<div id="bender-teams-sub-tab-bar" data-tab-style="classic" aria-hidden="true"></div>', unsafe_allow_html=True)
    _tab_cols = st.columns(len(opts))
    for _i, o in enumerate(opts):
        with _tab_cols[_i]:
            _is_sel = sub == o
            if st.button(o, key=f"teams_player_sub_{o.replace(' ', '_')}", type="primary" if _is_sel else "secondary"):
                st.session_state.bender_teams_player_sub = o
                st.rerun()
    sub = st.session_state.get("bender_teams_player_sub", "Assignments")
    st.divider()

    if sub == "Assignments":
        _render_player_assignments_tab(user_id, team_id, load_profile_fn, save_profile_fn)
    elif sub == "Team Progress":
        _render_player_team_progress_tab(user_id, team, profile, loader, team_name, load_profile_fn, save_profile_fn)
    elif sub == "Coach Feedback":
        _render_player_feedback_tab(user_id, load_profile_fn)


def _render_player_assignments_tab(
    user_id: str,
    team_id: str,
    load_profile_fn: Callable,
    save_profile_fn: Callable,
) -> None:
    """Assignments tab: Coach Assignments + Suggested Workouts combined."""
    st.markdown("#### Coach Assignments & Suggested Workouts")
    # Show coach focus again in context of assignments
    focus = get_team_focus(team_id)
    if focus:
        primary = focus.get("primary_focus", "")
        secondary = focus.get("secondary_focus", "")
        def _fl(key):
            return next((label for m, label in FOCUS_OPTIONS if m == key), (key or "").replace("_", " ").title())
        parts = [_fl(primary)] if primary else []
        if secondary:
            parts.append(_fl(secondary))
        if parts:
            st.caption(f"Coach priority: {' · '.join(parts)}")
    assignments = get_assignments_for_player(user_id, team_id)
    required = [a for a in assignments if (a.get("required_or_suggested") or "required") == "required"]
    suggested_assignments = [a for a in assignments if (a.get("required_or_suggested") or "") == "suggested"]
    completed_ids = set()
    for a in assignments:
        completed_ids.update(a.get("completed_by") or [])

    if not assignments:
        st.caption("No assignments yet. Your coach will assign workouts here.")
    else:
        for a in required:
            _render_assignment_card(a, user_id, load_profile_fn, save_profile_fn, completed_ids)
        if suggested_assignments:
            st.markdown("**Suggested by coach**")
            for a in suggested_assignments:
                _render_assignment_card(a, user_id, load_profile_fn, save_profile_fn, completed_ids)


def _render_player_team_progress_tab(
    user_id: str,
    team: dict,
    profile: dict,
    loader: Callable,
    team_name: str,
    load_profile_fn: Callable,
    save_profile_fn: Callable | None = None,
) -> None:
    """Team Progress tab: Your Progress + Team Activity + Team Info at bottom."""
    team_id = team.get("team_id", "")
    st.markdown("#### Your Progress")
    summary = get_team_activity_summary(team_id, loader)
    my_act = get_player_activity_summary(profile, 7)
    workouts_this_week = my_act.get("workouts_this_period", 0)
    team_avg = (summary.get("workouts_this_week", 0) / summary.get("total_players", 1)) if summary.get("total_players") else 0
    streak = my_act.get("streak", 0)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Workouts this week", workouts_this_week)
    with c2:
        st.metric("Team average", f"{team_avg:.1f}")
    with c3:
        st.metric("Current streak", f"{streak} days")

    # Category progress from completion_history
    hist = profile.get("completion_history") or []
    cats = {}
    for e in hist:
        m = (e.get("mode") or "").lower()
        if m in _ASSIGNMENT_CATEGORIES:
            label = _ASSIGNMENT_CATEGORIES[m]
            cats[label] = cats.get(label, 0) + 1
        else:
            cats["Other"] = cats.get("Other", 0) + 1
    if cats:
        st.caption("Category overview (all time)")
        for label in ["Performance", "Skating Mechanics", "Puck Mastery", "Conditioning", "Mobility Recovery"]:
            count = cats.get(label, 0)
            st.progress(min(1.0, count / 15), text=f"{label}: {count} workouts")

    st.divider()

    st.markdown("#### Team Activity")
    feed = get_recent_team_activity(team_id, loader, 15)
    if not feed:
        st.caption("No recent activity yet.")
    else:
        for item in feed[:12]:
            tpe = item.get("type", "")
            if tpe == "workout_completed":
                name = item.get("display_name", "Teammate")
                mode = (item.get("mode") or "workout").replace("_", " ").title()
                st.caption(f"· **{name}** completed a {mode} workout")
            elif tpe == "assignment":
                st.caption(f"· Coach assigned **{item.get('workout_title', 'a workout')}**")
            elif tpe == "feedback":
                st.caption("· Coach left feedback for a player")

    st.divider()

    st.markdown("#### Team Info")
    coach_name = team.get("coach_name") or (load_profile_fn(team.get("coach_user_id")) or {}).get("display_name") or "—"
    season = team.get("season", "") or "—"
    pos = profile.get("position", "") or "—"
    st.caption(f"**Team:** {team_name}")
    st.caption(f"**Coach:** {coach_name}")
    st.caption(f"**Season:** {season}")
    st.caption(f"**Your position:** {pos}")
    if save_profile_fn:
        if st.button("Leave team", key="player_leave_team_btn"):
            remove_member_from_team(team_id, user_id)
            remaining = [c for c in (profile.get("player_teams_cache") or []) if c.get("team_id") != team_id]
            prof = _remove_team_from_player_profile(load_profile_fn(user_id) or {}, team_id, remaining)
            save_profile_fn(prof)
            if st.session_state.get("current_user_id") == user_id and "current_profile" in st.session_state:
                st.session_state.current_profile = prof
            st.session_state.bender_teams_player_sub = "Assignments"
            st.rerun()


def _render_player_feedback_tab(user_id: str, load_profile_fn: Callable) -> None:
    st.markdown("#### Coach Feedback")
    feedbacks = get_feedback_for_player(user_id)
    if not feedbacks:
        st.caption("No feedback yet. Keep training — your coach may leave notes after workouts.")
    else:
        for f in feedbacks[:10]:
            coach_name = (load_profile_fn(f.get("coach_id")) or {}).get("display_name") or f.get("coach_id", "Coach")
            ts = (f.get("created_at") or "")[:10]
            st.markdown(f"**{f.get('feedback_type', 'Note').replace('_', ' ').title()}** — {coach_name} · {ts}")
            st.write(f.get("message", ""))
            st.caption("---")


def _render_assignment_card(
    a: dict,
    user_id: str,
    load_profile_fn: Callable,
    save_profile_fn: Callable,
    completed_ids: set,
) -> None:
    req = a.get("required_or_suggested", "required")
    is_required = req == "required"
    done = user_id in (a.get("completed_by") or [])
    due = a.get("due_date", "")
    overdue = False
    if due and not done:
        try:
            overdue = date.fromisoformat(due[:10]) < date.today()
        except (ValueError, TypeError):
            pass
    with st.container():
        badge = "Required" if is_required else "Suggested"
        due_str = f" · Due {due[:10]}" if due else ""
        if overdue:
            due_str += " · **Overdue**"
        st.markdown(f"**{a.get('workout_title', 'Workout')}** — {badge}{due_str}")
        st.caption(_assignment_category(a))
        if a.get("note_from_coach"):
            st.caption(f"Coach note: {a['note_from_coach']}")
        if a.get("workout_text"):
            with st.expander("View workout"):
                st.markdown((a.get("workout_text") or "")[:3000])
        if done:
            st.caption("Completed")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Start workout", key=f"assign_btn_{a.get('assignment_id', '')}"):
                    st.session_state.player_tab = "Training Session"
                    st.rerun()
            with col_b:
                if st.button("Mark complete", key=f"assign_done_{a.get('assignment_id', '')}"):
                    mark_assignment_completed(a.get("assignment_id"), user_id)
                    st.rerun()


# --- Player: Assigned workouts + feedback display ---
def render_player_assignments_and_feedback(
    user_id: str,
    team_id: str | None,
    load_profile_fn: Callable,
    on_complete_workout_callback=None,
):
    """Render assigned workouts and coach feedback for player in their main view."""
    teams = get_teams_for_user(user_id, load_profile_fn)
    if not teams:
        return
    team_id = team_id or (teams[0]["team_id"] if teams else None)
    assignments = get_assignments_for_player(user_id, team_id)
    feedbacks = get_feedback_for_player(user_id)
    if not assignments and not feedbacks:
        return
    st.markdown("### Assigned to you")
    for a in assignments:
        due = a.get("due_date", "")
        req = a.get("required_or_suggested", "required")
        badge = "Required" if req == "required" else "Suggested"
        st.markdown(f"**{a.get('workout_title', 'Workout')}** — {badge}" + (f" · Due {due[:10]}" if due else ""))
        if a.get("note_from_coach"):
            st.caption(f"Coach: {a['note_from_coach']}")
        if a.get("workout_text"):
            with st.expander("View workout"):
                st.markdown(a["workout_text"][:2000] + ("..." if len(a.get("workout_text", "")) > 2000 else ""))
    if feedbacks:
        st.markdown("### Coach feedback")
        for f in feedbacks[:5]:
            st.caption(f"**{f.get('feedback_type', '').replace('_', ' ').title()}** — {f.get('created_at', '')[:10]}")
            st.write(f.get("message", ""))
