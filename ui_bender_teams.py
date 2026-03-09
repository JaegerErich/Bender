"""
Bender Teams UI: Coach dashboard, roster, assignments, feedback.
Rendered as additive tabs/views. Does not replace existing player experience.
"""
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

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
        get_teams_coached_by,
        get_teams_for_user,
        get_player_activity_summary,
        get_team_activity_summary,
        get_recent_team_activity,
        is_team_coach,
        mark_assignment_completed,
        FEEDBACK_TYPES,
        ASSIGNED_TO_TEAM,
        ASSIGNED_TO_PLAYER,
    )
except ImportError:
    # Graceful fallback if module missing
    def _noop(*a, **k):
        return [] if "team" in str(a) + str(k) else None
    add_member_to_team = create_assignment = create_feedback = create_team = create_team_request = _noop
    has_pending_team_request = lambda *a: False
    get_assignments_for_player = get_assignments_for_team = get_feedback_for_player = get_feedback_for_team = lambda *a: []
    get_team_by_id = get_team_by_invite_code = lambda *a: None
    get_team_members = get_team_players = get_teams_for_user = get_teams_coached_by = lambda *a: []
    get_player_activity_summary = get_team_activity_summary = get_recent_team_activity = lambda *a: {}
    is_team_coach = lambda *a: False
    mark_assignment_completed = lambda *a: False
    FEEDBACK_TYPES = ()
    ASSIGNED_TO_TEAM = ASSIGNED_TO_PLAYER = ""


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
                        prof = load_profile_fn(uid) or {}
                        ids = list(prof.get("bender_team_ids") or [])
                        if t["team_id"] not in ids:
                            ids.append(t["team_id"])
                        prof["bender_team_ids"] = ids
                        prof["team"] = t.get("team_name", "").strip()
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
                        prof = load_profile_fn(uid) or {}
                        ids = list(prof.get("bender_team_ids") or [])
                        if t["team_id"] not in ids:
                            ids.append(t["team_id"])
                        prof["bender_team_ids"] = ids
                        prof["team"] = t.get("team_name", "").strip()
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
    players = get_team_players(team_id)
    if not players:
        st.caption(t.get("team_name", "Team"))
        st.info("Share your invite code above so players can join. Activity and metrics will appear here once they start training.")
        return
    st.caption(t.get("team_name", "Team"))
    st.caption(f"{t.get('age_group', '')} {t.get('level', '')} {t.get('season', '')}".strip() or "—")
    st.markdown("")  # spacer
    # Top metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Active this week", f"{summary.get('active_players_this_week', 0)} / {summary.get('total_players', 0)}")
    with m2:
        st.metric("Workouts this week", summary.get("workouts_this_week", 0))
    with m3:
        st.metric("Completion %", f"{summary.get('completion_percentage', 0)}%")
    with m4:
        st.metric("Avg minutes", f"{int(summary.get('avg_minutes_per_player', 0))}")
    # Roster snapshot + attention
    col_roster, col_attention = st.columns([2, 1])
    with col_roster:
        st.markdown("#### Roster snapshot")
        players = get_team_players(team_id)
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
def render_coach_roster(team_id: str, load_profile_fn: Callable, on_select_player: Callable[[str], None]):
    st.subheader("Roster")
    t = get_team_by_id(team_id)
    if not t:
        st.info("Share your invite code above so players can join. The roster will appear here once they join your team.")
        return
    players = get_team_players(team_id)
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
            cols = st.columns([3, 1, 1, 1, 1])
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
            st.divider()


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
def render_coach_assignments(team_id: str, load_profile_fn: Callable, generate_session_fn=None):
    st.subheader("Assignments")
    t = get_team_by_id(team_id)
    if not t:
        st.info("Share your invite code above so players can join. You can assign workouts once they join your team.")
        return
    assignments = get_assignments_for_team(team_id)
    if not assignments:
        st.info("No assignments yet. Assign workouts to your team or individual players using the form below once they join.")
    for a in reversed(assignments[-30:]):
        due = a.get("due_date", "")
        req = a.get("required_or_suggested", "required")
        if a.get("assigned_to_type") == ASSIGNED_TO_PLAYER:
            target = (load_profile_fn(a.get("assigned_to_id", "")) or {}).get("display_name") or a.get("assigned_to_id", "")
        else:
            target = "Full team"
        completed = a.get("completed_by", [])
        st.write(f"**{a.get('workout_title', 'Workout')}** — {req} · Due {due[:10] if due else '—'} · To: {target}")
        st.caption(f"Completed by: {len(completed)}")
    st.divider()
    st.markdown("#### New assignment")
    with st.expander("Assign to team or player"):
        to_type = st.radio("Assign to", ["team", "player"], key="new_assign_to")
        players = get_team_players(team_id)
        if to_type == "player":
            if not players:
                st.caption("No players on team yet. Invite players with your team code first.")
                pid = None
            else:
                pid = st.selectbox("Player", [p.get("user_id") for p in players], format_func=lambda x: (load_profile_fn(x) or {}).get("display_name") or x, key="new_assign_player")
        else:
            pid = None
        wt = st.text_input("Workout title", key="new_assign_title")
        wtext = st.text_area("Workout details", key="new_assign_text")
        due = st.date_input("Due date", key="new_assign_due", value=date.today() + timedelta(days=7))
        req = st.radio("Type", ["required", "suggested"], key="new_assign_req")
        note = st.text_input("Note", key="new_assign_note")
        if st.button("Create assignment", key="new_assign_btn"):
            if to_type == "player" and not pid:
                st.warning("Select a player or assign to full team.")
            else:
                target_id = pid if to_type == "player" and pid else team_id
                create_assignment(
                    team_id, st.session_state.current_user_id,
                    ASSIGNED_TO_PLAYER if to_type == "player" else ASSIGNED_TO_TEAM,
                    target_id,
                    workout_title=wt or "Workout", workout_text=wtext,
                    due_date=due.isoformat(), required_or_suggested=req, note_from_coach=note,
                )
                st.success("Assignment created.")
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


# --- Main coach dashboard router ---
def render_bender_teams_coach(
    load_profile_fn: Callable,
    save_profile_fn: Callable,
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
    team_names = [t.get("team_name", t["team_id"]) for t in coached]
    if "bender_teams_team_idx" not in st.session_state:
        st.session_state.bender_teams_team_idx = 0
    # Team selector row
    sel_idx = st.selectbox("Team", range(len(team_ids)), index=st.session_state.bender_teams_team_idx, format_func=lambda i: team_names[i], key="bender_teams_team_select")
    st.session_state.bender_teams_team_idx = min(sel_idx, len(team_ids) - 1)
    team_id = team_ids[sel_idx]
    # Invite code for current team (coach can share with players)
    _cur_team = coached[sel_idx]
    _invite_code = _cur_team.get("invite_code") or ""
    if _invite_code:
        st.info(f"**Invite code for players:** `{_invite_code}` — Share this code so players can join **{_cur_team.get('team_name', 'your team')}**.")
    sub = st.session_state.get("bender_teams_sub", "Overview")
    opts = ["Overview", "Roster", "Assignments", "Feedback", "Add Team", "Join team"]
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
            render_coach_roster(team_id, load_profile_fn, on_select)
    elif sub == "Assignments":
        render_coach_assignments(team_id, load_profile_fn)
    elif sub == "Feedback":
        render_coach_feedback(team_id, load_profile_fn)
    elif sub == "Add Team":
        _prof = load_profile_fn(uid)
        disp = (_prof or {}).get("display_name") or uid
        st.subheader("Create another team")
        st.caption("Submit a request to create a new team. An admin will review and approve.")
        with st.form("create_team_form_add_tab"):
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
    elif sub == "Join team":
        st.subheader("Join a team")
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
                        prof = load_profile_fn(uid) or {}
                        ids = list(prof.get("bender_team_ids") or [])
                        if t["team_id"] not in ids:
                            ids.append(t["team_id"])
                        prof["bender_team_ids"] = ids
                        prof["team"] = t.get("team_name", "").strip()
                        save_profile_fn(prof)
                        if st.session_state.get("current_user_id") == uid and "current_profile" in st.session_state:
                            st.session_state.current_profile = prof
                        st.success(f"You joined **{t.get('team_name', 'team')}**.")
                        st.rerun()
                    else:
                        st.info("You're already on this team.")


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
            prof = load_profile_fn(current_user_id) or {}
            ids = list(prof.get("bender_team_ids") or [])
            if t["team_id"] not in ids:
                ids.append(t["team_id"])
            prof["bender_team_ids"] = ids
            prof["team"] = t.get("team_name", "").strip()
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
    """Player portal: assignments, suggested workouts, team activity, progress, feedback, team info."""
    team_id = team.get("team_id", "")
    loader = _profile_loader(load_profile_fn)
    profile = load_profile_fn(user_id) or {}

    # Header: team name as portal title
    team_name = team.get("team_name", "Team")
    st.markdown(f"### {team_name}")
    st.caption("Your team locker room · Stay on track with your coach")
    st.divider()

    # --- 1. Coach Assignments (required first, then suggested) ---
    st.markdown("#### Coach Assignments")
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

    st.divider()

    # --- 2. Suggested Workouts (same as suggested in Coach Assignments above) ---
    st.markdown("#### Suggested Workouts")
    if not suggested_assignments:
        st.caption("No suggested workouts right now. Your coach may add recommendations above.")
    else:
        st.caption("Your coach’s recommendations are listed above under **Coach Assignments** with a **Suggested** badge.")

    st.divider()

    # --- 3. Team Activity ---
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

    # --- 4. Your Team Progress ---
    st.markdown("#### Your Team Progress")
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

    # --- 5. Coach Feedback ---
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

    st.divider()

    # --- 6. Team Info ---
    st.markdown("#### Team Info")
    coach_name = team.get("coach_name") or (load_profile_fn(team.get("coach_user_id")) or {}).get("display_name") or "—"
    season = team.get("season", "") or "—"
    pos = profile.get("position", "") or "—"
    st.caption(f"**Team:** {team_name}")
    st.caption(f"**Coach:** {coach_name}")
    st.caption(f"**Season:** {season}")
    st.caption(f"**Your position:** {pos}")


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
    teams = get_teams_for_user(user_id)
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
