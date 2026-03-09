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
        FEEDBACK_TYPES,
        ASSIGNED_TO_TEAM,
        ASSIGNED_TO_PLAYER,
    )
except ImportError:
    # Graceful fallback if module missing
    def _noop(*a, **k):
        return [] if "team" in str(a) + str(k) else None
    add_member_to_team = create_assignment = create_feedback = create_team = create_team_request = _noop
    get_assignments_for_player = get_assignments_for_team = get_feedback_for_player = get_feedback_for_team = lambda *a: []
    get_team_by_id = get_team_by_invite_code = lambda *a: None
    get_team_members = get_team_players = get_teams_for_user = get_teams_coached_by = lambda *a: []
    get_player_activity_summary = get_team_activity_summary = get_recent_team_activity = lambda *a: {}
    is_team_coach = lambda *a: False
    FEEDBACK_TYPES = ()
    ASSIGNED_TO_TEAM = ASSIGNED_TO_PLAYER = ""


def _profile_loader(load_profile_fn: Callable) -> Callable:
    def load(uid):
        return load_profile_fn(uid) if uid else None
    return load


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
                create_team_request(uid, disp, name, age_group=age_group or "", level=level or "", season=season or "")
                st.success("Team creation request submitted. An admin will review it. You'll see your team in Bender Teams once approved.")
                st.rerun()


# --- Coach: Overview dashboard ---
def render_coach_overview(team_id: str, load_profile_fn: Callable):
    t = get_team_by_id(team_id)
    if not t:
        st.warning("Team not found.")
        return
    loader = _profile_loader(load_profile_fn)
    summary = get_team_activity_summary(team_id, loader)
    st.subheader(t.get("team_name", "Team"))
    st.caption(f"{t.get('age_group', '')} {t.get('level', '')} {t.get('season', '')}".strip() or "—")
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
    st.divider()
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
    t = get_team_by_id(team_id)
    if not t:
        st.warning("Team not found.")
        return
    st.subheader("Roster")
    players = get_team_players(team_id)
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
    t = get_team_by_id(team_id)
    if not t:
        st.warning("Team not found.")
        return
    st.subheader("Assignments")
    assignments = get_assignments_for_team(team_id)
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
    t = get_team_by_id(team_id)
    if not t:
        st.warning("Team not found.")
        return
    st.subheader("Feedback")
    feed = get_feedback_for_team(team_id, 30)
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
    coached = get_teams_coached_by(uid)
    if not coached:
        st.subheader("Bender Teams")
        st.caption("Create a team to get started.")
        render_team_creation(load_profile_fn, save_profile_fn)
        return
    team_ids = [t["team_id"] for t in coached]
    team_names = [t.get("team_name", t["team_id"]) for t in coached]
    if "bender_teams_team_idx" not in st.session_state:
        st.session_state.bender_teams_team_idx = 0
    # Team selector row: dropdown + Add team button
    _col_sel, _col_add = st.columns([3, 1])
    with _col_sel:
        sel_idx = st.selectbox("Team", range(len(team_ids)), index=st.session_state.bender_teams_team_idx, format_func=lambda i: team_names[i], key="bender_teams_team_select")
    with _col_add:
        if st.button("+ Add team", key="teams_add_team_btn", help="Create another team (e.g. different age group)"):
            st.session_state.bender_teams_show_create = True
            st.rerun()
    st.session_state.bender_teams_team_idx = min(sel_idx, len(team_ids) - 1)
    team_id = team_ids[sel_idx]
    # Add another team (expander when coach already has teams)
    if st.session_state.get("bender_teams_show_create"):
        with st.expander("Create another team", expanded=True):
            _prof = load_profile_fn(uid)
            disp = (_prof or {}).get("display_name") or uid
            with st.form("create_team_form_existing"):
                c_name = st.text_input("Team name", placeholder="e.g. Eagles U14")
                c_age = st.text_input("Age group (optional)", placeholder="e.g. U14")
                c_level = st.selectbox("Level (optional)", ["", "Youth", "HS", "AA", "AAA", "Junior", "College"])
                c_season = st.text_input("Season (optional)", placeholder="e.g. 2024-25")
                _fc1, _fc2 = st.columns(2)
                with _fc1:
                    if st.form_submit_button("Submit request"):
                        name = (c_name or "").strip()
                        if name:
                            create_team_request(uid, disp, name, age_group=c_age or "", level=c_level or "", season=c_season or "")
                            st.session_state.bender_teams_show_create = False
                            st.success("Team creation request submitted. An admin will review it. You'll see your new team in the dropdown once approved.")
                            st.rerun()
                        else:
                            st.error("Enter a team name.")
                with _fc2:
                    if st.form_submit_button("Cancel"):
                        st.session_state.bender_teams_show_create = False
                        st.rerun()
    sub = st.session_state.get("bender_teams_sub", "Overview")
    opts = ["Overview", "Roster", "Assignments", "Feedback"]
    for o in opts:
        if st.button(o, key=f"teams_sub_{o}", type="primary" if sub == o else "secondary"):
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


# --- Player: Join flow ---
def handle_join_flow(join_code: str, current_user_id: str, load_profile_fn: Callable) -> tuple[bool, str]:
    """Handle ?join=CODE. Returns (success, message)."""
    if not join_code or not current_user_id:
        return False, ""
    t = get_team_by_invite_code(join_code)
    if not t:
        return False, "Invalid invite code."
    if add_member_to_team(t["team_id"], current_user_id, "player"):
        return True, f"You joined **{t.get('team_name', 'team')}**."
    return False, "You're already on this team."


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
