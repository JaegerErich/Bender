"""
Bender Teams: Team and coach management layer.
Adds teams, memberships, assignments, and feedback on top of existing Bender profiles.
Architecture: JSON files in data/ for teams, assignments, feedback. Profiles unchanged.
"""
from datetime import date, datetime, timedelta
from pathlib import Path
import json
import secrets
import string
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TEAMS_PATH = DATA_DIR / "teams.json"
TEAM_REQUESTS_PATH = DATA_DIR / "team_creation_requests.json"
ASSIGNMENTS_PATH = DATA_DIR / "assignments.json"
FEEDBACK_PATH = DATA_DIR / "feedback.json"

# Roles: coach, assistant_coach, player
ROLES = ("coach", "assistant_coach", "player")
COACH_ROLES = ("coach", "assistant_coach")

# Assignment types
ASSIGNED_TO_TEAM = "team"
ASSIGNED_TO_SUBGROUP = "subgroup"
ASSIGNED_TO_PLAYER = "player"

# Feedback types
FEEDBACK_TYPES = ("encouragement", "correction", "focus_area", "recovery_note", "praise")

# Subgroups (used for assignment filtering)
SUBGROUPS = ("forwards", "defense", "goalies", "rehab", "captains")


def _load_json(path: Path, default: list | dict) -> list | dict:
    if not path.exists():
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path: Path, data: list | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _generate_invite_code() -> str:
    """6-char alphanumeric invite code."""
    chars = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(chars) for _ in range(6))


# ---------------------------------------------------------------------------
# Team creation requests (admin approval flow)
# ---------------------------------------------------------------------------


def load_team_requests() -> list[dict]:
    """Load team creation requests. Status: pending, approved, denied."""
    return _load_json(TEAM_REQUESTS_PATH, [])


def save_team_requests(requests: list[dict]) -> None:
    _save_json(TEAM_REQUESTS_PATH, requests)


def create_team_request(
    requester_user_id: str,
    requester_display_name: str,
    team_name: str,
    *,
    age_group: str = "",
    level: str = "",
    season: str = "",
) -> dict:
    """Submit a team creation request. Requires admin approval."""
    requests = load_team_requests()
    req_id = f"tr_{secrets.token_hex(6)}"
    req = {
        "request_id": req_id,
        "requester_user_id": requester_user_id,
        "requester_display_name": requester_display_name,
        "team_name": team_name,
        "age_group": age_group or "",
        "level": level or "",
        "season": season or "",
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "reviewed_at": None,
        "reviewed_by": None,
    }
    requests.append(req)
    save_team_requests(requests)
    return req


def approve_team_request(request_id: str) -> dict | None:
    """Approve request and create the team. Returns the created team or None."""
    requests = load_team_requests()
    for req in requests:
        if req.get("request_id") == request_id and req.get("status") == "pending":
            team = create_team(
                req["team_name"],
                req["requester_user_id"],
                req["requester_display_name"],
                age_group=req.get("age_group", ""),
                level=req.get("level", ""),
                season=req.get("season", ""),
            )
            req["status"] = "approved"
            req["reviewed_at"] = datetime.now().isoformat()
            req["team_id"] = team["team_id"]
            req["invite_code"] = team["invite_code"]
            save_team_requests(requests)
            return team
    return None


def deny_team_request(request_id: str, reviewed_by: str | None = None) -> bool:
    """Mark request as denied."""
    requests = load_team_requests()
    for req in requests:
        if req.get("request_id") == request_id and req.get("status") == "pending":
            req["status"] = "denied"
            req["reviewed_at"] = datetime.now().isoformat()
            req["reviewed_by"] = reviewed_by or ""
            save_team_requests(requests)
            return True
    return False


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------


def load_teams() -> list[dict]:
    return _load_json(TEAMS_PATH, [])


def save_teams(teams: list[dict]) -> None:
    _save_json(TEAMS_PATH, teams)


def create_team(
    team_name: str,
    coach_user_id: str,
    coach_display_name: str,
    *,
    age_group: str = "",
    level: str = "",
    season: str = "",
    assistant_coaches: list[str] | None = None,
) -> dict:
    """Create a new team. Coach is added as first member with role=coach."""
    teams = load_teams()
    team_id = f"t_{secrets.token_hex(8)}"
    invite_code = _generate_invite_code()
    now = datetime.now().isoformat()
    team = {
        "team_id": team_id,
        "team_name": team_name,
        "age_group": age_group or "",
        "level": level or "",
        "season": season or "",
        "coach_user_id": coach_user_id,
        "coach_name": coach_display_name,
        "assistant_coaches": list(assistant_coaches or []),
        "invite_code": invite_code,
        "members": [
            {"user_id": coach_user_id, "role": "coach", "joined_at": now},
        ],
        "created_at": now,
    }
    teams.append(team)
    save_teams(teams)
    return team


def get_team_by_id(team_id: str) -> dict | None:
    for t in load_teams():
        if t.get("team_id") == team_id:
            return t
    return None


def get_team_by_invite_code(code: str) -> dict | None:
    c = (code or "").strip().upper()
    for t in load_teams():
        if (t.get("invite_code") or "").upper() == c:
            return t
    return None


def get_teams_for_user(user_id: str) -> list[dict]:
    """Teams where user is a member (any role)."""
    out = []
    for t in load_teams():
        for m in t.get("members", []):
            if m.get("user_id") == user_id:
                out.append(t)
                break
    return out


def get_teams_coached_by(user_id: str) -> list[dict]:
    """Teams where user is coach or assistant_coach."""
    out = []
    for t in load_teams():
        for m in t.get("members", []):
            if m.get("user_id") == user_id and m.get("role") in COACH_ROLES:
                out.append(t)
                break
    return out


def is_team_coach(user_id: str, team_id: str) -> bool:
    t = get_team_by_id(team_id)
    if not t:
        return False
    for m in t.get("members", []):
        if m.get("user_id") == user_id and m.get("role") in COACH_ROLES:
            return True
    return False


def add_member_to_team(team_id: str, user_id: str, role: str = "player") -> bool:
    """Add user to team. Returns False if already a member."""
    teams = load_teams()
    for t in teams:
        if t.get("team_id") != team_id:
            continue
        members = t.get("members", [])
        for m in members:
            if m.get("user_id") == user_id:
                return False
        members.append({"user_id": user_id, "role": role, "joined_at": datetime.now().isoformat()})
        t["members"] = members
        save_teams(teams)
        return True
    return False


def get_team_players(team_id: str) -> list[dict]:
    """Members with role=player."""
    t = get_team_by_id(team_id)
    if not t:
        return []
    return [m for m in t.get("members", []) if m.get("role") == "player"]


def get_team_members(team_id: str) -> list[dict]:
    t = get_team_by_id(team_id)
    return t.get("members", []) if t else []


# ---------------------------------------------------------------------------
# Assignments
# ---------------------------------------------------------------------------


def load_assignments() -> list[dict]:
    return _load_json(ASSIGNMENTS_PATH, [])


def save_assignments(assignments: list[dict]) -> None:
    _save_json(ASSIGNMENTS_PATH, assignments)


def create_assignment(
    team_id: str,
    assigned_by: str,
    assigned_to_type: str,
    assigned_to_id: str,
    *,
    workout_title: str = "",
    workout_text: str = "",
    workout_params: dict | None = None,
    due_date: str | None = None,
    required_or_suggested: str = "required",
    note_from_coach: str = "",
) -> dict:
    a = {
        "assignment_id": f"a_{secrets.token_hex(8)}",
        "team_id": team_id,
        "workout_title": workout_title or "Workout",
        "workout_text": workout_text or "",
        "workout_params": workout_params or {},
        "assigned_by": assigned_by,
        "assigned_to_type": assigned_to_type,
        "assigned_to_id": assigned_to_id,
        "due_date": due_date or "",
        "required_or_suggested": required_or_suggested,
        "note_from_coach": note_from_coach or "",
        "created_at": datetime.now().isoformat(),
        "completed_by": [],
    }
    assignments = load_assignments()
    assignments.append(a)
    save_assignments(assignments)
    return a


def get_assignments_for_team(team_id: str) -> list[dict]:
    return [a for a in load_assignments() if a.get("team_id") == team_id]


def get_assignments_for_player(user_id: str, team_id: str | None = None) -> list[dict]:
    """Assignments targeting this player (direct or via team)."""
    out = []
    for a in load_assignments():
        if a.get("assigned_to_type") == ASSIGNED_TO_PLAYER and a.get("assigned_to_id") == user_id:
            if team_id is None or a.get("team_id") == team_id:
                out.append(a)
        elif a.get("assigned_to_type") == ASSIGNED_TO_TEAM and (team_id is None or a.get("team_id") == team_id):
            out.append(a)
        elif a.get("assigned_to_type") == ASSIGNED_TO_SUBGROUP:
            if team_id and a.get("team_id") == team_id:
                out.append(a)
    return out


def mark_assignment_completed(assignment_id: str, user_id: str) -> bool:
    assignments = load_assignments()
    for a in assignments:
        if a.get("assignment_id") == assignment_id:
            completed = a.get("completed_by") or []
            if user_id not in completed:
                completed.append(user_id)
                a["completed_by"] = completed
                save_assignments(assignments)
            return True
    return False


def get_assignment_by_id(assignment_id: str) -> dict | None:
    for a in load_assignments():
        if a.get("assignment_id") == assignment_id:
            return a
    return None


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


def load_feedback() -> list[dict]:
    return _load_json(FEEDBACK_PATH, [])


def save_feedback(feedbacks: list[dict]) -> None:
    _save_json(FEEDBACK_PATH, feedbacks)


def create_feedback(
    player_id: str,
    coach_id: str,
    message: str,
    *,
    feedback_type: str = "encouragement",
    workout_id: str | None = None,
    visibility: str = "player_visible",
) -> dict:
    f = {
        "feedback_id": f"f_{secrets.token_hex(8)}",
        "player_id": player_id,
        "coach_id": coach_id,
        "workout_id": workout_id or "",
        "message": message or "",
        "feedback_type": feedback_type,
        "visibility": visibility,
        "created_at": datetime.now().isoformat(),
    }
    feedbacks = load_feedback()
    feedbacks.append(f)
    save_feedback(feedbacks)
    return f


def get_feedback_for_player(player_id: str, visible_only: bool = True) -> list[dict]:
    out = [x for x in load_feedback() if x.get("player_id") == player_id]
    if visible_only:
        out = [x for x in out if x.get("visibility") == "player_visible"]
    return sorted(out, key=lambda e: e.get("created_at", ""), reverse=True)


def get_feedback_for_team(team_id: str, limit: int = 50) -> list[dict]:
    """Recent feedback for players in this team (for activity feed)."""
    t = get_team_by_id(team_id)
    if not t:
        return []
    player_ids = {m["user_id"] for m in t.get("members", []) if m.get("role") == "player"}
    out = [x for x in load_feedback() if x.get("player_id") in player_ids]
    out = sorted(out, key=lambda e: e.get("created_at", ""), reverse=True)
    return out[:limit]


# ---------------------------------------------------------------------------
# Activity / Analytics (from existing profile data)
# ---------------------------------------------------------------------------


def get_player_activity_summary(profile: dict, days: int = 7) -> dict:
    """Derived from profile completion_history. Returns workouts_this_period, last_workout_date, streak."""
    hist = profile.get("completion_history") or []
    cutoff = date.today() - timedelta(days=days)
    workouts = 0
    last_date = None
    for e in hist:
        d = e.get("date") or ""
        try:
            dt = date.fromisoformat(d[:10]) if d else None
        except (ValueError, TypeError):
            dt = None
        if dt and dt >= cutoff:
            workouts += 1
        if dt and (last_date is None or dt > last_date):
            last_date = dt
    streak = profile.get("workout_streak") or 0
    try:
        from bender_leveling import _compute_streak
        streak = _compute_streak(hist)
    except Exception:
        pass
    return {
        "workouts_this_period": workouts,
        "last_workout_date": last_date.isoformat() if last_date else None,
        "streak": streak,
    }


def get_team_activity_summary(team_id: str, profile_loader) -> dict:
    """Aggregate activity for team members. profile_loader(user_id) -> profile dict."""
    players = get_team_players(team_id)
    today = date.today()
    week_start = today - timedelta(days=7)
    active_count = 0
    total_workouts = 0
    total_minutes = 0.0
    inactive_7_days: list[str] = []
    for m in players:
        uid = m.get("user_id")
        prof = profile_loader(uid) if uid else None
        if not prof:
            continue
        hist = prof.get("completion_history") or []
        workouts_this_week = 0
        mins_this_week = 0.0
        last_date = None
        for e in hist:
            d = e.get("date") or ""
            try:
                dt = date.fromisoformat(d[:10]) if d else None
            except (ValueError, TypeError):
                dt = None
            if dt:
                if dt >= week_start:
                    workouts_this_week += 1
                    mins_this_week += float(e.get("minutes", 0) or 0)
                if last_date is None or dt > last_date:
                    last_date = dt
        if workouts_this_week > 0:
            active_count += 1
        total_workouts += workouts_this_week
        total_minutes += mins_this_week
        if last_date is None or (today - last_date).days >= 7:
            inactive_7_days.append(uid)
    assignments = get_assignments_for_team(team_id)
    completed_assignments = sum(len(a.get("completed_by", [])) for a in assignments)
    total_instances = 0
    for a in assignments:
        if a.get("assigned_to_type") == ASSIGNED_TO_TEAM:
            total_instances += len(players)
        else:
            total_instances += 1
    completion_pct = (completed_assignments / total_instances * 100) if total_instances else 0
    return {
        "active_players_this_week": active_count,
        "total_players": len(players),
        "workouts_this_week": total_workouts,
        "total_training_minutes": total_minutes,
        "avg_minutes_per_player": total_minutes / active_count if active_count else 0,
        "completion_percentage": round(completion_pct, 1),
        "inactive_7_days": inactive_7_days,
    }


def get_recent_team_activity(team_id: str, profile_loader, limit: int = 20) -> list[dict]:
    """Feed of recent completions + assignments + feedback for dashboard."""
    feed = []
    t = get_team_by_id(team_id)
    if not t:
        return []
    player_ids = {m["user_id"]: m for m in t.get("members", [])}
    for uid in player_ids:
        prof = profile_loader(uid)
        if not prof:
            continue
        hist = (prof.get("completion_history") or [])[-50:]
        for e in reversed(hist):
            feed.append({
                "type": "workout_completed",
                "user_id": uid,
                "display_name": prof.get("display_name") or uid,
                "date": e.get("date"),
                "completed_at": e.get("completed_at"),
                "mode": e.get("mode"),
                "minutes": e.get("minutes"),
            })
    for a in get_assignments_for_team(team_id)[-30:]:
        feed.append({
            "type": "assignment",
            "assigned_by": a.get("assigned_by"),
            "assigned_to_type": a.get("assigned_to_type"),
            "workout_title": a.get("workout_title"),
            "created_at": a.get("created_at"),
        })
    for f in get_feedback_for_team(team_id, limit=30):
        feed.append({
            "type": "feedback",
            "player_id": f.get("player_id"),
            "coach_id": f.get("coach_id"),
            "created_at": f.get("created_at"),
        })
    feed.sort(key=lambda x: x.get("completed_at") or x.get("created_at") or "", reverse=True)
    return feed[:limit]
