"""Fix player portal: remove duplicate/wrong content and extract Coach Feedback to separate function."""
with open("ui_bender_teams.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace: wrong else block + duplicate Team Activity section
import re
content = re.sub(
    r'    else:\n        st\.caption\("Your coach.s recommendations[^"]*"\)\n\n    st\.divider\(\)\n\n    # --- 3\. Team Activity ---\n    st\.markdown\("#### Team Activity"\)\n    feed = get_recent_team_activity\(team_id, loader, 15\)\n    if not feed:\n        st\.caption\("No recent activity yet\."\)\n    else:\n        for item in feed\[:12\]:',
    "    else:\n        for item in feed[:12]:",
    content,
    count=1
)

# Replace "Your Team Progress" with "Your Progress" 
content = content.replace("#### Your Team Progress", "#### Your Progress")

# Replace Coach Feedback + Team Info: move Coach Feedback to _render_player_feedback_tab
# The Coach Feedback section (lines 666-677) and its divider should be moved.
# Team Info stays in Team Progress. So we just need to remove the Coach Feedback block
# from _render_player_team_progress_tab - it will be in a separate tab.
# The router calls _render_player_feedback_tab when sub == "Coach Feedback".
# So we need to add _render_player_feedback_tab and remove Coach Feedback from team progress tab.

old2 = """    st.divider()

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

    # --- 6. Team Info ---"""

new2 = """    st.divider()

    # Team Info at bottom of Team Progress tab
"""

content = content.replace(old2, new2)

# Add _render_player_feedback_tab before _render_assignment_card
old3 = """    st.caption(f"**Your position:** {pos}")


def _render_assignment_card("""

new3 = """    st.caption(f"**Your position:** {pos}")


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


def _render_assignment_card("""

content = content.replace(old3, new3)

with open("ui_bender_teams.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Done")
