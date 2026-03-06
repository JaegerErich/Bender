"""
Build a PDF of all drills that DON'T have videos yet, organized by tab then by equipment.
Excludes: Potential Drill Additions, NHL combine results. Skips circuits (no drill-level rows).
Use: checklist to film by equipment; only name, equipment, step_by_step, coaching_cues.

Run from repo root: python scripts/export_drills_needing_video_pdf.py
Writes: data/drills_needing_video.pdf
"""
from pathlib import Path
import json

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# Tabs to include (exclude nhl_combine_results, Potential Drill Additions, circuits)
TAB_FILES = {
    "Performance": "performance.json",
    "Shooting": "shooting.json",
    "Stickhandling": "stickhandling.json",
    "Skating Mechanics": "skating_mechanics.json",
    "Energy Systems": "energy_systems.json",
    "Mobility": "mobility.json",
    "Movement": "movement.json",
    "Speed & Agility": "speed_agility.json",
    "Warmup": "warmup.json",
}


def has_video(d: dict) -> bool:
    url = d.get("video_url") or ""
    if not isinstance(url, str):
        return False
    url = url.strip()
    return url.startswith("http")


def norm(s) -> str:
    if s is None:
        return ""
    return str(s).strip()


def equipment_key(eq: str) -> str:
    """Sort: None/empty last, then alphabetically."""
    if not eq or eq.lower() in ("none", "—", "-"):
        return "\xff"  # sort last
    return eq.lower()


def get_cues(d: dict) -> str:
    c = norm(d.get("coaching_cues") or d.get("notes") or "")
    return c


def load_tab(tab_label: str, filename: str) -> list[dict]:
    path = DATA_DIR / filename
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raw = [raw] if isinstance(raw, dict) else []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if has_video(item):
            continue
        name = norm(item.get("name") or "")
        if not name:
            continue
        out.append(item)
    return out


def build_by_tab_and_equipment() -> dict[str, dict[str, list[dict]]]:
    """Returns {tab_label: {equipment: [drill, ...]}}."""
    result: dict[str, dict[str, list[dict]]] = {}
    for tab_label, filename in TAB_FILES.items():
        drills = load_tab(tab_label, filename)
        by_equipment: dict[str, list[dict]] = {}
        for d in drills:
            eq = norm(d.get("equipment") or "")
            if not eq or eq.lower() in ("none", "null"):
                eq = "—"
            by_equipment.setdefault(eq, []).append(d)
        for eq in by_equipment:
            by_equipment[eq].sort(key=lambda x: norm(x.get("name") or ""))
        result[tab_label] = dict(sorted(by_equipment.items(), key=lambda p: equipment_key(p[0])))
    return result


def _safe(s: str) -> str:
    """Replace chars that can break fpdf."""
    if not s:
        return ""
    return s.replace("\x00", "").encode("latin-1", "replace").decode("latin-1")


def write_pdf(out_path: Path, by_tab: dict[str, dict[str, list[dict]]]) -> None:
    try:
        from fpdf import FPDF
        from fpdf.enums import XPos, YPos
    except ImportError:
        print("Install fpdf2: pip install fpdf2")
        return
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=10)
    pdf.set_margins(10, 10, 10)
    pdf.set_font("Helvetica", "", 7)
    line_h = 3.5
    small_h = 3

    for tab_label, by_equipment in by_tab.items():
        total_drills = sum(len(drills) for drills in by_equipment.values())
        if total_drills == 0:
            continue

        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 6, _safe(f"Tab: {tab_label}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 7)
        pdf.cell(0, line_h, _safe(f"Drills without video, by equipment ({total_drills} total)"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

        for equipment, drills in by_equipment.items():
            pdf.set_font("Helvetica", "B", 9)
            pdf.multi_cell(0, 5, _safe(f"Equipment: {equipment} ({len(drills)} drills)"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", "", 7)

            for d in drills:
                name = _safe(norm(d.get("name") or ""))
                steps = _safe(norm(d.get("step_by_step") or ""))
                cues = _safe(get_cues(d))
                eq_display = _safe(norm(d.get("equipment") or "—"))
                if not eq_display or eq_display.lower() in ("none", "null"):
                    eq_display = "—"

                pdf.set_font("Helvetica", "B", 7)
                pdf.multi_cell(0, small_h, name, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font("Helvetica", "", 7)
                pdf.cell(0, small_h, _safe(f"  Eq: {eq_display}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                if steps:
                    pdf.multi_cell(0, small_h, _safe(f"  Steps: {steps}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                if cues:
                    pdf.multi_cell(0, small_h, _safe(f"  Cues: {cues}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(0.5)

            pdf.ln(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_path))
    print(f"Wrote {out_path}")


def main() -> int:
    by_tab = build_by_tab_and_equipment()
    total = sum(
        len(drills)
        for tab_data in by_tab.values()
        for drills in tab_data.values()
    )
    print(f"Total drills without video: {total}")
    out_path = DATA_DIR / "drills_needing_video.pdf"
    write_pdf(out_path, by_tab)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
