"""
Generate a document showing how every stickhandling drill would look when
output in a generated workout (same format as the app).

Run from repo root: python scripts/export_stickhandling_generated_preview.py
Writes: data/stickhandling_generated_preview.md and data/stickhandling_generated_preview.pdf
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# Default reps for preview (matches typical block session)
DEFAULT_REPS = 2
BLOCK_LABELS = {
    "control": "A) Control/Touch",
    "quick_hands": "B) Quick Hands",
    "game_transfer": "C) Game Transfer",
    "decision": "D) Decision/Pressure",
    None: "Other (not used in block session)",
}


def _safe_pdf_text(s: str) -> str:
    """Ensure text is safe for PDF built-in fonts (Latin-1)."""
    if not s:
        return ""
    return s.encode("latin-1", errors="replace").decode("latin-1")


def write_pdf(
    out_path: Path,
    sections: list[tuple[str, list[tuple[str, str, str]]]],
    default_reps: int,
    work_sec: int,
    rest_sec: int,
) -> None:
    """Write a PDF. sections = [(section_title, [(line1, cue_equip_line, steps_line), ...]), ...]."""
    try:
        from fpdf import FPDF
    except ImportError:
        print("Skipping PDF: install fpdf2 with pip install fpdf2")
        return

    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=12)
    pdf.set_margins(12, 12, 12)
    pdf.add_page()
    pdf.set_font("Helvetica", "", 10)

    title = "Stickhandling drills — generated-style preview"
    pdf.set_font("Helvetica", "B", 14)
    pdf.multi_cell(0, 8, _safe_pdf_text(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(
        0, 5,
        _safe_pdf_text(
            "Every drill from the stickhandling database, formatted as in a generated workout. "
            f"Format: {work_sec}s work / {rest_sec}s rest per rep (preview uses {default_reps} reps per drill)."
        ),
        new_x="LMARGIN", new_y="NEXT"
    )
    pdf.ln(4)

    for section_title, drill_lines in sections:
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(0, 7, _safe_pdf_text(section_title), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        for line1, cue_equip, steps in drill_lines:
            pdf.set_font("Helvetica", "B", 9)
            pdf.multi_cell(0, 5, _safe_pdf_text(line1), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)
            if cue_equip:
                pdf.multi_cell(0, 5, _safe_pdf_text("  " + cue_equip), new_x="LMARGIN", new_y="NEXT")
            if steps:
                pdf.multi_cell(0, 5, _safe_pdf_text("  Steps: " + steps), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        pdf.ln(3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_path))
    print(f"Wrote PDF to {out_path}")


def main():
    import sys
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import json
    from bender_generate_v8_1 import (
        get,
        norm,
        _display_name,
        _video_marker_line,
        _stickhandling_special_equipment,
        _classify_stickhandling_block,
        STICKHANDLING_WORK_SEC,
        STICKHANDLING_REST_SEC,
    )

    path = DATA_DIR / "stickhandling.json"
    if not path.exists():
        print(f"Not found: {path}")
        return 1
    with open(path, encoding="utf-8") as f:
        drills = json.load(f)
    if not isinstance(drills, list):
        drills = [drills]

    default_reps = DEFAULT_REPS

    # Group by block, then sort by id within block
    by_block: dict = {}
    for d in drills:
        if not isinstance(d, dict):
            continue
        block = _classify_stickhandling_block(d)
        by_block.setdefault(block, []).append(d)
    for block in by_block:
        by_block[block].sort(key=lambda d: (norm(get(d, "id", "")), norm(get(d, "name", ""))))

    lines = [
        "# Stickhandling drills — generated-style preview",
        "",
        "Every drill from the stickhandling database, formatted exactly as they would appear in a generated workout.",
        f"Format: **{STICKHANDLING_WORK_SEC}s work / {STICKHANDLING_REST_SEC}s rest** per rep (preview uses {default_reps} reps per drill).",
        "",
        "---",
        "",
    ]

    pdf_sections: list[tuple[str, list[tuple[str, str, str]]]] = []

    for block in ("control", "quick_hands", "game_transfer", "decision", None):
        block_list = by_block.get(block, [])
        if not block_list:
            continue
        section_title = BLOCK_LABELS[block]
        lines.append(f"## {section_title}")
        lines.append("")

        drill_lines_pdf: list[tuple[str, str, str]] = []
        for d in block_list:
            name = _display_name(d)
            cue = norm(get(d, "coaching_cues", ""))
            if cue and "," in cue:
                cue = cue.split(",")[0].strip()
            eq = _stickhandling_special_equipment(d)
            steps = norm(get(d, "step_by_step", ""))
            video = _video_marker_line(d).strip()

            # Same format as build_stickhandling_blocks_session
            lines.append(f"- **{name}** | {default_reps} x {STICKHANDLING_WORK_SEC}s work / {STICKHANDLING_REST_SEC}s rest")
            parts = []
            if cue:
                parts.append(f"Cue: {cue}")
            if eq:
                parts.append(f"Equipment required: {eq}")
            if parts:
                lines.append(f"  {' | '.join(parts)}")
            if steps:
                lines.append(f"  Steps: {steps}")
            if video:
                lines.append(f"  `{video}`")
            lines.append("")

            cue_equip = " | ".join(parts) if parts else ""
            drill_lines_pdf.append((
                f"- {name} | {default_reps} x {STICKHANDLING_WORK_SEC}s work / {STICKHANDLING_REST_SEC}s rest",
                cue_equip,
                steps or "",
            ))
        pdf_sections.append((section_title, drill_lines_pdf))

    out_md = DATA_DIR / "stickhandling_generated_preview.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(drills)} drills to {out_md}")

    write_pdf(
        DATA_DIR / "stickhandling_generated_preview.pdf",
        sections=pdf_sections,
        default_reps=default_reps,
        work_sec=STICKHANDLING_WORK_SEC,
        rest_sec=STICKHANDLING_REST_SEC,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
