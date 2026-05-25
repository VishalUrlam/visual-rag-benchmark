"""Export SEED-Bench MCQ results to Excel with embedded images."""

from __future__ import annotations

import io
from pathlib import Path

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from PIL import Image as PILImage

from .seedbench_evaluator import SEEDBenchResult

_IMG_W = 120
_IMG_H = 90
_ROW_H = 70

_ANTHROPIC_MODEL = "Claude Opus 4.7"
_OPENAI_MODEL    = "GPT-5.5"


def _thumb(image_path: str, width: int, height: int) -> io.BytesIO:
    img = PILImage.open(image_path).convert("RGB")
    img.thumbnail((width * 2, height * 2), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return buf


def export_seedbench_excel(
    anthropic_result: SEEDBenchResult | None,
    openai_result: SEEDBenchResult | None,
    output_path: Path,
    anthropic_model: str = _ANTHROPIC_MODEL,
    openai_model: str = _OPENAI_MODEL,
) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "SEED-Bench Results"

    dark_fill  = PatternFill("solid", fgColor="1F3864")
    anth_fill  = PatternFill("solid", fgColor="2E4057")
    oai_fill   = PatternFill("solid", fgColor="1B4332")
    white_bold = Font(bold=True, color="FFFFFF", size=12)
    hdr_font   = Font(bold=True, color="FFFFFF", size=11)

    # ── Row 1: model name banner ──────────────────────────────────────────── #
    ws.row_dimensions[1].height = 28
    for col in range(1, 9):
        ws.cell(row=1, column=col).fill = dark_fill

    if anthropic_result:
        ws.merge_cells("I1:J1")
        c = ws.cell(row=1, column=9, value=anthropic_model)
        c.fill, c.font = anth_fill, white_bold
        c.alignment = Alignment(horizontal="center", vertical="center")

    if openai_result:
        ws.merge_cells("K1:L1")
        c = ws.cell(row=1, column=11, value=openai_model)
        c.fill, c.font = oai_fill, white_bold
        c.alignment = Alignment(horizontal="center", vertical="center")

    # ── Row 2: column headers ─────────────────────────────────────────────── #
    ws.row_dimensions[2].height = 30
    headers = [
        "Image", "Category", "Level", "Question",
        "A", "B", "C", "D",
        "Prediction", "✓ / ✗",
        "Prediction", "✓ / ✗",
    ]
    for col, h in enumerate(headers, 1):
        cell = ws.cell(row=2, column=col, value=h)
        cell.fill = dark_fill
        cell.font = hdr_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    # ── column widths ─────────────────────────────────────────────────────── #
    col_widths = [18, 20, 6, 38, 22, 22, 22, 22, 14, 8, 14, 8]
    for col, w in enumerate(col_widths, 1):
        ws.column_dimensions[get_column_letter(col)].width = w

    # ── data rows ─────────────────────────────────────────────────────────── #
    anth_map = {item.id: item for item in (anthropic_result.items if anthropic_result else [])}
    oai_map  = {item.id: item for item in (openai_result.items  if openai_result  else [])}
    all_ids  = list(dict.fromkeys(list(anth_map) + list(oai_map)))

    green = PatternFill("solid", fgColor="C6EFCE")
    red   = PatternFill("solid", fgColor="FFC7CE")
    opt_fill = PatternFill("solid", fgColor="EBF3FB")

    for row_idx, qid in enumerate(all_ids, 3):
        anth = anth_map.get(qid)
        oai  = oai_map.get(qid)
        ref  = anth or oai

        ws.row_dimensions[row_idx].height = _ROW_H

        # image
        if ref and Path(ref.image_path).exists():
            try:
                buf = _thumb(ref.image_path, _IMG_W, _IMG_H)
                xl_img = XLImage(buf)
                xl_img.width, xl_img.height = _IMG_W, _IMG_H
                ws.add_image(xl_img, f"A{row_idx}")
            except Exception:
                ws.cell(row=row_idx, column=1, value="[error]")
        else:
            ws.cell(row=row_idx, column=1, value="[missing]")

        # base columns
        for col, val in enumerate([None, ref.category if ref else "", ref.level if ref else "", ref.question if ref else ""], 1):
            if val is not None:
                ws.cell(row=row_idx, column=col, value=val).alignment = Alignment(vertical="center", wrap_text=True)

        # option columns A-D (cols 5-8)
        if ref:
            gt = ref.ground_truth
            for col_idx, letter in enumerate("ABCD", 5):
                c = ws.cell(row=row_idx, column=col_idx, value=ref.options.get(letter, ""))
                c.alignment = Alignment(vertical="center", wrap_text=True)
                if letter == gt:
                    c.fill = PatternFill("solid", fgColor="D9EAD3")
                    c.font = Font(bold=True)
                else:
                    c.fill = opt_fill

        # prediction columns
        def _write_pred(col: int, item) -> None:
            if item is None:
                ws.cell(row=row_idx, column=col, value="N/A").alignment = Alignment(vertical="center")
                ws.cell(row=row_idx, column=col + 1, value="").alignment = Alignment(vertical="center", horizontal="center")
                return
            ws.cell(row=row_idx, column=col, value=item.prediction).alignment = Alignment(vertical="center", horizontal="center")
            tick = ws.cell(row=row_idx, column=col + 1, value="✓" if item.correct else "✗")
            tick.alignment = Alignment(horizontal="center", vertical="center")
            tick.fill = green if item.correct else red
            tick.font = Font(bold=True, color="375623" if item.correct else "9C0006")

        _write_pred(9, anth)
        _write_pred(11, oai)

    # ── summary ───────────────────────────────────────────────────────────── #
    summary_row = len(all_ids) + 4
    ws.cell(row=summary_row, column=1, value="Summary").font = Font(bold=True, size=12)
    if anthropic_result:
        ws.cell(row=summary_row + 1, column=1, value=anthropic_model)
        ws.cell(row=summary_row + 1, column=2, value=f"{anthropic_result.accuracy:.1%}")
    if openai_result:
        ws.cell(row=summary_row + 2, column=1, value=openai_model)
        ws.cell(row=summary_row + 2, column=2, value=f"{openai_result.accuracy:.1%}")

    wb.save(output_path)
