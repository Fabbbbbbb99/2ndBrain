"""Extraction Quality Gate (EQG) diagnostics for PDF standards and books."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from second_brain.storage.models import VisualKind

UNMAPPED_GLYPH_RE = re.compile(r"[\ufffd\ue000-\uf8ff]")


@dataclass
class VisualCandidate:
    kind: VisualKind
    page_num: int
    bbox: list[float]
    reason: str


@dataclass
class PageQualityReport:
    page_num: int
    char_count: int
    single_char_ratio: float
    unmapped_glyphs: int
    image_area_ratio: float
    vector_rect_count: int
    broken_table_headers: int
    needs_vision: bool
    reasons: list[str] = field(default_factory=list)
    visuals: list[VisualCandidate] = field(default_factory=list)


def evaluate_pdf_page(
    *,
    page_num: int,
    text: str,
    page_width: float,
    page_height: float,
    images: list[dict[str, Any]],
    rects_or_lines_count: int,
    tables: list[list[list[str | None]]],
) -> PageQualityReport:
    """Inspect layout geometry, text quality, and table structure without an LLM."""
    stripped = text.strip()
    char_count = len(stripped)
    tokens = stripped.split()
    single_chars = sum(1 for tok in tokens if len(tok) == 1 and not tok.isdigit())
    single_char_ratio = round(single_chars / max(1, len(tokens)), 3)
    unmapped = len(UNMAPPED_GLYPH_RE.findall(stripped))

    page_area = max(1.0, page_width * page_height)
    img_area = 0.0
    visuals: list[VisualCandidate] = []
    for im in images:
        x0 = float(im.get("x0", 0.0))
        top = float(im.get("top", 0.0))
        x1 = float(im.get("x1", page_width))
        bottom = float(im.get("bottom", page_height))
        area = max(0.0, (x1 - x0) * (bottom - top))
        img_area += area
        if area / page_area >= 0.05:
            visuals.append(
                VisualCandidate(
                    kind="diagram",
                    page_num=page_num,
                    bbox=[round(x0, 1), round(top, 1), round(x1, 1), round(bottom, 1)],
                    reason="embedded_raster_figure",
                )
            )
    image_area_ratio = round(min(1.0, img_area / page_area), 3)

    broken_headers = 0
    for tbl in tables:
        if not tbl:
            continue
        header = tbl[0]
        if any(cell is None or not str(cell).strip() for cell in header):
            broken_headers += 1
        col_counts = {len(row) for row in tbl}
        if len(col_counts) > 1:
            broken_headers += 1

    reasons: list[str] = []
    if char_count < 80 and (image_area_ratio >= 0.05 or rects_or_lines_count >= 6):
        reasons.append("low_text_dense_graphics")
        if not visuals:
            visuals.append(
                VisualCandidate(
                    kind="diagram",
                    page_num=page_num,
                    bbox=[0.0, 0.0, round(page_width, 1), round(page_height, 1)],
                    reason="low_text_page",
                )
            )
    if unmapped >= 2:
        reasons.append("unmapped_glyphs")
        visuals.append(
            VisualCandidate(
                kind="equation",
                page_num=page_num,
                bbox=[0.0, 0.0, round(page_width, 1), round(page_height, 1)],
                reason="unmapped_math_glyphs",
            )
        )
    if len(tokens) >= 20 and single_char_ratio >= 0.30:
        reasons.append("single_char_math_scatter")
        visuals.append(
            VisualCandidate(
                kind="equation",
                page_num=page_num,
                bbox=[0.0, 0.0, round(page_width, 1), round(page_height, 1)],
                reason="equation_character_scatter",
            )
        )
    if broken_headers > 0:
        reasons.append("broken_or_merged_table")
        visuals.append(
            VisualCandidate(
                kind="table",
                page_num=page_num,
                bbox=[0.0, 0.0, round(page_width, 1), round(page_height, 1)],
                reason="broken_table_cells",
            )
        )
    elif rects_or_lines_count >= 30 and not tables and char_count >= 80:
        reasons.append("dense_vector_table_or_diagram")
        visuals.append(
            VisualCandidate(
                kind="table",
                page_num=page_num,
                bbox=[0.0, 0.0, round(page_width, 1), round(page_height, 1)],
                reason="unparsed_ruled_vector_region",
            )
        )

    return PageQualityReport(
        page_num=page_num,
        char_count=char_count,
        single_char_ratio=single_char_ratio,
        unmapped_glyphs=unmapped,
        image_area_ratio=image_area_ratio,
        vector_rect_count=rects_or_lines_count,
        broken_table_headers=broken_headers,
        needs_vision=bool(reasons),
        reasons=reasons,
        visuals=visuals,
    )
