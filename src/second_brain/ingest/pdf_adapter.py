"""Option 4D PDF ingestion adapter (pypdfium2 + pdfplumber + EQG + Lazy Vision crops)."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pdfplumber
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
import tiktoken

from second_brain.config import BrainConfig
from second_brain.ingest.eqg import PageQualityReport, evaluate_pdf_page
from second_brain.storage.models import AuthorityTier
from second_brain.storage.sqlite_store import SqliteStore, atomic_write_lf, lf_sha256

_ENC = tiktoken.get_encoding("o200k_base")

CLAUSE_HEADER_RE = re.compile(
    r"^(?:(?:Clause|Section)\s+)?(\d+(?:\.\d+)*)\s+([A-Za-z][^\n]{1,120})$"
)
ANNEX_HEADER_RE = re.compile(r"^(Annex\s+[A-Z])\b(?:\s*[:\-]?\s*(.*))?$", re.IGNORECASE)
CROSS_CLAUSE_RE = re.compile(r"\b(?:Clause|Section)\s+(\d+(?:\.\d+)*)\b")
CROSS_ANNEX_RE = re.compile(r"\b(Annex\s+[A-Z])\b", re.IGNORECASE)
EXT_STD_RE = re.compile(
    r"\b((?:(?:ISO|IEC|IEEE|ANSI|AIAA)(?:/(?:ISO|IEC|IEEE|ANSI|AIAA))*|MIL-STD)[-\s]+(?:[A-Z]-)?\d+(?:[-:]\d+)?[A-Z]?)\b"
)
_STD_PREFIX_RE = re.compile(
    r"^(?:(?:ISO|IEC|IEEE|ANSI|AIAA)(?:/(?:ISO|IEC|IEEE|ANSI|AIAA))*|MIL-STD)[-\s]+",
    re.IGNORECASE,
)
_BREADCRUMB_LINE_RE = re.compile(r"^(?:<!--\s*breadcrumb:[^\n]*-->\s*\n)+")

BOOK_ALIAS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bSEBoK\b"), "sebok"),
    (re.compile(r"\bINCOSE\s+(?:Systems\s+Engineering|SE)\s+Handbook\b", re.IGNORECASE), "incose-handbook"),
    (re.compile(r"\b(?:INCOSE\s+)?Needs\s+and\s+Requirements\s+Manual\b|\bINCOSE\s+NRM\b", re.IGNORECASE), "incose-nrm"),
    (re.compile(r"\bGuide\s+to\s+Writing\s+Requirements\b|\bGtWR\b", re.IGNORECASE), "incose-gtwr"),
    (re.compile(r"\bGuide\s+to\s+Needs\s+and\s+Requirements\b|\bGtNR\b", re.IGNORECASE), "incose-gtnr"),
]


def _strip_breadcrumb_header(text: str) -> str:
    """Strip injected leading breadcrumb comments before citation regex scanning."""
    return _BREADCRUMB_LINE_RE.sub("", text)


def _extract_std_code(label: str) -> str | None:
    """Extract normalized standard code (excluding bare copyright years like 2022)."""
    stripped = _STD_PREFIX_RE.sub("", label.strip())
    if not stripped or stripped == label.strip():
        return None
    no_year = re.sub(r":\d{4}$", "", stripped)
    code = re.sub(r"[\s_]+", "-", no_year).strip("-").lower()
    if not code:
        return None
    if re.fullmatch(r"\d{4}", code) and 1900 <= int(code) <= 2099:
        return None
    return code


def build_ingested_source_lookup(store: SqliteStore) -> dict[str, tuple[str, str, str]]:
    """Map normalized standard codes and slugs of ingested parsed docs to target metadata."""
    rows = store.conn.execute(
        "SELECT DISTINCT source_id, source_version FROM chunks WHERE store = 'parsed'"
    ).fetchall()
    lookup: dict[str, tuple[str, str, str]] = {}
    for r in rows:
        src_id = str(r["source_id"])
        src_ver = str(r["source_version"])
        entry = (src_id, _slugify(src_id, 32), _slugify(src_ver, 24))
        lookup[_slugify(src_id, 32)] = entry
        code = _extract_std_code(src_id)
        if code:
            lookup[code] = entry
    return lookup


def extract_standard_citations(
    text: str,
    *,
    current_source_id: str,
    ingested_lookup: dict[str, tuple[str, str, str]],
) -> list[tuple[str, float]]:
    """Extract cross-standard and external CITES (dst_uri, weight) tuples from chunk text."""
    clean = _strip_breadcrumb_header(text)
    current_code = _extract_std_code(current_source_id)
    found: dict[str, float] = {}

    for m in EXT_STD_RE.finditer(clean):
        raw = m.group(1)
        code = _extract_std_code(raw)
        if code is None:
            continue
        if code in ingested_lookup:
            tgt_src, tgt_safe_src, tgt_safe_ver = ingested_lookup[code]
            if tgt_src == current_source_id:
                continue
            dst = f"vault://{tgt_safe_src}@{tgt_safe_ver}#_doc"
            found[dst] = max(found.get(dst, 0.0), 0.95)
        else:
            if current_code and code == current_code:
                continue
            dst = f"vault://external@unresolved#{_slugify(raw, 40)}"
            found[dst] = max(found.get(dst, 0.0), 0.9)

    for pat, alias_key in BOOK_ALIAS_PATTERNS:
        if pat.search(clean) and alias_key in ingested_lookup:
            tgt_src, tgt_safe_src, tgt_safe_ver = ingested_lookup[alias_key]
            if tgt_src == current_source_id:
                continue
            dst = f"vault://{tgt_safe_src}@{tgt_safe_ver}#_doc"
            found[dst] = max(found.get(dst, 0.0), 0.95)

    return list(found.items())


def rebuild_cross_standard_edges(store: SqliteStore) -> dict[str, int]:
    """Rebuild cross-standard and external CITES edges cleanly without re-parsing PDFs."""
    ingested_lookup = build_ingested_source_lookup(store)
    store.conn.execute(
        "DELETE FROM edges WHERE rel_type = 'CITES' "
        "AND (dst_uri LIKE 'vault://external@unresolved#%' OR dst_uri LIKE '%#_doc')"
    )
    rows = store.conn.execute(
        "SELECT uri, source_id, text FROM chunks WHERE store = 'parsed'"
    ).fetchall()
    resolved_edges = 0
    external_edges = 0
    for r in rows:
        uri = str(r["uri"])
        src_id = str(r["source_id"])
        txt = str(r["text"])
        for dst, wt in extract_standard_citations(
            txt,
            current_source_id=src_id,
            ingested_lookup=ingested_lookup,
        ):
            if dst == uri:
                continue
            store.upsert_edge(uri, dst, "CITES", "EXTRACTED", wt, commit=False)
            if dst.startswith("vault://external@unresolved#"):
                external_edges += 1
            else:
                resolved_edges += 1
    store.conn.commit()
    return {
        "chunks_scanned": len(rows),
        "resolved_cross_std_edges": resolved_edges,
        "external_unresolved_edges": external_edges,
    }


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _slugify(value: str, max_len: int = 48) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip()).strip("_.")
    if not clean:
        clean = "sec"
    if len(clean) > max_len:
        h = hashlib.sha1(clean.encode("utf-8")).hexdigest()[:8]
        clean = f"{clean[: max_len - 9]}_{h}"
    return clean.lower()


def _norm_margin_line(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip())


def _strip_running_headers_footers(pages_lines: list[list[str]]) -> list[list[str]]:
    if len(pages_lines) < 4:
        return pages_lines
    top_counts: Counter[str] = Counter()
    bot_counts: Counter[str] = Counter()
    for lines in pages_lines:
        non_empty = [ln for ln in lines if ln.strip()]
        for ln in non_empty[:2]:
            top_counts[_norm_margin_line(ln)] += 1
        for ln in non_empty[-2:]:
            bot_counts[_norm_margin_line(ln)] += 1

    threshold = max(3, int(len(pages_lines) * 0.4))
    strip_top = {k for k, v in top_counts.items() if v >= threshold and k}
    strip_bot = {k for k, v in bot_counts.items() if v >= threshold and k}

    cleaned: list[list[str]] = []
    for lines in pages_lines:
        if not lines:
            cleaned.append([])
            continue
        out = list(lines)
        while out and (not out[0].strip() or _norm_margin_line(out[0]) in strip_top):
            if not out[0].strip():
                out.pop(0)
            elif _norm_margin_line(out[0]) in strip_top:
                out.pop(0)
        while out and (not out[-1].strip() or _norm_margin_line(out[-1]) in strip_bot):
            if not out[-1].strip():
                out.pop()
            elif _norm_margin_line(out[-1]) in strip_bot:
                out.pop()
        cleaned.append(out)
    return cleaned


@dataclass
class ParsedSection:
    clause_id: str
    clause_path: str
    major_group: str
    parent_clause_id: str | None
    title: str
    text: str
    page_nums: list[int] = field(default_factory=list)


@dataclass
class ParsedPdfResult:
    source_id: str
    source_version: str
    domain: str
    page_count: int
    sections: list[ParsedSection]
    page_reports: list[PageQualityReport]
    page_hashes: dict[int, str]
    skipped_clauses: list[str]
    toc_entries: int
    ocr_pages: list[int] = field(default_factory=list)
    native_low_text_pages: int = 0
    ocr_error: str | None = None


class PdfParserAdapter(Protocol):
    def parse(
        self,
        pdf_path: Path,
        source_id: str,
        source_version: str,
        domain: str,
    ) -> ParsedPdfResult: ...


def _format_table_markdown(table: list[list[str | None]]) -> str:
    if not table:
        return ""
    width = max(len(r) for r in table)
    rows = [[(str(c).replace("\n", " ").strip() if c is not None else "") for c in (r + [None] * (width - len(r)))] for r in table]
    header = rows[0]
    sep = ["---"] * width
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for r in rows[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _split_text_pieces(body_text: str, max_tokens: int) -> list[str]:
    """Split body text along paragraph, line, then word boundaries under max_tokens."""
    if not body_text.strip():
        return []

    units: list[str] = []
    for para in re.split(r"\n{2,}", body_text.strip()):
        para = para.strip()
        if not para:
            continue
        if count_tokens(para) <= max_tokens:
            units.append(para)
            continue
        for line in para.splitlines():
            line_s = line.strip()
            if not line_s:
                continue
            if count_tokens(line_s) <= max_tokens:
                units.append(line_s)
                continue
            words = line_s.split()
            buf: list[str] = []
            for w in words:
                cand = " ".join(buf + [w])
                if buf and count_tokens(cand) > max_tokens:
                    units.append(" ".join(buf))
                    buf = [w]
                else:
                    buf.append(w)
            if buf:
                units.append(" ".join(buf))

    chunks: list[str] = []
    cur: list[str] = []
    for unit in units:
        sep = "\n\n" if ("\n" not in unit and cur and "\n" not in cur[-1]) else "\n"
        cand = sep.join(cur + [unit]) if cur else unit
        if cur and count_tokens(cand) > max_tokens:
            chunks.append("\n".join(cur))
            cur = [unit]
        else:
            cur.append(unit)
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _split_oversized_section(
    sec: ParsedSection,
    used_ids: set[str],
    max_tokens: int = 600,
) -> list[ParsedSection]:
    """Split sections exceeding max_tokens so chunks fit the 512-WP embedder window."""
    if count_tokens(sec.text) <= max_tokens:
        used_ids.add(sec.clause_id)
        return [sec]

    lines = sec.text.splitlines()
    if lines and lines[0].startswith("## "):
        heading = lines[0]
        body = "\n".join(lines[1:]).strip()
    else:
        heading = f"## {sec.clause_id} {sec.title}".strip()
        body = sec.text.strip()

    head_toks = count_tokens(heading)
    budget = max(120, max_tokens - head_toks - 60)
    parts = _split_text_pieces(body, budget)
    if len(parts) <= 1:
        used_ids.add(sec.clause_id)
        return [sec]

    total = len(parts)
    out: list[ParsedSection] = []
    for idx, part_body in enumerate(parts, start=1):
        if idx == 1:
            cid = sec.clause_id
            parent_cid = sec.parent_clause_id
            cpath = f"{sec.clause_path} (Part 1/{total})"
            part_text = f"{heading}\n{part_body}\n"
        else:
            cid = f"{sec.clause_id}_p{idx}"
            while cid in used_ids:
                cid = f"{cid}_s"
            parent_cid = sec.clause_id
            cpath = f"{sec.clause_path} (Part {idx}/{total})"
            part_text = f"{heading} (Part {idx}/{total})\n{part_body}\n"
        used_ids.add(cid)
        out.append(
            ParsedSection(
                clause_id=cid,
                clause_path=cpath,
                major_group=sec.major_group,
                parent_clause_id=parent_cid,
                title=sec.title,
                text=part_text,
                page_nums=list(sec.page_nums),
            )
        )
    return out


def _ocr_result_to_text(ocr_result: Any) -> str:
    """Reconstruct visual reading lines from RapidOCR bounding-box tuples."""
    if not ocr_result:
        return ""
    boxes: list[tuple[float, float, float, str]] = []
    for item in ocr_result:
        if not item or len(item) < 2:
            continue
        pts, text = item[0], str(item[1] or "").strip()
        if not text:
            continue
        if len(item) >= 3 and item[2] is not None:
            try:
                if float(item[2]) < 0.30:
                    continue
            except (TypeError, ValueError):
                pass
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        x_min = min(xs)
        y_min = min(ys)
        y_max = max(ys)
        y_mid = 0.5 * (y_min + y_max)
        h = max(1.0, y_max - y_min)
        boxes.append((y_mid, x_min, h, text))

    if not boxes:
        return ""
    boxes.sort(key=lambda b: (b[0], b[1]))

    lines: list[list[tuple[float, float, float, str]]] = []
    cur_line: list[tuple[float, float, float, str]] = []
    line_y_mid = 0.0
    line_h = 1.0
    for b in boxes:
        y_mid, _x_min, h, _txt = b
        if not cur_line:
            cur_line = [b]
            line_y_mid = y_mid
            line_h = h
            continue
        if abs(y_mid - line_y_mid) <= max(6.0, 0.55 * line_h):
            cur_line.append(b)
            n = len(cur_line)
            line_y_mid = ((line_y_mid * (n - 1)) + y_mid) / n
            line_h = ((line_h * (n - 1)) + h) / n
        else:
            lines.append(cur_line)
            cur_line = [b]
            line_y_mid = y_mid
            line_h = h
    if cur_line:
        lines.append(cur_line)

    reconstructed: list[str] = []
    for line_boxes in lines:
        line_boxes.sort(key=lambda b: b[1])
        reconstructed.append(" ".join(b[3] for b in line_boxes))
    return "\n".join(reconstructed)


class PyPdfiumPlumberAdapter:
    """Fast MIT/Apache-2.0 PDF parser combining pypdfium2, lazy RapidOCR, and selective pdfplumber."""

    def __init__(
        self,
        *,
        enable_ocr: bool = True,
        ocr_scale: float = 200.0 / 72.0,
        ocr_min_image_ratio: float = 0.20,
    ) -> None:
        self.enable_ocr = enable_ocr
        self.ocr_scale = ocr_scale
        self.ocr_min_image_ratio = ocr_min_image_ratio

    def parse(
        self,
        pdf_path: Path,
        source_id: str,
        source_version: str,
        domain: str,
    ) -> ParsedPdfResult:
        toc_entries = 0
        toc_titles: dict[int, list[str]] = {}
        raw_pages_lines: list[list[str]] = []
        page_sizes: dict[int, tuple[float, float]] = {}
        page_images: dict[int, list[dict[str, float]]] = {}
        page_path_counts: dict[int, int] = {}
        page_hashes: dict[int, str] = {}
        candidate_table_pages: list[int] = []
        candidate_ocr_pages: list[int] = []
        native_low_text_pages = 0
        ocr_pages: list[int] = []
        ocr_error: str | None = None

        doc = pdfium.PdfDocument(str(pdf_path))
        try:
            page_count = len(doc)
            try:
                for item in doc.get_toc():
                    toc_entries += 1
                    if item.page_index is not None and item.title:
                        toc_titles.setdefault(item.page_index + 1, []).append(item.title.strip())
            except Exception:
                toc_entries = 0

            for idx in range(1, page_count + 1):
                page = doc[idx - 1]
                tp = None
                try:
                    w, h = page.get_size()
                    page_w = float(w or 595.0)
                    page_h = float(h or 842.0)
                    tp = page.get_textpage()
                    txt = tp.get_text_range() or ""

                    imgs: list[dict[str, float]] = []
                    path_cnt = 0
                    for obj in page.get_objects(max_depth=1):
                        if obj.type == pdfium_c.FPDF_PAGEOBJ_IMAGE:
                            try:
                                l, b, r, t = obj.get_bounds()
                                x0, x1 = min(float(l), float(r)), max(float(l), float(r))
                                y_low, y_high = min(float(b), float(t)), max(float(b), float(t))
                                top = max(0.0, min(page_h, page_h - y_high))
                                bottom = max(top, min(page_h, page_h - y_low))
                                imgs.append(
                                    {
                                        "x0": max(0.0, min(page_w, x0)),
                                        "top": top,
                                        "x1": max(0.0, min(page_w, x1)),
                                        "bottom": bottom,
                                    }
                                )
                            except Exception:
                                imgs.append(
                                    {
                                        "x0": 0.0,
                                        "top": 0.0,
                                        "x1": page_w,
                                        "bottom": page_h,
                                    }
                                )
                        elif obj.type == pdfium_c.FPDF_PAGEOBJ_PATH:
                            path_cnt += 1

                    page_sizes[idx] = (page_w, page_h)
                    page_images[idx] = imgs
                    page_path_counts[idx] = path_cnt
                    page_hashes[idx] = lf_sha256(txt)
                    raw_pages_lines.append(txt.splitlines())

                    stripped_len = len(txt.strip())
                    page_area = max(1.0, page_w * page_h)
                    img_area_ratio = min(
                        1.0,
                        sum(max(0.0, (im["x1"] - im["x0"]) * (im["bottom"] - im["top"])) for im in imgs) / page_area,
                    )
                    if stripped_len < 80:
                        native_low_text_pages += 1
                        if self.enable_ocr and img_area_ratio >= self.ocr_min_image_ratio:
                            candidate_ocr_pages.append(idx)
                    elif path_cnt >= 8:
                        candidate_table_pages.append(idx)
                finally:
                    if tp is not None:
                        tp.close()
                    page.close()

            if candidate_ocr_pages:
                try:
                    import numpy as np
                    from rapidocr_onnxruntime import RapidOCR
                except ImportError:
                    ocr_error = "rapidocr_onnxruntime is not installed"
                else:
                    ocr_engine = RapidOCR()
                    try:
                        for idx in candidate_ocr_pages:
                            page = doc[idx - 1]
                            bm = None
                            try:
                                bm = page.render(scale=self.ocr_scale)
                                pil_img = bm.to_pil()
                                arr = np.asarray(pil_img.convert("RGB"))
                            finally:
                                if bm is not None:
                                    bm.close()
                                page.close()
                            ocr_res, _ = ocr_engine(arr)
                            ocr_txt = _ocr_result_to_text(ocr_res)
                            native_txt = "\n".join(raw_pages_lines[idx - 1]).strip()
                            if len(ocr_txt.strip()) > len(native_txt) and len(ocr_txt.strip()) >= 40:
                                raw_pages_lines[idx - 1] = ocr_txt.splitlines()
                                page_hashes[idx] = lf_sha256(ocr_txt)
                                ocr_pages.append(idx)
                    finally:
                        del ocr_engine
        finally:
            doc.close()

        page_tables: dict[int, list[list[list[str | None]]]] = {}
        if candidate_table_pages:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for idx in candidate_table_pages:
                    pl_page = pdf.pages[idx - 1]
                    try:
                        page_tables[idx] = pl_page.extract_tables() or []
                    finally:
                        pl_page.flush_cache()

        ocr_pages_set = set(ocr_pages)
        page_reports: list[PageQualityReport] = []
        for idx in range(1, page_count + 1):
            page_w, page_h = page_sizes[idx]
            txt = "\n".join(raw_pages_lines[idx - 1])
            imgs_for_eqg = page_images[idx]
            if idx in ocr_pages_set and len(txt.strip()) >= 80:
                p_area = max(1.0, page_w * page_h)
                imgs_for_eqg = [
                    im
                    for im in imgs_for_eqg
                    if max(0.0, (im["x1"] - im["x0"]) * (im["bottom"] - im["top"])) / p_area < 0.65
                ]
            report = evaluate_pdf_page(
                page_num=idx,
                text=txt,
                page_width=page_w,
                page_height=page_h,
                images=imgs_for_eqg,
                rects_or_lines_count=page_path_counts[idx],
                tables=page_tables.get(idx, []),
            )
            page_reports.append(report)

        stripped_pages = _strip_running_headers_footers(raw_pages_lines)

        raw_sections: list[ParsedSection] = []
        current: ParsedSection | None = None
        seen_top_clauses: list[int] = []
        seen_clause_counts: Counter[str] = Counter()
        latest_clause_id_map: dict[str, str] = {}

        def _flush() -> None:
            nonlocal current
            if current is not None and current.text.strip():
                raw_sections.append(current)
            current = None

        for page_idx, lines in enumerate(stripped_pages, start=1):
            body_lines = list(lines)
            tables = page_tables.get(page_idx, [])
            if tables:
                body_lines.append("")
                for tbl in tables:
                    md_tbl = _format_table_markdown(tbl)
                    if md_tbl:
                        body_lines.append(md_tbl)
                        body_lines.append("")

            for line in body_lines:
                s = line.strip()
                m_clause = CLAUSE_HEADER_RE.match(s)
                m_annex = ANNEX_HEADER_RE.match(s) if not m_clause else None
                if m_clause:
                    _flush()
                    c_num = m_clause.group(1)
                    title = m_clause.group(2).strip()
                    parts = c_num.split(".")
                    if len(parts) == 1 and parts[0].isdigit():
                        seen_top_clauses.append(int(parts[0]))
                    major = f"c{parts[0].zfill(2)}" if parts[0].isdigit() else _slugify(parts[0], 16)
                    raw_parent = ".".join(parts[:-1]) if len(parts) > 1 else None
                    parent_id = latest_clause_id_map.get(raw_parent, raw_parent) if raw_parent else None
                    seen_clause_counts[c_num] += 1
                    unique_id = c_num if seen_clause_counts[c_num] == 1 else f"{c_num}_{seen_clause_counts[c_num]}"
                    latest_clause_id_map[c_num] = unique_id
                    breadcrumb = " > ".join([source_id] + [".".join(parts[: i + 1]) for i in range(len(parts))])
                    current = ParsedSection(
                        clause_id=unique_id,
                        clause_path=f"{breadcrumb} {title}".strip(),
                        major_group=major,
                        parent_clause_id=parent_id,
                        title=title,
                        text=f"## {c_num} {title}\n",
                        page_nums=[page_idx],
                    )
                elif m_annex:
                    _flush()
                    annex_id = m_annex.group(1).title()
                    title = (m_annex.group(2) or "").strip()
                    slug_id = _slugify(annex_id, 24)
                    seen_clause_counts[slug_id] += 1
                    unique_id = slug_id if seen_clause_counts[slug_id] == 1 else f"{slug_id}_{seen_clause_counts[slug_id]}"
                    current = ParsedSection(
                        clause_id=unique_id,
                        clause_path=f"{source_id} > {annex_id}" + (f" {title}" if title else ""),
                        major_group=slug_id,
                        parent_clause_id=None,
                        title=title or annex_id,
                        text=f"## {annex_id} {title}\n",
                        page_nums=[page_idx],
                    )
                else:
                    if current is None:
                        fallback_title = (toc_titles.get(page_idx) or ["Front Matter"])[0]
                        raw_p = f"p{page_idx:03d}"
                        seen_clause_counts[raw_p] += 1
                        unique_id = raw_p if seen_clause_counts[raw_p] == 1 else f"{raw_p}_{seen_clause_counts[raw_p]}"
                        current = ParsedSection(
                            clause_id=unique_id,
                            clause_path=f"{source_id} > Page {page_idx} ({fallback_title})",
                            major_group="c00_front",
                            parent_clause_id=None,
                            title=fallback_title,
                            text=f"## Page {page_idx} - {fallback_title}\n",
                            page_nums=[page_idx],
                        )
                    if page_idx not in current.page_nums:
                        current.page_nums.append(page_idx)
                    current.text += line + "\n"

        _flush()

        sections: list[ParsedSection] = []
        used_ids: set[str] = {s.clause_id for s in raw_sections}
        for sec in raw_sections:
            sections.extend(_split_oversized_section(sec, used_ids, max_tokens=600))

        skipped: list[str] = []
        if len(seen_top_clauses) >= 2:
            ordered = sorted(set(seen_top_clauses))
            for a, b in zip(ordered, ordered[1:]):
                if b - a > 1:
                    for missing_num in range(a + 1, b):
                        skipped.append(f"Clause {missing_num}")

        return ParsedPdfResult(
            source_id=source_id,
            source_version=source_version,
            domain=domain,
            page_count=page_count,
            sections=sections,
            page_reports=page_reports,
            page_hashes=page_hashes,
            skipped_clauses=skipped,
            toc_entries=toc_entries,
            ocr_pages=ocr_pages,
            native_low_text_pages=native_low_text_pages,
            ocr_error=ocr_error,
        )


def _apply_docling_upgrades(
    pdf_path: Path,
    parsed: ParsedPdfResult,
    table_pages: list[int],
    temp_out: Path,
) -> dict[int, str]:
    if not table_pages:
        return {}
    cmd = [
        sys.executable,
        "-m",
        "second_brain.ingest.docling_worker",
        "--pdf",
        str(pdf_path),
        "--pages",
        ",".join(str(p) for p in sorted(set(table_pages))),
        "--out",
        str(temp_out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=False)
    if proc.returncode != 0 or not temp_out.exists():
        return {}
    payload = json.loads(temp_out.read_text(encoding="utf-8"))
    if not payload.get("available"):
        return {}
    return {int(k): v for k, v in (payload.get("pages") or {}).items()}


def ingest_pdf_document(
    *,
    cfg: BrainConfig,
    store: SqliteStore,
    pdf_path: Path,
    domain: str,
    source_id: str,
    source_version: str,
    authority_tier: AuthorityTier = "primary_normative",
    run_docling_tables: bool = False,
    enable_ocr: bool = True,
    adapter: PdfParserAdapter | None = None,
) -> dict[str, Any]:
    """Run Option 4D PDF ingestion into state/parsed/<source_id> and SQLite."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF source not found: {pdf_path}")

    parser = adapter or PyPdfiumPlumberAdapter(enable_ocr=enable_ocr)
    parsed = parser.parse(pdf_path, source_id=source_id, source_version=source_version, domain=domain)
    gen = store.next_index_generation()

    safe_src = _slugify(source_id, 32)
    safe_ver = _slugify(source_version, 24)
    out_dir = cfg.parsed_dir / f"{safe_src}_{safe_ver}"
    out_dir.mkdir(parents=True, exist_ok=True)

    table_pages = [
        v.page_num
        for rep in parsed.page_reports
        for v in rep.visuals
        if v.kind == "table"
    ]
    docling_pages: dict[int, str] = {}
    if run_docling_tables and table_pages:
        docling_pages = _apply_docling_upgrades(
            pdf_path,
            parsed,
            table_pages,
            out_dir / "_docling_tables.json",
        )

    # Group sections into compact 1-file-per-major-clause Markdown files (<= 64-char filenames)
    by_group: dict[str, list[ParsedSection]] = {}
    for sec in parsed.sections:
        by_group.setdefault(sec.major_group, []).append(sec)

    clause_to_uri: dict[str, str] = {}
    for sec in parsed.sections:
        clause_slug = _slugify(sec.clause_id, 32)
        clause_to_uri[sec.clause_id] = f"vault://{safe_src}@{safe_ver}#{clause_slug}"

    try:
        pdf_rel = pdf_path.relative_to(cfg.vault_path).as_posix()
    except ValueError:
        pdf_rel = pdf_path.name

    ingested_lookup = build_ingested_source_lookup(store)
    self_entry = (source_id, safe_src, safe_ver)
    ingested_lookup[safe_src] = self_entry
    cur_code = _extract_std_code(source_id)
    if cur_code:
        ingested_lookup[cur_code] = self_entry

    chunks_written = 0
    edges_written = 0
    for group_name, secs in by_group.items():
        fname = _slugify(group_name, 48) + ".md"
        file_path = out_dir / fname
        rel_path = f"parsed/{safe_src}_{safe_ver}/{fname}"

        lines: list[str] = [
            "---",
            f"source_id: {source_id}",
            f"source_version: {source_version}",
            f"domain: {domain}",
            f"authority_tier: {authority_tier}",
            "---",
            "",
        ]
        spans: list[tuple[ParsedSection, int, int, str]] = []
        for sec in secs:
            body = sec.text.strip()
            for pnum in sec.page_nums:
                if pnum in docling_pages:
                    body += f"\n\n<!-- docling_table_upgrade page={pnum} -->\n{docling_pages[pnum].strip()}"
            full_block = f"<!-- breadcrumb: {sec.clause_path} -->\n{body}\n"
            start_line = len(lines) + 1
            block_lines = full_block.splitlines()
            lines.extend(block_lines)
            lines.append("")
            end_line = start_line + len(block_lines) - 1
            spans.append((sec, start_line, end_line, full_block))

        atomic_write_lf(file_path, "\n".join(lines))

        for sec, start_line, end_line, full_block in spans:
            uri = clause_to_uri[sec.clause_id]
            parent_uri = clause_to_uri.get(sec.parent_clause_id) if sec.parent_clause_id else None
            embed_key = lf_sha256(f"{cfg.vault_path.name}:bge-small-v1.5:v1:{full_block}")
            store.upsert_chunk(
                uri=uri,
                store="parsed",
                domain=domain,
                source_id=source_id,
                source_version=source_version,
                clause_path=sec.clause_path,
                parent_uri=parent_uri,
                authority_tier=authority_tier,
                file_relpath=rel_path,
                start_line=start_line,
                end_line=end_line,
                text=full_block,
                token_count=count_tokens(full_block),
                embed_key=embed_key,
                index_generation=gen,
                commit=False,
            )
            chunks_written += 1

            if parent_uri:
                store.upsert_edge(uri, parent_uri, "PART_OF", "EXTRACTED", 1.0, commit=False)
                edges_written += 1

            scan_text = _strip_breadcrumb_header(full_block)
            for m in CROSS_CLAUSE_RE.finditer(scan_text):
                target_id = m.group(1)
                if target_id != sec.clause_id:
                    dst = clause_to_uri.get(target_id, f"vault://{safe_src}@{safe_ver}#{_slugify(target_id, 32)}")
                    store.upsert_edge(uri, dst, "CITES", "EXTRACTED", 1.0, commit=False)
                    edges_written += 1

            for m in CROSS_ANNEX_RE.finditer(scan_text):
                annex_slug = _slugify(m.group(1), 24)
                dst = clause_to_uri.get(annex_slug, f"vault://{safe_src}@{safe_ver}#{annex_slug}")
                if dst != uri:
                    store.upsert_edge(uri, dst, "CITES", "EXTRACTED", 1.0, commit=False)
                    edges_written += 1

            for dst, wt in extract_standard_citations(
                full_block,
                current_source_id=source_id,
                ingested_lookup=ingested_lookup,
            ):
                if dst != uri:
                    store.upsert_edge(uri, dst, "CITES", "EXTRACTED", wt, commit=False)
                    edges_written += 1

    # Record visual nodes from EQG (clear stale pending_vision nodes for this version first)
    store.conn.execute(
        "DELETE FROM visual_nodes WHERE source_uri LIKE ? AND status = 'pending_vision'",
        (f"vault://{safe_src}@{safe_ver}#%",),
    )
    visuals_recorded = 0
    for rep in parsed.page_reports:
        page_uri = next(
            (clause_to_uri[s.clause_id] for s in parsed.sections if rep.page_num in s.page_nums),
            f"vault://{safe_src}@{safe_ver}#p{rep.page_num:03d}",
        )
        for v_idx, vis in enumerate(rep.visuals):
            vid = hashlib.sha256(f"{safe_src}:{safe_ver}:{rep.page_num}:{vis.kind}:{v_idx}".encode("utf-8")).hexdigest()[:16]
            repaired_now = docling_pages.get(rep.page_num) if vis.kind == "table" else None
            status = "certified" if repaired_now else "pending_vision"
            store.upsert_visual_node(
                visual_id=vid,
                source_uri=page_uri,
                pdf_relpath=pdf_rel,
                page_num=rep.page_num,
                bbox=vis.bbox,
                kind=vis.kind,
                status=status,
                page_sha256=parsed.page_hashes.get(rep.page_num, ""),
                repaired_md=repaired_now,
                commit=False,
            )
            visuals_recorded += 1

    store.conn.commit()

    scanned_only = parsed.page_count > 0 and parsed.native_low_text_pages == parsed.page_count
    report_payload = {
        "source_id": source_id,
        "source_version": source_version,
        "domain": domain,
        "page_count": parsed.page_count,
        "toc_entries": parsed.toc_entries,
        "chunks_written": chunks_written,
        "edges_written": edges_written,
        "visuals_flagged": visuals_recorded,
        "scanned_image_only_pdf": scanned_only,
        "ocr_pages_count": len(parsed.ocr_pages),
        "ocr_pages": parsed.ocr_pages,
        "ocr_error": parsed.ocr_error,
        "docling_tables_upgraded": sorted(docling_pages.keys()),
        "skipped_clauses": parsed.skipped_clauses,
        "flagged_pages": [
            {"page_num": r.page_num, "reasons": r.reasons, "char_count": r.char_count}
            for r in parsed.page_reports
            if r.needs_vision
        ],
    }
    atomic_write_lf(out_dir / "_ingest_report.json", json.dumps(report_payload, indent=2))
    return report_payload


def render_visual_crop(cfg: BrainConfig, visual_row: dict[str, Any]) -> Path:
    """Render a 200-DPI PNG crop for an EQG-flagged visual block using pypdfium2."""
    cfg.crops_dir.mkdir(parents=True, exist_ok=True)
    out_png = cfg.crops_dir / f"{visual_row['visual_id']}.png"
    if out_png.exists():
        return out_png

    pdf_candidate = Path(visual_row["pdf_relpath"])
    if not pdf_candidate.is_absolute():
        pdf_candidate = cfg.vault_path / pdf_candidate
    if not pdf_candidate.exists():
        raise FileNotFoundError(f"Cannot render visual crop; PDF missing: {pdf_candidate}")

    bbox = json.loads(visual_row["bbox_json"])
    doc = pdfium.PdfDocument(str(pdf_candidate))
    try:
        page = doc[int(visual_row["page_num"]) - 1]
        page_w, page_h = page.get_size()
        scale = 200.0 / 72.0
        pil_img = page.render(scale=scale).to_pil()
        x0, top, x1, bottom = [float(c) for c in bbox]
        crop_box = (
            max(0, int(x0 * scale)),
            max(0, int(top * scale)),
            min(pil_img.width, max(1, int(min(x1, page_w) * scale))),
            min(pil_img.height, max(1, int(min(bottom, page_h) * scale))),
        )
        cropped = pil_img.crop(crop_box)
        cropped.save(str(out_png), format="PNG")
    finally:
        doc.close()
    return out_png


def cache_visual_repair(
    *,
    cfg: BrainConfig,
    store: SqliteStore,
    visual_id: str,
    repaired_markdown: str,
    certify: bool = False,
) -> dict[str, Any]:
    """Cache multimodal host transcription keyed by page_sha256 and register searchable chunk."""
    row = store.conn.execute("SELECT * FROM visual_nodes WHERE visual_id = ?", (visual_id,)).fetchone()
    if not row:
        raise ValueError(f"Unknown visual_id: {visual_id!r}")

    status = "certified" if certify else "cached_unverified"
    store.conn.execute(
        "UPDATE visual_nodes SET status = ?, repaired_md = ? WHERE visual_id = ?",
        (status, repaired_markdown.replace("\r\n", "\n"), visual_id),
    )

    parent_chunk = store.get_chunk_by_uri(row["source_uri"])
    domain = parent_chunk.domain if parent_chunk else "global"
    source_id = parent_chunk.source_id if parent_chunk else "visual_repair"
    source_version = parent_chunk.source_version if parent_chunk else "1.0"
    repair_uri = f"vault://{_slugify(source_id, 32)}@{_slugify(source_version, 24)}#vis_{visual_id}"
    body = f"<!-- visual_repair: {visual_id} status={status} page_sha256={row['page_sha256']} -->\n{repaired_markdown.strip()}"

    store.upsert_chunk(
        uri=repair_uri,
        store="parsed",
        domain=domain,
        source_id=source_id,
        source_version=source_version,
        clause_path=f"{parent_chunk.clause_path if parent_chunk else source_id} > Visual {visual_id}",
        parent_uri=row["source_uri"],
        authority_tier="primary_reference" if certify else "working_memory",
        file_relpath=parent_chunk.file_relpath if parent_chunk else None,
        start_line=None,
        end_line=None,
        text=body,
        token_count=count_tokens(body),
        commit=False,
    )
    store.upsert_edge(repair_uri, row["source_uri"], "PART_OF", "EXTRACTED", 1.0, commit=True)
    return {
        "visual_id": visual_id,
        "repair_uri": repair_uri,
        "status": status,
        "page_sha256": row["page_sha256"],
    }
