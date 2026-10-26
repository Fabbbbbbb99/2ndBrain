"""
ingest_knowledge.py - Universal Initial Knowledge Base Builder for 2ndBrain

Ingests ANY folder of documents (PDFs, Markdown, Text, etc.) into the Obsidian vault:
1. Multi-Tier High-Fidelity PDF Processing Cascade:
   - Tier 1 (Fast Digital Path): PyMuPDF (fitz) native text & table extraction (<0.2s per PDF).
   - Tier 2 (Fallback 1): pdfplumber (spatial coordinate vector table recovery).
   - Tier 3 (Fallback 2): pypdf (pure Python digital text).
   - Tier 4 (Scanned Image Pipeline): PyMuPDF (fitz.get_pixmap) + RapidOCR.
2. Formulates structured concept notes in 01-Concepts/ with [[wikilinks]].
3. Stores primary reference extracts in 02-Sources/.
4. Dynamically registers new documents and MOCs in 00-Meta/Index.local.md.
5. Uses portable repo-relative POSIX keys in build/.ingest_manifest.json.
"""

import argparse
import contextlib
import hashlib
import html
import io
import json
import os
import re
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- PDF Processing Engines ---
# Fallback 1: pdfplumber (Spatial coordinate vector table recovery and layout text)
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# High-speed rendering & Scanned image rasterization: PyMuPDF / fitz
try:
    import pymupdf
    PYMUPDF_AVAILABLE = True
except ImportError:
    try:
        import fitz as pymupdf
        PYMUPDF_AVAILABLE = True
    except ImportError:
        PYMUPDF_AVAILABLE = False

# Fallback 2: pypdf (Pure Python PDF reader)
try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

# Scanned Image Pipeline: PyMuPDF (fitz.get_pixmap) + RapidOCR
try:
    from rapidocr_onnxruntime import RapidOCR
    RAPIDOCR_AVAILABLE = True
except ImportError:
    try:
        from rapidocr import RapidOCR
        RAPIDOCR_AVAILABLE = True
    except ImportError:
        RAPIDOCR_AVAILABLE = False


def normalize_source_key(path_or_str: Union[str, Path], repo_root: Optional[Path] = None) -> str:
    """
    Normalizes any source document path to a portable repo-relative POSIX key
    (e.g., 'source_materials/shared/sample.pdf') across Windows, macOS, and Linux.
    """
    root = (repo_root or REPO_ROOT).resolve()
    raw_posix = str(path_or_str).replace("\\", "/")

    # Portable extraction for any path inside source_materials/ regardless of host OS/user prefix
    marker = "source_materials/"
    if marker in raw_posix:
        idx = raw_posix.index(marker)
        return raw_posix[idx:]

    try:
        resolved = Path(path_or_str).resolve()
        return resolved.relative_to(root).as_posix()
    except Exception:
        return raw_posix


BOILERPLATE_PATTERNS = [
    r"(?i)Authorized licensed use limited to:.*?\n",
    r"(?i)Downloaded by .*?\n",
    r"©\s*.*?\n",
    r"(?i)Cambridge University Press.*?\n",
    r"(?i)The Edinburgh Building, Cambridge.*?\n",
    r"(?i)Published in the United States.*?\n",
    r"(?i)Information on this title:.*?\n",
    r"(?i)Library of Congress Cataloguing-in-Publication data.*?\n",
    r"(?i)A catalogue record for this publication.*?\n",
    r"(?i)This publication is in copyright.*?\n",
    r"(?i)Subject to statutory exception.*?\n",
    r"(?i)no reproduction of any part may take place.*?\n",
    r"(?i)ISBN\s*(?:-13:)?\s*[\d-]+\s*(?:hardback|paperback|ebook)?\n?",
    r"(?i)QA\d+[\.\d\w-]+\s*\d{4}\n?",
    r"(?i)Department of Electrical Engineering.*?\n",
    r"(?i)Electrical Engineering Department.*?\n",
    r"(?i)Printed in the United Kingdom.*?\n",
    r"(?i)CambridgeUniversityPress.*?\n",
    r"(?i)First published \d{4}.*?\n",
    r"(?i)All rights reserved.*?\n",
    r"(?i)arXiv:\d+\.\d+v\d+.*?\n",
]

SLOP_PATTERNS = [
    r"(?i)\bhere\'?s the thing:?\s*",
    r"(?i)\bin today\'?s (?:fast-paced|world|landscape|robotics)\b",
    r"(?i)\bit\'?s not (?:just )?about [^,\n]+, it\'?s about [^.\n]+\.?\s*",
    r"(?i)\bthe future (?:of [^.\n]+)? isn\'?t coming, it\'?s already here\.?\s*",
    r"(?i)\blet me be clear:?\s*",
    r"(?i)\bhope this helps!?\s*",
    r"(?i)\bat the end of the day,?\s*",
    r"(?i)\bwithout further ado,?\s*",
    r"(?i)\bin this (?:section|chapter|document), we (?:will )?dive deep into\b",
]


def clean_text(text: str) -> str:
    """Removes boilerplate, copyright watermarks, and AI slop patterns."""
    if not text:
        return ""
    for pat in BOILERPLATE_PATTERNS:
        text = re.sub(pat, "", text)
    for pat in SLOP_PATTERNS:
        text = re.sub(pat, "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def format_markdown_table(table_rows: List[List[Optional[str]]]) -> str:
    """Converts a 2D list of extracted table cells into a clean GitHub Flavored Markdown table."""
    if not table_rows or len(table_rows) < 1:
        return ""

    cleaned_rows = []
    for row in table_rows:
        cleaned_row = [re.sub(r"[\r\n\t]+", " ", (cell or "").strip()) for cell in row]
        if any(cleaned_row):
            cleaned_rows.append(cleaned_row)

    if not cleaned_rows:
        return ""

    num_cols = max(len(r) for r in cleaned_rows)
    padded_rows = [r + [""] * (num_cols - len(r)) for r in cleaned_rows]

    for r in padded_rows:
        for idx in range(len(r)):
            r[idx] = r[idx].replace("|", "\\|")

    header = padded_rows[0]
    separator = ["---"] * num_cols

    md_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for r in padded_rows[1:]:
        md_lines.append("| " + " | ".join(r) + " |")

    return "\n\n" + "\n".join(md_lines) + "\n\n"


def preserve_equations_and_math(text: str) -> str:
    """Detects mathematical expressions and preserves them in LaTeX math blocks."""
    if not text:
        return ""

    math_symbols = r"[∑∫∂√≈≠≤≥±×÷ΔαβγδεθλμστωϕψΩΦ\^_{}]"
    lines = text.splitlines()
    processed_lines = []
    in_math_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("$$") or stripped.startswith("\\["):
            processed_lines.append(line)
            continue

        has_symbols = bool(re.search(math_symbols, stripped))
        has_equality = "=" in stripped or "≈" in stripped or "∝" in stripped
        is_formula_candidate = (
            has_symbols
            and has_equality
            and len(stripped) < 120
            and not stripped.startswith("#")
            and not stripped.startswith("-")
            and not stripped.startswith("|")
        )

        if is_formula_candidate and not in_math_block:
            clean_formula = re.sub(r"(\d+)\s*\^\s*(\d+)", r"\1^{\2}", stripped)
            processed_lines.append(f"\n$$\n{clean_formula}\n$$\n")
        else:
            processed_lines.append(line)

    return "\n".join(processed_lines)


def detect_heuristic_tables(raw_text: str) -> str:
    """Fallback table detector for plain text or pypdf output with tab/multi-space columns."""
    lines = raw_text.splitlines()
    new_lines = []
    table_buffer = []

    for line in lines:
        parts = [p.strip() for p in re.split(r"\t|\s{3,}", line.strip()) if p.strip()]
        if len(parts) >= 3 and not line.strip().startswith("#"):
            table_buffer.append(parts)
        else:
            if table_buffer:
                if len(table_buffer) >= 2:
                    new_lines.append(format_markdown_table(table_buffer))
                else:
                    new_lines.append("   ".join(table_buffer[0]))
                table_buffer = []
            new_lines.append(line)

    if table_buffer:
        if len(table_buffer) >= 2:
            new_lines.append(format_markdown_table(table_buffer))
        else:
            new_lines.append("   ".join(table_buffer[0]))

    return "\n".join(new_lines)


def extract_ipynb(file_path: Path) -> Dict[str, Any]:
    """Extracts Markdown prose, LaTeX math, and fenced code cells from a Jupyter Notebook (.ipynb)."""
    try:
        data = json.loads(file_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {"text": "", "method": "ipynb_notebook", "num_pages": 1, "tables_found": 0}

    cells = data.get("cells", [])
    meta_lang = (
        data.get("metadata", {}).get("kernelspec", {}).get("language")
        or data.get("metadata", {}).get("language_info", {}).get("name")
        or "python"
    ).lower()

    blocks: List[str] = []
    for cell in cells:
        ctype = cell.get("cell_type", "")
        src = cell.get("source", [])
        cell_text = "".join(src) if isinstance(src, list) else str(src)
        cell_text = cell_text.strip()
        if not cell_text:
            continue
        if ctype == "markdown":
            blocks.append(clean_text(cell_text))
        elif ctype == "code":
            blocks.append(f"```{meta_lang}\n{cell_text}\n```")

    full_text = "\n\n".join(blocks)
    return {
        "text": full_text,
        "method": "ipynb_notebook",
        "num_pages": max(1, len(cells) // 4),
        "tables_found": full_text.count("| --- |"),
    }


def extract_vtt_transcript(file_path: Path) -> Dict[str, Any]:
    """Extracts clean lecture transcript paragraphs from WebVTT (.vtt) or SubRip (.srt) caption files."""
    try:
        raw = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {"text": "", "method": "vtt_transcript", "num_pages": 1, "tables_found": 0}

    cue_lines: List[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("WEBVTT") or s.startswith("NOTE") or "-->" in s:
            continue
        if s.isdigit():
            continue
        s_clean = re.sub(r"<[^>]+>", "", html.unescape(s)).strip()
        if s_clean and (not cue_lines or cue_lines[-1] != s_clean):
            cue_lines.append(s_clean)

    paragraphs: List[str] = []
    buf: List[str] = []
    char_acc = 0
    for cl in cue_lines:
        buf.append(cl)
        char_acc += len(cl)
        if char_acc >= 380 and cl.endswith((".", "?", "!")):
            paragraphs.append(" ".join(buf))
            buf = []
            char_acc = 0
    if buf:
        paragraphs.append(" ".join(buf))

    full_text = "\n\n".join(paragraphs)
    return {
        "text": full_text,
        "method": "vtt_transcript",
        "num_pages": max(1, len(full_text) // 2500),
        "tables_found": 0,
    }


def extract_html_document(file_path: Path) -> Dict[str, Any]:
    """Extracts textbook chapters and recitation slide decks from HTML while preserving LaTeX math and code."""
    try:
        raw = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {"text": "", "method": "html_textbook", "num_pages": 1, "tables_found": 0}

    # Remove script, style, svg, nav, and footer chrome
    cleaned = re.sub(r"<(script|style|svg|nav|footer|header)\b[^>]*>.*?</\1>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    # Convert headings and structural tags to markdown equivalents
    for level in range(1, 5):
        prefix = "#" * level
        cleaned = re.sub(
            rf"<h{level}\b[^>]*>(.*?)</h{level}>",
            lambda m, p=prefix: f"\n\n{p} {re.sub(r'<[^>]+>', '', m.group(1)).strip()}\n\n",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
    cleaned = re.sub(
        r"<pre\b[^>]*>(.*?)</pre>",
        lambda m: "\n\n```\n" + html.unescape(re.sub(r"<[^>]+>", "", m.group(1)).strip()) + "\n```\n\n",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(r"</?(p|div|section|article|li|tr|br)\b[^>]*>", "\n\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    cleaned = preserve_equations_and_math(clean_text(cleaned))

    return {
        "text": cleaned,
        "method": "html_textbook",
        "num_pages": max(1, len(cleaned) // 2500),
        "tables_found": 0,
    }


def extract_structured_or_raw_document(file_path: Path) -> Dict[str, Any]:
    """Extracts text from ROS interfaces, robot configs, MATLAB scripts, numeric logs, and documentation files."""
    ext = file_path.suffix.lower()
    fsize = file_path.stat().st_size

    # Fast check for multi-megabyte numeric trajectory/IMU dataset logs (.txt)
    if ext == ".txt" and fsize > 4096:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                sample = f.read(2048)
            if sample:
                num_chars = sum(c.isdigit() or c in " .-+eE\n\r\t,;" for c in sample)
                numeric_ratio = num_chars / len(sample)
                if numeric_ratio >= 0.68:
                    sample_lines = [l.strip() for l in sample.splitlines() if l.strip()][:15]
                    est_rows = max(len(sample_lines), int(fsize / max(12, len(sample) / max(1, len(sample_lines)))))
                    sample_block = "\n".join(sample_lines[:10])
                    summary_text = (
                        f"Numeric sensor / trajectory dataset log (`{file_path.name}`, "
                        f"size: {fsize / 1024:.1f} KB, ~{est_rows:,} rows).\n\n"
                        f"### Sample Trajectory / Sensor Rows\n```text\n{sample_block}\n```\n"
                    )
                    return {
                        "text": summary_text,
                        "method": "numeric_dataset_summary",
                        "num_pages": 1,
                        "tables_found": 0,
                    }
        except Exception:
            pass

    try:
        raw_content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {"text": "", "method": "raw_document", "num_pages": 1, "tables_found": 0}

    if ext in (".msg", ".srv", ".action", ".idl", ".proto"):
        body = raw_content.strip() or "# Empty interface definition (zero-argument request/response)."
        formatted = (
            f"Interface specification `{file_path.name}` (`{ext}` schema definition).\n\n"
            f"```{ext.lstrip('.')}\n{body}\n```\n"
        )
        return {"text": formatted, "method": "ros_interface", "num_pages": 1, "tables_found": 0}

    if ext in (".urdf", ".sdf", ".xacro", ".launch", ".yaml", ".yml", ".xml", ".cfg"):
        body = raw_content.strip()
        if len(body) > 45000:
            body = body[:45000] + f"\n... [Truncated at 45,000 chars; total file size: {fsize / 1024:.1f} KB]"
        formatted = preserve_equations_and_math(detect_heuristic_tables(body))
        return {
            "text": formatted,
            "method": "robot_config",
            "num_pages": max(1, len(formatted) // 3000),
            "tables_found": 0,
        }

    if ext == ".m":
        doc_lines = []
        for line in raw_content.splitlines():
            s = line.strip()
            if s.startswith("%"):
                doc_lines.append(s.lstrip("% ").strip())
            elif doc_lines:
                break
        header = "\n".join(doc_lines)
        formatted = (
            f"MATLAB / Octave robotics algorithm `{file_path.name}`.\n\n"
            + (f"{header}\n\n" if header else "")
            + f"```matlab\n{raw_content.strip()}\n```\n"
        )
        return {
            "text": formatted,
            "method": "matlab_script",
            "num_pages": max(1, len(formatted) // 3000),
            "tables_found": 0,
        }

    full_text = preserve_equations_and_math(detect_heuristic_tables(raw_content))
    return {
        "text": full_text,
        "method": "raw_document",
        "num_pages": max(1, len(full_text) // 3000),
        "tables_found": full_text.count("| --- |"),
    }


def extract_text_from_pdf(pdf_path: Path) -> Dict[str, Any]:
    """Extracts text from PDF with Table and Equation Preservation using Fast Digital-First Cascade."""
    cache_db = REPO_ROOT / "build" / ".pdf_extract_cache.db"
    if cache_db.exists():
        try:
            norm_k = normalize_source_key(pdf_path, REPO_ROOT)
            fsize = pdf_path.stat().st_size
            conn = sqlite3.connect(str(cache_db))
            row = conn.execute(
                "SELECT body, engine, num_pages, tables_found FROM pdf_cache "
                "WHERE norm_key = ? OR (source_file = ? AND file_size = ?) LIMIT 1",
                (norm_k, pdf_path.name, fsize),
            ).fetchone()
            conn.close()
            if row and row[0]:
                return {
                    "pages": [row[0]],
                    "method": row[1] or "pymupdf_digital_fast",
                    "num_pages": int(row[2] or 1),
                    "tables_found": int(row[3] or 0),
                }
        except Exception:
            pass

    pages_text = []
    num_pages = 0
    tables_found_total = 0

    if PYMUPDF_AVAILABLE:
        try:
            doc = pymupdf.open(str(pdf_path))
            doc_pages = len(doc)
            num_pages = doc_pages
            sample_chars = sum(
                len((doc[i].get_text("text") or "").strip()) for i in range(min(8, doc_pages))
            )
            if sample_chars > 80:
                pages_text = []
                tables_found_total = 0
                for page in doc:
                    page_parts = []
                    table_md_blocks = []
                    if doc_pages <= 60:
                        try:
                            tabs = page.find_tables()
                            if tabs and tabs.tables:
                                tables_found_total += len(tabs.tables)
                                for tab in tabs:
                                    md_t = format_markdown_table(tab.extract())
                                    if md_t:
                                        table_md_blocks.append(md_t)
                        except Exception:
                            pass

                    t = clean_text(page.get_text("text") or "")
                    t_tables = detect_heuristic_tables(t) if not table_md_blocks else t
                    t_math = preserve_equations_and_math(t_tables)
                    page_parts.append(t_math)
                    if table_md_blocks:
                        page_parts.append("\n### Extracted Tables\n" + "".join(table_md_blocks))
                    pages_text.append("\n\n".join(page_parts))

                doc.close()
                return {
                    "pages": pages_text,
                    "method": "pymupdf_digital_fast",
                    "num_pages": doc_pages,
                    "tables_found": tables_found_total,
                }
            doc.close()
        except Exception as e:
            print(f"    [pymupdf digital check]: {e}")

    if PDFPLUMBER_AVAILABLE:
        try:
            print(
                f"    [pdfplumber] Extracting '{pdf_path.name}' with spatial coordinate vector table recovery..."
            )
            with pdfplumber.open(str(pdf_path)) as pdf:
                pdf_num_pages = len(pdf.pages)
                sample_chars = 0
                for p_idx in range(min(5, pdf_num_pages)):
                    t = pdf.pages[p_idx].extract_text() or ""
                    sample_chars += len(t.strip())

                if sample_chars > 100:
                    pages_text = []
                    tables_found_total = 0
                    for page in pdf.pages:
                        page_parts = []
                        tables = page.extract_tables() or []
                        table_md_blocks = []
                        if tables:
                            tables_found_total += len(tables)
                            for t in tables:
                                md_t = format_markdown_table(t)
                                if md_t:
                                    table_md_blocks.append(md_t)

                        prose = page.extract_text(layout=True) or ""
                        prose_clean = clean_text(prose)
                        prose_with_math = preserve_equations_and_math(prose_clean)

                        page_parts.append(prose_with_math)
                        if table_md_blocks:
                            page_parts.append("\n### Extracted Tables\n" + "".join(table_md_blocks))

                        pages_text.append("\n\n".join(page_parts))

                    print(
                        f"    [pdfplumber] Successfully extracted {pdf_num_pages} pages "
                        f"({tables_found_total} tables recovered)."
                    )
                    return {
                        "pages": pages_text,
                        "method": "pdfplumber_with_tables",
                        "num_pages": pdf_num_pages,
                        "tables_found": tables_found_total,
                    }
        except Exception as e:
            print(f"    [pdfplumber fallback]: {e}")

    if PYMUPDF_AVAILABLE:
        try:
            doc = pymupdf.open(str(pdf_path))
            doc_pages = len(doc)
            sample_chars = sum(len((doc[i].get_text() or "").strip()) for i in range(min(5, doc_pages)))
            if sample_chars > 100:
                pages_text = []
                tables_found_total = 0
                for page in doc:
                    page_parts = []
                    table_md_blocks = []
                    try:
                        tabs = page.find_tables()
                        if tabs and tabs.tables:
                            tables_found_total += len(tabs.tables)
                            for tab in tabs:
                                md_t = format_markdown_table(tab.extract())
                                if md_t:
                                    table_md_blocks.append(md_t)
                    except Exception:
                        pass

                    t = clean_text(page.get_text("text") or "")
                    t_tables = detect_heuristic_tables(t) if not table_md_blocks else t
                    t_math = preserve_equations_and_math(t_tables)
                    page_parts.append(t_math)
                    if table_md_blocks:
                        page_parts.append("\n### Extracted Tables\n" + "".join(table_md_blocks))
                    pages_text.append("\n\n".join(page_parts))

                doc.close()
                return {
                    "pages": pages_text,
                    "method": "pymupdf_text",
                    "num_pages": doc_pages,
                    "tables_found": tables_found_total,
                }
            doc.close()
        except Exception as e:
            print(f"    [pymupdf text fallback]: {e}")

    if PYPDF_AVAILABLE:
        try:
            reader = pypdf.PdfReader(str(pdf_path))
            reader_pages = len(reader.pages)
            sample_chars = sum(
                len((reader.pages[p].extract_text() or "").strip()) for p in range(min(5, reader_pages))
            )
            if sample_chars > 100:
                pages_text = []
                for page in reader.pages:
                    t = clean_text(page.extract_text() or "")
                    t_tables = detect_heuristic_tables(t)
                    t_math = preserve_equations_and_math(t_tables)
                    pages_text.append(t_math)
                return {
                    "pages": pages_text,
                    "method": "pypdf_digital_text",
                    "num_pages": reader_pages,
                    "tables_found": 0,
                }
        except Exception:
            pass

    cached_candidates = [
        pdf_path.with_name(f"{pdf_path.stem}_extracted.txt"),
        pdf_path.with_name(f"{pdf_path.stem.replace(' ', '_')}_extracted.txt"),
        pdf_path.with_name(f"{pdf_path.stem.replace('_', ' ')}_extracted.txt"),
    ]
    for cached_txt in cached_candidates:
        if cached_txt.exists():
            t = cached_txt.read_text(encoding="utf-8", errors="ignore")
            t_math = preserve_equations_and_math(detect_heuristic_tables(clean_text(t)))
            return {
                "pages": [t_math],
                "method": "cached_ocr",
                "num_pages": num_pages or 1,
                "tables_found": 0,
            }

    if not (PYMUPDF_AVAILABLE and RAPIDOCR_AVAILABLE):
        print(
            f"    [Warning] PDF '{pdf_path.name}' is scanned, but PyMuPDF or RapidOCR is not available."
        )
        return {"pages": [], "method": "unreadable_scanned", "num_pages": num_pages or 1, "tables_found": 0}

    print(
        f"    [Scanned PDF Detected] Running PyMuPDF (fitz.get_pixmap) + RapidOCR on '{pdf_path.name}'..."
    )
    ocr = RapidOCR()
    doc = pymupdf.open(str(pdf_path))
    num_pages = len(doc)
    ocr_pages = []

    for i in range(num_pages):
        try:
            page = doc[i]
            pix = page.get_pixmap(dpi=150)
            png_bytes = pix.tobytes("png")
            ocr_img = np.array(Image.open(io.BytesIO(png_bytes)))
            result, _ = ocr(ocr_img)
            lines = [item[1] for item in result] if result else []
            ocr_text = "\n".join(lines)
            ocr_text = preserve_equations_and_math(detect_heuristic_tables(clean_text(ocr_text)))
            ocr_pages.append(ocr_text)
        except Exception as e:
            print(f"    [RapidOCR Error on page {i+1}]: {e}")
            ocr_pages.append("")
    doc.close()

    return {
        "pages": ocr_pages,
        "method": "scanned_fitz_rapidocr",
        "num_pages": num_pages,
        "tables_found": 0,
    }


class UniversalKnowledgeIngester:
    def __init__(self, vault_dir: str):
        self.vault_dir = Path(vault_dir).resolve()
        self.repo_root = self.vault_dir.parent
        self.concepts_dir = self.vault_dir / "01-Concepts"
        self.sources_dir = self.vault_dir / "02-Sources"
        self.meta_dir = self.vault_dir / "00-Meta"
        self.index_file = self.meta_dir / "Index.md"

        self.cache_dir = self.repo_root / "build"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.cache_dir / ".ingest_manifest.json"
        self._manifest_dirty = False
        self.manifest = self._load_manifest()
        self._index_cache: Optional[str] = None
        self._slug_owners: Dict[Any, str] = {}

        for k, v in self.manifest.items():
            if isinstance(v, dict) and v.get("slug"):
                try:
                    kp_parts = k.replace("\\", "/").split("/")
                    if "source_materials" in kp_parts:
                        idx = kp_parts.index("source_materials")
                        rh = "/".join(kp_parts[idx + 1 : -1][:3])
                    else:
                        rh = ""
                    self._slug_owners[(rh, v["slug"])] = k
                except Exception:
                    pass

        for d in [self.concepts_dir, self.sources_dir, self.meta_dir]:
            d.mkdir(parents=True, exist_ok=True)

        if self._manifest_dirty:
            self._save_manifest()
            self._manifest_dirty = False

    def _load_manifest(self) -> Dict[str, Any]:
        """Loads persistent document hash manifest and migrates absolute keys to repo-relative POSIX keys."""
        if not self.manifest_file.exists():
            return {}
        try:
            raw_data = json.loads(self.manifest_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

        normalized: Dict[str, Any] = {}
        for k, v in raw_data.items():
            nk = normalize_source_key(k, self.repo_root)
            if nk != k:
                self._manifest_dirty = True
            normalized[nk] = v
        return normalized

    def _save_manifest(self):
        """Saves persistent document hash manifest."""
        try:
            self.manifest_file.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"  [Warning] Could not save manifest: {e}")

    def _compute_sha256(self, file_path: Path) -> str:
        """Computes SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    SUPPORTED_EXTS = {
        ".pdf",
        ".txt",
        ".md",
        ".markdown",
        ".rst",
        ".adoc",
        ".ipynb",
        ".tex",
        ".dox",
        ".vtt",
        ".html",
        ".m",
        ".urdf",
        ".sdf",
        ".xacro",
        ".launch",
        ".yaml",
        ".yml",
        ".xml",
        ".cfg",
        ".msg",
        ".srv",
        ".action",
        ".idl",
        ".proto",
    }

    IGNORED_DIRS = {
        ".git",
        ".github",
        ".gitlab",
        ".circleci",
        ".devcontainer",
        ".vscode",
        ".bazelci",
        "3rdparty",
        "thirdparty",
        "node_modules",
        "__pycache__",
    }

    IGNORED_STEMS = {
        "license",
        "contributing",
        "contributors",
        "code_of_conduct",
        "authors",
        "copying",
        "copyright",
        "cmakelists",
        "changelog",
    }

    def sanitize_title(self, name: str) -> str:
        """Converts filename into clean title and directory slug."""
        clean = re.sub(r"[\(\)\[\]\{\}_]", " ", name).strip()
        clean = re.sub(r"\s+", " ", clean)
        return clean

    def slugify(self, text: str) -> str:
        s = re.sub(r"([a-z])([A-Z])", r"\g<1>-\g<2>", text)
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s-]", "", s).strip()
        slug = re.sub(r"[-\s_]+", "-", s)
        return slug[:65].rstrip("-")

    def _is_eligible_source_file(self, file_path: Path) -> bool:
        """Returns True if file_path is an eligible non-boilerplate technical source file."""
        if not file_path.exists() or not file_path.is_file():
            return False
        name = file_path.name
        name_lower = name.lower()
        if name.startswith(".") or name.startswith("~"):
            return False
        if name_lower.endswith(".pdf_tex") or name_lower.endswith("_extracted.txt") or name_lower.endswith("_extracted.json"):
            return False
        ext = file_path.suffix.lower()
        if ext not in self.SUPPORTED_EXTS:
            return False
        if any(part.lower() in self.IGNORED_DIRS or (part.startswith(".") and part not in (".", "..")) for part in file_path.parts[:-1]):
            return False
        stem_lower = file_path.stem.lower()
        if stem_lower in self.IGNORED_STEMS:
            return False
        if ext == ".pdf" and file_path.with_suffix(".pdf_tex").exists():
            return False
        if ext == ".html":
            parts_lower = {pt.lower() for pt in file_path.parts}
            if name_lower in {"404.html", "422.html", "500.html", "index.html", "search.html"}:
                return False
            if {"_templates", "fixtures", "static_resources"} & parts_lower:
                return False
            if not (
                name_lower.endswith(".slides.html")
                or "book" in parts_lower
                or "recitations" in parts_lower
            ):
                return False
        return True

    def clean_ingested_knowledge(self, preserve_caches: bool = True) -> Dict[str, int]:
        """
        Cleans all ingested notes from vault/01-Concepts and vault/02-Sources (preserving README.md
        and 03-Agent-Memory/), resets Index.local.md, and removes vault_vectors.db and .ingest_manifest.json.
        When preserve_caches=True, snapshots existing PDF text extracts and deterministic BGE embeddings
        into temporary SQLite caches in build/ so clean re-ingestion does not repeat CPU OCR/ONNX work.
        """
        stats = {"concepts_removed": 0, "sources_removed": 0, "pdf_cached": 0, "embeddings_cached": 0}
        pdf_cache_db = self.cache_dir / ".pdf_extract_cache.db"
        emb_cache_db = self.cache_dir / ".embedding_cache.db"

        if preserve_caches:
            try:
                if pdf_cache_db.exists():
                    pdf_cache_db.unlink()
                conn = sqlite3.connect(str(pdf_cache_db))
                conn.execute(
                    "CREATE TABLE pdf_cache ("
                    "norm_key TEXT PRIMARY KEY, source_file TEXT, file_size INTEGER, "
                    "body TEXT, engine TEXT, num_pages INTEGER, tables_found INTEGER)"
                )
                rows_to_insert = []
                for norm_k, meta in self.manifest.items():
                    if not norm_k.lower().endswith(".pdf") or not isinstance(meta, dict):
                        continue
                    slug = meta.get("slug", "")
                    if not slug:
                        continue
                    parts = norm_k.split("/")
                    rel_h = Path(*parts[parts.index("source_materials") + 1 : -1][:3]) if "source_materials" in parts and len(parts) > 2 else Path("")
                    src_note = (self.sources_dir / rel_h / f"{slug}-Source.md") if str(rel_h) != "." else (self.sources_dir / f"{slug}-Source.md")
                    if not src_note.exists():
                        continue
                    raw_note = src_note.read_text(encoding="utf-8", errors="ignore")
                    marker = "## Verbatim Extraction\n"
                    if marker not in raw_note:
                        continue
                    body = raw_note.split(marker, 1)[1].strip()
                    if len(body) < 50:
                        continue
                    m_pages = re.search(r"^total_pages:\s*(\d+)", raw_note, re.MULTILINE)
                    m_tables = re.search(r"^tables_preserved:\s*(\d+)", raw_note, re.MULTILINE)
                    m_engine = re.search(r'^engine:\s*"?([^"\n]+)"?', raw_note, re.MULTILINE)
                    rows_to_insert.append(
                        (
                            norm_k,
                            Path(norm_k).name,
                            int(meta.get("size") or 0),
                            body,
                            m_engine.group(1).strip() if m_engine else "pymupdf_digital_fast",
                            int(m_pages.group(1)) if m_pages else 1,
                            int(m_tables.group(1)) if m_tables else 0,
                        )
                    )
                if rows_to_insert:
                    conn.executemany(
                        "INSERT OR REPLACE INTO pdf_cache VALUES (?, ?, ?, ?, ?, ?, ?)",
                        rows_to_insert,
                    )
                    conn.commit()
                    stats["pdf_cached"] = len(rows_to_insert)
                conn.close()
            except Exception as e:
                print(f"  [Cache Warning] PDF cache snapshot note: {e}")

            vec_db = self.meta_dir / "vault_vectors.db"
            if vec_db.exists():
                try:
                    for wal_ext in ("-wal", "-shm"):
                        wf = self.meta_dir / f"vault_vectors.db{wal_ext}"
                        if wf.exists():
                            wf.unlink(missing_ok=True)
                    if emb_cache_db.exists():
                        emb_cache_db.unlink(missing_ok=True)
                    os.replace(vec_db, emb_cache_db)
                    conn_c = sqlite3.connect(str(emb_cache_db))
                    stats["embeddings_cached"] = conn_c.execute(
                        "SELECT COUNT(*) FROM child_chunks"
                    ).fetchone()[0]
                    conn_c.close()
                except Exception as e:
                    print(f"  [Cache Warning] Embedding cache snapshot note: {e}")

        for folder, key in [(self.concepts_dir, "concepts_removed"), (self.sources_dir, "sources_removed")]:
            if not folder.exists():
                continue
            for item in list(folder.iterdir()):
                if item.is_file() and item.name.lower() == "readme.md":
                    continue
                if item.is_dir():
                    cnt = sum(1 for _ in item.rglob("*.md"))
                    shutil.rmtree(item, ignore_errors=True)
                    stats[key] += cnt
                elif item.is_file():
                    item.unlink(missing_ok=True)
                    stats[key] += 1

        local_index = self.meta_dir / "Index.local.md"
        self._index_cache = "# 📚 Ingested Knowledge Bases (Local Index)\n\n"
        local_index.write_text(self._index_cache, encoding="utf-8")
        self._index_dirty = False

        for db_f in [
            self.meta_dir / "vault_vectors.db",
            self.meta_dir / "vault_vectors.db-wal",
            self.meta_dir / "vault_vectors.db-shm",
            self.manifest_file,
        ]:
            if db_f.exists():
                db_f.unlink(missing_ok=True)

        self.manifest = {}
        self._slug_owners = {}
        self._manifest_dirty = False
        return stats

    def ingest_document(
        self,
        file_path: Path,
        custom_title: Optional[str] = None,
        force: bool = False,
        quiet: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Ingests a single technical document into the vault with table, math, and structure preservation."""
        file_path = Path(file_path)
        if not self._is_eligible_source_file(file_path):
            return None

        norm_key = normalize_source_key(file_path, self.repo_root)
        legacy_abs_key = str(file_path.resolve())
        rel_key = f"source_materials/{file_path.name}"
        file_stat = file_path.stat()
        cached = (
            self.manifest.get(norm_key)
            or self.manifest.get(legacy_abs_key)
            or self.manifest.get(rel_key)
        )

        rel_hierarchy = ""
        extra_sub_parts: List[str] = []
        try:
            parts = norm_key.split("/")
            if "source_materials" in parts:
                idx = parts.index("source_materials")
                sub_parts = list(parts[idx + 1 : -1])
                if len(sub_parts) > 3:
                    extra_sub_parts = sub_parts[3:]
                    sub_parts = sub_parts[:3]
                if sub_parts:
                    rel_hierarchy = Path(*sub_parts)
        except Exception:
            pass

        target_concept_dir_check = (
            self.concepts_dir / rel_hierarchy if rel_hierarchy else self.concepts_dir
        )
        target_source_dir_check = (
            self.sources_dir / rel_hierarchy if rel_hierarchy else self.sources_dir
        )

        doc_name = custom_title if custom_title else file_path.stem
        if doc_name.lower().endswith(".slides"):
            doc_name = doc_name[:-7] + " slides"
        generic_names = {
            "readme",
            "package",
            "changelog",
            "cmakelists",
            "license",
            "index",
            "contributing",
            "todo",
            "version",
            "mainpage",
            "userdoc",
            "metadata",
            "model",
            "config",
            "params",
            "default",
            "example",
            "tutorial",
            "home",
        }
        if not custom_title:
            if extra_sub_parts:
                prefix_parts = [p for p in extra_sub_parts if p.lower() != doc_name.lower()]
                if prefix_parts:
                    doc_name = " ".join(prefix_parts[-2:] + [doc_name])
            elif doc_name.lower() in generic_names and len(file_path.parts) > 1:
                parent_folder = file_path.parts[-2]
                doc_name = f"{parent_folder} {doc_name}"

        clean_title = self.sanitize_title(doc_name)
        doc_slug = self.slugify(clean_title)

        if not force and cached:
            cached_slug = cached.get("slug", "")
            rel_h_key = Path(rel_hierarchy).as_posix() if rel_hierarchy else ""
            owner = self._slug_owners.get((rel_h_key, cached_slug))
            is_owner = owner in (norm_key, legacy_abs_key)
            notes_exist = bool(
                cached_slug
                and is_owner
                and (target_concept_dir_check / f"{cached_slug}-Overview.md").exists()
                and (target_source_dir_check / f"{cached_slug}-Source.md").exists()
            )
            if notes_exist:
                if (
                    cached.get("mtime") == file_stat.st_mtime
                    and cached.get("size") == file_stat.st_size
                ):
                    return None
                current_hash = self._compute_sha256(file_path)
                if cached.get("sha256") == current_hash:
                    self.manifest[norm_key] = cached
                    self.manifest[norm_key]["mtime"] = file_stat.st_mtime
                    self.manifest[norm_key]["size"] = file_stat.st_size
                    self._manifest_dirty = True
                    return None

        existing_overview = target_concept_dir_check / f"{doc_slug}-Overview.md"
        if existing_overview.exists():
            if cached and cached.get("slug"):
                doc_slug = cached["slug"]
            else:
                short_hash = hashlib.md5(norm_key.encode("utf-8")).hexdigest()[:6]
                doc_slug = f"{doc_slug[:58]}-{short_hash}"

        if not quiet:
            print(f"\n[Ingester] Processing: {clean_title} ({file_path.name})")

        ext = file_path.suffix.lower()
        full_text = ""
        total_pages = 1
        tables_found = 0
        engine_method = "plain_text"

        if ext == ".pdf":
            extract_res = extract_text_from_pdf(file_path)
            if not extract_res["pages"]:
                if not quiet:
                    print(f"  [Skipped] Could not extract text from {file_path.name}")
                return None
            full_text = "\n\n".join(extract_res["pages"])
            total_pages = extract_res["num_pages"]
            tables_found = extract_res.get("tables_found", 0)
            engine_method = extract_res.get("method", "pdf")
        elif ext == ".ipynb":
            extract_res = extract_ipynb(file_path)
            full_text = extract_res["text"]
            total_pages = extract_res["num_pages"]
            tables_found = extract_res["tables_found"]
            engine_method = extract_res["method"]
        elif ext == ".vtt":
            extract_res = extract_vtt_transcript(file_path)
            full_text = extract_res["text"]
            total_pages = extract_res["num_pages"]
            tables_found = extract_res["tables_found"]
            engine_method = extract_res["method"]
        elif ext == ".html":
            extract_res = extract_html_document(file_path)
            full_text = extract_res["text"]
            total_pages = extract_res["num_pages"]
            tables_found = extract_res["tables_found"]
            engine_method = extract_res["method"]
        elif ext in self.SUPPORTED_EXTS:
            extract_res = extract_structured_or_raw_document(file_path)
            full_text = extract_res["text"]
            total_pages = extract_res["num_pages"]
            tables_found = extract_res["tables_found"]
            engine_method = extract_res["method"]
        else:
            return None

        min_chars = (
            5
            if ext in {".msg", ".srv", ".action", ".idl", ".proto"}
            else 20
            if ext in {".cfg", ".launch", ".yaml", ".yml", ".xml", ".urdf", ".sdf", ".xacro"}
            else 50
        )
        if len(full_text.strip()) < min_chars:
            if not quiet:
                print("  [Skipped] Document text is too short or empty.")
            return None

        target_concept_dir = (
            self.concepts_dir / rel_hierarchy if rel_hierarchy else self.concepts_dir
        )
        target_source_dir = (
            self.sources_dir / rel_hierarchy if rel_hierarchy else self.sources_dir
        )
        target_concept_dir.mkdir(parents=True, exist_ok=True)
        target_source_dir.mkdir(parents=True, exist_ok=True)

        raw_paragraphs = [
            re.sub(r"\s+", " ", p).strip() for p in full_text.split("\n\n") if len(p.strip()) > 70
        ]
        valid_paragraphs = []
        for p in raw_paragraphs:
            p_lower = p.lower()
            if any(
                term in p_lower
                for term in [
                    "isbn",
                    "cataloguing",
                    "copyright",
                    "published by",
                    "all rights reserved",
                    "contents",
                    "table of contents",
                    "preface to the",
                    "printed in",
                    "department of electrical",
                    "university press",
                    "author index",
                    "subject index",
                    "cambridge university",
                    "stephen boyd",
                    "lieven vandenberghe",
                    "p. cm.",
                    "qa402",
                ]
            ):
                continue
            if re.search(r"\.\s*\.\s*\.\s*\.", p):
                continue
            cleaned_p = clean_text(p)
            if len(cleaned_p) > 80 and (cleaned_p.endswith(".") or "." in cleaned_p):
                valid_paragraphs.append(cleaned_p)

        source_note_name = f"{doc_slug}-Source.md"
        source_note_path = target_source_dir / source_note_name
        source_body = clean_text(full_text)
        if valid_paragraphs and valid_paragraphs[0] in source_body:
            idx_start = source_body.find(valid_paragraphs[0])
            if idx_start > 0:
                heading_idx = source_body.rfind("\n#", 0, idx_start)
                if heading_idx != -1 and (idx_start - heading_idx) < 200:
                    source_body = source_body[heading_idx:].strip()
                else:
                    source_body = source_body[idx_start:]

        source_content = f"""---
title: "{clean_title} - Primary Source Extract"
type: source
status: active
source_file: "{file_path.name}"
total_pages: {total_pages}
tables_preserved: {tables_found}
engine: "{engine_method}"
tags:
  - second-brain/source
  - reference/document
---

# {clean_title} - Source Extract

*Ingested from*: `{file_path.name}` ({total_pages} pages, {tables_found} tables preserved, engine: `{engine_method}`)

---

## Verbatim Extraction
{source_body}
"""
        source_note_path.write_text(source_content, encoding="utf-8")

        concept_note_name = f"{doc_slug}-Overview.md"
        concept_note_path = target_concept_dir / concept_note_name

        summary_intro = (
            "\n\n".join(valid_paragraphs[:2]) if valid_paragraphs else clean_text(full_text[:400])
        )
        takeaways = []
        for idx, p in enumerate(valid_paragraphs[:5]):
            first_sentence = p.split(". ")[0].strip()
            if len(first_sentence) > 20:
                detail = p.split(". ")[1][:140] if len(p.split(". ")) > 1 else ""
                takeaways.append(f"{idx+1}. **{first_sentence[:95]}**: {detail}")

        if not takeaways:
            takeaways = [
                f"1. **Core Problem**: Primary formulation and engineering specifications for {clean_title}.",
                f"2. **Governing Model**: Mathematical, interface, or algorithmic conventions defined in `{file_path.name}`.",
                f"3. **Operational Scope**: Ingests {total_pages} pages with {tables_found} structured data tables preserved.",
                "4. **Robotics Relevance**: Direct guidance for state estimation, navigation, and system control.",
                f"5. **Actionable Next Step**: Consult primary extract in [[{doc_slug}-Source]] for implementation details.",
            ]
        takeaways_md = "\n".join(takeaways[:5])

        concept_content = f"""---
title: "{clean_title} - Concept Overview"
type: concept
status: active
domain: robotics
tables_preserved: {tables_found}
engine: "{engine_method}"
tags:
  - second-brain/concept
  - concept/{doc_slug}
  - knowledge-base
---

# {clean_title}

## ⚡ Quick Reference & Executive Takeaways (Max 5 Actions)
{takeaways_md}

---

## 1. High-Density Executive Summary
{summary_intro}

---

## 2. Core Architecture & Verification Scope
- **Primary Source Extract**: [[{doc_slug}-Source]]
- **Document Scope**: {total_pages} pages / ~{len(full_text.split())} words
- **Preserved Tables**: {tables_found} structured tables
- **Ingestion Pipeline**: `{engine_method}`

```mermaid
flowchart TD
    Doc["{clean_title}"] --> S1["1. Mathematical Formulations & Standards"]
    Doc --> S2["2. Operational Interfaces & Dynamics"]
    Doc --> S3["3. Verification & Execution Criteria"]
```

---

## 3. Normative Technical Content & Formulas
{valid_paragraphs[2] if len(valid_paragraphs) > 2 else ''}

{valid_paragraphs[3] if len(valid_paragraphs) > 3 else ''}

---

## 🔗 Actionable Navigation & Related Links (Max 5)
- 📐 **Domain MOC**: [[{doc_slug}-MOC]]
- 🎙️ **Source Document**: [[{doc_slug}-Source]]
- 🗺️ **Master Index**: [[00-Meta/Index]]
"""
        concept_note_path.write_text(concept_content, encoding="utf-8")

        moc_name = f"{doc_slug}-MOC.md"
        moc_path = target_concept_dir / moc_name
        moc_content = f"""---
title: "{clean_title} - Map of Content (MOC)"
tags:
  - moc
  - {doc_slug}
---

# {clean_title} (MOC)

Map of Content for **{clean_title}**.

## Knowledge Navigation
- 📐 **Overview & Concepts**: [[{doc_slug}-Overview]]
- 🎙️ **Primary Text Extract**: [[{doc_slug}-Source]]
- 🗺️ **Master Vault Index**: [[00-Meta/Index]]
"""
        moc_path.write_text(moc_content, encoding="utf-8")

        self.register_in_index(clean_title, doc_slug, moc_name, flush=not quiet)

        try:
            self.manifest[norm_key] = {
                "mtime": file_stat.st_mtime,
                "size": file_stat.st_size,
                "sha256": self._compute_sha256(file_path),
                "slug": doc_slug,
            }
            rel_h_key = Path(rel_hierarchy).as_posix() if rel_hierarchy else ""
            self._slug_owners[(rel_h_key, doc_slug)] = norm_key
            self._manifest_dirty = True
        except Exception as e:
            print(f"  [Warning] Failed to update manifest entry for {file_path.name}: {e}")

        if not quiet:
            print(
                f"  [Success] Ingested into [[{doc_slug}-MOC]] "
                f"({total_pages} pages, {tables_found} tables preserved, engine: {engine_method})"
            )
        return {
            "title": clean_title,
            "slug": doc_slug,
            "concept_note": concept_note_name,
            "source_note": source_note_name,
            "moc": moc_name,
            "tables_found": tables_found,
            "engine": engine_method,
        }

    def register_in_index(self, title: str, slug: str, moc_name: str, flush: bool = True):
        """Appends new document MOC to 00-Meta/Index.local.md (git-ignored) if not already present."""
        local_index = self.meta_dir / "Index.local.md"
        if self._index_cache is None:
            if not local_index.exists():
                self._index_cache = "# 📚 Ingested Knowledge Bases (Local Index)\n\n"
                local_index.write_text(self._index_cache, encoding="utf-8")
            else:
                self._index_cache = local_index.read_text(encoding="utf-8")

        if f"[[{slug}-MOC" in self._index_cache or moc_name in self._index_cache:
            return

        wikilink_target = f"[[{slug}-MOC|{title}]]"
        new_entry = f"- 📘 {wikilink_target}: Complete concept overview and source extracts.\n"
        self._index_cache += new_entry
        self._index_dirty = True
        if flush:
            self._flush_index()

    def _flush_index(self):
        """Writes buffered Index.local.md updates to disk."""
        if getattr(self, "_index_dirty", False) and self._index_cache is not None:
            local_index = self.meta_dir / "Index.local.md"
            local_index.write_text(self._index_cache, encoding="utf-8")
            self._index_dirty = False

    def ingest_directory(
        self, source_dir: Path, force: bool = False, update_vectors: bool = True, quiet: bool = True
    ) -> List[Dict[str, Any]]:
        """Ingests all valid documents found within source_dir recursively."""
        source_dir = Path(source_dir)
        if not source_dir.exists():
            print(f"[Error] Source directory '{source_dir}' does not exist.")
            return []

        results = []
        for p in sorted(source_dir.rglob("*")):
            if not self._is_eligible_source_file(p):
                continue
            try:
                res = self.ingest_document(p, force=force, quiet=quiet)
                if res:
                    results.append(res)
                    if len(results) % 250 == 0:
                        print(f"  [Ingester Progress] {len(results)} documents ingested (latest: {p.name})...")
                        if self._manifest_dirty:
                            self._save_manifest()
                            self._manifest_dirty = False
                        self._flush_index()
            except Exception as e:
                print(f"  [Error] Failed to ingest {p.name}: {e}")

        if self._manifest_dirty:
            self._save_manifest()
            self._manifest_dirty = False
        self._flush_index()

        if update_vectors:
            try:
                from hybrid_search import HybridSearchEngine

                search_engine = HybridSearchEngine(self.vault_dir)
                indexed = search_engine.index_vault()
                if indexed > 0:
                    print(
                        f"[Ingester] Updated local hierarchical vector database: {indexed} notes indexed."
                    )
            except Exception as e:
                print(f"[Ingester] Vector update note: {e}")

        print(
            f"\n[Ingester] Completed: Ingested {len(results)} documents into vault at '{self.vault_dir}'"
        )
        return results


def main():
    parser = argparse.ArgumentParser(description="Universal Knowledge Base Ingestion Engine for 2ndBrain")
    parser.add_argument(
        "--source", required=True, help="Directory containing documents to ingest (PDFs, text, etc.)"
    )
    parser.add_argument("--vault", required=True, help="Target Obsidian vault directory")
    parser.add_argument("--force", action="store_true", help="Force re-ingestion of unchanged documents")

    args = parser.parse_args()
    ingester = UniversalKnowledgeIngester(args.vault)
    ingester.ingest_directory(Path(args.source), force=args.force)


if __name__ == "__main__":
    main()
