"""
ingest_knowledge.py - Universal Initial Knowledge Base Builder for 2ndBrain

Ingests ANY folder of documents (PDFs, Markdown, Text, etc.) into the Obsidian vault:
1. Detects text format (digital vector PDF vs. scanned raster image).
2. Table & Equation Preservation:
   - Uses pdfplumber to detect, extract, and format structured tables into clean Markdown tables (| Col | Col |).
   - Detects mathematical formulas and expressions, wrapping them in LaTeX blocks ($$ ... $$).
3. Neural OCR fallback (rapidocr + pypdfium2) for scanned pages.
4. Formulates structured concept notes in 01-Concepts/ with [[wikilinks]].
5. Stores primary reference extracts in 03-Sources/.
6. Dynamically registers new documents and MOCs in 00-Meta/Index.md.
"""

import sys
import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import pypdf

# Table extraction support
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# Optional OCR imports
try:
    import pypdfium2 as pdfium
    from rapidocr_onnxruntime import RapidOCR
    import numpy as np
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


BOILERPLATE_PATTERNS = [
    r'(?i)Authorized licensed use limited to:.*?\n',
    r'(?i)Downloaded by .*?\n',
    r'©\s*.*?\n',
    r'(?i)Cambridge University Press.*?\n',
    r'(?i)The Edinburgh Building, Cambridge.*?\n',
    r'(?i)Published in the United States.*?\n',
    r'(?i)Information on this title:.*?\n',
    r'(?i)Library of Congress Cataloguing-in-Publication data.*?\n',
    r'(?i)A catalogue record for this publication.*?\n',
    r'(?i)This publication is in copyright.*?\n',
    r'(?i)Subject to statutory exception.*?\n',
    r'(?i)no reproduction of any part may take place.*?\n',
    r'(?i)ISBN\s*(?:-13:)?\s*[\d-]+\s*(?:hardback|paperback|ebook)?\n?',
    r'(?i)QA\d+[\.\d\w-]+\s*\d{4}\n?',
    r'(?i)Department of Electrical Engineering.*?\n',
    r'(?i)Electrical Engineering Department.*?\n',
    r'(?i)Printed in the United Kingdom.*?\n',
    r'(?i)CambridgeUniversityPress.*?\n',
    r'(?i)First published \d{4}.*?\n',
    r'(?i)All rights reserved.*?\n',
    r'(?i)arXiv:\d+\.\d+v\d+.*?\n',
]

SLOP_PATTERNS = [
    r'(?i)\bhere\'?s the thing:?\s*',
    r'(?i)\bin today\'?s (?:fast-paced|world|landscape|robotics)\b',
    r'(?i)\bit\'?s not (?:just )?about [^,\n]+, it\'?s about [^.\n]+\.?\s*',
    r'(?i)\bthe future (?:of [^.\n]+)? isn\'?t coming, it\'?s already here\.?\s*',
    r'(?i)\blet me be clear:?\s*',
    r'(?i)\bhope this helps!?\s*',
    r'(?i)\bat the end of the day,?\s*',
    r'(?i)\bwithout further ado,?\s*',
    r'(?i)\bin this (?:section|chapter|document), we (?:will )?dive deep into\b',
]


def clean_text(text: str) -> str:
    """Removes boilerplate, copyright watermarks, and AI slop patterns."""
    if not text:
        return ""
    for pat in BOILERPLATE_PATTERNS:
        text = re.sub(pat, '', text)
    for pat in SLOP_PATTERNS:
        text = re.sub(pat, '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def format_markdown_table(table_rows: List[List[Optional[str]]]) -> str:
    """Converts a 2D list of extracted table cells into a clean GitHub Flavored Markdown table."""
    if not table_rows or len(table_rows) < 1:
        return ""

    cleaned_rows = []
    for row in table_rows:
        cleaned_row = [re.sub(r'[\r\n\t]+', ' ', (cell or '').strip()) for cell in row]
        if any(cleaned_row):
            cleaned_rows.append(cleaned_row)

    if not cleaned_rows:
        return ""

    num_cols = max(len(r) for r in cleaned_rows)
    padded_rows = [r + [''] * (num_cols - len(r)) for r in cleaned_rows]

    # Escape pipes inside cells
    for r in padded_rows:
        for idx in range(len(r)):
            r[idx] = r[idx].replace('|', '\\|')

    header = padded_rows[0]
    separator = ['---'] * num_cols

    md_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |"
    ]
    for r in padded_rows[1:]:
        md_lines.append("| " + " | ".join(r) + " |")

    return "\n\n" + "\n".join(md_lines) + "\n\n"


def preserve_equations_and_math(text: str) -> str:
    """
    Detects mathematical expressions, formulas, and Greek symbol sequences,
    preserving them in LaTeX math blocks ($$ ... $$ or $ ... $).
    """
    if not text:
        return ""

    math_symbols = r'[∑∫∂√≈≠≤≥±×÷ΔαβγδεθλμστωϕψΩΦ\^_{}]'
    
    # Check lines that look like mathematical formulas
    lines = text.splitlines()
    processed_lines = []
    in_math_block = False

    for line in lines:
        stripped = line.strip()
        
        # If line contains explicit LaTeX math delimiters, leave intact
        if stripped.startswith("$$") or stripped.startswith("\\["):
            processed_lines.append(line)
            continue

        # Detect standalone math formulas (e.g., equations with '=' and math operators, or fractions)
        has_symbols = bool(re.search(math_symbols, stripped))
        has_equality = "=" in stripped or "≈" in stripped or "∝" in stripped
        is_formula_candidate = (
            has_symbols and has_equality and 
            len(stripped) < 120 and 
            not stripped.startswith("#") and
            not stripped.startswith("-") and
            not stripped.startswith("|")
        )

        if is_formula_candidate and not in_math_block:
            # Wrap as LaTeX equation block
            clean_formula = re.sub(r'(\d+)\s*\^\s*(\d+)', r'\1^{\2}', stripped)
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
        # Check if line has 2+ column-like gaps (3 or more spaces or tabs)
        parts = [p.strip() for p in re.split(r'\t|\s{3,}', line.strip()) if p.strip()]
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


def extract_text_from_pdf(pdf_path: Path) -> Dict[str, Any]:
    """
    Extracts text from PDF with Table and Equation Preservation.
    1. Uses pdfplumber to extract structured tables into Markdown format and preserve math formulas.
    2. Falls back to pypdf with heuristic table formatting.
    3. If text is unreadable (scanned raster image), automatically applies local neural OCR.
    """
    pages_text = []
    num_pages = 0
    tables_found_total = 0

    # --- Strategy A: pdfplumber (High-Fidelity Tables & Text) ---
    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                num_pages = len(pdf.pages)
                sample_chars = 0
                for p_idx in range(min(5, num_pages)):
                    t = pdf.pages[p_idx].extract_text() or ""
                    sample_chars += len(t.strip())

                if sample_chars > 100:
                    print(f"    [pdfplumber] Extracting {num_pages} pages with Table & Equation preservation...")
                    for page in pdf.pages:
                        page_parts = []
                        # 1. Extract and format structured tables
                        tables = page.extract_tables() or []
                        table_md_blocks = []
                        if tables:
                            tables_found_total += len(tables)
                            for t in tables:
                                md_t = format_markdown_table(t)
                                if md_t:
                                    table_md_blocks.append(md_t)

                        # 2. Extract prose text with layout preservation
                        prose = page.extract_text(layout=True) or ""
                        prose_clean = clean_text(prose)
                        prose_with_math = preserve_equations_and_math(prose_clean)

                        page_parts.append(prose_with_math)
                        if table_md_blocks:
                            page_parts.append("\n### Extracted Tables\n" + "".join(table_md_blocks))

                        pages_text.append("\n\n".join(page_parts))

                    return {
                        "pages": pages_text, 
                        "method": "pdfplumber_with_tables", 
                        "num_pages": num_pages,
                        "tables_found": tables_found_total
                    }
        except Exception as e:
            print(f"    [pdfplumber fallback to pypdf]: {e}")

    # --- Strategy B: pypdf (Digital Text with Heuristic Tables & Math) ---
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        num_pages = len(reader.pages)
        sample_chars = 0
        for p in range(min(5, num_pages)):
            t = reader.pages[p].extract_text() or ""
            sample_chars += len(t.strip())

        if sample_chars > 100:
            print(f"    [pypdf] Extracting {num_pages} pages with heuristic table and math preservation...")
            for page in reader.pages:
                t = clean_text(page.extract_text() or "")
                t_tables = detect_heuristic_tables(t)
                t_math = preserve_equations_and_math(t_tables)
                pages_text.append(t_math)
            return {"pages": pages_text, "method": "pypdf_digital_text", "num_pages": num_pages, "tables_found": 0}
    except Exception:
        pass

    # Check if pre-extracted OCR cache exists
    cached_candidates = [
        pdf_path.with_name(f"{pdf_path.stem}_extracted.txt"),
        pdf_path.with_name(f"{pdf_path.stem.replace(' ', '_')}_extracted.txt"),
        pdf_path.with_name(f"{pdf_path.stem.replace('_', ' ')}_extracted.txt")
    ]
    for cached_txt in cached_candidates:
        if cached_txt.exists():
            print(f"    [Cached OCR Text Found] Loading {cached_txt.name}...")
            t = cached_txt.read_text(encoding="utf-8", errors="ignore")
            t_math = preserve_equations_and_math(detect_heuristic_tables(clean_text(t)))
            return {"pages": [t_math], "method": "cached_ocr", "num_pages": num_pages, "tables_found": 0}

    # Otherwise, fallback to local neural OCR
    if not OCR_AVAILABLE:
        print(f"    [Warning] PDF '{pdf_path.name}' is scanned, but OCR libraries (pypdfium2, rapidocr) are not installed.")
        return {"pages": [], "method": "unreadable_scanned", "num_pages": num_pages, "tables_found": 0}

    print(f"    [Scanned PDF Detected] Running local neural OCR on {num_pages} pages...")
    ocr = RapidOCR()
    pdf_doc = pdfium.PdfDocument(str(pdf_path))
    ocr_pages = []

    for i in range(num_pages):
        try:
            img = pdf_doc[i].render(scale=1.5).to_pil()
            result, _ = ocr(np.array(img))
            lines = [item[1] for item in result] if result else []
            ocr_text = "\n".join(lines)
            ocr_text = preserve_equations_and_math(detect_heuristic_tables(clean_text(ocr_text)))
            ocr_pages.append(ocr_text)
        except Exception:
            ocr_pages.append("")

    return {"pages": ocr_pages, "method": "neural_ocr", "num_pages": num_pages, "tables_found": 0}


class UniversalKnowledgeIngester:
    def __init__(self, vault_dir: str):
        self.vault_dir = Path(vault_dir).resolve()
        self.concepts_dir = self.vault_dir / "01-Concepts"
        self.sources_dir = self.vault_dir / "03-Sources"
        self.meta_dir = self.vault_dir / "00-Meta"
        self.index_file = self.meta_dir / "Index.md"

        for d in [self.concepts_dir, self.sources_dir, self.meta_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def sanitize_title(self, name: str) -> str:
        """Converts filename into clean title and directory slug."""
        clean = re.sub(r'[\(\)\[\]\{\}_]', ' ', name).strip()
        clean = re.sub(r'\s+', ' ', clean)
        return clean

    def slugify(self, text: str) -> str:
        s = re.sub(r'[^\w\s-]', '', text).strip()
        slug = re.sub(r'[-\s]+', '-', s)
        return slug[:65].rstrip('-')

    def ingest_document(self, file_path: Path, custom_title: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Ingests a single document (.pdf, .txt, .md) into the vault with table & equation preservation."""
        doc_name = custom_title if custom_title else file_path.stem
        clean_title = self.sanitize_title(doc_name)
        doc_slug = self.slugify(clean_title)

        print(f"\n[Ingester] Processing: {clean_title} ({file_path.name})")

        full_text = ""
        total_pages = 1
        tables_found = 0

        if file_path.suffix.lower() == ".pdf":
            extract_res = extract_text_from_pdf(file_path)
            if not extract_res["pages"]:
                print(f"  [Skipped] Could not extract text from {file_path.name}")
                return None
            full_text = "\n\n".join(extract_res["pages"])
            total_pages = extract_res["num_pages"]
            tables_found = extract_res.get("tables_found", 0)
        elif file_path.suffix.lower() in [".txt", ".md", ".rst"]:
            try:
                raw_content = file_path.read_text(encoding="utf-8", errors="ignore")
                full_text = preserve_equations_and_math(detect_heuristic_tables(raw_content))
            except Exception as e:
                print(f"  [Error] Reading {file_path.name}: {e}")
                return None
        else:
            print(f"  [Skipped] Unsupported format: {file_path.suffix}")
            return None

        if len(full_text.strip()) < 50:
            print(f"  [Skipped] Document text is too short or empty.")
            return None

        # Create target directories
        target_concept_dir = self.concepts_dir / doc_slug
        target_source_dir = self.sources_dir / doc_slug
        target_concept_dir.mkdir(parents=True, exist_ok=True)
        target_source_dir.mkdir(parents=True, exist_ok=True)

        # 1. Clean and filter paragraphs for genuine conceptual signal
        raw_paragraphs = [re.sub(r'\s+', ' ', p).strip() for p in full_text.split("\n\n") if len(p.strip()) > 70]
        valid_paragraphs = []
        for p in raw_paragraphs:
            p_lower = p.lower()
            # Filter out copyright, cataloguing, and publisher frontmatter
            if any(term in p_lower for term in [
                "isbn", "cataloguing", "copyright", "published by", "all rights reserved",
                "contents", "table of contents", "preface to the", "printed in",
                "department of electrical", "university press", "author index", "subject index",
                "cambridge university", "stephen boyd", "lieven vandenberghe", "p. cm.", "qa402"
            ]):
                continue
            if re.search(r'\.\s*\.\s*\.\s*\.', p): # TOC dots
                continue
            cleaned_p = clean_text(p)
            if len(cleaned_p) > 80 and (cleaned_p.endswith('.') or '.' in cleaned_p):
                valid_paragraphs.append(cleaned_p)

        # 1. Write Raw Source File in 03-Sources/
        source_note_name = f"{doc_slug}-Source.md"
        source_note_path = target_source_dir / source_note_name
        source_body = clean_text(full_text[:35000])
        if valid_paragraphs and valid_paragraphs[0] in source_body:
            idx_start = source_body.find(valid_paragraphs[0])
            if idx_start > 0:
                source_body = source_body[idx_start:]

        source_content = f"""---
title: "{clean_title} - Primary Source Extract"
type: source
status: active
source_file: "{file_path.name}"
total_pages: {total_pages}
tables_preserved: {tables_found}
tags:
  - second-brain/source
  - reference/document
---

# {clean_title} - Source Extract

*Ingested from*: `{file_path.name}` ({total_pages} pages, {tables_found} tables preserved)

---

## Verbatim Extraction
{source_body}
"""
        source_note_path.write_text(source_content, encoding="utf-8")

        # 2. Extract Headings and Core Concept Notes in 01-Concepts/
        concept_note_name = f"{doc_slug}-Overview.md"
        concept_note_path = target_concept_dir / concept_note_name
        
        # Build "I Have ADHD" Action-First Executive Takeaways (Max 5 items)
        summary_intro = "\n\n".join(valid_paragraphs[:2]) if valid_paragraphs else clean_text(full_text[:400])
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
                f"4. **Robotics Relevance**: Direct guidance for state estimation, navigation, and system control.",
                f"5. **Actionable Next Step**: Consult primary extract in [[{doc_slug}-Source]] for implementation details."
            ]
        takeaways_md = "\n".join(takeaways[:5])

        concept_content = f"""---
title: "{clean_title} - Concept Overview"
type: concept
status: active
domain: robotics
tables_preserved: {tables_found}
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

        # 3. Create Map of Content (MOC)
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

        # 4. Register in 00-Meta/Index.md
        self.register_in_index(clean_title, doc_slug, moc_name)

        print(f"  [Success] Ingested into [[{doc_slug}-MOC]] ({total_pages} pages, {tables_found} tables preserved)")
        return {
            "title": clean_title,
            "slug": doc_slug,
            "concept_note": concept_note_name,
            "source_note": source_note_name,
            "moc": moc_name,
            "tables_found": tables_found
        }

    def register_in_index(self, title: str, slug: str, moc_name: str):
        """Appends new document MOC to 00-Meta/Index.local.md (git-ignored) if not already present."""
        local_index = self.meta_dir / "Index.local.md"
        if not local_index.exists():
            local_index.write_text("# 📚 Ingested Knowledge Bases (Local Index)\n\n", encoding="utf-8")

        content = local_index.read_text(encoding="utf-8")
        wikilink_target = f"[[{slug}-MOC|{title}]]"
        
        if slug in content or moc_name in content:
            return  # Already registered

        new_entry = f"- 📘 {wikilink_target}: Complete concept overview and source extracts.\n"
        local_index.write_text(content + new_entry, encoding="utf-8")

    def ingest_directory(self, source_dir: Path) -> List[Dict[str, Any]]:
        """Ingests all valid documents found within source_dir recursively."""
        source_dir = Path(source_dir)
        if not source_dir.exists():
            print(f"[Error] Source directory '{source_dir}' does not exist.")
            return []

        results = []
        supported_exts = {".pdf", ".txt", ".md", ".rst"}

        for p in sorted(source_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in supported_exts:
                # Skip temporary files and pre-extracted raw text sidecars
                if p.name.startswith("~") or p.name.startswith("."):
                    continue
                if p.name.endswith("_extracted.txt") or p.name.endswith("_extracted.json"):
                    continue
                res = self.ingest_document(p)
                if res:
                    results.append(res)

        # Auto-update local hierarchical vector embeddings for newly ingested notes
        try:
            from hybrid_search import HybridSearchEngine
            search_engine = HybridSearchEngine(self.vault_dir)
            indexed = search_engine.index_vault()
            if indexed > 0:
                print(f"[Ingester] Updated local hierarchical vector database: {indexed} notes indexed.")
        except Exception as e:
            print(f"[Ingester] Vector update note: {e}")

        print(f"\n[Ingester] Completed: Ingested {len(results)} documents into vault at '{self.vault_dir}'")
        return results


def main():
    parser = argparse.ArgumentParser(description="Universal Knowledge Base Ingestion Engine for 2ndBrain")
    parser.add_argument("--source", required=True, help="Directory containing documents to ingest (PDFs, text, etc.)")
    parser.add_argument("--vault", required=True, help="Target Obsidian vault directory")

    args = parser.parse_args()
    ingester = UniversalKnowledgeIngester(args.vault)
    ingester.ingest_directory(Path(args.source))


if __name__ == "__main__":
    main()
