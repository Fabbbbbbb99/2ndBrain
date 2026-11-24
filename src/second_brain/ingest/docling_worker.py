"""Isolated subprocess worker for optional Docling TableFormer table upgrades.

Never imported by braind. Runs out-of-process only when optional `docling` is installed,
with OCR and formula enrichment strictly disabled to prevent RAM spikes on 15.3 GB hosts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Out-of-process selective Docling table parser")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--pages", required=True, help="Comma-separated 1-based page numbers")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError:
        print(json.dumps({"available": False, "error": "docling is not installed in this environment"}))
        return 3

    page_nums = sorted({int(p.strip()) for p in args.pages.split(",") if p.strip()})
    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_formula_enrichment = False
    opts.do_table_structure = True

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    results: dict[str, str] = {}
    for page_num in page_nums:
        res = converter.convert(args.pdf, page_range=(page_num, page_num))
        results[str(page_num)] = res.document.export_to_markdown()

    Path(args.out).write_text(json.dumps({"available": True, "pages": results}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
