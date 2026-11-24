"""Generic Living Engineering Documentation parser (Sphinx RST + Markdown/MyST/MkDocs).

Supports arbitrary versioned documentation repositories without hardcoding specific frameworks.
Resolves conf.py substitutions, version tab directives, and include/literalinclude files
with explicit UNRESOLVED_INCLUDE error reporting (zero silent drops).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from docutils.frontend import get_default_settings
from docutils.parsers.rst import Parser as RstParser
from docutils.utils import new_document
from markdown_it import MarkdownIt
import tiktoken

from second_brain.config import BrainConfig
from second_brain.storage.sqlite_store import SqliteStore, atomic_write_lf, lf_sha256

_ENC = tiktoken.get_encoding("o200k_base")
_MD = MarkdownIt("commonmark")

CONF_SUBST_RE = re.compile(r"\.\.\s+\|([A-Za-z0-9_.-]+)\|\s+replace::\s+(.+)")
CONF_VAR_RE = re.compile(r"^([A-Za-z0-9_]+)\s*=\s*['\"]([^'\"]+)['\"]\s*$", re.MULTILINE)
INCLUDE_RE = re.compile(r"^(\s*)\.\.\s+(include|literalinclude)::\s+(.+?)\s*$")
TABS_DIRECTIVE_RE = re.compile(r"^(\s*)\.\.\s+(tabs|group-tab|code-tab|tab|only)::\s*(.*)$")
MD_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)#\s]+\.(?:md|rst))(?:#[^)]*)?\)")


def _slugify(value: str, max_len: int = 48) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip()).strip("_.")
    if not clean:
        clean = "doc"
    if len(clean) > max_len:
        h = hashlib.sha1(clean.encode("utf-8")).hexdigest()[:8]
        clean = f"{clean[: max_len - 9]}_{h}"
    return clean.lower()


def load_conf_substitutions(docs_root: Path, source_version: str) -> dict[str, str]:
    subs: dict[str, str] = {
        "DISTRO": source_version,
        "distro": source_version,
        "DISTRO_TITLE": source_version.title(),
        "VERSION": source_version,
    }
    for conf in docs_root.rglob("conf.py"):
        text = conf.read_text(encoding="utf-8", errors="replace")
        for m in CONF_SUBST_RE.finditer(text):
            subs[m.group(1)] = m.group(2).strip()
        for m in CONF_VAR_RE.finditer(text):
            key, val = m.group(1), m.group(2).strip()
            if key.upper() in ("DISTRO", "VERSION", "RELEASE"):
                subs[key] = val
    return subs


def _apply_substitutions(text: str, subs: dict[str, str]) -> tuple[str, int]:
    applied = 0
    for key, val in subs.items():
        for pat in (f"|{key}|", f"{{{key}}}"):
            if pat in text:
                applied += text.count(pat)
                text = text.replace(pat, val)
    return text, applied


def _filter_version_tabs(lines: list[str], source_version: str) -> tuple[list[str], int]:
    """Keep active source_version blocks inside .. tabs:: / .. group-tab:: / .. only:: directives."""
    out: list[str] = []
    i = 0
    directives_resolved = 0
    target_lower = source_version.lower()

    while i < len(lines):
        line = lines[i]
        m = TABS_DIRECTIVE_RE.match(line)
        if not m:
            out.append(line)
            i += 1
            continue

        indent = len(m.group(1))
        kind = m.group(2)
        arg = m.group(3).strip().lower()
        directives_resolved += 1

        if kind == "tabs":
            i += 1
            continue

        # Collect indented body of this tab / only block
        i += 1
        block: list[str] = []
        while i < len(lines):
            nxt = lines[i]
            if nxt.strip() == "":
                block.append(nxt)
                i += 1
                continue
            nxt_indent = len(nxt) - len(nxt.lstrip(" "))
            if nxt_indent <= indent:
                break
            if nxt.lstrip().startswith(":"):
                # Directive option line inside tab
                i += 1
                continue
            block.append(nxt[min(len(nxt), indent + 3) :] if nxt.startswith(" " * (indent + 3)) else nxt.lstrip())
            i += 1

        # Decide whether block matches target source_version
        tokens = [t.strip() for t in re.split(r"[\s,/|]+", arg) if t.strip()]
        matches = (not tokens) or (target_lower in tokens) or ("all" in tokens) or any(
            target_lower == tok for tok in tokens
        )
        # If tag mentions common non-matching version names explicitly and excludes target, drop it
        if matches:
            out.extend(block)
    return out, directives_resolved


def _resolve_includes(
    lines: list[str],
    file_path: Path,
    docs_root: Path,
    unresolved: list[dict[str, str]],
) -> tuple[list[str], int]:
    out: list[str] = []
    resolved_cnt = 0
    for line in lines:
        m = INCLUDE_RE.match(line)
        if not m:
            out.append(line)
            continue
        kind = m.group(2)
        target_rel = m.group(3).strip()
        if target_rel.startswith("<") and target_rel.endswith(">"):
            out.append(line)
            continue

        candidate = (file_path.parent / target_rel).resolve()
        if not candidate.exists():
            alt = (docs_root / target_rel.lstrip("/")).resolve()
            candidate = alt if alt.exists() else candidate

        if not candidate.exists():
            unresolved.append(
                {
                    "file": file_path.relative_to(docs_root).as_posix(),
                    "directive": kind,
                    "target": target_rel,
                    "error": "UNRESOLVED_INCLUDE",
                }
            )
            out.append(f"[UNRESOLVED_INCLUDE: {kind} {target_rel}]")
            continue

        resolved_cnt += 1
        inc_text = candidate.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
        if kind == "literalinclude":
            lang = candidate.suffix.lstrip(".") or "text"
            out.append(f"```{lang}")
            out.extend(inc_text.splitlines())
            out.append("```")
        else:
            out.extend(inc_text.splitlines())
    return out, resolved_cnt


def _split_rst_sections(text: str, doc_title: str) -> list[tuple[str, str]]:
    """Split RST using docutils parser validation + underline heading segmentation."""
    settings = get_default_settings(RstParser)
    settings.report_level = 5
    settings.halt_level = 6
    doc = new_document("<rst>", settings=settings)
    RstParser().parse(text, doc)

    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    cur_title = doc_title
    cur_lines: list[str] = []
    i = 0
    while i < len(lines):
        if i + 1 < len(lines) and lines[i].strip() and re.match(r"^[=\-~^\"`#*+]{3,}$", lines[i + 1].strip()):
            if len(lines[i + 1].strip()) >= len(lines[i].strip()):
                if cur_lines:
                    sections.append((cur_title, cur_lines))
                cur_title = lines[i].strip()
                cur_lines = [f"## {cur_title}"]
                i += 2
                continue
        cur_lines.append(lines[i])
        i += 1
    if cur_lines:
        sections.append((cur_title, cur_lines))
    return [(t, "\n".join(ls).strip()) for t, ls in sections if "\n".join(ls).strip()]


def _split_md_sections(text: str, doc_title: str) -> list[tuple[str, str]]:
    _MD.parse(text)
    sections: list[tuple[str, list[str]]] = []
    cur_title = doc_title
    cur_lines: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        if not in_fence:
            hm = re.match(r"^(#{1,3})\s+(.+)$", line.strip())
            if hm:
                if cur_lines:
                    sections.append((cur_title, cur_lines))
                cur_title = hm.group(2).strip()
                cur_lines = [line]
                continue
        cur_lines.append(line)
    if cur_lines:
        sections.append((cur_title, cur_lines))
    return [(t, "\n".join(ls).strip()) for t, ls in sections if "\n".join(ls).strip()]


def ingest_living_docs_dir(
    *,
    cfg: BrainConfig,
    store: SqliteStore,
    docs_root: Path,
    domain: str,
    source_id: str,
    source_version: str,
) -> dict[str, Any]:
    """Ingest any versioned Sphinx RST or Markdown engineering documentation tree."""
    if not source_version.strip():
        raise ValueError("source_version is mandatory for living engineering documentation.")
    if not docs_root.exists():
        raise FileNotFoundError(f"Living docs root not found: {docs_root}")

    gen = store.next_index_generation()
    subs = load_conf_substitutions(docs_root, source_version)
    safe_src = _slugify(source_id, 32)
    safe_ver = _slugify(source_version, 24)
    out_dir = cfg.parsed_dir / f"{safe_src}_{safe_ver}"
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_files = sorted(
        [p for p in docs_root.rglob("*") if p.is_file() and p.suffix.lower() in (".rst", ".md")],
        key=lambda p: p.as_posix(),
    )

    unresolved_includes: list[dict[str, str]] = []
    directives_resolved = 0
    substitutions_applied = 0
    chunks_written = 0
    edges_written = 0

    for fpath in doc_files:
        rel_doc = fpath.relative_to(docs_root).as_posix()
        raw = fpath.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
        lines, res_inc = _resolve_includes(raw.splitlines(), fpath, docs_root, unresolved_includes)
        directives_resolved += res_inc
        lines, tab_cnt = _filter_version_tabs(lines, source_version)
        directives_resolved += tab_cnt
        processed, sub_cnt = _apply_substitutions("\n".join(lines), subs)
        substitutions_applied += sub_cnt

        doc_slug = _slugify(rel_doc.replace("/", "_"), 44)
        out_name = f"{doc_slug}.md"
        out_path = out_dir / out_name
        out_rel = f"parsed/{safe_src}_{safe_ver}/{out_name}"

        if fpath.suffix.lower() == ".rst":
            secs = _split_rst_sections(processed, fpath.stem)
        else:
            secs = _split_md_sections(processed, fpath.stem)
        if not secs and processed.strip():
            secs = [(fpath.stem, processed.strip())]

        md_lines = [
            "---",
            f"source_id: {source_id}",
            f"source_version: {source_version}",
            f"domain: {domain}",
            f"source_file: {rel_doc}",
            "---",
            "",
        ]
        spans: list[tuple[str, str, int, int, str]] = []
        for idx, (sec_title, sec_body) in enumerate(secs):
            sec_id = _slugify(f"{doc_slug}_{idx}_{sec_title}", 40)
            crumb = f"{source_id} ({source_version}) > {rel_doc} > {sec_title}"
            block = f"<!-- breadcrumb: {crumb} -->\n{sec_body}\n"
            start_ln = len(md_lines) + 1
            b_lines = block.splitlines()
            md_lines.extend(b_lines)
            md_lines.append("")
            end_ln = start_ln + len(b_lines) - 1
            spans.append((sec_id, crumb, start_ln, end_ln, block))

        atomic_write_lf(out_path, "\n".join(md_lines))
        doc_root_uri = f"vault://{safe_src}@{safe_ver}#{doc_slug}"

        for sec_id, crumb, start_ln, end_ln, block in spans:
            uri = f"vault://{safe_src}@{safe_ver}#{sec_id}"
            tok_cnt = len(_ENC.encode(block))
            embed_key = lf_sha256(f"{cfg.vault_path.name}:bge-small-v1.5:v1:{block}")
            store.upsert_chunk(
                uri=uri,
                store="parsed",
                domain=domain,
                source_id=source_id,
                source_version=source_version,
                clause_path=crumb,
                parent_uri=doc_root_uri if uri != doc_root_uri else None,
                authority_tier="primary_reference",
                file_relpath=out_rel,
                start_line=start_ln,
                end_line=end_ln,
                text=block,
                token_count=tok_cnt,
                embed_key=embed_key,
                index_generation=gen,
                commit=False,
            )
            chunks_written += 1
            if uri != doc_root_uri:
                store.upsert_edge(uri, doc_root_uri, "PART_OF", "EXTRACTED", 1.0, commit=False)
                edges_written += 1

            for m in MD_LINK_RE.finditer(block):
                target = _slugify(m.group(1).replace("/", "_"), 44)
                dst = f"vault://{safe_src}@{safe_ver}#{target}"
                if dst != uri:
                    store.upsert_edge(uri, dst, "CITES", "EXTRACTED", 1.0, commit=False)
                    edges_written += 1

    store.conn.commit()
    report = {
        "source_id": source_id,
        "source_version": source_version,
        "domain": domain,
        "files_processed": len(doc_files),
        "chunks_written": chunks_written,
        "edges_written": edges_written,
        "directives_resolved": directives_resolved,
        "substitutions_applied": substitutions_applied,
        "unresolved_includes": unresolved_includes,
    }
    atomic_write_lf(out_dir / "_ingest_report.json", json.dumps(report, indent=2))
    return report
