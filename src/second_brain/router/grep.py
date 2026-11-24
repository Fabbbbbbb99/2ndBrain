"""Exact literal and ReDoS-guarded regex verbatim search over parsed/ and wiki/ chunks."""
from __future__ import annotations

import re
import time
from typing import Any

import regex as regex_mod

from second_brain.config import BrainConfig
from second_brain.ingest.pdf_adapter import _strip_breadcrumb_header
from second_brain.observability.tracer import TraceLogger, new_trace_id
from second_brain.router.packer import count_tokens
from second_brain.router.retrieval import AUTHORITY_BOOST, _structural_noise_multiplier
from second_brain.storage.models import EvidenceEnvelope, GrepMatch, GrepOverflowEntry, GrepResult
from second_brain.storage.sqlite_store import SqliteStore

_AUTHORITY_RANK = {
    "primary_normative": 0,
    "primary_reference": 1,
    "compiled_synthesis": 2,
    "working_memory": 3,
}
_BACKREF_RE = re.compile(r"\\(?:[1-9]|k<|g<)|\(\?P=")
_LOOKAROUND_RE = re.compile(r"\(\?(?:[=!]|<[=!])")
_REGEX_META = set(r"\.^$*+?{}[]|()")
_SAFE_GREP_WINDOW_CHARS = 240
_PER_CHUNK_REGEX_TIMEOUT_S = 0.05
_TOTAL_GREP_DEADLINE_S = 1.0


def _extract_safe_literal_anchor(pattern: str) -> str | None:
    """Return a mandatory >=3-char literal run when safe from alternation/optionality."""
    if any(ch in pattern for ch in "|?*[]{}"):
        return None
    runs: list[str] = []
    buf: list[str] = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\" and i + 1 < len(pattern):
            nxt = pattern[i + 1]
            if nxt in _REGEX_META or nxt in " -_":
                buf.append(nxt)
            else:
                if buf:
                    runs.append("".join(buf))
                    buf.clear()
            i += 2
            continue
        if ch in _REGEX_META:
            if buf:
                runs.append("".join(buf))
                buf.clear()
        else:
            buf.append(ch)
        i += 1
    if buf:
        runs.append("".join(buf))
    candidates = [r for r in runs if len(r) >= 3]
    return max(candidates, key=len) if candidates else None


def _snap_window_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Snap window start/end cleanly to nearby whitespace or newline boundaries."""
    n = len(text)
    s = max(0, start)
    e = min(n, end)
    if s > 0:
        for idx in range(s, max(0, s - 32), -1):
            if text[idx - 1] in "\n \t":
                s = idx
                break
    if e < n:
        for idx in range(e, min(n, e + 32)):
            if text[idx] in "\n \t":
                e = idx
                break
    return s, e


def _build_coalesced_snippet(text: str, spans: list[tuple[int, int]]) -> tuple[str, int]:
    """Merge +-240-char windows around match spans, returning up to 2 excerpts and uncovered count."""
    windows: list[list[int]] = []
    for st, en in spans:
        ws, we = _snap_window_bounds(text, st - _SAFE_GREP_WINDOW_CHARS, en + _SAFE_GREP_WINDOW_CHARS)
        if windows and ws <= windows[-1][1] + 32:
            windows[-1][1] = max(windows[-1][1], we)
        else:
            windows.append([ws, we])

    kept = windows[:2]
    parts: list[str] = []
    for ws, we in kept:
        excerpt = text[ws:we].strip()
        prefix = "... " if ws > 0 else ""
        suffix = " ..." if we < len(text) else ""
        parts.append(f"{prefix}{excerpt}{suffix}")

    uncovered = sum(
        1
        for st, en in spans
        if not any(ws <= st and en <= we for ws, we in kept)
    )
    return "\n...\n".join(parts), uncovered


def execute_grep(
    *,
    cfg: BrainConfig,
    store: SqliteStore,
    tracer: TraceLogger,
    pattern: str,
    domain_scope: list[str] | None = None,
    source_id: str | None = None,
    source_version: str | None = None,
    regex: bool = False,
    case_sensitive: bool = False,
    max_matches: int = 10,
    token_budget: int | None = None,
    parent_trace_id: str | None = None,
    harness_id: str = "cli",
) -> GrepResult:
    """Run literal or ReDoS-guarded regex search across parsed/ and wiki/ chunks."""
    t0 = time.perf_counter()
    trace_id = new_trace_id()
    if not pattern or not pattern.strip():
        raise ValueError("INVALID_PATTERN: pattern must be a non-empty string.")

    budget = max(256, min(int(token_budget or cfg.pack_token_budget), 8000))
    limit = max(1, min(int(max_matches), 25))
    src_id = source_id.strip() if source_id and source_id.strip() else None
    src_ver = source_version.strip() if source_version and source_version.strip() else None

    compiled_re: Any = None
    if regex:
        if len(pattern) > 200:
            raise ValueError("REGEX_UNSAFE: regex pattern exceeds 200-character limit.")
        if _BACKREF_RE.search(pattern) or _LOOKAROUND_RE.search(pattern):
            raise ValueError("REGEX_UNSAFE: backreferences and lookarounds are disallowed in brain_grep.")
        flags = 0 if case_sensitive else regex_mod.IGNORECASE
        try:
            compiled_re = regex_mod.compile(pattern, flags=flags)
        except Exception as exc:
            raise ValueError(f"INVALID_REGEX: {exc}") from exc

    clauses = ["store IN ('parsed', 'wiki')"]
    params: list[Any] = []
    if domain_scope:
        ph = ",".join("?" for _ in domain_scope)
        clauses.append(f"domain IN ({ph})")
        params.extend(domain_scope)
    if src_id:
        clauses.append("LOWER(source_id) = LOWER(?)")
        params.append(src_id)
    if src_ver:
        clauses.append("(source_version = ? OR source_version = 'global')")
        params.append(src_ver)

    if not regex:
        if case_sensitive:
            clauses.append("INSTR(text, ?) > 0")
            params.append(pattern)
        else:
            clauses.append("INSTR(LOWER(text), LOWER(?)) > 0")
            params.append(pattern)
    else:
        anchor = _extract_safe_literal_anchor(pattern)
        if anchor:
            if case_sensitive:
                clauses.append("INSTR(text, ?) > 0")
                params.append(anchor)
            else:
                clauses.append("INSTR(LOWER(text), LOWER(?)) > 0")
                params.append(anchor)

    where_sql = " AND ".join(clauses)
    rows = store.conn.execute(
        f"""
        SELECT uri, store, domain, source_id, source_version, clause_path, authority_tier,
               file_relpath, start_line, end_line, content_sha256, text, token_count
        FROM chunks
        WHERE {where_sql}
        """,
        params,
    ).fetchall()

    matched: list[dict[str, Any]] = []
    total_occurrences = 0
    deadline_started = time.monotonic()
    needle_cmp = pattern if case_sensitive else pattern.lower()

    for r in rows:
        raw_text = str(r["text"])
        stripped = _strip_breadcrumb_header(raw_text)
        prefix_len = len(raw_text) - len(stripped)
        bc_lines = raw_text[:prefix_len].count("\n") if prefix_len > 0 else 0

        spans: list[tuple[int, int]] = []
        if not regex:
            hay = stripped if case_sensitive else stripped.lower()
            idx = 0
            step = max(1, len(needle_cmp))
            while True:
                pos = hay.find(needle_cmp, idx)
                if pos == -1:
                    break
                spans.append((pos, pos + len(pattern)))
                idx = pos + step
        else:
            if (time.monotonic() - deadline_started) > _TOTAL_GREP_DEADLINE_S:
                raise ValueError("REGEX_TIMEOUT: brain_grep exceeded 1.0s scan deadline.")
            try:
                for m in compiled_re.finditer(stripped, timeout=_PER_CHUNK_REGEX_TIMEOUT_S):
                    if m.end() == m.start():
                        continue
                    spans.append((m.start(), m.end()))
            except TimeoutError as exc:
                raise ValueError(f"REGEX_TIMEOUT: pattern exceeded {_PER_CHUNK_REGEX_TIMEOUT_S}s on chunk {r['uri']}.") from exc

        if not spans:
            continue

        total_occurrences += len(spans)
        env = EvidenceEnvelope(
            uri=r["uri"],
            store=r["store"],
            authority_tier=r["authority_tier"],
            domain=r["domain"],
            source_id=r["source_id"],
            source_version=r["source_version"],
            clause_path=r["clause_path"],
            text=raw_text,
            content_sha256=r["content_sha256"],
            token_count=int(r["token_count"]),
        )
        noise_mult = _structural_noise_multiplier(env)
        first_line_text = next((ln.strip() for ln in stripped.splitlines() if ln.strip()), "")
        cpath = str(r["clause_path"] or "")
        if not regex:
            c_cmp = cpath if case_sensitive else cpath.lower()
            f_cmp = first_line_text if case_sensitive else first_line_text.lower()
            heading_hit = (needle_cmp in c_cmp) or (needle_cmp in f_cmp)
        else:
            heading_hit = bool(compiled_re.search(cpath) or compiled_re.search(first_line_text))

        first_offset_lines = stripped[: spans[0][0]].count("\n")
        match_line = (
            int(r["start_line"]) + bc_lines + first_offset_lines
            if r["start_line"] is not None
            else (first_offset_lines + 1)
        )
        matched.append(
            {
                "row": r,
                "stripped": stripped,
                "spans": spans,
                "noise_flag": 0 if noise_mult == 1.0 else 1,
                "auth_rank": _AUTHORITY_RANK.get(str(r["authority_tier"]), 9),
                "heading_flag": 0 if heading_hit else 1,
                "match_count": len(spans),
                "match_line": match_line,
            }
        )

    matched.sort(
        key=lambda m: (
            m["noise_flag"],
            m["auth_rank"],
            m["heading_flag"],
            -min(m["match_count"], 5),
            str(m["row"]["uri"]),
        )
    )

    warnings: list[str] = []
    if not matched and (src_id or src_ver or domain_scope):
        scope_clauses = ["store IN ('parsed', 'wiki')"]
        scope_params: list[Any] = []
        if domain_scope:
            scope_clauses.append(f"domain IN ({','.join('?' for _ in domain_scope)})")
            scope_params.extend(domain_scope)
        avail = [
            f"{ar['source_id']}@{ar['source_version']}"
            for ar in store.conn.execute(
                f"SELECT DISTINCT source_id, source_version FROM chunks WHERE {' AND '.join(scope_clauses)} ORDER BY source_id LIMIT 15",
                scope_params,
            ).fetchall()
        ]
        warnings.append(f"Zero matches in requested filter scope. Available sources: {avail}")

    total_matching_chunks = len(matched)
    top_hits = matched[:limit]

    # Dynamic token-fit selection: full chunks when everything fits inside max_matches and budget
    full_texts_tokens = [count_tokens(m["stripped"]) for m in top_hits]
    use_full_mode = bool(
        top_hits
        and total_matching_chunks <= limit
        and sum(full_texts_tokens) <= budget
    )

    items: list[GrepMatch] = []
    overflow: list[GrepOverflowEntry] = []
    total_tokens = 0
    truncated = False
    render_mode = "full" if use_full_mode else "snippets"

    if use_full_mode:
        for m, tok_cnt in zip(top_hits, full_texts_tokens):
            r = m["row"]
            is_stale = bool(r["store"] == "wiki" and store.check_compiled_staleness(r["uri"]))
            if is_stale:
                warnings.append(f"[STALE] Compiled page {r['uri']} references updated parsed/ content.")
            items.append(
                GrepMatch(
                    uri=r["uri"],
                    store=r["store"],
                    domain=r["domain"],
                    source_id=r["source_id"],
                    source_version=r["source_version"],
                    authority_tier=r["authority_tier"],
                    clause_path=r["clause_path"],
                    text=m["stripped"],
                    match_count=m["match_count"],
                    match_line=m["match_line"],
                    more_matches_in_chunk=0,
                    token_count=tok_cnt,
                    truncated=False,
                    stale=is_stale,
                )
            )
            total_tokens += tok_cnt
    else:
        manifest_budget = int(budget * 0.15) if total_matching_chunks > 1 else 0
        excerpt_budget = budget - manifest_budget
        overflow_candidates: list[dict[str, Any]] = []

        for m in top_hits:
            r = m["row"]
            snippet, uncovered = _build_coalesced_snippet(m["stripped"], m["spans"])
            tok_cnt = count_tokens(snippet)
            if items and (total_tokens + tok_cnt) > excerpt_budget:
                truncated = True
                overflow_candidates.append(m)
                continue
            if not items and tok_cnt > budget:
                # Guarantee at least top-1 excerpt bounded to budget
                truncated = True
                max_chars = max(120, int(budget * 3))
                snippet = snippet[:max_chars].rstrip() + " ..."
                tok_cnt = count_tokens(snippet)

            is_stale = bool(r["store"] == "wiki" and store.check_compiled_staleness(r["uri"]))
            if is_stale:
                warnings.append(f"[STALE] Compiled page {r['uri']} references updated parsed/ content.")
            items.append(
                GrepMatch(
                    uri=r["uri"],
                    store=r["store"],
                    domain=r["domain"],
                    source_id=r["source_id"],
                    source_version=r["source_version"],
                    authority_tier=r["authority_tier"],
                    clause_path=r["clause_path"],
                    text=snippet,
                    match_count=m["match_count"],
                    match_line=m["match_line"],
                    more_matches_in_chunk=uncovered,
                    token_count=tok_cnt,
                    truncated=(uncovered > 0 or len(snippet) < len(m["stripped"])),
                    stale=is_stale,
                )
            )
            total_tokens += tok_cnt

        overflow_candidates.extend(matched[limit:])
        if overflow_candidates:
            truncated = True
            man_used = 0
            man_cap = max(manifest_budget, budget - total_tokens)
            for m in overflow_candidates[:50]:
                r = m["row"]
                entry_cost = count_tokens(f"{r['uri']} {r['clause_path']}") + 6
                if man_used + entry_cost > man_cap:
                    break
                overflow.append(
                    GrepOverflowEntry(
                        uri=r["uri"],
                        clause_path=r["clause_path"],
                        authority_tier=r["authority_tier"],
                        match_count=m["match_count"],
                    )
                )
                man_used += entry_cost
                total_tokens += entry_cost

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    res = GrepResult(
        trace_id=trace_id,
        pattern=pattern,
        regex=regex,
        case_sensitive=case_sensitive,
        render_mode=render_mode,  # type: ignore[arg-type]
        total_matching_chunks=total_matching_chunks,
        total_occurrences=total_occurrences,
        returned_chunks=len(items),
        total_tokens=min(budget, total_tokens),
        token_budget=budget,
        truncated=truncated,
        items=items,
        overflow_manifest=overflow,
        warnings=warnings,
        elapsed_ms=elapsed_ms,
    )

    tracer.write(
        {
            "event": "grep",
            "trace_id": trace_id,
            "parent_trace_id": parent_trace_id,
            "harness_id": harness_id,
            "pattern": pattern[:120],
            "regex": regex,
            "case_sensitive": case_sensitive,
            "domain_scope": domain_scope,
            "source_id": src_id,
            "source_version": src_ver,
            "render_mode": render_mode,
            "total_matching_chunks": total_matching_chunks,
            "total_occurrences": total_occurrences,
            "returned_chunks": len(items),
            "truncated": truncated,
            "elapsed_ms": elapsed_ms,
        }
    )
    return res
