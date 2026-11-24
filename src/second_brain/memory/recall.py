"""Shared Store 2 memory embedding, hybrid ranking, Slot 1 gating, and read-only brain_recall."""
from __future__ import annotations

import re
import time
from typing import Any

import numpy as np

from second_brain.config import BrainConfig
from second_brain.ingest.embed_worker import make_embed_key
from second_brain.observability.tracer import TraceLogger, new_trace_id
from second_brain.storage.models import (
    MemoryRecallItem,
    MemoryRecallResult,
    MemoryRecord,
    normalize_project_tag,
)
from second_brain.storage.sqlite_store import SqliteStore

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_STOPWORDS = {"the", "a", "an", "is", "are", "how", "what", "when", "why", "in", "for", "of", "to", "and", "or"}
_VALID_RECALL_STATUSES = {"active", "staged", "superseded", "expired", "rejected", "all"}


def lex_tokens(raw: str) -> set[str]:
    """Extract lowercase >=2-char non-stopword alphanumeric tokens."""
    tokens = _TOKEN_RE.findall(raw or "")
    kept = {t.lower() for t in tokens if len(t) >= 2 and t.lower() not in _STOPWORDS}
    if not kept:
        kept = {t.lower() for t in tokens if len(t) >= 2}
    return kept


def lex_coverage_and_hit(q_toks: set[str], content: str) -> tuple[float, bool]:
    """Return normalized query coverage in [0, 1] and deterministic lexical hit flag."""
    if not q_toks:
        return 0.0, False
    m_toks = lex_tokens(content)
    shared = len(q_toks & m_toks)
    cov = shared / len(q_toks)
    hit = (shared >= 2) or (len(q_toks) == 1 and shared == 1)
    return cov, hit


def embed_memory_text(
    models: Any,
    text: str,
    model_id: str = "BAAI/bge-small-en-v1.5",
) -> tuple[bytes | None, str | None, np.ndarray | None]:
    """Compute unit-L2 float32[384] bytes + embed_key, returning None triples if embedder fails."""
    cleaned = text.strip()
    if not cleaned:
        return None, None, None
    try:
        emb = models.get_embedder()
        raw = np.asarray(next(iter(emb.embed([cleaned]))), dtype=np.float32)
        norm = float(np.linalg.norm(raw))
        unit = (raw / norm).astype(np.float32) if norm > 0.0 else raw.astype(np.float32)
        emb_key = make_embed_key(model_id, "v1.5", "v1", cleaned)
        return unit.tobytes(), emb_key, unit
    except Exception:
        return None, None, None


def _unit_vec(vec: np.ndarray | None) -> np.ndarray | None:
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 0.0:
        return (arr / norm).astype(np.float32)
    return arr


def rank_memory_candidates(
    candidates: list[tuple[MemoryRecord, bytes | None, str | None]],
    *,
    query: str | None,
    q_vec: np.ndarray | None,
    project_tag: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Score memory candidates by unit-cosine + lexical coverage + project_tag match."""
    q_unit = _unit_vec(q_vec)
    q_toks = lex_tokens(query) if (query and query.strip()) else set()
    norm_tag = normalize_project_tag(project_tag)
    null_emb_count = 0
    scored: list[dict[str, Any]] = []

    for rec, emb_bytes, _emb_key in candidates:
        cos_sim: float | None = None
        if q_unit is not None and emb_bytes is not None:
            vec = np.frombuffer(emb_bytes, dtype=np.float32)
            if vec.shape == q_unit.shape:
                cos_sim = round(float(np.dot(q_unit, vec)), 4)
        elif q_unit is not None and emb_bytes is None:
            null_emb_count += 1

        lex_cov, lex_hit = lex_coverage_and_hit(q_toks, rec.content)
        tag_bonus = 0.05 if (norm_tag and rec.project_tag == norm_tag) else 0.0
        if cos_sim is not None:
            hybrid = 0.75 * max(0.0, cos_sim) + 0.25 * lex_cov + tag_bonus
        else:
            hybrid = lex_cov + tag_bonus

        scored.append(
            {
                "rec": rec,
                "similarity": cos_sim,
                "lexical_score": round(lex_cov, 4),
                "lexical_hit": lex_hit,
                "hybrid_score": round(hybrid, 4),
                "exact_project": bool(norm_tag and rec.project_tag == norm_tag),
            }
        )

    # Sort descending by hybrid score while preserving newest effective_from / created_at on ties
    scored.sort(key=lambda item: (item["rec"].effective_from, item["rec"].created_at), reverse=True)
    if query and query.strip():
        scored.sort(key=lambda item: (1 if item["exact_project"] else 0, item["hybrid_score"]), reverse=True)

    warnings: list[str] = []
    if q_unit is not None and null_emb_count > 0:
        warnings.append(
            f"{null_emb_count} active/candidate memory record(s) have NULL embeddings; lexical fallback applied."
        )
    return scored, warnings


def select_slot1_memories(
    *,
    cfg: BrainConfig,
    store: SqliteStore,
    query: str,
    q_vec: np.ndarray | None,
    domain_scope: list[str],
    version_scope: str | None = None,
    project_tag: str | None = None,
    slot1_token_budget: int = 375,
) -> tuple[list[tuple[MemoryRecord, float]], dict[str, Any], list[dict[str, str]], list[str]]:
    """Rank full active pool in scope; pin preferences (<=50% cap) and gate decisions/facts."""
    norm_tag = normalize_project_tag(project_tag)
    candidates = store.get_memory_candidates(
        domain_scope=domain_scope,
        project_tag=norm_tag,
        include_untagged_for_project=True,
        version_scope=version_scope,
        status="active",
        limit=500,
    )
    ranked, warnings = rank_memory_candidates(
        candidates,
        query=query,
        q_vec=q_vec,
        project_tag=norm_tag,
    )
    if q_vec is None and candidates:
        warnings.append("memory_embedder_unavailable_lexical_fallback")

    prefs: list[dict[str, Any]] = []
    others: list[dict[str, Any]] = []
    for item in ranked:
        rec: MemoryRecord = item["rec"]
        if rec.ledger_type == "preferences":
            # Prevent foreign project preferences from leaking into untagged queries
            if norm_tag is None and rec.project_tag is not None:
                continue
            prefs.append(item)
        else:
            others.append(item)

    # Preferences: exact project_tag match first, capped at 50% of Slot 1 budget (~187t default)
    prefs.sort(
        key=lambda x: (1 if x["exact_project"] else 0, x["hybrid_score"], x["rec"].effective_from, x["rec"].created_at),
        reverse=True,
    )

    from second_brain.router.packer import count_tokens

    pref_cap = max(64, int(slot1_token_budget * 0.50))
    pref_used = 0
    selected: list[tuple[MemoryRecord, float]] = []
    dropped_gate: list[dict[str, str]] = []

    for item in prefs:
        rec = item["rec"]
        est_tok = count_tokens(rec.content) + 28
        if selected and (pref_used + est_tok) > pref_cap:
            dropped_gate.append({"uri": f"memory://{rec.id}", "reason": "preference_slot_cap"})
            continue
        selected.append((rec, float(item["hybrid_score"])))
        pref_used += est_tok

    # Decisions & facts: must clear calibrated BGE cosine gate OR deterministic lexical hit
    for item in others:
        rec = item["rec"]
        sim = item["similarity"]
        passes_vec = sim is not None and sim >= cfg.memory_sim_threshold
        passes_lex = bool(item["lexical_hit"])
        if passes_vec or passes_lex:
            selected.append((rec, float(item["hybrid_score"])))
        else:
            dropped_gate.append({"uri": f"memory://{rec.id}", "reason": "below_memory_relevance_gate"})

    kept_count = len(selected)
    stats = {
        "memories_considered": len(candidates),
        "memories_kept": kept_count,
        "memories_dropped_gate": len(dropped_gate),
    }
    return selected, stats, dropped_gate, warnings


def execute_recall(
    *,
    cfg: BrainConfig,
    store: SqliteStore,
    models: Any,
    tracer: TraceLogger,
    query: str | None = None,
    domain_scope: list[str] | None = None,
    project_tag: str | None = None,
    ledger_type: str | None = None,
    version_scope: str | None = None,
    status: str = "active",
    limit: int = 15,
    parent_trace_id: str | None = None,
    harness_id: str = "cli",
) -> MemoryRecallResult:
    """Read-only Store 2 memory search & structured staged/active/superseded inspection."""
    t0 = time.perf_counter()
    trace_id = new_trace_id()
    if status not in _VALID_RECALL_STATUSES:
        raise ValueError(f"INVALID_STATUS: status must be one of {sorted(_VALID_RECALL_STATUSES)}, got {status!r}")
    if ledger_type and ledger_type not in {"preferences", "decisions", "facts"}:
        raise ValueError(f"INVALID_LEDGER_TYPE: ledger_type={ledger_type!r}")

    norm_tag = normalize_project_tag(project_tag)
    cap = max(1, min(int(limit), 50))
    candidates = store.get_memory_candidates(
        domain_scope=domain_scope,
        project_tag=norm_tag,
        include_untagged_for_project=False,
        ledger_type=ledger_type,
        version_scope=version_scope,
        status=status,
        limit=2000,
    )

    warnings: list[str] = []
    q_vec: np.ndarray | None = None
    has_query = bool(query and query.strip())
    if has_query and candidates:
        try:
            emb = models.get_embedder()
            q_vec = np.asarray(next(iter(emb.embed([query.strip()]))), dtype=np.float32)  # type: ignore[union-attr]
        except Exception as exc:
            q_vec = None
            warnings.append(f"memory_embedder_unavailable_lexical_fallback: {exc}")

    scored, rank_warnings = rank_memory_candidates(
        candidates,
        query=query if has_query else None,
        q_vec=q_vec,
        project_tag=norm_tag,
    )
    warnings.extend(rank_warnings)

    page = scored[:cap]
    conflict_map = store.get_conflict_targets([entry["rec"].id for entry in page if entry["rec"].status == "staged"])

    items: list[MemoryRecallItem] = []
    for entry in page:
        rec: MemoryRecord = entry["rec"]
        hint = f"brain certify --id {rec.id} --decision approve" if rec.status == "staged" else None
        items.append(
            MemoryRecallItem(
                id=rec.id,
                domain=rec.domain,
                ledger_type=rec.ledger_type,
                project_tag=rec.project_tag,
                effective_from=rec.effective_from,
                effective_to=rec.effective_to,
                version_scope=rec.version_scope,
                status=rec.status,
                supersedes_id=rec.supersedes_id,
                content=rec.content,
                source_ref=rec.source_ref,
                reason_code=rec.reason_code,
                created_at=rec.created_at,
                similarity=entry["similarity"],
                lexical_score=entry["lexical_score"],
                hybrid_score=entry["hybrid_score"],
                conflict_target_id=conflict_map.get(rec.id),
                certify_hint=hint,
            )
        )

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    res = MemoryRecallResult(
        trace_id=trace_id,
        status=status,
        project_tag=norm_tag,
        total_matched=len(scored),
        returned_count=len(items),
        items=items,
        warnings=warnings,
        elapsed_ms=elapsed_ms,
    )
    tracer.write(
        {
            "event": "recall",
            "trace_id": trace_id,
            "parent_trace_id": parent_trace_id,
            "harness_id": harness_id,
            "status": status,
            "domain_scope": domain_scope,
            "project_tag": norm_tag,
            "ledger_type": ledger_type,
            "query_present": has_query,
            "total_matched": len(scored),
            "returned_count": len(items),
            "elapsed_ms": elapsed_ms,
        }
    )
    return res
