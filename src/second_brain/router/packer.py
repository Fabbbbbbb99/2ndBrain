"""Dual-Track Verbatim Grounding Pack assembler with tiktoken o200k_base budget and relevance pruning."""
from __future__ import annotations

import hashlib

import tiktoken

from second_brain.config import BrainConfig
from second_brain.ingest.pdf_adapter import render_visual_crop
from second_brain.storage.models import (
    EvidenceEnvelope,
    GroundingPack,
    IndexCoverage,
    LinkedClauseRef,
    MemoryRecord,
    RouteDecision,
    StoreStatus,
    VisualCropNotice,
)
from second_brain.storage.sqlite_store import SqliteStore

_ENC = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _memory_to_envelope(rec: MemoryRecord, score: float = 1.0) -> EvidenceEnvelope:
    header = (
        f"[Memory {rec.id} | {rec.ledger_type} | effective={rec.effective_from}"
        + (f"..{rec.effective_to}" if rec.effective_to else "..open")
        + (f" | project={rec.project_tag}" if rec.project_tag else "")
        + (f" | scope={rec.version_scope}" if rec.version_scope else "")
        + "]\n"
    )
    body = header + rec.content.strip()
    sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return EvidenceEnvelope(
        uri=f"memory://{rec.id}",
        store="memory",
        authority_tier="working_memory",
        domain=rec.domain,
        source_id=f"memory_{rec.ledger_type}",
        source_version=rec.version_scope or "global",
        clause_path=f"Memory > {rec.ledger_type} > {rec.id}",
        text=body,
        content_sha256=sha,
        token_count=count_tokens(body),
        status=rec.status,
        score=round(float(score), 4),
    )


def _summarize_oversized(env: EvidenceEnvelope, max_tokens: int) -> EvidenceEnvelope:
    lines = [ln for ln in env.text.splitlines() if ln.strip()]
    preview_lines = lines[:6]
    summary = "\n".join(preview_lines) + f"\n[TRUNCATED_ATOMIC_BLOCK: fetch full text via brain_get_clause(uri={env.uri!r})]"
    tokens = _ENC.encode(summary)
    if len(tokens) > max_tokens:
        summary = _ENC.decode(tokens[: max(32, max_tokens - 24)]) + f"\n[TRUNCATED: fetch {env.uri}]"
    return env.model_copy(
        update={
            "text": summary,
            "token_count": count_tokens(summary),
            "truncated": True,
            "fetch_span_handle": env.uri,
        }
    )


def _prune_by_relevance(
    cands: list[EvidenceEnvelope],
    *,
    reranked: bool,
    max_items: int,
    dropped: list[dict[str, str]],
) -> list[EvidenceEnvelope]:
    if not cands:
        return []
    kept = [cands[0]]
    top_score = cands[0].score
    for cand in cands[1:]:
        if len(kept) >= max_items:
            dropped.append({"uri": cand.uri, "reason": "max_pack_items_cap"})
            continue
        if reranked:
            if (top_score > 0 and cand.score < 0) or (cand.score < top_score - 4.0):
                dropped.append({"uri": cand.uri, "reason": "below_relevance_cutoff"})
                continue
        else:
            if top_score > 0 and cand.score < 0.55 * top_score:
                dropped.append({"uri": cand.uri, "reason": "below_relevance_cutoff"})
                continue
        kept.append(cand)
    return kept


def assemble_grounding_pack(
    *,
    cfg: BrainConfig,
    store: SqliteStore,
    trace_id: str,
    query: str,
    domain_scope: list[str],
    source_version_filter: str | None,
    route: RouteDecision | None,
    candidates: list[EvidenceEnvelope],
    memories: list[Any],
    index_coverage: IndexCoverage,
    store_statuses: list[StoreStatus],
    token_budget: int,
    stage_latencies_ms: dict[str, float],
    warnings: list[str],
    reranked: bool = False,
) -> tuple[GroundingPack, list[dict[str, str]]]:
    """Assemble <= token_budget Grounding Pack with 15/55/20/10 slot quotas and relevance pruning."""
    mem_budget = int(token_budget * 0.15)
    verbatim_budget = int(token_budget * 0.55)
    wiki_budget = int(token_budget * 0.20)

    selected: list[EvidenceEnvelope] = []
    dropped: list[dict[str, str]] = []
    used_tokens = 0

    # Slot 1: Active Store 2 memory overrides (up to 15%)
    mem_used = 0
    for item in memories:
        rec, mem_score = item if isinstance(item, tuple) else (item, 1.0)
        env = _memory_to_envelope(rec, score=float(mem_score))
        if mem_used + env.token_count <= mem_budget:
            selected.append(env)
            mem_used += env.token_count
            used_tokens += env.token_count
        else:
            dropped.append({"uri": env.uri, "reason": "memory_slot_budget"})

    pruned_cands = _prune_by_relevance(candidates, reranked=reranked, max_items=5, dropped=dropped)
    verbatim_cands = [c for c in pruned_cands if c.store == "parsed"]
    wiki_cands = [c for c in pruned_cands if c.store == "wiki"]

    # Slot 2: Verbatim parsed/ clauses (up to 55% + unused memory slack)
    verb_limit = verbatim_budget + max(0, mem_budget - mem_used)
    verb_used = 0
    for cand in verbatim_cands:
        if verb_used >= verb_limit:
            dropped.append({"uri": cand.uri, "reason": "verbatim_slot_budget"})
            continue
        remaining = verb_limit - verb_used
        if cand.token_count <= remaining:
            selected.append(cand)
            verb_used += cand.token_count
            used_tokens += cand.token_count
        elif remaining >= 140:
            compact = _summarize_oversized(cand, remaining)
            selected.append(compact)
            verb_used += compact.token_count
            used_tokens += compact.token_count
            dropped.append({"uri": cand.uri, "reason": "oversized_block_summarized"})
        else:
            dropped.append({"uri": cand.uri, "reason": "verbatim_slot_budget"})

    # Slot 3: Compiled wiki summaries (up to 20% + unused verbatim slack), verified against upstream SHA-256
    wiki_limit = wiki_budget + max(0, verb_limit - verb_used)
    wiki_used = 0
    for cand in wiki_cands:
        if wiki_used >= wiki_limit:
            dropped.append({"uri": cand.uri, "reason": "wiki_slot_budget"})
            continue
        is_stale = store.check_compiled_staleness(cand.uri)
        item = cand.model_copy(update={"stale": is_stale, "status": "stale" if is_stale else cand.status})
        if is_stale and f"[STALE] Compiled page {cand.uri} upstream changed" not in warnings:
            warnings.append(f"[STALE] Compiled page {cand.uri} references updated parsed/ content.")
        remaining = wiki_limit - wiki_used
        if item.token_count <= remaining:
            selected.append(item)
            wiki_used += item.token_count
            used_tokens += item.token_count
        elif remaining >= 120:
            compact = _summarize_oversized(item, remaining)
            selected.append(compact)
            wiki_used += compact.token_count
            used_tokens += compact.token_count
        else:
            dropped.append({"uri": cand.uri, "reason": "wiki_slot_budget"})

    # Slot 4: 1-hop Linked Clause Manifest + Lazy Vision crops
    selected_uris = [s.uri for s in selected if s.store != "memory"]
    neighbors: list[LinkedClauseRef] = []
    if selected_uris:
        raw_neighbors = store.get_neighbors(selected_uris, domain_scope, limit=16)
        raw_neighbors.sort(
            key=lambda nb: (0 if nb.provenance == "EXTRACTED" else 1, -(nb.confidence or 0.0))
        )
        manifest_budget = max(0, token_budget - used_tokens)
        for nb in raw_neighbors:
            if len(neighbors) >= 12:
                break
            if nb.uri in selected_uris:
                continue
            if nb.rel_type == "INFERRED_RELATED" and (nb.confidence or 0.0) < 0.85:
                continue
            cost = count_tokens(f"{nb.uri} {nb.rel_type} {nb.clause_path or ''}") + 8
            if cost <= manifest_budget:
                neighbors.append(nb)
                used_tokens += cost
                manifest_budget -= cost

    visual_notices: list[VisualCropNotice] = []
    if selected_uris:
        for vn in store.get_visual_nodes_for_uris(selected_uris):
            if vn["status"] == "pending_vision":
                crop_path_str = vn.get("crop_path") or ""
                if not crop_path_str:
                    try:
                        crop_path_str = str(render_visual_crop(cfg, vn))
                        store.conn.execute(
                            "UPDATE visual_nodes SET crop_path = ? WHERE visual_id = ?",
                            (crop_path_str, vn["visual_id"]),
                        )
                        store.conn.commit()
                    except Exception as exc:
                        warnings.append(f"Visual crop error for {vn['visual_id']}: {exc}")
                notice = VisualCropNotice(
                    visual_id=vn["visual_id"],
                    source_uri=vn["source_uri"],
                    kind=vn["kind"],
                    page_num=int(vn["page_num"]),
                    crop_path=crop_path_str,
                    status=vn["status"],
                )
                visual_notices.append(notice)

    pack = GroundingPack(
        trace_id=trace_id,
        query=query,
        domain_scope=domain_scope,
        source_version_filter=source_version_filter,
        route=route,
        index_coverage=index_coverage,
        store_statuses=store_statuses,
        items=selected,
        linked_manifest=neighbors,
        visual_notices=visual_notices,
        total_tokens=min(token_budget, used_tokens),
        token_budget=token_budget,
        stage_latencies_ms=stage_latencies_ms,
        warnings=warnings,
    )
    return pack, dropped
