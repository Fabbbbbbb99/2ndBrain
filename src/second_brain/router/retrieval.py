"""Hybrid retrieval combining partitioned FTS5, partitioned vec0, graph expansion, and Gate 1 MiniLM reranking."""
from __future__ import annotations

import math
import re
import time
from typing import Any

import numpy as np

from second_brain.config import BrainConfig
from second_brain.memory.recall import select_slot1_memories
from second_brain.observability.tracer import TraceLogger
from second_brain.router.packer import assemble_grounding_pack
from second_brain.router.system1 import load_specialist_specs, route_query
from second_brain.storage.models import EvidenceEnvelope, GroundingPack, normalize_project_tag
from second_brain.storage.sqlite_store import SqliteStore, serialize_unit_int8

AUTHORITY_BOOST = {
    "primary_normative": 1.25,
    "primary_reference": 1.15,
    "compiled_synthesis": 0.90,
    "working_memory": 1.00,
}

_INDEX_PATH_RE = re.compile(r"(?:>\s*\d+\s+index\b|>\s*index\b)", re.IGNORECASE)
_TOC_LINE_RE = re.compile(r"(?:\.{5,}|\u2026{3,}|(?:\.\s){4,})\s*[ivxlcdm\d]+\s*$", re.IGNORECASE)
_PAGE_NUM_END_RE = re.compile(r"\s\d{1,4}\s*$")
STRUCTURAL_NOISE_MULT = 0.05


def _structural_noise_multiplier(env: EvidenceEnvelope) -> float:
    """Demote back-of-book alphabetical index chunks, empty heading stubs, and TOC pages in parsed/."""
    if env.store != "parsed":
        return 1.0
    clause_path = env.clause_path or ""
    if _INDEX_PATH_RE.search(clause_path):
        return STRUCTURAL_NOISE_MULT
    lines = [ln.strip() for ln in env.text.splitlines() if ln.strip()]
    body_lines = [ln for ln in lines if not ln.startswith("<!-- breadcrumb:") and not ln.startswith("#")]
    if not body_lines:
        return STRUCTURAL_NOISE_MULT
    toc_hits = sum(1 for ln in body_lines if _TOC_LINE_RE.search(ln))
    if toc_hits >= 3 and (toc_hits / len(body_lines)) >= 0.40:
        return STRUCTURAL_NOISE_MULT
    low_path = clause_path.lower()
    if ("front matter" in low_path or "contents" in low_path) and len(body_lines) >= 8:
        pg_hits = sum(1 for ln in body_lines if _PAGE_NUM_END_RE.search(ln) and len(ln) < 85)
        if pg_hits / len(body_lines) >= 0.60:
            return STRUCTURAL_NOISE_MULT
    return 1.0


def _to_fts_query(raw: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9_]+", raw)
    stop = {"the", "a", "an", "is", "are", "how", "what", "when", "why", "in", "for", "of", "to", "and", "or"}
    kept = [t for t in tokens if len(t) >= 2 and t.lower() not in stop]
    if not kept:
        kept = [t for t in tokens if len(t) >= 2]
    if not kept:
        return ""
    return " OR ".join(dict.fromkeys(kept[:12]))


def _run_partition_search(
    store: SqliteStore,
    fts_q: str,
    q_bytes: bytes,
    domains: list[str],
    effective_version: str | None,
    source_id: str | None = None,
) -> tuple[list[tuple[EvidenceEnvelope, float]], list[tuple[EvidenceEnvelope, float]], float, float]:
    t_fts0 = time.perf_counter()
    fts_hits = store.search_fts(
        fts_q, domain_scope=domains, version_filter=effective_version, limit=25, source_id=source_id
    )
    t_fts_ms = round((time.perf_counter() - t_fts0) * 1000.0, 2)

    t_vec0 = time.perf_counter()
    vec_hits = store.search_vec(
        q_bytes, domain_scope=domains, version_filter=effective_version, limit=25, source_id=source_id
    )
    t_vec_ms = round((time.perf_counter() - t_vec0) * 1000.0, 2)
    return fts_hits, vec_hits, t_fts_ms, t_vec_ms


def execute_query(
    *,
    cfg: BrainConfig,
    store: SqliteStore,
    models: Any,
    tracer: TraceLogger,
    query: str,
    domain_scope: list[str] | None = None,
    source_version: str | None = None,
    source_id: str | None = None,
    project_tag: str | None = None,
    token_budget: int | None = None,
    include_rerank: bool = True,
    parent_trace_id: str | None = None,
    harness_id: str = "cli",
) -> GroundingPack:
    """Run read-only System 1 route -> partitioned FTS5 + vec0 + graph -> Gate 1 MiniLM -> Pack."""
    budget = token_budget or cfg.pack_token_budget
    latencies: dict[str, float] = {}
    warnings: list[str] = []
    norm_source_id = source_id.strip() if source_id and source_id.strip() else None
    norm_project_tag = normalize_project_tag(project_tag)

    route = route_query(
        cfg=cfg,
        store=store,
        models=models,
        tracer=tracer,
        query=query,
        explicit_domains=domain_scope,
        version_filter=source_version,
        parent_trace_id=parent_trace_id,
        harness_id=harness_id,
    )
    warnings.extend(route.warnings)
    trace_id = route.trace_id
    active_domains = list(route.selected_domains)
    latencies["t_laya_ms"] = route.laya_latency_ms
    effective_version = source_version

    # Embed query once
    t_emb0 = time.perf_counter()
    embedder = models.get_embedder()
    q_vec = np.asarray(next(iter(embedder.embed([query]))), dtype=np.float32)
    q_bytes = serialize_unit_int8(q_vec)
    latencies["t_embed_ms"] = round((time.perf_counter() - t_emb0) * 1000.0, 2)

    fts_q = _to_fts_query(query)
    fts_hits, vec_hits, t_fts_ms, t_vec_ms = _run_partition_search(
        store, fts_q, q_bytes, active_domains, effective_version, norm_source_id
    )

    widened_fallback = False
    if domain_scope is None and (not fts_hits) and (not vec_hits):
        all_domains = list(load_specialist_specs(cfg.vault_path).keys())
        if set(all_domains) != set(active_domains):
            fts_hits, vec_hits, t_fts_2, t_vec_2 = _run_partition_search(
                store, fts_q, q_bytes, all_domains, effective_version, norm_source_id
            )
            t_fts_ms = round(t_fts_ms + t_fts_2, 2)
            t_vec_ms = round(t_vec_ms + t_vec_2, 2)
            widened_fallback = True
            warnings.append(
                f"System 1 partition {active_domains} returned 0 hits; widened retrieval to {all_domains}; logged for Laya retraining."
            )
            active_domains = all_domains

    latencies["t_fts_ms"] = t_fts_ms
    latencies["t_vec_ms"] = t_vec_ms

    if not effective_version:
        ver_params = list(active_domains)
        src_filter = ""
        if norm_source_id:
            src_filter = " AND LOWER(source_id) = LOWER(?)"
            ver_params.append(norm_source_id)
        distinct_vers = [
            r["source_version"]
            for r in store.conn.execute(
                f"SELECT DISTINCT source_version FROM chunks WHERE domain IN ({','.join('?' for _ in active_domains)}) AND store='parsed'{src_filter}",
                ver_params,
            ).fetchall()
        ]
        if len(distinct_vers) > 1:
            warnings.append(
                f"Multiple source_version editions exist in {active_domains}: {distinct_vers}. Pass source_version to pin one edition."
            )

    cov = store.get_index_coverage(active_domains)
    if cov.index_state in ("partial_embeddings", "fast_bm25_only"):
        warnings.append(
            f"Index coverage is {cov.index_state} ({cov.coverage_pct}% embedded; {cov.pending_chunks} pending)."
        )

    # RRF Fusion + authority precedence
    rrf_k = 60.0
    fused_scores: dict[str, float] = {}
    by_uri: dict[str, EvidenceEnvelope] = {}
    for rank, (env, _raw) in enumerate(fts_hits, start=1):
        by_uri[env.uri] = env
        boost = AUTHORITY_BOOST.get(env.authority_tier, 1.0) * _structural_noise_multiplier(env)
        fused_scores[env.uri] = fused_scores.get(env.uri, 0.0) + boost / (rrf_k + rank)

    for rank, (env, _dist) in enumerate(vec_hits, start=1):
        by_uri[env.uri] = env
        boost = AUTHORITY_BOOST.get(env.authority_tier, 1.0) * _structural_noise_multiplier(env)
        fused_scores[env.uri] = fused_scores.get(env.uri, 0.0) + boost / (rrf_k + rank)

    # 1-hop graph boost for parent / cited / certified INFERRED_RELATED chunks inside active_domains
    t_gr0 = time.perf_counter()
    seed_uris = [u for u, _ in sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]]
    if seed_uris:
        for nb in store.get_neighbors(seed_uris, active_domains, limit=10, exclude_inferred=True):
            if nb.provenance == "EXTRACTED":
                child = store.get_chunk_by_uri(nb.uri)
                if (
                    child
                    and child.domain in active_domains
                    and (not effective_version or child.source_version in (effective_version, "global"))
                    and (not norm_source_id or child.source_id.lower() == norm_source_id.lower())
                ):
                    by_uri.setdefault(child.uri, child)
                    child_noise = _structural_noise_multiplier(child)
                    fused_scores[child.uri] = fused_scores.get(child.uri, 0.0) + (0.35 * child_noise) / (rrf_k + 5.0)
        has_inferred = store.conn.execute("SELECT 1 FROM edges WHERE rel_type='INFERRED_RELATED' LIMIT 1").fetchone()
        if has_inferred:
            from second_brain.graph.import_edges import load_certified_edge_hashes

            certified_hashes = load_certified_edge_hashes(cfg)
            for seed in seed_uris:
                inferred_refs, _ = store.get_inferred_links(seed, limit=6, certified_hashes=certified_hashes)
                for nb in inferred_refs:
                    if nb.stale:
                        continue
                    child = store.get_chunk_by_uri(nb.uri)
                    if (
                        child
                        and child.domain in active_domains
                        and (not effective_version or child.source_version in (effective_version, "global"))
                        and (not norm_source_id or child.source_id.lower() == norm_source_id.lower())
                    ):
                        by_uri.setdefault(child.uri, child)
                        child_noise = _structural_noise_multiplier(child)
                        conf = float(nb.confidence) if nb.confidence is not None else 0.80
                        fused_scores[child.uri] = fused_scores.get(child.uri, 0.0) + (
                            0.15 * conf * child_noise
                        ) / (rrf_k + 5.0)
    latencies["t_graph_ms"] = round((time.perf_counter() - t_gr0) * 1000.0, 2)

    ranked_uris = [u for u, _ in sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)]
    candidates = [by_uri[u].model_copy(update={"score": round(fused_scores[u], 5)}) for u in ranked_uris[:8]]

    # Gate 1 Confidence-Gated MiniLM Cross-Encoder Rerank (Top <= 8, max 400 chars)
    t_rr0 = time.perf_counter()
    reranked_flag = False
    fts_top1 = fts_hits[0][0].uri if fts_hits else None
    vec_top1 = vec_hits[0][0].uri if vec_hits else None
    top1_rrf = fused_scores[ranked_uris[0]] if ranked_uris else 0.0
    top2_rrf = fused_scores[ranked_uris[1]] if len(ranked_uris) > 1 else 0.0
    norm_gap = (top1_rrf - top2_rrf) / max(top1_rrf, 1e-6) if len(ranked_uris) >= 2 else 1.0

    strong_agreement = bool(
        fts_top1 and vec_top1 and fts_top1 == vec_top1 and _structural_noise_multiplier(by_uri[fts_top1]) == 1.0
    )
    if include_rerank and len(candidates) > 1 and (not strong_agreement) and norm_gap < 0.15:
        reranker = models.get_reranker()
        docs = [c.text[:400] for c in candidates]
        rr_scores = list(reranker.rerank(query, docs))
        reranked = []
        for cand, rr_s in zip(candidates, rr_scores):
            auth_b = AUTHORITY_BOOST.get(cand.authority_tier, 1.0)
            noise_m = _structural_noise_multiplier(cand)
            adj = float(rr_s) + math.log(auth_b) + (math.log(noise_m) if noise_m < 1.0 else 0.0)
            reranked.append(cand.model_copy(update={"score": round(adj, 5)}))
        reranked.sort(key=lambda c: c.score, reverse=True)
        candidates = reranked
        reranked_flag = True
    latencies["t_rerank_ms"] = round((time.perf_counter() - t_rr0) * 1000.0, 2)

    t_mem0 = time.perf_counter()
    selected_memories, mem_stats, mem_dropped, mem_warnings = select_slot1_memories(
        cfg=cfg,
        store=store,
        query=query,
        q_vec=q_vec,
        domain_scope=active_domains,
        version_scope=effective_version,
        project_tag=norm_project_tag,
        slot1_token_budget=int(budget * 0.15),
    )
    latencies["t_mem_rank_ms"] = round((time.perf_counter() - t_mem0) * 1000.0, 2)
    warnings.extend(mem_warnings)
    statuses = store.get_store_statuses()

    t_pk0 = time.perf_counter()
    pack, dropped = assemble_grounding_pack(
        cfg=cfg,
        store=store,
        trace_id=trace_id,
        query=query,
        domain_scope=active_domains,
        source_version_filter=effective_version,
        route=route,
        candidates=candidates,
        memories=selected_memories,
        index_coverage=cov,
        store_statuses=statuses,
        token_budget=budget,
        stage_latencies_ms=latencies,
        warnings=warnings,
        reranked=reranked_flag,
    )
    dropped.extend(mem_dropped)
    latencies["t_pack_ms"] = round((time.perf_counter() - t_pk0) * 1000.0, 2)

    if pack.items:
        top_topic = pack.items[0].clause_path[:80]
        q_count = store.record_query_topic(top_topic, route.primary_domain)
        if q_count >= 3 and pack.items[0].store == "parsed":
            pack.warnings.append(
                f"Topic {top_topic!r} reached {q_count} queries. Run `brain compile` or call `brain_compile` to file a synthesis page."
            )

    tracer.write(
        {
            "event": "query",
            "trace_id": trace_id,
            "parent_trace_id": parent_trace_id,
            "harness_id": harness_id,
            "query": query,
            "domain_scope": active_domains,
            "source_version": effective_version,
            "source_id": norm_source_id,
            "project_tag": norm_project_tag,
            "gate": route.gate,
            "widened_fallback": widened_fallback,
            "reranked": reranked_flag,
            "memories_considered": mem_stats["memories_considered"],
            "memories_kept": mem_stats["memories_kept"],
            "memories_dropped_gate": mem_stats["memories_dropped_gate"],
            "stage_latencies_ms": latencies,
            "index_coverage": cov.model_dump(),
            "pack_tokens": pack.total_tokens,
            "token_budget": budget,
            "retrieved_uris": [it.uri for it in pack.items],
            "dropped": dropped,
        }
    )
    return pack
