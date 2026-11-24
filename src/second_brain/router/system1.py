"""Kahneman System 1 Router combining Gate 0 deterministic rules, synchronous Laya ONNX, and k-NN exemplars."""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from second_brain.config import BrainConfig
from second_brain.observability.tracer import TraceLogger, new_trace_id
from second_brain.storage.manifest import load_taxonomy
from second_brain.storage.models import FanOutBranch, RouteDecision
from second_brain.storage.sqlite_store import SqliteStore

EXACT_STD_RE = re.compile(
    r"\b((?:ISO|IEC|IEEE|MIL-STD)[-\s]*\d+(?:[-:]\d+)?|(?:ANSI|AIAA)[-\s/]*(?:AIAA[-\s]*)?G[-\s]*\d+[A-Z]?|INCOSE(?:[-\s]+(?:GtWR|GtNR|NRM|Handbook|SEH))?|SEBoK)\b",
    re.IGNORECASE,
)
EXACT_CLAUSE_RE = re.compile(r"\b(?:Clause|Section|Annex)\s+(\d+(?:\.\d+)*|[A-Z])\b", re.IGNORECASE)
EXPLICIT_URI_RE = re.compile(r"\b(?:vault|memory|repo)://[^\s]+")


def load_specialist_specs(vault_path: Path) -> dict[str, dict[str, Any]]:
    spec_dir = vault_path / "specialists"
    specs: dict[str, dict[str, Any]] = {}
    if not spec_dir.exists():
        return specs
    for yf in sorted(spec_dir.glob("*.yaml")):
        raw = yaml.safe_load(yf.read_text(encoding="utf-8")) or {}
        if "domain_id" in raw:
            specs[str(raw["domain_id"])] = raw
    return specs


def _gate0_exact_match(
    query: str,
    store: SqliteStore,
    specs: dict[str, dict[str, Any]],
) -> tuple[list[str], str | None]:
    m_uri = EXPLICIT_URI_RE.search(query)
    if m_uri:
        env = store.get_chunk_by_uri(m_uri.group(0))
        if env:
            return [env.domain], "gate0_exact_id"

    m_std = EXACT_STD_RE.search(query)
    m_clause = EXACT_CLAUSE_RE.search(query)
    if m_std:
        norm = re.sub(r"[^a-zA-Z0-9]+", "%", m_std.group(1))
        rows = store.conn.execute(
            "SELECT DISTINCT domain FROM chunks WHERE source_id LIKE ? OR clause_path LIKE ? LIMIT 3",
            (f"%{norm}%", f"%{norm}%"),
        ).fetchall()
        found = [r["domain"] for r in rows if r["domain"] in specs]
        if found:
            return found, "gate0_exact_id"

    if m_clause:
        cval = m_clause.group(1)
        rows = store.conn.execute(
            "SELECT DISTINCT domain FROM chunks WHERE clause_path LIKE ? LIMIT 3",
            (f"%{cval}%",),
        ).fetchall()
        found = [r["domain"] for r in rows if r["domain"] in specs]
        if found:
            return found, "gate0_exact_id"

    return [], None


def _apply_taxonomy_alias_priors(
    query: str,
    probs: dict[str, float],
    taxonomy: dict[str, Any],
    specs: dict[str, dict[str, Any]],
) -> tuple[dict[str, float], dict[str, int]]:
    norm_q = query.replace("_", " ")
    tax_domains = taxonomy.get("domains") or {}
    match_counts: dict[str, int] = {}
    boosted = dict(probs)
    for dom in specs:
        aliases = list((tax_domains.get(dom) or {}).get("aliases") or [])
        hits = 0
        for alias in aliases:
            a_str = str(alias).strip()
            if not a_str:
                continue
            if re.search(rf"\b{re.escape(a_str)}\b", norm_q, re.IGNORECASE):
                hits += 1
        if hits > 0:
            match_counts[dom] = hits
            boost = min(0.60, 0.35 * hits)
            boosted[dom] = boosted.get(dom, 0.0) + boost

    if match_counts:
        total = sum(boosted.values()) or 1.0
        boosted = {k: round(v / total, 4) for k, v in boosted.items()}
    return boosted, match_counts


def _domains_matching_version(
    store: SqliteStore,
    specs: dict[str, dict[str, Any]],
    version_filter: str,
) -> set[str]:
    rows = store.conn.execute(
        "SELECT DISTINCT domain FROM chunks WHERE source_version = ?",
        (version_filter,),
    ).fetchall()
    matched = {r["domain"] for r in rows if r["domain"] in specs}
    for dom, spec in specs.items():
        if version_filter in (spec.get("source_versions") or []):
            matched.add(dom)
    return matched


def route_query(
    *,
    cfg: BrainConfig,
    store: SqliteStore,
    models: Any,
    tracer: TraceLogger,
    query: str,
    explicit_domains: list[str] | None = None,
    version_filter: str | None = None,
    parent_trace_id: str | None = None,
    harness_id: str = "cli",
) -> RouteDecision:
    """Execute System 1 routing (Gate 0 -> synchronous Laya 421M INT8 -> taxonomy/k-NN calibration)."""
    t_start = time.perf_counter()
    trace_id = new_trace_id()
    specs = load_specialist_specs(cfg.vault_path)
    if not specs:
        raise RuntimeError(f"No specialist specs found under {cfg.vault_path / 'specialists'}")

    warnings: list[str] = []
    raw_laya_probs: dict[str, float] | None = None
    alias_matches: dict[str, int] = {}

    if explicit_domains:
        unknown = [d for d in explicit_domains if d not in specs]
        if unknown:
            raise ValueError(f"Unknown domain_scope requested: {unknown}. Valid: {sorted(specs.keys())}")
        selected = list(dict.fromkeys(explicit_domains))
        probs = {d: round(1.0 / len(selected), 4) for d in selected}
        gate = "gate0_explicit"
        laya_ms = 0.0
    else:
        g0_domains, g0_gate = _gate0_exact_match(query, store, specs)
        if g0_domains and g0_gate:
            selected = g0_domains[: cfg.max_fanout_domains]
            probs = {d: round(1.0 / len(selected), 4) for d in selected}
            gate = "gate0_exact_id"
            laya_ms = 0.0
        elif len(specs) == 1:
            only_dom = next(iter(specs.keys()))
            selected = [only_dom]
            probs = {only_dom: 1.0}
            gate = "gate0_single_domain"
            laya_ms = 0.0
        else:
            # Synchronous Laya 421M INT8 ONNX pass (Option P2 locked)
            t_laya0 = time.perf_counter()
            laya = models.get_laya()
            criteria = {dom: str(spec.get("description", dom))[:120] for dom, spec in specs.items()}
            laya_out = laya.predict_choice(
                state=query,
                instructions="Select the primary engineering domain for this technical request.",
                criteria=criteria,
                cap_tokens=256,
            )
            laya_ms = round((time.perf_counter() - t_laya0) * 1000.0, 2)
            raw_laya_probs = dict(laya_out["probabilities"])
            probs = dict(raw_laya_probs)
            gate = "laya_system1"

            # Apply deterministic taxonomy alias priors when running un-finetuned stock Laya
            if not laya.manifest.is_purpose_trained:
                taxonomy, _ = load_taxonomy(cfg.vault_path)
                probs, alias_matches = _apply_taxonomy_alias_priors(query, probs, taxonomy, specs)

            # Local certified-exemplar k-NN calibration using already-loaded embedder
            exemplar_count = int(store.conn.execute("SELECT COUNT(*) FROM exemplars").fetchone()[0])
            if exemplar_count > 0:
                emb = models.get_embedder()
                q_vec = np.asarray(next(iter(emb.embed([query]))), dtype=np.float32)
                knn = store.query_exemplars_knn(q_vec, top_k=3)
                if knn and knn[0]["similarity"] >= 0.88 and knn[0].get("expected_action", "approve") == "approve" and knn[0]["expected_domain"] in specs:
                    top_dom = knn[0]["expected_domain"]
                    probs[top_dom] = round(max(probs.get(top_dom, 0.0), 0.78), 4)
                    total = sum(probs.values()) or 1.0
                    probs = {k: round(v / total, 4) for k, v in probs.items()}
                    gate = "knn_override"

            ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
            selected = [ordered[0][0]]
            if len(ordered) > 1 and cfg.max_fanout_domains >= 2:
                top1_p, top2_p = ordered[0][1], ordered[1][1]
                if top2_p >= 0.25 and (top1_p - top2_p) <= 0.35:
                    selected.append(ordered[1][0])

            # Version affinity filter when source_version pin is requested
            if version_filter:
                ver_domains = _domains_matching_version(store, specs, version_filter)
                if ver_domains:
                    filtered = [d for d in selected if d in ver_domains]
                    if not filtered:
                        ordered_ver = [d for d, _ in ordered if d in ver_domains]
                        filtered = ordered_ver[: cfg.max_fanout_domains]
                    if filtered != selected:
                        warnings.append(
                            f"Version affinity filter adjusted selected_domains from {selected} to {filtered} for source_version={version_filter!r}."
                        )
                        selected = filtered

    multi_domain = len(selected) > 1
    branches: list[FanOutBranch] = []
    per_branch_budget = cfg.pack_token_budget
    aggregate_budget = min(cfg.fanout_token_budget, per_branch_budget * len(selected))

    for dom in selected:
        spec = specs[dom]
        versions = [version_filter] if version_filter else list(spec.get("source_versions") or [])
        branches.append(
            FanOutBranch(
                domain=dom,
                confidence=probs.get(dom, 0.0),
                scoped_query=query,
                specialist_uri=f"vault://specialists/{dom}.yaml",
                source_versions=versions,
                token_budget=min(per_branch_budget, int(spec.get("token_budget", per_branch_budget))),
            )
        )

    decision = RouteDecision(
        trace_id=trace_id,
        primary_domain=selected[0],
        selected_domains=selected,
        probabilities=probs,
        multi_domain=multi_domain,
        gate=gate,  # type: ignore[arg-type]
        source_versions=[version_filter] if version_filter else [],
        branches=branches,
        aggregate_token_budget=aggregate_budget,
        laya_latency_ms=laya_ms,
        warnings=warnings,
    )

    tracer.write(
        {
            "event": "route",
            "trace_id": trace_id,
            "parent_trace_id": parent_trace_id,
            "harness_id": harness_id,
            "query": query,
            "gate": gate,
            "selected_domains": selected,
            "raw_laya_probs": raw_laya_probs,
            "alias_matches": alias_matches,
            "probabilities": probs,
            "laya_latency_ms": laya_ms,
            "total_ms": round((time.perf_counter() - t_start) * 1000.0, 2),
        }
    )
    return decision
