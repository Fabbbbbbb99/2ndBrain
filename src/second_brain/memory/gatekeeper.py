"""Deterministic Write Gatekeeper with pairwise Laya noul conflict detection."""
from __future__ import annotations

from typing import Any

from second_brain.config import BrainConfig
from second_brain.memory.ledger import append_ledger_entry, new_memory_id, write_inbox_item
from second_brain.memory.recall import embed_memory_text, rank_memory_candidates
from second_brain.observability.tracer import TraceLogger, new_trace_id
from second_brain.storage.models import MemoryRecord, normalize_project_tag
from second_brain.storage.sqlite_store import SqliteStore, utc_now_iso

CONFLICT_THRESHOLD = 0.45
MAX_MEMORY_CHARS = 4000


def process_memory_write(
    *,
    cfg: BrainConfig,
    store: SqliteStore,
    models: Any,
    tracer: TraceLogger,
    content: str,
    domain: str = "global",
    ledger_type: str = "facts",
    project_tag: str | None = None,
    source_ref: str = "session:manual",
    explicit_user_save: bool = False,
    is_domain_claim: bool = False,
    is_ontology_edit: bool = False,
    effective_from: str | None = None,
    effective_to: str | None = None,
    version_scope: str | None = None,
    supersedes_id: str | None = None,
) -> dict[str, Any]:
    """Evaluate conflicts pairwise under Laya 1,024-token cap and enforce hard write staging rules."""
    cleaned_content = content.strip()
    if len(cleaned_content) > MAX_MEMORY_CHARS:
        raise ValueError(f"CONTENT_TOO_LONG: memory content exceeds {MAX_MEMORY_CHARS} characters.")

    trace_id = new_trace_id()
    now_iso = utc_now_iso()
    eff_from = effective_from or now_iso[:10]
    norm_tag = normalize_project_tag(project_tag)
    emb_bytes, emb_key, new_vec = embed_memory_text(models, cleaned_content)

    # Retrieve top active memories in project scope using hybrid relevance (not pure 5-newest LIFO)
    candidates = store.get_memory_candidates(
        [domain] if domain != "global" else ["global"],
        project_tag=norm_tag,
        include_untagged_for_project=bool(norm_tag),
        strict_untagged_only=(norm_tag is None),
        version_scope=version_scope,
        status="active",
        limit=500,
    )
    ranked, _ = rank_memory_candidates(
        candidates,
        query=cleaned_content,
        q_vec=new_vec,
        project_tag=norm_tag,
    )
    existing = [
        item["rec"]
        for item in ranked
        if not (supersedes_id and item["rec"].id == supersedes_id)
    ][:5]

    laya_manifest = None
    laya_advisory: dict[str, Any] = {"conflict_checks": [], "max_conflict_prob": 0.0}
    max_conflict_prob = 0.0
    conflict_target_id: str | None = None

    try:
        laya = models.get_laya()
        laya_manifest = laya.manifest
        for prev in existing:
            pair_state = (
                f"Existing record ({prev.id}): {prev.content[:400]}\n"
                f"New candidate claim: {cleaned_content[:400]}"
            )
            noul = laya.predict_noul(
                state=pair_state,
                instructions="Does the new candidate claim contradict or conflict with the existing record?",
                true_desc="yes, the new claim contradicts or changes the existing record",
                false_desc="no, the two statements are compatible",
                cap_tokens=1024,
            )
            p_conf = float(noul["p_true"])
            laya_advisory["conflict_checks"].append({"id": prev.id, "p_conflict": p_conf})
            if p_conf > max_conflict_prob:
                max_conflict_prob = p_conf
                conflict_target_id = prev.id
        laya_advisory["max_conflict_prob"] = round(max_conflict_prob, 4)
    except Exception as exc:
        laya_advisory["laya_error"] = str(exc)

    trained_auto_ok = bool(
        cfg.allow_laya_auto_commit
        and laya_manifest is not None
        and laya_manifest.is_purpose_trained
        and laya_manifest.allow_laya_auto_commit
    )

    reason_code: str | None = None
    if max_conflict_prob >= CONFLICT_THRESHOLD:
        reason_code = "conflict_detected"
    elif is_domain_claim:
        reason_code = "domain_claim_requires_certify"
    elif is_ontology_edit:
        reason_code = "ontology_edit_requires_certify"
    elif (not explicit_user_save) and (not trained_auto_ok):
        reason_code = "ai_proposal_staged_until_trained"

    status = "staged" if reason_code else "active"
    rec = MemoryRecord(
        id=new_memory_id(),
        domain=domain,
        ledger_type=ledger_type,  # type: ignore[arg-type]
        project_tag=norm_tag,
        effective_from=eff_from,
        effective_to=effective_to,
        version_scope=version_scope,
        status=status,  # type: ignore[arg-type]
        supersedes_id=supersedes_id,
        content=cleaned_content,
        source_ref=source_ref,
        reason_code=reason_code,
        laya_advisory=laya_advisory,
        created_at=now_iso,
    )

    store.upsert_memory_record(rec, embedding_bytes=emb_bytes, embed_key=emb_key, commit=False)
    if conflict_target_id and reason_code == "conflict_detected":
        store.upsert_edge(
            f"memory://{rec.id}",
            f"memory://{conflict_target_id}",
            "CONFLICTS_WITH",
            "INFERRED",
            confidence=max_conflict_prob,
            commit=False,
        )

    if status == "active":
        written_path = append_ledger_entry(cfg.vault_path, rec)
    else:
        written_path = write_inbox_item(cfg.vault_path, rec)
    store.conn.commit()

    tracer.write(
        {
            "event": "memory_write",
            "trace_id": trace_id,
            "memory_id": rec.id,
            "domain": domain,
            "project_tag": norm_tag,
            "candidate_pool_size": len(candidates),
            "embedding_persisted": emb_bytes is not None,
            "status": status,
            "reason_code": reason_code,
            "max_conflict_prob": max_conflict_prob,
            "file": str(written_path),
        }
    )

    return {
        "trace_id": trace_id,
        "memory_id": rec.id,
        "project_tag": norm_tag,
        "status": status,
        "reason_code": reason_code,
        "file_path": str(written_path),
        "conflict_target_id": conflict_target_id if reason_code == "conflict_detected" else None,
        "laya_advisory": laya_advisory,
    }
