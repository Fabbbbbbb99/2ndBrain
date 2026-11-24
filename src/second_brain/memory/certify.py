"""Human certification workflow (`brain certify`) feeding local k-NN exemplars and hard-negative traces."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from second_brain.config import BrainConfig
from second_brain.memory.ledger import append_ledger_entry, read_inbox_items
from second_brain.memory.recall import embed_memory_text
from second_brain.observability.tracer import TraceLogger, new_trace_id
from second_brain.storage.models import normalize_project_tag
from second_brain.storage.sqlite_store import SqliteStore, utc_now_iso


def list_inbox_items(cfg: BrainConfig) -> list[dict[str, Any]]:
    items = read_inbox_items(cfg.vault_path)
    return [
        {
            "file": str(path),
            **rec.model_dump(),
        }
        for path, rec in items
    ]


def apply_certification(
    *,
    cfg: BrainConfig,
    store: SqliteStore,
    models: Any,
    tracer: TraceLogger,
    item_id: str,
    decision: str,
    corrected_domain: str | None = None,
    edited_content: str | None = None,
    corrected_project_tag: str | None = None,
) -> dict[str, Any]:
    """Promote or reject a staged inbox item, updating ledger, memory embedding, and exemplars k-NN."""
    inbox_items = read_inbox_items(cfg.vault_path)
    match = next(((p, r) for p, r in inbox_items if r.id == item_id), None)
    if match is None:
        raise FileNotFoundError(f"Staged inbox item not found: {item_id!r}")

    inbox_path, rec = match
    final_domain = corrected_domain or rec.domain
    final_content = edited_content.strip() if edited_content else rec.content
    final_status = "active" if decision == "approve" else "rejected"
    final_project_tag = (
        normalize_project_tag(corrected_project_tag) if corrected_project_tag is not None else rec.project_tag
    )

    emb_bytes, emb_key, vec = embed_memory_text(models, final_content) if decision == "approve" else (None, None, None)

    updated = rec.model_copy(
        update={
            "domain": final_domain,
            "project_tag": final_project_tag,
            "content": final_content,
            "status": final_status,
        }
    )
    store.upsert_memory_record(updated, embedding_bytes=emb_bytes, embed_key=emb_key, commit=False)

    ledger_path: Path | None = None
    if final_status == "active":
        ledger_path = append_ledger_entry(cfg.vault_path, updated)

    # Index certified decision as an instant local k-NN exemplar for System 1 calibration
    exemplar_added = False
    if decision == "approve" and vec is not None:
        try:
            store.add_exemplar(
                exemplar_id=f"ex_{updated.id}",
                query_text=final_content[:500],
                expected_domain=final_domain,
                expected_action=decision,
                embedding_f32=vec,
                model_sha="bge-small-en-v1.5",
                commit=False,
            )
            exemplar_added = True
        except Exception:
            exemplar_added = False

    store.conn.commit()
    inbox_path.unlink(missing_ok=True)

    trace_id = new_trace_id()
    cert_log = cfg.logs_dir / "certify_traces.jsonl"
    cert_entry = {
        "timestamp": utc_now_iso(),
        "trace_id": trace_id,
        "memory_id": updated.id,
        "decision": decision,
        "original_domain": rec.domain,
        "final_domain": final_domain,
        "project_tag": final_project_tag,
        "reason_code": rec.reason_code,
        "text": final_content,
        "embedding_persisted": emb_bytes is not None,
        "hard_negative": bool(corrected_domain and corrected_domain != rec.domain) or (decision == "reject"),
    }
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    with cert_log.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(cert_entry, ensure_ascii=False) + "\n")

    tracer.write({"event": "certify", **cert_entry})

    return {
        "trace_id": trace_id,
        "memory_id": updated.id,
        "status": final_status,
        "domain": final_domain,
        "project_tag": final_project_tag,
        "ledger_path": str(ledger_path) if ledger_path else None,
        "exemplar_added": exemplar_added,
    }
