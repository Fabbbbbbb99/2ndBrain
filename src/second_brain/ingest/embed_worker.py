"""Phase 2 incremental background ONNX embedding worker with SHA-256 keying and preemption."""
from __future__ import annotations

import os
import threading
import time
from contextlib import nullcontext
from typing import Any, Callable

import psutil

from second_brain.config import BrainConfig
from second_brain.storage.sqlite_store import SqliteStore, lf_sha256, serialize_unit_int8


def make_embed_key(model_id: str, model_rev: str, normalizer_ver: str, text: str) -> str:
    lf_text = text.replace("\r\n", "\n")
    return lf_sha256(f"{model_id}:{model_rev}:{normalizer_ver}:{lf_text}")


def embed_pending_chunks(
    *,
    cfg: BrainConfig,
    store: SqliteStore,
    embedder: Any,
    batch_size: int = 8,
    max_chunks: int | None = None,
    should_pause: Callable[[], bool] | None = None,
    low_priority: bool = False,
    write_lock: threading.Lock | None = None,
) -> dict[str, Any]:
    """Embed pending SQLite chunks incrementally while pausing cleanly between micro-batches."""
    proc = psutil.Process(os.getpid())
    prev_prio = None
    if low_priority:
        try:
            prev_prio = proc.nice()
            proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        except Exception:
            prev_prio = None

    processed = 0
    failed = 0
    paused_preemptions = 0
    micro_batch = max(1, min(batch_size, 16))
    lock_ctx = (lambda: write_lock) if write_lock is not None else nullcontext

    try:
        while True:
            if max_chunks is not None and processed >= max_chunks:
                break
            if should_pause is not None and should_pause():
                paused_preemptions += 1
                time.sleep(0.05)
                continue

            limit = micro_batch if max_chunks is None else min(micro_batch, max_chunks - processed)
            with lock_ctx():
                rows = store.pending_embeddings(limit=limit)
            if not rows:
                break

            texts = [r["text"] for r in rows]
            try:
                vecs = list(embedder.embed(texts))
                with lock_ctx():
                    for r, vec in zip(rows, vecs):
                        emb_key = r["embed_key"] or make_embed_key("BAAI/bge-small-en-v1.5", "v1.5", "v1", r["text"])
                        store.set_chunk_embedding(
                            uri=r["uri"],
                            embed_key=emb_key,
                            embedding_bytes=serialize_unit_int8(vec),
                            status="ready",
                            commit=False,
                        )
                        processed += 1
                    store.conn.commit()
            except Exception:
                with lock_ctx():
                    for r in rows:
                        emb_key = r["embed_key"] or "err"
                        store.set_chunk_embedding(r["uri"], emb_key, None, status="failed", commit=False)
                        failed += 1
                    store.conn.commit()
    finally:
        if prev_prio is not None:
            try:
                proc.nice(prev_prio)
            except Exception:
                pass

    with lock_ctx():
        cov = store.get_index_coverage()
    return {
        "embedded_now": processed,
        "failed_now": failed,
        "paused_preemptions": paused_preemptions,
        "index_coverage": cov.model_dump(),
    }
