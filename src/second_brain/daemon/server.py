"""Resident shared braind daemon owning the ONNX model pool, SQLite WAL writer, and Vault lock."""
from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import psutil

from second_brain.config import BrainConfig, load_config
from second_brain.daemon.ipc import NamedPipeIpcServer
from second_brain.observability.tracer import TraceLogger, new_trace_id
from second_brain.storage.manifest import load_and_verify_manifest
from second_brain.storage.models import ClauseLookupResult
from second_brain.storage.spi import CodeAstAdapter, StoreUnconfiguredError
from second_brain.storage.sqlite_store import SqliteStore

API_VERSION = "2"
ADMIN_ONLY_RPCS = frozenset(
    {
        "ingest_pdf",
        "ingest_living_docs",
        "embed_pending",
        "certify_apply",
        "import_edges_apply",
        "model_swap",
        "model_rollback",
        "shutdown",
    }
)


class SharedModelPool:
    """Lazy-loading, single-instance CPU ONNX pool shared across all shims."""

    def __init__(self, cfg: BrainConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._embedder: Any = None
        self._reranker: Any = None
        self._laya: Any = None

    def get_embedder(self) -> Any:
        with self._lock:
            if self._embedder is None:
                from fastembed import TextEmbedding

                self.cfg.fastembed_dir.mkdir(parents=True, exist_ok=True)
                self._embedder = TextEmbedding(
                    model_name="BAAI/bge-small-en-v1.5",
                    cache_dir=str(self.cfg.fastembed_dir),
                    threads=min(6, self.cfg.intra_op_num_threads),
                    providers=["CPUExecutionProvider"],
                    enable_cpu_mem_arena=self.cfg.enable_cpu_mem_arena,
                )
            return self._embedder

    def get_reranker(self) -> Any:
        with self._lock:
            if self._reranker is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                self.cfg.fastembed_dir.mkdir(parents=True, exist_ok=True)
                self._reranker = TextCrossEncoder(
                    model_name="Xenova/ms-marco-MiniLM-L-6-v2",
                    cache_dir=str(self.cfg.fastembed_dir),
                    threads=min(6, self.cfg.intra_op_num_threads),
                    providers=["CPUExecutionProvider"],
                    enable_cpu_mem_arena=self.cfg.enable_cpu_mem_arena,
                )
            return self._reranker

    def get_laya(self) -> Any:
        with self._lock:
            if self._laya is None:
                from second_brain.router.laya_runtime import LayaOnnxRuntime

                self._laya = LayaOnnxRuntime.from_dir(
                    self.cfg.laya_dir,
                    vault_path=self.cfg.vault_path,
                    intra_op_num_threads=self.cfg.intra_op_num_threads,
                    inter_op_num_threads=self.cfg.inter_op_num_threads,
                    enable_cpu_mem_arena=self.cfg.enable_cpu_mem_arena,
                )
            return self._laya

    def reload_laya(self, model_dir: Path | None = None) -> dict[str, Any]:
        with self._lock:
            target = model_dir or self.cfg.laya_dir
            from second_brain.router.laya_runtime import LayaOnnxRuntime

            new_rt = LayaOnnxRuntime.from_dir(
                target,
                vault_path=self.cfg.vault_path,
                intra_op_num_threads=self.cfg.intra_op_num_threads,
                inter_op_num_threads=self.cfg.inter_op_num_threads,
                enable_cpu_mem_arena=self.cfg.enable_cpu_mem_arena,
            )
            self._laya = new_rt
            return new_rt.manifest.model_dump()

    def loaded_models(self) -> dict[str, bool]:
        return {
            "embedder": self._embedder is not None,
            "reranker": self._reranker is not None,
            "laya": self._laya is not None,
        }


class BrainDaemon:
    """Single-writer resident engine daemon serving IPC requests."""

    def __init__(self, cfg: BrainConfig) -> None:
        self.cfg = cfg
        self.manifest, self.taxonomy = load_and_verify_manifest(cfg.vault_path)
        self.store = SqliteStore(cfg.db_path)
        self._sync_db_meta()
        self.tracer = TraceLogger(cfg.logs_dir)
        self.models = SharedModelPool(cfg)
        self.code_adapter = CodeAstAdapter()
        self._write_lock = threading.Lock()
        self._rpc_lock = threading.Lock()
        self._active_interactive_rpcs = 0
        self._started_at = time.monotonic()
        self._ipc: NamedPipeIpcServer | None = None
        if self.cfg.certified_edges_path.exists():
            from second_brain.graph.import_edges import sync_certified_edges_from_vault

            sync_certified_edges_from_vault(self.cfg, self.store)

    @contextmanager
    def _read_store(self) -> Iterator[SqliteStore]:
        rstore = SqliteStore(self.cfg.db_path, readonly=True)
        try:
            yield rstore
        finally:
            rstore.close()

    def _sync_db_meta(self) -> None:
        self.store.set_meta(
            {
                "vault_schema_version": self.manifest.vault_schema_version,
                "ontology_hash": self.manifest.ontology_hash,
                "parser_version": self.manifest.parser_version,
                "embed_model_id": self.manifest.embed_model_id,
                "rerank_model_id": self.manifest.rerank_model_id,
                "laya_model_id": self.manifest.laya_model_id,
            }
        )

    @property
    def active_interactive_rpcs(self) -> int:
        with self._rpc_lock:
            return self._active_interactive_rpcs

    def _begin_rpc(self) -> None:
        with self._rpc_lock:
            self._active_interactive_rpcs += 1

    def _end_rpc(self) -> None:
        with self._rpc_lock:
            self._active_interactive_rpcs = max(0, self._active_interactive_rpcs - 1)

    def handle_rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method in ADMIN_ONLY_RPCS and params.get("harness_id") == "mcp-stdio":
            raise PermissionError(f"RPC method {method!r} is CLI-only and disallowed over MCP stdio.")
        self._begin_rpc()
        try:
            dispatch = {
                "ping": self._rpc_ping,
                "shutdown": self._rpc_shutdown,
                "status": self._rpc_status,
                "get_clause": self._rpc_get_clause,
                "grep": self._rpc_grep,
                "recall": self._rpc_recall,
                "route": self._rpc_route,
                "query": self._rpc_query,
                "remember": self._rpc_remember,
                "compile": self._rpc_compile,
                "cache_visual_repair": self._rpc_cache_visual_repair,
                "ingest_pdf": self._rpc_ingest_pdf,
                "ingest_living_docs": self._rpc_ingest_living_docs,
                "embed_pending": self._rpc_embed_pending,
                "certify_list": self._rpc_certify_list,
                "certify_apply": self._rpc_certify_apply,
                "import_edges_apply": self._rpc_import_edges_apply,
                "model_status": self._rpc_model_status,
                "model_swap": self._rpc_model_swap,
                "model_rollback": self._rpc_model_rollback,
                "code_search": self._rpc_code_search,
            }
            fn = dispatch.get(method)
            if fn is None:
                raise ValueError(f"Unknown RPC method: {method!r}")
            return fn(params)
        finally:
            self._end_rpc()

    def _rpc_ping(self, _params: dict[str, Any]) -> dict[str, Any]:
        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        return {
            "pong": True,
            "api_version": API_VERSION,
            "pid": os.getpid(),
            "pipe_name": self.cfg.pipe_name,
            "uptime_s": round(time.monotonic() - self._started_at, 2),
            "private_mb": round(getattr(mem, "private", mem.rss) / (1024 * 1024), 2),
            "loaded_models": self.models.loaded_models(),
        }

    def _rpc_shutdown(self, _params: dict[str, Any]) -> dict[str, Any]:
        if self._ipc is not None:
            threading.Timer(0.05, self._ipc.request_stop).start()
        return {"stopping": True, "pid": os.getpid()}

    def _rpc_status(self, params: dict[str, Any]) -> dict[str, Any]:
        from second_brain.memory.certify import list_inbox_items

        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        domain_scope = params.get("domain_scope")
        with self._read_store() as rstore:
            cov = rstore.get_index_coverage(domain_scope)
            stores = [*rstore.get_store_statuses(), self.code_adapter.health()]
            staged_cnt = rstore.count_memories_by_status("staged", domain_scope=domain_scope)
            null_emb_cnt = rstore.count_memories_missing_embeddings()
        inbox_cnt = len(list_inbox_items(self.cfg))
        return {
            "api_version": API_VERSION,
            "pid": os.getpid(),
            "vault_path": str(self.cfg.vault_path),
            "state_dir": str(self.cfg.state_dir),
            "private_mb": round(getattr(mem, "private", mem.rss) / (1024 * 1024), 2),
            "loaded_models": self.models.loaded_models(),
            "manifest": self.manifest.model_dump(),
            "index_coverage": cov.model_dump(),
            "store_statuses": [s.model_dump() for s in stores],
            "staged_memory_count": staged_cnt,
            "inbox_file_count": inbox_cnt,
            "staged_sync_drift": staged_cnt != inbox_cnt,
            "memory_null_embedding_count": null_emb_cnt,
        }

    def _rpc_get_clause(self, params: dict[str, Any]) -> dict[str, Any]:
        from second_brain.graph.import_edges import load_certified_edge_hashes

        uri = str(params["uri"])
        include_text = bool(params.get("include_text", True))
        neighbor_limit = max(0, min(int(params.get("neighbor_limit", 12)), 50))
        if uri.startswith("repo://"):
            self.code_adapter.get_by_uri(uri)

        trace_id = new_trace_id()
        with self._read_store() as rstore:
            env = rstore.get_chunk_by_uri(uri)
            if env is None:
                res = ClauseLookupResult(found=False, uri=uri).model_dump()
                self.tracer.write({"event": "get_clause", "trace_id": trace_id, "uri": uri, "found": False})
                return res

            if env.store == "wiki":
                env.stale = rstore.check_compiled_staleness(uri)
                if env.stale:
                    env.status = "stale"
            neighbors = rstore.get_neighbors([uri], [env.domain], limit=neighbor_limit, exclude_inferred=True)
            children, ch_tot = rstore.get_clause_children(uri, limit=neighbor_limit)
            cited_by, cb_tot = rstore.get_clause_cited_by(uri, limit=neighbor_limit)
            cert_hashes = load_certified_edge_hashes(self.cfg)
            inferred_links, inf_tot = rstore.get_inferred_links(
                uri, limit=neighbor_limit, certified_hashes=cert_hashes
            )
            visuals = rstore.get_visual_nodes_for_uris([uri])

        item_dict = env.model_dump()
        text_omitted = False
        if not include_text:
            item_dict["text"] = ""
            text_omitted = True

        self.tracer.write(
            {
                "event": "get_clause",
                "trace_id": trace_id,
                "uri": uri,
                "found": True,
                "include_text": include_text,
                "children_total": ch_tot,
                "cited_by_total": cb_tot,
                "inferred_links_total": inf_tot,
            }
        )
        return ClauseLookupResult(
            found=True,
            uri=uri,
            item=item_dict,
            text_omitted=text_omitted,
            neighbors=neighbors,
            children=children,
            cited_by=cited_by,
            inferred_links=inferred_links,
            totals={
                "neighbors": len(neighbors),
                "children": ch_tot,
                "cited_by": cb_tot,
                "inferred_links": inf_tot,
            },
            visual_nodes=visuals,
        ).model_dump()

    def _rpc_grep(self, params: dict[str, Any]) -> dict[str, Any]:
        from second_brain.router.grep import execute_grep

        with self._read_store() as rstore:
            res = execute_grep(
                cfg=self.cfg,
                store=rstore,
                tracer=self.tracer,
                pattern=str(params["pattern"]),
                domain_scope=params.get("domain_scope"),
                source_id=params.get("source_id"),
                source_version=params.get("source_version"),
                regex=bool(params.get("regex", False)),
                case_sensitive=bool(params.get("case_sensitive", False)),
                max_matches=int(params.get("max_matches", 10)),
                token_budget=int(params.get("token_budget", self.cfg.pack_token_budget)),
                parent_trace_id=params.get("parent_trace_id"),
                harness_id=str(params.get("harness_id", "mcp")),
            )
        return res.model_dump()

    def _rpc_recall(self, params: dict[str, Any]) -> dict[str, Any]:
        from second_brain.memory.recall import execute_recall

        with self._read_store() as rstore:
            res = execute_recall(
                cfg=self.cfg,
                store=rstore,
                models=self.models,
                tracer=self.tracer,
                query=params.get("query"),
                domain_scope=params.get("domain_scope"),
                project_tag=params.get("project_tag"),
                ledger_type=params.get("ledger_type"),
                version_scope=params.get("version_scope"),
                status=str(params.get("status", "active")),
                limit=int(params.get("limit", 15)),
                parent_trace_id=params.get("parent_trace_id"),
                harness_id=str(params.get("harness_id", "mcp")),
            )
        return res.model_dump()

    def _rpc_code_search(self, params: dict[str, Any]) -> dict[str, Any]:
        workspace_root = params.get("workspace_root")
        adapter = CodeAstAdapter(workspace_root=workspace_root)
        try:
            adapter.search(
                query=str(params.get("query", "")),
                domain_scope=list(params.get("domain_scope") or []),
                version_filter=params.get("version_filter"),
                limit=int(params.get("limit", 10)),
            )
        except StoreUnconfiguredError as exc:
            return {"status": adapter.health().model_dump(), "error": str(exc), "items": []}
        return {"status": adapter.health().model_dump(), "items": []}

    def _rpc_route(self, params: dict[str, Any]) -> dict[str, Any]:
        from second_brain.router.system1 import route_query

        with self._write_lock:
            decision = route_query(
                cfg=self.cfg,
                store=self.store,
                models=self.models,
                tracer=self.tracer,
                query=str(params["query"]),
                explicit_domains=params.get("domain_scope"),
                version_filter=params.get("source_version"),
                parent_trace_id=params.get("parent_trace_id"),
                harness_id=str(params.get("harness_id", "mcp")),
            )
        return decision.model_dump()

    def _rpc_query(self, params: dict[str, Any]) -> dict[str, Any]:
        from second_brain.router.retrieval import execute_query

        with self._write_lock:
            pack = execute_query(
                cfg=self.cfg,
                store=self.store,
                models=self.models,
                tracer=self.tracer,
                query=str(params["query"]),
                domain_scope=params.get("domain_scope"),
                source_version=params.get("source_version"),
                source_id=params.get("source_id"),
                project_tag=params.get("project_tag"),
                token_budget=int(params.get("token_budget", self.cfg.pack_token_budget)),
                include_rerank=not bool(params.get("no_rerank", False)),
                parent_trace_id=params.get("parent_trace_id"),
                harness_id=str(params.get("harness_id", "mcp")),
            )
        return pack.model_dump()

    def _rpc_remember(self, params: dict[str, Any]) -> dict[str, Any]:
        from second_brain.memory.gatekeeper import process_memory_write

        with self._write_lock:
            return process_memory_write(
                cfg=self.cfg,
                store=self.store,
                models=self.models,
                tracer=self.tracer,
                content=str(params["content"]),
                domain=str(params.get("domain", "global")),
                ledger_type=str(params.get("ledger_type", "facts")),
                project_tag=params.get("project_tag"),
                source_ref=str(params.get("source_ref", "session:manual")),
                explicit_user_save=bool(params.get("explicit_user_save", False)),
                is_domain_claim=bool(params.get("is_domain_claim", False)),
                is_ontology_edit=bool(params.get("is_ontology_edit", False)),
                effective_from=params.get("effective_from"),
                effective_to=params.get("effective_to"),
                version_scope=params.get("version_scope"),
                supersedes_id=params.get("supersedes_id"),
            )

    def _rpc_compile(self, params: dict[str, Any]) -> dict[str, Any]:
        from second_brain.specialists.compiler import compile_wiki_page

        with self._write_lock:
            return compile_wiki_page(
                cfg=self.cfg,
                store=self.store,
                domain=str(params["domain"]),
                slug=str(params["slug"]),
                title=str(params["title"]),
                summary_markdown=str(params["summary_markdown"]),
                cited_uris=list(params["cited_uris"]),
                source_version=str(params.get("source_version", "compiled")),
            )

    def _rpc_cache_visual_repair(self, params: dict[str, Any]) -> dict[str, Any]:
        from second_brain.ingest.pdf_adapter import cache_visual_repair

        with self._write_lock:
            return cache_visual_repair(
                cfg=self.cfg,
                store=self.store,
                visual_id=str(params["visual_id"]),
                repaired_markdown=str(params["repaired_markdown"]),
                certify=bool(params.get("certify", False)),
            )

    def _rpc_ingest_pdf(self, params: dict[str, Any]) -> dict[str, Any]:
        from second_brain.ingest.pdf_adapter import ingest_pdf_document

        with self._write_lock:
            result = ingest_pdf_document(
                cfg=self.cfg,
                store=self.store,
                pdf_path=Path(str(params["pdf_path"])),
                domain=str(params["domain"]),
                source_id=str(params["source_id"]),
                source_version=str(params["source_version"]),
                authority_tier=str(params.get("authority_tier", "primary_normative")),
                run_docling_tables=bool(params.get("run_docling_tables", False)),
            )
        if not bool(params.get("lexical_only", False)):
            from second_brain.ingest.embed_worker import embed_pending_chunks

            emb_stats = embed_pending_chunks(
                cfg=self.cfg,
                store=self.store,
                embedder=self.models.get_embedder(),
                batch_size=int(params.get("embed_batch_size", 32)),
                max_chunks=params.get("embed_limit"),
                should_pause=lambda: self.active_interactive_rpcs > 1,
                write_lock=self._write_lock,
            )
            result["embedding"] = emb_stats
        return result

    def _rpc_ingest_living_docs(self, params: dict[str, Any]) -> dict[str, Any]:
        from second_brain.ingest.living_docs import ingest_living_docs_dir

        with self._write_lock:
            result = ingest_living_docs_dir(
                cfg=self.cfg,
                store=self.store,
                docs_root=Path(str(params["docs_root"])),
                domain=str(params["domain"]),
                source_id=str(params["source_id"]),
                source_version=str(params["source_version"]),
            )
        if not bool(params.get("lexical_only", False)):
            from second_brain.ingest.embed_worker import embed_pending_chunks

            emb_stats = embed_pending_chunks(
                cfg=self.cfg,
                store=self.store,
                embedder=self.models.get_embedder(),
                batch_size=int(params.get("embed_batch_size", 32)),
                max_chunks=params.get("embed_limit"),
                should_pause=lambda: self.active_interactive_rpcs > 1,
                write_lock=self._write_lock,
            )
            result["embedding"] = emb_stats
        return result

    def _rpc_embed_pending(self, params: dict[str, Any]) -> dict[str, Any]:
        from second_brain.ingest.embed_worker import embed_pending_chunks

        return embed_pending_chunks(
            cfg=self.cfg,
            store=self.store,
            embedder=self.models.get_embedder(),
            batch_size=int(params.get("batch_size", 32)),
            max_chunks=params.get("max_chunks"),
            should_pause=lambda: self.active_interactive_rpcs > 1,
            write_lock=self._write_lock,
        )

    def _rpc_certify_list(self, _params: dict[str, Any]) -> dict[str, Any]:
        from second_brain.memory.certify import list_inbox_items

        with self._write_lock:
            return {"items": list_inbox_items(self.cfg)}

    def _rpc_certify_apply(self, params: dict[str, Any]) -> dict[str, Any]:
        from second_brain.memory.certify import apply_certification

        with self._write_lock:
            return apply_certification(
                cfg=self.cfg,
                store=self.store,
                models=self.models,
                tracer=self.tracer,
                item_id=str(params["item_id"]),
                decision=str(params["decision"]),
                corrected_domain=params.get("corrected_domain"),
                edited_content=params.get("edited_content"),
                corrected_project_tag=params.get("corrected_project_tag"),
            )

    def _rpc_import_edges_apply(self, params: dict[str, Any]) -> dict[str, Any]:
        from second_brain.graph.import_edges import apply_certified_edges

        rev_path = Path(str(params["review_path"])) if params.get("review_path") else None
        with self._write_lock:
            return apply_certified_edges(
                cfg=self.cfg,
                store=self.store,
                tracer=self.tracer,
                review_path=rev_path,
                edges_payload=params.get("edges"),
                replace_inferred=bool(params.get("replace_inferred", False)),
                dry_run=bool(params.get("dry_run", False)),
            )

    def _rpc_model_status(self, _params: dict[str, Any]) -> dict[str, Any]:
        from second_brain.router.laya_runtime import read_laya_manifest

        manifest = read_laya_manifest(self.cfg.laya_dir, self.cfg.vault_path)
        return {
            "active_dir": str(self.cfg.laya_dir),
            "manifest": manifest.model_dump(),
            "loaded": self.models.loaded_models(),
        }

    def _rpc_model_swap(self, params: dict[str, Any]) -> dict[str, Any]:
        from second_brain.router.laya_runtime import swap_laya_model

        with self._write_lock:
            info = swap_laya_model(self.cfg, Path(str(params["model_dir"])))
            self.models.reload_laya(self.cfg.laya_dir)
            return info

    def _rpc_model_rollback(self, _params: dict[str, Any]) -> dict[str, Any]:
        from second_brain.router.laya_runtime import rollback_laya_model

        with self._write_lock:
            info = rollback_laya_model(self.cfg)
            self.models.reload_laya(self.cfg.laya_dir)
            return info

    def serve(self) -> None:
        self._ipc = NamedPipeIpcServer(
            pipe_name=self.cfg.pipe_name,
            lock_path=self.cfg.lock_path,
            handler=self.handle_rpc,
            idle_ttl_seconds=self.cfg.daemon_idle_ttl_seconds,
        )
        try:
            self._ipc.serve_forever()
        finally:
            self.store.close()


def main() -> None:
    cfg = load_config()
    daemon = BrainDaemon(cfg)
    daemon.serve()


if __name__ == "__main__":
    main()
