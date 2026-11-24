"""Thin CLI shim (`brain`) communicating with braind and local utilities."""
from __future__ import annotations

import argparse
from importlib import resources
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import tomllib
import winreg
from pathlib import Path
from typing import Any

import onnxruntime as ort
import sqlite_vec
import yaml

from second_brain.config import ConfigError, default_config_path, load_config
from second_brain.daemon.ipc import NamedPipeIpcClient, is_daemon_running, spawn_daemon_background
from second_brain.observability.tracer import TraceLogger
from second_brain.router.laya_runtime import LayaManifestError, pull_stock_models, read_laya_manifest
from second_brain.storage.manifest import ManifestDriftError, load_and_verify_manifest, load_taxonomy
from second_brain.storage.sqlite_store import atomic_write_lf

VAULT_TEMPLATE_MAP: dict[str, str] = {
    "gitignore.tmpl": ".gitignore",
    "githooks/pre-commit.tmpl": ".githooks/pre-commit",
    "githooks/pre-push.tmpl": ".githooks/pre-push",
    "obsidian/app.json": ".obsidian/app.json",
    "obsidian/appearance.json": ".obsidian/appearance.json",
    "obsidian/core-plugins.json": ".obsidian/core-plugins.json",
    "ontology/taxonomy.yaml": "ontology/taxonomy.yaml",
    "specialists/example_domain.yaml": "specialists/example_domain.yaml",
    "memory/decisions.md": "memory/decisions.md",
    "memory/facts.md": "memory/facts.md",
    "memory/preferences.md": "memory/preferences.md",
    "index.md": "index.md",
    "log.md": "log.md",
    "vault_manifest.yaml": "vault_manifest.yaml",
}


def _resolve_config_arg(config_arg: str | None) -> Path:
    return Path(config_arg).expanduser().resolve() if config_arg else default_config_path()


def _materialize_vault_templates(vault: Path) -> list[str]:
    """Copy bundled starter Vault files into missing target paths without overwriting user content."""
    tmpl_root = resources.files("second_brain.templates.vault")
    has_existing_specialists = any((vault / "specialists").glob("*.yaml"))
    created: list[str] = []

    for src_rel, dst_rel in VAULT_TEMPLATE_MAP.items():
        if src_rel == "specialists/example_domain.yaml" and has_existing_specialists:
            continue
        dst_path = vault / dst_rel
        if dst_path.exists():
            continue
        src_node = tmpl_root
        for part in src_rel.split("/"):
            src_node = src_node.joinpath(part)
        atomic_write_lf(dst_path, src_node.read_text(encoding="utf-8"))
        created.append(dst_rel)
    return created


def _default_antigravity_skills_dir() -> Path:
    cfg_skills = Path.home() / ".gemini" / "config" / "skills"
    legacy_root = Path.home() / ".gemini" / "antigravity"
    if (Path.home() / ".gemini" / "config").exists() or not legacy_root.exists():
        return cfg_skills
    return legacy_root / "skills"


def _check_long_paths_enabled() -> bool:
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\FileSystem",
        ) as key:
            val, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
            return bool(int(val) == 1)
    except OSError:
        return False


def _check_onedrive_desktop(vault_path: Path) -> bool:
    return "onedrive" in str(vault_path).lower()


def _run_stdio_framing_check(cfg_path: Path | None = None) -> dict[str, Any]:
    """Spawn brain-mcp / python -m second_brain.shims.mcp_proxy and inspect raw stdout bytes for stray CR or noise."""
    init_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "brain-doctor", "version": "0.1.0"},
        },
    }
    raw_in = (json.dumps(init_req) + "\n").encode("utf-8")
    env = os.environ.copy()
    if cfg_path is not None:
        env["BRAIN_CONFIG_PATH"] = str(cfg_path)
    proc = subprocess.run(
        [sys.executable, "-m", "second_brain.shims.mcp_proxy"],
        input=raw_in,
        capture_output=True,
        env=env,
        timeout=15,
        check=False,
    )
    out = proc.stdout
    has_cr = b"\r" in out
    lines = [ln for ln in out.split(b"\n") if ln]
    valid_json = False
    if lines:
        try:
            parsed = json.loads(lines[0].decode("utf-8"))
            valid_json = isinstance(parsed, dict) and parsed.get("id") == 1
        except Exception:
            valid_json = False
    return {
        "exit_code": proc.returncode,
        "bytes_out": len(out),
        "has_cr": has_cr,
        "valid_jsonrpc_init": valid_json,
        "passed": (not has_cr) and valid_json,
    }


def _cmd_init(args: argparse.Namespace) -> int:
    cfg_path = _resolve_config_arg(args.config)
    if args.vault:
        vault = Path(args.vault).expanduser().resolve()
    elif args.sync_manifest and cfg_path.exists():
        vault = load_config(config_path=cfg_path, require_existing_vault=False).vault_path
    else:
        vault = (Path.home() / "2ndBrain_Vault").resolve()

    if args.state:
        state = Path(args.state).expanduser().resolve()
    elif args.sync_manifest and cfg_path.exists():
        state = load_config(config_path=cfg_path, require_existing_vault=False).state_dir
    else:
        state = (Path(os.environ.get("LOCALAPPDATA", ".")) / "2ndBrain" / "state").resolve()

    if cfg_path.exists() and (not args.force) and (not args.sync_manifest):
        existing_raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
        raw_vault = existing_raw.get("vault_path")
        if raw_vault:
            existing_vault = Path(str(raw_vault)).expanduser().resolve()
            if existing_vault != vault and existing_vault.exists():
                raise ConfigError(
                    f"Config at {cfg_path} already points to existing vault {existing_vault}. "
                    "Pass --force to overwrite or --config to target an isolated config file."
                )

    for sub in ("raw", "domains", "memory/inbox", "ontology", "specialists", ".obsidian", ".githooks"):
        (vault / sub).mkdir(parents=True, exist_ok=True)
    for sub in ("models", "parsed", "crops", "logs"):
        (state / sub).mkdir(parents=True, exist_ok=True)

    created_templates = _materialize_vault_templates(vault)

    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    if not cfg_path.exists() or args.force:
        threads = min(8, max(2, (os.cpu_count() or 4) // 2))
        content = (
            "# 2ndBrain Local Configuration\n"
            f"vault_path = {json.dumps(str(vault))}\n"
            f"state_dir = {json.dumps(str(state))}\n"
            "allow_laya_auto_commit = false\n"
            "max_fanout_domains = 2\n"
            "pack_token_budget = 2500\n"
            "fanout_token_budget = 12000\n"
            f"intra_op_num_threads = {threads}\n"
            "inter_op_num_threads = 1\n"
            "enable_cpu_mem_arena = false\n"
            "warm_cascade_p95_budget_ms = 1350\n"
            "daemon_idle_ttl_seconds = 1800\n"
        )
        atomic_write_lf(cfg_path, content)

    if args.sync_manifest or ("vault_manifest.yaml" in created_templates):
        _, digest = load_taxonomy(vault)
        mp = vault / "vault_manifest.yaml"
        data = yaml.safe_load(mp.read_text(encoding="utf-8"))
        data["ontology_hash"] = digest
        atomic_write_lf(mp, yaml.safe_dump(data, sort_keys=False))

    print(
        json.dumps(
            {
                "initialized": True,
                "config_path": str(cfg_path),
                "vault": str(vault),
                "state": str(state),
                "templates_created": created_templates,
            },
            indent=2,
        )
    )
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    cfg_path = _resolve_config_arg(args.config)
    cfg = load_config(config_path=cfg_path)
    manifest = None
    manifest_valid = True
    manifest_error: str | None = None
    try:
        manifest, _ = load_and_verify_manifest(cfg.vault_path)
    except ManifestDriftError as exc:
        manifest_valid = False
        manifest_error = str(exc)

    providers = ort.get_available_providers()
    cpu_ok = "CPUExecutionProvider" in providers and "DmlExecutionProvider" not in providers

    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    vec_ver = conn.execute("SELECT vec_version()").fetchone()[0]
    fts5_ok = any("FTS5" in r[0] for r in conn.execute("PRAGMA compile_options").fetchall())
    conn.close()

    long_paths = _check_long_paths_enabled()
    onedrive_risk = _check_onedrive_desktop(cfg.vault_path) or _check_onedrive_desktop(cfg.state_dir)
    onedrive_ok = (not onedrive_risk) or bool(args.allow_onedrive)

    laya_ok = False
    laya_error: str | None = None
    try:
        read_laya_manifest(cfg.laya_dir, vault_path=cfg.vault_path)
        if not (cfg.laya_dir / "tokenizer.json").exists():
            raise LayaManifestError(f"Missing tokenizer.json in {cfg.laya_dir}")
        laya_ok = True
    except Exception as exc:
        laya_error = str(exc)

    models_ok = laya_ok or bool(args.allow_missing_models)
    running, lock_info = is_daemon_running(cfg.pipe_name, cfg.lock_path)

    stdio_report = _run_stdio_framing_check(cfg_path) if args.stdio_check else None
    all_ok = (
        cpu_ok
        and fts5_ok
        and manifest_valid
        and onedrive_ok
        and models_ok
        and (stdio_report is None or stdio_report["passed"])
    )

    report = {
        "ok": all_ok,
        "python": sys.version.split()[0],
        "config_path": str(cfg_path),
        "ort_providers": providers,
        "cpu_only_ort_ok": cpu_ok,
        "sqlite_version": sqlite3.sqlite_version,
        "sqlite_vec_version": vec_ver,
        "fts5_enabled": fts5_ok,
        "windows_long_paths_enabled": long_paths,
        "onedrive_sync_risk": onedrive_risk,
        "vault_manifest_valid": manifest_valid,
        "manifest_error": manifest_error,
        "ontology_hash": manifest.ontology_hash if manifest else None,
        "laya_manifest_present": laya_ok,
        "laya_valid": laya_ok,
        "laya_error": laya_error,
        "daemon_running": running,
        "daemon_lock": lock_info,
        "stdio_check": stdio_report,
    }
    print(json.dumps(report, indent=2))
    return 0 if all_ok else 2


def _client_from_cfg() -> tuple[Any, NamedPipeIpcClient]:
    cfg = load_config()
    client = NamedPipeIpcClient(
        pipe_name=cfg.pipe_name,
        vault_path=cfg.vault_path,
        state_dir=cfg.state_dir,
        auto_spawn=True,
    )
    return cfg, client


def _cmd_daemon(args: argparse.Namespace) -> int:
    cfg = load_config()
    if args.action == "status":
        running, info = is_daemon_running(cfg.pipe_name, cfg.lock_path)
        res: dict[str, Any] = {"running": running, "pipe_name": cfg.pipe_name, "lock": info}
        if running:
            res["status"] = NamedPipeIpcClient(cfg.pipe_name, cfg.vault_path, cfg.state_dir, auto_spawn=False).call("status")
        print(json.dumps(res, indent=2))
        return 0
    if args.action == "start":
        running, _ = is_daemon_running(cfg.pipe_name, cfg.lock_path)
        if not running:
            spawn_daemon_background(cfg.vault_path, cfg.state_dir)
            for _ in range(100):
                ok, _ = is_daemon_running(cfg.pipe_name, cfg.lock_path)
                if ok:
                    break
                time.sleep(0.05)
        ping = NamedPipeIpcClient(cfg.pipe_name, cfg.vault_path, cfg.state_dir, auto_spawn=False).call("ping")
        print(json.dumps(ping, indent=2))
        return 0
    if args.action == "stop":
        running, _ = is_daemon_running(cfg.pipe_name, cfg.lock_path)
        if not running:
            print(json.dumps({"stopped": False, "reason": "not running"}))
            return 0
        resp = NamedPipeIpcClient(cfg.pipe_name, cfg.vault_path, cfg.state_dir, auto_spawn=False).call("shutdown")
        print(json.dumps(resp, indent=2))
        return 0
    return 1


def _cmd_query(args: argparse.Namespace) -> int:
    _, client = _client_from_cfg()
    res = client.call(
        "query",
        {
            "query": args.text,
            "domain_scope": args.domain or None,
            "source_version": args.version,
            "source_id": args.source_id,
            "project_tag": args.project_tag,
            "token_budget": args.budget,
            "no_rerank": args.no_rerank,
            "harness_id": "cli",
        },
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


def _cmd_grep(args: argparse.Namespace) -> int:
    _, client = _client_from_cfg()
    res = client.call(
        "grep",
        {
            "pattern": args.pattern,
            "domain_scope": args.domain or None,
            "source_id": args.source_id,
            "source_version": args.version,
            "regex": args.regex,
            "case_sensitive": args.case_sensitive,
            "max_matches": args.max_matches,
            "token_budget": args.budget,
            "harness_id": "cli",
        },
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


def _cmd_recall(args: argparse.Namespace) -> int:
    _, client = _client_from_cfg()
    res = client.call(
        "recall",
        {
            "query": args.query,
            "domain_scope": args.domain or None,
            "project_tag": args.project_tag,
            "ledger_type": args.ledger,
            "version_scope": args.version,
            "status": args.status,
            "limit": args.limit,
            "harness_id": "cli",
        },
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


def _cmd_embed(args: argparse.Namespace) -> int:
    _, client = _client_from_cfg()
    res = client.call(
        "embed_pending",
        {
            "batch_size": args.batch_size,
            "max_chunks": args.max_chunks,
        },
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    _, client = _client_from_cfg()
    if args.kind == "pdf":
        res = client.call(
            "ingest_pdf",
            {
                "pdf_path": str(Path(args.path).expanduser().resolve()),
                "domain": args.domain,
                "source_id": args.source_id,
                "source_version": args.version,
                "lexical_only": args.lexical_only,
                "run_docling_tables": args.docling_tables,
            },
        )
    else:
        res = client.call(
            "ingest_living_docs",
            {
                "docs_root": str(Path(args.path).expanduser().resolve()),
                "domain": args.domain,
                "source_id": args.source_id,
                "source_version": args.version,
                "lexical_only": args.lexical_only,
            },
        )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


def _cmd_certify(args: argparse.Namespace) -> int:
    _, client = _client_from_cfg()
    if not args.item_id:
        print(json.dumps(client.call("certify_list"), indent=2, ensure_ascii=False))
        return 0
    res = client.call(
        "certify_apply",
        {
            "item_id": args.item_id,
            "decision": args.decision,
            "corrected_domain": args.domain,
            "edited_content": args.content,
            "corrected_project_tag": args.project_tag,
        },
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


def _cmd_import_edges(args: argparse.Namespace) -> int:
    if args.action == "stage":
        from second_brain.graph.import_edges import stage_graphify_edges
        from second_brain.storage.sqlite_store import SqliteStore

        cfg = load_config()
        store = SqliteStore(cfg.db_path, readonly=True)
        try:
            res = stage_graphify_edges(
                cfg=cfg,
                store=store,
                graph_path=Path(args.input_path).expanduser().resolve(),
                out_path=Path(args.out).expanduser().resolve(),
                min_confidence=args.min_confidence,
                cross_doc_only=args.cross_doc_only,
            )
        finally:
            store.close()
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    if args.action == "apply":
        _, client = _client_from_cfg()
        res = client.call(
            "import_edges_apply",
            {
                "review_path": str(Path(args.input_path).expanduser().resolve()),
                "replace_inferred": args.replace_inferred,
                "dry_run": args.dry_run,
            },
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    return 1


def _cmd_compile(args: argparse.Namespace) -> int:
    _, client = _client_from_cfg()
    summary = Path(args.summary_file).read_text(encoding="utf-8") if args.summary_file else args.summary
    res = client.call(
        "compile",
        {
            "domain": args.domain,
            "slug": args.slug,
            "title": args.title,
            "summary_markdown": summary,
            "cited_uris": args.cite,
            "source_version": args.version,
        },
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


def _cmd_export_skills(args: argparse.Namespace) -> int:
    from second_brain.specialists.exporters.antigravity import export_antigravity_skills

    cfg = load_config(config_path=_resolve_config_arg(args.config))
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _default_antigravity_skills_dir()
    res = export_antigravity_skills(
        cfg=cfg,
        target_dir=out_dir,
        dry_run=args.dry_run,
        force=args.force,
        prune=not args.no_prune,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    cfg = load_config()
    tracer = TraceLogger(cfg.logs_dir)
    rec = tracer.find_trace(args.trace_id)
    if rec is None:
        print(json.dumps({"found": False, "trace_id": args.trace_id}))
        return 1
    print(json.dumps(rec, indent=2, ensure_ascii=False))
    return 0


def _cmd_model(args: argparse.Namespace) -> int:
    if args.action == "pull":
        cfg = load_config(config_path=_resolve_config_arg(args.config))
        res = pull_stock_models(
            cfg,
            allow_unverified=bool(args.allow_unverified),
            skip_fastembed=bool(args.skip_fastembed),
        )
        print(json.dumps(res, indent=2))
        return 0
    _, client = _client_from_cfg()
    if args.action == "status":
        print(json.dumps(client.call("model_status"), indent=2))
        return 0
    if args.action == "swap":
        if not args.model_dir:
            raise ConfigError("brain model swap requires --model-dir")
        print(json.dumps(client.call("model_swap", {"model_dir": str(Path(args.model_dir).expanduser().resolve())}), indent=2))
        return 0
    if args.action == "rollback":
        print(json.dumps(client.call("model_rollback"), indent=2))
        return 0
    return 1


def _cmd_graph_export(args: argparse.Namespace) -> int:
    import webbrowser
    from second_brain.observability.graph_export import export_interactive_graph_html
    from second_brain.storage.sqlite_store import SqliteStore

    cfg = load_config()
    if args.rebuild_cross_refs:
        running, _ = is_daemon_running(cfg.pipe_name, cfg.lock_path)
        if running:
            print(
                json.dumps(
                    {
                        "error": "DAEMON_RUNNING",
                        "message": "Stop braind (`brain daemon stop`) before running --rebuild-cross-refs.",
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            return 1

    out_path = Path(args.out).expanduser().resolve() if args.out else (cfg.state_dir / "graph.html")
    store = SqliteStore(cfg.db_path)
    try:
        res = export_interactive_graph_html(
            cfg=cfg,
            store=store,
            output_path=out_path,
            domain_filter=args.domain,
            rebuild_cross_refs=args.rebuild_cross_refs,
        )
    finally:
        store.close()
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.open:
        webbrowser.open(out_path.resolve().as_uri())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="brain", description="2ndBrain CLI shim")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init")
    p_init.add_argument("--vault", default=None)
    p_init.add_argument("--state", default=None)
    p_init.add_argument("--config", default=None)
    p_init.add_argument("--force", action="store_true")
    p_init.add_argument("--sync-manifest", action="store_true")
    p_init.set_defaults(func=_cmd_init)

    p_doc = sub.add_parser("doctor")
    p_doc.add_argument("--config", default=None)
    p_doc.add_argument("--stdio-check", action="store_true")
    p_doc.add_argument("--allow-missing-models", action="store_true")
    p_doc.add_argument("--allow-onedrive", action="store_true")
    p_doc.set_defaults(func=_cmd_doctor)

    p_dmn = sub.add_parser("daemon")
    p_dmn.add_argument("action", choices=["start", "stop", "status"])
    p_dmn.set_defaults(func=_cmd_daemon)

    p_q = sub.add_parser("query")
    p_q.add_argument("text")
    p_q.add_argument("--domain", action="append")
    p_q.add_argument("--version", default=None)
    p_q.add_argument("--source-id", default=None)
    p_q.add_argument("--project-tag", default=None)
    p_q.add_argument("--budget", type=int, default=2500)
    p_q.add_argument("--no-rerank", action="store_true")
    p_q.set_defaults(func=_cmd_query)

    p_grep = sub.add_parser("grep")
    p_grep.add_argument("pattern")
    p_grep.add_argument("--domain", action="append")
    p_grep.add_argument("--source-id", default=None)
    p_grep.add_argument("--version", default=None)
    p_grep.add_argument("--regex", action="store_true")
    p_grep.add_argument("--case-sensitive", action="store_true")
    p_grep.add_argument("--max-matches", type=int, default=10)
    p_grep.add_argument("--budget", type=int, default=2500)
    p_grep.set_defaults(func=_cmd_grep)

    p_rec = sub.add_parser("recall")
    p_rec.add_argument("--query", default=None)
    p_rec.add_argument("--domain", action="append")
    p_rec.add_argument("--project-tag", default=None)
    p_rec.add_argument("--ledger", choices=["preferences", "decisions", "facts"], default=None)
    p_rec.add_argument("--version", default=None)
    p_rec.add_argument("--status", choices=["active", "staged", "superseded", "expired", "rejected", "all"], default="active")
    p_rec.add_argument("--limit", type=int, default=15)
    p_rec.set_defaults(func=_cmd_recall)

    p_emb = sub.add_parser("embed")
    p_emb.add_argument("--batch-size", type=int, default=16)
    p_emb.add_argument("--max-chunks", type=int, default=None)
    p_emb.set_defaults(func=_cmd_embed)

    p_ing = sub.add_parser("ingest")
    p_ing.add_argument("kind", choices=["pdf", "docs"])
    p_ing.add_argument("path")
    p_ing.add_argument("--domain", required=True)
    p_ing.add_argument("--source-id", required=True)
    p_ing.add_argument("--version", required=True)
    p_ing.add_argument("--lexical-only", action="store_true")
    p_ing.add_argument("--docling-tables", action="store_true")
    p_ing.set_defaults(func=_cmd_ingest)

    p_cert = sub.add_parser("certify")
    p_cert.add_argument("--id", dest="item_id", default=None)
    p_cert.add_argument("--decision", choices=["approve", "reject"], default="approve")
    p_cert.add_argument("--domain", default=None)
    p_cert.add_argument("--content", default=None)
    p_cert.add_argument("--project-tag", default=None)
    p_cert.set_defaults(func=_cmd_certify)

    p_imp = sub.add_parser("import-edges")
    p_imp.add_argument("action", choices=["stage", "apply"])
    p_imp.add_argument("input_path")
    p_imp.add_argument("--out", default="review_edges.yaml")
    p_imp.add_argument("--min-confidence", type=float, default=0.80)
    p_imp.add_argument("--cross-doc-only", action="store_true")
    p_imp.add_argument("--replace-inferred", action="store_true")
    p_imp.add_argument("--dry-run", action="store_true")
    p_imp.set_defaults(func=_cmd_import_edges)

    p_comp = sub.add_parser("compile")
    p_comp.add_argument("--domain", required=True)
    p_comp.add_argument("--slug", required=True)
    p_comp.add_argument("--title", required=True)
    p_comp.add_argument("--summary", default="")
    p_comp.add_argument("--summary-file", default=None)
    p_comp.add_argument("--cite", action="append", required=True)
    p_comp.add_argument("--version", default="compiled")
    p_comp.set_defaults(func=_cmd_compile)

    p_exp = sub.add_parser("export-skills")
    p_exp.add_argument("--harness", choices=["antigravity"], default="antigravity")
    p_exp.add_argument("--config", default=None)
    p_exp.add_argument("--output-dir", default=None)
    p_exp.add_argument("--dry-run", action="store_true")
    p_exp.add_argument("--force", action="store_true")
    p_exp.add_argument("--no-prune", action="store_true")
    p_exp.set_defaults(func=_cmd_export_skills)

    p_ex = sub.add_parser("explain")
    p_ex.add_argument("trace_id")
    p_ex.set_defaults(func=_cmd_explain)

    p_mod = sub.add_parser("model")
    p_mod.add_argument("action", choices=["status", "swap", "rollback", "pull"])
    p_mod.add_argument("--config", default=None)
    p_mod.add_argument("--model-dir", default=None)
    p_mod.add_argument("--skip-fastembed", action="store_true")
    p_mod.add_argument("--allow-unverified", action="store_true")
    p_mod.set_defaults(func=_cmd_model)

    p_g = sub.add_parser("graph-export")
    p_g.add_argument("--out", default=None)
    p_g.add_argument("--domain", default=None)
    p_g.add_argument("--rebuild-cross-refs", action="store_true")
    p_g.add_argument("--open", action="store_true")
    p_g.set_defaults(func=_cmd_graph_export)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ConfigError, LayaManifestError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
