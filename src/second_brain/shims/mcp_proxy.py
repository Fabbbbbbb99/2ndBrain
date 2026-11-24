"""Thin stateless MCP stdio proxy enforcing FD-1 isolation and LF-only JSON-RPC framing."""
from __future__ import annotations

import os
import sys

# CRITICAL FD-1 GUARD: execute BEFORE importing native modules (onnxruntime, pypdfium2, huggingface_hub)
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import io
from contextlib import suppress
from typing import Any

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from second_brain.config import load_config
from second_brain.daemon.ipc import NamedPipeIpcClient

if sys.platform == "win32":
    with suppress(ImportError):
        from mcp.os.win32.utilities import rebind_std_handle_to_fd
else:
    def rebind_std_handle_to_fd(_fd: int) -> None:
        return None


def _claim_clean_stdio() -> tuple[io.TextIOWrapper, io.TextIOWrapper, callable]:
    """Claim FD 0/1 exclusively for UTF-8 LF JSON-RPC framing and divert process stdout to stderr."""
    raw_in_fd = os.dup(0)
    raw_out_fd = os.dup(1)
    null_fd = os.open(os.devnull, os.O_RDONLY)
    try:
        err_dup = os.dup(2)
    except OSError:
        err_dup = os.open(os.devnull, os.O_WRONLY)

    os.dup2(null_fd, 0)
    os.dup2(err_dup, 1)
    os.close(null_fd)
    os.close(err_dup)
    if sys.platform == "win32":
        with suppress(OSError):
            rebind_std_handle_to_fd(0)
            rebind_std_handle_to_fd(1)

    stdin_bin = os.fdopen(raw_in_fd, "rb", buffering=0, closefd=False)
    stdout_bin = os.fdopen(raw_out_fd, "wb", buffering=0, closefd=False)
    stdin_text = io.TextIOWrapper(stdin_bin, encoding="utf-8", errors="replace", newline=None)
    stdout_text = io.TextIOWrapper(stdout_bin, encoding="utf-8", newline="\n")

    def restore() -> None:
        with suppress(Exception):
            stdout_text.flush()
        with suppress(OSError):
            os.dup2(raw_in_fd, 0)
            os.dup2(raw_out_fd, 1)
            if sys.platform == "win32":
                rebind_std_handle_to_fd(0)
                rebind_std_handle_to_fd(1)
            os.close(raw_in_fd)
            os.close(raw_out_fd)

    return stdin_text, stdout_text, restore


def _build_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="brain_route",
            description="Read-only System 1 domain routing and multi-domain specialist Fan-Out Routing Plan.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "domain_scope": {"type": "array", "items": {"type": "string"}},
                    "source_version": {"type": "string"},
                    "parent_trace_id": {"type": "string"},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="brain_query",
            description="Read-only domain/version-partitioned Grounding Pack query (<= 2,500 tokens, verbatim citations + relevant Store 2 memories).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "domain_scope": {"type": "array", "items": {"type": "string"}},
                    "source_version": {"type": "string"},
                    "source_id": {"type": "string"},
                    "project_tag": {"type": "string"},
                    "token_budget": {"type": "integer", "minimum": 256, "maximum": 8000},
                    "no_rerank": {"type": "boolean"},
                    "parent_trace_id": {"type": "string"},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="brain_grep",
            description="Read-only exact literal or ReDoS-guarded regex search over parsed/ and wiki/ chunks (use for exact clause IDs, symbols, or phrases).",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "domain_scope": {"type": "array", "items": {"type": "string"}},
                    "source_id": {"type": "string"},
                    "source_version": {"type": "string"},
                    "regex": {"type": "boolean"},
                    "case_sensitive": {"type": "boolean"},
                    "max_matches": {"type": "integer", "minimum": 1, "maximum": 25},
                    "token_budget": {"type": "integer", "minimum": 256, "maximum": 8000},
                    "parent_trace_id": {"type": "string"},
                },
                "required": ["pattern"],
            },
        ),
        types.Tool(
            name="brain_get_clause",
            description="Read-only O(1) verbatim clause/chunk lookup by typed provenance URI with outbound neighbors, children, cited_by backlinks, and inferred_links.",
            inputSchema={
                "type": "object",
                "properties": {
                    "uri": {"type": "string"},
                    "include_text": {"type": "boolean"},
                    "neighbor_limit": {"type": "integer", "minimum": 0, "maximum": 50},
                },
                "required": ["uri"],
            },
        ),
        types.Tool(
            name="brain_status",
            description="Read-only daemon health, memory RSS, index coverage, staged memory backlog, and store status check.",
            inputSchema={
                "type": "object",
                "properties": {"domain_scope": {"type": "array", "items": {"type": "string"}}},
            },
        ),
        types.Tool(
            name="brain_remember",
            description="Explicit Store 2 memory write routed through the Deterministic Write Gatekeeper.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "domain": {"type": "string"},
                    "ledger_type": {"type": "string", "enum": ["preferences", "decisions", "facts"]},
                    "project_tag": {"type": "string"},
                    "source_ref": {"type": "string"},
                    "explicit_user_save": {"type": "boolean"},
                    "is_domain_claim": {"type": "boolean"},
                    "is_ontology_edit": {"type": "boolean"},
                    "effective_from": {"type": "string"},
                    "effective_to": {"type": "string"},
                    "version_scope": {"type": "string"},
                    "supersedes_id": {"type": "string"},
                },
                "required": ["content"],
            },
        ),
        types.Tool(
            name="brain_recall",
            description="Read-only Store 2 memory search and structured inspection (active, staged, superseded, rejected) by project_tag, ledger_type, and query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "domain_scope": {"type": "array", "items": {"type": "string"}},
                    "project_tag": {"type": "string"},
                    "ledger_type": {"type": "string", "enum": ["preferences", "decisions", "facts"]},
                    "version_scope": {"type": "string"},
                    "status": {"type": "string", "enum": ["active", "staged", "superseded", "expired", "rejected", "all"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    "parent_trace_id": {"type": "string"},
                },
            },
        ),
        types.Tool(
            name="brain_compile",
            description="Explicit CQS writeback creating a Compiled Truth Markdown page citing verified parsed/ URIs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "domain": {"type": "string"},
                    "slug": {"type": "string"},
                    "title": {"type": "string"},
                    "summary_markdown": {"type": "string"},
                    "cited_uris": {"type": "array", "items": {"type": "string"}},
                    "source_version": {"type": "string"},
                },
                "required": ["domain", "slug", "title", "summary_markdown", "cited_uris"],
            },
        ),
        types.Tool(
            name="brain_cache_visual_repair",
            description="Cache multimodal host transcription (LaTeX/Mermaid/Markdown) for an EQG-flagged visual crop.",
            inputSchema={
                "type": "object",
                "properties": {
                    "visual_id": {"type": "string"},
                    "repaired_markdown": {"type": "string"},
                    "certify": {"type": "boolean"},
                },
                "required": ["visual_id", "repaired_markdown"],
            },
        ),
    ]


TOOL_TO_RPC = {
    "brain_route": "route",
    "brain_query": "query",
    "brain_grep": "grep",
    "brain_get_clause": "get_clause",
    "brain_status": "status",
    "brain_remember": "remember",
    "brain_recall": "recall",
    "brain_compile": "compile",
    "brain_cache_visual_repair": "cache_visual_repair",
}


async def run_mcp_stdio() -> None:
    cfg = load_config()
    client = NamedPipeIpcClient(
        pipe_name=cfg.pipe_name,
        vault_path=cfg.vault_path,
        state_dir=cfg.state_dir,
        auto_spawn=True,
    )

    async def on_list_tools(_ctx: Any, _params: types.PaginatedRequestParams | None) -> types.ListToolsResult:
        return types.ListToolsResult(tools=_build_tools())

    async def on_call_tool(_ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        import json

        rpc_method = TOOL_TO_RPC.get(params.name)
        if rpc_method is None:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Unknown tool: {params.name}")],
                isError=True,
            )
        args = dict(params.arguments or {})
        args.setdefault("harness_id", "mcp-stdio")
        try:
            result = await anyio.to_thread.run_sync(client.call, rpc_method, args)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))],
                structuredContent=result if isinstance(result, dict) else {"result": result},
            )
        except Exception as exc:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"{type(exc).__name__}: {exc}")],
                isError=True,
            )

    server = Server(
        "second-brain",
        version="0.1.0",
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )

    stdin_text, stdout_text, restore = _claim_clean_stdio()
    try:
        async with stdio_server(stdin=anyio.wrap_file(stdin_text), stdout=anyio.wrap_file(stdout_text)) as (
            read_stream,
            write_stream,
        ):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    finally:
        restore()


def main() -> None:
    anyio.run(run_mcp_stdio)


if __name__ == "__main__":
    main()
