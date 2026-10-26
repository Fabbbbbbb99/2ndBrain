"""
mcp_server.py - Universal Model Context Protocol (MCP) Server for 2ndBrain

Allows ANY MCP-compatible LLM (Claude Code, Claude Desktop, Cursor, Windsurf,
Antigravity, Zed, Continue.dev) to dynamically query, review, recall, and
crystallize memories in the Second Brain.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    try:
        from fastmcp import FastMCP
    except ImportError:
        print(
            "[Error] Neither 'mcp' nor 'fastmcp' is installed in the active Python environment.",
            file=sys.stderr,
        )
        print("[Error] Install using: pip install mcp", file=sys.stderr)
        sys.exit(1)

from laya_router import router
from memory_distiller import ConversationalMemoryDistiller

mcp = FastMCP("2ndBrain-Universal-Server")


def resolve_binary(binary_name: str) -> str:
    """Finds binary on PATH or in active python environment."""
    found = shutil.which(binary_name)
    if found:
        return found
    scripts_dir = Path(sys.prefix) / "Scripts"
    candidate_win = scripts_dir / f"{binary_name}.exe"
    if candidate_win.exists():
        return str(candidate_win)
    bin_dir = Path(sys.prefix) / "bin"
    candidate_posix = bin_dir / binary_name
    if candidate_posix.exists():
        return str(candidate_posix)
    return binary_name


def get_default_vault() -> str:
    """Dynamically resolves the vault directory across any OS or clone location."""
    env_root = os.environ.get("KNOWLEDGE_BASE_ROOT")
    if env_root and Path(env_root).exists():
        return str((Path(env_root) / "vault").resolve())

    env_vault = os.environ.get("SECOND_BRAIN_VAULT")
    if env_vault and Path(env_vault).exists():
        return str(Path(env_vault).resolve())

    local_template = current_dir.parent / "vault"
    if not local_template.exists():
        local_template = current_dir.parent / "vault_template"
    if local_template.exists():
        return str(local_template.resolve())

    cwd = Path.cwd()
    for candidate in [
        cwd / "vault",
        cwd / "vault_template",
        cwd / "Second Brain" / "vault",
        cwd / "Second Brain" / "vault_template",
    ]:
        if candidate.exists():
            return str(candidate.resolve())

    home_vault = Path.home() / ".se_brain" / "vault"
    if home_vault.exists():
        return str(home_vault.resolve())

    return str(local_template.resolve())


def get_default_target() -> str:
    """Dynamically resolves the repository target root across any OS or working directory."""
    for env_var in ("SECOND_BRAIN_TARGET", "KNOWLEDGE_BASE_ROOT"):
        val = os.environ.get(env_var)
        if val and Path(val).exists():
            return str(Path(val).resolve())
    return str(current_dir.parent.resolve())


def _resolve_crg_repo(target_dir: str) -> str:
    """Resolves the CRG repo root, preferring source_materials when present."""
    t_path = Path(target_dir).resolve()
    sm_path = t_path / "source_materials"
    if sm_path.exists() and any(sm_path.iterdir()):
        return str(sm_path)
    return str(t_path)


DEFAULT_VAULT = get_default_vault()
DEFAULT_TARGET = get_default_target()


def get_distiller(vault_path: Optional[str] = None) -> ConversationalMemoryDistiller:
    vault = vault_path or DEFAULT_VAULT
    return ConversationalMemoryDistiller(vault)


@mcp.tool()
def brain_query(question: str, vault_path: Optional[str] = None, target_dir: Optional[str] = None) -> str:
    """
    Intelligently answers a question using 2ndBrain.
    Routes via Laya (System 1) to AST Code Graph, Semantic GraphRAG, or Obsidian Vault,
    performs Parent-Child hierarchical search with rich context, and prepends active learned rules.
    """
    vault = vault_path or DEFAULT_VAULT
    target = target_dir or DEFAULT_TARGET

    build_dir = Path(vault).resolve().parent / "build"
    os.environ["CRG_DATA_DIR"] = str((build_dir / "ast").resolve())
    crg_repo = _resolve_crg_repo(target)

    decision = router.route_query(question)
    engine = decision["decision"]
    confidence = decision["confidence"]

    distiller = get_distiller(vault)
    memories = distiller.recall_relevant_memories(question)
    memory_prefix = ""
    if memories:
        memory_prefix = "[Active Learned Memories]:\n" + "\n".join(memories) + "\n\n"

    def search_vault(vault_p: str, q: str) -> str:
        try:
            from hybrid_search import HybridSearchEngine

            h_engine = HybridSearchEngine(vault_p)
            matches = h_engine.search(q, top_k=4, return_parent_context=True)
            if matches:
                out = []
                for m in matches:
                    sec = m.get("section_title", "General")
                    out.append(
                        f"- **[[{m['stem']}]]** > *{sec}* "
                        f"(Semantic: {m['semantic_similarity']}%, RRF: {m['rrf_score']}):\n"
                        f"  {m['snippet']}..."
                    )
                return "\n\n".join(out)
        except Exception:
            pass

        q_words = [w.lower() for w in re.findall(r"\w{3,}", q)]
        scored = []
        for md_file in Path(vault_p).rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8", errors="ignore")
                cnt = sum(content.lower().count(w) for w in q_words)
                if cnt > 0:
                    scored.append((cnt, md_file, content))
            except Exception:
                continue
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return "No matching notes found in vault."
        out = []
        for cnt, f, text in scored[:3]:
            clean_text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL).strip()
            lines = [l.strip() for l in clean_text.splitlines() if l.strip() and not l.startswith("#")]
            snippet = " ".join(lines[:3])[:300]
            out.append(f"- **[[{f.stem}]]** ({f.parent.name}):\n  {snippet}...")
        return "\n\n".join(out)

    if engine == "CRG_AST_CALL_GRAPH":
        from sync_brain import query_code_ast

        ast_text = query_code_ast(question, build_dir, crg_repo)
        result_text = ast_text or (
            "=== AST Code Graph Notice ===\nNo AST matches found. Falling back to Obsidian Vault Search:\n\n"
            + search_vault(vault, question)
        )
    elif engine == "OBSIDIAN_VAULT_SEARCH":
        result_text = "=== Obsidian Vault Matches ===\n" + search_vault(vault, question)
    else:
        from sync_brain import query_semantic_graph

        result_text = query_semantic_graph(question, build_dir, vault)

    return f"{memory_prefix}[Routed via {engine} (Confidence: {confidence})]\n{result_text}"


@mcp.tool()
def brain_blast_radius(symbol: str, target_dir: Optional[str] = None) -> str:
    """
    Calculates the exact syntactic blast radius and dependents for a function, class,
    or module using Tree-sitter AST via Code-Review-Graph.
    """
    if not symbol or not symbol.strip() or symbol.strip() in (".", "*", "%"):
        return "Please specify a valid function, class, or method name (e.g. computeVelocityCommands)."

    target = target_dir or DEFAULT_TARGET
    build_dir = Path(get_default_vault()).parent / "build"
    os.environ["CRG_DATA_DIR"] = str((build_dir / "ast").resolve())
    crg_repo = _resolve_crg_repo(target)

    from sync_brain import query_code_ast

    return query_code_ast(symbol, build_dir, crg_repo, pattern="callers_of") or (
        f"No blast radius found for {symbol}."
    )


@mcp.tool()
def brain_remember(
    rule_or_decision: str, context: str = "", category: str = "fact", vault_path: Optional[str] = None
) -> str:
    """
    Permanently crystallizes a memory into the Obsidian second brain across 4 structured pillars:
    - category='task': Actionable roadmap milestone or task item in 03-Agent-Memory/01-Tasks/
    - category='decision': Architectural Decision Record (ADR) in 03-Agent-Memory/02-Decisions/
    - category='fact': Permanent system invariant or project truth in 03-Agent-Memory/03-Facts/
    - category='skill': Reusable execution recipe or debugging workflow in 03-Agent-Memory/04-Skills/
    """
    vault = vault_path or DEFAULT_VAULT
    distiller = get_distiller(vault)
    saved_path = distiller.distill_turn(
        rule_or_decision, context or "User explicit instruction", category=category
    )
    if saved_path:
        return f"Successfully crystallized memory note [{category.upper()}]: {saved_path.name}"
    return "Memory was evaluated as transient or duplicate by Laya."


@mcp.tool()
def brain_remember_task(
    title: str,
    details: str,
    status: str = "pending",
    priority: str = "medium",
    milestone: str = "",
    vault_path: Optional[str] = None,
) -> str:
    """Explicitly creates an actionable task or roadmap milestone under 03-Agent-Memory/01-Tasks/."""
    vault = vault_path or DEFAULT_VAULT
    distiller = get_distiller(vault)
    saved = distiller.remember_task(
        title=title, details=details, status=status, priority=priority, milestone=milestone
    )
    return f"Successfully saved task to 01-Tasks: {saved.name}"


@mcp.tool()
def brain_remember_decision(
    title: str,
    rationale: str,
    alternatives: Optional[List[str]] = None,
    impact: str = "",
    vault_path: Optional[str] = None,
) -> str:
    """Explicitly records an Architectural Decision Record (ADR) under 03-Agent-Memory/02-Decisions/."""
    vault = vault_path or DEFAULT_VAULT
    distiller = get_distiller(vault)
    saved = distiller.remember_decision(
        title=title, rationale=rationale, alternatives=alternatives, impact=impact
    )
    return f"Successfully saved decision ADR to 02-Decisions: {saved.name}"


@mcp.tool()
def brain_remember_fact(
    statement: str,
    source: str = "project_grounding",
    domain: str = "robotics",
    vault_path: Optional[str] = None,
) -> str:
    """Explicitly records an invariant system truth or hardware specification under 03-Agent-Memory/03-Facts/."""
    vault = vault_path or DEFAULT_VAULT
    distiller = get_distiller(vault)
    saved = distiller.remember_fact(statement=statement, source=source, domain=domain)
    return f"Successfully saved system fact to 03-Facts: {saved.name}"


@mcp.tool()
def brain_remember_skill(
    skill_name: str, steps: List[str], trigger: str = "", vault_path: Optional[str] = None
) -> str:
    """Explicitly records a learned execution recipe or troubleshooting procedure under 03-Agent-Memory/04-Skills/."""
    vault = vault_path or DEFAULT_VAULT
    distiller = get_distiller(vault)
    saved = distiller.remember_skill(skill_name=skill_name, steps=steps, trigger=trigger)
    return f"Successfully saved execution skill to 04-Skills: {saved.name}"


@mcp.tool()
def brain_recall(topic: str, vault_path: Optional[str] = None) -> str:
    """Recalls all active rules, user preferences, and architectural decisions relevant to a topic."""
    vault = vault_path or DEFAULT_VAULT
    distiller = get_distiller(vault)
    memories = distiller.recall_relevant_memories(topic)
    if not memories:
        return f"No active memories found regarding '{topic}'."
    return "=== Active Second Brain Memories ===\n" + "\n".join(memories)


@mcp.tool()
def brain_archive_session(
    topic: str,
    summary: str,
    key_decisions: Optional[List[str]] = None,
    dialogue_snippet: Optional[str] = None,
    vault_path: Optional[str] = None,
) -> str:
    """Compacts and permanently archives a conversation dialogue into 03-Agent-Memory/Sessions/."""
    vault = vault_path or DEFAULT_VAULT
    distiller = get_distiller(vault)
    dialogue_list = [{"role": "Dialogue", "content": dialogue_snippet}] if dialogue_snippet else None
    saved = distiller.archive_session(
        topic=topic, summary=summary, dialogue=dialogue_list, key_decisions=key_decisions
    )
    return f"Successfully archived session into persistent memory: {saved.name}"


@mcp.tool()
def brain_consolidate_memory(dry_run: bool = False, vault_path: Optional[str] = None) -> str:
    """Executes the Memory Consolidation 'Sleep Cycle'."""
    vault = vault_path or DEFAULT_VAULT
    distiller = get_distiller(vault)
    result = distiller.consolidate_memories(dry_run=dry_run)
    mode_str = "PREVIEW (Dry Run)" if result["dry_run"] else "APPLIED"
    return (
        f"=== Memory Consolidation ({mode_str}) ===\n"
        f"- Archived (Superseded & Decayed): {result['archived_count']}\n"
        f"- Promoted to Architecture Guidelines: {result['promoted_count']}\n"
        f"- Merged Redundancies: {result['merged_count']}\n"
        f"- Active Memory Rules Remaining: {result['active_count']}\n"
        f"- Audit Report Generated: {result['report_path']}"
    )


@mcp.tool()
def brain_sync(target_dir: Optional[str] = None, vault_path: Optional[str] = None) -> str:
    """Synchronizes the AST code graph, Graphify semantic graph, and Hierarchical Vector embeddings."""
    vault = vault_path or DEFAULT_VAULT
    target = target_dir or DEFAULT_TARGET
    sync_script = current_dir / "sync_brain.py"
    res = subprocess.run(
        [sys.executable, str(sync_script), "sync", "--target", target, "--vault", vault],
        cwd=target,
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return res.stdout or res.stderr or "Sync completed."


@mcp.tool()
def brain_ingest(source_dir: str, vault_path: Optional[str] = None) -> str:
    """Ingests external documents (PDFs, markdown, text) from a folder into the Obsidian vault."""
    vault = vault_path or DEFAULT_VAULT
    try:
        from ingest_knowledge import UniversalKnowledgeIngester
    except ImportError:
        from .ingest_knowledge import UniversalKnowledgeIngester
    ingester = UniversalKnowledgeIngester(vault)
    results = ingester.ingest_directory(Path(source_dir))
    return (
        f"Successfully ingested {len(results)} documents into vault at '{vault}' "
        f"with Table and Equation preservation."
    )


@mcp.tool()
def brain_lint(vault_path: Optional[str] = None, auto_fix: bool = False) -> str:
    """Audits the Obsidian second brain for broken [[wikilinks]], island/orphan notes, and schema compliance."""
    vault = vault_path or DEFAULT_VAULT
    try:
        from vault_linter import VaultLinter
    except ImportError:
        from .vault_linter import VaultLinter
    linter = VaultLinter(vault)
    report = linter.audit(auto_fix=auto_fix)
    return linter.format_report(report)


if __name__ == "__main__":
    mcp.run()
