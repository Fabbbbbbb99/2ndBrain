"""
mcp_server.py - Universal Model Context Protocol (MCP) Server for 2ndBrain

Allows ANY MCP-compatible LLM (Claude Code, Claude Desktop, Cursor, Windsurf, 
Antigravity, Zed, Continue.dev) to dynamically query, review, recall, and 
crystallize memories in the Second Brain.

Exposes 9 Universal Brain Tools:
1. brain_query: Intelligent routing with Parent-Child hierarchical search & active memory recall.
2. brain_blast_radius: Deterministic AST code call graph & blast radius evaluation.
3. brain_remember: Crystallizes durable rules, constraints, and user preferences into Obsidian.
4. brain_recall: Recalls active rules with recency decay and reinforcement weighting.
5. brain_archive_session: Compacts and archives an entire conversation dialogue into 04-Agent-Memory/Sessions/.
6. brain_consolidate_memory: Runs the memory consolidation "sleep cycle" (prunes, merges, promotes).
7. brain_sync: Synchronizes AST code graph, semantic graph, and hierarchical vector embeddings.
8. brain_ingest: Ingests documents into the vault with table & equation preservation.
9. brain_lint: Audits vault health (0-100 score) and auto-repairs broken wikilinks.
"""

import os
import sys
import subprocess
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add parent directory to path to import sibling scripts
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    try:
        from fastmcp import FastMCP
    except ImportError:
        print("[Error] Neither 'mcp' nor 'fastmcp' is installed in the active Python environment.", file=sys.stderr)
        print("[Error] Install using: pip install mcp", file=sys.stderr)
        sys.exit(1)

from laya_router import router
import shutil
from memory_distiller import ConversationalMemoryDistiller

# Initialize FastMCP Server
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
    env_vault = os.environ.get("SECOND_BRAIN_VAULT")
    if env_vault and Path(env_vault).exists():
        return env_vault

    local_template = current_dir.parent / "vault_template"
    if local_template.exists():
        return str(local_template)

    cwd = Path.cwd()
    for candidate in [cwd / "vault_template", cwd / "vault", cwd / "Second Brain" / "vault_template"]:
        if candidate.exists():
            return str(candidate.resolve())

    home_vault = Path.home() / ".2ndbrain" / "vault"
    if home_vault.exists():
        return str(home_vault)

    return str(local_template)


DEFAULT_VAULT = get_default_vault()
DEFAULT_TARGET = os.environ.get("SECOND_BRAIN_TARGET", str(Path.cwd() if Path.cwd() != current_dir else current_dir.parent.parent))


def get_distiller(vault_path: Optional[str] = None) -> ConversationalMemoryDistiller:
    vault = vault_path or DEFAULT_VAULT
    return ConversationalMemoryDistiller(vault)


@mcp.tool()
def brain_query(question: str, vault_path: Optional[str] = None, target_dir: Optional[str] = None) -> str:
    """
    Intelligently answers a question using the 2ndBrain.
    Routes via Laya (System 1 <35ms) to AST Code Graph, Semantic GraphRAG, or Obsidian Vault,
    performs Parent-Child hierarchical search with rich context, and prepends active learned rules.
    """
    vault = vault_path or DEFAULT_VAULT
    target = target_dir or DEFAULT_TARGET

    decision = router.route_query(question)
    engine = decision["decision"]
    confidence = decision["confidence"]

    # Active memory recall
    distiller = get_distiller(vault)
    memories = distiller.recall_relevant_memories(question)
    memory_prefix = ""
    if memories:
        memory_prefix = "[Active Learned Memories]:\n" + "\n".join(memories) + "\n\n"

    def search_vault(vault_path: str, q: str) -> str:
        try:
            from hybrid_search import HybridSearchEngine
            h_engine = HybridSearchEngine(vault_path)
            matches = h_engine.search(q, top_k=4, return_parent_context=True)
            if matches:
                out = []
                for m in matches:
                    sec = m.get("section_title", "General")
                    out.append(f"- **[[{m['stem']}]]** > *{sec}* (Semantic: {m['semantic_similarity']}%, RRF: {m['rrf_score']}):\n  {m['snippet']}...")
                return "\n\n".join(out)
        except Exception:
            pass

        q_words = [w.lower() for w in re.findall(r'\w{3,}', q)]
        scored = []
        for md_file in Path(vault_path).rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
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
        words = [w.strip("?\"'.,") for w in question.split() if w.strip("?\"'.,")]
        symbol = words[-1] if words else "."
        crg_bin = resolve_binary("code-review-graph")
        if shutil.which(crg_bin) is None and not Path(crg_bin).exists():
            result_text = f"=== AST Code Graph Notice ===\nNote: 'code-review-graph' is not installed (pip install code-review-graph).\nFalling back to Obsidian Vault Search:\n\n" + search_vault(vault, question)
        else:
            cmd = f'"{crg_bin}" query callers_of {symbol}'
            res = subprocess.run(cmd, cwd=target, shell=True, capture_output=True, text=True, encoding="utf-8")
            result_text = f"=== AST Code Graph Result (callers_of {symbol}) ===\n" + (res.stdout or res.stderr or "No callers found.")
    elif engine == "OBSIDIAN_VAULT_SEARCH":
        result_text = "=== Obsidian Vault Matches ===\n" + search_vault(vault, question)
    else:
        graphify_bin = resolve_binary("graphify")
        cmd = f'"{graphify_bin}" query "{question}"'
        res = subprocess.run(cmd, cwd=target, shell=True, capture_output=True, text=True, encoding="utf-8")
        if res.returncode == 0 and res.stdout and "graph file not found" not in res.stdout:
            result_text = f"=== Graphify Semantic GraphRAG Result ===\n{res.stdout}"
        else:
            result_text = f"=== Obsidian Vault Search (Semantic Fallback) ===\n" + search_vault(vault, question)

    return f"{memory_prefix}[Routed via {engine} (Confidence: {confidence})]\n{result_text}"


@mcp.tool()
def brain_blast_radius(symbol: str, target_dir: Optional[str] = None) -> str:
    """
    Calculates the exact syntactic blast radius and dependents for a function, class,
    or module using Tree-sitter AST via Code-Review-Graph.
    """
    target = target_dir or DEFAULT_TARGET
    crg_bin = resolve_binary("code-review-graph")
    if shutil.which(crg_bin) is None and not Path(crg_bin).exists():
        return f"AST code intelligence requires 'code-review-graph' (run: pip install code-review-graph). Symbol '{symbol}' could not be evaluated via AST."
    cmd = f'"{crg_bin}" query callers_of {symbol}'
    res = subprocess.run(cmd, cwd=target, shell=True, capture_output=True, text=True, encoding="utf-8")
    return res.stdout or res.stderr or f"No blast radius found for {symbol}."


@mcp.tool()
def brain_remember(rule_or_decision: str, context: str = "", category: str = "fact", vault_path: Optional[str] = None) -> str:
    """
    Permanently crystallizes a memory into the Obsidian second brain across 4 structured pillars:
    - category='task': Actionable roadmap milestone or task item in 04-Agent-Memory/01-Tasks/
    - category='decision': Architectural Decision Record (ADR) in 04-Agent-Memory/02-Decisions/
    - category='fact': Permanent system invariant or project truth in 04-Agent-Memory/03-Facts/
    - category='skill': Reusable execution recipe or debugging workflow in 04-Agent-Memory/04-Skills/
    """
    vault = vault_path or DEFAULT_VAULT
    distiller = get_distiller(vault)
    saved_path = distiller.distill_turn(rule_or_decision, context or "User explicit instruction", category=category)
    if saved_path:
        return f"Successfully crystallized memory note [{category.upper()}]: {saved_path.name}"
    return "Memory was evaluated as transient or duplicate by Laya."


@mcp.tool()
def brain_remember_task(title: str, details: str, status: str = "pending", priority: str = "medium", milestone: str = "", vault_path: Optional[str] = None) -> str:
    """
    Explicitly creates an actionable task or roadmap milestone under 04-Agent-Memory/01-Tasks/.
    """
    vault = vault_path or DEFAULT_VAULT
    distiller = get_distiller(vault)
    saved = distiller.remember_task(title=title, details=details, status=status, priority=priority, milestone=milestone)
    return f"Successfully saved task to 01-Tasks: {saved.name}"


@mcp.tool()
def brain_remember_decision(title: str, rationale: str, alternatives: Optional[List[str]] = None, impact: str = "", vault_path: Optional[str] = None) -> str:
    """
    Explicitly records an Architectural Decision Record (ADR) under 04-Agent-Memory/02-Decisions/.
    """
    vault = vault_path or DEFAULT_VAULT
    distiller = get_distiller(vault)
    saved = distiller.remember_decision(title=title, rationale=rationale, alternatives=alternatives, impact=impact)
    return f"Successfully saved decision ADR to 02-Decisions: {saved.name}"


@mcp.tool()
def brain_remember_fact(statement: str, source: str = "project_grounding", domain: str = "robotics", vault_path: Optional[str] = None) -> str:
    """
    Explicitly records an invariant system truth or hardware specification under 04-Agent-Memory/03-Facts/.
    """
    vault = vault_path or DEFAULT_VAULT
    distiller = get_distiller(vault)
    saved = distiller.remember_fact(statement=statement, source=source, domain=domain)
    return f"Successfully saved system fact to 03-Facts: {saved.name}"


@mcp.tool()
def brain_remember_skill(skill_name: str, steps: List[str], trigger: str = "", vault_path: Optional[str] = None) -> str:
    """
    Explicitly records a learned execution recipe or troubleshooting procedure under 04-Agent-Memory/04-Skills/.
    """
    vault = vault_path or DEFAULT_VAULT
    distiller = get_distiller(vault)
    saved = distiller.remember_skill(skill_name=skill_name, steps=steps, trigger=trigger)
    return f"Successfully saved execution skill to 04-Skills: {saved.name}"



@mcp.tool()
def brain_recall(topic: str, vault_path: Optional[str] = None) -> str:
    """
    Recalls all active rules, user preferences, and architectural decisions relevant to a topic.
    """
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
    vault_path: Optional[str] = None
) -> str:
    """
    Compacts and permanently archives a conversation dialogue into 04-Agent-Memory/Sessions/.
    Enables future sessions across any LLM to recall past discussions, questions, and decisions.
    """
    vault = vault_path or DEFAULT_VAULT
    distiller = get_distiller(vault)
    dialogue_list = [{"role": "Dialogue", "content": dialogue_snippet}] if dialogue_snippet else None
    saved = distiller.archive_session(
        topic=topic,
        summary=summary,
        dialogue=dialogue_list,
        key_decisions=key_decisions
    )
    return f"Successfully archived session into persistent memory: {saved.name}"


@mcp.tool()
def brain_consolidate_memory(dry_run: bool = False, vault_path: Optional[str] = None) -> str:
    """
    Executes the Memory Consolidation "Sleep Cycle":
    1. Archives permanently superseded or dead/decayed rules into 04-Agent-Memory/Archive/.
    2. Promotes highly-reinforced rules (count >= 3) to permanent Architectural Guidelines.
    3. Merges near-duplicate rules across active memories.
    4. Produces a consolidation audit report.
    Pass dry_run=True to preview actions without altering files.
    """
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
    """
    Synchronizes the AST code graph, Graphify semantic graph, and Hierarchical Vector embeddings.
    """
    vault = vault_path or DEFAULT_VAULT
    target = target_dir or DEFAULT_TARGET
    sync_script = current_dir / "sync_brain.py"
    cmd = f'"{sys.executable}" "{sync_script}" sync --target "{target}" --vault "{vault}"'
    res = subprocess.run(cmd, cwd=target, shell=True, capture_output=True, text=True, encoding="utf-8")
    return res.stdout or res.stderr or "Sync completed."


@mcp.tool()
def brain_ingest(source_dir: str, vault_path: Optional[str] = None) -> str:
    """
    Ingests external documents (PDFs, markdown, text) from a folder into the Obsidian vault.
    Preserves structured tables as Markdown tables and preserves math equations in LaTeX format.
    Automatically handles digital text extraction and neural OCR for scanned documents.
    """
    vault = vault_path or DEFAULT_VAULT
    try:
        from ingest_knowledge import UniversalKnowledgeIngester
    except ImportError:
        from .ingest_knowledge import UniversalKnowledgeIngester
    ingester = UniversalKnowledgeIngester(vault)
    results = ingester.ingest_directory(Path(source_dir))
    return f"Successfully ingested {len(results)} documents into vault at '{vault}' with Table and Equation preservation."


@mcp.tool()
def brain_lint(vault_path: Optional[str] = None, auto_fix: bool = False) -> str:
    """
    Audits the Obsidian second brain for broken [[wikilinks]], island/orphan notes,
    schema violations, and index completeness. Computes a quantitative health score (0-100).
    Pass auto_fix=True to automatically repair high-confidence broken links and map orphans into Index.md.
    """
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
