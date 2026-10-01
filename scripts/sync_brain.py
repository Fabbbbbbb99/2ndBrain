"""
sync_brain.py - Master Orchestrator for 2ndBrain

Binds together:
1. Laya (System 1 Fast Reflex & Query Router)
2. Code-Review-Graph (AST Call Graph & Blast Radius)
3. Graphify (Semantic GraphRAG & Community Detection)
4. Obsidian (Markdown Vault with Bidirectional Memory)
5. Hybrid Search (Parent-Child Hierarchical Dense Vector + BM25 RRF)
6. Conversational Memory Distillation & Consolidation ("Sleep Cycle")
"""

import os
import sys
import re
import subprocess
import argparse
import shutil
from pathlib import Path

# Ensure current script directory is in sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from laya_router import router
    from memory_distiller import ConversationalMemoryDistiller
    from vault_linter import VaultLinter
    from hybrid_search import HybridSearchEngine
except ImportError:
    from .laya_router import router
    from .memory_distiller import ConversationalMemoryDistiller
    from .vault_linter import VaultLinter
    from .hybrid_search import HybridSearchEngine


def get_default_vault() -> str:
    """Dynamically resolves the vault directory across any OS, repository location, or environment."""
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

    canonical = Path("C:/Users/Fabian/Desktop/Y2T1/RSE2802 Concept Defintion/Second Brain/vault_template")
    if canonical.exists():
        return str(canonical)

    return str(local_template)

DEFAULT_VAULT = get_default_vault()


def resolve_binary(binary_name: str) -> str:
    """Finds binary on PATH or in active python environment (Windows Scripts/ or POSIX bin/)."""
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


def run_command(cmd, cwd=None):
    """Executes a shell command cross-platform and streams output."""
    if cmd and isinstance(cmd, list):
        cmd[0] = resolve_binary(cmd[0])
        cmd_str = subprocess.list2cmdline(cmd) if sys.platform == "win32" else " ".join(f'"{c}"' if " " in c else c for c in cmd)
    else:
        cmd_str = cmd

    print(f"[2ndBrain] Running: {cmd_str}")
    result = subprocess.run(cmd_str, cwd=cwd, shell=True, capture_output=True, text=True, encoding="utf-8")
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0 and "graph file not found" not in result.stderr:
        print(f"[Error] {result.stderr}", file=sys.stderr)
    return result


def sync_code_ast(target_dir: str):
    """Runs code-review-graph build or update."""
    print(f"\n--- [1/3] Updating Structural Code Graph (AST) ---")
    crg_bin = resolve_binary("code-review-graph")
    if shutil.which(crg_bin) is None and not Path(crg_bin).exists():
        print("[2ndBrain] Note: code-review-graph is not installed on PATH.")
        print("[2ndBrain] Install with: pip install code-review-graph (skipping AST indexing).")
        return
    crg_cmd = ["code-review-graph", "update"]
    res = run_command(crg_cmd, cwd=target_dir)
    if res.returncode != 0:
        print("[2ndBrain] Tip: code-review-graph build will be initialized on first run.")
        run_command(["code-review-graph", "build"], cwd=target_dir)


def sync_semantic_graph(target_dir: str, vault_dir: str, update_only: bool = True):
    """Runs graphify with obsidian export."""
    print(f"\n--- [2/3] Updating Semantic Knowledge Graph (Graphify) ---")
    cmd = ["graphify", str(Path(target_dir).resolve()), "--code-only", "--obsidian", "--obsidian-dir", str(Path(vault_dir).resolve())]
    if update_only:
        cmd.append("--update")
    run_command(cmd)


def sync_vector_embeddings(vault_dir: str, force: bool = False):
    """Indexes and updates local Parent-Child hierarchical vector embeddings via fastembed."""
    print(f"\n--- [3/3] Updating Hierarchical Vector Embeddings (fastembed) ---")
    try:
        engine = HybridSearchEngine(vault_dir)
        count = engine.index_vault(force=force)
        print(f"[2ndBrain] Successfully indexed {count} new/modified notes into Parent-Child vector database.")
    except Exception as e:
        print(f"[2ndBrain] Note on embeddings: {e}")


def route_and_query(query_text: str, vault_dir: str, target_dir: str):
    """Evaluates query with Laya System 1 and routes to the best engine."""
    print(f"\n[2ndBrain] Evaluating query through Laya (System 1)...")
    decision = router.route_query(query_text)
    print(f" -> Routed to: {decision['decision']} (Confidence: {decision['confidence']})")
    print(f" -> Rationale: {decision['reasoning']}\n")

    # Check for active memories to inject
    distiller = ConversationalMemoryDistiller(vault_dir)
    memories = distiller.recall_relevant_memories(query_text)
    if memories:
        print("[2ndBrain] Recalled Active Memories:")
        for m in memories:
            print(f"   {m}")
        print()

    engine = decision["decision"]

    def search_vault(v_dir: str, q_text: str):
        # 1. Try Hierarchical Parent-Child Search (Dense Vectors + BM25)
        try:
            h_engine = HybridSearchEngine(v_dir)
            matches = h_engine.search(q_text, top_k=4, return_parent_context=True)
            if matches:
                print(f"\n[Hierarchical Knowledge Matches ({len(matches)} sections found via Parent-Child Search)]:")
                for m in matches:
                    sec = m.get("section_title", "General Section")
                    print(f"- **[[{m['stem']}]]** > *{sec}* (Semantic: {m['semantic_similarity']}%, RRF: {m['rrf_score']}):\n  {m['snippet']}...\n")
                return
        except Exception:
            pass

        # 2. Fallback to basic text scan if vector engine unavailable
        v = Path(v_dir)
        scored = []
        tokens = [t.lower() for t in re.findall(r"\w+", q_text) if len(t) > 3]
        for f in v.glob("**/*.md"):
            try:
                txt = f.read_text(encoding="utf-8", errors="ignore")
                score = sum(txt.lower().count(tok) for tok in tokens)
                if score > 0:
                    scored.append((score, f, txt))
            except Exception:
                continue
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            print("No matching notes found in vault.")
            return
        print(f"\n[Vault Keyword Matches ({len(scored)} notes found)]:")
        for cnt, f, text in scored[:4]:
            clean = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL).strip()
            lines = [l.strip() for l in clean.splitlines() if l.strip() and not l.startswith("#")]
            snippet = " ".join(lines[:3])[:300]
            print(f"- **[[{f.stem}]]** ({f.parent.name}):\n  {snippet}...\n")

    if engine == "CRG_AST_CALL_GRAPH":
        print(f"[Executing AST Query via code-review-graph]:")
        crg_bin = resolve_binary("code-review-graph")
        if shutil.which(crg_bin) is None and not Path(crg_bin).exists():
            print("[2ndBrain] Note: code-review-graph is not installed (pip install code-review-graph).")
            print("[2ndBrain] Falling back to Obsidian Vault Search...")
            search_vault(vault_dir, query_text)
            return

        q_lower = query_text.lower()
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", query_text)
        if quoted:
            symbol = quoted[0]
        else:
            match = re.search(r"\b(?:module|class|function|method|symbol|def)\s+([a-zA-Z_][a-zA-Z0-9_]*)", query_text, re.IGNORECASE)
            if match:
                symbol = match.group(1)
            else:
                words = [w.strip("?\"'.,") for w in query_text.split() if w.strip("?\"'.,") and w.lower() not in ("it", "this", "that", "them", "who", "what", "which", "how")]
                symbol = words[-1] if words else "."

        pattern = "callers_of"
        if "who calls" in q_lower or "callers" in q_lower:
            pattern = "callers_of"
        elif "callee" in q_lower or "calls" in q_lower:
            pattern = "callees_of"
        elif "test" in q_lower:
            pattern = "tests_for"
        elif "import" in q_lower:
            pattern = "importers_of" if "who" in q_lower else "imports_of"
        elif "inherit" in q_lower:
            pattern = "inheritors_of"
        elif "summary" in q_lower:
            pattern = "file_summary"

        print(f" -> Pattern: {pattern}, Target: {symbol}")
        res = run_command(["code-review-graph", "query", pattern, symbol], cwd=target_dir)
        if res.returncode != 0:
            print("[AST Query returned non-zero - Falling back to Obsidian Vault Search]")
            search_vault(vault_dir, query_text)
    elif engine == "OBSIDIAN_VAULT_SEARCH":
        search_vault(vault_dir, query_text)
    else:
        # Default to Graphify query, falling back to Vault Search if graph is absent
        print(f"[Executing Semantic GraphRAG Query via Graphify]:")
        res = run_command(["graphify", "query", f"\"{query_text}\""], cwd=target_dir)
        if res.returncode != 0 or not res.stdout or "graph file not found" in res.stdout:
            print("[Graphify index not built yet - Falling back to Obsidian Vault Search]")
            search_vault(vault_dir, query_text)


def main():
    parser = argparse.ArgumentParser(description="2ndBrain - Dynamic Local Second Brain Orchestrator")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand: sync
    sync_p = subparsers.add_parser("sync", help="Synchronize AST, Semantic Graph, and Obsidian Vault")
    sync_p.add_argument("--target", default=".", help="Target code/doc directory to index")
    sync_p.add_argument("--vault", default=DEFAULT_VAULT, help=f"Path to Obsidian vault (default: {DEFAULT_VAULT})")
    sync_p.add_argument("--full", action="store_true", help="Force full rebuild instead of incremental update")

    # Subcommand: query
    query_p = subparsers.add_parser("query", help="Route and execute a query across the Second Brain")
    query_p.add_argument("text", help="Question or prompt to answer")
    query_p.add_argument("--vault", default=DEFAULT_VAULT, help=f"Path to Obsidian vault (default: {DEFAULT_VAULT})")
    query_p.add_argument("--target", default=".", help="Target codebase directory")

    # Subcommand: distill
    distill_p = subparsers.add_parser("distill", help="Distill conversational transcript into durable memory notes")
    distill_p.add_argument("--transcript", required=True, help="Path to transcript.jsonl")
    distill_p.add_argument("--vault", default=DEFAULT_VAULT, help=f"Path to Obsidian vault (default: {DEFAULT_VAULT})")

    # Subcommand: session (Automated Session Dialogue Archival)
    session_p = subparsers.add_parser("session", help="Archive a conversation dialogue or meeting into persistent memory")
    session_p.add_argument("--topic", required=True, help="Topic or title of the session")
    session_p.add_argument("--summary", required=True, help="Executive summary of key discussions")
    session_p.add_argument("--decisions", nargs="*", default=[], help="Key takeaways or decisions reached")
    session_p.add_argument("--vault", default=DEFAULT_VAULT, help=f"Path to Obsidian vault (default: {DEFAULT_VAULT})")

    # Subcommand: consolidate (Periodic Memory Consolidation / Sleep Cycle)
    cons_p = subparsers.add_parser("consolidate", help="Run Memory Consolidation sleep cycle (prune, merge, promote)")
    cons_p.add_argument("--dry-run", action="store_true", help="Preview consolidation actions without modifying vault")
    cons_p.add_argument("--vault", default=DEFAULT_VAULT, help=f"Path to Obsidian vault (default: {DEFAULT_VAULT})")

    # Subcommand: ingest (Universal Knowledge Ingestion)
    ingest_p = subparsers.add_parser("ingest", help="Ingest external materials (PDFs, docs) into Obsidian vault")
    ingest_p.add_argument("--source", required=True, help="Directory containing documents to ingest")
    ingest_p.add_argument("--vault", default=DEFAULT_VAULT, help=f"Path to Obsidian vault (default: {DEFAULT_VAULT})")

    # Subcommand: lint (Vault Health & Integrity Linter)
    lint_p = subparsers.add_parser("lint", help="Audit vault for broken links, island notes, and schema compliance")
    lint_p.add_argument("--vault", default=DEFAULT_VAULT, help=f"Path to Obsidian vault (default: {DEFAULT_VAULT})")
    lint_p.add_argument("--fix", action="store_true", help="Auto-repair high-confidence broken links and register orphans")

    # Subcommand: blast (AST Blast Radius)
    blast_p = subparsers.add_parser("blast", help="Compute AST blast radius and callers for a symbol or file")
    blast_p.add_argument("symbol", nargs="?", default="", help="Function, class, or method name to trace")
    blast_p.add_argument("--file", help="File to trace blast radius for")
    blast_p.add_argument("--target", default=".", help="Target repository directory")

    # Subcommand: remember (Explicit Rule Crystallization)
    remember_p = subparsers.add_parser("remember", help="Explicitly crystallize a learned rule or decision")
    remember_p.add_argument("rule", help="Rule or decision statement")
    remember_p.add_argument("--category", choices=["task", "decision", "fact", "skill"], default=None, help="Memory pillar: task, decision, fact, or skill")
    remember_p.add_argument("--tag", default="general", help="Category tag (e.g. security, architecture)")
    remember_p.add_argument("--vault", default=DEFAULT_VAULT, help=f"Path to Obsidian vault (default: {DEFAULT_VAULT})")

    args = parser.parse_args()

    if args.command == "sync":
        sync_code_ast(args.target)
        sync_semantic_graph(args.target, args.vault, update_only=not args.full)
        sync_vector_embeddings(args.vault, force=args.full)
        print("\n[2ndBrain] Sync complete! Your Second Brain vault is up to date.")
    elif args.command == "query":
        route_and_query(args.text, args.vault, args.target)
    elif args.command == "distill":
        distiller = ConversationalMemoryDistiller(args.vault)
        count = distiller.process_transcript_jsonl(args.transcript)
        print(f"\n[2ndBrain] Distillation complete! Crystallized {count} durable memory notes.")
    elif args.command == "session":
        distiller = ConversationalMemoryDistiller(args.vault)
        saved = distiller.archive_session(
            topic=args.topic,
            summary=args.summary,
            key_decisions=args.decisions
        )
        print(f"\n[2ndBrain] Session archived successfully: {saved.name}")
    elif args.command == "consolidate":
        distiller = ConversationalMemoryDistiller(args.vault)
        result = distiller.consolidate_memories(dry_run=args.dry_run)
        mode_str = "PREVIEW (Dry Run)" if result["dry_run"] else "APPLIED"
        print(f"\n--- [2ndBrain] Memory Consolidation ({mode_str}) ---")
        print(f"  • Archived (Superseded & Decayed): {result['archived_count']}")
        print(f"  • Promoted to Architecture Guidelines: {result['promoted_count']}")
        print(f"  • Merged Redundancies: {result['merged_count']}")
        print(f"  • Active Memory Rules Remaining: {result['active_count']}")
        print(f"  • Report: {result['report_path']}")
    elif args.command == "ingest":
        try:
            from ingest_knowledge import UniversalKnowledgeIngester
        except ImportError:
            from .ingest_knowledge import UniversalKnowledgeIngester
        ingester = UniversalKnowledgeIngester(args.vault)
        ingester.ingest_directory(Path(args.source))
    elif args.command == "lint":
        linter = VaultLinter(args.vault)
        report = linter.audit(auto_fix=args.fix)
        print(linter.format_report(report))
    elif args.command == "blast":
        target_sym = args.symbol or (Path(args.file).stem if args.file else ".")
        crg_bin = resolve_binary("code-review-graph")
        if shutil.which(crg_bin) is None and not Path(crg_bin).exists():
            print("[2ndBrain] Note: code-review-graph is not installed.")
            print("[2ndBrain] Install with: pip install code-review-graph")
        else:
            print(f"\n[2ndBrain] Computing AST blast radius for '{target_sym}'...")
            run_command(["code-review-graph", "query", "callers_of", target_sym], cwd=args.target)
    elif args.command == "remember":
        distiller = ConversationalMemoryDistiller(args.vault)
        cat = args.category or (args.tag if args.tag in ("task", "decision", "fact", "skill") else None)
        saved = distiller.distill_turn(args.rule, f"Explicitly remembered via CLI with tag: {args.tag}", category=cat)
        if saved:
            print(f"[2ndBrain] Successfully crystallized rule into Obsidian: {saved.name}")
        else:
            print(f"[2ndBrain] Rule was evaluated as duplicate or transient.")
    else:
        parser.print_help()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
