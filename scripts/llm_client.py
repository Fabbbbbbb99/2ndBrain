"""
llm_client.py - Universal LLM Adapter for 2ndBrain

Provides standard function calling / tool definitions and memory injection for:
- OpenAI (GPT-4o, GPT-4o-mini, o1, o3, etc.)
- Anthropic (Claude 3.7 Sonnet, Claude 3.5 Sonnet, Claude 3 Opus)
- Google Gemini (Gemini 2.5 Pro, Flash via google-genai)
- Ollama / Local Models (Llama 3, DeepSeek-R1 / V3, Qwen 2.5, Mistral)
- LangChain, AutoGen, CrewAI, and LlamaIndex
"""

import json
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
    from hybrid_search import HybridSearchEngine
    from laya_router import router
    from memory_distiller import ConversationalMemoryDistiller
    from vault_linter import VaultLinter
except ImportError:
    from .hybrid_search import HybridSearchEngine
    from .laya_router import router
    from .memory_distiller import ConversationalMemoryDistiller
    from .vault_linter import VaultLinter


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
    """Dynamically resolves the vault directory across any OS, environment, or clone location."""
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


class UniversalBrainAdapter:
    """
    Universal bridge between 2ndBrain and ANY LLM provider.
    Supports pre-flight prompt enrichment, function calling tools, and post-flight distillation.
    """

    def __init__(
        self,
        vault_dir: Optional[str] = None,
        vault_path: Optional[str] = None,
        target_dir: Optional[str] = None,
    ):
        chosen_vault = vault_dir or vault_path or get_default_vault()
        self.vault_dir = Path(chosen_vault).resolve()
        self.target_dir = Path(target_dir or get_default_target()).resolve()
        self.distiller = ConversationalMemoryDistiller(str(self.vault_dir))

    def enrich_prompt(self, user_prompt: str) -> str:
        """
        Pre-flight memory injection for ANY LLM.
        Scans Obsidian for active, non-superseded rules/decisions and prepends them.
        """
        memories = self.distiller.recall_relevant_memories(user_prompt)
        if not memories:
            return user_prompt

        prefix = "### [2ndBrain Grounding Context]\n"
        prefix += "The following persistent rules, user preferences, and architectural decisions apply to this project:\n"
        for m in memories:
            prefix += f"{m}\n"
        prefix += "\n"
        return prefix + user_prompt

    def distill_turn(
        self,
        user_msg: str,
        assistant_msg: str,
        session_id: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Optional[Path]:
        """Post-flight distillation for ANY LLM."""
        return self.distiller.distill_turn(
            user_msg, assistant_msg, session_id=session_id, category=category
        )

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Standard OpenAI tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "brain_query",
                    "description": "Query the 2ndBrain knowledge engine for code call graphs, architecture, or past meeting decisions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The technical question or search query",
                            }
                        },
                        "required": ["question"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "brain_blast_radius",
                    "description": "Get callers, dependents, and affected code for a symbol using Tree-sitter AST via Code-Review-Graph.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "The function, class, or method name",
                            }
                        },
                        "required": ["symbol"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "brain_remember",
                    "description": "Save a learned rule, constraint, or architectural decision into the Obsidian second brain across the 4 memory pillars.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "rule_text": {
                                "type": "string",
                                "description": "The exact rule or decision statement",
                            },
                            "category": {
                                "type": "string",
                                "enum": ["task", "decision", "fact", "skill"],
                                "default": "fact",
                            },
                        },
                        "required": ["rule_text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "brain_recall",
                    "description": "Recall active rules, user preferences, and architectural decisions relevant to a topic.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string", "description": "The topic or keyword to recall"}
                        },
                        "required": ["topic"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "brain_sync",
                    "description": "Synchronize AST code graph, Graphify semantic graph, and vector embeddings with the Obsidian vault.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "brain_lint",
                    "description": "Audit vault health, broken links, orphan notes, and schema compliance. Pass auto_fix=True to repair.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "auto_fix": {
                                "type": "boolean",
                                "description": "Whether to auto-fix broken links",
                                "default": False,
                            }
                        },
                    },
                },
            },
        ]

    def get_anthropic_tools(self) -> List[Dict[str, Any]]:
        """Anthropic Messages API tool definitions."""
        return [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "input_schema": t["function"]["parameters"],
            }
            for t in self.get_openai_tools()
        ]

    def get_gemini_tools(self) -> List[Dict[str, Any]]:
        """Google Gemini FunctionDeclaration definitions."""
        return [{"function_declarations": [t["function"] for t in self.get_openai_tools()]}]

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Executes any brain tool call requested by ANY LLM model."""
        build_dir = self.vault_dir.parent / "build"
        os.environ["CRG_DATA_DIR"] = str((build_dir / "ast").resolve())
        crg_repo = _resolve_crg_repo(str(self.target_dir))

        if tool_name == "brain_query":
            q = arguments.get("question", "")
            return self._execute_query(q)

        elif tool_name == "brain_blast_radius":
            sym = (arguments.get("symbol", "") or "").strip()
            if not sym or sym in (".", "*", "%"):
                return "Please specify a valid function, class, or method name (e.g. computeVelocityCommands)."
            from sync_brain import query_code_ast

            return query_code_ast(sym, build_dir, crg_repo, pattern="callers_of") or (
                f"No callers found for symbol '{sym}'."
            )

        elif tool_name == "brain_remember":
            rule = arguments.get("rule_text", "")
            cat = arguments.get("category", "fact")
            saved = self.distiller.distill_turn(
                rule, f"Explicit memory update via LLM ({cat})", category=cat
            )
            if saved:
                return f"Successfully crystallized rule into Obsidian: [[{saved.stem}]]"
            return "Memory was evaluated as transient or duplicate by Laya."

        elif tool_name == "brain_recall":
            topic = arguments.get("topic", "")
            memories = self.distiller.recall_relevant_memories(topic)
            if not memories:
                return f"No active memories found regarding '{topic}'."
            return "=== Active Second Brain Memories ===\n" + "\n".join(memories)

        elif tool_name == "brain_sync":
            sync_script = current_dir / "sync_brain.py"
            res = subprocess.run(
                [
                    sys.executable,
                    str(sync_script),
                    "sync",
                    "--target",
                    str(self.target_dir),
                    "--vault",
                    str(self.vault_dir),
                ],
                cwd=self.target_dir,
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            return res.stdout or res.stderr or "Sync completed."

        elif tool_name == "brain_lint":
            auto_fix = arguments.get("auto_fix", False)
            linter = VaultLinter(str(self.vault_dir))
            report = linter.audit(auto_fix=auto_fix)
            return linter.format_report(report)

        return f"Unknown tool: '{tool_name}'"

    def _execute_query(self, question: str) -> str:
        """Internal helper for full brain query with multi-engine routing and fallback."""
        decision = router.route_query(question)
        engine = decision["decision"]
        confidence = decision["confidence"]

        build_dir = self.vault_dir.parent / "build"
        os.environ["CRG_DATA_DIR"] = str((build_dir / "ast").resolve())
        crg_repo = _resolve_crg_repo(str(self.target_dir))

        memories = self.distiller.recall_relevant_memories(question)
        memory_prefix = ""
        if memories:
            memory_prefix = "=== Active Learned Memories ===\n" + "\n".join(memories) + "\n\n"

        def search_vault(v_path: Path, q: str) -> str:
            try:
                h_engine = HybridSearchEngine(str(v_path))
                matches = h_engine.search(q, top_k=4)
                if matches:
                    out = []
                    for m in matches:
                        out.append(
                            f"- **[[{m['stem']}]]** (Semantic: {m['semantic_similarity']}%, RRF: {m['rrf_score']}):\n"
                            f"  {m['snippet']}..."
                        )
                    return "\n\n".join(out)
            except Exception:
                pass

            q_words = [w.lower() for w in re.findall(r"\w{3,}", q)]
            scored = []
            for md_file in v_path.rglob("*.md"):
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
            body = ast_text or (
                "[No AST matches found. Falling back to vault search]\n\n"
                + search_vault(self.vault_dir, question)
            )
        elif engine == "OBSIDIAN_VAULT_SEARCH":
            body = "=== Obsidian Vault Matches ===\n" + search_vault(self.vault_dir, question)
        else:
            from sync_brain import query_semantic_graph

            body = query_semantic_graph(question, build_dir, self.vault_dir)

        return f"{memory_prefix}[Routed via {engine} (Confidence: {confidence})]\n{body}"


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Returns OpenAI-compatible function calling definitions for all 2ndBrain tools."""
    return UniversalBrainAdapter().get_openai_tools()


UniversalLLMClient = UniversalBrainAdapter


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    adapter = UniversalBrainAdapter()
    print("=== 2ndBrain Universal Adapter Self-Test ===\n")
    print(f"Vault Directory:  {adapter.vault_dir}")
    print(f"Target Directory: {adapter.target_dir}\n")
    print(f"OpenAI Tools:     {len(adapter.get_openai_tools())} tools exported")
    print(f"Anthropic Tools:  {len(adapter.get_anthropic_tools())} tools exported")
    print(f"Gemini Tools:     {len(adapter.get_gemini_tools()[0]['function_declarations'])} functions exported")
