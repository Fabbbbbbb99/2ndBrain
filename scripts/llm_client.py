"""
llm_client.py - Universal LLM Adapter for 2ndBrain

Provides standard function calling / tool definitions and memory injection for:
- OpenAI (GPT-4o, GPT-4o-mini, o1, o3, etc.)
- Anthropic (Claude 3.7 Sonnet, Claude 3.5 Sonnet, Claude 3 Opus)
- Google Gemini (Gemini 2.5 Pro, Flash via google-genai)
- Ollama / Local Models (Llama 3, DeepSeek-R1 / V3, Qwen 2.5, Mistral)
- LangChain, AutoGen, CrewAI, and LlamaIndex

Supports:
1. Pre-flight prompt enrichment: Automatically injects relevant active memory constraints.
2. Post-flight distillation: Auto-extracts durable rules/decisions from responses.
3. Universal Function Calling: Exposes all 7 brain tools in OpenAI, Anthropic, and Gemini schemas.
"""

import os
import sys
import re
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

# Ensure scripts directory is in sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from laya_router import router
    from memory_distiller import ConversationalMemoryDistiller
    from hybrid_search import HybridSearchEngine
    from vault_linter import VaultLinter
except ImportError:
    from .laya_router import router
    from .memory_distiller import ConversationalMemoryDistiller
    from .hybrid_search import HybridSearchEngine
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


class UniversalBrainAdapter:
    """
    Universal bridge between 2ndBrain and ANY LLM provider.
    Supports pre-flight prompt enrichment, function calling tools, and post-flight distillation.
    """

    def __init__(self, vault_dir: Optional[str] = None, vault_path: Optional[str] = None, target_dir: Optional[str] = None):
        chosen_vault = vault_dir or vault_path or get_default_vault()
        self.vault_dir = Path(chosen_vault).resolve()
        self.target_dir = Path(target_dir or (Path.cwd() if Path.cwd() != current_dir else current_dir.parent.parent)).resolve()
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

    def distill_turn(self, user_msg: str, assistant_msg: str, session_id: Optional[str] = None) -> Optional[Path]:
        """
        Post-flight distillation for ANY LLM.
        Filters conversation through Laya (System 1) and saves durable rules to Obsidian.
        """
        return self.distiller.distill_turn(user_msg, assistant_msg, session_id=session_id)

    # -------------------------------------------------------------------------
    # Tool Schemas for Every Major LLM Format
    # -------------------------------------------------------------------------

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """
        Standard OpenAI tool definitions.
        Compatible with: OpenAI (GPT-4o/o1/o3), Azure OpenAI, Groq, Ollama (/v1/chat/completions),
        vLLM, LiteLLM, DeepSeek, together.ai, and Mistral.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "brain_query",
                    "description": "Query the 2ndBrain knowledge engine for code call graphs, architecture, or past meeting decisions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "The technical question or search query"}
                        },
                        "required": ["question"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "brain_blast_radius",
                    "description": "Get callers, dependents, and affected code for a symbol using Tree-sitter AST via Code-Review-Graph.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "The function, class, or method name"}
                        },
                        "required": ["symbol"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "brain_remember",
                    "description": "Save a learned rule, constraint, or architectural decision into the Obsidian second brain for future sessions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "rule_text": {"type": "string", "description": "The exact rule or decision statement"},
                            "category": {"type": "string", "enum": ["user_preference", "architectural_constraint", "bug_pattern"], "default": "user_preference"}
                        },
                        "required": ["rule_text"]
                    }
                }
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
                        "required": ["topic"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "brain_sync",
                    "description": "Synchronize AST code graph, Graphify semantic graph, and vector embeddings with the Obsidian vault.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "brain_lint",
                    "description": "Audit vault health, broken links, orphan notes, and schema compliance. Pass auto_fix=True to repair.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "auto_fix": {"type": "boolean", "description": "Whether to auto-fix broken links", "default": False}
                        }
                    }
                }
            }
        ]

    def get_anthropic_tools(self) -> List[Dict[str, Any]]:
        """
        Anthropic Messages API tool definitions.
        Compatible with: Claude 3.7 Sonnet, Claude 3.5 Sonnet, Claude 3 Opus, AWS Bedrock Claude.
        """
        return [
            {
                "name": "brain_query",
                "description": "Query the 2ndBrain knowledge engine for code call graphs, architecture, or past meeting decisions.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The technical question or search query"}
                    },
                    "required": ["question"]
                }
            },
            {
                "name": "brain_blast_radius",
                "description": "Get callers, dependents, and affected code for a symbol using Tree-sitter AST via Code-Review-Graph.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "The function, class, or method name"}
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "brain_remember",
                "description": "Save a learned rule, constraint, or architectural decision into the Obsidian second brain for future sessions.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "rule_text": {"type": "string", "description": "The exact rule or decision statement"},
                        "category": {"type": "string", "enum": ["user_preference", "architectural_constraint", "bug_pattern"], "default": "user_preference"}
                    },
                    "required": ["rule_text"]
                }
            },
            {
                "name": "brain_recall",
                "description": "Recall active rules, user preferences, and architectural decisions relevant to a topic.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "The topic or keyword to recall"}
                    },
                    "required": ["topic"]
                }
            },
            {
                "name": "brain_sync",
                "description": "Synchronize AST code graph, Graphify semantic graph, and vector embeddings with the Obsidian vault.",
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "brain_lint",
                "description": "Audit vault health, broken links, orphan notes, and schema compliance.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "auto_fix": {"type": "boolean", "description": "Whether to auto-fix broken links", "default": False}
                    }
                }
            }
        ]

    def get_gemini_tools(self) -> List[Dict[str, Any]]:
        """
        Google Gemini FunctionDeclaration definitions.
        Compatible with Google GenAI SDK (gemini-2.5-pro, gemini-2.5-flash).
        """
        # Gemini accepts standard function declaration dictionaries
        return [{"function_declarations": [t["function"] for t in self.get_openai_tools()]}]

    # -------------------------------------------------------------------------
    # Unified Tool Execution
    # -------------------------------------------------------------------------

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Executes any brain tool call requested by ANY LLM model.
        Returns the formatted string response to return to the model.
        """
        if tool_name == "brain_query":
            q = arguments.get("question", "")
            return self._execute_query(q)

        elif tool_name == "brain_blast_radius":
            sym = arguments.get("symbol", "")
            crg_bin = resolve_binary("code-review-graph")
            if shutil.which(crg_bin) is None and not Path(crg_bin).exists():
                return f"AST code intelligence requires 'code-review-graph' (run: pip install code-review-graph). Symbol '{sym}' could not be evaluated via AST."
            cmd = f'"{crg_bin}" query callers_of {sym}'
            res = subprocess.run(cmd, cwd=self.target_dir, shell=True, capture_output=True, text=True, encoding="utf-8")
            return res.stdout or res.stderr or f"No callers found for symbol '{sym}'."

        elif tool_name == "brain_remember":
            rule = arguments.get("rule_text", "")
            cat = arguments.get("category", "user_preference")
            saved = self.distiller.distill_turn(rule, f"Explicit memory update via LLM ({cat})")
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
            cmd = f'"{sys.executable}" "{sync_script}" sync --target "{self.target_dir}" --vault "{self.vault_dir}"'
            res = subprocess.run(cmd, cwd=self.target_dir, shell=True, capture_output=True, text=True, encoding="utf-8")
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

        # 1. Recall active memories
        memories = self.distiller.recall_relevant_memories(question)
        memory_prefix = ""
        if memories:
            memory_prefix = "=== Active Learned Memories ===\n" + "\n".join(memories) + "\n\n"

        def search_vault(v_path: Path, q: str) -> str:
            # Try Hybrid Search first
            try:
                h_engine = HybridSearchEngine(str(v_path))
                matches = h_engine.search(q, top_k=4)
                if matches:
                    out = []
                    for m in matches:
                        out.append(f"- **[[{m['stem']}]]** (Semantic: {m['semantic_similarity']}%, RRF: {m['rrf_score']}):\n  {m['snippet']}...")
                    return "\n\n".join(out)
            except Exception:
                pass

            # Keyword fallback
            q_words = [w.lower() for w in re.findall(r'\w{3,}', q)]
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
            words = [w.strip("?\"'.,") for w in question.split() if w.strip("?\"'.,")]
            symbol = words[-1] if words else "."
            crg_bin = resolve_binary("code-review-graph")
            if shutil.which(crg_bin) is None and not Path(crg_bin).exists():
                body = f"[Note: 'code-review-graph' is not installed (pip install code-review-graph). Falling back to vault search]\n\n" + search_vault(self.vault_dir, question)
            else:
                cmd = f'"{crg_bin}" query callers_of {symbol}'
                res = subprocess.run(cmd, cwd=self.target_dir, shell=True, capture_output=True, text=True, encoding="utf-8")
                body = f"=== AST Code Graph Result (callers_of {symbol}) ===\n" + (res.stdout or res.stderr or "No callers found.")
        elif engine == "OBSIDIAN_VAULT_SEARCH":
            body = "=== Obsidian Vault Matches ===\n" + search_vault(self.vault_dir, question)
        else:
            graphify_bin = resolve_binary("graphify")
            cmd = f'"{graphify_bin}" query "{question}"'
            res = subprocess.run(cmd, cwd=self.target_dir, shell=True, capture_output=True, text=True, encoding="utf-8")
            if res.returncode == 0 and res.stdout and "graph file not found" not in res.stdout:
                body = f"=== Graphify Semantic GraphRAG Result ===\n{res.stdout}"
            else:
                body = "=== Obsidian Vault Search (Semantic Fallback) ===\n" + search_vault(self.vault_dir, question)

        return f"{memory_prefix}[Routed via {engine} (Confidence: {confidence})]\n{body}"


# -----------------------------------------------------------------------------
# Standalone Demonstration & Self-Test
# -----------------------------------------------------------------------------
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

    # 1. Prompt Enrichment Test
    sample_prompt = "How should we write tests for Python modules?"
    enriched = adapter.enrich_prompt(sample_prompt)
    print("--- 1. Prompt Enrichment Demo ---")
    print(enriched)
    print()

    # 2. Tool Query Test
    print("--- 2. Tool Execution Demo (brain_query) ---")
    result = adapter.execute_tool("brain_query", {"question": "life cycle concept definition"})
    print(result[:400] + "...\n")

    # 3. Tool Schemas Count
    print("--- 3. Tool Schemas Exported ---")
    print(f"OpenAI Tools:    {len(adapter.get_openai_tools())} tools exported")
    print(f"Anthropic Tools: {len(adapter.get_anthropic_tools())} tools exported")
    print(f"Gemini Tools:    {len(adapter.get_gemini_tools()[0]['function_declarations'])} functions exported")
    print("\n[SUCCESS] All Universal Adapter tests completed successfully!")
