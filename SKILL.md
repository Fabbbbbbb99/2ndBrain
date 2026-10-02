---
name: 2ndBrain
description: "Dynamic local-first second brain for LLMs. Orchestrates Laya (System 1 fast reflex routing & triage), Code-Review-Graph (Tree-sitter AST & blast radius), Graphify (semantic GraphRAG & community detection), Obsidian (bi-directional markdown memory vault & conversational learning), and 4-Pillar Persistent Memory (Tasks, Decisions, Facts, Skills). Use when the user asks to index, query, review, update, or recall knowledge from their second brain."
---

# /2ndBrain

A local-first, dual-process (System 1 Reflex + System 2 Reasoning) second brain that turns code, documents, and live conversation turns into a persistent, queryable knowledge engine.

---

## Capabilities & Architecture

| Layer | Engine | Responsibility |
| :--- | :--- | :--- |
| **L0: System 1 Reflex** | **Laya** (ModernBERT-large 421M / Calibrated Heuristic Engine, <1ms) | Ultra-fast routing, document novelty gating, blast-radius risk scoring, frontmatter tagging, and conversation memory filtering via typed schemas (Choice, Score, Noul). |
| **L1: Structural Code** | **Code-Review-Graph** (Tree-sitter AST & SQLite) | Precise function call trees, callers/callees, import hierarchies, and Git diff blast radius without LLM hallucination. |
| **L2: Semantic Concepts** | **Graphify** (GraphRAG) | Architectural community detection (Leiden/Louvain), god nodes, bridge concepts, and cross-document relationships. |
| **L3: Knowledge Vault** | **Obsidian** (Markdown + [[wikilinks]]) | Human-in-the-loop navigation, persistent ADRs, and self-evolving prompt/conversational memory. |
| **L4: Vector Index** | **Hybrid Search** (Parent-Child Dense + SQLite BM25) | Decomposes notes into Parent Sections and matches on granular Child Chunks (384-d `fastembed` BGE-small-en-v1.5) fused with BM25 via Reciprocal Rank Fusion (RRF). Injects complete, rich parent clauses directly to LLMs. |
| **L5: Persistent Memory** | **4-Pillar Memory Ontology** (Local-First SQLite/MD) | Categorizes long-term context into `01-Tasks`, `02-Decisions`, `03-Facts`, and `04-Skills` that survive across sessions and model handoffs. |

---

## 🏛️ The 4 Memory Pillars (`04-Agent-Memory/`)

1. **📋 `01-Tasks/`**: Actionable work items, sprint backlogs, GSD roadmaps, and execution status (`pending`, `in_progress`, `completed`).
2. **⚖️ `02-Decisions/`**: Architectural Decision Records (ADRs) with rationale, evaluated alternatives, and trade-offs.
3. **📌 `03-Facts/`**: System invariants, hardware pinouts, REP standards, and OS/distro rules.
4. **🛠️ `04-Skills/`**: Reusable execution recipes, zero-copy code patterns, and troubleshooting procedures.

---

## Usage Commands

The orchestrator CLI is located in `scripts/sync_brain.py`. By default, `--vault` automatically discovers your vault via `$env:SECOND_BRAIN_VAULT`, sibling `vault_template/`, `./vault`, or `~/.2ndbrain/vault`.

```powershell
# Set helper alias for convenience (adjust path if running outside repo root):
$SB = "scripts/sync_brain.py"

# 1. Ingest external knowledge materials (PDFs, docs) into Obsidian vault (with table & equation preservation)
python $SB ingest --source "path/to/materials"

# 2. Synchronize the second brain (AST + Graphify + fastembed Vectors + Obsidian)
python $SB sync --target "."

# 3. Intelligent query routed by Laya (System 1) with Parent-Child hierarchical search & active memory recall
python $SB query "How does module X work and who calls it?"

# 4. Compute AST blast radius and callers for a symbol or file
python $SB blast <symbol_or_function_name>

# 5. Explicitly crystallize a durable memory across pillars (--category: task, decision, fact, skill)
python $SB remember "Always use Pydantic models for data validation" --category fact --tag "architecture"
python $SB remember "Use gRPC instead of REST for internal microservices" --category decision --tag "networking"
python $SB remember "Implement user authentication flow" --category task --tag "auth"

# 6. Archive a conversation dialogue or meeting into persistent memory
python $SB session --topic "Architecture Review" --summary "Decided on microservices boundary." --decisions "Use gRPC for internal RPC"

# 7. Run Memory Consolidation ("Sleep Cycle") to prune, merge, and promote rules
python $SB consolidate
python $SB consolidate --dry-run

# 8. Distill durable rules and decisions from a chat transcript
python $SB distill --transcript "path/to/transcript.jsonl"

# 9. Audit vault health, broken links, and unindexed notes
python $SB lint

# 10. Start Universal FastMCP Server (for Claude Code, Cursor, Windsurf)
python "scripts/mcp_server.py"

# 11. Manage GSD Atomic Tasks (Create, List, Start, Done)
python "scripts/gsd_task.py" create "Synthesize USBL" --goal "Extract Water Linked UGPS" --timebox 30m
python "scripts/gsd_task.py" list
python "scripts/gsd_task.py" start "synthesize_usbl"
python "scripts/gsd_task.py" done "synthesize_usbl"

# 12. Install Git Pre-Commit Hook (Rule 5 Git Isolation + Quality Guard)
python "scripts/install_git_hooks.py"
```

---

## Universal LLM Compatibility

This second brain works with **any LLM environment**:

- **Google Antigravity**: Type `@2ndBrain <question>`, `@robbie <question>`, or run the CLI commands.
- **Claude Desktop / Claude Code**: Add `scripts/mcp_server.py` to your MCP configuration (`claude_desktop_config.json`). Exposes 13 native tools (`brain_query`, `brain_blast_radius`, `brain_remember`, `brain_remember_task`, `brain_remember_decision`, `brain_remember_fact`, `brain_remember_skill`, `brain_recall`, `brain_archive_session`, `brain_consolidate_memory`, `brain_sync`, `brain_ingest`, `brain_lint`).
- **Cursor IDE / Windsurf**: Add `scripts/mcp_server.py` to Settings -> Features -> MCP Servers.
- **OpenAI, Anthropic & Gemini Python SDKs**: Use `UniversalBrainAdapter` from `scripts/llm_client.py` to auto-enrich prompts or export standard function calling tools.
- **Local Models (Ollama, LM Studio, Llama 3, DeepSeek-R1)**: Prepend `configs/SYSTEM_PROMPT.md` to your system prompt, or run `UniversalBrainAdapter.enrich_prompt(prompt)` before calling local APIs.

---

## Agent Operational Protocol

When activated by the user, follow these procedures:

### Step 1: System 1 Triage & Prompt Recall
1. Evaluate query with Laya router logic:
   ```powershell
   python scripts/sync_brain.py query "<user_prompt>"
   ```
2. Check for active memories in `04-Agent-Memory/` (`01-Tasks`, `02-Decisions`, `03-Facts`, `04-Skills`) using `ConversationalMemoryDistiller.recall_relevant_memories()`.
3. If past constraints exist, cite them before generating code or architecture plans:
   > *"[2ndBrain Grounding]: Applying active memory rule: [[Rule-Name]]."*

### Step 2: Query Routing & Parent-Child Context
- If the question targets **function callers, class hierarchies, or blast radius**:
  - Run `python $SB blast <symbol>` (queries `code-review-graph` AST).
- If the question targets **high-level architecture, module boundaries, or concepts**:
  - Query `graphify query "<question>"`.
- If the question targets **past decisions, meetings, or preferences**:
  - Search `04-Agent-Memory/02-Decisions/`, `04-Agent-Memory/03-Facts/`, or `04-Agent-Memory/Sessions/`.

### Step 3: Conversational Memory Distillation & Archival
1. Whenever the user provides an explicit instruction, correction, or architectural decision during a chat turn:
   - Evaluate with Laya: *"Does this turn establish a persistent project constraint, user preference, or bug fix?"*
   - If probability $P \ge 0.7$, call `ConversationalMemoryDistiller.distill_turn(user_prompt, model_response)`.
2. When concluding a major design discussion or milestone session:
   - Call `ConversationalMemoryDistiller.archive_session(topic, summary, key_decisions)` to save a permanent record under `04-Agent-Memory/Sessions/`.
3. Periodically run `python $SB consolidate` to clean up decayed or superseded rules, merge near-duplicates, and promote recurring rules into permanent architecture guidelines.

### Step 4: Incremental Knowledge Sync
When files are added or modified:
1. Run `python $SB sync --target "."` to update the AST index, refresh Graphify semantic clusters, and rebuild SQLite Parent-Child vector embeddings.

---

## Honesty & Grounding Rules
- **No Hallucinated Edges**: Never claim a function calls another unless confirmed by `code-review-graph` or source AST.
- **Zero Token Waste**: Route queries with Laya first before invoking expensive generative models.
- **Transparent Operational Mode**: Distinguish between the fast calibrated heuristic reflex engine and optional full neural ModernBERT runs.
- **Human Transparency**: Every concept, session archive, and learned memory must be human-readable and editable inside Obsidian.

---

## ⚡ Operational Protocols: "I Have ADHD" & "No AI Slop"

All autonomous agents executing under `/2ndBrain` must strictly adhere to:

### 1. "I Have ADHD" Protocol (`ayghri/i-have-adhd`)
- **Action-First Execution**: Begin every response, plan, or note directly with the concrete action, terminal command, or code diff. Never start with conversational pleasantries, recaps, or throat-clearing.
- **5-Step Cognitive Ceiling**: All execution plans, task backlogs, and concept summaries MUST be broken into numbered lists capped at a MAXIMUM of 5 items.
- **Single Concrete Next Step**: Every task, note, and turn must conclude with a single, unambiguous, executable next step.
- **Scope & Duration Estimates**: Assign realistic scope and time estimates (e.g. `[~15m]`, `[~1h]`) to pending items in `01-Tasks/`.
- **Zero Fluff**: Ban "Hope this helps!", "Let me know if you need anything else", and rhetorical cheerleading.

### 2. "No AI Slop" Quality Standard (`petergyang/no-ai-slop`)
- **Banned Rhetorical Tics**: Strict prohibition against AI throat-clearing openers (*"In today's fast-moving..."*, *"Let's dive in..."*), faux-profound conclusions (*"The future is here..."*, *"At the end of the day..."*), and artificial binary contrasts (*"It's not about X; it's about Y"*).
- **High Information Density**: Prioritize mathematical formulations ($\LaTeX$), formal types, exact C++/Python interface signatures, ROS 2 QoS profiles, and verified call graphs over generic prose summaries.
- **Empirical Evidence & Grounding**: Every engineering statement must link to an exact source file, line number, REP, or peer-reviewed citation. Never hallucinate API parameters or unverified dependencies.
- **Anti-Boilerplate**: Strip publisher metadata, copyright blocks, ISBN dumps, and cataloguing tables from ingested documents. Ban happy-path-only stubs lacking error handling or type definitions.
