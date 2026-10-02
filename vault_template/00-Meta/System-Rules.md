---
title: "System Rules for LLMs"
tags:
  - second-brain/rules
---

# Operational Rules for AI Models Reading / Writing in this Vault

When you (an LLM agent such as Claude, Gemini, Antigravity, or Cursor) interact with this vault:

### 1. Linking Rule
- Always use Obsidian wikilinks: `[[Note Name]]` or `[[Folder/Note Name|Anchor Text]]`.
- Link directly to established entities in `01-Concepts/` or code modules in `02-Codebase/`.

### 2. Frontmatter Standards
- Every note MUST begin with YAML frontmatter:
  ```yaml
  ---
  type: concept | module | transcript | correction | decision
  created: YYYY-MM-DD
  tags:
    - second-brain/...
  ---
  ```

### 3. Honesty & Source Attribution
- Never invent dependencies or call relations. Ground code relationships in `code-review-graph` AST truth.
- When citing facts, link to the source document in `03-Sources/Transcripts/` or the specific file path.

### 4. Memory Persistence
- When the user gives an explicit preference (e.g. "Always use X", "Never do Y"), crystallize it into `04-Agent-Memory/` immediately.

### 5. Git Isolation Rule (Strict Framework Only)
- NEVER stage or commit domain knowledge notes, ingested materials/sources, or dynamic memory notes (`04-Agent-Memory/*/*.md`) to Git.
- Git strictly tracks the core framework code, engines, CLI scripts, documentation, and directory scaffolding READMEs only.
- All user knowledge base notes and learned conversational memories are 100% private, sovereign, and local.

### 6. "I Have ADHD" Operational Protocol (ayghri/i-have-adhd Standard)
- **Action-First Execution**: Never begin responses, notes, or execution logs with pleasantries, recaps, throat-clearing, or conversational filler. Start directly with the actionable result, terminal command, or code diff.
- **5-Step Cognitive Ceiling**: All execution plans, task backlogs, and concept summaries MUST be broken into numbered lists capped at a MAXIMUM of 5 items. Never present walls of unchecked text.
- **Explicit Next Action**: Every task, note, and turn must conclude with a single, unambiguous, executable next step.
- **Scope & Duration Estimates**: Assign realistic scope and time estimates (e.g. `[~15m]`, `[~1h]`) to pending items in `01-Tasks/`.
- **Zero Fluff**: Ban "Hope this helps!", "Let me know if you need anything else", and rhetorical cheerleading.

### 7. "No AI Slop" Quality Standard (petergyang/no-ai-slop Standard)
- **Banned Rhetorical Tics**: Strict prohibition against AI throat-clearing openers (*"In today's fast-moving..."*, *"Let's dive in..."*), faux-profound conclusions (*"The future is here..."*, *"At the end of the day..."*), and artificial binary contrasts (*"It's not about X; it's about Y"*).
- **High Information Density**: Prioritize mathematical formulations ($\LaTeX$), formal types, exact C++/Python interface signatures, ROS 2 QoS profiles, and verified call graphs over generic prose summaries.
- **Empirical Evidence & Grounding**: Every engineering statement must link to an exact source file, line number, REP, or peer-reviewed citation. Never hallucinate API parameters or unverified dependencies.
- **Anti-Boilerplate**: Strip publisher metadata, copyright blocks, ISBN dumps, and cataloguing tables from ingested documents. Ban happy-path-only stubs lacking error handling or type definitions.

### 8. GSD (Get Stuff Done) Work-Unit Specification
- **Atomic Work-Units**: All tasks under `04-Agent-Memory/01-Tasks/` must define exactly 1 measurable goal and maximum 3–5 discrete checklist items.
- **Strict Time-Boxing**: Every work-unit must be scoped to $\le 45$ minutes (`timebox: 30m`). Larger tasks must be decomposed into sequential atomic units.
- **Binary Definition of Done**: Verification criteria must be pass/fail checks (e.g. exit code 0, 0 linter errors) rather than qualitative descriptions.
- **Mandatory Failure Rollback**: Every task must declare an explicit, executable rollback command (e.g. `git restore <target>`) in case of unexpected failure.
- **Pre-Commit Enforcement**: The pre-commit quality gate (`scripts/pre_commit_hook.py`) automatically blocks any commit violating Git Isolation or vault health.

