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
