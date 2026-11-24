# 2ndBrain Agent Grounding Rules (Claude Desktop / Cursor / Generic MCP Clients)

Paste or adapt these instructions into your AI client's custom instructions (`.cursorrules`, `CLAUDE.md`, or system prompt). Antigravity users can alternatively run `brain export-skills` to generate per-domain `SKILL.md` files from `<VAULT>/specialists/*.yaml`.

## Retrieval & Grounding
- Query `second-brain` MCP tools before answering domain-specific questions:
  - `brain_route`: Inspect System 1 domain routing plan.
  - `brain_query`: Retrieve bounded evidence packs (`<= 2,500` tokens) with verbatim `vault://` citations and active Store 2 memories.
  - `brain_grep`: Run exact literal or regex scans across `parsed/` and `wiki/`.
  - `brain_get_clause`: Fetch O(1) full clause text, children, and inbound `cited_by` / `inferred_links`.
- Always cite the returned `vault://` URI, source ID, and version tag.
- On `[STALE]` wiki pages, re-read upstream `parsed/` URIs via `brain_get_clause`.
- On `VISUAL_CROP_REQUIRED`, inspect the returned `crop_path` image and record the transcription via `brain_cache_visual_repair(certify=false)`.

## Memory & Compilation Governance
- Store durable preferences, decisions, and facts via `brain_remember` (`ledger_type` in `{preferences, decisions, facts}`, optional lowercase `project_tag`, and `source_ref`).
- Inspect `staged` items with `brain_recall(status="staged")`; certify via CLI (`brain certify --id <id> --decision approve`).
- Call `brain_compile` only when explicitly requested by the user.
