---
title: "Local-First 4-Pillar Agent Memory over Cloud SaaS"
type: decision
status: active
created: 2026-10-01
last_reinforced: 2026-10-01
confidence: 1.0
tags:
  - second-brain/memory
  - memory/decision
  - adr
---

# ⚖️ Decision: Local-First 4-Pillar Agent Memory over Cloud SaaS

> [!IMPORTANT] Architectural Decision Record (ADR)
> **Decision**: Adopt the 4-Pillar ontology (Tasks, Decisions, Facts, Skills) natively within Robbie's local SQLite/Markdown second brain rather than relying on external cloud memory platforms (e.g. ProjectBrain).
> **Status**: Active | **Date**: 2026-10-01

## Rationale
1. **Privacy & Security**: Robotics software contains proprietary algorithms, hardware pinouts, and custom URDF kinematics that must not be exfiltrated to third-party cloud servers.
2. **Deterministic Speed**: Local SQLite and BGE-small dense embeddings query in under 5ms without network latency or API rate limits.
3. **Offline Resilience**: Works seamlessly in air-gapped lab environments and inside WSL2 Linux without internet connectivity.

## Impact & Trade-offs
- Robbie maintains complete data sovereignty on disk (`C:\Users\Fabian\Desktop\Robbie\vault_template\04-Agent-Memory`).
- All memories remain directly human-readable and editable in Obsidian.

---
*Linked to [[04-Agent-Memory-MOC]]*
