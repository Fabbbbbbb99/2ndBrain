# 2ndBrain Guidelines for Claude Code

When working in this repository:

1. **System 1 Triage & Memory Recall**:
   - Before executing complex refactors, use the `second-brain` MCP server tool `brain_recall(topic="...")` to verify if the user has recorded constraints or past decisions in Obsidian.

2. **AST Grounding (No Code Hallucination)**:
   - When checking callers, dependencies, or blast radius, call `brain_blast_radius(symbol="...")`. Ground all statements in AST truth.

3. **Continuous Conversational Learning**:
   - If the user provides a persistent preference (e.g. "Always use X", "Never do Y"), immediately call `brain_remember(rule_or_decision="...", category="user_preference")`.
