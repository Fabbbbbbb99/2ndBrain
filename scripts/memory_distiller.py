"""
memory_distiller.py - Conversational & Prompt Memory Distillation for 2ndBrain

Captures insights, user preferences, corrections, and architectural decisions
from conversations/prompts, filters them through Laya (System 1), and crystallizes
them into Obsidian markdown notes with [[wikilinks]].

Includes:
- Memory Reinforcement (detects duplicate/repeated constraints and increments confidence & usage).
- Memory Decay (applies exponential half-life decay based on recency and reinforcement).
- Conflict Resolution & Superseding (detects contradictions/overrides and marks old rules as superseded).
- Automated Session Dialogue Archival (compacts dialogues into 04-Agent-Memory/Sessions/).
- Periodic Memory Consolidation / Sleep Cycle (prunes superseded/decayed rules, merges near-duplicates, promotes durable rules to Architecture guidelines).
"""

import os
import json
import re
import math
import difflib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

try:
    from laya_router import router
except ImportError:
    from .laya_router import router


class ConversationalMemoryDistiller:
    def __init__(self, vault_dir: str):
        self.vault_dir = Path(vault_dir).resolve()
        self.memory_dir = self.vault_dir / "04-Agent-Memory"
        # 4-Pillar Structured Ontology (Tasks, Decisions, Facts, Skills)
        self.tasks_dir = self.memory_dir / "01-Tasks"
        self.decisions_dir = self.memory_dir / "02-Decisions"
        self.facts_dir = self.memory_dir / "03-Facts"
        self.skills_dir = self.memory_dir / "04-Skills"
        # Legacy & Auxiliary memory directories for continuity
        self.corrections_dir = self.memory_dir / "Corrections"
        self.sessions_dir = self.memory_dir / "Sessions"
        self.archive_dir = self.memory_dir / "Archive"
        self.reports_dir = self.memory_dir / "Consolidation-Reports"
        self.concepts_decisions_dir = self.vault_dir / "01-Concepts" / "Decisions"
        self.guidelines_dir = self.vault_dir / "01-Concepts" / "Architecture"

        # Ensure all directories exist
        for d in [self.tasks_dir, self.decisions_dir, self.facts_dir, self.skills_dir,
                 self.corrections_dir, self.sessions_dir, self.archive_dir, self.reports_dir,
                 self.concepts_decisions_dir, self.guidelines_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _parse_note_metadata(self, note_path: Path) -> Dict[str, Any]:
        """Extracts YAML frontmatter, rule statement, and status from a memory note."""
        try:
            content = note_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return {}

        meta: Dict[str, Any] = {
            "path": note_path,
            "status": "active",
            "type": "user_preference",
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "last_reinforced": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "reinforcement_count": 1,
            "confidence": 0.8,
            "superseded_by": None,
            "supersedes": None,
            "statement": "",
            "raw_content": content
        }

        # Parse YAML frontmatter
        fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if fm_match:
            fm_text = fm_match.group(1)
            for line in fm_text.splitlines():
                if ":" in line and not line.strip().startswith("-"):
                    k, v = line.split(":", 1)
                    k = k.strip().lower()
                    v = v.strip().strip('"\'')
                    if k == "title":
                        meta["title"] = v
                    elif k == "status":
                        meta["status"] = v.lower()
                    elif k == "type":
                        meta["type"] = v
                    elif k == "created":
                        meta["created"] = v
                    elif k == "last_reinforced":
                        meta["last_reinforced"] = v
                    elif k == "reinforcement_count":
                        try:
                            meta["reinforcement_count"] = int(v)
                        except ValueError:
                            pass
                    elif k == "confidence":
                        try:
                            meta["confidence"] = float(v)
                        except ValueError:
                            pass
                    elif k == "superseded_by":
                        meta["superseded_by"] = v if v != "null" else None
                    elif k == "supersedes":
                        meta["supersedes"] = v if v != "null" else None

        # Extract statement
        stmt_match = re.search(r"\*\*Statement\*\*:\s*(.*)", content)
        if stmt_match:
            meta["statement"] = stmt_match.group(1).strip()
        elif meta.get("title"):
            meta["statement"] = meta["title"]
        else:
            meta["statement"] = note_path.stem

        return meta

    def _update_note_frontmatter(self, note_path: Path, updates: Dict[str, Any], append_body_warning: Optional[str] = None):
        """Safely updates YAML frontmatter fields in a note."""
        try:
            content = note_path.read_text(encoding="utf-8")
        except Exception:
            return

        fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if not fm_match:
            return

        fm_lines = fm_match.group(1).splitlines()
        new_fm_lines = []
        handled_keys = set()

        for line in fm_lines:
            if ":" in line and not line.strip().startswith("-"):
                k = line.split(":", 1)[0].strip().lower()
                if k in updates:
                    val = updates[k]
                    val_str = f'"{val}"' if isinstance(val, str) and not val.startswith("[[") else str(val).lower() if isinstance(val, bool) else str(val)
                    new_fm_lines.append(f"{k}: {val_str}")
                    handled_keys.add(k)
                    continue
            new_fm_lines.append(line)

        # Append any unhandled new keys
        for k, v in updates.items():
            if k not in handled_keys:
                val_str = f'"{v}"' if isinstance(v, str) and not v.startswith("[[") else str(v)
                new_fm_lines.append(f"{k}: {val_str}")

        body = content[fm_match.end():]
        if append_body_warning and append_body_warning not in body:
            body = f"\n{append_body_warning}\n" + body

        new_content = "---\n" + "\n".join(new_fm_lines) + "\n---\n" + body
        note_path.write_text(new_content, encoding="utf-8")

    def _detect_conflict_or_reinforcement(self, new_rule: str, user_msg: str) -> Tuple[Optional[Path], bool]:
        """
        Scans existing memory notes.
        Returns: (matching_note_path, is_reinforcement)
        """
        all_notes = (
            list(self.tasks_dir.glob("*.md")) +
            list(self.decisions_dir.glob("*.md")) +
            list(self.facts_dir.glob("*.md")) +
            list(self.skills_dir.glob("*.md")) +
            list(self.corrections_dir.glob("*.md")) +
            list(self.concepts_decisions_dir.glob("*.md"))
        )
        new_rule_lower = new_rule.lower()
        user_lower = user_msg.lower()
        new_tokens = set(re.findall(r"\w{3,}", new_rule_lower))

        override_phrases = ["instead of", "from now on", "supersede", "replace", "no longer", "stop using", "prefer"]
        has_override_intent = any(op in user_lower or op in new_rule_lower for op in override_phrases)

        negation_terms = {"never", "not", "dont", "don't", "avoid", "stop", "prohibit", "disallow"}
        affirmative_terms = {"always", "must", "prefer", "use", "require", "shall", "ensure"}

        new_has_negation = bool(new_tokens & negation_terms)
        new_has_affirmative = bool(new_tokens & affirmative_terms)

        for p in all_notes:
            if p.name.lower() in ("readme.md", "index.md"):
                continue
            meta = self._parse_note_metadata(p)
            if meta.get("status") == "superseded":
                continue

            existing_stmt = meta.get("statement", "").lower()
            if not existing_stmt:
                continue

            # 1. Check for Reinforcement (Exact or high semantic match >= 0.80)
            ratio = difflib.SequenceMatcher(None, new_rule_lower, existing_stmt).ratio()
            if ratio >= 0.80:
                return p, True  # Reinforce existing

            # 2. Check for Conflict / Superseding
            exist_tokens = set(re.findall(r"\w{3,}", existing_stmt))
            if not exist_tokens or not new_tokens:
                continue

            common_tokens = new_tokens & exist_tokens
            overlap_ratio = len(common_tokens) / min(len(new_tokens), len(exist_tokens))

            exist_has_negation = bool(exist_tokens & negation_terms)
            exist_has_affirmative = bool(exist_tokens & affirmative_terms)

            # Polarity flip (e.g. was always, now never) or explicit override on same topic
            is_polarity_flip = (new_has_negation and exist_has_affirmative) or (new_has_affirmative and exist_has_negation)

            if overlap_ratio >= 0.45 and (is_polarity_flip or has_override_intent):
                return p, False  # Conflict found! Supersede old note

        return None, False

    def distill_turn(self, user_msg: str, assistant_msg: str, session_id: Optional[str] = None, category: Optional[str] = None) -> Optional[Path]:
        """
        Evaluates a single conversation turn. If Laya identifies a durable rule or decision:
        1. Checks if it reinforces an existing rule.
        2. Checks if it conflicts with an existing rule.
        3. Otherwise, writes a new timestamped memory note into Obsidian.
        """
        eval_result = router.evaluate_conversational_turn(user_msg, assistant_msg)

        if not eval_result["is_durable_memory"]:
            return None  # Filtered out as transient noise

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        m_type = eval_result["memory_type"]
        rule_text = eval_result["rule_text"] or user_msg.strip()

        # Check for existing match (reinforcement or conflict)
        matched_note, is_reinforce = self._detect_conflict_or_reinforcement(rule_text, user_msg)

        if matched_note and is_reinforce:
            # Reinforce existing note
            meta = self._parse_note_metadata(matched_note)
            new_count = meta.get("reinforcement_count", 1) + 1
            new_conf = min(0.99, meta.get("confidence", 0.8) + 0.05)
            self._update_note_frontmatter(matched_note, {
                "last_reinforced": now_str,
                "reinforcement_count": new_count,
                "confidence": round(new_conf, 2),
                "status": "active"
            }, append_body_warning=f"- **Reinforced on {now_str}**: `{rule_text[:70]}`")
            print(f"[Memory Distiller] Reinforced active rule: {matched_note.name} (Count: {new_count}, Confidence: {new_conf:.2f})")
            return matched_note

        # Create slug for new rule
        clean_words = re.sub(r"[^\w\s-]", "", rule_text.lower()).split()[:5]
        slug = "-".join(clean_words) or f"rule_{timestamp_str}"
        file_name = f"{now_str}_{slug}.md"

        cat = (category or "").lower()
        if cat in ("task", "tasks", "01-tasks"):
            target_path = self.tasks_dir / file_name
            m_type = "task"
        elif cat in ("decision", "decisions", "02-decisions", "adr"):
            target_path = self.decisions_dir / file_name
            m_type = "decision"
        elif cat in ("fact", "facts", "03-facts", "invariant"):
            target_path = self.facts_dir / file_name
            m_type = "fact"
        elif cat in ("skill", "skills", "04-skills", "procedure", "workflow"):
            target_path = self.skills_dir / file_name
            m_type = "skill"
        elif m_type in ("architectural_constraint", "decision"):
            target_path = self.decisions_dir / file_name
        elif m_type in ("fact", "system_fact"):
            target_path = self.facts_dir / file_name
        elif m_type in ("task", "todo"):
            target_path = self.tasks_dir / file_name
        elif m_type in ("skill", "workflow"):
            target_path = self.skills_dir / file_name
        else:
            target_path = self.corrections_dir / file_name

        supersedes_ref = None
        if matched_note and not is_reinforce:
            # Conflict detected: Supersede old note
            old_meta = self._parse_note_metadata(matched_note)
            supersedes_ref = f"[[{matched_note.stem}]]"
            self._update_note_frontmatter(matched_note, {
                "status": "superseded",
                "superseded_by": f"[[{target_path.stem}]]"
            }, append_body_warning=f"> [!WARNING] Superseded Rule\n> This rule was superseded on {now_str} by [[{target_path.stem}]].")
            print(f"[Memory Distiller] Superseded conflicting rule: {matched_note.name} -> [[{target_path.stem}]]")

        supersedes_frontmatter = f'supersedes: "{supersedes_ref}"\n' if supersedes_ref else ""
        supersedes_body = f"- **Supersedes**: {supersedes_ref}\n" if supersedes_ref else ""

        note_content = f"""---
type: {m_type}
status: active
created: {now_str}
last_reinforced: {now_str}
reinforcement_count: 1
confidence: {eval_result['probability']}
superseded_by: null
{supersedes_frontmatter}session_id: "{session_id or 'adhoc'}"
tags:
  - second-brain/memory
  - memory/{m_type}
---

# {rule_text[:80]}

> [!IMPORTANT] Learned Rule / Decision
> **Statement**: {rule_text}

{supersedes_body}### Context & Origin
- **Triggering User Prompt**: 
  > {user_msg.strip()}
- **Model Grounding**:
  {assistant_msg.strip()[:300]}...

### Linked Concepts
- [[04-Agent-Memory]]
"""
        target_path.write_text(note_content, encoding="utf-8")
        print(f"[Memory Distiller] Crystallized new memory note: {target_path.name}")
        return target_path


    # =========================================================================
    # 4-PILLAR EXPLICIT MEMORY HELPERS
    # =========================================================================
    def remember_task(self, title: str, details: str, status: str = "pending", priority: str = "medium", milestone: str = "") -> Path:
        """Records an actionable task or roadmap milestone in 04-Agent-Memory/01-Tasks/."""
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        slug = "-".join(re.sub(r"[^\w\s-]", "", title.lower()).split()[:5]) or "task"
        target_path = self.tasks_dir / f"{now_str}_{slug}.md"
        content = f"""---
title: "{title}"
type: task
status: {status}
priority: {priority}
milestone: "{milestone}"
created: {now_str}
last_reinforced: {now_str}
tags:
  - second-brain/memory
  - memory/task
---

# 📋 Task: {title}

> [!NOTE] Task Details
> **Status**: {status} | **Priority**: {priority} | **Milestone**: {milestone or 'General'}

## Description
{details.strip()}

### Verification Criteria
- [ ] Code implementation complete
- [ ] Unit tests pass
- [ ] Documented in project roadmap
"""
        target_path.write_text(content, encoding="utf-8")
        print(f"[Memory Distiller] Recorded task: {target_path.name}")
        return target_path

    def remember_decision(self, title: str, rationale: str, alternatives: Optional[List[str]] = None, impact: str = "") -> Path:
        """Records an Architectural Decision Record (ADR) in 04-Agent-Memory/02-Decisions/."""
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        slug = "-".join(re.sub(r"[^\w\s-]", "", title.lower()).split()[:5]) or "decision"
        target_path = self.decisions_dir / f"{now_str}_{slug}.md"
        alt_str = "\n".join(f"- {a}" for a in (alternatives or [])) or "- None considered"
        content = f"""---
title: "{title}"
type: decision
status: active
created: {now_str}
last_reinforced: {now_str}
confidence: 0.95
tags:
  - second-brain/memory
  - memory/decision
  - adr
---

# ⚖️ Decision: {title}

> [!IMPORTANT] Architectural Decision Record (ADR)
> **Decision**: {title}
> **Status**: Active | **Date**: {now_str}

## Rationale
{rationale.strip()}

## Alternatives Considered
{alt_str}

## Impact & Trade-offs
{impact.strip() or 'Establishes project standard.'}
"""
        target_path.write_text(content, encoding="utf-8")
        print(f"[Memory Distiller] Recorded decision: {target_path.name}")
        return target_path

    def remember_fact(self, statement: str, source: str = "project_grounding", domain: str = "robotics") -> Path:
        """Records an invariant truth or system configuration in 04-Agent-Memory/03-Facts/."""
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        slug = "-".join(re.sub(r"[^\w\s-]", "", statement.lower()).split()[:5]) or "fact"
        target_path = self.facts_dir / f"{now_str}_{slug}.md"
        content = f"""---
title: "{statement[:80]}"
type: fact
status: active
domain: {domain}
created: {now_str}
last_reinforced: {now_str}
confidence: 1.0
tags:
  - second-brain/memory
  - memory/fact
---

# 📌 Fact: {statement[:80]}

> [!NOTE] System Invariant
> **Fact**: {statement}
> **Domain**: {domain} | **Source**: {source}
"""
        target_path.write_text(content, encoding="utf-8")
        print(f"[Memory Distiller] Recorded fact: {target_path.name}")
        return target_path

    def remember_skill(self, skill_name: str, steps: List[str], trigger: str = "") -> Path:
        """Records a learned execution recipe or procedure in 04-Agent-Memory/04-Skills/."""
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        slug = "-".join(re.sub(r"[^\w\s-]", "", skill_name.lower()).split()[:5]) or "skill"
        target_path = self.skills_dir / f"{now_str}_{slug}.md"
        steps_str = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
        content = f"""---
title: "{skill_name}"
type: skill
status: active
trigger: "{trigger}"
created: {now_str}
last_reinforced: {now_str}
tags:
  - second-brain/memory
  - memory/skill
---

# 🛠️ Skill: {skill_name}

> [!TIP] Execution Recipe
> **Trigger**: {trigger or 'Invoked when solving ' + skill_name}

## Step-by-Step Procedure
{steps_str}
"""
        target_path.write_text(content, encoding="utf-8")
        print(f"[Memory Distiller] Recorded skill: {target_path.name}")
        return target_path

    # =========================================================================
    # IMPROVEMENT 3: Automated Session Dialogue Archival
    # =========================================================================
    def archive_session(
        self,
        topic: str,
        summary: str,
        dialogue: Optional[List[Dict[str, str]]] = None,
        key_decisions: Optional[List[str]] = None,
        session_id: Optional[str] = None
    ) -> Path:
        """
        Compacts and archives an entire conversation dialogue into a persistent session note:
        Stored in: 04-Agent-Memory/Sessions/YYYY-MM-DD_<topic-slug>.md
        """
        now_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        now_time = datetime.now(timezone.utc).strftime("%H%M%S")
        s_id = session_id or f"sess_{now_date}_{now_time}"

        # Build clean filename slug
        clean_topic = re.sub(r'[^\w\s-]', '', topic.strip())
        slug = re.sub(r'[-\s]+', '-', clean_topic.lower())[:45].rstrip('-') or "session"
        file_name = f"{now_date}_{slug}.md"
        target_path = self.sessions_dir / file_name

        # Format key decisions list
        decisions_block = ""
        if key_decisions:
            decisions_block = "\n".join(f"- 📌 **Takeaway**: {d}" for d in key_decisions)
        else:
            decisions_block = "- General discussion and exploration."

        # Format dialogue transcript
        dialogue_block = ""
        if dialogue:
            formatted_turns = []
            for turn in dialogue:
                role = turn.get("role", "User").capitalize()
                text = turn.get("content", "").strip()
                if text:
                    formatted_turns.append(f"**{role}**:\n> {text[:800]}\n")
            dialogue_block = "\n".join(formatted_turns)
        else:
            dialogue_block = "*Detailed turn-by-turn dialogue omitted for conciseness.*"

        content = f"""---
title: "{topic} - Session Archive"
type: memory/session
date: {now_date}
session_id: "{s_id}"
tags:
  - second-brain/session
  - memory/session
---

# 📝 Session: {topic}

- **Date**: `{now_date}`
- **Session Identifier**: `{s_id}`
- **Master Index**: [[00-Meta/Index]]

---

## 1. Executive Summary
{summary.strip()}

---

## 2. Key Decisions & Takeaways
{decisions_block}

---

## 3. Dialogue Highlights & Context
{dialogue_block}

---

## 4. Memory Cross-References
- [[04-Agent-Memory]]
- [[04-Agent-Memory/Sessions/README|All Sessions]]
"""
        target_path.write_text(content, encoding="utf-8")
        print(f"[Memory Distiller] Successfully archived session dialogue into: {target_path.name}")

        # Update Sessions README index if present
        readme = self.sessions_dir / "README.md"
        if readme.exists():
            entry = f"- [[{target_path.stem}|{now_date}: {topic}]]"
            rm_text = readme.read_text(encoding="utf-8")
            if target_path.stem not in rm_text:
                readme.write_text(rm_text.strip() + f"\n{entry}\n", encoding="utf-8")

        # Automatically update vector embeddings for this session note
        try:
            from hybrid_search import HybridSearchEngine
            engine = HybridSearchEngine(self.vault_dir)
            engine.index_note(target_path, force=True)
        except Exception:
            pass

        return target_path

    # =========================================================================
    # IMPROVEMENT 4: Periodic Memory Consolidation ("Sleep Cycle")
    # =========================================================================
    def consolidate_memories(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Executes the Memory Consolidation ("Sleep Cycle") routine:
        1. Archives permanently superseded notes and dead/decayed rules into 04-Agent-Memory/Archive/.
        2. Detects high-reinforcement durable rules (count >= 3) and synthesizes them into
           permanent Architectural Guidelines (01-Concepts/Architecture/Project-Guidelines.md).
        3. Identifies and merges near-duplicate rules across active memories.
        4. Generates a comprehensive consolidation audit report.
        """
        now_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today = datetime.now(timezone.utc).date()

        all_notes = [
            p for p in (list(self.corrections_dir.glob("*.md")) + list(self.decisions_dir.glob("*.md")))
            if p.name.lower() not in ("readme.md", "index.md")
        ]

        archived: List[Dict[str, Any]] = []
        promoted: List[Dict[str, Any]] = []
        merged: List[Dict[str, Any]] = []
        active_remaining: List[Dict[str, Any]] = []

        # 1. Inspect notes for Superseding & Exponential Decay
        for p in all_notes:
            meta = self._parse_note_metadata(p)
            status = meta.get("status", "active")
            count = meta.get("reinforcement_count", 1)
            conf = meta.get("confidence", 0.8)

            try:
                last_date = datetime.strptime(meta.get("last_reinforced", ""), "%Y-%m-%d").date()
                delta_days = max(0, (today - last_date).days)
            except Exception:
                delta_days = 0

            decay_factor = math.exp(-0.05 * delta_days)

            # Rule A: Superseded notes or severely decayed un-reinforced notes
            if status == "superseded" or (delta_days >= 60 and count == 1 and decay_factor < 0.05):
                archived.append({
                    "path": p,
                    "stem": p.stem,
                    "reason": "superseded" if status == "superseded" else "expired_decay",
                    "statement": meta.get("statement", p.stem)
                })
                if not dry_run:
                    dest = self.archive_dir / p.name
                    shutil.move(str(p), str(dest))
                continue

            # Rule B: High-confidence permanent rules eligible for promotion
            if count >= 3 or (conf >= 0.95 and count >= 2):
                promoted.append({
                    "path": p,
                    "stem": p.stem,
                    "count": count,
                    "confidence": conf,
                    "statement": meta.get("statement", p.stem)
                })

            active_remaining.append(meta)

        # 2. Near-Duplicate Merging across remaining active notes
        seen_pairs = set()
        for i in range(len(active_remaining)):
            for j in range(i + 1, len(active_remaining)):
                m1 = active_remaining[i]
                m2 = active_remaining[j]
                stmt1 = m1.get("statement", "").lower()
                stmt2 = m2.get("statement", "").lower()
                if not stmt1 or not stmt2:
                    continue

                sim = difflib.SequenceMatcher(None, stmt1, stmt2).ratio()
                if sim >= 0.78:
                    pair_key = (m1["path"].stem, m2["path"].stem)
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        # Keep note with higher reinforcement or older created
                        keeper, dupe = (m1, m2) if m1.get("reinforcement_count", 1) >= m2.get("reinforcement_count", 1) else (m2, m1)
                        merged.append({
                            "keeper": keeper["path"].name,
                            "duplicate": dupe["path"].name,
                            "similarity": round(sim * 100, 1)
                        })
                        if not dry_run:
                            # Merge count into keeper
                            new_count = keeper.get("reinforcement_count", 1) + dupe.get("reinforcement_count", 1)
                            self._update_note_frontmatter(keeper["path"], {
                                "reinforcement_count": new_count,
                                "last_reinforced": now_date
                            }, append_body_warning=f"- **Consolidation**: Merged redundant rule `[[{dupe['path'].stem}]]` on {now_date}.")
                            # Mark dupe as superseded
                            self._update_note_frontmatter(dupe["path"], {
                                "status": "superseded",
                                "superseded_by": f"[[{keeper['path'].stem}]]"
                            })

        # 3. Promote Proven Guidelines to 01-Concepts/Architecture/Project-Guidelines.md
        guidelines_path = self.guidelines_dir / "Project-Guidelines.md"
        if promoted and not dry_run:
            guidelines_content = f"""---
title: "Project Architectural Guidelines & Permanent Conventions"
type: concept/architecture
updated: {now_date}
tags:
  - architecture/guidelines
  - permanent-conventions
---

# 🏛️ Project Guidelines & Consolidated Conventions

*Promoted automatically by 2ndBrain Memory Consolidation on {now_date}.*

---

## Proven Architectural Rules & System Constraints
"""
            for prom in promoted:
                guidelines_content += f"\n### [[{prom['stem']}]]\n- **Policy**: {prom['statement']}\n- **Confidence**: {prom['confidence']} (Reinforced {prom['count']}x across sessions)\n"

            guidelines_content += f"\n---\n*Linked Meta*: [[00-Meta/Index]]\n"
            guidelines_path.write_text(guidelines_content, encoding="utf-8")

        # 4. Generate Consolidation Audit Report
        report_path = self.reports_dir / f"{now_date}_Consolidation.md"
        report_md = f"""---
title: "Memory Consolidation Report - {now_date}"
type: report/consolidation
date: {now_date}
dry_run: {str(dry_run).lower()}
---

# 💤 Memory Consolidation Audit Report ({now_date})

**Execution Mode**: `{'DRY RUN (Preview Only)' if dry_run else 'APPLIED'}`

---

## Executive Summary
- **Total Memory Rules Audited**: {len(all_notes)}
- **Archived / Pruned (Superseded & Decayed)**: {len(archived)}
- **Promoted to Permanent Guidelines**: {len(promoted)}
- **Near-Duplicate Pairs Merged**: {len(merged)}
- **Active Operational Memory Rules Remaining**: {len(active_remaining) - len(archived)}

---

## 1. Archived Rules
"""
        if archived:
            for a in archived:
                report_md += f"- 📦 `[[{a['stem']}]]`: {a['statement']} *(Reason: {a['reason']})*\n"
        else:
            report_md += "- No decayed or superseded rules required archiving.\n"

        report_md += "\n## 2. Promoted Guidelines (Durable Rules)\n"
        if promoted:
            for p in promoted:
                report_md += f"- 🌟 `[[{p['stem']}]]`: {p['statement']} *(Count: {p['count']}x, Conf: {p['confidence']})*\n"
        else:
            report_md += "- No rules currently meet the promotion threshold (reinforcement >= 3).\n"

        report_md += "\n## 3. Merged Redundancies\n"
        if merged:
            for m in merged:
                report_md += f"- 🔄 Merged `{m['duplicate']}` into `{m['keeper']}` (Similarity: {m['similarity']}%)\n"
        else:
            report_md += "- No duplicate rule pairs detected.\n"

        if not dry_run:
            report_path.write_text(report_md, encoding="utf-8")

        # Re-index vault vectors after consolidation
        if not dry_run:
            try:
                from hybrid_search import HybridSearchEngine
                engine = HybridSearchEngine(self.vault_dir)
                engine.index_vault(force=False)
            except Exception:
                pass

        return {
            "archived_count": len(archived),
            "promoted_count": len(promoted),
            "merged_count": len(merged),
            "active_count": len(active_remaining) - len(archived),
            "report_path": str(report_path),
            "dry_run": dry_run
        }

    def process_transcript_jsonl(self, transcript_path: str, session_id: str = "session") -> int:
        """Parses an entire conversation transcript (.jsonl) and extracts durable memories."""
        path = Path(transcript_path)
        if not path.exists():
            print(f"[Memory Distiller] Transcript not found: {transcript_path}")
            return 0

        crystallized_count = 0
        last_user_msg = ""

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    step = json.loads(line)
                    step_type = step.get("type", "")
                    content = step.get("content", "")

                    if step_type == "USER_INPUT" or step.get("source") == "USER_EXPLICIT":
                        last_user_msg = content
                    elif step_type == "PLANNER_RESPONSE" and last_user_msg:
                        saved = self.distill_turn(last_user_msg, content, session_id=session_id)
                        if saved:
                            crystallized_count += 1
                        last_user_msg = ""
                except Exception:
                    continue

        return crystallized_count

    def recall_relevant_memories(self, prompt: str, max_items: int = 5) -> List[str]:
        """
        Searches active memory notes for relevance to the prompt using:
        1. Status gating: Filters out 'superseded' or 'deprecated' rules.
        2. Recency half-life decay: exp(-0.05 * days_since_last_reinforced).
        3. Reinforcement boost: 1.0 + 0.2 * (count - 1).
        4. Confidence & keyword overlap.
        """
        prompt_words = set(re.findall(r"\w{3,}", prompt.lower()))
        today = datetime.now(timezone.utc).date()
        scored_memories = []

        all_notes = (
            list(self.tasks_dir.glob("*.md")) +
            list(self.decisions_dir.glob("*.md")) +
            list(self.facts_dir.glob("*.md")) +
            list(self.skills_dir.glob("*.md")) +
            list(self.corrections_dir.glob("*.md")) +
            list(self.concepts_decisions_dir.glob("*.md")) +
            list(self.sessions_dir.glob("*.md"))
        )

        for p in all_notes:
            if p.name.lower() in ("readme.md", "index.md"):
                continue
            meta = self._parse_note_metadata(p)

            # Skip superseded or archived rules
            if meta.get("status") in ("superseded", "archived"):
                continue

            content = meta.get("raw_content", "")
            content_words = set(re.findall(r"\w{3,}", content.lower()))
            overlap = len(prompt_words.intersection(content_words))

            if overlap > 0:
                # 1. Parse date for decay calculation
                try:
                    last_date = datetime.strptime(meta.get("last_reinforced", ""), "%Y-%m-%d").date()
                    delta_days = max(0, (today - last_date).days)
                except Exception:
                    delta_days = 0

                # 2. Compute exponential decay (~14-day half-life: lambda = 0.05)
                decay_factor = math.exp(-0.05 * delta_days)

                # 3. Compute reinforcement factor
                count = meta.get("reinforcement_count", 1)
                reinforce_factor = 1.0 + 0.2 * max(0, count - 1)

                # 4. Confidence
                confidence = meta.get("confidence", 0.8)

                # Composite weight
                composite_weight = overlap * confidence * decay_factor * reinforce_factor

                statement = meta.get("statement", p.stem)
                status_tag = f"[Reinforced {count}x]" if count > 1 else ""
                m_type_badge = f"[{meta.get('type', 'memory').upper()}]"
                scored_memories.append((composite_weight, f"- {m_type_badge} [[{p.stem}]] {status_tag}: {statement}".strip()))

        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in scored_memories[:max_items]]
