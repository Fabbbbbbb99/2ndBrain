"""
vault_linter.py - Vault Health Linter & Orphan Link Resolver for 2ndBrain

Audits an Obsidian markdown vault for:
1. Broken wikilinks ([[Note]] pointing to non-existent files) with cached fuzzy match suggestions.
2. Island / Orphan notes (zero incoming and zero outgoing links).
3. Schema / Frontmatter compliance (YAML delimiters, type, tags).
4. Index synchronization (MOC hubs on disk not listed in 00-Meta/Index.md or Index.local.md).
5. Automatic reconciliation of 03-Agent-Memory/Agent-Memory-MOC.md against local 4-Pillar files.
6. Quantitative Vault Health Score (0 - 100).
"""

import difflib
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class VaultLinter:
    EXCLUDED_META_NAMES = (
        "readme.md",
        "index.md",
        "index.local.md",
        "system-rules.md",
        "agent-memory-moc.md",
    )

    def __init__(self, vault_dir: str):
        self.vault_dir = Path(vault_dir).resolve()
        self.wikilink_pattern = re.compile(r"\[\[(.*?)\]\]")
        self.frontmatter_pattern = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

    def _get_all_markdown_files(self) -> List[Path]:
        """Returns all markdown files in the vault, ignoring hidden/system files andTemplates."""
        files = []
        for p in self.vault_dir.rglob("*.md"):
            rel_parts = p.relative_to(self.vault_dir).parts
            if any(part.startswith(".") for part in rel_parts):
                continue
            if "Templates" in rel_parts:
                continue
            files.append(p)
        return sorted(files)

    def _extract_links_and_frontmatter(
        self, file_path: Path
    ) -> Tuple[Set[str], Dict[str, Any], bool, str]:
        """Extracts wikilinks, parsed frontmatter, frontmatter validity flag, and inspectable text."""
        is_source_extract = file_path.name.endswith("-Source.md")
        try:
            if is_source_extract:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read(1536)
            else:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return set(), {}, False, ""

        fm_match = self.frontmatter_pattern.match(content)
        has_valid_fm = False
        fm_dict: Dict[str, Any] = {}
        if fm_match:
            has_valid_fm = True
            fm_text = fm_match.group(1)
            for line in fm_text.splitlines():
                if ":" in line and not line.strip().startswith("-"):
                    k, v = line.split(":", 1)
                    fm_dict[k.strip().lower()] = v.strip()

        if is_source_extract and "## Verbatim Extraction" in content:
            content = content.split("## Verbatim Extraction", 1)[0]

        # For generated -Overview.md notes, exclude Section 3 verbatim formulas/code from wikilink parsing
        link_scan_text = content
        if file_path.name.endswith("-Overview.md") and "## 3. Normative Technical Content" in link_scan_text:
            parts = link_scan_text.split("## 3. Normative Technical Content", 1)
            tail = ""
            if "## 🔗 Actionable Navigation" in parts[1]:
                tail = "## 🔗 Actionable Navigation" + parts[1].split("## 🔗 Actionable Navigation", 1)[1]
            link_scan_text = parts[0] + "\n" + tail

        content_no_code = re.sub(r"```.*?```", "", link_scan_text, flags=re.DOTALL)
        content_no_code = re.sub(r"`[^`\n]+`", "", content_no_code)
        content_no_code = re.sub(r"\.\. mermaid::.*?(?=\n\S|\Z)", "", content_no_code, flags=re.DOTALL)

        raw_links = self.wikilink_pattern.findall(content_no_code)
        clean_links = set()
        for link in raw_links:
            if "][" in link or link.startswith("[") or link.endswith("]"):
                continue
            target = link.split("|")[0].split("#")[0].strip()
            if not target or any(ch in target for ch in ("\n", "\r", "$", "{", "}", "=", ";", "(", ")")):
                continue
            clean_links.add(target)

        return clean_links, fm_dict, has_valid_fm, content

    def reconcile_agent_memory_moc(self) -> int:
        """
        Synchronizes 03-Agent-Memory/Agent-Memory-MOC.md with actual local markdown files
        in 01-Tasks, 02-Decisions, 03-Facts, and 04-Skills.
        Removes broken links to non-existent local files and registers untracked local notes.
        """
        memory_dir = self.vault_dir / "03-Agent-Memory"
        moc_file = memory_dir / "Agent-Memory-MOC.md"
        if not memory_dir.exists() or not moc_file.exists():
            return 0

        pillars = [
            (
                "01-Tasks",
                "### 1. 📋 [[01-Tasks]]",
                "*Actionable work items, sprint backlogs, milestone checklists, and execution states.*",
            ),
            (
                "02-Decisions",
                "### 2. ⚖️ [[02-Decisions]]",
                "*Architectural Decision Records (ADRs), technical trade-offs, and design rationales.*",
            ),
            (
                "03-Facts",
                "### 3. 📌 [[03-Facts]]",
                "*Permanent system invariants, hardware configurations, and mathematical conventions.*",
            ),
            (
                "04-Skills",
                "### 4. 🛠️ [[04-Skills]]",
                "*Reusable execution recipes, debugging procedures, and procedural workflows.*",
            ),
        ]

        lines = [
            "---",
            'title: "Agent Memory Map of Content (MOC)"',
            "type: moc",
            "tags:",
            "  - se-brain/meta",
            "  - agent-memory",
            "  - moc",
            "---",
            "",
            "# 🧠 Agent Memory Map of Content (MOC)",
            "",
            "Map of Content for the local-first persistent memory engine. Structured across **4 Core Memory Pillars**:",
            "",
            "---",
            "",
            "## 🏛️ The 4 Memory Pillars",
            "",
        ]

        total_registered = 0
        for folder_name, heading, desc in pillars:
            lines.append(heading)
            lines.append(desc)
            p_dir = memory_dir / folder_name
            if p_dir.exists():
                notes = sorted(
                    [f for f in p_dir.glob("*.md") if f.name.lower() != "readme.md"],
                    key=lambda x: x.name,
                )
                for note in notes:
                    _, fm_dict, _, _ = self._extract_links_and_frontmatter(note)
                    raw_title = fm_dict.get("title") or note.stem.replace("_", " ").title()
                    clean_title = re.sub(r'["\'\n\r]', "", str(raw_title)).strip()
                    lines.append(f"- [[{note.stem}|{clean_title}]]")
                    total_registered += 1
            lines.append("")

        lines.extend(
            [
                "---",
                "",
                "## 📦 Auxiliary Memory Repositories",
                "- **`Sessions/`**: Compacted historical session transcripts and multi-agent dialogue archives.",
                "- **`Consolidation-Reports/`**: Audit logs generated during periodic memory consolidation.",
                "- **`Archive/`**: Superseded and decayed rules preserved for historical reference.",
                "",
                "---",
                "[[00-Meta/Index|Index]] | [[03-Agent-Memory/Agent-Memory-MOC|Agent-Memory-MOC]]",
                "",
            ]
        )

        new_content = "\n".join(lines)
        try:
            old_content = moc_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            old_content = ""

        if new_content != old_content:
            moc_file.write_text(new_content, encoding="utf-8")

        return total_registered

    def audit(self, auto_fix: bool = False) -> Dict[str, Any]:
        """Runs vault health audit and returns detailed diagnostics and health metrics."""
        # Always keep Agent-Memory-MOC.md synchronized with local 4-Pillar notes on disk
        self.reconcile_agent_memory_moc()

        all_files = self._get_all_markdown_files()
        all_stems: Set[str] = {f.stem for f in all_files}
        all_stems_list: List[str] = sorted(all_stems)
        stem_to_file: Dict[str, Path] = {f.stem.lower(): f for f in all_files}

        rel_path_to_stem: Dict[str, str] = {}
        vault_folders: Set[str] = {
            "00-meta",
            "01-concepts",
            "02-sources",
            "03-agent-memory",
            "01-tasks",
            "02-decisions",
            "03-facts",
            "04-skills",
            "sessions",
            "archive",
            "consolidation-reports",
        }
        for f in all_files:
            rel_no_ext = f.relative_to(self.vault_dir).with_suffix("")
            rel_path_to_stem[rel_no_ext.as_posix().lower()] = f.stem
            for parent in rel_no_ext.parents:
                if parent.name:
                    vault_folders.add(parent.as_posix().lower())
                    vault_folders.add(parent.name.lower())

        slop_regex = re.compile(
            r"\b(?:here'?s the thing|in today'?s fast-paced|it'?s not (?:just )?about [^,\n]+, it'?s about|the future isn'?t coming|let me be clear|hope this helps|without further ado)\b",
            re.IGNORECASE,
        )

        outgoing_edges: Dict[Path, Set[str]] = {}
        incoming_edges: Dict[str, Set[Path]] = {f.stem: set() for f in all_files}
        broken_links: List[Dict[str, Any]] = []
        schema_issues: List[Dict[str, Any]] = []
        suggestion_cache: Dict[str, Optional[str]] = {}
        fixed_links_count = 0
        fixed_orphans_count = 0
        use_fast_triplet_index = len(all_files) > 1000

        for file_path in all_files:
            stem = file_path.stem
            # Fast path for verified 3-note ingested triplets in large vaults (avoids 41k Windows Defender CreateFile hooks)
            if use_fast_triplet_index and stem.endswith(("-Overview", "-MOC", "-Source")):
                if stem.endswith("-Overview"):
                    base_slug = stem[: -len("-Overview")]
                    moc_stem = f"{base_slug}-MOC"
                    src_stem = f"{base_slug}-Source"
                    if moc_stem in all_stems and src_stem in all_stems:
                        outgoing_edges[file_path] = {moc_stem, src_stem, "Index"}
                        incoming_edges[moc_stem].add(file_path)
                        incoming_edges[src_stem].add(file_path)
                        if "Index" in incoming_edges:
                            incoming_edges["Index"].add(file_path)
                        continue
                elif stem.endswith("-MOC"):
                    base_slug = stem[: -len("-MOC")]
                    ov_stem = f"{base_slug}-Overview"
                    src_stem = f"{base_slug}-Source"
                    if ov_stem in all_stems and src_stem in all_stems:
                        outgoing_edges[file_path] = {ov_stem, src_stem, "Index"}
                        incoming_edges[ov_stem].add(file_path)
                        incoming_edges[src_stem].add(file_path)
                        if "Index" in incoming_edges:
                            incoming_edges["Index"].add(file_path)
                        continue
                elif stem.endswith("-Source"):
                    base_slug = stem[: -len("-Source")]
                    ov_stem = f"{base_slug}-Overview"
                    moc_stem = f"{base_slug}-MOC"
                    if ov_stem in all_stems and moc_stem in all_stems:
                        outgoing_edges[file_path] = set()
                        continue

            links, fm_dict, has_valid_fm, raw_text = self._extract_links_and_frontmatter(file_path)
            outgoing_edges[file_path] = links

            if file_path.name.lower() not in self.EXCLUDED_META_NAMES:
                if not has_valid_fm:
                    schema_issues.append(
                        {
                            "file": file_path.relative_to(self.vault_dir),
                            "issue": "Missing YAML frontmatter (--- header ---)",
                        }
                    )
                elif "type" not in fm_dict and "tags" not in fm_dict and "category" not in fm_dict:
                    schema_issues.append(
                        {
                            "file": file_path.relative_to(self.vault_dir),
                            "issue": "Frontmatter missing 'type' or 'tags' metadata",
                        }
                    )

                if raw_text and not file_path.name.endswith(("-Overview.md", "-MOC.md")):
                    slop_matches = slop_regex.findall(raw_text)
                    if slop_matches:
                        schema_issues.append(
                            {
                                "file": file_path.relative_to(self.vault_dir),
                                "issue": f"Contains prohibited AI slop phrases: {len(slop_matches)} detected",
                            }
                        )

            for target in links:
                target_clean = target.replace("\\", "/").strip("/")
                target_base = target_clean.split("/")[-1]

                if target in all_stems or target_base in all_stems:
                    matched_stem = target if target in all_stems else target_base
                    incoming_edges[matched_stem].add(file_path)
                elif target_base.lower() in stem_to_file:
                    correct_stem = stem_to_file[target_base.lower()].stem
                    incoming_edges[correct_stem].add(file_path)
                elif target_clean.lower() in rel_path_to_stem:
                    matched_stem = rel_path_to_stem[target_clean.lower()]
                    incoming_edges[matched_stem].add(file_path)
                elif target_clean.lower() in vault_folders or target_base.lower() in vault_folders:
                    pass
                elif (self.vault_dir / f"{target_clean}.md").exists():
                    pass
                elif target_clean.lower() in (
                    "note name",
                    "folder/note name",
                    "wikilinks",
                    "concept-name",
                    "targetnote",
                    "index.local",
                    "00-meta/index.local",
                ):
                    pass
                else:
                    if target_base not in suggestion_cache:
                        t_len = len(target_base)
                        t_first = target_base[:1].lower()
                        shortlist = [
                            s
                            for s in all_stems_list
                            if abs(len(s) - t_len) <= 6 and s[:1].lower() == t_first
                        ]
                        candidates = difflib.get_close_matches(
                            target_base, shortlist or all_stems_list[:500], n=1, cutoff=0.7
                        )
                        suggestion_cache[target_base] = candidates[0] if candidates else None
                    broken_links.append(
                        {
                            "source_file": file_path,
                            "broken_target": target,
                            "suggestion": suggestion_cache[target_base],
                        }
                    )

        if auto_fix and broken_links:
            still_broken: List[Dict[str, Any]] = []
            for bl in broken_links:
                if bl["suggestion"]:
                    src = bl["source_file"]
                    try:
                        content = src.read_text(encoding="utf-8")
                        pattern = rf"\[\[{re.escape(bl['broken_target'])}(\|.*?)?\]\]"

                        def repl(m, sugg=bl["suggestion"]):
                            alias = m.group(1) or ""
                            return f"[[{sugg}{alias}]]"

                        new_content = re.sub(pattern, repl, content)
                        if new_content != content:
                            src.write_text(new_content, encoding="utf-8")
                            fixed_links_count += 1
                            incoming_edges.setdefault(bl["suggestion"], set()).add(src)
                            continue
                    except Exception:
                        pass
                still_broken.append(bl)
            broken_links = still_broken

        orphan_notes: List[Path] = []
        for file_path in all_files:
            stem = file_path.stem
            if file_path.name.lower() in self.EXCLUDED_META_NAMES:
                continue
            is_island = (
                len(outgoing_edges.get(file_path, set())) == 0
                and len(incoming_edges.get(stem, set())) == 0
            )
            if is_island:
                orphan_notes.append(file_path)

        # Check Index.md + Index.local.md coverage for MOC hubs
        index_file = self.vault_dir / "00-Meta" / "Index.md"
        local_index = self.vault_dir / "00-Meta" / "Index.local.md"
        active_index = local_index if local_index.exists() else index_file

        index_stems_lower: Set[str] = set()
        for idx_path in (index_file, local_index):
            if idx_path.exists():
                idx_links, _, _, _ = self._extract_links_and_frontmatter(idx_path)
                for lk in idx_links:
                    lk_clean = lk.replace("\\", "/").strip("/")
                    index_stems_lower.add(lk_clean.lower())
                    index_stems_lower.add(lk_clean.split("/")[-1].lower())

        unindexed_notes: List[Path] = []
        for file_path in all_files:
            if file_path.name.lower() in self.EXCLUDED_META_NAMES:
                continue
            rel_parts = file_path.relative_to(self.vault_dir).parts
            if len(rel_parts) > 1 and rel_parts[0] in ("01-Concepts", "02-Sources"):
                # -Overview.md and -Source.md belong under their -MOC.md hub
                if file_path.name.endswith(("-Overview.md", "-Source.md")):
                    if len(incoming_edges.get(file_path.stem, set())) > 0:
                        continue
                # Check if linked in Index.md / Index.local.md or linked by another indexed note
                if (
                    file_path.stem.lower() not in index_stems_lower
                    and len(incoming_edges.get(file_path.stem, set())) == 0
                ):
                    unindexed_notes.append(file_path)

        if auto_fix and unindexed_notes and active_index.exists():
            try:
                idx_text = active_index.read_text(encoding="utf-8")
                new_entries: List[str] = []
                for un in unindexed_notes:
                    if f"[[{un.stem}" not in idx_text:
                        new_entries.append(f"- [[{un.stem}]] (`{un.parent.name}`)")
                if new_entries:
                    if "### 📦 Auto-Discovered Unindexed Notes" not in idx_text:
                        idx_text = idx_text.rstrip() + "\n\n### 📦 Auto-Discovered Unindexed Notes\n"
                    idx_text = idx_text.rstrip() + "\n" + "\n".join(new_entries) + "\n"
                    active_index.write_text(idx_text, encoding="utf-8")
                    fixed_orphans_count = len(new_entries)
                    unindexed_notes = []
            except Exception:
                pass

        total_files_count = len(all_files) or 1
        broken_penalty = min(40, len(broken_links) * 4)
        orphan_penalty = min(30, (len(orphan_notes) / total_files_count) * 50)
        schema_penalty = min(30, (len(schema_issues) / total_files_count) * 30)
        health_score = float(max(0, round(100 - broken_penalty - orphan_penalty - schema_penalty, 1)))

        return {
            "health_score": health_score,
            "total_files": len(all_files),
            "broken_links": broken_links,
            "orphan_notes": orphan_notes,
            "orphans": orphan_notes,
            "unindexed_notes": unindexed_notes,
            "schema_issues": schema_issues,
            "fixed_links": fixed_links_count,
            "fixed_orphans": fixed_orphans_count,
            "vault_path": str(self.vault_dir),
        }

    def format_report(self, results: Dict[str, Any]) -> str:
        """Formats the audit results into a clean markdown / terminal report."""
        score = results["health_score"]
        if score >= 90:
            status_badge = "🟢 EXCELLENT"
        elif score >= 70:
            status_badge = "🟡 GOOD (Minor Issues)"
        else:
            status_badge = "🔴 NEEDS ATTENTION"

        lines = [
            "=" * 60,
            "🧠 2ndBrain Vault Health Audit Report",
            "=" * 60,
            f"Vault: {results['vault_path']}",
            f"Total Markdown Notes: {results['total_files']}",
            f"Overall Health Score: {score}/100 [{status_badge}]",
            "-" * 60,
        ]

        broken = results["broken_links"]
        lines.append(f"\n1. 🔗 Broken Wikilinks: {len(broken)}")
        if broken:
            for b in broken[:10]:
                src = b["source_file"].relative_to(self.vault_dir)
                sugg = f" (Did you mean: [[{b['suggestion']}]])" if b["suggestion"] else " (No suggestion)"
                lines.append(f"   ❌ [[{b['broken_target']}]] in `{src}`{sugg}")
            if len(broken) > 10:
                lines.append(f"   ... and {len(broken) - 10} more broken links.")
        else:
            lines.append("   ✅ All wikilinks resolve to valid files!")

        orphans = results["orphan_notes"]
        lines.append(f"\n2. 🏝️ Island Notes (0 in / 0 out): {len(orphans)}")
        if orphans:
            for o in orphans[:8]:
                lines.append(f"   ⚠️ `{o.relative_to(self.vault_dir)}`")
            if len(orphans) > 8:
                lines.append(f"   ... and {len(orphans) - 8} more island notes.")
        else:
            lines.append("   ✅ No isolated island notes detected.")

        unindexed = results["unindexed_notes"]
        lines.append(f"\n3. 📇 Unindexed Notes (Missing from Index.md): {len(unindexed)}")
        if unindexed:
            for u in unindexed[:8]:
                lines.append(f"   📝 `{u.relative_to(self.vault_dir)}`")
            if len(unindexed) > 8:
                lines.append(f"   ... and {len(unindexed) - 8} more unindexed notes.")
        else:
            lines.append("   ✅ All concept/source notes are mapped in Index.md!")

        schemas = results["schema_issues"]
        lines.append(f"\n4. 📋 Frontmatter Schema Issues: {len(schemas)}")
        if schemas:
            for s in schemas[:6]:
                lines.append(f"   ⚠️ `{s['file']}`: {s['issue']}")
            if len(schemas) > 6:
                lines.append(f"   ... and {len(schemas) - 6} more schema warnings.")
        else:
            lines.append("   ✅ All markdown notes have valid YAML frontmatter.")

        if results.get("fixed_links", 0) > 0 or results.get("fixed_orphans", 0) > 0:
            lines.append("\n" + "-" * 60)
            lines.append("🛠️ Auto-Fix Actions Applied:")
            lines.append(f"   - Fuzzy-repaired broken links: {results['fixed_links']}")
            lines.append(f"   - Registered unindexed notes into Index.md: {results['fixed_orphans']}")

        lines.append("=" * 60)
        return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    def _find_default_vault():
        env_vault = os.environ.get("SECOND_BRAIN_VAULT")
        if env_vault and Path(env_vault).exists():
            return env_vault
        env_root = os.environ.get("KNOWLEDGE_BASE_ROOT")
        if env_root and (Path(env_root) / "vault").exists():
            return str(Path(env_root) / "vault")
        sp = Path(__file__).resolve().parent.parent
        for p in [sp / "vault", sp / "vault_template", Path("vault"), Path("vault_template")]:
            if p.exists() and p.is_dir():
                return str(p)
        return str(sp / "vault")

    default_vault = _find_default_vault()
    vault = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else default_vault
    auto_fix = "--fix" in sys.argv
    linter = VaultLinter(vault)
    report = linter.audit(auto_fix=auto_fix)
    print(linter.format_report(report))
