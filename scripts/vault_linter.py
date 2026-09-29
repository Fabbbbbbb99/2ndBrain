"""
vault_linter.py - Vault Health Linter & Orphan Link Resolver for 2ndBrain

Audits an Obsidian markdown vault for:
1. Broken wikilinks ([[Note]] pointing to non-existent files) with fuzzy match suggestions.
2. Island / Orphan notes (zero incoming and zero outgoing links).
3. Schema / Frontmatter compliance (YAML delimiters, type, tags).
4. Index synchronization (notes on disk not listed in 00-Meta/Index.md).
5. Quantitative Vault Health Score (0 - 100).
6. Optional auto-repair (--fix) for high-confidence fuzzy link repairs and MOC registration.
"""

import os
import re
import difflib
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional


class VaultLinter:
    def __init__(self, vault_dir: str):
        self.vault_dir = Path(vault_dir).resolve()
        self.wikilink_pattern = re.compile(r"\[\[(.*?)\]\]")
        self.frontmatter_pattern = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

    def _get_all_markdown_files(self) -> List[Path]:
        """Returns all markdown files in the vault, ignoring hidden/system files."""
        files = []
        for p in self.vault_dir.rglob("*.md"):
            if any(part.startswith(".") for part in p.parts):
                continue
            files.append(p)
        return sorted(files)

    def _extract_links_and_frontmatter(self, file_path: Path) -> Tuple[Set[str], Dict[str, Any], bool]:
        """Extracts wikilinks, parsed frontmatter, and frontmatter validity flag."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return set(), {}, False

        # Extract wikilinks
        raw_links = self.wikilink_pattern.findall(content)
        clean_links = set()
        for link in raw_links:
            # Handle aliases: [[TargetNote|Display Name]] or [[TargetNote#Section]]
            target = link.split("|")[0].split("#")[0].strip()
            if target:
                clean_links.add(target)

        # Extract frontmatter
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

        return clean_links, fm_dict, has_valid_fm

    def audit(self, auto_fix: bool = False) -> Dict[str, Any]:
        """
        Runs comprehensive vault health audit.
        Returns detailed diagnostics and health metrics.
        """
        all_files = self._get_all_markdown_files()
        # Collect all stem names and relative path representations (without .md)
        all_stems: Set[str] = {f.stem for f in all_files}
        stem_to_file: Dict[str, Path] = {f.stem.lower(): f for f in all_files}
        
        # Also register relative paths (e.g. "00-Meta/System-Rules") and folder names with READMEs/Index
        rel_paths_no_ext: Set[str] = {
            f.relative_to(self.vault_dir).with_suffix("").as_posix().lower()
            for f in all_files
        }
        vault_folders: Set[str] = {
            p.relative_to(self.vault_dir).as_posix().lower()
            for p in self.vault_dir.rglob("*") if p.is_dir()
        }

        # Track incoming / outgoing graph edges
        outgoing_edges: Dict[Path, Set[str]] = {}
        incoming_edges: Dict[str, Set[Path]] = {f.stem: set() for f in all_files}
        broken_links: List[Dict[str, Any]] = []
        schema_issues: List[Dict[str, Any]] = []
        fixed_links_count = 0
        fixed_orphans_count = 0

        # Scan each file
        for file_path in all_files:
            links, fm_dict, has_valid_fm = self._extract_links_and_frontmatter(file_path)
            outgoing_edges[file_path] = links

            # Check schema compliance (ignore top-level README.md and system rules)
            if file_path.name.lower() not in ("readme.md", "index.md", "index.local.md"):
                if not has_valid_fm:
                    schema_issues.append({
                        "file": file_path.relative_to(self.vault_dir),
                        "issue": "Missing YAML frontmatter (--- header ---)"
                    })
                elif "type" not in fm_dict and "tags" not in fm_dict:
                    schema_issues.append({
                        "file": file_path.relative_to(self.vault_dir),
                        "issue": "Frontmatter missing 'type' or 'tags' metadata"
                    })

            # Check links
            for target in links:
                target_clean = target.replace("\\", "/").strip("/")
                target_base = target_clean.split("/")[-1]

                # Match by exact stem, case-insensitive stem, relative path, or folder link
                if target in all_stems or target_base in all_stems:
                    matched_stem = target if target in all_stems else target_base
                    incoming_edges[matched_stem].add(file_path)
                elif target_base.lower() in stem_to_file:
                    correct_stem = stem_to_file[target_base.lower()].stem
                    incoming_edges[correct_stem].add(file_path)
                elif target_clean.lower() in rel_paths_no_ext:
                    # Valid relative path link
                    pass
                elif target_clean.lower() in vault_folders:
                    # Link to folder (e.g. [[01-Concepts]] or [[04-Agent-Memory]])
                    pass
                elif target_clean.lower() in ("note name", "folder/note name", "wikilinks", "concept-name", "targetnote"):
                    # Documentation syntax example placeholders
                    pass
                else:
                    # Broken link - find suggestions
                    candidates = difflib.get_close_matches(target_base, list(all_stems), n=2, cutoff=0.7)
                    suggestion = candidates[0] if candidates else None
                    broken_links.append({
                        "source_file": file_path,
                        "broken_target": target,
                        "suggestion": suggestion
                    })

        # Auto-fix broken links if requested
        if auto_fix and broken_links:
            for bl in broken_links:
                if bl["suggestion"]:
                    src = bl["source_file"]
                    try:
                        content = src.read_text(encoding="utf-8")
                        pattern = rf"\[\[{re.escape(bl['broken_target'])}(\|.*?)?\]\]"
                        
                        def repl(m):
                            alias = m.group(1) or ""
                            return f"[[{bl['suggestion']}{alias}]]"

                        new_content = re.sub(pattern, repl, content)
                        if new_content != content:
                            src.write_text(new_content, encoding="utf-8")
                            fixed_links_count += 1
                    except Exception:
                        pass

        # Identify Island / Orphan notes (0 incoming, 0 outgoing, excluding README/Index)
        orphan_notes: List[Path] = []
        for file_path in all_files:
            stem = file_path.stem
            if file_path.name.lower() in ("readme.md", "index.md", "system-rules.md"):
                continue
            is_island = len(outgoing_edges.get(file_path, set())) == 0 and len(incoming_edges.get(stem, set())) == 0
            if is_island:
                orphan_notes.append(file_path)

        # Check Index.md coverage
        index_file = self.vault_dir / "00-Meta" / "Index.md"
        local_index = self.vault_dir / "00-Meta" / "Index.local.md"
        active_index = local_index if local_index.exists() else index_file
        unindexed_notes: List[Path] = []
        if active_index.exists():
            index_content = active_index.read_text(encoding="utf-8", errors="ignore")
            index_links = set(self.wikilink_pattern.findall(index_content))
            for file_path in all_files:
                if file_path.name.lower() in ("readme.md", "index.md", "index.local.md", "system-rules.md"):
                    continue
                # If note belongs to 01-Concepts or 03-Sources but not linked in index
                rel_parts = file_path.relative_to(self.vault_dir).parts
                if len(rel_parts) > 1 and rel_parts[0] in ("01-Concepts", "03-Sources"):
                    if file_path.stem not in index_links:
                        unindexed_notes.append(file_path)

        # Auto-fix: register orphans or unindexed into Index.md
        if auto_fix and unindexed_notes and active_index.exists():
            try:
                idx_text = active_index.read_text(encoding="utf-8")
                append_lines = ["\n\n### 📦 Auto-Discovered Unindexed Notes\n"]
                for un in unindexed_notes[:20]:
                    append_lines.append(f"- [[{un.stem}]] (`{un.parent.name}`)")
                idx_text += "\n".join(append_lines)
                active_index.write_text(idx_text, encoding="utf-8")
                fixed_orphans_count = len(unindexed_notes)
            except Exception:
                pass

        # Calculate quantitative health score (0 - 100)
        total_files_count = len(all_files) or 1
        broken_penalty = min(40, len(broken_links) * 4)
        orphan_penalty = min(30, (len(orphan_notes) / total_files_count) * 50)
        schema_penalty = min(30, (len(schema_issues) / total_files_count) * 30)
        health_score = max(0, round(100 - broken_penalty - orphan_penalty - schema_penalty))

        return {
            "health_score": health_score,
            "total_files": len(all_files),
            "broken_links": broken_links,
            "orphan_notes": orphan_notes,
            "unindexed_notes": unindexed_notes,
            "schema_issues": schema_issues,
            "fixed_links": fixed_links_count,
            "fixed_orphans": fixed_orphans_count,
            "vault_path": str(self.vault_dir)
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
            f"🧠 2ndBrain Vault Health Audit Report",
            "=" * 60,
            f"Vault: {results['vault_path']}",
            f"Total Markdown Notes: {results['total_files']}",
            f"Overall Health Score: {score}/100 [{status_badge}]",
            "-" * 60
        ]

        # 1. Broken Links
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

        # 2. Island / Orphan Notes
        orphans = results["orphan_notes"]
        lines.append(f"\n2. 🏝️ Island Notes (0 in / 0 out): {len(orphans)}")
        if orphans:
            for o in orphans[:8]:
                lines.append(f"   ⚠️ `{o.relative_to(self.vault_dir)}`")
            if len(orphans) > 8:
                lines.append(f"   ... and {len(orphans) - 8} more island notes.")
        else:
            lines.append("   ✅ No isolated island notes detected.")

        # 3. Unindexed Notes
        unindexed = results["unindexed_notes"]
        lines.append(f"\n3. 📇 Unindexed Notes (Missing from Index.md): {len(unindexed)}")
        if unindexed:
            for u in unindexed[:8]:
                lines.append(f"   📝 `{u.relative_to(self.vault_dir)}`")
            if len(unindexed) > 8:
                lines.append(f"   ... and {len(unindexed) - 8} more unindexed notes.")
        else:
            lines.append("   ✅ All concept/source notes are mapped in Index.md!")

        # 4. Schema Compliance
        schemas = results["schema_issues"]
        lines.append(f"\n4. 📋 Frontmatter Schema Issues: {len(schemas)}")
        if schemas:
            for s in schemas[:6]:
                lines.append(f"   ⚠️ `{s['file']}`: {s['issue']}")
            if len(schemas) > 6:
                lines.append(f"   ... and {len(schemas) - 6} more schema warnings.")
        else:
            lines.append("   ✅ All markdown notes have valid YAML frontmatter.")

        # Fix summary if run with --fix
        if results.get("fixed_links", 0) > 0 or results.get("fixed_orphans", 0) > 0:
            lines.append("\n" + "-" * 60)
            lines.append(f"🛠️ Auto-Fix Actions Applied:")
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
    vault = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "../vault_template"
    auto_fix = "--fix" in sys.argv
    linter = VaultLinter(vault)
    report = linter.audit(auto_fix=auto_fix)
    print(linter.format_report(report))
