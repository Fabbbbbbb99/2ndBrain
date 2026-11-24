"""Antigravity skill exporter with SHA-256 lock tracking, --dry-run, --force, and orphan pruning."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from second_brain.config import BrainConfig
from second_brain.specialists.schema import SpecialistSpec, load_all_specialists
from second_brain.storage.sqlite_store import atomic_write_lf

LOCK_FILENAME = ".2ndbrain_export_lock.json"


def _render_skill_markdown(spec: SpecialistSpec) -> str:
    versions_line = ", ".join(spec.source_versions) if spec.source_versions else "any (pin explicitly when querying)"
    domains_json = json.dumps(spec.allowed_domains)
    return (
        f"---\n"
        f"name: 2ndbrain-{spec.domain_id}\n"
        f"description: {spec.description}\n"
        f"---\n\n"
        f"# {spec.display_name}\n\n"
        f"<!-- Generated from specialists/{spec.domain_id}.yaml (spec_version={spec.spec_version}). -->\n\n"
        f"## Grounding Contract\n"
        f"- **Allowed Domains:** `{domains_json}`\n"
        f"- **Pinned Versions:** `{versions_line}`\n"
        f"- **Token Budget:** `{spec.token_budget}` tokens (`tiktoken o200k_base`)\n"
        f"- **Citation Policy:** `{spec.citation_policy}`\n\n"
        f"## Execution Protocol\n"
        f"1. Call `second-brain:brain_query` (`domain_scope={domains_json}`) for concept/clause packs, or `second-brain:brain_grep` for exact literal/regex scans across `parsed/` and `wiki/`.\n"
        f"2. Call `second-brain:brain_recall` to inspect Store 2 memories (`active`, `staged`, `superseded`) filtered by `project_tag` or `ledger_type`.\n"
        f"3. On `[STALE]`, re-read upstream `parsed/` URIs via `second-brain:brain_get_clause` (also inspects `children`, `cited_by`, and `inferred_links`). On `VISUAL_CROP_REQUIRED`, inspect the crop path and call `second-brain:brain_cache_visual_repair`.\n\n"
        f"## Specialist Rules\n\n"
        f"{spec.instructions.strip()}\n"
    )


def export_antigravity_skills(
    *,
    cfg: BrainConfig,
    target_dir: Path,
    dry_run: bool = False,
    force: bool = False,
    prune: bool = True,
) -> dict[str, Any]:
    specs = load_all_specialists(cfg.vault_path)
    lock_path = target_dir / LOCK_FILENAME
    prev_lock: dict[str, dict[str, str]] = {}
    if lock_path.exists():
        prev_lock = json.loads(lock_path.read_text(encoding="utf-8"))

    new_lock: dict[str, dict[str, str]] = {}
    written: list[str] = []
    skipped_modified: list[str] = []
    pruned: list[str] = []
    active_ids = {s.domain_id for s in specs}

    for spec in specs:
        skill_dir = target_dir / f"2ndbrain-{spec.domain_id}"
        skill_file = skill_dir / "SKILL.md"
        rendered = _render_skill_markdown(spec)
        new_sha = hashlib.sha256(rendered.encode("utf-8")).hexdigest()

        if skill_file.exists():
            cur_bytes = skill_file.read_bytes().replace(b"\r\n", b"\n")
            cur_sha = hashlib.sha256(cur_bytes).hexdigest()
            prev_sha = (prev_lock.get(spec.domain_id) or {}).get("sha256")
            if prev_sha and cur_sha != prev_sha and not force:
                skipped_modified.append(str(skill_file))
                new_lock[spec.domain_id] = {"path": str(skill_file), "sha256": prev_sha}
                continue
            if not prev_sha and cur_sha != new_sha and not force:
                skipped_modified.append(str(skill_file))
                continue

        if not dry_run:
            atomic_write_lf(skill_file, rendered)
        written.append(str(skill_file))
        new_lock[spec.domain_id] = {"path": str(skill_file), "sha256": new_sha}

    if prune:
        for old_id, info in prev_lock.items():
            if old_id in active_ids:
                continue
            old_path = Path(info["path"])
            if old_path.exists():
                cur_sha = hashlib.sha256(old_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                if cur_sha != info.get("sha256") and not force:
                    skipped_modified.append(str(old_path))
                    new_lock[old_id] = info
                    continue
                if not dry_run:
                    old_path.unlink(missing_ok=True)
                    try:
                        old_path.parent.rmdir()
                    except OSError:
                        pass
                pruned.append(str(old_path))

    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_lf(lock_path, json.dumps(new_lock, indent=2))

    return {
        "harness": "antigravity",
        "dry_run": dry_run,
        "target_dir": str(target_dir),
        "written": written,
        "skipped_modified": skipped_modified,
        "pruned": pruned,
        "lock_file": str(lock_path),
    }
