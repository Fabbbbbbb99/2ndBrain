"""Append-only YAML-frontmatter Markdown ledger and per-item staging inbox."""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import frontmatter

from second_brain.storage.models import MemoryRecord
from second_brain.storage.sqlite_store import atomic_write_lf


def new_memory_id() -> str:
    ts = int(time.time() * 1000)
    return f"mem_{ts}_{uuid.uuid4().hex[:8]}"


def append_ledger_entry(vault_path: Path, rec: MemoryRecord) -> Path:
    """Append a certified or explicit memory record to memory/<ledger_type>.md without mutating prior rows."""
    target = vault_path / "memory" / f"{rec.ledger_type}.md"
    existing = ""
    if target.exists():
        existing = target.read_text(encoding="utf-8").replace("\r\n", "\n")
    else:
        existing = f"---\nledger_type: {rec.ledger_type}\nschema_version: \"1.0.0\"\n---\n# {rec.ledger_type.title()}\n"

    block = (
        f"\n## [{rec.id}] {rec.domain} ({rec.status})\n"
        f"- **project_tag:** `{rec.project_tag or 'none'}`\n"
        f"- **effective_from:** `{rec.effective_from}`\n"
        f"- **effective_to:** `{rec.effective_to or 'open'}`\n"
        f"- **version_scope:** `{rec.version_scope or 'global'}`\n"
        f"- **supersedes_id:** `{rec.supersedes_id or 'none'}`\n"
        f"- **source_ref:** `{rec.source_ref}`\n"
        f"- **created_at:** `{rec.created_at}`\n\n"
        f"{rec.content.strip()}\n"
    )
    atomic_write_lf(target, existing.rstrip("\n") + "\n" + block)
    return target


def write_inbox_item(vault_path: Path, rec: MemoryRecord) -> Path:
    """Write a standalone staging file memory/inbox/<id>.md with strict YAML frontmatter."""
    inbox_dir = vault_path / "memory" / "inbox"
    inbox_dir.mkdir(parents=True, exist_ok=True)
    path = inbox_dir / f"{rec.id}.md"
    post = frontmatter.Post(
        rec.content.strip() + "\n",
        id=rec.id,
        domain=rec.domain,
        ledger_type=rec.ledger_type,
        project_tag=rec.project_tag,
        effective_from=rec.effective_from,
        effective_to=rec.effective_to,
        version_scope=rec.version_scope,
        status=rec.status,
        supersedes_id=rec.supersedes_id,
        source_ref=rec.source_ref,
        reason_code=rec.reason_code,
        laya_advisory=rec.laya_advisory,
        created_at=rec.created_at,
    )
    atomic_write_lf(path, frontmatter.dumps(post) + "\n")
    return path


def read_inbox_items(vault_path: Path) -> list[tuple[Path, MemoryRecord]]:
    inbox_dir = vault_path / "memory" / "inbox"
    if not inbox_dir.exists():
        return []
    out: list[tuple[Path, MemoryRecord]] = []
    for md in sorted(inbox_dir.glob("*.md")):
        post = frontmatter.loads(md.read_text(encoding="utf-8"))
        meta = dict(post.metadata)
        meta["content"] = post.content.strip()
        out.append((md, MemoryRecord(**meta)))
    return out
