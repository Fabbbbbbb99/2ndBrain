#!/usr/bin/env python3
"""
gsd_task.py - GSD (Get Stuff Done) Task Engine for 2ndBrain.
Enforces Rule 6 (ADHD 5-Step Ceiling) and Rule 8 (GSD Atomic Work-Units).
"""

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_vault_dir(vault_dir: Optional[str] = None) -> Path:
    if vault_dir:
        return Path(vault_dir).resolve()
    env_vault = os.environ.get("SECOND_BRAIN_VAULT")
    if env_vault and Path(env_vault).exists():
        return Path(env_vault).resolve()
    env_root = os.environ.get("KNOWLEDGE_BASE_ROOT")
    if env_root and (Path(env_root) / "vault").exists():
        return (Path(env_root) / "vault").resolve()
    if (REPO_ROOT / "vault").exists():
        return (REPO_ROOT / "vault").resolve()
    return (REPO_ROOT / "vault_template").resolve()


def slugify(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text).strip()
    return re.sub(r"[-\s]+", "_", s).lower()[:50]


def _register_task_in_moc(vault_path: Path, task_path: Path, title: str):
    moc_file = vault_path / "03-Agent-Memory" / "Agent-Memory-MOC.md"
    if not moc_file.exists():
        return
    try:
        content = moc_file.read_text(encoding="utf-8")
        stem = task_path.stem
        if f"[[{stem}|" in content or f"[[{stem}]]" in content:
            return
        clean_title = f"TASK: {title}".replace('"', "").strip()
        link_entry = f"- [[{stem}|{clean_title}]]"
        for marker in ("### 1. 📋 [[01-Tasks]]", "### 🎯 Active & Completed Tasks"):
            if marker in content:
                marker_idx = content.find(marker)
                next_nl = content.find("\n\n", marker_idx)
                if next_nl != -1:
                    new_content = content[:next_nl] + f"\n{link_entry}" + content[next_nl:]
                else:
                    new_content = content.replace(marker, f"{marker}\n{link_entry}")
                moc_file.write_text(new_content, encoding="utf-8")
                return
    except Exception:
        pass


def create_task(
    title: str,
    goal: str,
    timebox: str = "30m",
    target_files: str = "",
    steps: Optional[List[str]] = None,
    vault_dir: Optional[str] = None,
) -> Path:
    vault_path = _resolve_vault_dir(vault_dir)
    tasks_dir = vault_path / "03-Agent-Memory" / "01-Tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    slug = slugify(title)
    filename = f"{date_str}_{slug}.md"
    file_path = tasks_dir / filename

    action_items = (steps or [
        "Initialize context and isolate target modules.",
        "Implement core architectural changes or synthesis.",
        "Run automated verification gates.",
    ])[:5]
    checklist_md = "\n".join(f"- [ ] {idx}. {item}" for idx, item in enumerate(action_items, 1))

    content = f"""---
title: "TASK: {title}"
type: task
category: task
status: backlog
timebox: {timebox}
priority: P1
created: {date_str}
tags:
  - second-brain/task
  - task/gsd
---

# 🎯 Task: {title}

### 1. Goal (Atomic & Measurable)
- {goal}

### 2. Constraints & Time-Box
- **Estimated Duration**: `{timebox}` (Max 45m)
- **Target Scope**:
  - `{target_files if target_files else 'General codebase'}`

### 3. Action Checklist (Max 3-5 Discrete Actions)
{checklist_md}

### 4. Definition of Done (Binary Verifiable Gates)
- [ ] Verification command executed with zero exit code.
- [ ] Vault linter audit clean (0 broken links, 0 AI slop).

### 5. Failure Rollback Plan
- **Rollback Command**: `git restore .`

---
*Navigation*: [[00-Meta/Index|Index]] | [[03-Agent-Memory/Agent-Memory-MOC|Agent-Memory-MOC]]
"""
    file_path.write_text(content, encoding="utf-8")
    _register_task_in_moc(vault_path, file_path, title)
    print(f"✅ Created GSD Task: {file_path.name}")
    return file_path


def create_gsd_task(
    title: str,
    goal: str,
    steps: Optional[List[str]] = None,
    timebox: str = "30m",
    target_files: str = "",
    vault_dir: Optional[str] = None,
) -> Path:
    return create_task(
        title=title,
        goal=goal,
        timebox=timebox,
        target_files=target_files,
        steps=steps,
        vault_dir=vault_dir,
    )


def list_tasks(vault_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    vault_path = _resolve_vault_dir(vault_dir)
    tasks_dir = vault_path / "03-Agent-Memory" / "01-Tasks"
    if not tasks_dir.exists():
        print("No tasks directory found.")
        return []

    tasks = []
    for p in sorted(tasks_dir.glob("*.md")):
        if p.name.lower() == "readme.md":
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            status = "backlog"
            m = re.search(r"status:\s*([\w-]+)", text)
            if m:
                status = m.group(1).lower()
            title = p.stem
            tm = re.search(r'title:\s*["\']?(.*?)["\']?$', text, re.MULTILINE)
            if tm:
                title = tm.group(1)
            timebox = "30m"
            tbm = re.search(r"timebox:\s*([\w-]+)", text)
            if tbm:
                timebox = tbm.group(1)
            tasks.append({"file": p, "title": title, "status": status, "timebox": timebox})
        except Exception:
            continue

    print(f"\n📋 Active GSD Tasks ({len(tasks)} total):")
    print("-" * 65)
    for t in tasks:
        icon = "⏳" if t["status"] == "in-progress" else ("✅" if t["status"] == "completed" else "📌")
        print(f" {icon} [{t['status'].upper():11}] ({t['timebox']}) {t['file'].name}")
    print("-" * 65)
    return tasks


def update_status(filename: str, new_status: str, vault_dir: Optional[str] = None):
    vault_path = _resolve_vault_dir(vault_dir)
    tasks_dir = vault_path / "03-Agent-Memory" / "01-Tasks"
    target = None
    for p in tasks_dir.glob("*.md"):
        if filename in p.name:
            target = p
            break
    if not target:
        print(f"❌ Task matching '{filename}' not found.")
        return

    content = target.read_text(encoding="utf-8")
    updated = re.sub(r"status:\s*[\w-]+", f"status: {new_status}", content)
    target.write_text(updated, encoding="utf-8")
    print(f"✅ Updated task '{target.name}' status -> {new_status}")


def main():
    parser = argparse.ArgumentParser(description="GSD Work-Unit Manager")
    subparsers = parser.add_subparsers(dest="command")

    p_create = subparsers.add_parser("create")
    p_create.add_argument("title", help="Short title of the task")
    p_create.add_argument("--goal", required=True, help="Atomic 1-sentence goal")
    p_create.add_argument("--timebox", default="30m", help="Timebox (e.g. 25m, 45m)")
    p_create.add_argument("--files", default="", help="Target files/scope")

    subparsers.add_parser("list")

    p_start = subparsers.add_parser("start")
    p_start.add_argument("filename", help="Task filename or substring")

    p_done = subparsers.add_parser("done")
    p_done.add_argument("filename", help="Task filename or substring")

    args = parser.parse_args()

    if args.command == "create":
        create_task(args.title, args.goal, args.timebox, args.files)
    elif args.command == "list":
        list_tasks()
    elif args.command == "start":
        update_status(args.filename, "in-progress")
    elif args.command == "done":
        update_status(args.filename, "completed")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
