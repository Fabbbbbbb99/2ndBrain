#!/usr/bin/env python3
"""
gsd_task.py - GSD (Get Stuff Done) Task Engine for 2ndBrain.
Enforces Rule 6 (ADHD 5-Step Ceiling) and Rule 8 (GSD Atomic Work-Units).
"""

import sys
import re
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = REPO_ROOT / "vault_template" / "04-Agent-Memory" / "01-Tasks"
TEMPLATE_PATH = REPO_ROOT / "vault_template" / "00-Meta" / "Templates" / "Template-GSD-Task.md"


def slugify(text: str) -> str:
    s = re.sub(r'[^\w\s-]', '', text).strip()
    return re.sub(r'[-\s]+', '_', s).lower()[:50]


def create_task(title: str, goal: str, timebox: str = "30m", target_files: str = "") -> Path:
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    slug = slugify(title)
    filename = f"{date_str}_{slug}.md"
    file_path = TASKS_DIR / filename

    content = f"""---
title: "TASK: {title}"
type: task
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

### 3. Action Checklist (Max 3–5 Discrete Actions)
- [ ] 1. Initialize context and isolate target modules.
- [ ] 2. Implement core architectural changes or synthesis.
- [ ] 3. Run automated verification gates.

### 4. Definition of Done (Binary Verifiable Gates)
- [ ] Verification command executed with zero exit code.
- [ ] Vault linter audit clean (0 broken links, 0 AI slop).

### 5. Failure Rollback Plan
- **Rollback Command**: `git restore .`
"""
    file_path.write_text(content, encoding="utf-8")
    print(f"✅ Created GSD Task: {file_path.relative_to(REPO_ROOT)}")
    return file_path


def list_tasks() -> List[Dict[str, Any]]:
    if not TASKS_DIR.exists():
        print("No tasks directory found.")
        return []

    tasks = []
    for p in sorted(TASKS_DIR.glob("*.md")):
        if p.name.lower() == "readme.md":
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            status = "backlog"
            m = re.search(r'status:\s*([\w-]+)', text)
            if m:
                status = m.group(1).lower()
            title = p.stem
            tm = re.search(r'title:\s*["\']?(.*?)["\']?$', text, re.MULTILINE)
            if tm:
                title = tm.group(1)
            timebox = "30m"
            tbm = re.search(r'timebox:\s*([\w-]+)', text)
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


def update_status(filename: str, new_status: str):
    target = None
    for p in TASKS_DIR.glob("*.md"):
        if filename in p.name:
            target = p
            break
    if not target:
        print(f"❌ Task matching '{filename}' not found.")
        return

    content = target.read_text(encoding="utf-8")
    updated = re.sub(r'status:\s*[\w-]+', f'status: {new_status}', content)
    target.write_text(updated, encoding="utf-8")
    print(f"✅ Updated task '{target.name}' status -> {new_status}")


def main():
    parser = argparse.ArgumentParser(description="GSD Work-Unit Manager")
    subparsers = parser.add_subparsers(dest="command")

    # Create
    p_create = subparsers.add_parser("create")
    p_create.add_argument("title", help="Short title of the task")
    p_create.add_argument("--goal", required=True, help="Atomic 1-sentence goal")
    p_create.add_argument("--timebox", default="30m", help="Timebox (e.g. 25m, 45m)")
    p_create.add_argument("--files", default="", help="Target files/scope")

    # List
    subparsers.add_parser("list")

    # Start
    p_start = subparsers.add_parser("start")
    p_start.add_argument("filename", help="Task filename or substring")

    # Done
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
