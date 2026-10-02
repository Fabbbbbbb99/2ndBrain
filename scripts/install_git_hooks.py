#!/usr/bin/env python3
"""
install_git_hooks.py - Installs git pre-commit quality gate into .git/hooks/pre-commit
"""

import sys
import os
import stat
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
GIT_HOOKS_DIR = REPO_ROOT / ".git" / "hooks"
PRE_COMMIT_DEST = GIT_HOOKS_DIR / "pre-commit"

HOOK_SHELL_SCRIPT = """#!/usr/bin/env bash
# 2ndBrain Pre-Commit Hook

# Detect working Python interpreter (avoids Windows Microsoft Store stubs)
PYTHON_BIN=""
for candidate in python python3 py; do
    if "$candidate" -c "import sys; sys.exit(0)" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "[Pre-Commit Error] No functional Python interpreter found on PATH."
    exit 1
fi

"$PYTHON_BIN" scripts/pre_commit_hook.py
"""

def install_hook():
    if not GIT_HOOKS_DIR.exists():
        print(f"[Error] .git/hooks directory not found at {GIT_HOOKS_DIR}")
        sys.exit(1)

    PRE_COMMIT_DEST.write_text(HOOK_SHELL_SCRIPT, encoding="utf-8")
    # Make executable on Unix/Git Bash
    st = os.stat(str(PRE_COMMIT_DEST))
    os.chmod(str(PRE_COMMIT_DEST), st.st_mode | stat.S_IEXEC)

    print(f"✅ Pre-commit hook successfully installed at: {PRE_COMMIT_DEST}")

if __name__ == "__main__":
    install_hook()
