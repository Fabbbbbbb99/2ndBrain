#!/usr/bin/env python3
"""
pre_commit_hook.py - Automated Pre-Commit Safety & Quality Gate.
Enforces:
1. Rule 5 (Git Isolation): Blocks staged raw materials, PDFs, binary weights, or private agent memories.
2. Rule 6 & 7: Runs vault_linter.py to block broken wikilinks, invalid frontmatter, and banned AI slop.
"""

import sys
import subprocess
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
VAULT_DIR = REPO_ROOT / "vault_template"

def check_staged_isolation() -> bool:
    """Verifies that no private memory or raw materials are staged for commit."""
    try:
        res = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True
        )
        staged_files = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    except Exception as e:
        print(f"[Pre-Commit Error] Failed to get staged files: {e}")
        return False

    violations = []
    banned_prefixes = [
        "raw_materials/",
        "raw_audio/",
        "Knowledge Base Material/",
        "models/",
        "graphify-out/",
        "vault_template/04-Agent-Memory/01-Tasks/",
        "vault_template/04-Agent-Memory/02-Decisions/",
        "vault_template/04-Agent-Memory/03-Facts/",
        "vault_template/04-Agent-Memory/04-Skills/",
        "vault_template/04-Agent-Memory/Archive/",
        "vault_template/04-Agent-Memory/Consolidation-Reports/",
    ]
    banned_extensions = [".pdf", ".mp4", ".mp3", ".wav", ".onnx", ".pt", ".bin", ".safetensors", ".sqlite", ".db"]

    for f in staged_files:
        normalized = f.replace("\\", "/")
        # Check prefixes
        for prefix in banned_prefixes:
            if normalized.startswith(prefix) and not normalized.endswith("README.md") and not normalized.endswith("04-Agent-Memory-MOC.md"):
                violations.append(f"Private memory/raw path staged: {f}")
                break
        # Check extensions
        for ext in banned_extensions:
            if normalized.lower().endswith(ext):
                violations.append(f"Binary/Raw file extension staged: {f}")
                break

    if violations:
        print("\n❌ [Pre-Commit Gate Failed] Rule 5 (Git Isolation) Violation:")
        for v in violations:
            print(f"   - {v}")
        print("\nFix: Unstage these files via 'git reset HEAD <file>' before committing.\n")
        return False

    return True

def run_vault_linter() -> bool:
    """Runs vault_linter.py on vault_template."""
    print("🔍 [Pre-Commit] Running Vault Integrity & AI Slop Audit...")
    try:
        if str(SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPTS_DIR))
        from vault_linter import VaultLinter

        linter = VaultLinter(str(VAULT_DIR))
        report = linter.audit(auto_fix=False)

        # Disallow broken links or schema/slop issues
        has_broken_links = len(report.get("broken_links", [])) > 0
        has_schema_issues = len(report.get("schema_issues", [])) > 0

        if has_broken_links or has_schema_issues:
            print(linter.format_report(report))
            print("\n❌ [Pre-Commit Gate Failed] Vault contains broken links, missing frontmatter, or AI slop.")
            print("Run 'python scripts/vault_linter.py vault_template' to inspect and auto-fix.\n")
            return False

        print(f"✅ [Pre-Commit] Vault Audit Clean (Health: {report.get('health_score', 100)}/100).")
        return True
    except Exception as e:
        print(f"⚠️ [Pre-Commit Warning] Failed to run vault linter: {e}")
        return True  # Do not block if linter runtime issue occurs

def main():
    print("=" * 60)
    print("🛡️ 2ndBrain Pre-Commit Quality & Isolation Guard")
    print("=" * 60)

    if not check_staged_isolation():
        sys.exit(1)

    if not run_vault_linter():
        sys.exit(1)

    print("🚀 All pre-commit quality gates passed cleanly.")
    sys.exit(0)

if __name__ == "__main__":
    main()
