"""
bootstrap.py - Cross-Desktop Bootstrapper for 2ndBrain

Prepares a freshly cloned or existing 2ndBrain repository on any desktop (Windows, macOS, Linux):
1. Creates all required vault, build, and staging directory scaffolding.
2. Initializes configs/sources.local.yaml and vault/00-Meta/Index.local.md if absent.
3. Cleans stale backup files and reconciles 03-Agent-Memory/Agent-Memory-MOC.md.
4. Installs Git pre-commit hooks.
5. Renders and installs Antigravity skills (2ndBrain, robbie, marie) into ~/.gemini/config/skills/
   with the current machine's exact repository path.
6. Generates machine-local MCP configuration files (configs/claude_desktop_config.local.json
   and configs/cursor_mcp.local.json).
7. Ensures Laya System 1 weights (models/laya-knowledge-base/model.safetensors) exist;
   automatically runs dataset generation and ModernBERT training on fresh clones unless --skip-laya is passed.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


def ensure_directories(repo_root: Path) -> List[Path]:
    required_dirs = [
        repo_root / "vault" / "00-Meta" / "Templates",
        repo_root / "vault" / "01-Concepts",
        repo_root / "vault" / "02-Sources",
        repo_root / "vault" / "03-Agent-Memory" / "01-Tasks",
        repo_root / "vault" / "03-Agent-Memory" / "02-Decisions",
        repo_root / "vault" / "03-Agent-Memory" / "03-Facts",
        repo_root / "vault" / "03-Agent-Memory" / "04-Skills",
        repo_root / "vault" / "03-Agent-Memory" / "Sessions",
        repo_root / "vault" / "03-Agent-Memory" / "Archive",
        repo_root / "vault" / "03-Agent-Memory" / "Consolidation-Reports",
        repo_root / "build" / "ast",
        repo_root / "build" / "graphify-out",
        repo_root / "data",
        repo_root / "models",
        repo_root / "source_materials" / "shared",
        repo_root / "source_materials" / "robbie",
        repo_root / "source_materials" / "marie",
    ]
    created = []
    for d in required_dirs:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created.append(d)
    return created


def ensure_local_configs_and_clean_artifacts(repo_root: Path) -> None:
    # 1. Initialize sources.local.yaml from sources.example.yaml if absent
    example_yaml = repo_root / "configs" / "sources.example.yaml"
    local_yaml = repo_root / "configs" / "sources.local.yaml"
    if example_yaml.exists() and not local_yaml.exists():
        shutil.copyfile(example_yaml, local_yaml)
        print(f"  [Config] Created {local_yaml.relative_to(repo_root)} from template.")

    # 2. Initialize Index.local.md if absent, or strip legacy polluted auto-discovered tail
    index_local = repo_root / "vault" / "00-Meta" / "Index.local.md"
    if not index_local.exists():
        index_local.write_text(
            "---\n"
            'title: "2ndBrain Local Domain Index"\n'
            "type: moc\n"
            "tags:\n"
            "  - second-brain/meta\n"
            "  - moc\n"
            "---\n\n"
            "# 🗺️ Local Domain Index (Index.local.md)\n\n"
            "[[00-Meta/Index|Index]]\n",
            encoding="utf-8",
        )
    else:
        raw = index_local.read_text(encoding="utf-8", errors="ignore")
        marker = "### 📦 Auto-Discovered Unindexed Notes"
        if marker in raw:
            cleaned = raw.split(marker)[0].rstrip() + "\n"
            index_local.write_text(cleaned, encoding="utf-8")
            print("  [Clean] Stripped legacy polluted tail from vault/00-Meta/Index.local.md.")

    # 3. Remove stale Index.local.md.bak if present
    bak_file = repo_root / "vault" / "00-Meta" / "Index.local.md.bak"
    if bak_file.exists():
        bak_file.unlink()
        print("  [Clean] Removed stale vault/00-Meta/Index.local.md.bak.")

    # 4. Remove nested build/graphify-out/graphify-out if present
    nested_graphify = repo_root / "build" / "graphify-out" / "graphify-out"
    if nested_graphify.exists():
        shutil.rmtree(nested_graphify, ignore_errors=True)
        print("  [Clean] Removed redundant nested build/graphify-out/graphify-out/.")

    # 5. Remove legacy root-level graphify-out/ (all Graphify output lives in build/graphify-out/)
    root_graphify = repo_root / "graphify-out"
    if root_graphify.exists():
        shutil.rmtree(root_graphify, ignore_errors=True)
        print("  [Clean] Removed legacy root graphify-out/.")


def reconcile_vault_memory(repo_root: Path) -> int:
    from vault_linter import VaultLinter

    linter = VaultLinter(str(repo_root / "vault"))
    count = linter.reconcile_agent_memory_moc()
    print(f"  [Vault] Reconciled Agent-Memory-MOC.md ({count} active local 4-Pillar notes linked).")
    return count


def install_skills(repo_root: Path, skills_root: Path = None) -> Dict[str, Path]:
    if skills_root is None:
        skills_root = Path.home() / ".gemini" / "config" / "skills"

    skill_sources = {
        "2ndBrain": repo_root / "SKILL.md",
        "robbie": repo_root / "configs" / "skills" / "robbie" / "SKILL.md",
        "marie": repo_root / "configs" / "skills" / "marie" / "SKILL.md",
    }

    installed: Dict[str, Path] = {}
    repo_posix = repo_root.as_posix()

    for skill_name, src_path in skill_sources.items():
        if not src_path.exists():
            continue
        template_text = src_path.read_text(encoding="utf-8")
        rendered_text = template_text.replace("{{KNOWLEDGE_BASE_ROOT}}", str(repo_root))
        rendered_text = rendered_text.replace("{{KNOWLEDGE_BASE_ROOT_POSIX}}", repo_posix)

        dest_dir = skills_root / skill_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / "SKILL.md"
        dest_file.write_text(rendered_text, encoding="utf-8")
        installed[skill_name] = dest_file
        print(f"  [Skill] Installed {skill_name:8s} -> {dest_file}")

    return installed


def generate_mcp_configs(repo_root: Path) -> List[Path]:
    mcp_script = (repo_root / "scripts" / "mcp_server.py").as_posix()
    vault_dir = (repo_root / "vault").as_posix()
    py_exec = Path(sys.executable).as_posix()

    payload = {
        "mcpServers": {
            "second-brain": {
                "command": py_exec,
                "args": [mcp_script],
                "env": {
                    "SECOND_BRAIN_VAULT": vault_dir,
                    "KNOWLEDGE_BASE_ROOT": repo_root.as_posix(),
                },
            }
        }
    }

    out_files = [
        repo_root / "configs" / "claude_desktop_config.local.json",
        repo_root / "configs" / "cursor_mcp.local.json",
    ]
    for out_path in out_files:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"  [MCP]   Generated {out_path.relative_to(repo_root)}")

    return out_files


def install_hooks(repo_root: Path) -> None:
    hook_installer = repo_root / "scripts" / "install_git_hooks.py"
    if hook_installer.exists() and (repo_root / ".git").exists():
        subprocess.run(
            [sys.executable, str(hook_installer)],
            cwd=str(repo_root),
            check=False,
        )


def ensure_laya_model(repo_root: Path, skip_laya: bool = False, retrain_laya: bool = False) -> bool:
    model_weights = repo_root / "models" / "laya-knowledge-base" / "model.safetensors"
    if model_weights.exists() and not retrain_laya:
        size_mb = model_weights.stat().st_size / (1024 * 1024)
        print(f"  [Laya]  Neural weights verified at {model_weights.relative_to(repo_root)} ({size_mb:,.1f} MB).")
        return True

    if skip_laya:
        print("  [Laya]  Skipped neural training (--skip-laya). Calibrated <0.1ms heuristic engine active.")
        return False

    print("  [Laya]  Neural weights missing or --retrain-laya requested. Bootstrapping ModernBERT-large...")
    train_data = repo_root / "data" / "train_data.jsonl"
    val_data = repo_root / "data" / "val_data.jsonl"
    if not train_data.exists() or not val_data.exists() or retrain_laya:
        gen_script = repo_root / "scripts" / "generate_laya_dataset.py"
        subprocess.run([sys.executable, str(gen_script)], cwd=str(repo_root), check=True)

    train_script = repo_root / "scripts" / "train_laya.py"
    subprocess.run(
        [sys.executable, str(train_script), "--device", "auto"],
        cwd=str(repo_root),
        check=True,
    )
    return model_weights.exists()


def bootstrap(
    repo_root: Path = REPO_ROOT,
    skip_laya: bool = False,
    retrain_laya: bool = False,
    skills_root: Path = None,
) -> Dict[str, object]:
    print("=" * 65)
    print(f"🚀 2ndBrain Cross-Desktop Bootstrap ({repo_root})")
    print("=" * 65)

    created_dirs = ensure_directories(repo_root)
    ensure_local_configs_and_clean_artifacts(repo_root)
    reconciled_count = reconcile_vault_memory(repo_root)
    install_hooks(repo_root)
    installed_skills = install_skills(repo_root, skills_root=skills_root)
    mcp_configs = generate_mcp_configs(repo_root)
    laya_ready = ensure_laya_model(repo_root, skip_laya=skip_laya, retrain_laya=retrain_laya)

    print("=" * 65)
    print("✅ Bootstrap Complete.")
    print("=" * 65)
    return {
        "created_dirs": len(created_dirs),
        "reconciled_memories": reconciled_count,
        "installed_skills": list(installed_skills.keys()),
        "mcp_configs": [str(p) for p in mcp_configs],
        "laya_neural_ready": laya_ready,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap 2ndBrain on any desktop.")
    parser.add_argument(
        "--skip-laya",
        action="store_true",
        help="Skip neural ModernBERT training if weights are absent.",
    )
    parser.add_argument(
        "--retrain-laya",
        action="store_true",
        help="Force regeneration and retraining of Laya neural weights.",
    )
    args = parser.parse_args()
    bootstrap(REPO_ROOT, skip_laya=args.skip_laya, retrain_laya=args.retrain_laya)
