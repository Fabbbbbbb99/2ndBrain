"""Pure-stdlib Windows auto-detect installer for 2ndBrain v2."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent


def _fail(msg: str, code: int = 2) -> int:
    print(f"[ERROR] {msg}", file=sys.stderr)
    return code


def _default_localappdata() -> Path:
    val = os.environ.get("LOCALAPPDATA")
    return Path(val) if val else (Path.home() / "AppData" / "Local")


def _default_antigravity_mcp_config() -> Path:
    cfg = Path.home() / ".gemini" / "config" / "mcp_config.json"
    legacy = Path.home() / ".gemini" / "antigravity" / "mcp_config.json"
    if (Path.home() / ".gemini" / "config").exists() or not legacy.exists():
        return cfg
    return legacy


def _write_lf_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))
    os.replace(tmp, path)


def _configure_git_hooks(repo_dir: Path) -> bool:
    if not (repo_dir / ".git").exists():
        return False
    res = subprocess.run(
        ["git", "-C", str(repo_dir), "config", "core.hooksPath", ".githooks"],
        capture_output=True,
        text=True,
        check=False,
    )
    return res.returncode == 0


def _generate_mcp_snippets(repo_root: Path, brain_mcp_exe: Path, cfg_path: Path) -> list[Path]:
    payload = {
        "mcpServers": {
            "second-brain": {
                "command": str(brain_mcp_exe),
                "args": [],
                "env": {
                    "BRAIN_CONFIG_PATH": str(cfg_path),
                },
            }
        }
    }
    text = json.dumps(payload, indent=2) + "\n"
    out_dir = repo_root / "configs" / "generated"
    targets = [
        out_dir / "antigravity.local.json",
        out_dir / "claude_desktop.local.json",
        out_dir / "cursor.local.json",
    ]
    for target in targets:
        _write_lf_atomic(target, text)
    return targets


def _merge_mcp_config(target: Path, brain_mcp_exe: Path, cfg_path: Path) -> dict[str, str]:
    backup_path = ""
    data: dict[str, object] = {}
    if target.exists():
        raw = target.read_text(encoding="utf-8")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Refusing to modify invalid JSON in {target}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Expected top-level JSON object in {target}")
        data = parsed
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup = target.with_name(f"{target.name}.bak.{stamp}")
        shutil.copy2(target, backup)
        backup_path = str(backup)

    servers = data.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        raise RuntimeError(f"Expected 'mcpServers' object in {target}")
    servers["second-brain"] = {
        "command": str(brain_mcp_exe),
        "args": [],
        "env": {"BRAIN_CONFIG_PATH": str(cfg_path)},
    }
    _write_lf_atomic(target, json.dumps(data, indent=2) + "\n")
    return {"target": str(target), "backup": backup_path}


def main(argv: list[str] | None = None) -> int:
    if sys.platform != "win32":
        return _fail("2ndBrain v2 currently supports Windows only (Named Pipe IPC and Windows credential checks).")
    if sys.version_info < (3, 13):
        return _fail(f"Python 3.13+ is required; running {sys.version.split()[0]}.")
    if "windowsapps" in sys.executable.lower():
        return _fail("Detected Microsoft Store PythonApps stub. Install full Python 3.13+ from python.org.")

    local_app = _default_localappdata()
    parser = argparse.ArgumentParser(description="Auto-detect Windows setup for 2ndBrain v2")
    parser.add_argument("--vault", default=str(Path.home() / "2ndBrain_Vault"))
    parser.add_argument("--state", default=str(local_app / "2ndBrain" / "state"))
    parser.add_argument("--config", default=str(local_app / "2ndBrain" / "config.toml"))
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-fastembed", action="store_true")
    parser.add_argument("--allow-unverified-model", action="store_true")
    parser.add_argument("--allow-onedrive", action="store_true")
    parser.add_argument("--register-mcp", action="store_true")
    parser.add_argument("--mcp-config", default=None)
    parser.add_argument("--skills-dir", default=None)
    parser.add_argument("--prune-skills", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    vault = Path(args.vault).expanduser().resolve()
    state = Path(args.state).expanduser().resolve()
    cfg_path = Path(args.config).expanduser().resolve()

    if (("onedrive" in str(vault).lower()) or ("onedrive" in str(state).lower())) and (not args.allow_onedrive):
        return _fail(
            f"OneDrive-synced paths risk SQLite/Obsidian locks (vault={vault}, state={state}). "
            "Pass --vault outside OneDrive or re-run with --allow-onedrive."
        )

    venv_dir = REPO_ROOT / ".venv"
    venv_python = venv_dir / "Scripts" / "python.exe"
    brain_exe = venv_dir / "Scripts" / "brain.exe"
    brain_mcp_exe = venv_dir / "Scripts" / "brain-mcp.exe"

    if not venv_python.exists():
        print(f"[1/6] Creating virtual environment at {venv_dir} ...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    else:
        print(f"[1/6] Reusing virtual environment at {venv_dir}")

    print("[2/6] Installing second-brain package in editable mode ...")
    subprocess.run([str(venv_python), "-m", "pip", "install", "-e", str(REPO_ROOT)], check=True)

    env = os.environ.copy()
    env["BRAIN_CONFIG_PATH"] = str(cfg_path)

    print(f"[3/6] Initializing external Vault ({vault}) and config ({cfg_path}) ...")
    init_cmd = [
        str(brain_exe),
        "init",
        "--vault",
        str(vault),
        "--state",
        str(state),
        "--config",
        str(cfg_path),
        "--sync-manifest",
    ]
    if args.force:
        init_cmd.append("--force")
    subprocess.run(init_cmd, env=env, check=True)

    if args.skip_models:
        print("[4/6] Skipping model downloads (--skip-models). Run `brain model pull` later.")
    else:
        print("[4/6] Downloading & verifying stock ONNX models ...")
        pull_cmd = [str(brain_exe), "model", "pull", "--config", str(cfg_path)]
        if args.skip_fastembed:
            pull_cmd.append("--skip-fastembed")
        if args.allow_unverified_model:
            pull_cmd.append("--allow-unverified")
        subprocess.run(pull_cmd, env=env, check=True)

    repo_hooks_ok = _configure_git_hooks(REPO_ROOT)
    vault_hooks_ok = _configure_git_hooks(vault)
    print(f"[5/6] Git hooks configured: repo={repo_hooks_ok}, vault={vault_hooks_ok}")

    snippets = _generate_mcp_snippets(REPO_ROOT, brain_mcp_exe, cfg_path)
    mcp_reg_info: dict[str, str] | None = None
    if args.register_mcp:
        target_mcp = Path(args.mcp_config).expanduser().resolve() if args.mcp_config else _default_antigravity_mcp_config()
        mcp_reg_info = _merge_mcp_config(target_mcp, brain_mcp_exe, cfg_path)
        exp_cmd = [str(brain_exe), "export-skills", "--config", str(cfg_path)]
        if not args.prune_skills:
            exp_cmd.append("--no-prune")
        if args.skills_dir:
            exp_cmd.extend(["--output-dir", str(Path(args.skills_dir).expanduser().resolve())])
        subprocess.run(exp_cmd, env=env, check=True)

    print("[6/6] Running brain doctor --stdio-check ...")
    doc_cmd = [str(brain_exe), "doctor", "--stdio-check", "--config", str(cfg_path)]
    if args.skip_models:
        doc_cmd.append("--allow-missing-models")
    if args.allow_onedrive:
        doc_cmd.append("--allow-onedrive")
    subprocess.run(doc_cmd, env=env, check=True)

    summary = {
        "setup_complete": True,
        "repo_root": str(REPO_ROOT),
        "vault_path": str(vault),
        "state_dir": str(state),
        "config_path": str(cfg_path),
        "generated_mcp_snippets": [str(p) for p in snippets],
        "registered_mcp": mcp_reg_info,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
