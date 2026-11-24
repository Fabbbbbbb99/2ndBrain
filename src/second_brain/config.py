"""Strict configuration loader with zero silent fallbacks."""
from __future__ import annotations

import hashlib
import os
import tomllib
from pathlib import Path
from pydantic import BaseModel, Field, field_validator


class ConfigError(RuntimeError):
    """Raised when configuration or required directories are missing or invalid."""


class BrainConfig(BaseModel):
    vault_path: Path
    state_dir: Path
    allow_laya_auto_commit: bool = False
    max_fanout_domains: int = Field(default=2, ge=1, le=5)
    pack_token_budget: int = Field(default=2500, ge=256, le=8000)
    fanout_token_budget: int = Field(default=12000, ge=1000, le=32000)
    intra_op_num_threads: int = Field(default=8, ge=1, le=16)
    inter_op_num_threads: int = Field(default=1, ge=1, le=4)
    enable_cpu_mem_arena: bool = False
    warm_cascade_p95_budget_ms: int = Field(default=1350, ge=100)
    daemon_idle_ttl_seconds: int = Field(default=1800, ge=30)
    active_laya_model_dir: str = "laya_stock"
    memory_sim_threshold: float = Field(default=0.62, ge=0.0, le=1.0)

    @field_validator("vault_path", "state_dir", mode="before")
    @classmethod
    def _coerce_path(cls, value: str | Path) -> Path:
        p = Path(value).expanduser()
        if not p.is_absolute():
            raise ConfigError(f"Configured path must be absolute: {value!r}")
        return p.resolve()

    @property
    def db_path(self) -> Path:
        return self.state_dir / "brain.sqlite"

    @property
    def models_dir(self) -> Path:
        return self.state_dir / "models"

    @property
    def laya_dir(self) -> Path:
        import json

        ptr = self.state_dir / "model_pointer.json"
        if ptr.exists():
            try:
                active = Path(json.loads(ptr.read_text(encoding="utf-8"))["active_dir"])
                return active if active.is_absolute() else (self.models_dir / active).resolve()
            except Exception:
                pass
        return self.models_dir / self.active_laya_model_dir

    @property
    def fastembed_dir(self) -> Path:
        return self.models_dir / "fastembed"

    @property
    def parsed_dir(self) -> Path:
        return self.state_dir / "parsed"

    @property
    def crops_dir(self) -> Path:
        return self.state_dir / "crops"

    @property
    def logs_dir(self) -> Path:
        return self.state_dir / "logs"

    @property
    def lock_path(self) -> Path:
        return self.state_dir / "daemon.lock"

    @property
    def pipe_name(self) -> str:
        digest = hashlib.sha256(str(self.vault_path).lower().encode("utf-8")).hexdigest()[:12]
        return rf"\\.\pipe\second_brain_{digest}"

    @property
    def certified_edges_path(self) -> Path:
        return self.vault_path / "ontology" / "certified_edges.jsonl"


def default_config_path() -> Path:
    env_cfg = os.environ.get("BRAIN_CONFIG_PATH")
    if env_cfg:
        return Path(env_cfg).expanduser().resolve()
    local_app = os.environ.get("LOCALAPPDATA")
    if not local_app:
        raise ConfigError("LOCALAPPDATA environment variable is unset and BRAIN_CONFIG_PATH was not provided.")
    return Path(local_app) / "2ndBrain" / "config.toml"


def load_config(
    config_path: Path | None = None,
    vault_override: Path | str | None = None,
    state_override: Path | str | None = None,
    require_existing_vault: bool = True,
) -> BrainConfig:
    cfg_path = config_path or default_config_path()
    data: dict[str, object] = {}
    if cfg_path.exists():
        data = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    elif not (vault_override or os.environ.get("BRAIN_VAULT_PATH")):
        raise ConfigError(
            f"Missing config file at {cfg_path}. Run `brain init --vault <path>` or set BRAIN_CONFIG_PATH / BRAIN_VAULT_PATH."
        )

    env_vault = os.environ.get("BRAIN_VAULT_PATH")
    env_state = os.environ.get("BRAIN_STATE_DIR")
    if vault_override is not None:
        data["vault_path"] = str(vault_override)
    elif env_vault:
        data["vault_path"] = env_vault
    if state_override is not None:
        data["state_dir"] = str(state_override)
    elif env_state:
        data["state_dir"] = env_state
    elif "state_dir" not in data:
        local_app = os.environ.get("LOCALAPPDATA")
        if not local_app:
            raise ConfigError("Missing state_dir in config and LOCALAPPDATA is unset.")
        data["state_dir"] = str(Path(local_app) / "2ndBrain" / "state")

    if "vault_path" not in data:
        raise ConfigError("Missing vault_path in configuration. Provide --vault or BRAIN_VAULT_PATH.")

    cfg = BrainConfig(**data)
    if require_existing_vault and not cfg.vault_path.exists():
        raise ConfigError(f"Configured vault_path does not exist: {cfg.vault_path}")
    return cfg
