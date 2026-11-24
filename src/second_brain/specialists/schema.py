"""Pydantic schema validation for specialists/<domain>.yaml profiles."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class SpecialistSpec(BaseModel):
    spec_version: str = "1.0.0"
    domain_id: str
    display_name: str
    description: str
    allowed_domains: list[str]
    source_versions: list[str] = Field(default_factory=list)
    token_budget: int = Field(default=2500, ge=256, le=8000)
    citation_policy: Literal["strict_verbatim_normative", "version_pinned_reference", "synthesis"] = "version_pinned_reference"
    instructions: str


def load_all_specialists(vault_path: Path) -> list[SpecialistSpec]:
    spec_dir = vault_path / "specialists"
    if not spec_dir.exists():
        return []
    out: list[SpecialistSpec] = []
    for yf in sorted(spec_dir.glob("*.yaml")):
        raw = yaml.safe_load(yf.read_text(encoding="utf-8")) or {}
        out.append(SpecialistSpec(**raw))
    return out
