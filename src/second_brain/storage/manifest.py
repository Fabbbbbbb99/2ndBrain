"""Vault manifest validation and fail-fast drift detection."""
from __future__ import annotations

import hashlib
from pathlib import Path
import yaml
from pydantic import BaseModel


class ManifestDriftError(RuntimeError):
    """Raised when vault_manifest.yaml or ontology/taxonomy.yaml drifts out of sync."""


class VaultManifest(BaseModel):
    vault_schema_version: str
    ontology_hash: str
    parser_version: str
    embed_model_id: str
    rerank_model_id: str
    laya_model_id: str


def compute_file_sha256(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def load_taxonomy(vault_path: Path) -> tuple[dict, str]:
    tax_path = vault_path / "ontology" / "taxonomy.yaml"
    if not tax_path.exists():
        raise ManifestDriftError(f"Missing ontology taxonomy file: {tax_path}")
    raw_bytes = tax_path.read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha256(raw_bytes).hexdigest()
    parsed = yaml.safe_load(raw_bytes.decode("utf-8")) or {}
    return parsed, digest


def load_and_verify_manifest(vault_path: Path) -> tuple[VaultManifest, dict]:
    manifest_path = vault_path / "vault_manifest.yaml"
    if not manifest_path.exists():
        raise ManifestDriftError(f"Missing vault_manifest.yaml at {manifest_path}")
    manifest = VaultManifest(**(yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}))
    taxonomy, actual_hash = load_taxonomy(vault_path)
    if manifest.ontology_hash != actual_hash:
        raise ManifestDriftError(
            f"Ontology hash mismatch: vault_manifest.yaml has {manifest.ontology_hash[:12]}, "
            f"but ontology/taxonomy.yaml hashes to {actual_hash[:12]}. Run `brain init --sync-manifest` or inspect diff."
        )
    return manifest, taxonomy
