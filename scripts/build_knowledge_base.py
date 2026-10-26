"""
build_knowledge_base.py - Master Builder for 2ndBrain Knowledge Base.
Ingests staging documents, synchronizes CRG AST & Graphify, builds hierarchical vector indices, and audits vault integrity.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from hybrid_search import HybridSearchEngine
from ingest_knowledge import UniversalKnowledgeIngester
from sync_brain import sync_code_ast, sync_semantic_graph
from vault_linter import VaultLinter

PROJECT_ROOT = SCRIPTS_DIR.parent
VAULT_DIR = Path(
    os.environ.get(
        "VAULT_DIR",
        PROJECT_ROOT / ("vault" if (PROJECT_ROOT / "vault").exists() else "vault_template"),
    )
)
RAW_DIR = Path(
    os.environ.get(
        "RAW_DIR",
        PROJECT_ROOT
        / ("source_materials" if (PROJECT_ROOT / "source_materials").exists() else "raw_materials"),
    )
)
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_yaml_or_json(path: Path) -> Dict[str, Any]:
    """Safely loads YAML or JSON configuration."""
    if not path.exists():
        return {}
    content = path.read_text(encoding="utf-8")
    try:
        import yaml

        return yaml.safe_load(content) or {}
    except ImportError:
        import json

        return json.loads(content)


def resolve_source_path(raw_path: str, base_dir: Path) -> Path:
    """Resolves relative or absolute paths against project root or raw dir."""
    p = Path(raw_path)
    if p.is_absolute():
        return p
    if (PROJECT_ROOT / p).exists():
        return PROJECT_ROOT / p
    if (base_dir / p).exists():
        return base_dir / p
    if p.parts and p.parts[0] == base_dir.name:
        return PROJECT_ROOT / p
    return base_dir / p


def ingest_from_config(
    ingester: UniversalKnowledgeIngester, config_path: Path, raw_dir: Path, force: bool = False
) -> int:
    """Ingests documents specified in a manifest config file (mapped_sources first, then directory sources)."""
    config = load_yaml_or_json(config_path)
    total_docs = 0

    # 1. Pattern-to-Title Mappings first (so mapped lectures receive human-readable titles)
    mapped_sources: List[Dict[str, Any]] = config.get("mapped_sources", [])
    for item in mapped_sources:
        folder = resolve_source_path(item.get("directory", ""), raw_dir)
        mapping: Dict[str, str] = item.get("mapping", {})
        if not folder.exists():
            print(f"[Warning] Mapped directory not found: {folder}")
            continue
        print(f"\n---> Ingesting mapped documents in: {folder.name}")
        for doc_file in sorted(folder.rglob("*")):
            if not doc_file.is_file():
                continue
            name_lower = doc_file.name.lower()
            if "_orig" in name_lower:
                continue
            for key, clean_title in mapping.items():
                if key.lower() in name_lower:
                    try:
                        res = ingester.ingest_document(
                            doc_file, custom_title=clean_title, force=force, quiet=True
                        )
                        if res:
                            total_docs += 1
                    except Exception as e:
                        print(f"     [Error ingesting {doc_file.name}]: {e}")
                    break

    # 2. Directory & File Sources
    sources: List[str] = config.get("sources", [])
    for src in sources:
        target_path = resolve_source_path(src, raw_dir)
        if not target_path.exists():
            print(f"[Warning] Source path not found: {target_path}")
            continue
        print(f"\n---> Ingesting source: {target_path}")
        try:
            if target_path.is_file():
                res = ingester.ingest_document(target_path, force=force, quiet=True)
                if res:
                    total_docs += 1
            else:
                results = ingester.ingest_directory(
                    target_path, force=force, update_vectors=False, quiet=True
                )
                total_docs += len(results)
                print(f"     Ingested {len(results)} items from {target_path.name}")
        except Exception as e:
            print(f"     [Error ingesting {target_path.name}]: {e}")

    if getattr(ingester, "_manifest_dirty", False):
        ingester._save_manifest()
        ingester._manifest_dirty = False
    ingester._flush_index()

    return total_docs


def ingest_auto_discovery(
    ingester: UniversalKnowledgeIngester, raw_dir: Path, force: bool = False
) -> int:
    """Recursively auto-discovers and ingests all documents inside source_materials/ or raw_materials/."""
    if not raw_dir.exists():
        print(f"[Info] Staging directory {raw_dir} does not exist. Skipping auto-discovery.")
        return 0
    print(f"\n---> Auto-discovering documents in: {raw_dir}")
    results = ingester.ingest_directory(raw_dir, force=force, update_vectors=False, quiet=True)
    print(f"     Ingested {len(results)} items via auto-discovery.")
    return len(results)


def main():
    parser = argparse.ArgumentParser(description="2nd Brain Master Knowledge Base Builder")
    parser.add_argument(
        "--vault", type=str, default=str(VAULT_DIR), help="Path to target vault directory"
    )
    parser.add_argument(
        "--raw", type=str, default=str(RAW_DIR), help="Path to staging source_materials directory"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to custom sources YAML/JSON manifest"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean all ingested vault notes, vector DB, and manifest before full re-ingestion",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force full re-ingestion and vector re-indexing"
    )
    parser.add_argument(
        "--skip-index", action="store_true", help="Skip SQLite/vector embedding indexing"
    )
    parser.add_argument("--skip-lint", action="store_true", help="Skip vault integrity linter")
    args = parser.parse_args()

    vault_path = Path(args.vault).resolve()
    raw_path = Path(args.raw).resolve()

    print("=" * 60)
    print("🚀 Starting 2nd Brain Knowledge Base Build")
    print(f"Target Vault : {vault_path}")
    print(f"Raw Staging  : {raw_path}")
    print("=" * 60)

    ingester = UniversalKnowledgeIngester(str(vault_path))

    if args.clean:
        print("\n" + "=" * 60)
        print("🧹 Cleaning Ingested Knowledge (Preserving Agent Memory & Templates)...")
        print("=" * 60)
        clean_stats = ingester.clean_ingested_knowledge(preserve_caches=True)
        print(
            f"✅ Cleaned {clean_stats['concepts_removed']} concept notes and "
            f"{clean_stats['sources_removed']} source notes "
            f"(cached {clean_stats['pdf_cached']} PDF extracts, "
            f"{clean_stats['embeddings_cached']} deterministic chunks)."
        )

    total_docs = 0

    config_file: Optional[Path] = None
    if args.config and Path(args.config).exists():
        config_file = Path(args.config).resolve()
    elif (CONFIGS_DIR / "sources.local.yaml").exists():
        config_file = CONFIGS_DIR / "sources.local.yaml"
    elif (CONFIGS_DIR / "sources.yaml").exists():
        config_file = CONFIGS_DIR / "sources.yaml"

    if config_file:
        print(f"Loading ingestion manifest: {config_file.relative_to(PROJECT_ROOT)}")
        total_docs += ingest_from_config(ingester, config_file, raw_path, force=args.force)
        # Also run auto-discovery for any directory not listed in config
        total_docs += ingest_auto_discovery(ingester, raw_path, force=False)
    else:
        print("No ingestion manifest found. Falling back to recursive auto-discovery.")
        total_docs += ingest_auto_discovery(ingester, raw_path, force=args.force)

    # Synchronize CRG AST and Graphify Semantic Graph + Vault Concept Bridge
    print("\n" + "=" * 60)
    print("🕸️ Synchronizing CRG AST & Graphify Semantic Graph...")
    print("=" * 60)
    try:
        sync_code_ast(str(PROJECT_ROOT))
        sync_semantic_graph(str(PROJECT_ROOT), str(vault_path))
    except Exception as e:
        print(f"⚠️ Graph sync note: {e}")

    # Hierarchical Vector Indexing
    if not args.skip_index:
        print("\n" + "=" * 60)
        print("🧠 Building Hierarchical Vector Embeddings (Parent-Child Dense + BM25)...")
        print("=" * 60)
        try:
            search_engine = HybridSearchEngine(str(vault_path))
            indexed_count = search_engine.index_vault(force=args.force)
            print(f"✅ Indexed {indexed_count} notes into SQLite Parent-Child Vector Database.")
        except Exception as e:
            print(f"⚠️ Vector Indexing note: {e}")

    # Remove temporary build caches if created during --clean
    for tmp_cache in (
        PROJECT_ROOT / "build" / ".pdf_extract_cache.db",
        PROJECT_ROOT / "build" / ".embedding_cache.db",
    ):
        if tmp_cache.exists():
            tmp_cache.unlink(missing_ok=True)

    # Vault Integrity Audit
    if not args.skip_lint:
        print("\n" + "=" * 60)
        print("🔍 Auditing Vault Integrity (Vault Linter)...")
        print("=" * 60)
        try:
            linter = VaultLinter(str(vault_path))
            report = linter.audit(auto_fix=True)
            print(linter.format_report(report))
        except Exception as e:
            print(f"⚠️ Lint note: {e}")

    print("\n" + "=" * 60)
    print(f"🎉 Build Complete! Total Processed Documents: {total_docs}")
    print("=" * 60)


if __name__ == "__main__":
    main()
