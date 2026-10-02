"""
build_knowledge_base.py - Master Builder for Robbie Knowledge Base.
Ingests foundational robotics documents, specifications, and curricula,
then builds the hierarchical vector index and audits vault health.
"""

import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add current scripts directory to path
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ingest_knowledge import UniversalKnowledgeIngester
from hybrid_search import HybridSearchEngine
from vault_linter import VaultLinter

ROBBIE_DIR = SCRIPTS_DIR.parent
VAULT_DIR = ROBBIE_DIR / "vault_template"
RAW_DIR = ROBBIE_DIR / "raw_materials"

SOURCES_TO_INGEST = [
    # Books & Curricula PDFs & Markdown Sources
    RAW_DIR / "references" / "Academic_Curricula_and_University_Programs" / "Stanford_EE364a_Convex_Optimization",
    RAW_DIR / "references" / "Academic_Curricula_and_University_Programs" / "UCL_Reinforcement_Learning_David_Silver",
    RAW_DIR / "references" / "Academic_Curricula_and_University_Programs" / "MIT_6_832_Underactuated_Robotics_Russ_Tedrake",
    RAW_DIR / "references" / "Reference_Books" / "Nonlinear_Dynamics_and_Chaos_Steven_Strogatz",
    RAW_DIR / "references" / "Marine_and_AUV" / "Fossen" / "Book",
    RAW_DIR / "references" / "Dynamics_and_Lie_Theory" / "Micro_Lie_Theory",
    RAW_DIR / "references" / "Dynamics_and_Lie_Theory" / "Pinocchio",
    
    # Official Standards & Architecture Specifications
    RAW_DIR / "references" / "ROS_Enhancement_Proposals_REPs" / "rep",
    RAW_DIR / "references" / "ROS2_Design_Architecture" / "design",
    RAW_DIR / "references" / "Reference_Books" / "Kalman_and_Bayesian_Filters_in_Python_Roger_Labbe",
    
    # Official ROS 2 Architecture & Concepts (Humble, Jazzy)
    RAW_DIR / "ros2_docs" / "humble" / "source" / "Concepts",
    RAW_DIR / "ros2_docs" / "jazzy" / "source" / "Concepts",
]

def main():
    print("=" * 60)
    print("🚀 Starting Comprehensive Knowledge Base Build for Robbie")
    print(f"Target Vault: {VAULT_DIR}")
    print("=" * 60)

    ingester = UniversalKnowledgeIngester(str(VAULT_DIR))
    total_docs = 0

    for source_path in SOURCES_TO_INGEST:
        if not source_path.exists():
            print(f"\n[Warning] Source path not found: {source_path}")
            continue
        print(f"\n---> Ingesting from: {source_path.relative_to(ROBBIE_DIR)}")
        try:
            results = ingester.ingest_directory(source_path)
            total_docs += len(results)
            print(f"     Ingested {len(results)} items from {source_path.name}")
        except Exception as e:
            print(f"     [Error during ingestion]: {e}")

    print("\n" + "=" * 60)
    print("🧠 Building Hierarchical Vector Embeddings (Parent-Child Dense + BM25)...")
    print("=" * 60)
    try:
        search_engine = HybridSearchEngine(str(VAULT_DIR))
        indexed_count = search_engine.index_vault(force=False)
        print(f"✅ Indexed {indexed_count} notes into SQLite Parent-Child Vector Database.")
    except Exception as e:
        print(f"⚠️ Vector Indexing note: {e}")

    print("\n" + "=" * 60)
    print("🔍 Auditing Vault Integrity (Vault Linter)...")
    print("=" * 60)
    try:
        linter = VaultLinter(str(VAULT_DIR))
        report = linter.audit(auto_fix=True)
        print(linter.format_report(report))
    except Exception as e:
        print(f"⚠️ Lint note: {e}")

    print("\n" + "=" * 60)
    print(f"🎉 Knowledge Base Build Finished! Total Ingested Documents: {total_docs}")
    print("=" * 60)

if __name__ == "__main__":
    main()
