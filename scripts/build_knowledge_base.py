"""
build_knowledge_base.py - Master Builder for Robbie Knowledge Base.
Ingests foundational robotics documents, specifications, curricula, and MIT algorithm lectures,
then builds the hierarchical vector index and audits vault health.
"""

import os
import sys
from pathlib import Path
from typing import Dict

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
CURRICULA_DIR = RAW_DIR / "references" / "Academic_Curricula_and_University_Programs"

SOURCES_TO_INGEST = [
    # Books & Curricula PDFs & Markdown Sources
    CURRICULA_DIR / "Stanford_EE364a_Convex_Optimization",
    CURRICULA_DIR / "UCL_Reinforcement_Learning_David_Silver",
    CURRICULA_DIR / "MIT_6_832_Underactuated_Robotics_Russ_Tedrake",
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

# Curated MIT Algorithm Lecture Maps
MIT_6006_MAP = {
    "lec01": "MIT 6.006 Lec 01 Algorithmic Thinking Peak Finding",
    "lec02": "MIT 6.006 Lec 02 Models of Computation Document Distance",
    "lec03": "MIT 6.006 Lec 03 Insertion Sort Merge Sort",
    "lec04": "MIT 6.006 Lec 04 Heaps and Heap Sort Priority Queues",
    "lec05": "MIT 6.006 Lec 05 Binary Search Trees BST Sort",
    "lec06": "MIT 6.006 Lec 06 AVL Balanced Trees AVL Sort",
    "lec07": "MIT 6.006 Lec 07 Counting Sort Radix Sort",
    "lec08": "MIT 6.006 Lec 08 Hashing with Chaining",
    "lec09": "MIT 6.006 Lec 09 Table Doubling Karp Rabin String Matching",
    "lec10": "MIT 6.006 Lec 10 Open Addressing Cryptographic Hashing",
    "lec11": "MIT 6.006 Lec 11 Integer Arithmetic Karatsuba Multiplication",
    "lec12": "MIT 6.006 Lec 12 Square Roots Newtons Method",
    "lec13": "MIT 6.006 Lec 13 Breadth First Search BFS Shortest Paths",
    "lec14": "MIT 6.006 Lec 14 Depth First Search DFS Topological Sorting",
    "lec15": "MIT 6.006 Lec 15 Single Source Shortest Paths DAGs",
    "lec16": "MIT 6.006 Lec 16 Dijkstra Algorithm Priority Queue",
    "lec17": "MIT 6.006 Lec 17 Bellman Ford Negative Weight Cycles",
    "lec18": "MIT 6.006 Lec 18 Speeding Up Dijkstra Bidirectional Search",
    "lec19": "MIT 6.006 Lec 19 Dynamic Programming I Fibonacci Shortest Paths",
    "lec20": "MIT 6.006 Lec 20 Dynamic Programming II Text Justification Blackjack",
    "lec21": "MIT 6.006 Lec 21 Dynamic Programming III Knapsack Edit Distance",
    "lec22": "MIT 6.006 Lec 22 Dynamic Programming IV Guitar Fingering Tetris",
    "lec23": "MIT 6.006 Lec 23 Computational Complexity P vs NP",
    "lec24": "MIT 6.006 Lec 24 Topics in Algorithms Research",
}

MIT_6046_MAP = {
    "lec1.pdf": "MIT 6.046J Lec 01 Asymptotic Analysis Insertion Mergesort",
    "lec2.pdf": "MIT 6.046J Lec 02 Recurrences Substitution Master Method",
    "lec3.pdf": "MIT 6.046J Lec 03 Divide and Conquer Strassen Matrix Mult",
    "lec4.pdf": "MIT 6.046J Lec 04 Quicksort Randomized Algorithms",
    "lec5.pdf": "MIT 6.046J Lec 05 Linear Time Sorting Radix Sort Lower Bounds",
    "lec6.pdf": "MIT 6.046J Lec 06 Order Statistics Median Selection",
    "lec7.pdf": "MIT 6.046J Lec 07 Hashing Hash Functions Universal Hashing",
    "lec8.pdf": "MIT 6.046J Lec 08 Universal and Perfect Hashing",
    "lec9.pdf": "MIT 6.046J Lec 09 Relation of BSTs to Quicksort",
    "lec10.pdf": "MIT 6.046J Lec 10 Red Black Trees Rotations Insertions",
    "lec11.pdf": "MIT 6.046J Lec 11 Augmenting Data Structures Interval Trees",
    "lec12.pdf": "MIT 6.046J Lec 12 Skip Lists Randomized Search",
    "lec13.pdf": "MIT 6.046J Lec 13 Amortized Analysis Potential Method",
    "lec14.pdf": "MIT 6.046J Lec 14 Competitive Analysis Self Organizing Lists",
    "lec15.pdf": "MIT 6.046J Lec 15 Dynamic Programming Longest Common Subseq",
    "lec16.pdf": "MIT 6.046J Lec 16 Greedy Algorithms Minimum Spanning Trees",
    "lec17.pdf": "MIT 6.046J Lec 17 Shortest Paths Dijkstra BFS Properties",
    "lec18.pdf": "MIT 6.046J Lec 18 Shortest Paths Bellman Ford Linear Programming",
    "lec19.pdf": "MIT 6.046J Lec 19 All Pairs Shortest Paths Floyd Warshall Johnson",
    "dyn_multi_alg": "MIT 6.046J Lec 22 Dynamic Multithreaded Algorithms Work Span",
    "L24": "MIT 6.046J Lec 24 Cache Oblivious Algorithms and Memory Layouts",
}


def ingest_curated_mit_algorithms(ingester: UniversalKnowledgeIngester) -> int:
    """Ingests curated typed lecture PDFs for MIT 6.006 and 6.046J."""
    print("\n---> Ingesting Curated MIT Algorithms (6.006 & 6.046J)")
    total = 0
    all_lectures: Dict[Path, str] = {}

    # 6.006
    dir_6006 = CURRICULA_DIR / "MIT_6_006_Algorithms"
    if dir_6006.exists():
        for pdf in dir_6006.rglob("*.pdf"):
            name = pdf.name.lower()
            if "_orig" in name:
                continue
            for key, clean_title in MIT_6006_MAP.items():
                if f"_{key}." in name or f"_{key}_" in name:
                    all_lectures[pdf] = clean_title
                    break

    # 6.046J
    dir_6046 = CURRICULA_DIR / "MIT_6_046J_Algorithms"
    if dir_6046.exists():
        for pdf in dir_6046.rglob("*.pdf"):
            name = pdf.name.lower()
            for key, clean_title in MIT_6046_MAP.items():
                if key.lower() in name:
                    all_lectures[pdf] = clean_title
                    break

    for pdf_path, clean_title in sorted(all_lectures.items(), key=lambda x: x[1]):
        try:
            res = ingester.ingest_document(pdf_path, custom_title=clean_title)
            if res:
                total += 1
        except Exception as e:
            print(f"     [Error ingesting {pdf_path.name}]: {e}")

    print(f"     Ingested {total} curated MIT algorithm lectures.")
    return total


def main():
    print("=" * 60)
    print("🚀 Starting Comprehensive Knowledge Base Build for Robbie")
    print(f"Target Vault: {VAULT_DIR}")
    print("=" * 60)

    ingester = UniversalKnowledgeIngester(str(VAULT_DIR))
    total_docs = 0

    # Ingest core directory sources
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

    # Ingest curated MIT algorithms
    total_docs += ingest_curated_mit_algorithms(ingester)

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
