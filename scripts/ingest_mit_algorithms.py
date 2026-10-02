#!/usr/bin/env python3
"""
ingest_mit_algorithms.py - Curated Ingestion Engine for MIT 6.006 and MIT 6.046J.
Extracts primary typed lecture PDFs, maps them to human-readable academic titles,
and ingests them into the Robbie/2ndBrain vault adhering to Rule 6 (ADHD) and Rule 7 (No Slop).
"""

import sys
import re
from pathlib import Path
from typing import Dict, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
VAULT_DIR = REPO_ROOT / "vault_template"
CURRICULA_DIR = REPO_ROOT / "raw_materials" / "references" / "Academic_Curricula_and_University_Programs"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ingest_knowledge import UniversalKnowledgeIngester
from hybrid_search import HybridSearchEngine
from vault_linter import VaultLinter

# Mappings: regex pattern on filename -> human-readable topic name
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


def find_mit_6006_lectures() -> Dict[Path, str]:
    dir_path = CURRICULA_DIR / "MIT_6_006_Algorithms"
    results = {}
    if not dir_path.exists():
        return results

    for pdf in dir_path.rglob("*.pdf"):
        name = pdf.name.lower()
        if "_orig" in name:
            continue  # Skip handwritten scans, prefer typed notes
        for key, clean_title in MIT_6006_MAP.items():
            if f"_{key}." in name or f"_{key}_" in name:
                results[pdf] = clean_title
                break
    return results


def find_mit_6046_lectures() -> Dict[Path, str]:
    dir_path = CURRICULA_DIR / "MIT_6_046J_Algorithms"
    results = {}
    if not dir_path.exists():
        return results

    for pdf in dir_path.rglob("*.pdf"):
        name = pdf.name.lower()
        for key, clean_title in MIT_6046_MAP.items():
            if key.lower() in name:
                results[pdf] = clean_title
                break
    return results


def main():
    print("=" * 60)
    print("📚 Curated MIT Algorithms Knowledge Base Ingestion")
    print("   MIT 6.006 (Demaine/Devadas) & MIT 6.046J (Leiserson/Demaine)")
    print("=" * 60)

    lec_6006 = find_mit_6006_lectures()
    lec_6046 = find_mit_6046_lectures()

    print(f"Found {len(lec_6006)} typed lecture notes for MIT 6.006")
    print(f"Found {len(lec_6046)} typed lecture notes for MIT 6.046J")

    ingester = UniversalKnowledgeIngester(str(VAULT_DIR))
    total_ingested = 0

    all_lectures = {**lec_6006, **lec_6046}

    for pdf_path, clean_title in sorted(all_lectures.items(), key=lambda x: x[1]):
        try:
            res = ingester.ingest_document(pdf_path, custom_title=clean_title)
            if res:
                total_ingested += 1
        except Exception as e:
            print(f"     [Error ingesting {pdf_path.name}]: {e}")

    print("\n" + "=" * 60)
    print(f"🎉 Completed Ingesting {total_ingested} Curated Algorithm Lectures!")
    print("=" * 60)

    # Re-index hybrid search
    print("\n🧠 Updating SQLite Vector & BM25 Index...")
    try:
        search_engine = HybridSearchEngine(str(VAULT_DIR))
        cnt = search_engine.index_vault(force=False)
        print(f"✅ Indexed {cnt} notes into hybrid vector database.")
    except Exception as e:
        print(f"⚠️ Vector index note: {e}")

    # Run linter audit
    print("\n🔍 Auditing Vault Integrity...")
    try:
        linter = VaultLinter(str(VAULT_DIR))
        report = linter.audit(auto_fix=True)
        print(linter.format_report(report))
    except Exception as e:
        print(f"⚠️ Lint note: {e}")


if __name__ == "__main__":
    main()
