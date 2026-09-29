"""
generate_laya_dataset.py - Advanced Synthetic Dataset Generator with Ambiguous Borderline Distillation

Features:
- Standard clear-cut domain samples across AST, Graphify, and Vault.
- Borderline / multi-intent mixed developer queries with soft teacher distributions (e.g. 50/50 splits).
- Diverse blast radius and memory distillation instances.
"""

import os
import json
import random
from typing import List, Dict, Any

AST_PATTERNS = [
    "Find all callers of {func}",
    "Who calls {func} across the repository?",
    "Show me the call tree for {func}",
    "Which modules import {module}?",
    "What functions does {func} call internally?",
    "What is the code blast radius if I modify {func} in {file}?",
    "Trace callees of {class_name} constructor",
    "Find all references to class {class_name}",
    "Check if changing {file} will break any dependent functions",
    "List all functions defined inside {file}",
    "Who uses {func} in scripts?",
    "Does {file} have any circular dependencies with other scripts?",
    "Show me the function signature and callers for {func}",
    "Find every occurrence where {func} is invoked",
    "Trace the import chain of {module}",
    "Identify all callee nodes connected to {func}",
    "Which files depend directly on {file}?",
    "Show AST parse nodes for {class_name}",
    "Calculate the exact syntactic blast radius for editing {func}",
    "Where is {class_name} instantiated across the codebase?",
    "yo who calls {func}?",
    "inspect call hierarchy for {func}",
    "find all symbols imported from {module} in {file}",
    "which functions are affected if I delete {func}?",
    "give me the tree-sitter AST nodes for {file}",
    "is {func} called by any external scripts?",
    "show all usages of {func} across the whole project",
    "parse {file} and list all function declarations",
    "who is calling {class_name} in {file}?",
    "get callers of {func} and their line numbers",
    "show callers, callees, and imports of {file}",
    "check if {func} has any recursive calls",
    "find broken references after renaming {func}",
    "extract caller graph for {func} using tree-sitter",
    "where does {module} get imported in scripts?",
]

GRAPHIFY_PATTERNS = [
    "Explain the high-level architecture of {concept}",
    "What are the main concept clusters in the system?",
    "How does {concept} conceptually connect to {other_concept}?",
    "Show me the bridge nodes between {concept} and {other_concept}",
    "What are the central god nodes in our architecture graph?",
    "Give me an overview of the {concept} subsystem boundaries",
    "Explain the Leiden community detection clusters for our modules",
    "How does data flow from {concept} to {other_concept}?",
    "Summarize the conceptual relationships in the knowledge graph",
    "What are the peripheral vs central components of {concept}?",
    "Explain the architectural hierarchy between System 1 and System 2",
    "Map out the conceptual dependencies of {concept}",
    "How are domain concepts grouped into communities in graphify?",
    "Show the macro architecture overview of 2ndBrain",
    "Which modules act as semantic hubs in the knowledge graph?",
    "Explain the interaction design between {concept} and external agents",
    "What is the overall architecture of this system?",
    "Explain the structural boundaries of our multi-engine stack",
    "How does System 1 fast reflex hand off to System 2 deep grounding?",
    "Show community detection graph for the entire project",
    "Which files form the core architectural backbone?",
    "Explain the conceptual design of our memory distillation pipeline",
    "What are the key abstractions connecting {concept} and {other_concept}?",
    "Provide a high-level architectural summary of the repository",
    "How do the different layers interact in the dual-process model?",
    "Identify architectural bottlenecks and god nodes across the project",
    "What are the primary clusters identified by Louvain/Leiden in graphify?",
    "Give me an executive architecture overview of 2ndBrain",
]

VAULT_PATTERNS = [
    "What was our decision on {decision_topic}?",
    "Find the ADR concerning {decision_topic}",
    "Search vault notes for {decision_topic}",
    "What are the active user preferences regarding {preference_topic}?",
    "Recall the rule about {preference_topic} from our session notes",
    "Show me meeting transcripts discussing {decision_topic}",
    "What did Fabian say about {preference_topic}?",
    "Search specifications in 01-Concepts for {decision_topic}",
    "Retrieve past architectural decisions about {decision_topic}",
    "What notes in the vault mention [[{note_link}]]?",
    "Find our documentation on {decision_topic} in 00-Meta",
    "What corrections did the user provide during last session?",
    "Look up the specification for {decision_topic}",
    "Check 04-Agent-Memory for rules on {preference_topic}",
    "What were the conclusions from the design review note?",
    "Search my Obsidian vault for references to {decision_topic}",
    "What did we agree on in ADR-002 regarding {decision_topic}?",
    "Retrieve past consultation notes about {decision_topic}",
    "What are the system rules documented in 00-Meta/System-Rules.md?",
    "Search notes tagged with #adr or #architecture",
    "Summarize the user constraints recorded in 04-Agent-Memory",
    "Did we decide to use {decision_topic} or an alternative?",
    "Find notes linking to [[{note_link}]] in the vault",
    "What was the reasoning behind our {decision_topic} choice?",
    "What user feedback was logged in the last session?",
    "Search the vault for markdown notes explaining {decision_topic}",
    "Check meeting transcripts for mentions of {preference_topic}",
]

# Ambiguous multi-intent queries: Teaching the model soft distributions
# [CRG_AST, GRAPHIFY, VAULT]
AMBIGUOUS_CHOICE_SAMPLES = [
    ("Explain the architecture of the AST parser and show who calls it", [0.46, 0.46, 0.08]),
    ("Search ADR-001 and trace callers of LayaDecisionEngine", [0.46, 0.08, 0.46]),
    ("How do the Obsidian vault notes connect to the system architecture graph?", [0.08, 0.46, 0.46]),
    ("Show me the modules in 02-Codebase and their AST parse trees", [0.48, 0.10, 0.42]),
    ("Check the blast radius of changing our architecture decisions", [0.46, 0.46, 0.08]),
    ("Review the high level design of hybrid search and its callers", [0.46, 0.46, 0.08]),
    ("What does ADR-003 say about callers of SQLite database?", [0.46, 0.08, 0.46]),
    ("Map out the conceptual modules and list the functions in each", [0.46, 0.46, 0.08]),
    ("How does the memory distiller fit into the architecture notes?", [0.08, 0.46, 0.46]),
    ("Find the file defining hybrid search and summarize its architectural role", [0.46, 0.46, 0.08]),
    ("Who calls the functions described in the System-Architecture spec?", [0.46, 0.08, 0.46]),
    ("Summarize the Leiden clusters and check which functions are imported", [0.46, 0.46, 0.08]),
    ("Look up the meeting transcript where we discussed callers of route_query", [0.46, 0.08, 0.46]),
    ("Show the god nodes in graphify and the AST for each one", [0.46, 0.46, 0.08]),
    ("What are the architectural trade-offs recorded in the vault for code review graph?", [0.08, 0.46, 0.46]),
]

ENTITIES = {
    "func": [
        "hybrid_search", "route_query", "score_blast_radius", "distill_turn", 
        "ingest_file", "lint_vault", "sync_brain", "execute_tool", "parse_ast",
        "calculate_rrf_score", "extract_wikilinks", "load_laya_agent", "score_ingestion_novelty",
        "evaluate_conversational_turn", "build_sequence", "collate_items", "fit_temperature"
    ],
    "file": [
        "scripts/laya_router.py", "scripts/hybrid_search.py", "scripts/mcp_server.py",
        "scripts/memory_distiller.py", "scripts/vault_linter.py", "scripts/sync_brain.py",
        "scripts/ingest_knowledge.py", "scripts/train_laya.py"
    ],
    "module": [
        "tree_sitter", "fastembed", "sqlite3", "laya", "transformers", "pydantic", "torch", "safetensors"
    ],
    "class_name": [
        "LayaDecisionEngine", "HybridSearcher", "MemoryDistiller", "VaultLinter", "MCPKnowledgeServer",
        "DecisionModel", "Agent", "LayaRouter"
    ],
    "concept": [
        "Dual-Process Cognitive Architecture", "System 1 Fast Reflex", "System 2 Deep Grounding",
        "Multi-Engine Knowledge Stack", "Memory Distillation Lifecycle", "Semantic GraphRAG",
        "Code-Review-Graph AST Pipeline", "Hybrid Search RRF Fusion", "Obsidian Vault Knowledge Base"
    ],
    "other_concept": [
        "Obsidian Markdown Vault", "Laya ModernBERT Router", "Tree-sitter SQLite Graph",
        "Agent Long-Term Memory", "Fastembed Vector Index", "Zero-Hallucination Grounding",
        "Continuous Learning Loop", "Community Detection Clusters"
    ],
    "decision_topic": [
        "ModernBERT vs standard BERT", "fastembed vs sentence-transformers", "BM25 plus dense vector fusion",
        "local-first architecture vs cloud APIs", "atomic notes and wikilinks convention", "SQLite schema for AST",
        "temperature calibration bounds", "Draw.io vs Mermaid diagram rendering", "single commit repository policy"
    ],
    "preference_topic": [
        "snake_case naming conventions", "never writing outside vault_template", "git commit history preservation",
        "dark mode diagrams", "drawio vs mermaid format", "single forward pass routing", "type annotations on Python functions"
    ],
    "note_link": [
        "System-Architecture", "ADR-001-Laya-Router", "Fastembed-Retrieval-Spec", "Coding-Style-Guidelines",
        "Dual-Process-Cognitive-Model", "Agent-Memory-Spec"
    ]
}

SCORE_SAMPLES = [
    ("Adding a comment or docstring to scripts/hybrid_search.py", 1),
    ("Updating formatting and markdown table in README.md", 1),
    ("Fixing a typo in a log message inside vault_linter.py", 1),
    ("Adding blank lines for PEP8 compliance in scripts/sync_brain.py", 1),
    ("Updating repository description in package metadata", 1),
    ("Adding an optional helper parameter with a default value to route_query()", 2),
    ("Refactoring internal list comprehension into a generator inside ingest_file()", 2),
    ("Adding a new optional cache lookup step before calling the database", 2),
    ("Writing a unit test verifying hybrid search scoring logic", 2),
    ("Adding type hints to private helper functions in mcp_server.py", 2),
    ("Renaming the public method route_query to evaluate_query across laya_router.py", 3),
    ("Changing the return type of hybrid_search from List[str] to Dict[str, Any]", 3),
    ("Updating SQLite index creation to include an additional compound column", 3),
    ("Modifying default RRF k constant from 60 to 45 in hybrid search", 3),
    ("Refactoring memory distiller decay function from linear to exponential", 3),
    ("Updating SQLite table schema for code-review-graph without running a migration", 4),
    ("Rewriting the tokenizer encoding pipeline in common.py used by all modules", 4),
    ("Changing the global database connection string and connection pooling protocol", 4),
    ("Switching embedding model weights from 384-dimensional to 768-dimensional", 4),
    ("Modifying the public MCP server protocol response schema", 4),
    ("Deleting scripts/hybrid_search.py and replacing it with an incompatible interface", 5),
    ("Dropping the primary tables in the active knowledge base SQLite database", 5),
    ("Modifying core AST parser grammar rules affecting all language bindings", 5),
    ("Hardcoding an invalid API token and removing environment variable fallback", 5),
    ("Removing the entire 04-Agent-Memory directory and wiping persistent rules", 5)
]

NOUL_SAMPLES = [
    ("User: Always generate diagrams using Draw.io XML and high-res PNG, never Mermaid.", True),
    ("User: Remember that my project path is C:/Users/Fabian/Desktop/Second Brain.", True),
    ("User: When writing Python scripts, always use type annotations on public functions.", True),
    ("User: Correction: never commit model weights or .pt files to the repository.", True),
    ("User: Rule update: use fastembed with BAAI/bge-small-en-v1.5 for dense search.", True),
    ("User: Prefer snake_case for all script filenames and module names.", True),
    ("User: Never modify user_vault directly; always propose edits in staging.", True),
    ("User: Make sure all git updates amend the single root commit without intermediate commits.", True),
    ("User: Always verify model execution on CPU before falling back to heuristics.", True),
    ("User: Ensure all wikilinks in generated notes use double bracket notation [[Note-Name]].", True),
    ("User: What is 2 + 2?", False),
    ("User: Can you explain how quicksort works?", False),
    ("User: Show me the first 10 lines of README.md.", False),
    ("User: Please run git status.", False),
    ("User: Check what time it is.", False),
    ("User: Thanks, that looks good.", False),
    ("User: Where is the documentation located?", False),
    ("User: Could you list the files in the assets directory?", False),
    ("User: Why did the previous test fail?", False),
    ("User: Help me write a regex for email validation.", False),
    ("User: How does ModernBERT differ from classic RoBERTa?", False),
    ("User: What is the current version of PyTorch installed?", False)
]

# Ambiguous Noul cases: Mild suggestions or thoughts that are borderline
# [P(false), P(true)]
AMBIGUOUS_NOUL_SAMPLES = [
    ("User: I think maybe we could consider using black for formatting at some point.", [0.55, 0.45]),
    ("User: In some projects I usually like camelCase but let's see.", [0.58, 0.42]),
    ("User: That was a helpful explanation of the router.", [0.85, 0.15]),
    ("User: Could it be better to add caching here? Not sure yet.", [0.60, 0.40]),
    ("User: Interesting, let's keep that in mind for later.", [0.50, 0.50]),
    ("User: I might test this with another model next week.", [0.65, 0.35]),
    ("User: Perhaps we should document this somewhere.", [0.45, 0.55]),
    ("User: Let me think about whether we want to keep that rule.", [0.52, 0.48]),
]


def fill_template(template: str) -> str:
    res = template
    for key, values in ENTITIES.items():
        placeholder = "{" + key + "}"
        while placeholder in res:
            res = res.replace(placeholder, random.choice(values), 1)
    return res


def generate_dataset(output_dir: str = "data", train_ratio: float = 0.85):
    os.makedirs(output_dir, exist_ok=True)
    all_samples = []

    # 1. Clear-cut Routing Choice samples (~900 instances)
    for _ in range(10):
        for pattern in AST_PATTERNS:
            all_samples.append({
                "state": fill_template(pattern),
                "question": "Which engine should handle this knowledge query?",
                "type": "choice",
                "options": ["CRG_AST_CALL_GRAPH", "GRAPHIFY_SEMANTIC_GRAPHRAG", "OBSIDIAN_VAULT_SEARCH"],
                "target": "CRG_AST_CALL_GRAPH"
            })
        for pattern in GRAPHIFY_PATTERNS:
            all_samples.append({
                "state": fill_template(pattern),
                "question": "Which engine should handle this knowledge query?",
                "type": "choice",
                "options": ["CRG_AST_CALL_GRAPH", "GRAPHIFY_SEMANTIC_GRAPHRAG", "OBSIDIAN_VAULT_SEARCH"],
                "target": "GRAPHIFY_SEMANTIC_GRAPHRAG"
            })
        for pattern in VAULT_PATTERNS:
            all_samples.append({
                "state": fill_template(pattern),
                "question": "Which engine should handle this knowledge query?",
                "type": "choice",
                "options": ["CRG_AST_CALL_GRAPH", "GRAPHIFY_SEMANTIC_GRAPHRAG", "OBSIDIAN_VAULT_SEARCH"],
                "target": "OBSIDIAN_VAULT_SEARCH"
            })

    # 2. Ambiguous multi-intent Choice samples with soft distribution targets (~150 instances)
    for _ in range(10):
        for text, soft_dist in AMBIGUOUS_CHOICE_SAMPLES:
            all_samples.append({
                "state": text,
                "question": "Which engine should handle this knowledge query?",
                "type": "choice",
                "options": ["CRG_AST_CALL_GRAPH", "GRAPHIFY_SEMANTIC_GRAPHRAG", "OBSIDIAN_VAULT_SEARCH"],
                "target_dist": soft_dist,
                "target": "CRG_AST_CALL_GRAPH" if soft_dist[0] == max(soft_dist) else ("GRAPHIFY_SEMANTIC_GRAPHRAG" if soft_dist[1] == max(soft_dist) else "OBSIDIAN_VAULT_SEARCH")
            })

    # 3. Score samples (Blast Radius, ~250 instances)
    for _ in range(10):
        for text, score in SCORE_SAMPLES:
            all_samples.append({
                "state": text,
                "question": "Rate the structural blast radius risk of this proposed modification",
                "type": "score",
                "range": [1, 5],
                "target": score
            })

    # 4. Noul samples (Memory Distillation, ~220 instances)
    for _ in range(10):
        for text, noul_val in NOUL_SAMPLES:
            all_samples.append({
                "state": text,
                "question": "Does this turn contain an enduring user preference or rule to remember?",
                "type": "noul",
                "target": noul_val
            })

    # 5. Ambiguous Noul samples (~80 instances)
    for _ in range(10):
        for text, soft_dist in AMBIGUOUS_NOUL_SAMPLES:
            all_samples.append({
                "state": text,
                "question": "Does this turn contain an enduring user preference or rule to remember?",
                "type": "noul",
                "target_dist": soft_dist,
                "target": soft_dist[1] >= 0.5
            })

    random.seed(42)
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    train_path = os.path.join(output_dir, "train_data.jsonl")
    val_path = os.path.join(output_dir, "val_data.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s) + "\n")

    print(f"[Dataset Generator] Generated {len(train_samples)} train samples -> {train_path}")
    print(f"[Dataset Generator] Generated {len(val_samples)} val samples -> {val_path}")


if __name__ == "__main__":
    generate_dataset()
