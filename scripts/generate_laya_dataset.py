"""
generate_laya_dataset.py - Synthetic Dataset Generator with Ambiguous Borderline Distillation

Features:
- Standard clear-cut domain samples across AST, Graphify, and Vault.
- Borderline / multi-intent mixed developer queries with soft teacher distributions.
- Diverse blast radius and memory distillation instances.
- Anchored to REPO_ROOT / "data" so execution works from any working directory.
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

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
    "What is the mathematical update formula for {class_name} in source code?",
    "Where is the state update equation for {func} implemented?",
    "Show the algorithm update formula in {file}",
    "Inspect the mathematical formula and code implementation of {func}",
    "Find the update step equation for {class_name} in the codebase",
    "What is the formula for {func} update in the code?",
    "What is the formula for FAST-LIO2 ikd-Tree update in source code?",
    "Show the covariance update formula in {class_name}",
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
    "What did the lead developer say about {preference_topic}?",
    "Search specifications in 01-Concepts for {decision_topic}",
    "Retrieve past architectural decisions about {decision_topic}",
    "What notes in the vault mention [[{note_link}]]?",
    "Find our documentation on {decision_topic} in 00-Meta",
    "What corrections did the user provide during last session?",
    "Look up the specification for {decision_topic}",
    "Check 03-Agent-Memory for rules on {preference_topic}",
    "What were the conclusions from the design review note?",
    "Search my Obsidian vault for references to {decision_topic}",
    "What did we agree on in ADR-002 regarding {decision_topic}?",
    "Retrieve past consultation notes about {decision_topic}",
    "What are the system rules documented in 00-Meta/System-Rules.md?",
    "Search notes tagged with #adr or #architecture",
    "Summarize the user constraints recorded in 03-Agent-Memory",
    "Did we decide to use {decision_topic} or an alternative?",
    "Find notes linking to [[{note_link}]] in the vault",
    "What was the reasoning behind our {decision_topic} choice?",
    "What user feedback was logged in the last session?",
    "Search the vault for markdown notes explaining {decision_topic}",
    "Check meeting transcripts for mentions of {preference_topic}",
    "What are the specific requirements in ISO 15288 clause 6.4?",
    "Find the exact verification table in ISO 29148",
    "Show the stakeholder needs clause in ISO 15288",
    "What is the definition of architecture viewpoint in ISO 42010?",
    "Retrieve the verbatim text of SEBoK Section 3 from 02-Sources",
    "Search vault sources for table of ergonomics processes in ISO 9241-220",
    "Look up the system life cycle definition in [[iso-15288-2023-Overview]]",
    "Find the exact reliability allocation formula in ANSI G-043B",
    "Retrieve the overview note for [[incose-systems-engineering-handbook-5th-edition-Overview]]",
    "Look up our task memory in 03-Agent-Memory/01-Tasks",
    "What facts did we record in 03-Agent-Memory/03-Facts?",
    "Search the vault for the exact definition of mission assurance",
    "Find notes in 02-Sources describing architecture evaluation criteria in ISO 42030",
    "What does ISO 25030 specify for software quality requirements verification?",
]

AMBIGUOUS_CHOICE_SAMPLES = [
    ("Explain the architecture of the AST parser and show who calls it", [0.46, 0.46, 0.08]),
    ("Search ADR-001 and trace callers of LayaDecisionEngine", [0.46, 0.08, 0.46]),
    ("How do the Obsidian vault notes connect to the system architecture graph?", [0.08, 0.46, 0.46]),
    ("Show me the source modules and their AST parse trees", [0.48, 0.10, 0.42]),
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
        "hybrid_search",
        "route_query",
        "score_blast_radius",
        "distill_turn",
        "ingest_file",
        "lint_vault",
        "sync_brain",
        "execute_tool",
        "parse_ast",
        "calculate_rrf_score",
        "extract_wikilinks",
        "load_laya_agent",
        "score_ingestion_novelty",
        "evaluate_conversational_turn",
        "build_sequence",
        "collate_items",
        "fit_temperature",
        "laserMapping",
        "CreateMapBuilder",
        "compute_odom",
        "ekf_predict",
        "ekf_update",
        "publish_cmd_vel",
        "solve_ik",
        "track_trajectory",
        "filter_scan",
        "register_frame",
    ],
    "file": [
        "scripts/laya_router.py",
        "scripts/hybrid_search.py",
        "scripts/mcp_server.py",
        "scripts/memory_distiller.py",
        "scripts/vault_linter.py",
        "scripts/sync_brain.py",
        "scripts/ingest_knowledge.py",
        "scripts/train_laya.py",
        "src/cartographer_ros/node.cpp",
        "src/FAST_LIO/laserMapping.cpp",
        "src/slam_toolbox/slam_toolbox.cpp",
        "src/nav2_controller/controller_server.cpp",
        "src/micro_ros/executor.c",
    ],
    "module": [
        "tree_sitter",
        "fastembed",
        "sqlite3",
        "laya",
        "transformers",
        "pydantic",
        "torch",
        "safetensors",
        "rclcpp",
        "rclpy",
        "nav2_core",
        "ros2_control",
        "micro_ros_msgs",
        "geometry_msgs",
        "sensor_msgs",
    ],
    "class_name": [
        "LayaDecisionEngine",
        "HybridSearcher",
        "MemoryDistiller",
        "VaultLinter",
        "MCPKnowledgeServer",
        "DecisionModel",
        "Agent",
        "LayaRouter",
        "LaserMapping",
        "SlamToolbox",
        "PurePursuitController",
        "SystemInterface",
        "BehaviorTreeEngine",
    ],
    "concept": [
        "Dual-Process Cognitive Architecture",
        "System 1 Fast Reflex",
        "System 2 Deep Grounding",
        "Multi-Engine Knowledge Stack",
        "Memory Distillation Lifecycle",
        "Semantic GraphRAG",
        "Code-Review-Graph AST Pipeline",
        "Hybrid Search RRF Fusion",
        "Obsidian Vault Knowledge Base",
        "ROS 2 Jazzy and Humble Middleware Architecture",
        "Nav2 Navigation Stack and Behavior Trees",
        "FAST-LIO2 and LIO-SAM 3D LiDAR Odometry",
        "Cartographer and SLAM Toolbox 2D Mapping",
        "ros2_control HardwareInterface and Lifecycle",
        "Gazebo Harmonic and ros_gz Bridge Architecture",
        "Fossen 6-DOF Equations of Motion and Hydrodynamics",
        "Acoustic Navigation USBL and DVL Integration",
        "Micro-ROS Firmware on STM32 and ESP32",
        "REP-103 Units and Coordinate Conventions",
        "REP-105 Coordinate Frames earth map odom base_link",
        "REP-147 Marine Vehicle Coordinate Conventions",
    ],
    "other_concept": [
        "Obsidian Markdown Vault",
        "Laya ModernBERT Router",
        "Tree-sitter SQLite Graph",
        "Agent Long-Term Memory",
        "Fastembed Vector Index",
        "Zero-Hallucination Grounding",
        "Continuous Learning Loop",
        "Community Detection Clusters",
        "Regulated Pure Pursuit Controller",
        "SmacPlanner2D Hybrid-A* Path Planning",
        "SocketCAN and CANopen Fieldbus Communication",
        "FreeRTOS Micro-ROS Architecture",
        "SensorDataQoS Reliability and Durability",
        "URDF and SDFormat Physics",
    ],
    "decision_topic": [
        "ModernBERT vs standard BERT",
        "fastembed vs sentence-transformers",
        "BM25 plus dense vector fusion",
        "local-first architecture vs cloud APIs",
        "atomic notes and wikilinks convention",
        "SQLite schema for AST",
        "temperature calibration bounds",
        "single commit repository policy",
        "Robbie YAGNI coding guardrail for single standard nodes",
        "closed-form analytical kinematics vs nonlinear solvers",
        "Regulated Pure Pursuit controller for outdoor path tracking",
        "Gazebo Harmonic ros_gz bridge stamped vs unstamped cmd_vel",
        "SingleThreadedExecutor vs MultiThreadedExecutor callback groups",
        "WSL2 Ubuntu 24.04 network configuration",
    ],
    "preference_topic": [
        "snake_case naming conventions",
        "never writing outside vault",
        "git commit history preservation",
        "single forward pass routing",
        "type annotations on Python functions",
        "Robbie minimal baseline under 150 LOC",
        "empirical source citations with line numbers",
        "ROS 2 REP standards compliance",
        "local data sovereignty in vault/03-Agent-Memory",
    ],
    "note_link": [
        "System-Architecture",
        "ADR-001-Laya-Router",
        "Fastembed-Retrieval-Spec",
        "Coding-Style-Guidelines",
        "Dual-Process-Cognitive-Model",
        "Agent-Memory-Spec",
        "Agent-Memory-MOC",
        "2026-10-01_local_first_4_pillar_agent_memory",
        "2026-10-03_gazebo_harmonic_and_ros_gz_simulation_architecture",
        "rep-0103-Overview",
        "rep-0105-Overview",
        "rep-0117-Overview",
        "rep-0119-Overview",
        "rep-0138-Overview",
    ],
}

SCORE_SAMPLES = [
    ("Adding a comment or docstring to scripts/hybrid_search.py", 1),
    ("Updating formatting and markdown table in README.md", 1),
    ("Fixing a typo in a log message inside vault_linter.py", 1),
    ("Adding blank lines for PEP8 compliance in scripts/sync_brain.py", 1),
    ("Updating repository description in package metadata", 1),
    ("Fixing spelling in code comments of laya_router.py", 1),
    ("Updating Markdown headings in SKILL.md", 1),
    ("Adding an optional helper parameter with a default value to route_query()", 2),
    ("Refactoring internal list comprehension into a generator inside ingest_file()", 2),
    ("Adding a new optional cache lookup step before calling the database", 2),
    ("Writing a unit test verifying hybrid search scoring logic", 2),
    ("Adding type hints to private helper functions in mcp_server.py", 2),
    ("Adding a new debug logging line inside sync_brain.py", 2),
    ("Adding an internal helper method _format_timestamp in memory_distiller.py", 2),
    ("Renaming the public method route_query to evaluate_query across laya_router.py", 3),
    ("Changing the return type of hybrid_search from List[str] to Dict[str, Any]", 3),
    ("Updating SQLite index creation to include an additional compound column", 3),
    ("Modifying default RRF k constant from 60 to 45 in hybrid search", 3),
    ("Refactoring memory distiller decay function from linear to exponential", 3),
    ("Changing default parameter values for UniversalKnowledgeIngester constructor", 3),
    ("Adding a required argument to the public search_vault API function", 3),
    ("Updating SQLite table schema for code-review-graph without running a migration", 4),
    ("Rewriting the tokenizer encoding pipeline in common.py used by all modules", 4),
    ("Changing the global database connection string and connection pooling protocol", 4),
    ("Switching embedding model weights from 384-dimensional to 768-dimensional", 4),
    ("Modifying the public MCP server protocol response schema", 4),
    ("Refactoring the base AbstractEngine class inherited by all search engines", 4),
    ("Replacing the underlying networkx graph representation with an incompatible C-binding", 4),
    ("Deleting scripts/hybrid_search.py and replacing it with an incompatible interface", 5),
    ("Dropping the primary tables in the active knowledge base SQLite database", 5),
    ("Modifying core AST parser grammar rules affecting all language bindings", 5),
    ("Hardcoding an invalid API token and removing environment variable fallback", 5),
    ("Removing the entire 03-Agent-Memory directory and wiping persistent rules", 5),
    ("Wiping all stored embeddings in vault_vectors.db without recovery backup", 5),
    ("Deleting the root models directory and breaking all runtime routing inference", 5),
]

NOUL_SAMPLES = [
    ("User: Always generate diagrams using Draw.io XML and high-res PNG, never Mermaid.", True),
    ("User: Remember that my project path is /workspace/second-brain.", True),
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
    ("User: What is the current version of PyTorch installed?", False),
]

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


def generate_dataset(output_dir: Optional[str] = None, train_ratio: float = 0.85):
    target_dir = Path(output_dir) if output_dir else (REPO_ROOT / "data")
    target_dir.mkdir(parents=True, exist_ok=True)
    all_samples: List[Dict[str, Any]] = []

    for _ in range(10):
        for pattern in AST_PATTERNS:
            all_samples.append(
                {
                    "state": fill_template(pattern),
                    "question": "Which engine should handle this knowledge query?",
                    "type": "choice",
                    "options": [
                        "CRG_AST_CALL_GRAPH",
                        "GRAPHIFY_SEMANTIC_GRAPHRAG",
                        "OBSIDIAN_VAULT_SEARCH",
                    ],
                    "target": "CRG_AST_CALL_GRAPH",
                }
            )
        for pattern in GRAPHIFY_PATTERNS:
            all_samples.append(
                {
                    "state": fill_template(pattern),
                    "question": "Which engine should handle this knowledge query?",
                    "type": "choice",
                    "options": [
                        "CRG_AST_CALL_GRAPH",
                        "GRAPHIFY_SEMANTIC_GRAPHRAG",
                        "OBSIDIAN_VAULT_SEARCH",
                    ],
                    "target": "GRAPHIFY_SEMANTIC_GRAPHRAG",
                }
            )
        for pattern in VAULT_PATTERNS:
            all_samples.append(
                {
                    "state": fill_template(pattern),
                    "question": "Which engine should handle this knowledge query?",
                    "type": "choice",
                    "options": [
                        "CRG_AST_CALL_GRAPH",
                        "GRAPHIFY_SEMANTIC_GRAPHRAG",
                        "OBSIDIAN_VAULT_SEARCH",
                    ],
                    "target": "OBSIDIAN_VAULT_SEARCH",
                }
            )

    for _ in range(10):
        for text, soft_dist in AMBIGUOUS_CHOICE_SAMPLES:
            all_samples.append(
                {
                    "state": text,
                    "question": "Which engine should handle this knowledge query?",
                    "type": "choice",
                    "options": [
                        "CRG_AST_CALL_GRAPH",
                        "GRAPHIFY_SEMANTIC_GRAPHRAG",
                        "OBSIDIAN_VAULT_SEARCH",
                    ],
                    "target_dist": soft_dist,
                    "target": (
                        "CRG_AST_CALL_GRAPH"
                        if soft_dist[0] == max(soft_dist)
                        else (
                            "GRAPHIFY_SEMANTIC_GRAPHRAG"
                            if soft_dist[1] == max(soft_dist)
                            else "OBSIDIAN_VAULT_SEARCH"
                        )
                    ),
                }
            )

    for _ in range(10):
        for text, score in SCORE_SAMPLES:
            all_samples.append(
                {
                    "state": text,
                    "question": "Rate the structural blast radius risk of this proposed modification",
                    "type": "score",
                    "range": [1, 5],
                    "target": score,
                }
            )

    for _ in range(10):
        for text, noul_val in NOUL_SAMPLES:
            all_samples.append(
                {
                    "state": text,
                    "question": "Does this turn contain an enduring user preference or rule to remember?",
                    "type": "noul",
                    "target": noul_val,
                }
            )

    for _ in range(10):
        for text, soft_dist in AMBIGUOUS_NOUL_SAMPLES:
            all_samples.append(
                {
                    "state": text,
                    "question": "Does this turn contain an enduring user preference or rule to remember?",
                    "type": "noul",
                    "target_dist": soft_dist,
                    "target": soft_dist[1] >= 0.5,
                }
            )

    random.seed(42)
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    train_path = target_dir / "train_data.jsonl"
    val_path = target_dir / "val_data.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s) + "\n")

    print(f"[Dataset Generator] Generated {len(train_samples)} train samples -> {train_path}")
    print(f"[Dataset Generator] Generated {len(val_samples)} val samples -> {val_path}")
    return str(train_path), str(val_path)


if __name__ == "__main__":
    generate_dataset()
