"""
sync_brain.py - Master Orchestrator for 2ndBrain

Binds together:
1. Laya (System 1 Fast Reflex & Query Router)
2. Code-Review-Graph (AST Call Graph & Blast Radius)
3. Graphify (Semantic GraphRAG & Community Detection)
4. Obsidian (Markdown Vault with Bidirectional Memory)
5. Hybrid Search (Parent-Child Hierarchical Dense Vector + BM25 RRF)
6. Conversational Memory Distillation & Consolidation ("Sleep Cycle")
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from hybrid_search import HybridSearchEngine
    from laya_router import router
    from memory_distiller import ConversationalMemoryDistiller
    from vault_linter import VaultLinter
except ImportError:
    from .hybrid_search import HybridSearchEngine
    from .laya_router import router
    from .memory_distiller import ConversationalMemoryDistiller
    from .vault_linter import VaultLinter


def get_default_vault() -> str:
    """Dynamically resolves the vault directory across any OS, repository location, or environment."""
    env_root = os.environ.get("KNOWLEDGE_BASE_ROOT")
    if env_root and Path(env_root).exists():
        return str((Path(env_root) / "vault").resolve())

    env_vault = os.environ.get("SECOND_BRAIN_VAULT")
    if env_vault and Path(env_vault).exists():
        return str(Path(env_vault).resolve())

    local_template = current_dir.parent / "vault"
    if not local_template.exists():
        local_template = current_dir.parent / "vault_template"
    if local_template.exists():
        return str(local_template.resolve())

    cwd = Path.cwd()
    for candidate in [
        cwd / "vault",
        cwd / "vault_template",
        cwd / "Second Brain" / "vault",
        cwd / "Second Brain" / "vault_template",
    ]:
        if candidate.exists():
            return str(candidate.resolve())

    home_vault = Path.home() / ".se_brain" / "vault"
    if home_vault.exists():
        return str(home_vault.resolve())

    return str(local_template.resolve())


def get_default_target() -> str:
    """Dynamically resolves the repository target root across any OS or working directory."""
    for env_var in ("SECOND_BRAIN_TARGET", "KNOWLEDGE_BASE_ROOT"):
        val = os.environ.get(env_var)
        if val and Path(val).exists():
            return str(Path(val).resolve())
    return str(current_dir.parent.resolve())


DEFAULT_VAULT = get_default_vault()
DEFAULT_TARGET = get_default_target()


def resolve_binary(binary_name: str) -> str:
    """Finds binary on PATH or in active python environment (Windows Scripts/ or POSIX bin/)."""
    found = shutil.which(binary_name)
    if found:
        return found
    scripts_dir = Path(sys.prefix) / "Scripts"
    candidate_win = scripts_dir / f"{binary_name}.exe"
    if candidate_win.exists():
        return str(candidate_win)
    bin_dir = Path(sys.prefix) / "bin"
    candidate_posix = bin_dir / binary_name
    if candidate_posix.exists():
        return str(candidate_posix)
    return binary_name


def run_command(cmd, cwd=None):
    """Executes a subprocess safely with shell=False for argument lists."""
    if isinstance(cmd, list) and cmd:
        exec_cmd = list(cmd)
        exec_cmd[0] = resolve_binary(str(exec_cmd[0]))
        display_str = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in exec_cmd)
        print(f"[SyncBrain] Running: {display_str}")
        result = subprocess.run(
            exec_cmd,
            cwd=cwd,
            shell=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    else:
        print(f"[SyncBrain] Running: {cmd}")
        result = subprocess.run(
            cmd,
            cwd=cwd,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0 and "graph file not found" not in result.stderr:
        print(f"[Error] {result.stderr}", file=sys.stderr)
    return result


def _resolve_crg_repo(target_dir: str) -> str:
    """Resolves the code repository root, preferring source_materials when present."""
    t_path = Path(target_dir).resolve()
    sm_path = t_path / "source_materials"
    if sm_path.exists() and any(sm_path.iterdir()):
        return str(sm_path)
    return str(t_path)


def sync_agent_memory_moc(vault_dir: str):
    """Reconciles and audits Agent-Memory-MOC.md with disk state."""
    print("\n--- [0/3] Reconciling Agent Memory Map of Content (MOC) ---")
    try:
        distiller = ConversationalMemoryDistiller(vault_dir)
        distiller.reconcile_agent_memory_moc()
    except Exception as e:
        print(f"[SyncBrain] Warning: Could not reconcile Agent-Memory-MOC: {e}")


def sync_code_ast(target_dir: str) -> int:
    """Updates build/ast/graph.db with all code files including .hxx, .ino, .ipynb, .jl, and .m."""
    print("\n--- [1/3] Updating Structural Code Graph (AST) ---")

    build_dir = Path(get_default_vault()).parent / "build" / "ast"
    build_dir.mkdir(parents=True, exist_ok=True)
    db_path = build_dir / "graph.db"
    os.environ["CRG_DATA_DIR"] = str(build_dir.resolve())
    crg_repo = Path(_resolve_crg_repo(target_dir)).resolve()

    try:
        import hashlib
        import code_review_graph.parser as crg_parser
        import code_review_graph.graph as crg_graph

        crg_parser.EXTENSION_TO_LANGUAGE.update({".hxx": "cpp", ".hh": "cpp", ".ino": "cpp"})
        store = crg_graph.GraphStore(str(db_path))
        existing_files = set(store.get_all_files())

        code_exts = {
            ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx", ".hh",
            ".py", ".ino", ".jl", ".sh", ".lua", ".rb", ".js", ".ts",
            ".ipynb", ".m",
        }
        ignored_dirs = {
            ".git", ".github", ".gitlab", ".circleci", ".devcontainer",
            ".vscode", ".bazelci", "3rdparty", "thirdparty", "node_modules", "__pycache__",
        }

        missing_files = []
        for f in sorted(crg_repo.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in code_exts:
                continue
            if any(pt.lower() in ignored_dirs or (pt.startswith(".") and pt not in (".", "..")) for pt in f.parts[:-1]):
                continue
            if f.name.startswith(".") or f.name.startswith("~"):
                continue
            posix_p = f.resolve().as_posix()
            if posix_p not in existing_files:
                missing_files.append((f, posix_p))

        if not missing_files:
            total_f = len(existing_files)
            store.close()
            print(f"[SyncBrain] AST graph is up to date ({total_f} code files indexed).")
            return 0

        print(f"[SyncBrain] Parsing {len(missing_files)} new code files (.hxx, .ino, .ipynb, .jl, .m, .py)...")
        parser = crg_parser.CodeParser(crg_repo)
        batch = []
        added_count = 0
        for f, posix_p in missing_files:
            try:
                raw = f.read_bytes()
                fhash = hashlib.sha256(raw).hexdigest()
                lang = crg_parser.EXTENSION_TO_LANGUAGE.get(f.suffix.lower(), "code")
                nodes, edges = parser.parse_bytes(f, raw)
                if not any(n.kind == "File" for n in nodes):
                    line_cnt = max(1, raw.count(b"\n") + 1)
                    nodes.insert(
                        0,
                        crg_parser.NodeInfo(
                            kind="File",
                            name=posix_p,
                            file_path=posix_p,
                            line_start=1,
                            line_end=line_cnt,
                            language=lang,
                        ),
                    )
                batch.append((posix_p, nodes, edges, fhash))
                added_count += 1
                if len(batch) >= 200:
                    store.store_file_batch(batch)
                    batch.clear()
            except Exception as e:
                print(f"  [AST Parse Warning] {f.name}: {e}")

        if batch:
            store.store_file_batch(batch)
            batch.clear()

        total_after = len(store.get_all_files())
        store.close()
        print(f"[SyncBrain] Indexed {added_count} new code files into CRG AST (total: {total_after} files).")
        return added_count
    except ImportError:
        crg_bin = resolve_binary("code-review-graph")
        if shutil.which(crg_bin) is None and not Path(crg_bin).exists():
            print("[SyncBrain] Note: code-review-graph is not installed on PATH.")
            return 0
        crg_cmd = ["code-review-graph", "update", "--repo", str(crg_repo), "--data-dir", str(build_dir.resolve())]
        run_command(crg_cmd, cwd=str(crg_repo))
        return 0


def sync_semantic_graph(target_dir: str, vault_dir: str, update_only: bool = True) -> Dict[str, int]:
    """Synchronizes Graphify graph.json with all CRG AST code files and Vault Concept Overview nodes."""
    print("\n--- [2/3] Updating Semantic Knowledge Graph (Graphify + Vault Concept Bridge) ---")
    repo_root = Path(vault_dir).resolve().parent
    build_dir = repo_root / "build"
    graphify_out = build_dir / "graphify-out"
    graphify_out.mkdir(parents=True, exist_ok=True)

    nested_dup = graphify_out / "graphify-out"
    if nested_dup.exists() and nested_dup.is_dir():
        shutil.rmtree(nested_dup, ignore_errors=True)

    graph_json_path = graphify_out / "graph.json"
    sm_root = Path(_resolve_crg_repo(target_dir)).resolve()

    import json
    import sqlite3

    if graph_json_path.exists():
        try:
            with open(graph_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {"directed": False, "multigraph": False, "graph": {}, "nodes": [], "links": [], "hyperedges": []}
    else:
        data = {"directed": False, "multigraph": False, "graph": {}, "nodes": [], "links": [], "hyperedges": []}

    nodes = data.setdefault("nodes", [])
    links = data.setdefault("links", [])
    existing_node_ids = {n.get("id") for n in nodes if n.get("id")}
    existing_source_files = {n.get("source_file") for n in nodes if n.get("source_file")}

    code_added = 0
    concepts_bridged = 0

    # 1. Bridge any code files in build/ast/graph.db not yet represented in graph.json
    ast_db = build_dir / "ast" / "graph.db"
    if ast_db.exists():
        try:
            conn = sqlite3.connect(str(ast_db))
            rows = conn.execute(
                "SELECT kind, name, file_path, line_start FROM nodes"
            ).fetchall()
            conn.close()
            sm_posix = sm_root.as_posix().rstrip("/") + "/"
            for kind, name, fp, lstart in rows:
                if not fp:
                    continue
                rel_fp = fp[len(sm_posix):] if fp.startswith(sm_posix) else fp
                if rel_fp in existing_source_files and kind == "File":
                    continue
                if rel_fp not in existing_source_files:
                    label = Path(name).name if kind == "File" else name
                    nid = re.sub(r"[^a-z0-9]+", "_", f"{rel_fp}_{label}".lower()).strip("_")
                    if nid in existing_node_ids:
                        continue
                    nodes.append(
                        {
                            "id": nid,
                            "label": label,
                            "_callable": kind in ("Function", "Class"),
                            "_callable_class": kind == "Class",
                            "_origin": "ast",
                            "community": 1,
                            "community_name": "/".join(rel_fp.split("/")[:2]) or "Robotics Code",
                            "file_type": "code",
                            "norm_label": label.lower(),
                            "source_file": rel_fp,
                            "source_location": f"L{lstart or 1}",
                        }
                    )
                    existing_node_ids.add(nid)
                    existing_source_files.add(rel_fp)
                    code_added += 1
        except Exception as e:
            print(f"  [Graphify AST Bridge Note]: {e}")

    # 2. Bridge all ingested documents / Vault 01-Concepts/*-Overview.md notes into graph.json
    manifest_path = build_dir / ".ingest_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for norm_k, meta in manifest.items():
                if not isinstance(meta, dict):
                    continue
                slug = meta.get("slug", "")
                if not slug:
                    continue
                rel_sf = norm_k[len("source_materials/"):] if norm_k.startswith("source_materials/") else norm_k
                if rel_sf in existing_source_files:
                    continue
                nid = f"concept_{slug.replace('-', '_')}"
                if nid in existing_node_ids:
                    nid = f"{nid}_{ abs(hash(rel_sf)) % 100000 }"
                domain_parts = rel_sf.split("/")[:2]
                domain_name = "/".join(domain_parts) if domain_parts else "vault_concepts"
                hub_id = f"domain_hub_{re.sub(r'[^a-z0-9]+', '_', domain_name.lower()).strip('_')}"
                if hub_id not in existing_node_ids:
                    nodes.append(
                        {
                            "id": hub_id,
                            "label": domain_name,
                            "_callable": False,
                            "_callable_class": False,
                            "_origin": "vault_concept_bridge",
                            "community": 2,
                            "community_name": domain_name,
                            "file_type": "concept_hub",
                            "norm_label": domain_name.lower(),
                            "source_file": domain_name,
                            "source_location": "L1",
                        }
                    )
                    existing_node_ids.add(hub_id)

                label = slug.replace("-", " ").title()
                nodes.append(
                    {
                        "id": nid,
                        "label": label,
                        "_callable": False,
                        "_callable_class": False,
                        "_origin": "vault_concept_bridge",
                        "community": 2,
                        "community_name": domain_name,
                        "file_type": "document",
                        "norm_label": slug.replace("-", " "),
                        "source_file": rel_sf,
                        "vault_slug": f"{slug}-Overview",
                        "source_location": "L1",
                    }
                )
                links.append(
                    {
                        "source": nid,
                        "target": hub_id,
                        "relation": "documented_in_domain",
                        "_origin": "vault_concept_bridge",
                        "confidence": "EXTRACTED",
                        "confidence_score": 1.0,
                        "source_file": rel_sf,
                        "source_location": "L1",
                        "weight": 1.0,
                    }
                )
                existing_node_ids.add(nid)
                existing_source_files.add(rel_sf)
                concepts_bridged += 1
        except Exception as e:
            print(f"  [Graphify Concept Bridge Note]: {e}")

    if code_added > 0 or concepts_bridged > 0:
        with open(graph_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(
            f"[SyncBrain] Updated Graphify graph.json: +{code_added} AST code files, "
            f"+{concepts_bridged} Vault concept nodes (total nodes: {len(nodes)})."
        )
    else:
        print(f"[SyncBrain] Graphify graph.json is up to date ({len(nodes)} nodes).")

    ensure_graphify_sqlite_index(build_dir, graph_data=data if (code_added > 0 or concepts_bridged > 0) else None)
    return {"code_added": code_added, "concepts_bridged": concepts_bridged, "total_nodes": len(nodes)}


def ensure_graphify_sqlite_index(build_dir: Path, graph_data: Any = None) -> Path:
    """Builds or refreshes build/graphify-out/graph_index.db from graph.json for <50ms semantic queries."""
    import json
    import sqlite3

    graphify_out = Path(build_dir) / "graphify-out"
    graphify_out.mkdir(parents=True, exist_ok=True)
    graph_json_path = graphify_out / "graph.json"
    db_path = graphify_out / "graph_index.db"

    if not graph_json_path.exists() and graph_data is None:
        return db_path

    if (
        graph_data is None
        and db_path.exists()
        and graph_json_path.exists()
        and db_path.stat().st_mtime >= graph_json_path.stat().st_mtime
    ):
        try:
            conn = sqlite3.connect(str(db_path))
            cnt = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            conn.close()
            if cnt > 0:
                return db_path
        except Exception:
            pass

    if graph_data is None:
        with open(graph_json_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

    nodes = graph_data.get("nodes", [])
    links = graph_data.get("links", [])

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("DROP TABLE IF EXISTS nodes")
    conn.execute("DROP TABLE IF EXISTS links")
    conn.execute(
        """
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            label TEXT,
            norm_label TEXT,
            community_name TEXT,
            file_type TEXT,
            source_file TEXT,
            origin TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE links (
            source TEXT,
            target TEXT,
            relation TEXT,
            weight REAL
        )
        """
    )

    node_tuples = [
        (
            str(n.get("id", "")),
            str(n.get("label", "")),
            str(n.get("norm_label") or n.get("label", "")).lower(),
            str(n.get("community_name", "")),
            str(n.get("file_type", "")),
            str(n.get("source_file", "")),
            str(n.get("_origin", "")),
        )
        for n in nodes
        if n.get("id")
    ]
    link_tuples = [
        (
            str(lk.get("source", "")),
            str(lk.get("target", "")),
            str(lk.get("relation", "related_to")),
            float(lk.get("weight", 1.0) or 1.0),
        )
        for lk in links
        if lk.get("source") and lk.get("target")
    ]

    conn.executemany("INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)", node_tuples)
    conn.executemany("INSERT INTO links VALUES (?, ?, ?, ?)", link_tuples)
    conn.execute("CREATE INDEX idx_nodes_norm ON nodes(norm_label)")
    conn.execute("CREATE INDEX idx_nodes_sf ON nodes(source_file)")
    conn.execute("CREATE INDEX idx_links_src ON links(source)")
    conn.execute("CREATE INDEX idx_links_tgt ON links(target)")
    conn.commit()
    conn.close()
    return db_path


def extract_ast_symbol_and_pattern(query_text: str, default_pattern: str = "callers_of"):
    """Extracts the primary code symbol and AST query pattern from a natural language question or symbol."""
    q_clean = query_text.strip()
    q_lower = q_clean.lower()

    pattern = default_pattern
    if "who calls" in q_lower or "callers" in q_lower or "blast" in q_lower:
        pattern = "callers_of"
    elif "callee" in q_lower or "what does" in q_lower:
        pattern = "callees_of"
    elif "test" in q_lower:
        pattern = "tests_for"
    elif "import" in q_lower:
        pattern = "importers_of" if "who" in q_lower else "imports_of"
    elif "inherit" in q_lower or "subclass" in q_lower or "derive" in q_lower:
        pattern = "inheritors_of"
    elif "summary" in q_lower:
        pattern = "file_summary"

    # 1. Explicit single/double/backtick quoted symbol
    quoted = re.findall(r"[`'\"]([^`'\"]+)[`'\"]", q_clean)
    if quoted:
        return quoted[0].strip(), pattern

    # 2. Explicit keyword prefix: "function foo", "method bar", "class Baz", "calls computeVelocityCommands"
    kw_match = re.search(
        r"\b(?:module|class|struct|function|method|symbol|def|calls|callers\s+of|callees\s+of|for)\s+([a-zA-Z_][a-zA-Z0-9_:]*)",
        q_clean,
        re.IGNORECASE,
    )
    if kw_match:
        cand = kw_match.group(1).strip(":")
        if cand.lower() not in ("the", "a", "an", "in", "on", "of", "to"):
            return cand, pattern

    # 3. Prefer camelCase, PascalCase, snake_case, or C++ qualified (::) identifiers
    tokens = [w.strip("?\"'.,()[]{}") for w in q_clean.split() if w.strip("?\"'.,()[]{}")]
    stop_words = {
        "who", "what", "which", "where", "when", "how", "does", "do", "is", "are",
        "the", "in", "of", "for", "to", "from", "with", "and", "or", "calls",
        "caller", "callers", "callee", "callees", "function", "method", "class",
        "module", "file", "work", "used", "use", "find", "show", "list", "trace",
        "blast", "radius", "impact", "depend", "depends", "dependents",
    }
    code_like = []
    for tok in tokens:
        if tok.lower() in stop_words:
            continue
        if "::" in tok or "_" in tok or (any(c.isupper() for c in tok[1:]) and any(c.islower() for c in tok)):
            code_like.append(tok)
    if code_like:
        return code_like[0], pattern

    filtered = [t for t in tokens if t.lower() not in stop_words]
    return (filtered[-1] if filtered else (tokens[-1] if tokens else ".")), pattern


def _like_escape(s: str) -> str:
    """Escapes special SQLite LIKE characters (%, _, \\)."""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def query_code_ast(
    query_text_or_symbol: str,
    build_dir: Path,
    crg_repo: str = "",
    pattern: Any = None,
) -> str:
    """
    Executes a deterministic AST query directly against build/ast/graph.db in <15ms,
    automatically disambiguating overloaded C++/Python methods across all matching definitions.
    """
    import sqlite3

    symbol, inferred_pattern = extract_ast_symbol_and_pattern(
        query_text_or_symbol, default_pattern=pattern or "callers_of"
    )
    active_pattern = pattern or inferred_pattern

    if not symbol or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_:.]*", symbol):
        return ""

    ast_db = Path(build_dir) / "ast" / "graph.db"
    if not ast_db.exists():
        return ""

    short_sym = symbol.split("::")[-1].split(".")[-1]
    if not short_sym or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", short_sym):
        return ""

    try:
        conn = sqlite3.connect(str(ast_db))
        defs = conn.execute(
            """
            SELECT kind, name, qualified_name, file_path, line_start, line_end
            FROM nodes
            WHERE name = ?
            ORDER BY CASE WHEN kind IN ('Function', 'Class') THEN 0 ELSE 1 END, file_path
            LIMIT 200
            """,
            (short_sym,),
        ).fetchall()

        if not defs:
            defs = conn.execute(
                """
                SELECT kind, name, qualified_name, file_path, line_start, line_end
                FROM nodes
                WHERE name LIKE ? ESCAPE '\\'
                LIMIT 20
                """,
                (f"%{_like_escape(short_sym)}%",),
            ).fetchall()

        # Edge endpoints are stored as bare names (unresolved) or full qualified names (indexed).
        keys = [short_sym, symbol] + [d[2] for d in defs]
        keys = list(dict.fromkeys(k for k in keys if k))
        ph = ",".join("?" for _ in keys)

        if active_pattern == "callees_of":
            edges = conn.execute(
                f"""
                SELECT kind, source_qualified, target_qualified, file_path, line
                FROM edges
                WHERE source_qualified IN ({ph}) AND kind IN ('CALLS', 'IMPORTS_FROM', 'INHERITS')
                ORDER BY CASE WHEN target_qualified LIKE '%TEST(%' OR file_path LIKE '%/test%' OR file_path LIKE '%benchmark%' THEN 1 ELSE 0 END, line
                LIMIT 35
                """,
                keys,
            ).fetchall()
        elif active_pattern in ("imports_of", "importers_of"):
            col = "source_qualified" if active_pattern == "imports_of" else "target_qualified"
            edges = conn.execute(
                f"""
                SELECT kind, source_qualified, target_qualified, file_path, line
                FROM edges
                WHERE {col} IN ({ph}) AND kind = 'IMPORTS_FROM'
                ORDER BY line
                LIMIT 35
                """,
                keys,
            ).fetchall()
        elif active_pattern == "inheritors_of":
            edges = conn.execute(
                f"""
                SELECT kind, source_qualified, target_qualified, file_path, line
                FROM edges
                WHERE target_qualified IN ({ph}) AND kind = 'INHERITS'
                ORDER BY line
                LIMIT 35
                """,
                keys,
            ).fetchall()
        elif active_pattern == "tests_for":
            edges = conn.execute(
                f"""
                SELECT kind, source_qualified, target_qualified, file_path, line
                FROM edges
                WHERE source_qualified IN ({ph}) AND kind = 'TESTED_BY'
                ORDER BY line
                LIMIT 35
                """,
                keys,
            ).fetchall()
        else:
            edges = conn.execute(
                f"""
                SELECT kind, source_qualified, target_qualified, file_path, line
                FROM edges
                WHERE target_qualified IN ({ph}) AND kind IN ('CALLS', 'INHERITS', 'IMPORTS_FROM')
                ORDER BY CASE WHEN kind = 'CALLS' THEN 0 ELSE 1 END,
                         CASE WHEN source_qualified LIKE '%TEST(%' OR file_path LIKE '%/test%' OR file_path LIKE '%benchmark%' THEN 1 ELSE 0 END,
                         line
                LIMIT 40
                """,
                keys,
            ).fetchall()
        conn.close()
    except Exception as e:
        return f"[AST SQLite Error]: {e}"

    if not defs and not edges:
        return ""

    def _short(qname: str) -> str:
        if "::" in qname and "/" in qname.split("::", 1)[0]:
            return qname.split("::", 1)[1]
        if "source_materials/" in qname:
            return qname.split("source_materials/")[-1]
        return qname

    def _rel(fp: str) -> str:
        return fp.split("source_materials/")[-1] if "source_materials/" in fp else fp

    out = [f"=== AST Code Graph Result ({active_pattern}: {symbol}) ==="]
    if defs:
        out.append(f"\n[Definitions Found ({len(defs)})]:")
        for kind, name, qname, fp, lstart, lend in defs[:12]:
            out.append(f"  - [{kind}] `{_short(qname)}` ({_rel(fp)}:L{lstart}-L{lend})")

    if edges:
        out.append(f"\n[AST Structural Edges ({len(edges)})]:")
        for ekind, src_q, tgt_q, fp, line in edges[:25]:
            out.append(f"  - [{ekind}] `{_short(src_q)}` -> `{_short(tgt_q)}` ({_rel(fp)}:L{line})")
    else:
        out.append("\n[AST Structural Edges]: No direct incoming/outgoing edges recorded for this symbol.")

    return "\n".join(out)


def query_semantic_graph(
    query_text: str,
    build_dir: Path,
    vault_dir: str,
    top_k: int = 6,
) -> str:
    """
    Executes a <1.0s Semantic GraphRAG query using build/graphify-out/graph_index.db
    for community/topology relationships fused with HybridSearchEngine parent sections.
    """
    import sqlite3

    db_path = ensure_graphify_sqlite_index(Path(build_dir))
    words = [w.lower() for w in re.findall(r"\w{3,}", query_text)]
    stop_words = {
        "what", "where", "when", "which", "who", "how", "why", "does", "are", "the",
        "and", "for", "with", "from", "into", "between", "explain", "describe",
        "architecture", "overview", "difference", "work", "works", "used", "using",
    }
    terms = [w for w in words if w not in stop_words]
    if not terms:
        # If no meaningful terms remain after stripping stop-words, do not perform arbitrary substring matches
        try:
            h_engine = HybridSearchEngine(vault_dir)
            matches = h_engine.search(query_text, top_k=4, return_parent_context=True)
            if matches:
                out = ["=== Graphify Semantic GraphRAG Result ===", "\n[Grounded Vault Sections (Parent-Child Hybrid RRF)]:"]
                for m in matches:
                    sec = m.get("section_title", "General Section")
                    out.append(
                        f"- **[[{m['stem']}]]** > *{sec}* "
                        f"(Semantic: {m['semantic_similarity']}%, RRF: {m['rrf_score']}):\n"
                        f"  {m['snippet']}..."
                    )
                return "\n".join(out)
        except Exception:
            pass
        return "=== Graphify Semantic GraphRAG Result ===\nNo specific concepts identified for semantic graph exploration."

    matched_nodes = []
    neighbor_edges = []
    if db_path.exists() and terms:
        try:
            conn = sqlite3.connect(str(db_path))
            like_clauses = " OR ".join("(norm_label LIKE ? ESCAPE '\\' OR source_file LIKE ? ESCAPE '\\')" for _ in terms[:4])
            params = []
            for t in terms[:4]:
                esc_t = f"%{_like_escape(t)}%"
                params.extend([esc_t, esc_t])
            rows = conn.execute(
                f"""
                SELECT id, label, community_name, file_type, source_file, origin
                FROM nodes
                WHERE {like_clauses}
                LIMIT 400
                """,
                params,
            ).fetchall()

            best_by_label: Dict[str, tuple] = {}
            for nid, label, comm, ftype, sfile, orig in rows:
                ll = (label or "").lower()
                sfl = (sfile or "").lower()
                label_hits = sum(1 for t in terms if t in ll)
                path_hits = sum(1 for t in terms if t in sfl)
                sc = label_hits * 4.0 + path_hits * 0.5
                if orig == "vault_concept_bridge" or ftype in ("document", "concept_hub"):
                    sc += 2.0
                if "/test" in sfl or "test_" in sfl:
                    sc -= 1.5
                key = ll.strip()
                if key not in best_by_label or sc > best_by_label[key][0]:
                    best_by_label[key] = (sc, nid, label, comm, ftype, sfile)
            scored_nodes = sorted(best_by_label.values(), key=lambda x: x[0], reverse=True)
            matched_nodes = scored_nodes[:top_k]

            if matched_nodes:
                top_ids = [m[1] for m in matched_nodes[:5]]
                ph = ",".join("?" for _ in top_ids)
                neighbor_edges = conn.execute(
                    f"""
                    SELECT l.source, n1.label, l.relation, l.target, n2.label, n2.community_name
                    FROM links l
                    LEFT JOIN nodes n1 ON l.source = n1.id
                    LEFT JOIN nodes n2 ON l.target = n2.id
                    WHERE l.source IN ({ph}) OR l.target IN ({ph})
                    LIMIT 20
                    """,
                    top_ids + top_ids,
                ).fetchall()
            conn.close()
        except Exception:
            pass

    out = ["=== Graphify Semantic GraphRAG Result ==="]
    if matched_nodes:
        out.append(f"\n[Knowledge Graph Nodes ({len(matched_nodes)} matched communities/entities)]:")
        for _, nid, label, comm, ftype, sfile in matched_nodes:
            out.append(f"  - **{label}** [Community: `{comm}` | Type: `{ftype}` | Source: `{sfile}`]")
    if neighbor_edges:
        out.append(f"\n[1-Hop Graph Topology Links ({min(12, len(neighbor_edges))} edges)]:")
        for src_id, src_lbl, rel, tgt_id, tgt_lbl, tgt_comm in neighbor_edges[:12]:
            out.append(f"  - `{src_lbl or src_id}` --({rel})--> `{tgt_lbl or tgt_id}` [{tgt_comm or 'Graph'}]")

    try:
        h_engine = HybridSearchEngine(vault_dir)
        matches = h_engine.search(query_text, top_k=4, return_parent_context=True)
        if matches:
            out.append(
                f"\n[Grounded Vault Sections ({len(matches)} sections via Parent-Child Hybrid RRF)]:"
            )
            for m in matches:
                sec = m.get("section_title", "General Section")
                out.append(
                    f"- **[[{m['stem']}]]** > *{sec}* "
                    f"(Semantic: {m['semantic_similarity']}%, RRF: {m['rrf_score']}):\n"
                    f"  {m['snippet']}..."
                )
    except Exception:
        pass

    return "\n".join(out)


def sync_vector_embeddings(vault_dir: str, force: bool = False):
    """Indexes and updates local Parent-Child hierarchical vector embeddings via fastembed."""
    print("\n--- [3/3] Updating Hierarchical Vector Embeddings (fastembed) ---")
    try:
        engine = HybridSearchEngine(vault_dir)
        count = engine.index_vault(force=force)
        print(
            f"[SyncBrain] Successfully indexed {count} new/modified notes into Parent-Child vector database."
        )
    except Exception as e:
        print(f"[SyncBrain] Note on embeddings: {e}")


def route_and_query(query_text: str, vault_dir: str, target_dir: str):
    """Evaluates query with Laya System 1 and routes to the best engine."""
    print("\n[SyncBrain] Evaluating query through Laya (System 1)...")
    decision = router.route_query(query_text)
    print(f" -> Routed to: {decision['decision']} (Confidence: {decision['confidence']})")
    print(f" -> Rationale: {decision['reasoning']}\n")

    distiller = ConversationalMemoryDistiller(vault_dir)
    memories = distiller.recall_relevant_memories(query_text)
    if memories:
        print("[SyncBrain] Recalled Active Memories:")
        for m in memories:
            print(f"   {m}")
        print()

    engine = decision["decision"]

    build_dir = Path(get_default_vault()).parent / "build"
    os.environ["CRG_DATA_DIR"] = str((build_dir / "ast").resolve())
    crg_repo = _resolve_crg_repo(target_dir)

    def search_vault(v_dir: str, q_text: str):
        try:
            h_engine = HybridSearchEngine(v_dir)
            matches = h_engine.search(q_text, top_k=4, return_parent_context=True)
            if matches:
                print(
                    f"\n[Hierarchical Knowledge Matches ({len(matches)} sections found via Parent-Child Search)]:"
                )
                for m in matches:
                    sec = m.get("section_title", "General Section")
                    print(
                        f"- **[[{m['stem']}]]** > *{sec}* "
                        f"(Semantic: {m['semantic_similarity']}%, RRF: {m['rrf_score']}):\n"
                        f"  {m['snippet']}...\n"
                    )
                return
        except Exception:
            pass

        v = Path(v_dir)
        scored = []
        tokens = [t.lower() for t in re.findall(r"\w+", q_text) if len(t) > 3]
        for f in v.glob("**/*.md"):
            try:
                txt = f.read_text(encoding="utf-8", errors="ignore")
                score = sum(txt.lower().count(tok) for tok in tokens)
                if score > 0:
                    scored.append((score, f, txt))
            except Exception:
                continue
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            print("No matching notes found in vault.")
            return
        print(f"\n[Vault Keyword Matches ({len(scored)} notes found)]:")
        for cnt, f, text in scored[:4]:
            clean = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL).strip()
            lines = [l.strip() for l in clean.splitlines() if l.strip() and not l.startswith("#")]
            snippet = " ".join(lines[:3])[:300]
            print(f"- **[[{f.stem}]]** ({f.parent.name}):\n  {snippet}...\n")

    if engine == "CRG_AST_CALL_GRAPH":
        print("[Executing AST Query via build/ast/graph.db]:")
        ast_text = query_code_ast(query_text, build_dir, crg_repo)
        if ast_text:
            print(ast_text)
        else:
            print("[AST Query returned no matches - Falling back to Obsidian Vault Search]")
            search_vault(vault_dir, query_text)
    elif engine == "OBSIDIAN_VAULT_SEARCH":
        search_vault(vault_dir, query_text)
    else:
        print("[Executing Semantic GraphRAG Query via graph_index.db + Hybrid Search]:")
        print(query_semantic_graph(query_text, build_dir, vault_dir))


def main():
    parser = argparse.ArgumentParser(description="2ndBrain - Dynamic Local Second Brain Orchestrator")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    sync_p = subparsers.add_parser("sync", help="Synchronize AST, Semantic Graph, and Obsidian Vault")
    sync_p.add_argument("--target", default=DEFAULT_TARGET, help="Target code/doc directory to index")
    sync_p.add_argument("--vault", default=DEFAULT_VAULT, help=f"Path to Obsidian vault (default: {DEFAULT_VAULT})")
    sync_p.add_argument("--full", action="store_true", help="Force full rebuild instead of incremental update")

    query_p = subparsers.add_parser("query", help="Route and execute a query across the Second Brain")
    query_p.add_argument("text", help="Question or prompt to answer")
    query_p.add_argument("--vault", default=DEFAULT_VAULT, help=f"Path to Obsidian vault (default: {DEFAULT_VAULT})")
    query_p.add_argument("--target", default=DEFAULT_TARGET, help="Target codebase directory")

    distill_p = subparsers.add_parser(
        "distill", help="Distill conversational transcript into durable memory notes"
    )
    distill_p.add_argument("--transcript", required=True, help="Path to transcript.jsonl")
    distill_p.add_argument("--vault", default=DEFAULT_VAULT, help=f"Path to Obsidian vault (default: {DEFAULT_VAULT})")

    session_p = subparsers.add_parser(
        "session", help="Archive a conversation dialogue or meeting into persistent memory"
    )
    session_p.add_argument("--topic", required=True, help="Topic or title of the session")
    session_p.add_argument("--summary", required=True, help="Executive summary of key discussions")
    session_p.add_argument("--decisions", nargs="*", default=[], help="Key takeaways or decisions reached")
    session_p.add_argument("--vault", default=DEFAULT_VAULT, help=f"Path to Obsidian vault (default: {DEFAULT_VAULT})")

    cons_p = subparsers.add_parser(
        "consolidate", help="Run Memory Consolidation sleep cycle (prune, merge, promote)"
    )
    cons_p.add_argument(
        "--dry-run", action="store_true", help="Preview consolidation actions without modifying vault"
    )
    cons_p.add_argument("--vault", default=DEFAULT_VAULT, help=f"Path to Obsidian vault (default: {DEFAULT_VAULT})")

    ingest_p = subparsers.add_parser(
        "ingest", help="Ingest external materials (PDFs, docs) into Obsidian vault"
    )
    ingest_p.add_argument("--source", required=True, help="Directory containing documents to ingest")
    ingest_p.add_argument("--vault", default=DEFAULT_VAULT, help=f"Path to Obsidian vault (default: {DEFAULT_VAULT})")

    lint_p = subparsers.add_parser(
        "lint", help="Audit vault for broken links, island notes, and schema compliance"
    )
    lint_p.add_argument("--vault", default=DEFAULT_VAULT, help=f"Path to Obsidian vault (default: {DEFAULT_VAULT})")
    lint_p.add_argument(
        "--fix", action="store_true", help="Auto-repair high-confidence broken links and register orphans"
    )

    blast_p = subparsers.add_parser("blast", help="Compute AST blast radius and callers for a symbol or file")
    blast_p.add_argument("symbol", nargs="?", default="", help="Function, class, or method name to trace")
    blast_p.add_argument("--file", help="File to trace blast radius for")
    blast_p.add_argument("--target", default=DEFAULT_TARGET, help="Target repository directory")

    remember_p = subparsers.add_parser("remember", help="Explicitly crystallize a learned rule or decision")
    remember_p.add_argument("rule", help="Rule or decision statement")
    remember_p.add_argument(
        "--category",
        choices=["task", "decision", "fact", "skill"],
        default=None,
        help="Memory pillar: task, decision, fact, or skill",
    )
    remember_p.add_argument("--tag", default="general", help="Category tag (e.g. security, architecture)")
    remember_p.add_argument("--vault", default=DEFAULT_VAULT, help=f"Path to Obsidian vault (default: {DEFAULT_VAULT})")

    args = parser.parse_args()

    if args.command == "sync":
        sync_agent_memory_moc(args.vault)
        sync_code_ast(args.target)
        sync_semantic_graph(args.target, args.vault, update_only=not args.full)
        sync_vector_embeddings(args.vault, force=args.full)
        print("\n[SyncBrain] Sync complete! Your Second Brain vault is up to date.")
    elif args.command == "query":
        route_and_query(args.text, args.vault, args.target)
    elif args.command == "distill":
        distiller = ConversationalMemoryDistiller(args.vault)
        count = distiller.process_transcript_jsonl(args.transcript)
        print(f"\n[SyncBrain] Distillation complete! Crystallized {count} durable memory notes.")
    elif args.command == "session":
        distiller = ConversationalMemoryDistiller(args.vault)
        saved = distiller.archive_session(
            topic=args.topic, summary=args.summary, key_decisions=args.decisions
        )
        print(f"\n[SyncBrain] Session archived successfully: {saved.name}")
    elif args.command == "consolidate":
        distiller = ConversationalMemoryDistiller(args.vault)
        result = distiller.consolidate_memories(dry_run=args.dry_run)
        mode_str = "PREVIEW (Dry Run)" if result["dry_run"] else "APPLIED"
        print(f"\n--- [SyncBrain] Memory Consolidation ({mode_str}) ---")
        print(f"  • Archived (Superseded & Decayed): {result['archived_count']}")
        print(f"  • Promoted to Architecture Guidelines: {result['promoted_count']}")
        print(f"  • Merged Redundancies: {result['merged_count']}")
        print(f"  • Active Memory Rules Remaining: {result['active_count']}")
        print(f"  • Report: {result['report_path']}")
    elif args.command == "ingest":
        try:
            from ingest_knowledge import UniversalKnowledgeIngester
        except ImportError:
            from .ingest_knowledge import UniversalKnowledgeIngester
        ingester = UniversalKnowledgeIngester(args.vault)
        ingester.ingest_directory(Path(args.source))
    elif args.command == "lint":
        linter = VaultLinter(args.vault)
        report = linter.audit(auto_fix=args.fix)
        print(linter.format_report(report))
    elif args.command == "blast":
        target_sym = args.symbol or (Path(args.file).stem if args.file else ".")
        build_dir = Path(get_default_vault()).parent / "build"
        print(f"\n[SyncBrain] Computing AST blast radius for '{target_sym}'...")
        print(
            query_code_ast(target_sym, build_dir, _resolve_crg_repo(args.target), pattern="callers_of")
            or f"No AST blast radius found for {target_sym}."
        )
    elif args.command == "remember":
        distiller = ConversationalMemoryDistiller(args.vault)
        cat = args.category or (args.tag if args.tag in ("task", "decision", "fact", "skill") else None)
        saved = distiller.distill_turn(
            args.rule, f"Explicitly remembered via CLI with tag: {args.tag}", category=cat
        )
        if saved:
            print(f"[SyncBrain] Successfully crystallized rule into Obsidian: {saved.name}")
        else:
            print("[SyncBrain] Rule was evaluated as duplicate or transient.")
    else:
        parser.print_help()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
