"""
hybrid_search.py - Local-First Hybrid Search Engine for 2ndBrain

Features:
1. Normalized Parent-Child Hierarchical Chunk Retrieval:
   - Stores Parent Sections once in `parent_sections` (capped at <=2800 chars).
   - Stores Child Chunks (~380 chars) and 384-d embeddings in `child_chunks`.
   - Excludes index/meta hubs (`00-Meta/*`, `*-MOC.md`, `README.md`, `Agent-Memory-MOC.md`).
   - Indexes `*-Overview.md`, `*-Source.md`, and `03-Agent-Memory/` 4-pillar notes.
2. Dense Semantic Vector Search via fastembed (BAAI/bge-small-en-v1.5, 384-d, 100% offline)
   with vectorized NumPy cosine similarity.
3. Sparse Keyword / BM25 Scoring with exact token weighting.
4. Reciprocal Rank Fusion (RRF) for merged ranking and lazy parent context loading.
"""

import argparse
import contextlib
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    import numpy as np
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
    np = None


class HybridSearchEngine:
    MAX_PARENT_CHARS = 2800
    TARGET_PARENT_PART_CHARS = 2200

    def __init__(self, vault_dir: str, db_path: Optional[str] = None):
        self.vault_dir = Path(vault_dir).resolve()
        self.db_path = Path(db_path) if db_path else self.vault_dir / "00-Meta" / "vault_vectors.db"
        self._model: Any = None
        self._init_db()

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Opens a SQLite connection and guarantees deterministic handle closure on Windows."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Initializes normalized SQLite tables for parent sections, child chunks, and note embeddings."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS note_embeddings (
                    rel_path TEXT PRIMARY KEY,
                    stem TEXT,
                    title TEXT,
                    snippet TEXT,
                    content TEXT,
                    embedding BLOB,
                    mtime REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_stem ON note_embeddings(stem)")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS parent_sections (
                    parent_id TEXT PRIMARY KEY,
                    rel_path TEXT,
                    stem TEXT,
                    title TEXT,
                    section_header TEXT,
                    parent_content TEXT,
                    file_mtime REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ps_rel_path ON parent_sections(rel_path)")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS child_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    parent_id TEXT,
                    rel_path TEXT,
                    child_content TEXT,
                    embedding BLOB,
                    file_mtime REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cc_rel_path ON child_chunks(rel_path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cc_parent_id ON child_chunks(parent_id)")

            # Create backward-compatible view if legacy physical table is not present
            existing_hc = conn.execute(
                "SELECT type FROM sqlite_master WHERE name = 'hierarchical_chunks'"
            ).fetchone()
            if existing_hc is None:
                conn.execute("""
                    CREATE VIEW IF NOT EXISTS hierarchical_chunks AS
                    SELECT c.chunk_id, c.rel_path, p.stem, p.title,
                           p.section_header AS section_title,
                           c.parent_id, c.child_content, p.parent_content,
                           c.embedding, c.file_mtime AS mtime
                    FROM child_chunks c
                    LEFT JOIN parent_sections p ON c.parent_id = p.parent_id
                """)

    def migrate_legacy_db(self) -> Dict[str, Any]:
        """
        Migrates a legacy physical `hierarchical_chunks` table into normalized
        `parent_sections` + `child_chunks` tables in place, prunes excluded meta/MOC rows,
        replaces `hierarchical_chunks` with a view, and runs VACUUM.
        """
        before_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        if not self.db_path.exists():
            return {"before_mb": 0.0, "after_mb": 0.0, "parent_sections": 0, "child_chunks": 0}

        new_db_path = self.db_path.with_name(f"{self.db_path.stem}_migrated.db")
        if new_db_path.exists():
            new_db_path.unlink()

        where_keep = """
            rel_path NOT LIKE '00-Meta/%'
            AND rel_path NOT LIKE '%-MOC.md'
            AND LOWER(rel_path) NOT LIKE '%readme.md'
            AND LOWER(rel_path) NOT LIKE '%agent-memory-moc.md'
        """

        conn = sqlite3.connect(new_db_path)
        try:
            conn.execute("PRAGMA journal_mode=OFF")
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA cache_size=-200000")

            conn.execute("""
                CREATE TABLE note_embeddings (
                    rel_path TEXT PRIMARY KEY,
                    stem TEXT,
                    title TEXT,
                    snippet TEXT,
                    content TEXT,
                    embedding BLOB,
                    mtime REAL
                )
            """)
            conn.execute("""
                CREATE TABLE parent_sections (
                    parent_id TEXT PRIMARY KEY,
                    rel_path TEXT,
                    stem TEXT,
                    title TEXT,
                    section_header TEXT,
                    parent_content TEXT,
                    file_mtime REAL
                )
            """)
            conn.execute("""
                CREATE TABLE child_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    parent_id TEXT,
                    rel_path TEXT,
                    child_content TEXT,
                    embedding BLOB,
                    file_mtime REAL
                )
            """)

            conn.execute("ATTACH DATABASE ? AS legacy", (str(self.db_path),))
            hc_row = conn.execute(
                "SELECT type FROM legacy.sqlite_master WHERE name = 'hierarchical_chunks'"
            ).fetchone()

            conn.execute("BEGIN")
            conn.execute(f"""
                INSERT OR REPLACE INTO main.note_embeddings
                (rel_path, stem, title, snippet, content, embedding, mtime)
                SELECT rel_path, stem, title, snippet, SUBSTR(content, 1, 3000), embedding, mtime
                FROM legacy.note_embeddings
                WHERE {where_keep}
            """)

            if hc_row and hc_row[0] == "table":
                conn.execute(f"""
                    INSERT OR REPLACE INTO main.parent_sections
                    (parent_id, rel_path, stem, title, section_header, parent_content, file_mtime)
                    SELECT parent_id,
                           rel_path,
                           stem,
                           title,
                           section_title,
                           SUBSTR(parent_content, 1, 2800),
                           MAX(mtime)
                    FROM legacy.hierarchical_chunks
                    WHERE {where_keep}
                    GROUP BY parent_id
                """)
                conn.execute(f"""
                    INSERT OR REPLACE INTO main.child_chunks
                    (chunk_id, parent_id, rel_path, child_content, embedding, file_mtime)
                    SELECT chunk_id, parent_id, rel_path, child_content, embedding, mtime
                    FROM legacy.hierarchical_chunks
                    WHERE {where_keep}
                """)
            else:
                conn.execute(f"""
                    INSERT OR REPLACE INTO main.parent_sections
                    SELECT * FROM legacy.parent_sections
                    WHERE {where_keep}
                """)
                conn.execute(f"""
                    INSERT OR REPLACE INTO main.child_chunks
                    SELECT * FROM legacy.child_chunks
                    WHERE {where_keep}
                """)

            conn.execute("CREATE INDEX idx_parent_rel_path ON parent_sections(rel_path)")
            conn.execute("CREATE INDEX idx_child_rel_path ON child_chunks(rel_path)")
            conn.execute("CREATE INDEX idx_child_parent_id ON child_chunks(parent_id)")
            conn.execute("""
                CREATE VIEW hierarchical_chunks AS
                SELECT c.chunk_id, c.rel_path, p.stem, p.title,
                       p.section_header AS section_title,
                       c.parent_id, c.child_content, p.parent_content,
                       c.embedding, c.file_mtime AS mtime
                FROM child_chunks c
                LEFT JOIN parent_sections p ON c.parent_id = p.parent_id
            """)
            conn.commit()

            migrated_parents = conn.execute("SELECT COUNT(*) FROM main.parent_sections").fetchone()[0]
            migrated_children = conn.execute("SELECT COUNT(*) FROM main.child_chunks").fetchone()[0]
            conn.execute("DETACH DATABASE legacy")
        finally:
            conn.close()

        os.replace(new_db_path, self.db_path)
        after_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "before_mb": round(before_bytes / (1024 * 1024), 2),
            "after_mb": round(after_bytes / (1024 * 1024), 2),
            "parent_sections": migrated_parents,
            "child_chunks": migrated_children,
        }

    @property
    def model(self):
        """Lazy-loads the embedding model on first call; degrades cleanly if offline."""
        if self._model is None and FASTEMBED_AVAILABLE:
            try:
                self._model = TextEmbedding(
                    model_name="BAAI/bge-small-en-v1.5",
                    threads=os.cpu_count() or 8,
                )
            except Exception as e:
                print(f"[HybridSearch] Warning: FastEmbed model unavailable ({e}). Falling back to BM25.")
                self._model = False
        return self._model if self._model is not False else None

    def _is_indexable_note(self, file_path: Path) -> bool:
        """
        Determines whether a vault markdown file should be embedded in `vault_vectors.db`.
        Includes `-Overview.md`, `-Source.md`, and `03-Agent-Memory/` content notes.
        Excludes `00-Meta/*`, `*-MOC.md`, `README.md`, and `Agent-Memory-MOC.md`.
        """
        if not file_path.exists() or file_path.suffix.lower() != ".md":
            return False
        try:
            rel = file_path.resolve().relative_to(self.vault_dir)
        except ValueError:
            return False

        if any(part.startswith(".") for part in rel.parts):
            return False

        name_lower = file_path.name.lower()
        if name_lower in (
            "readme.md",
            "index.md",
            "index.local.md",
            "system-rules.md",
            "agent-memory-moc.md",
        ):
            return False

        if file_path.name.endswith("-MOC.md"):
            return False

        if rel.parts and rel.parts[0] == "00-Meta":
            return False

        return True

    def _clean_text(self, markdown_text: str) -> Tuple[str, str, str]:
        """Extracts title, clean content (no frontmatter), and concise snippet."""
        clean = re.sub(r"^---\s*\n.*?\n---\s*\n", "", markdown_text, flags=re.DOTALL).strip()
        lines = [l.strip() for l in clean.splitlines() if l.strip()]

        title = ""
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
                break

        body_lines = [l for l in lines if not l.startswith("#")]
        snippet = " ".join(body_lines[:4])[:350] if body_lines else clean[:350]
        return title, clean, snippet

    def _split_oversized_block(self, text: str, max_part_chars: int = 2200) -> List[str]:
        """
        Splits an oversized block using double newlines, then single newlines,
        then hard character slices so no parent section exceeds MAX_PARENT_CHARS.
        """
        units: List[str] = []
        for para in text.split("\n\n"):
            p_str = para.strip()
            if not p_str:
                continue
            if len(p_str) <= max_part_chars:
                units.append(p_str)
            else:
                for line in p_str.split("\n"):
                    l_str = line.strip()
                    if not l_str:
                        continue
                    if len(l_str) <= max_part_chars:
                        units.append(l_str)
                    else:
                        for start in range(0, len(l_str), max_part_chars):
                            units.append(l_str[start:start + max_part_chars])

        parts: List[str] = []
        buf: List[str] = []
        buf_len = 0
        for u in units:
            sep_len = 1 if buf else 0
            if buf and (buf_len + sep_len + len(u) > max_part_chars):
                parts.append("\n".join(buf)[: self.MAX_PARENT_CHARS])
                buf = [u]
                buf_len = len(u)
            else:
                buf.append(u)
                buf_len += sep_len + len(u)

        if buf:
            parts.append("\n".join(buf)[: self.MAX_PARENT_CHARS])

        return parts or [text[: self.MAX_PARENT_CHARS]]

    def _decompose_into_parent_sections(self, clean_content: str, default_title: str) -> List[Tuple[str, str]]:
        """
        Decomposes markdown content into Parent Sections based on markdown headings
        (H1, H2, H3) and enforces a strict <= 2800 char ceiling per parent section.
        """
        if not clean_content.strip():
            return [(default_title, "")]

        header_pattern = re.compile(r"(?m)^(#{1,3}\s+[^\n]+)")
        splits = header_pattern.split(clean_content)

        sections: List[Tuple[str, str]] = []
        current_title = default_title

        if splits and not header_pattern.match(splits[0]):
            lead_text = splits[0].strip()
            if lead_text:
                sections.append((default_title, lead_text))
            splits = splits[1:]

        i = 0
        while i < len(splits):
            chunk = splits[i].strip()
            if header_pattern.match(chunk):
                sec_heading = re.sub(r"^#{1,3}\s+", "", chunk).strip()
                body = splits[i + 1].strip() if i + 1 < len(splits) else ""
                i += 2
                full_sec = f"{chunk}\n\n{body}".strip()
                sections.append((sec_heading, full_sec))
            else:
                if chunk:
                    sections.append((current_title, chunk))
                i += 1

        if not sections:
            sections = [(default_title, clean_content)]

        refined_sections: List[Tuple[str, str]] = []
        for sec_title, sec_body in sections:
            if len(sec_body) > self.MAX_PARENT_CHARS:
                sub_parts = self._split_oversized_block(sec_body, max_part_chars=self.TARGET_PARENT_PART_CHARS)
                for idx, part_text in enumerate(sub_parts, start=1):
                    label = f"{sec_title} (Part {idx})" if len(sub_parts) > 1 else sec_title
                    refined_sections.append((label, part_text[: self.MAX_PARENT_CHARS]))
            else:
                refined_sections.append((sec_title, sec_body))

        return refined_sections

    def _split_into_child_chunks(self, parent_text: str, target_size: int = 380, overlap: int = 60) -> List[str]:
        """Splits a parent section into small child chunks (~380 chars) preserving sentence/line boundaries."""
        if len(parent_text) <= target_size:
            return [parent_text]

        sentences = re.split(r"(?<=[.!?\n])\s+", parent_text)
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue

            if len(s_clean) > target_size:
                if current_chunk:
                    joined = " ".join(current_chunk).strip()
                    if joined:
                        chunks.append(joined)
                    current_chunk = []
                    current_len = 0
                for start in range(0, len(s_clean), target_size - overlap):
                    chunks.append(s_clean[start:start + target_size])
                continue

            if current_len + len(s_clean) > target_size and current_chunk:
                joined = " ".join(current_chunk).strip()
                if joined:
                    chunks.append(joined)
                overlap_words = " ".join(current_chunk)[-overlap:].split()
                current_chunk = overlap_words[1:] if len(overlap_words) > 1 else []
                current_len = sum(len(w) + 1 for w in current_chunk)

            current_chunk.append(s_clean)
            current_len += len(s_clean) + 1

        if current_chunk:
            joined = " ".join(current_chunk).strip()
            if joined and (not chunks or joined != chunks[-1]):
                chunks.append(joined)

        return chunks or [parent_text[:target_size]]

    def _build_note_records(
        self, file_path: Path, rel_path: str, file_mtime: float
    ) -> Optional[Tuple[Tuple[Any, ...], List[Tuple[Any, ...]], List[Dict[str, Any]], List[str]]]:
        """Builds normalized parent_sections, child_chunks, and embed_texts for a single note."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None

        title, clean_content, snippet = self._clean_text(content)
        title = title or file_path.stem
        file_stem = file_path.stem

        parent_sections = self._decompose_into_parent_sections(clean_content, title)
        parent_rows: List[Tuple[Any, ...]] = []
        child_records: List[Dict[str, Any]] = []
        embed_texts: List[str] = []

        is_source_note = rel_path.startswith("02-Sources/") or file_stem.endswith("-Source")
        c_target = 1800 if is_source_note else 380
        c_overlap = 150 if is_source_note else 60

        for sec_idx, (sec_title, parent_text) in enumerate(parent_sections):
            parent_id = f"{rel_path}#sec-{sec_idx}"
            parent_rows.append(
                (parent_id, rel_path, file_stem, title, sec_title, parent_text, file_mtime)
            )
            children = self._split_into_child_chunks(parent_text, target_size=c_target, overlap=c_overlap)
            for c_idx, child_text in enumerate(children):
                chunk_id = f"{parent_id}-c{c_idx}"
                embed_input = f"{title} > {sec_title}\n{child_text}".strip()
                embed_texts.append(embed_input)
                child_records.append(
                    {
                        "chunk_id": chunk_id,
                        "parent_id": parent_id,
                        "rel_path": rel_path,
                        "child_content": child_text,
                        "file_mtime": file_mtime,
                    }
                )

        note_meta = (rel_path, file_stem, title, snippet, clean_content[:3000], file_mtime)
        return note_meta, parent_rows, child_records, embed_texts

    def index_note(self, file_path: Path, force: bool = False) -> bool:
        """Indexes a single note into normalized `parent_sections` and `child_chunks` tables."""
        if not self._is_indexable_note(file_path):
            return False

        rel_path = file_path.resolve().relative_to(self.vault_dir).as_posix()
        file_mtime = file_path.stat().st_mtime

        with self._connect() as conn:
            row = conn.execute(
                "SELECT file_mtime FROM parent_sections WHERE rel_path = ? LIMIT 1",
                (rel_path,),
            ).fetchone()
            if row and not force and row[0] is not None and abs(float(row[0]) - file_mtime) < 1e-4:
                return False

        built = self._build_note_records(file_path, rel_path, file_mtime)
        if built is None:
            return False
        note_meta, parent_rows, child_records, embed_texts = built

        embeddings_bytes: List[Optional[bytes]] = [None] * len(child_records)
        if self.model and np is not None and embed_texts:
            try:
                for idx, vec in enumerate(self.model.embed(embed_texts)):
                    embeddings_bytes[idx] = np.array(vec, dtype=np.float32).tobytes()
            except Exception as e:
                print(f"[HybridSearch] Note embedding warning for {rel_path}: {e}")

        with self._connect() as conn:
            conn.execute("DELETE FROM parent_sections WHERE rel_path = ?", (rel_path,))
            conn.execute("DELETE FROM child_chunks WHERE rel_path = ?", (rel_path,))

            conn.executemany(
                """
                INSERT OR REPLACE INTO parent_sections
                (parent_id, rel_path, stem, title, section_header, parent_content, file_mtime)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                parent_rows,
            )
            conn.executemany(
                """
                INSERT OR REPLACE INTO child_chunks
                (chunk_id, parent_id, rel_path, child_content, embedding, file_mtime)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        rec["chunk_id"],
                        rec["parent_id"],
                        rec["rel_path"],
                        rec["child_content"],
                        embeddings_bytes[i],
                        rec["file_mtime"],
                    )
                    for i, rec in enumerate(child_records)
                ],
            )

            primary_emb = embeddings_bytes[0] if embeddings_bytes else None
            rel_p, f_stem, title, snippet, clean_trunc, f_mt = note_meta
            conn.execute(
                """
                INSERT OR REPLACE INTO note_embeddings
                (rel_path, stem, title, snippet, content, embedding, mtime)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (rel_p, f_stem, title, snippet, clean_trunc, primary_emb, f_mt),
            )

        return True

    def index_vault(self, force: bool = False) -> int:
        """Scans and indexes all eligible markdown notes in the vault using normalized Parent-Child chunks."""
        all_md_files = [
            p for p in self.vault_dir.rglob("*.md")
            if self._is_indexable_note(p)
        ]

        existing_rel_paths = {p.resolve().relative_to(self.vault_dir).as_posix() for p in all_md_files}
        existing_mtimes: Dict[str, float] = {}

        with self._connect() as conn:
            for tbl, col in (
                ("parent_sections", "file_mtime"),
                ("child_chunks", "file_mtime"),
                ("note_embeddings", "mtime"),
            ):
                rows = conn.execute(f"SELECT rel_path, MAX({col}) FROM {tbl} GROUP BY rel_path").fetchall()
                for dp, mt in rows:
                    if dp not in existing_rel_paths:
                        conn.execute(f"DELETE FROM {tbl} WHERE rel_path = ?", (dp,))
                    elif tbl == "parent_sections" and mt is not None:
                        existing_mtimes[dp] = float(mt)

        pending_files: List[Tuple[Path, str, float]] = []
        restored_from_cache = 0
        emb_cache_db = self.vault_dir.parent / "build" / ".embedding_cache.db"
        cached_contents: Dict[str, str] = {}
        if emb_cache_db.exists() and not force:
            try:
                c_conn = sqlite3.connect(str(emb_cache_db))
                for rp, c_txt in c_conn.execute("SELECT rel_path, content FROM note_embeddings"):
                    if rp and c_txt:
                        cached_contents[rp] = c_txt
                c_conn.close()
            except Exception:
                cached_contents = {}

        reusable_from_cache: List[Tuple[str, float]] = []
        for p in all_md_files:
            rel_p = p.resolve().relative_to(self.vault_dir).as_posix()
            f_mtime = p.stat().st_mtime
            if not force:
                known_mtime = existing_mtimes.get(rel_p)
                if known_mtime is not None and abs(known_mtime - f_mtime) < 1e-4:
                    continue
                if rel_p in cached_contents:
                    try:
                        raw_md = p.read_text(encoding="utf-8", errors="ignore")
                        _, clean_c, _ = self._clean_text(raw_md)
                        if clean_c[:3000] == cached_contents[rel_p]:
                            reusable_from_cache.append((rel_p, f_mtime))
                            continue
                    except Exception:
                        pass
            pending_files.append((p, rel_p, f_mtime))

        if reusable_from_cache and emb_cache_db.exists():
            try:
                with self._connect() as conn:
                    conn.execute("ATTACH DATABASE ? AS emb_cache", (str(emb_cache_db),))
                    conn.execute(
                        "CREATE TEMP TABLE IF NOT EXISTS reusable_notes "
                        "(rel_path TEXT PRIMARY KEY, new_mtime REAL)"
                    )
                    conn.execute("DELETE FROM temp.reusable_notes")
                    conn.executemany(
                        "INSERT OR REPLACE INTO temp.reusable_notes VALUES (?, ?)",
                        reusable_from_cache,
                    )
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO main.note_embeddings
                        (rel_path, stem, title, snippet, content, embedding, mtime)
                        SELECT n.rel_path, n.stem, n.title, n.snippet, n.content, n.embedding, r.new_mtime
                        FROM emb_cache.note_embeddings n
                        INNER JOIN temp.reusable_notes r ON n.rel_path = r.rel_path
                        """
                    )
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO main.parent_sections
                        (parent_id, rel_path, stem, title, section_header, parent_content, file_mtime)
                        SELECT p.parent_id, p.rel_path, p.stem, p.title, p.section_header, p.parent_content, r.new_mtime
                        FROM emb_cache.parent_sections p
                        INNER JOIN temp.reusable_notes r ON p.rel_path = r.rel_path
                        """
                    )
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO main.child_chunks
                        (chunk_id, parent_id, rel_path, child_content, embedding, file_mtime)
                        SELECT c.chunk_id, c.parent_id, c.rel_path, c.child_content, c.embedding, r.new_mtime
                        FROM emb_cache.child_chunks c
                        INNER JOIN temp.reusable_notes r ON c.rel_path = r.rel_path
                        """
                    )
                    conn.commit()
                    conn.execute("DETACH DATABASE emb_cache")
                    restored_from_cache = len(reusable_from_cache)
                    print(
                        f"[HybridSearch] Restored {restored_from_cache} unchanged notes from deterministic embedding cache."
                    )
            except Exception as e:
                print(f"[HybridSearch] Cache restore fallback: {e}")
                reusable_set = {rp for rp, _ in reusable_from_cache}
                for p in all_md_files:
                    rel_p = p.resolve().relative_to(self.vault_dir).as_posix()
                    if rel_p in reusable_set:
                        pending_files.append((p, rel_p, p.stat().st_mtime))
                restored_from_cache = 0

        if not pending_files:
            return restored_from_cache

        total_pending = len(pending_files)
        print(f"[HybridSearch] Indexing {total_pending} new/modified notes in batches...")

        indexed_count = 0
        batch_size = 128
        for b_start in range(0, total_pending, batch_size):
            batch = pending_files[b_start:b_start + batch_size]
            batch_parent_rows: List[Tuple[Any, ...]] = []
            batch_child_records: List[Dict[str, Any]] = []
            batch_embed_texts: List[str] = []
            batch_note_records: List[Tuple[str, str, str, str, str, float, int]] = []

            for p, rel_path, file_mtime in batch:
                built = self._build_note_records(p, rel_path, file_mtime)
                if built is None:
                    continue
                note_meta, parent_rows, child_records, embed_texts = built
                first_chunk_idx = len(batch_child_records)
                batch_parent_rows.extend(parent_rows)
                batch_child_records.extend(child_records)
                batch_embed_texts.extend(embed_texts)
                rel_p, f_stem, title, snippet, clean_trunc, f_mt = note_meta
                batch_note_records.append(
                    (rel_p, f_stem, title, snippet, clean_trunc, f_mt, first_chunk_idx)
                )

            if not batch_note_records:
                continue

            embeddings_bytes: List[Optional[bytes]] = [None] * len(batch_child_records)
            if self.model and np is not None and batch_embed_texts:
                try:
                    for idx, vec in enumerate(self.model.embed(batch_embed_texts, batch_size=256)):
                        embeddings_bytes[idx] = np.array(vec, dtype=np.float32).tobytes()
                except Exception as e:
                    print(f"[HybridSearch] Batch embedding warning at {b_start}: {e}")

            with self._connect() as conn:
                rel_tuples = [(nr[0],) for nr in batch_note_records]
                conn.executemany("DELETE FROM parent_sections WHERE rel_path = ?", rel_tuples)
                conn.executemany("DELETE FROM child_chunks WHERE rel_path = ?", rel_tuples)

                conn.executemany(
                    """
                    INSERT OR REPLACE INTO parent_sections
                    (parent_id, rel_path, stem, title, section_header, parent_content, file_mtime)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch_parent_rows,
                )
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO child_chunks
                    (chunk_id, parent_id, rel_path, child_content, embedding, file_mtime)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            rec["chunk_id"],
                            rec["parent_id"],
                            rec["rel_path"],
                            rec["child_content"],
                            embeddings_bytes[i],
                            rec["file_mtime"],
                        )
                        for i, rec in enumerate(batch_child_records)
                    ],
                )
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO note_embeddings
                    (rel_path, stem, title, snippet, content, embedding, mtime)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            rel_p,
                            f_stem,
                            t_str,
                            snip,
                            c_text,
                            embeddings_bytes[first_idx] if first_idx < len(embeddings_bytes) else None,
                            f_mt,
                        )
                        for rel_p, f_stem, t_str, snip, c_text, f_mt, first_idx in batch_note_records
                    ],
                )

            indexed_count += len(batch_note_records)
            print(
                f"[HybridSearch] Progress: {indexed_count}/{total_pending} notes "
                f"({len(batch_child_records)} child chunks in batch)"
            )

        return restored_from_cache + indexed_count

    def search(
        self,
        query: str,
        top_k: int = 5,
        return_parent_context: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Executes Two-Stage Coarse-to-Fine Parent-Child Hierarchical Search across the vault:
        1. Stage 1 (Coarse Filter): Scores `note_embeddings` (1 row per note, no JOIN) via vectorized
           BGE cosine similarity + title/stem/snippet keyword matching, plus fast SQLite `LIKE`
           matching on `child_chunks` for top query tokens, selecting the top ~150 candidate notes.
        2. Stage 2 (Fine RRF): Loads `child_chunks` joined with `parent_sections` ONLY for those
           candidate notes, computes exact BM25 + dense cosine Reciprocal Rank Fusion (RRF),
           deduplicates by `parent_id`, and fetches `parent_content` only for the final `top_k` hits.
        """
        query_words = [w.lower() for w in re.findall(r"\w{3,}", query)]
        if not query or not query.strip() or not re.search(r"\w", query):
            return []

        q_unit = None
        expected_bytes = 0

        if self.model and np is not None:
            try:
                query_vectors = list(self.model.embed([query]))
                if query_vectors:
                    q_vec = np.asarray(query_vectors[0], dtype=np.float32)
                    q_norm = float(np.linalg.norm(q_vec)) or 1.0
                    q_unit = q_vec / q_norm
                    expected_bytes = q_unit.shape[0] * 4
            except Exception:
                q_unit = None

        legacy_fallback = False
        legacy_parent_map: Dict[str, str] = {}

        with self._connect() as conn:
            note_rows = conn.execute(
                "SELECT rel_path, stem, title, snippet, embedding FROM note_embeddings"
            ).fetchall()

            has_child_chunks = (
                conn.execute("SELECT 1 FROM child_chunks LIMIT 1").fetchone() is not None
            )

            if not has_child_chunks:
                if not note_rows:
                    return []
                legacy_rows = conn.execute(
                    "SELECT rel_path, stem, title, snippet, content, embedding FROM note_embeddings"
                ).fetchall()
                legacy_fallback = True
                rows = [
                    (r[0], r[0], r[1], r[2], "Overview", r[0], r[3], r[5])
                    for r in legacy_rows
                ]
                legacy_parent_map = {r[0]: r[4] for r in legacy_rows}
            elif len(note_rows) > 300:
                # Stage 1: Coarse Candidate Note Selection across note_embeddings
                note_dense: Dict[int, float] = {}
                if q_unit is not None and np is not None:
                    v_idx: List[int] = []
                    v_blobs: List[np.ndarray] = []
                    for i, r in enumerate(note_rows):
                        b = r[4]
                        if b and len(b) == expected_bytes:
                            v_idx.append(i)
                            v_blobs.append(np.frombuffer(b, dtype=np.float32))
                    if v_blobs:
                        mat = np.vstack(v_blobs)
                        norms = np.linalg.norm(mat, axis=1)
                        norms[norms == 0] = 1.0
                        sims = (mat @ q_unit) / norms
                        for pos, doc_i in enumerate(v_idx):
                            note_dense[doc_i] = float(sims[pos])

                note_scores: List[Tuple[float, str]] = []
                for i, (rpath, stem, title, snippet, _) in enumerate(note_rows):
                    kw = 0.0
                    sl = (stem or "").lower()
                    tl = (title or "").lower()
                    snl = (snippet or "").lower()
                    for w in query_words:
                        if w in sl:
                            kw += 4.0
                        if w in tl:
                            kw += 3.5
                        if w in snl:
                            kw += 1.5
                    sim = note_dense.get(i, 0.0)
                    note_scores.append((sim * 12.0 + kw, rpath))

                note_scores.sort(key=lambda x: x[0], reverse=True)
                cand_rel_paths = [rp for _, rp in note_scores[:150]]
                cand_set = set(cand_rel_paths)

                # Augment with direct SQLite child_content substring matches for specific tokens
                specific_words = sorted(query_words, key=len, reverse=True)[:3]
                if specific_words:
                    like_clauses = " OR ".join("child_content LIKE ?" for _ in specific_words)
                    like_params = [f"%{w}%" for w in specific_words]
                    extra_rows = conn.execute(
                        f"SELECT DISTINCT rel_path FROM child_chunks WHERE {like_clauses} LIMIT 100",
                        like_params,
                    ).fetchall()
                    for (erp,) in extra_rows:
                        if erp not in cand_set:
                            cand_set.add(erp)
                            cand_rel_paths.append(erp)

                placeholders = ",".join("?" for _ in cand_rel_paths)
                rows = conn.execute(
                    f"""
                    SELECT c.chunk_id, c.rel_path, p.stem, p.title, p.section_header,
                           c.parent_id, c.child_content, c.embedding
                    FROM child_chunks c
                    LEFT JOIN parent_sections p ON c.parent_id = p.parent_id
                    WHERE c.rel_path IN ({placeholders})
                    """,
                    cand_rel_paths,
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT c.chunk_id, c.rel_path, p.stem, p.title, p.section_header,
                           c.parent_id, c.child_content, c.embedding
                    FROM child_chunks c
                    LEFT JOIN parent_sections p ON c.parent_id = p.parent_id
                    """
                ).fetchall()

        if not rows:
            return []

        doc_count = len(rows)

        # --- 1. Sparse Keyword Scoring on Candidate Child Chunks & Titles ---
        sparse_scores: List[Tuple[float, int]] = []
        for idx, (_, rel_path, stem, title, section_title, _, child_content, _) in enumerate(rows):
            score = 0.0
            t_lower = (title or "").lower()
            st_lower = (section_title or "").lower()
            stem_lower = (stem or "").lower()
            c_lower = (child_content or "").lower()

            for w in query_words:
                if w in stem_lower:
                    score += 4.0
                if w in t_lower:
                    score += 3.5
                if w in st_lower:
                    score += 3.0
                if w in c_lower:
                    score += min(5.0, c_lower.count(w) * 1.0)

            sparse_scores.append((score, idx))

        sparse_scores.sort(key=lambda x: x[0], reverse=True)
        sparse_ranks: Dict[int, int] = {idx: rank + 1 for rank, (_, idx) in enumerate(sparse_scores)}

        # --- 2. Vectorized Dense Semantic Scoring on Candidate Child Chunks ---
        dense_ranks: Dict[int, int] = {}
        cosine_sims: Dict[int, float] = {}

        if q_unit is not None and np is not None:
            valid_indices: List[int] = []
            valid_blobs: List[np.ndarray] = []

            for idx, row in enumerate(rows):
                emb_blob = row[7]
                if emb_blob and len(emb_blob) == expected_bytes:
                    valid_indices.append(idx)
                    valid_blobs.append(np.frombuffer(emb_blob, dtype=np.float32))

            if valid_blobs:
                mat = np.vstack(valid_blobs)
                norms = np.linalg.norm(mat, axis=1)
                norms[norms == 0] = 1.0
                sims = (mat @ q_unit) / norms

                dense_scores: List[Tuple[float, int]] = []
                for pos, doc_idx in enumerate(valid_indices):
                    s_val = float(sims[pos])
                    cosine_sims[doc_idx] = s_val
                    dense_scores.append((s_val, doc_idx))

                dense_scores.sort(key=lambda x: x[0], reverse=True)
                dense_ranks = {doc_idx: rank + 1 for rank, (_, doc_idx) in enumerate(dense_scores)}

        # --- 3. Reciprocal Rank Fusion & Hierarchical Parent Aggregation ---
        k_rrf = 60.0
        parent_results: Dict[str, Dict[str, Any]] = {}

        for idx, (chunk_id, rel_path, stem, title, section_title, parent_id, child_content, _) in enumerate(rows):
            r_sparse = sparse_ranks.get(idx, doc_count)
            r_dense = dense_ranks.get(idx, doc_count)
            cos_sim = cosine_sims.get(idx, 0.0)
            sparse_raw = sparse_scores[r_sparse - 1][0] if 0 <= r_sparse - 1 < len(sparse_scores) else 0.0

            rrf_score = (1.0 / (k_rrf + r_sparse)) + (1.0 / (k_rrf + r_dense))

            if sparse_raw <= 0.0 and cos_sim < 0.45 and (not dense_ranks or r_dense > 20):
                continue
            if r_sparse > 25 and cos_sim < 0.45 and (not dense_ranks or r_dense > 20):
                continue

            if parent_id not in parent_results or rrf_score > parent_results[parent_id]["rrf_score"]:
                parent_results[parent_id] = {
                    "rel_path": rel_path,
                    "stem": stem or Path(rel_path).stem,
                    "title": title or stem or Path(rel_path).stem,
                    "section_title": section_title or "General",
                    "parent_id": parent_id,
                    "child_snippet": child_content,
                    "rrf_score": round(rrf_score, 4),
                    "semantic_similarity": round(cos_sim * 100, 1),
                    "sparse_rank": r_sparse,
                    "dense_rank": r_dense,
                }

        all_sorted = sorted(parent_results.values(), key=lambda x: x["rrf_score"], reverse=True)
        sorted_results = []
        counts_per_note: Dict[str, int] = {}
        for r in all_sorted:
            rp = r["rel_path"]
            if counts_per_note.get(rp, 0) < 2:
                counts_per_note[rp] = counts_per_note.get(rp, 0) + 1
                sorted_results.append(r)
                if len(sorted_results) >= top_k:
                    break

        if not sorted_results:
            return []

        # --- 4. Lazy-Load Parent Section Content Only for Top-K Winners ---
        winner_pids = [r["parent_id"] for r in sorted_results]
        parent_text_map: Dict[str, str] = {}
        if legacy_fallback:
            parent_text_map = legacy_parent_map
        else:
            placeholders = ",".join("?" for _ in winner_pids)
            with self._connect() as conn:
                p_rows = conn.execute(
                    f"SELECT parent_id, parent_content FROM parent_sections WHERE parent_id IN ({placeholders})",
                    winner_pids,
                ).fetchall()
                parent_text_map = {pid: (pcontent or "") for pid, pcontent in p_rows}

        for item in sorted_results:
            pid = item["parent_id"]
            p_text = parent_text_map.get(pid, item["child_snippet"])
            item["parent_content"] = p_text if return_parent_context else item["child_snippet"]
            item["snippet"] = p_text[:500] if return_parent_context else item["child_snippet"][:300]

        return sorted_results


def main():
    parser = argparse.ArgumentParser(description="2ndBrain Hybrid Search Engine & Vector DB Maintenance")
    parser.add_argument("--vault", default=None, help="Path to Obsidian vault")
    parser.add_argument("--migrate", action="store_true", help="Migrate legacy hierarchical_chunks table to normalized schema and VACUUM")
    parser.add_argument("--index", action="store_true", help="Incrementally index vault notes")
    parser.add_argument("--force", action="store_true", help="Force re-index of all notes")
    parser.add_argument("--query", default=None, help="Run a hybrid search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    vault_dir = args.vault or os.environ.get("SECOND_BRAIN_VAULT") or str(repo_root / "vault")
    engine = HybridSearchEngine(vault_dir)

    if args.migrate:
        stats = engine.migrate_legacy_db()
        print(
            f"[HybridSearch] Migration complete: {stats['before_mb']} MB -> {stats['after_mb']} MB "
            f"({stats['parent_sections']:,} parent sections, {stats['child_chunks']:,} child chunks)"
        )
    if args.index:
        count = engine.index_vault(force=args.force)
        print(f"[HybridSearch] Indexed {count} notes.")
    if args.query:
        results = engine.search(args.query, top_k=args.top_k)
        for i, r in enumerate(results, 1):
            print(
                f"{i}. [{r['rel_path']}] ({r['section_title']}) "
                f"RRF={r['rrf_score']} Sim={r['semantic_similarity']}%"
            )


if __name__ == "__main__":
    main()
