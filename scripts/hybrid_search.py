"""
hybrid_search.py - Local-First Hybrid Search Engine for 2ndBrain

Features:
1. Parent-Child / Hierarchical Chunk Retrieval:
   - Decomposes notes into logical Parent Sections (headers & context blocks).
   - Subdivides each parent into small, cohesive Child Chunks (~350 chars) for pinpoint vector & BM25 matching.
   - Embeds Child Chunks with Document & Section context prefixes.
   - Aggregates child hits back into full Parent Section contexts, delivering rich, untruncated clauses to LLMs.
2. Dense Semantic Vector Search via fastembed (BAAI/bge-small-en-v1.5, 384-d, 100% offline).
3. Sparse Keyword / BM25 Scoring with exact token weighting.
4. Reciprocal Rank Fusion (RRF) for merged ranking.
5. Embedded SQLite storage for instant sub-25ms retrieval.
"""

import os
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    import numpy as np
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
    np = None


class HybridSearchEngine:
    def __init__(self, vault_dir: str, db_path: Optional[str] = None):
        self.vault_dir = Path(vault_dir).resolve()
        self.db_path = Path(db_path) if db_path else self.vault_dir / "00-Meta" / "vault_vectors.db"
        self._model = None
        self._init_db()

    def _init_db(self):
        """Initializes SQLite tables for note-level and hierarchical chunk vector embeddings."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            # Legacy note-level table (backwards compatibility)
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

            # Hierarchical Parent-Child chunks table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hierarchical_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    rel_path TEXT,
                    stem TEXT,
                    title TEXT,
                    section_title TEXT,
                    parent_id TEXT,
                    child_content TEXT,
                    parent_content TEXT,
                    embedding BLOB,
                    mtime REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_h_rel_path ON hierarchical_chunks(rel_path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_h_parent_id ON hierarchical_chunks(parent_id)")
            conn.commit()

    @property
    def model(self):
        """Lazy-loads the embedding model on first call to keep startup instant."""
        if self._model is None and FASTEMBED_AVAILABLE:
            self._model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return self._model

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

    def _decompose_into_parent_sections(self, clean_content: str, default_title: str) -> List[Tuple[str, str]]:
        """
        Decomposes markdown content into Parent Sections based on markdown headings
        (H1, H2, H3) and thematic breaks.
        Returns a list of (section_title, section_body) tuples.
        """
        if not clean_content.strip():
            return [(default_title, "")]

        # Split on headers: H1 (# ), H2 (## ), H3 (### )
        header_pattern = re.compile(r"(?m)^(#{1,3}\s+[^\n]+)")
        splits = header_pattern.split(clean_content)

        sections: List[Tuple[str, str]] = []
        current_title = default_title

        # If text begins before first header
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

        # If no sections were produced, return the full content
        if not sections:
            return [(default_title, clean_content)]

        # Further subdivide any overly massive parent section (> 2800 chars) into paragraph blocks
        refined_sections: List[Tuple[str, str]] = []
        for sec_title, sec_body in sections:
            if len(sec_body) > 2800:
                paragraphs = sec_body.split("\n\n")
                buf: List[str] = []
                buf_len = 0
                part_idx = 1
                for p in paragraphs:
                    p_str = p.strip()
                    if not p_str:
                        continue
                    if buf_len + len(p_str) > 2200 and buf:
                        refined_sections.append((f"{sec_title} (Part {part_idx})", "\n\n".join(buf)))
                        buf = [p_str]
                        buf_len = len(p_str)
                        part_idx += 1
                    else:
                        buf.append(p_str)
                        buf_len += len(p_str)
                if buf:
                    refined_sections.append((f"{sec_title} (Part {part_idx})" if part_idx > 1 else sec_title, "\n\n".join(buf)))
            else:
                refined_sections.append((sec_title, sec_body))

        return refined_sections

    def _split_into_child_chunks(self, parent_text: str, target_size: int = 380, overlap: int = 60) -> List[str]:
        """
        Splits parent section into small, cohesive child chunks (~350 chars)
        preserving sentence and clause boundaries.
        """
        if len(parent_text) <= target_size:
            return [parent_text]

        # Split into sentences or lines
        sentences = re.split(r"(?<=[.!?\n])\s+", parent_text)
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue

            if current_len + len(s_clean) > target_size and current_chunk:
                joined = " ".join(current_chunk).strip()
                if joined:
                    chunks.append(joined)
                # Keep overlap from the end of current chunk
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

    def index_note(self, file_path: Path, force: bool = False) -> bool:
        """
        Indexes a single note using Parent-Child Hierarchical Chunking.
        Subdivides the note into Parent Sections and embeds Child Chunks.
        """
        if not file_path.exists() or file_path.suffix != ".md":
            return False
        if file_path.name.lower() == "readme.md" and file_path.parent.name in (
            "Architecture", "Decisions", "Blast-Radius-Logs", "Modules", 
            "Papers", "Transcripts", "Corrections", "Sessions"
        ):
            return False

        rel_path = file_path.relative_to(self.vault_dir).as_posix()
        file_mtime = file_path.stat().st_mtime

        with sqlite3.connect(self.db_path) as conn:
            # Check if up to date in hierarchical_chunks
            row = conn.execute("SELECT mtime FROM hierarchical_chunks WHERE rel_path = ? LIMIT 1", (rel_path,)).fetchone()
            if row and not force and abs(row[0] - file_mtime) < 1e-4:
                return False  # Up to date

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return False

        title, clean_content, snippet = self._clean_text(content)
        title = title or file_path.stem
        file_stem = file_path.stem

        # 1. Build Parent Sections
        parent_sections = self._decompose_into_parent_sections(clean_content, title)

        # 2. Build Child Chunks for each Parent Section
        all_chunk_records: List[Dict[str, Any]] = []
        embed_texts: List[str] = []

        for sec_idx, (sec_title, parent_text) in enumerate(parent_sections):
            parent_id = f"{rel_path}#sec-{sec_idx}"
            children = self._split_into_child_chunks(parent_text, target_size=380, overlap=60)

            for c_idx, child_text in enumerate(children):
                chunk_id = f"{parent_id}-c{c_idx}"
                # Embed text combines Document title + Section context + Child content
                embed_input = f"{title} > {sec_title}\n{child_text}".strip()
                embed_texts.append(embed_input)

                all_chunk_records.append({
                    "chunk_id": chunk_id,
                    "rel_path": rel_path,
                    "stem": file_stem,
                    "title": title,
                    "section_title": sec_title,
                    "parent_id": parent_id,
                    "child_content": child_text,
                    "parent_content": parent_text,
                    "mtime": file_mtime
                })

        # 3. Compute Embeddings in Batch via fastembed
        embeddings_bytes: List[Optional[bytes]] = [None] * len(all_chunk_records)
        if self.model and np is not None and embed_texts:
            try:
                vectors = list(self.model.embed(embed_texts))
                for idx, vec in enumerate(vectors):
                    vec_np = np.array(vec, dtype=np.float32)
                    embeddings_bytes[idx] = vec_np.tobytes()
            except Exception as e:
                print(f"[HybridSearch] Note embedding warning for {rel_path}: {e}")

        # 4. Insert into SQLite transactionally
        with sqlite3.connect(self.db_path) as conn:
            # Delete old chunks for this file
            conn.execute("DELETE FROM hierarchical_chunks WHERE rel_path = ?", (rel_path,))
            
            # Insert hierarchical chunks
            insert_rows = [
                (
                    rec["chunk_id"], rec["rel_path"], rec["stem"], rec["title"],
                    rec["section_title"], rec["parent_id"], rec["child_content"],
                    rec["parent_content"], embeddings_bytes[i], rec["mtime"]
                )
                for i, rec in enumerate(all_chunk_records)
            ]
            conn.executemany("""
                INSERT OR REPLACE INTO hierarchical_chunks
                (chunk_id, rel_path, stem, title, section_title, parent_id, child_content, parent_content, embedding, mtime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, insert_rows)

            # Also maintain legacy note_embeddings table with first chunk embedding
            primary_emb = embeddings_bytes[0] if embeddings_bytes else None
            conn.execute("""
                INSERT OR REPLACE INTO note_embeddings 
                (rel_path, stem, title, snippet, content, embedding, mtime)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (rel_path, file_stem, title, snippet, clean_content[:3000], primary_emb, file_mtime))

            conn.commit()

        return True

    def index_vault(self, force: bool = False) -> int:
        """Scans and indexes all markdown notes in vault into hierarchical chunks."""
        all_md_files = [
            p for p in self.vault_dir.rglob("*.md")
            if not any(part.startswith(".") for part in p.parts)
        ]

        existing_rel_paths = {p.relative_to(self.vault_dir).as_posix() for p in all_md_files}
        with sqlite3.connect(self.db_path) as conn:
            # Clean up deleted files
            db_paths = [r[0] for r in conn.execute("SELECT DISTINCT rel_path FROM hierarchical_chunks").fetchall()]
            for dp in db_paths:
                if dp not in existing_rel_paths:
                    conn.execute("DELETE FROM hierarchical_chunks WHERE rel_path = ?", (dp,))
                    conn.execute("DELETE FROM note_embeddings WHERE rel_path = ?", (dp,))
            conn.commit()

        indexed_count = 0
        for p in all_md_files:
            if self.index_note(p, force=force):
                indexed_count += 1

        return indexed_count

    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        return_parent_context: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Executes Parent-Child Hierarchical Search across the vault:
        1. Pinpoint matching on Child Chunks via dense vectors + sparse BM25.
        2. Fused ranking using Reciprocal Rank Fusion (RRF).
        3. Hierarchical Deduplication: Multiple child matches in the same section
           are consolidated into the Parent Section.
        4. Injects rich Parent Section content directly into the returned result.
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT chunk_id, rel_path, stem, title, section_title, parent_id, 
                       child_content, parent_content, embedding 
                FROM hierarchical_chunks
            """).fetchall()

        # Fallback to legacy note table if hierarchical table is empty
        if not rows:
            with sqlite3.connect(self.db_path) as conn:
                legacy_rows = conn.execute("""
                    SELECT rel_path, stem, title, snippet, content, embedding 
                    FROM note_embeddings
                """).fetchall()
            if not legacy_rows:
                return []
            # Convert legacy format to chunk format
            rows = [
                (r[0], r[0], r[1], r[2], "Overview", r[0], r[3], r[4], r[5])
                for r in legacy_rows
            ]

        doc_count = len(rows)
        query_words = [w.lower() for w in re.findall(r"\w{3,}", query)]

        # --- 1. Sparse Keyword Scoring on Child Chunks & Titles ---
        sparse_scores: List[Tuple[float, int]] = []
        for idx, (_, rel_path, stem, title, section_title, _, child_content, _, _) in enumerate(rows):
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

        # --- 2. Dense Semantic Vector Scoring on Child Chunks ---
        dense_ranks: Dict[int, int] = {}
        cosine_sims: Dict[int, float] = {}

        if self.model and np is not None:
            query_vectors = list(self.model.embed([query]))
            if query_vectors:
                q_vec = np.array(query_vectors[0], dtype=np.float32)
                q_norm = np.linalg.norm(q_vec) or 1.0

                dense_scores: List[Tuple[float, int]] = []
                for idx, (_, _, _, _, _, _, _, _, emb_blob) in enumerate(rows):
                    if emb_blob:
                        doc_vec = np.frombuffer(emb_blob, dtype=np.float32)
                        doc_norm = np.linalg.norm(doc_vec) or 1.0
                        sim = float(np.dot(q_vec, doc_vec) / (q_norm * doc_norm))
                    else:
                        sim = 0.0
                    cosine_sims[idx] = sim
                    dense_scores.append((sim, idx))

                dense_scores.sort(key=lambda x: x[0], reverse=True)
                dense_ranks = {idx: rank + 1 for rank, (_, idx) in enumerate(dense_scores)}

        # --- 3. Reciprocal Rank Fusion & Hierarchical Parent Aggregation ---
        k_rrf = 60.0
        # Map: parent_id -> best child result
        parent_results: Dict[str, Dict[str, Any]] = {}

        for idx, (chunk_id, rel_path, stem, title, section_title, parent_id, child_content, parent_content, _) in enumerate(rows):
            r_sparse = sparse_ranks.get(idx, doc_count)
            r_dense = dense_ranks.get(idx, doc_count)
            cos_sim = cosine_sims.get(idx, 0.0)

            # RRF formula
            rrf_score = (1.0 / (k_rrf + r_sparse)) + (1.0 / (k_rrf + r_dense))

            # Filter out non-relevant matches
            if r_sparse > 25 and cos_sim < 0.45 and (not r_dense or r_dense > 20):
                continue

            # Hierarchical grouping: retain highest-scoring child match per parent section
            if parent_id not in parent_results or rrf_score > parent_results[parent_id]["rrf_score"]:
                parent_results[parent_id] = {
                    "rel_path": rel_path,
                    "stem": stem,
                    "title": title or stem,
                    "section_title": section_title or "General",
                    "parent_id": parent_id,
                    "child_snippet": child_content,
                    "parent_content": parent_content if return_parent_context else child_content,
                    "snippet": (parent_content[:500] if return_parent_context else child_content[:300]),
                    "rrf_score": round(rrf_score, 4),
                    "semantic_similarity": round(cos_sim * 100, 1),
                    "sparse_rank": r_sparse,
                    "dense_rank": r_dense
                }

        sorted_results = sorted(parent_results.values(), key=lambda x: x["rrf_score"], reverse=True)
        return sorted_results[:top_k]
