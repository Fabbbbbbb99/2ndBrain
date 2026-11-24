"""Single-writer SQLite WAL + FTS5 + domain/version-partitioned sqlite-vec storage."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sqlite_vec

from second_brain.storage.models import (
    AuthorityTier,
    EdgeProvenance,
    EdgeType,
    EvidenceEnvelope,
    IndexCoverage,
    LinkedClauseRef,
    MemoryRecord,
    StoreStatus,
    VisualKind,
    VisualStatus,
    normalize_project_tag,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def lf_sha256(text: str) -> str:
    normalized = text.replace("\r\n", "\n").encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def serialize_unit_int8(vec: np.ndarray | list[float]) -> bytes:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    unit = arr / norm if norm > 0.0 else arr
    q = np.clip(np.round(unit * 127.0), -127, 127).astype(np.int8)
    return sqlite_vec.serialize_int8(q.tolist())


def atomic_replace_windows(src: Path, dst: Path, retries: int = 8, delay_s: float = 0.05) -> None:
    """Replace dst with src using bounded retry backoff for transient Windows Defender/Obsidian locks."""
    for attempt in range(retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay_s * (attempt + 1))


def atomic_write_lf(path: Path, text: str) -> None:
    """Write LF-normalized UTF-8 text via temp file + atomic retry replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lf_text = text.replace("\r\n", "\n")
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_bytes(lf_text.encode("utf-8"))
    atomic_replace_windows(tmp, path)


class SqliteStore:
    """Single-writer SQLite storage backing Store 1 (wiki/parsed) and Store 2 (memory)."""

    VECTOR_DIM = 384

    def __init__(self, db_path: Path, readonly: bool = False) -> None:
        self.db_path = Path(db_path)
        self.readonly = readonly
        if not readonly:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = self._open_connection()
        if not readonly:
            self.init_schema()

    def _open_connection(self) -> sqlite3.Connection:
        if self.readonly:
            uri = f"file:{self.db_path.as_posix()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        else:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        if not self.readonly:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=10000;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                uri TEXT PRIMARY KEY,
                store TEXT NOT NULL CHECK(store IN ('wiki', 'parsed', 'memory', 'code')),
                domain TEXT NOT NULL,
                source_id TEXT NOT NULL,
                source_version TEXT NOT NULL,
                clause_path TEXT NOT NULL,
                parent_uri TEXT,
                authority_tier TEXT NOT NULL CHECK(
                    authority_tier IN ('primary_normative', 'primary_reference', 'compiled_synthesis', 'working_memory')
                ),
                file_relpath TEXT,
                start_line INTEGER,
                end_line INTEGER,
                content_sha256 TEXT NOT NULL,
                text TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                embedding_status TEXT NOT NULL DEFAULT 'pending' CHECK(embedding_status IN ('pending', 'ready', 'failed')),
                embed_key TEXT,
                index_generation INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_domain_ver ON chunks(domain, source_version);
            CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id, source_version);
            CREATE INDEX IF NOT EXISTS idx_chunks_embed_status ON chunks(embedding_status);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                uri UNINDEXED,
                clause_path,
                text,
                content='chunks',
                content_rowid='rowid'
            );

            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, uri, clause_path, text)
                VALUES (new.rowid, new.uri, new.clause_path, new.text);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, uri, clause_path, text)
                VALUES ('delete', old.rowid, old.uri, old.clause_path, old.text);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, uri, clause_path, text)
                VALUES ('delete', old.rowid, old.uri, old.clause_path, old.text);
                INSERT INTO chunks_fts(rowid, uri, clause_path, text)
                VALUES (new.rowid, new.uri, new.clause_path, new.text);
            END;

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                domain text partition key,
                source_version text partition key,
                embedding int8[{self.VECTOR_DIM}],
                +uri text
            );

            CREATE TABLE IF NOT EXISTS edges (
                src_uri TEXT NOT NULL,
                dst_uri TEXT NOT NULL,
                rel_type TEXT NOT NULL CHECK(
                    rel_type IN ('CITES', 'PART_OF', 'SUPERSEDES', 'CONFLICTS_WITH', 'APPLIES_TO', 'INFERRED_RELATED')
                ),
                provenance TEXT NOT NULL CHECK(provenance IN ('EXTRACTED', 'INFERRED')),
                confidence REAL NOT NULL DEFAULT 1.0,
                PRIMARY KEY (src_uri, dst_uri, rel_type)
            );
            CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_uri);

            CREATE TABLE IF NOT EXISTS memory_records (
                id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                ledger_type TEXT NOT NULL DEFAULT 'facts',
                project_tag TEXT,
                effective_from TEXT NOT NULL,
                effective_to TEXT,
                version_scope TEXT,
                status TEXT NOT NULL CHECK(status IN ('staged', 'active', 'superseded', 'expired', 'rejected')),
                supersedes_id TEXT,
                content TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                reason_code TEXT,
                created_at TEXT NOT NULL,
                embedding BLOB,
                embed_key TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_memory_domain_status ON memory_records(domain, status);

            CREATE TABLE IF NOT EXISTS visual_nodes (
                visual_id TEXT PRIMARY KEY,
                source_uri TEXT NOT NULL,
                pdf_relpath TEXT NOT NULL,
                page_num INTEGER NOT NULL,
                bbox_json TEXT NOT NULL,
                kind TEXT NOT NULL CHECK(kind IN ('table', 'equation', 'diagram')),
                status TEXT NOT NULL CHECK(status IN ('pending_vision', 'cached_unverified', 'certified')),
                page_sha256 TEXT NOT NULL,
                crop_path TEXT,
                repaired_md TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_visual_source ON visual_nodes(source_uri, status);

            CREATE TABLE IF NOT EXISTS exemplars (
                id TEXT PRIMARY KEY,
                query_text TEXT NOT NULL,
                expected_domain TEXT NOT NULL,
                expected_action TEXT NOT NULL,
                embedding BLOB NOT NULL,
                model_sha TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_exemplars_domain ON exemplars(expected_domain);

            CREATE TABLE IF NOT EXISTS query_counts (
                topic_key TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 1,
                last_queried_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS compiled_deps (
                compiled_uri TEXT NOT NULL,
                upstream_uri TEXT NOT NULL,
                upstream_sha256 TEXT NOT NULL,
                PRIMARY KEY (compiled_uri, upstream_uri)
            );
            """
        )
        existing_mem_cols = {
            r["name"] for r in self.conn.execute("PRAGMA table_info(memory_records)").fetchall()
        }
        if "project_tag" not in existing_mem_cols:
            self.conn.execute("ALTER TABLE memory_records ADD COLUMN project_tag TEXT")
        if "embedding" not in existing_mem_cols:
            self.conn.execute("ALTER TABLE memory_records ADD COLUMN embedding BLOB")
        if "embed_key" not in existing_mem_cols:
            self.conn.execute("ALTER TABLE memory_records ADD COLUMN embed_key TEXT")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_project_status ON memory_records(project_tag, status)"
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_src_rel ON edges(src_uri, rel_type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_dst_rel ON edges(dst_uri, rel_type)")
        self.conn.commit()

    def set_meta(self, pairs: dict[str, str]) -> None:
        with self.conn:
            for k, v in pairs.items():
                self.conn.execute(
                    "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (k, str(v)),
                )

    def get_meta(self) -> dict[str, str]:
        return {r["key"]: r["value"] for r in self.conn.execute("SELECT key, value FROM meta").fetchall()}

    def next_index_generation(self) -> int:
        meta = self.get_meta()
        gen = int(meta.get("index_generation", "0")) + 1
        self.set_meta({"index_generation": str(gen)})
        return gen

    def upsert_chunk(
        self,
        *,
        uri: str,
        store: str,
        domain: str,
        source_id: str,
        source_version: str,
        clause_path: str,
        parent_uri: str | None,
        authority_tier: AuthorityTier,
        file_relpath: str | None,
        start_line: int | None,
        end_line: int | None,
        text: str,
        token_count: int,
        embed_key: str | None = None,
        embedding_bytes: bytes | None = None,
        index_generation: int = 1,
        commit: bool = True,
    ) -> int:
        sha = lf_sha256(text)
        now = utc_now_iso()
        existing = self.conn.execute(
            "SELECT rowid, content_sha256, embed_key, embedding_status FROM chunks WHERE uri = ?",
            (uri,),
        ).fetchone()

        if embedding_bytes is not None:
            emb_status = "ready"
        elif existing and existing["embed_key"] == embed_key and existing["embedding_status"] == "ready":
            emb_status = "ready"
        else:
            emb_status = "pending"

        self.conn.execute(
            """
            INSERT INTO chunks(
                uri, store, domain, source_id, source_version, clause_path, parent_uri,
                authority_tier, file_relpath, start_line, end_line, content_sha256, text,
                token_count, embedding_status, embed_key, index_generation, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(uri) DO UPDATE SET
                store=excluded.store,
                domain=excluded.domain,
                source_id=excluded.source_id,
                source_version=excluded.source_version,
                clause_path=excluded.clause_path,
                parent_uri=excluded.parent_uri,
                authority_tier=excluded.authority_tier,
                file_relpath=excluded.file_relpath,
                start_line=excluded.start_line,
                end_line=excluded.end_line,
                content_sha256=excluded.content_sha256,
                text=excluded.text,
                token_count=excluded.token_count,
                embedding_status=CASE
                    WHEN excluded.embedding_status = 'ready' THEN 'ready'
                    WHEN chunks.embed_key = excluded.embed_key AND chunks.embedding_status = 'ready' THEN 'ready'
                    ELSE 'pending'
                END,
                embed_key=excluded.embed_key,
                index_generation=excluded.index_generation,
                updated_at=excluded.updated_at
            """,
            (
                uri,
                store,
                domain,
                source_id,
                source_version,
                clause_path,
                parent_uri,
                authority_tier,
                file_relpath,
                start_line,
                end_line,
                sha,
                text.replace("\r\n", "\n"),
                token_count,
                emb_status,
                embed_key,
                index_generation,
                now,
            ),
        )
        rowid = int(self.conn.execute("SELECT rowid FROM chunks WHERE uri = ?", (uri,)).fetchone()[0])

        if embedding_bytes is not None:
            self._upsert_vec_row(rowid, domain, source_version, embedding_bytes, uri)
        elif existing and emb_status == "pending":
            self.conn.execute("DELETE FROM chunks_vec WHERE rowid = ?", (rowid,))

        if commit:
            self.conn.commit()
        return rowid

    def _upsert_vec_row(
        self,
        rowid: int,
        domain: str,
        source_version: str,
        embedding_bytes: bytes,
        uri: str,
    ) -> None:
        self.conn.execute("DELETE FROM chunks_vec WHERE rowid = ?", (rowid,))
        self.conn.execute(
            "INSERT INTO chunks_vec(rowid, domain, source_version, embedding, uri) VALUES (?, ?, ?, vec_int8(?), ?)",
            (rowid, domain, source_version, embedding_bytes, uri),
        )

    def set_chunk_embedding(
        self,
        uri: str,
        embed_key: str,
        embedding_bytes: bytes | None,
        status: str = "ready",
        commit: bool = True,
    ) -> None:
        row = self.conn.execute("SELECT rowid, domain, source_version FROM chunks WHERE uri = ?", (uri,)).fetchone()
        if not row:
            return
        self.conn.execute(
            "UPDATE chunks SET embedding_status = ?, embed_key = ?, updated_at = ? WHERE uri = ?",
            (status, embed_key, utc_now_iso(), uri),
        )
        if status == "ready" and embedding_bytes is not None:
            self._upsert_vec_row(int(row["rowid"]), row["domain"], row["source_version"], embedding_bytes, uri)
        if commit:
            self.conn.commit()

    def upsert_edge(
        self,
        src_uri: str,
        dst_uri: str,
        rel_type: EdgeType,
        provenance: EdgeProvenance,
        confidence: float = 1.0,
        commit: bool = True,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO edges(src_uri, dst_uri, rel_type, provenance, confidence)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(src_uri, dst_uri, rel_type) DO UPDATE SET
                provenance=excluded.provenance,
                confidence=excluded.confidence
            """,
            (src_uri, dst_uri, rel_type, provenance, confidence),
        )
        if commit:
            self.conn.commit()

    def get_chunk_by_uri(self, uri: str) -> EvidenceEnvelope | None:
        row = self.conn.execute("SELECT * FROM chunks WHERE uri = ?", (uri,)).fetchone()
        if not row:
            return None
        return self._row_to_envelope(row)

    def _row_to_envelope(self, row: sqlite3.Row, score: float = 0.0, stale: bool = False) -> EvidenceEnvelope:
        return EvidenceEnvelope(
            uri=row["uri"],
            store=row["store"],
            authority_tier=row["authority_tier"],
            domain=row["domain"],
            source_id=row["source_id"],
            source_version=row["source_version"],
            clause_path=row["clause_path"],
            text=row["text"],
            content_sha256=row["content_sha256"],
            token_count=row["token_count"],
            status="stale" if stale else "active",
            score=score,
            stale=stale,
            file_relpath=row["file_relpath"],
            start_line=row["start_line"],
            end_line=row["end_line"],
        )

    def search_fts(
        self,
        fts_query: str,
        domain_scope: list[str],
        version_filter: str | None = None,
        limit: int = 20,
        source_id: str | None = None,
    ) -> list[tuple[EvidenceEnvelope, float]]:
        if not domain_scope or not fts_query.strip():
            return []
        placeholders = ",".join("?" for _ in domain_scope)
        params: list[Any] = [fts_query, *domain_scope]
        ver_clause = ""
        if version_filter:
            ver_clause = "AND (c.source_version = ? OR c.source_version = 'global')"
            params.append(version_filter)
        src_clause = ""
        if source_id and source_id.strip():
            src_clause = "AND LOWER(c.source_id) = LOWER(?)"
            params.append(source_id.strip())
        params.append(limit)
        sql = f"""
            SELECT c.*, bm25(chunks_fts) AS rank_score
            FROM chunks_fts
            JOIN chunks c ON c.rowid = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
              AND c.domain IN ({placeholders})
              {ver_clause}
              {src_clause}
            ORDER BY rank_score ASC
            LIMIT ?
        """
        rows = self.conn.execute(sql, params).fetchall()
        out: list[tuple[EvidenceEnvelope, float]] = []
        for idx, r in enumerate(rows):
            env = self._row_to_envelope(r, score=1.0 / (idx + 1))
            out.append((env, float(r["rank_score"])))
        return out

    def search_vec(
        self,
        query_embedding_bytes: bytes,
        domain_scope: list[str],
        version_filter: str | None = None,
        limit: int = 20,
        source_id: str | None = None,
    ) -> list[tuple[EvidenceEnvelope, float]]:
        if not domain_scope:
            return []
        placeholders = ",".join("?" for _ in domain_scope)
        src_norm = source_id.strip().lower() if source_id and source_id.strip() else None
        vec_k = min(max(limit * 6, 60), 250) if src_norm else limit
        params: list[Any] = [query_embedding_bytes, vec_k, *domain_scope]
        ver_clause = ""
        if version_filter:
            ver_clause = "AND source_version IN (?, 'global')"
            params.append(version_filter)
        sql = f"""
            SELECT uri, distance
            FROM chunks_vec
            WHERE embedding MATCH vec_int8(?)
              AND k = ?
              AND domain IN ({placeholders})
              {ver_clause}
        """
        vec_rows = self.conn.execute(sql, params).fetchall()
        out: list[tuple[EvidenceEnvelope, float]] = []
        for vr in vec_rows:
            chunk = self.conn.execute("SELECT * FROM chunks WHERE uri = ?", (vr["uri"],)).fetchone()
            if not chunk:
                continue
            if src_norm and str(chunk["source_id"]).lower() != src_norm:
                continue
            dist = float(vr["distance"])
            sim = 1.0 / (1.0 + max(dist, 0.0))
            out.append((self._row_to_envelope(chunk, score=sim), dist))
            if len(out) >= limit:
                break
        return out

    def get_neighbors(
        self,
        uris: list[str],
        domain_scope: list[str],
        limit: int = 12,
        *,
        exclude_inferred: bool = False,
    ) -> list[LinkedClauseRef]:
        if not uris or not domain_scope:
            return []
        u_ph = ",".join("?" for _ in uris)
        d_ph = ",".join("?" for _ in domain_scope)
        inf_clause = "AND e.provenance != 'INFERRED'" if exclude_inferred else ""
        sql = f"""
            SELECT e.dst_uri, e.rel_type, e.provenance, e.confidence, c.clause_path, c.source_version, c.token_count
            FROM edges e
            LEFT JOIN chunks c ON c.uri = e.dst_uri
            WHERE e.src_uri IN ({u_ph})
              AND (c.domain IS NULL OR c.domain IN ({d_ph}))
              {inf_clause}
            ORDER BY CASE e.provenance WHEN 'EXTRACTED' THEN 0 ELSE 1 END, e.confidence DESC
            LIMIT ?
        """
        rows = self.conn.execute(sql, [*uris, *domain_scope, limit]).fetchall()
        return [
            LinkedClauseRef(
                uri=r["dst_uri"],
                rel_type=r["rel_type"],
                provenance=r["provenance"],
                clause_path=r["clause_path"],
                source_version=r["source_version"],
                token_count=r["token_count"],
                confidence=float(r["confidence"]) if r["confidence"] is not None else None,
                direction="out",
            )
            for r in rows
        ]

    def get_clause_children(self, uri: str, limit: int = 12) -> tuple[list[LinkedClauseRef], int]:
        rows = self.conn.execute(
            """
            SELECT e.src_uri, e.rel_type, e.provenance, e.confidence,
                   c.clause_path, c.source_version, c.token_count, c.start_line
            FROM edges e
            LEFT JOIN chunks c ON c.uri = e.src_uri
            WHERE e.dst_uri = ? AND e.rel_type = 'PART_OF'
            """,
            (uri,),
        ).fetchall()

        def _sort_key(r: sqlite3.Row) -> tuple[int, int, int, str]:
            src = str(r["src_uri"])
            if src.startswith(uri + "_p"):
                m = re.search(r"_p(\d+)$", src)
                if m:
                    return (0, int(m.group(1)), 0, src)
            sline = int(r["start_line"]) if r["start_line"] is not None else 999999
            return (1, 0, sline, src)

        ordered = sorted(rows, key=_sort_key)
        refs = [
            LinkedClauseRef(
                uri=r["src_uri"],
                rel_type=r["rel_type"],
                provenance=r["provenance"],
                clause_path=r["clause_path"],
                source_version=r["source_version"],
                token_count=r["token_count"],
                confidence=float(r["confidence"]) if r["confidence"] is not None else None,
                direction="in",
            )
            for r in ordered[: max(0, limit)]
        ]
        return refs, len(rows)

    def get_clause_cited_by(self, uri: str, limit: int = 12) -> tuple[list[LinkedClauseRef], int]:
        rows = self.conn.execute(
            """
            SELECT e.src_uri, e.rel_type, e.provenance, e.confidence,
                   c.clause_path, c.source_version, c.token_count
            FROM edges e
            LEFT JOIN chunks c ON c.uri = e.src_uri
            WHERE e.dst_uri = ? AND e.rel_type = 'CITES' AND e.provenance = 'EXTRACTED'
            ORDER BY COALESCE(c.clause_path, e.src_uri) ASC, e.src_uri ASC
            """,
            (uri,),
        ).fetchall()
        refs = [
            LinkedClauseRef(
                uri=r["src_uri"],
                rel_type=r["rel_type"],
                provenance=r["provenance"],
                clause_path=r["clause_path"],
                source_version=r["source_version"],
                token_count=r["token_count"],
                confidence=float(r["confidence"]) if r["confidence"] is not None else None,
                direction="in",
            )
            for r in rows[: max(0, limit)]
        ]
        return refs, len(rows)

    def get_inferred_links(
        self,
        uri: str,
        limit: int = 12,
        certified_hashes: dict[tuple[str, str], tuple[str, str]] | None = None,
    ) -> tuple[list[LinkedClauseRef], int]:
        self_row = self.conn.execute("SELECT content_sha256 FROM chunks WHERE uri = ?", (uri,)).fetchone()
        self_sha = str(self_row["content_sha256"]) if self_row else None
        rows = self.conn.execute(
            """
            SELECT e.src_uri, e.dst_uri, e.rel_type, e.provenance, e.confidence,
                   c.clause_path, c.source_version, c.token_count, c.content_sha256 AS other_sha
            FROM edges e
            LEFT JOIN chunks c ON c.uri = CASE WHEN e.src_uri = ? THEN e.dst_uri ELSE e.src_uri END
            WHERE e.rel_type = 'INFERRED_RELATED' AND (e.src_uri = ? OR e.dst_uri = ?)
            """,
            (uri, uri, uri),
        ).fetchall()

        seen: dict[str, tuple[float, LinkedClauseRef]] = {}
        for r in rows:
            is_out = r["src_uri"] == uri
            other_uri = str(r["dst_uri"] if is_out else r["src_uri"])
            conf = float(r["confidence"]) if r["confidence"] is not None else 0.0
            is_stale = False
            if certified_hashes is not None:
                a, b = (uri, other_uri) if uri < other_uri else (other_uri, uri)
                pair_hashes = certified_hashes.get((a, b))
                if pair_hashes is not None:
                    exp_a, exp_b = pair_hashes
                    cur_a = self_sha if a == uri else r["other_sha"]
                    cur_b = self_sha if b == uri else r["other_sha"]
                    is_stale = (cur_a != exp_a) or (cur_b != exp_b)
            ref = LinkedClauseRef(
                uri=other_uri,
                rel_type=r["rel_type"],
                provenance=r["provenance"],
                clause_path=r["clause_path"],
                source_version=r["source_version"],
                token_count=r["token_count"],
                confidence=conf,
                direction="out" if is_out else "in",
                stale=is_stale,
            )
            prev = seen.get(other_uri)
            if prev is None or conf > prev[0]:
                seen[other_uri] = (conf, ref)

        ordered = [
            ref
            for _, ref in sorted(
                seen.values(),
                key=lambda item: (-item[0], item[1].uri),
            )
        ]
        return ordered[: max(0, limit)], len(ordered)

    def apply_inferred_edges(
        self,
        edge_rows: list[dict[str, Any]],
        *,
        replace_inferred: bool = False,
    ) -> dict[str, int]:
        deleted = 0
        inserted = 0
        with self.conn:
            if replace_inferred:
                cur = self.conn.execute(
                    "DELETE FROM edges WHERE rel_type = 'INFERRED_RELATED' AND provenance = 'INFERRED' AND src_uri LIKE 'vault://%'"
                )
                deleted = int(cur.rowcount or 0)
            for row in edge_rows:
                cur = self.conn.execute(
                    """
                    INSERT INTO edges(src_uri, dst_uri, rel_type, provenance, confidence)
                    VALUES (?, ?, 'INFERRED_RELATED', 'INFERRED', ?)
                    ON CONFLICT(src_uri, dst_uri, rel_type) DO UPDATE SET
                        confidence=excluded.confidence
                    WHERE edges.provenance != 'EXTRACTED'
                    """,
                    (str(row["src_uri"]), str(row["dst_uri"]), float(row.get("confidence", 0.8))),
                )
                inserted += int(cur.rowcount or 0)
        return {"deleted": deleted, "inserted": inserted}

    def get_index_coverage(self, domain_scope: list[str] | None = None) -> IndexCoverage:
        where = ""
        params: list[Any] = []
        if domain_scope:
            ph = ",".join("?" for _ in domain_scope)
            where = f"WHERE domain IN ({ph})"
            params.extend(domain_scope)
        row = self.conn.execute(
            f"""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN embedding_status = 'ready' THEN 1 ELSE 0 END) AS ready_cnt,
                SUM(CASE WHEN embedding_status = 'pending' THEN 1 ELSE 0 END) AS pending_cnt,
                SUM(CASE WHEN embedding_status = 'failed' THEN 1 ELSE 0 END) AS failed_cnt
            FROM chunks {where}
            """,
            params,
        ).fetchone()
        total = int(row["total"] or 0)
        ready = int(row["ready_cnt"] or 0)
        pending = int(row["pending_cnt"] or 0)
        failed = int(row["failed_cnt"] or 0)
        if total == 0:
            return IndexCoverage(index_state="empty", coverage_pct=100.0)
        pct = round((ready / total) * 100.0, 2)
        if ready == 0:
            state = "fast_bm25_only"
        elif pending > 0 or failed > 0:
            state = "partial_embeddings"
        else:
            state = "complete"
        return IndexCoverage(
            fts_chunks=total,
            embedded_chunks=ready,
            pending_chunks=pending,
            failed_chunks=failed,
            coverage_pct=pct,
            index_state=state,
        )

    def get_store_statuses(self) -> list[StoreStatus]:
        c_wiki = int(self.conn.execute("SELECT COUNT(*) FROM chunks WHERE store='wiki'").fetchone()[0])
        c_parsed = int(self.conn.execute("SELECT COUNT(*) FROM chunks WHERE store='parsed'").fetchone()[0])
        c_mem = int(self.conn.execute("SELECT COUNT(*) FROM memory_records WHERE status='active'").fetchone()[0])
        return [
            StoreStatus(store="wiki", state="active", detail="Domain wiki synthesis store", chunk_count=c_wiki),
            StoreStatus(store="parsed", state="active", detail="Verbatim parsed standards and docs", chunk_count=c_parsed),
            StoreStatus(store="memory", state="active", detail="Temporal memory ledger", chunk_count=c_mem),
        ]

    def _row_to_memory_record(self, r: sqlite3.Row) -> MemoryRecord:
        keys = r.keys()
        return MemoryRecord(
            id=r["id"],
            domain=r["domain"],
            ledger_type=r["ledger_type"],
            project_tag=r["project_tag"] if "project_tag" in keys else None,
            effective_from=r["effective_from"],
            effective_to=r["effective_to"],
            version_scope=r["version_scope"],
            status=r["status"],
            supersedes_id=r["supersedes_id"],
            content=r["content"],
            source_ref=r["source_ref"],
            reason_code=r["reason_code"],
            created_at=r["created_at"],
        )

    def upsert_memory_record(
        self,
        rec: MemoryRecord,
        *,
        embedding_bytes: bytes | None = None,
        embed_key: str | None = None,
        commit: bool = True,
    ) -> None:
        norm_tag = normalize_project_tag(rec.project_tag)
        self.conn.execute(
            """
            INSERT INTO memory_records(
                id, domain, ledger_type, project_tag, effective_from, effective_to, version_scope,
                status, supersedes_id, content, source_ref, reason_code, created_at, embedding, embed_key
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                domain=excluded.domain,
                ledger_type=excluded.ledger_type,
                project_tag=excluded.project_tag,
                effective_to=excluded.effective_to,
                version_scope=excluded.version_scope,
                status=excluded.status,
                supersedes_id=excluded.supersedes_id,
                content=excluded.content,
                source_ref=excluded.source_ref,
                reason_code=excluded.reason_code,
                embedding=CASE
                    WHEN excluded.embedding IS NOT NULL THEN excluded.embedding
                    WHEN excluded.content != memory_records.content THEN NULL
                    ELSE memory_records.embedding
                END,
                embed_key=CASE
                    WHEN excluded.embedding IS NOT NULL THEN excluded.embed_key
                    WHEN excluded.content != memory_records.content THEN NULL
                    ELSE memory_records.embed_key
                END
            """,
            (
                rec.id,
                rec.domain,
                rec.ledger_type,
                norm_tag,
                rec.effective_from,
                rec.effective_to,
                rec.version_scope,
                rec.status,
                rec.supersedes_id,
                rec.content,
                rec.source_ref,
                rec.reason_code,
                rec.created_at,
                embedding_bytes,
                embed_key,
            ),
        )
        if rec.supersedes_id and rec.status == "active":
            self.conn.execute(
                "UPDATE memory_records SET status = 'superseded' WHERE id = ? AND status = 'active'",
                (rec.supersedes_id,),
            )
            self.upsert_edge(
                f"memory://{rec.id}",
                f"memory://{rec.supersedes_id}",
                "SUPERSEDES",
                "EXTRACTED",
                confidence=1.0,
                commit=False,
            )
        if commit:
            self.conn.commit()

    def update_memory_embedding(
        self,
        memory_id: str,
        embedding_bytes: bytes,
        embed_key: str,
        commit: bool = True,
    ) -> None:
        self.conn.execute(
            "UPDATE memory_records SET embedding = ?, embed_key = ? WHERE id = ?",
            (embedding_bytes, embed_key, memory_id),
        )
        if commit:
            self.conn.commit()

    def get_memory_candidates(
        self,
        domain_scope: list[str] | None = None,
        *,
        project_tag: str | None = None,
        include_untagged_for_project: bool = True,
        strict_untagged_only: bool = False,
        ledger_type: str | None = None,
        version_scope: str | None = None,
        status: str = "active",
        as_of_iso: str | None = None,
        limit: int = 2000,
    ) -> list[tuple[MemoryRecord, bytes | None, str | None]]:
        clauses: list[str] = []
        params: list[Any] = []

        if status == "active":
            as_of = as_of_iso or utc_now_iso()
            clauses.append("status = 'active'")
            clauses.append("effective_from <= ?")
            clauses.append("(effective_to IS NULL OR effective_to = '' OR effective_to >= ?)")
            params.extend([as_of, as_of])
        elif status != "all":
            clauses.append("status = ?")
            params.append(status)

        if domain_scope:
            ph = ",".join("?" for _ in domain_scope)
            clauses.append(f"(domain IN ({ph}) OR domain = 'global')")
            params.extend(domain_scope)

        norm_tag = normalize_project_tag(project_tag)
        if strict_untagged_only:
            clauses.append("project_tag IS NULL")
        elif norm_tag is not None:
            if include_untagged_for_project:
                clauses.append("(project_tag = ? OR project_tag IS NULL)")
            else:
                clauses.append("project_tag = ?")
            params.append(norm_tag)

        if ledger_type:
            clauses.append("ledger_type = ?")
            params.append(ledger_type)

        if version_scope:
            clauses.append("(version_scope IS NULL OR version_scope = '' OR version_scope = ?)")
            params.append(version_scope)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = self.conn.execute(
            f"""
            SELECT * FROM memory_records
            {where}
            ORDER BY effective_from DESC, created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        out: list[tuple[MemoryRecord, bytes | None, str | None]] = []
        for r in rows:
            keys = r.keys()
            emb = bytes(r["embedding"]) if ("embedding" in keys and r["embedding"] is not None) else None
            ekey = str(r["embed_key"]) if ("embed_key" in keys and r["embed_key"] is not None) else None
            out.append((self._row_to_memory_record(r), emb, ekey))
        return out

    def get_active_memories(
        self,
        domain_scope: list[str],
        version_scope: str | None = None,
        as_of_iso: str | None = None,
        limit: int = 20,
    ) -> list[MemoryRecord]:
        if not domain_scope:
            return []
        rows = self.get_memory_candidates(
            domain_scope=domain_scope,
            version_scope=version_scope,
            status="active",
            as_of_iso=as_of_iso,
            limit=limit,
        )
        return [rec for rec, _emb, _ekey in rows]

    def count_memories_by_status(self, status: str = "staged", domain_scope: list[str] | None = None) -> int:
        if domain_scope:
            ph = ",".join("?" for _ in domain_scope)
            row = self.conn.execute(
                f"SELECT COUNT(*) FROM memory_records WHERE status = ? AND (domain IN ({ph}) OR domain = 'global')",
                [status, *domain_scope],
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) FROM memory_records WHERE status = ?", (status,)).fetchone()
        return int(row[0] if row else 0)

    def count_memories_missing_embeddings(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM memory_records WHERE status = 'active' AND embedding IS NULL"
        ).fetchone()
        return int(row[0] if row else 0)

    def get_conflict_targets(self, memory_ids: list[str]) -> dict[str, str]:
        if not memory_ids:
            return {}
        src_uris = [f"memory://{mid}" for mid in memory_ids]
        ph = ",".join("?" for _ in src_uris)
        rows = self.conn.execute(
            f"""
            SELECT src_uri, dst_uri FROM edges
            WHERE rel_type = 'CONFLICTS_WITH' AND src_uri IN ({ph})
            ORDER BY confidence DESC
            """,
            src_uris,
        ).fetchall()
        out: dict[str, str] = {}
        for r in rows:
            mid = str(r["src_uri"]).removeprefix("memory://")
            if mid not in out:
                out[mid] = str(r["dst_uri"]).removeprefix("memory://")
        return out

    def upsert_visual_node(
        self,
        *,
        visual_id: str,
        source_uri: str,
        pdf_relpath: str,
        page_num: int,
        bbox: list[float] | tuple[float, ...],
        kind: VisualKind,
        status: VisualStatus,
        page_sha256: str,
        crop_path: str | None = None,
        repaired_md: str | None = None,
        commit: bool = True,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO visual_nodes(
                visual_id, source_uri, pdf_relpath, page_num, bbox_json, kind,
                status, page_sha256, crop_path, repaired_md
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(visual_id) DO UPDATE SET
                status=excluded.status,
                page_sha256=excluded.page_sha256,
                crop_path=COALESCE(excluded.crop_path, visual_nodes.crop_path),
                repaired_md=COALESCE(excluded.repaired_md, visual_nodes.repaired_md)
            """,
            (
                visual_id,
                source_uri,
                pdf_relpath,
                page_num,
                json.dumps(list(bbox)),
                kind,
                status,
                page_sha256,
                crop_path,
                repaired_md,
            ),
        )
        if commit:
            self.conn.commit()

    def get_visual_nodes_for_uris(self, uris: list[str]) -> list[dict[str, Any]]:
        if not uris:
            return []
        ph = ",".join("?" for _ in uris)
        rows = self.conn.execute(
            f"SELECT * FROM visual_nodes WHERE source_uri IN ({ph})",
            uris,
        ).fetchall()
        return [dict(r) for r in rows]

    def add_exemplar(
        self,
        exemplar_id: str,
        query_text: str,
        expected_domain: str,
        expected_action: str,
        embedding_f32: np.ndarray,
        model_sha: str,
        commit: bool = True,
    ) -> None:
        arr = np.asarray(embedding_f32, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 0.0:
            arr = arr / norm
        self.conn.execute(
            """
            INSERT INTO exemplars(id, query_text, expected_domain, expected_action, embedding, model_sha, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                expected_domain=excluded.expected_domain,
                expected_action=excluded.expected_action,
                embedding=excluded.embedding,
                model_sha=excluded.model_sha
            """,
            (exemplar_id, query_text, expected_domain, expected_action, arr.tobytes(), model_sha, utc_now_iso()),
        )
        if commit:
            self.conn.commit()

    def query_exemplars_knn(self, query_f32: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT id, query_text, expected_domain, expected_action, embedding FROM exemplars").fetchall()
        if not rows:
            return []
        q = np.asarray(query_f32, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm > 0.0:
            q = q / q_norm
        scored = []
        for r in rows:
            vec = np.frombuffer(r["embedding"], dtype=np.float32)
            if vec.shape != q.shape:
                continue
            sim = float(np.dot(q, vec))
            scored.append(
                {
                    "id": r["id"],
                    "query_text": r["query_text"],
                    "expected_domain": r["expected_domain"],
                    "expected_action": r["expected_action"],
                    "similarity": round(sim, 4),
                }
            )
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def record_query_topic(self, topic_key: str, domain: str) -> int:
        now = utc_now_iso()
        self.conn.execute(
            """
            INSERT INTO query_counts(topic_key, domain, count, last_queried_at)
            VALUES (?, ?, 1, ?)
            ON CONFLICT(topic_key) DO UPDATE SET
                count = query_counts.count + 1,
                last_queried_at = excluded.last_queried_at
            """,
            (topic_key, domain, now),
        )
        self.conn.commit()
        return int(self.conn.execute("SELECT count FROM query_counts WHERE topic_key = ?", (topic_key,)).fetchone()[0])

    def register_compiled_deps(self, compiled_uri: str, deps: dict[str, str]) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM compiled_deps WHERE compiled_uri = ?", (compiled_uri,))
            for up_uri, up_sha in deps.items():
                self.conn.execute(
                    "INSERT INTO compiled_deps(compiled_uri, upstream_uri, upstream_sha256) VALUES (?, ?, ?)",
                    (compiled_uri, up_uri, up_sha),
                )

    def check_compiled_staleness(self, compiled_uri: str) -> bool:
        rows = self.conn.execute(
            """
            SELECT d.upstream_sha256, c.content_sha256
            FROM compiled_deps d
            LEFT JOIN chunks c ON c.uri = d.upstream_uri
            WHERE d.compiled_uri = ?
            """,
            (compiled_uri,),
        ).fetchall()
        if not rows:
            return False
        return any(r["content_sha256"] != r["upstream_sha256"] for r in rows)

    def pending_embeddings(self, limit: int = 32) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT uri, domain, source_version, text, embed_key FROM chunks WHERE embedding_status = 'pending' LIMIT ?",
            (limit,),
        ).fetchall()

    def integrity_check(self) -> bool:
        res = self.conn.execute("PRAGMA integrity_check;").fetchone()
        return bool(res and res[0] == "ok")

    @classmethod
    def atomic_swap_database(cls, tmp_db: Path, target_db: Path) -> None:
        probe = cls(tmp_db, readonly=False)
        try:
            if not probe.integrity_check():
                raise RuntimeError(f"Integrity check failed for rebuilt database: {tmp_db}")
            probe.conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        finally:
            probe.close()
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(target_db) + suffix)
            if sidecar.exists():
                sidecar.unlink()
            tmp_side = Path(str(tmp_db) + suffix)
            if tmp_side.exists():
                tmp_side.unlink()
        atomic_replace_windows(tmp_db, target_db)
