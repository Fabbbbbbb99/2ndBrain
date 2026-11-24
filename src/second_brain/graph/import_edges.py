"""2-step human-certified Graphify INFERRED_RELATED edge importer and Vault JSONL ledger."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

from second_brain.config import BrainConfig
from second_brain.observability.tracer import TraceLogger, new_trace_id
from second_brain.router.retrieval import _structural_noise_multiplier
from second_brain.storage.models import EvidenceEnvelope
from second_brain.storage.sqlite_store import SqliteStore, atomic_write_lf, utc_now_iso

_LINE_RE = re.compile(r"L?(\d+)", re.IGNORECASE)


def _normalize_relpath_candidates(raw_path: str, cfg: BrainConfig) -> list[str]:
    """Normalize a Windows/POSIX Graphify source_file into candidate chunks.file_relpath keys."""
    norm = raw_path.replace("\\", "/").strip()
    if norm.startswith("./"):
        norm = norm[2:]
    low = norm.lower()
    for base in (cfg.state_dir.as_posix().rstrip("/") + "/", cfg.vault_path.as_posix().rstrip("/") + "/"):
        base_low = base.lower()
        if low.startswith(base_low):
            norm = norm[len(base) :]
            low = norm.lower()
            break
    parsed_base = cfg.parsed_dir.as_posix().rstrip("/").lower() + "/"
    if low.startswith(parsed_base):
        norm = "parsed/" + norm[len(parsed_base) :]
        low = norm.lower()

    cands = [low]
    if not low.startswith("parsed/") and not low.startswith("domains/"):
        cands.append(f"parsed/{low}")
    return cands


def _parse_line_num(loc: Any) -> int | None:
    if loc is None:
        return None
    if isinstance(loc, int):
        return loc
    m = _LINE_RE.search(str(loc))
    return int(m.group(1)) if m else None


def _resolve_confidence(link: dict[str, Any]) -> tuple[str, float]:
    label = str(link.get("confidence", "INFERRED")).strip().upper()
    raw_score = link.get("confidence_score")
    if raw_score is not None:
        return label, float(raw_score)
    if label == "EXTRACTED":
        return label, 0.90
    if label == "INFERRED":
        return label, 0.80
    return label, 0.40


def stage_graphify_edges(
    *,
    cfg: BrainConfig,
    store: SqliteStore,
    graph_path: Path,
    out_path: Path,
    min_confidence: float = 0.80,
    cross_doc_only: bool = False,
) -> dict[str, Any]:
    """Resolve Graphify (source_file, source_location) links onto chunks URIs into review.yaml."""
    raw_bytes = graph_path.read_bytes()
    graph_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    data = json.loads(raw_bytes.decode("utf-8"))
    nodes = data.get("nodes", [])
    links = data.get("links") if "links" in data else data.get("edges", [])

    # Preload chunks indexed by lowercase file_relpath
    rows = store.conn.execute(
        """
        SELECT uri, store, domain, source_id, source_version, clause_path, authority_tier,
               file_relpath, start_line, end_line, content_sha256, text, token_count
        FROM chunks
        WHERE store IN ('parsed', 'wiki')
          AND file_relpath IS NOT NULL
          AND start_line IS NOT NULL
          AND end_line IS NOT NULL
        """
    ).fetchall()

    by_file: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        env = EvidenceEnvelope(
            uri=r["uri"],
            store=r["store"],
            authority_tier=r["authority_tier"],
            domain=r["domain"],
            source_id=r["source_id"],
            source_version=r["source_version"],
            clause_path=r["clause_path"],
            text=r["text"],
            content_sha256=r["content_sha256"],
            token_count=int(r["token_count"]),
        )
        entry = {
            "uri": str(r["uri"]),
            "source_id": str(r["source_id"]),
            "clause_path": str(r["clause_path"]),
            "start_line": int(r["start_line"]),
            "end_line": int(r["end_line"]),
            "sha256": str(r["content_sha256"]),
            "is_noise": _structural_noise_multiplier(env) < 1.0,
        }
        rel_low = str(r["file_relpath"]).replace("\\", "/").lower()
        by_file.setdefault(rel_low, []).append(entry)

    for spans in by_file.values():
        spans.sort(key=lambda item: (item["start_line"], item["end_line"], item["uri"]))

    extracted_pairs: set[tuple[str, str]] = set()
    for er in store.conn.execute("SELECT src_uri, dst_uri FROM edges WHERE provenance = 'EXTRACTED'").fetchall():
        extracted_pairs.add((str(er["src_uri"]), str(er["dst_uri"])))
        extracted_pairs.add((str(er["dst_uri"]), str(er["src_uri"])))

    node_to_chunk: dict[str, dict[str, Any]] = {}
    unmapped_nodes: list[dict[str, Any]] = []
    for n in nodes:
        nid = str(n.get("id", ""))
        src_file = n.get("source_file")
        line_num = _parse_line_num(n.get("source_location"))
        if not nid or not src_file or line_num is None:
            unmapped_nodes.append({"id": nid, "source_file": src_file, "source_location": n.get("source_location")})
            continue

        matched_chunk: dict[str, Any] | None = None
        for cand_rel in _normalize_relpath_candidates(str(src_file), cfg):
            spans = by_file.get(cand_rel)
            if not spans:
                continue
            for sp in spans:
                if sp["start_line"] <= line_num <= sp["end_line"]:
                    matched_chunk = sp
                    break
            if matched_chunk is None:
                # Allow trailing blank separator line (end_line + 1)
                for sp in spans:
                    if sp["start_line"] <= line_num <= (sp["end_line"] + 1):
                        matched_chunk = sp
                        break
            if matched_chunk is not None:
                break

        if matched_chunk is None:
            unmapped_nodes.append({"id": nid, "source_file": src_file, "source_location": n.get("source_location")})
        else:
            node_to_chunk[nid] = matched_chunk

    staged_map: dict[tuple[str, str], dict[str, Any]] = {}
    stats = {
        "total_nodes": len(nodes),
        "mapped_nodes": len(node_to_chunk),
        "unmapped_nodes": len(unmapped_nodes),
        "total_links": len(links),
        "skipped_ambiguous": 0,
        "skipped_low_confidence": 0,
        "skipped_unmapped_endpoint": 0,
        "skipped_self_loop": 0,
        "skipped_noise_chunk": 0,
        "skipped_same_source": 0,
        "skipped_existing_extracted": 0,
        "staged_edges": 0,
    }

    for link in links:
        conf_label, conf_score = _resolve_confidence(link)
        if conf_label == "AMBIGUOUS":
            stats["skipped_ambiguous"] += 1
            continue
        if conf_score < min_confidence:
            stats["skipped_low_confidence"] += 1
            continue
        src_chunk = node_to_chunk.get(str(link.get("source", "")))
        dst_chunk = node_to_chunk.get(str(link.get("target", "")))
        if src_chunk is None or dst_chunk is None:
            stats["skipped_unmapped_endpoint"] += 1
            continue
        if src_chunk["uri"] == dst_chunk["uri"]:
            stats["skipped_self_loop"] += 1
            continue
        if src_chunk["is_noise"] or dst_chunk["is_noise"]:
            stats["skipped_noise_chunk"] += 1
            continue
        if cross_doc_only and src_chunk["source_id"] == dst_chunk["source_id"]:
            stats["skipped_same_source"] += 1
            continue
        if (src_chunk["uri"], dst_chunk["uri"]) in extracted_pairs:
            stats["skipped_existing_extracted"] += 1
            continue

        a, b = (
            (src_chunk, dst_chunk)
            if src_chunk["uri"] < dst_chunk["uri"]
            else (dst_chunk, src_chunk)
        )
        key = (a["uri"], b["uri"])
        prev = staged_map.get(key)
        if prev is None or conf_score > float(prev["confidence"]):
            staged_map[key] = {
                "src_uri": a["uri"],
                "src_clause_path": a["clause_path"],
                "dst_uri": b["uri"],
                "dst_clause_path": b["clause_path"],
                "rel_type": "INFERRED_RELATED",
                "original_relation": str(link.get("relation", "related")),
                "confidence": round(conf_score, 4),
                "src_sha256": a["sha256"],
                "dst_sha256": b["sha256"],
                "cross_doc": a["source_id"] != b["source_id"],
                "keep": True,
            }

    edges_out = sorted(
        staged_map.values(),
        key=lambda item: (0 if item["cross_doc"] else 1, -float(item["confidence"]), item["src_uri"], item["dst_uri"]),
    )
    stats["staged_edges"] = len(edges_out)

    manifest_doc = {
        "schema_version": "1.0",
        "generated_at": utc_now_iso(),
        "source_graph": str(graph_path.resolve()),
        "source_graph_sha256": graph_sha256,
        "min_confidence": min_confidence,
        "cross_doc_only": cross_doc_only,
        "stats": stats,
        "unmapped_sample": unmapped_nodes[:20],
        "edges": edges_out,
    }
    atomic_write_lf(out_path, yaml.safe_dump(manifest_doc, sort_keys=False, allow_unicode=True))
    return {"out_path": str(out_path), **stats}


def load_certified_edge_hashes(cfg: BrainConfig) -> dict[tuple[str, str], tuple[str, str]]:
    """Load certified endpoint SHA-256 pairs keyed by canonical (src_uri, dst_uri)."""
    path = cfg.certified_edges_path
    if not path.exists():
        return {}
    out: dict[tuple[str, str], tuple[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        src = str(obj["src_uri"])
        dst = str(obj["dst_uri"])
        s_sha = str(obj.get("src_sha256", ""))
        d_sha = str(obj.get("dst_sha256", ""))
        if src < dst:
            out[(src, dst)] = (s_sha, d_sha)
        else:
            out[(dst, src)] = (d_sha, s_sha)
    return out


def sync_certified_edges_from_vault(cfg: BrainConfig, store: SqliteStore) -> int:
    """Re-apply existing Vault certified_edges.jsonl rows whose URIs exist in SQLite."""
    path = cfg.certified_edges_path
    if not path.exists():
        return 0
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if store.get_chunk_by_uri(str(obj["src_uri"])) and store.get_chunk_by_uri(str(obj["dst_uri"])):
            rows.append(obj)
    if not rows:
        return 0
    res = store.apply_inferred_edges(rows, replace_inferred=False)
    return int(res["inserted"])


def apply_certified_edges(
    *,
    cfg: BrainConfig,
    store: SqliteStore,
    tracer: TraceLogger,
    review_path: Path | None = None,
    edges_payload: list[dict[str, Any]] | None = None,
    replace_inferred: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate reviewed edges, persist Vault certified_edges.jsonl, and commit SQLite INFERRED_RELATED rows."""
    trace_id = new_trace_id()
    if edges_payload is None:
        if review_path is None or not review_path.exists():
            raise FileNotFoundError(f"Review YAML not found: {review_path}")
        doc = yaml.safe_load(review_path.read_text(encoding="utf-8")) or {}
        raw_edges = list(doc.get("edges") or [])
    else:
        raw_edges = list(edges_payload)

    now_iso = utc_now_iso()
    validated: list[dict[str, Any]] = []
    drift_warnings: list[str] = []
    skipped_dropped = 0

    for row in raw_edges:
        if not bool(row.get("keep", True)):
            skipped_dropped += 1
            continue
        rel_type = str(row.get("rel_type", "INFERRED_RELATED"))
        if rel_type != "INFERRED_RELATED":
            raise ValueError(f"Unsupported rel_type in edge review manifest: {rel_type!r} (only INFERRED_RELATED allowed)")
        raw_src = str(row["src_uri"]).strip()
        raw_dst = str(row["dst_uri"]).strip()
        if raw_src == raw_dst:
            raise ValueError(f"Self-loop disallowed in edge review manifest: {raw_src!r}")
        src_chunk = store.get_chunk_by_uri(raw_src)
        dst_chunk = store.get_chunk_by_uri(raw_dst)
        if src_chunk is None or dst_chunk is None:
            missing = raw_src if src_chunk is None else raw_dst
            raise ValueError(f"Edge endpoint URI not found in chunks: {missing!r}")

        if raw_src < raw_dst:
            a_uri, b_uri = raw_src, raw_dst
            a_cur_sha, b_cur_sha = src_chunk.content_sha256, dst_chunk.content_sha256
            a_staged_sha = row.get("src_sha256")
            b_staged_sha = row.get("dst_sha256")
        else:
            a_uri, b_uri = raw_dst, raw_src
            a_cur_sha, b_cur_sha = dst_chunk.content_sha256, src_chunk.content_sha256
            a_staged_sha = row.get("dst_sha256")
            b_staged_sha = row.get("src_sha256")

        if (a_staged_sha and a_staged_sha != a_cur_sha) or (b_staged_sha and b_staged_sha != b_cur_sha):
            drift_warnings.append(f"Chunk SHA-256 drifted since stage for edge ({a_uri} <-> {b_uri}); recording live hash.")

        validated.append(
            {
                "src_uri": a_uri,
                "dst_uri": b_uri,
                "rel_type": "INFERRED_RELATED",
                "provenance": "INFERRED",
                "confidence": round(float(row.get("confidence", 0.80)), 4),
                "src_sha256": a_cur_sha,
                "dst_sha256": b_cur_sha,
                "original_relation": str(row.get("original_relation", "related")),
                "certified_at": now_iso,
            }
        )

    # Deduplicate canonically keeping max confidence
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in validated:
        k = (entry["src_uri"], entry["dst_uri"])
        prev = deduped.get(k)
        if prev is None or float(entry["confidence"]) > float(prev["confidence"]):
            deduped[k] = entry
    final_rows = sorted(deduped.values(), key=lambda x: (x["src_uri"], x["dst_uri"]))

    db_counts = {"deleted": 0, "inserted": 0}
    if not dry_run:
        existing_vault: dict[tuple[str, str], dict[str, Any]] = {}
        if not replace_inferred and cfg.certified_edges_path.exists():
            for line in cfg.certified_edges_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                existing_vault[(str(obj["src_uri"]), str(obj["dst_uri"]))] = obj
        for row in final_rows:
            existing_vault[(row["src_uri"], row["dst_uri"])] = row
        vault_sorted = sorted(existing_vault.values(), key=lambda x: (x["src_uri"], x["dst_uri"]))
        jsonl_text = "\n".join(json.dumps(r, ensure_ascii=False) for r in vault_sorted)
        if jsonl_text:
            jsonl_text += "\n"
        atomic_write_lf(cfg.certified_edges_path, jsonl_text)
        db_counts = store.apply_inferred_edges(final_rows, replace_inferred=replace_inferred)

    tracer.write(
        {
            "event": "import_edges",
            "trace_id": trace_id,
            "dry_run": dry_run,
            "replace_inferred": replace_inferred,
            "kept_edges": len(final_rows),
            "pruned_edges": skipped_dropped,
            "deleted": db_counts["deleted"],
            "inserted": db_counts["inserted"],
            "drift_warnings": len(drift_warnings),
        }
    )

    return {
        "trace_id": trace_id,
        "dry_run": dry_run,
        "replace_inferred": replace_inferred,
        "kept_edges": len(final_rows),
        "pruned_edges": skipped_dropped,
        "deleted_inferred": db_counts["deleted"],
        "inserted_inferred": db_counts["inserted"],
        "certified_edges_path": str(cfg.certified_edges_path),
        "drift_warnings": drift_warnings,
    }
