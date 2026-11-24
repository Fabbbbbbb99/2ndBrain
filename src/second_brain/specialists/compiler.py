"""Explicit CQS wiki compiler (`brain_compile`) validating parsed/ provenance hashes."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import tiktoken

from second_brain.config import BrainConfig
from second_brain.storage.sqlite_store import SqliteStore, atomic_write_lf, lf_sha256, utc_now_iso

_ENC = tiktoken.get_encoding("o200k_base")


def _slug(val: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", val.strip()).strip("_").lower()[:48] or "topic"


def compile_wiki_page(
    *,
    cfg: BrainConfig,
    store: SqliteStore,
    domain: str,
    slug: str,
    title: str,
    summary_markdown: str,
    cited_uris: list[str],
    source_version: str = "compiled",
) -> dict[str, Any]:
    """Validate all cited parsed/ URIs and persist a Compiled Truth page under domains/<domain>/."""
    if not cited_uris:
        raise ValueError("brain_compile requires at least one verified cited URI in cited_uris.")

    deps: dict[str, str] = {}
    for uri in cited_uris:
        chunk = store.get_chunk_by_uri(uri)
        if chunk is None:
            raise ValueError(f"Cannot compile wiki page: cited URI does not exist in SQLite: {uri!r}")
        deps[uri] = chunk.content_sha256

    safe_dom = _slug(domain)
    safe_slug = _slug(slug)
    wiki_uri = f"vault://wiki_{safe_dom}@{_slug(source_version)}#{safe_slug}"
    out_file = cfg.vault_path / "domains" / safe_dom / f"{safe_slug}.md"
    rel_path = f"domains/{safe_dom}/{safe_slug}.md"

    now = utc_now_iso()
    upstream_yaml = "\n".join(f"  {u!r}: {h!r}" for u, h in deps.items())
    citations_md = "\n".join(f"- `{u}` (`sha256:{h[:12]}`)" for u, h in deps.items())

    page_text = (
        f"---\n"
        f"uri: {wiki_uri!r}\n"
        f"domain: {safe_dom}\n"
        f"authority_tier: compiled_synthesis\n"
        f"source_version: {source_version!r}\n"
        f"updated_at: {now!r}\n"
        f"upstream_hashes:\n{upstream_yaml}\n"
        f"---\n\n"
        f"# {title.strip()}\n\n"
        f"## Compiled Truth\n\n"
        f"{summary_markdown.strip()}\n\n"
        f"## Verified Source Citations\n\n"
        f"{citations_md}\n\n"
        f"## Timeline\n\n"
        f"- `{now[:10]}` Compiled from {len(deps)} verified source chunks.\n"
    )
    atomic_write_lf(out_file, page_text)

    store.upsert_chunk(
        uri=wiki_uri,
        store="wiki",
        domain=domain,
        source_id=f"wiki_{safe_dom}",
        source_version=source_version,
        clause_path=f"{domain} > Compiled > {title.strip()}",
        parent_uri=None,
        authority_tier="compiled_synthesis",
        file_relpath=rel_path,
        start_line=1,
        end_line=len(page_text.splitlines()),
        text=page_text,
        token_count=len(_ENC.encode(page_text)),
        embed_key=lf_sha256(f"{cfg.vault_path.name}:bge-small-v1.5:v1:{page_text}"),
        commit=False,
    )
    store.register_compiled_deps(wiki_uri, deps)
    for cited in cited_uris:
        store.upsert_edge(wiki_uri, cited, "CITES", "EXTRACTED", 1.0, commit=False)
    store.conn.commit()

    # Update index.md and log.md cleanly
    index_path = cfg.vault_path / "index.md"
    idx_txt = index_path.read_text(encoding="utf-8").replace("\r\n", "\n") if index_path.exists() else "# 2ndBrain Wiki Index\n"
    link_line = f"- [{title.strip()}]({rel_path}) (`{wiki_uri}`)"
    if rel_path not in idx_txt:
        atomic_write_lf(index_path, idx_txt.rstrip("\n") + "\n" + link_line + "\n")

    log_path = cfg.vault_path / "log.md"
    log_txt = log_path.read_text(encoding="utf-8").replace("\r\n", "\n") if log_path.exists() else "# 2ndBrain Activity Log\n"
    log_entry = f"\n## [{now[:10]}] compile | {title.strip()}\n- Compiled `{wiki_uri}` citing {len(deps)} verified URIs.\n"
    atomic_write_lf(log_path, log_txt.rstrip("\n") + "\n" + log_entry)

    return {
        "compiled": True,
        "wiki_uri": wiki_uri,
        "file_path": str(out_file),
        "cited_count": len(deps),
        "upstream_hashes": deps,
    }
