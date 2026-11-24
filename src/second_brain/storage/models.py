"""Canonical Pydantic contracts shared across all 3 stores, router, daemon, and MCP tools."""
from __future__ import annotations

import re
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator

StoreKind = Literal["wiki", "parsed", "memory", "code"]
AuthorityTier = Literal[
    "primary_normative",
    "primary_reference",
    "compiled_synthesis",
    "working_memory",
]
EdgeType = Literal[
    "CITES",
    "PART_OF",
    "SUPERSEDES",
    "CONFLICTS_WITH",
    "APPLIES_TO",
    "INFERRED_RELATED",
]
EdgeProvenance = Literal["EXTRACTED", "INFERRED"]
EmbeddingStatus = Literal["pending", "ready", "failed"]
MemoryStatus = Literal["staged", "active", "superseded", "expired", "rejected"]
VisualKind = Literal["table", "equation", "diagram"]
VisualStatus = Literal["pending_vision", "cached_unverified", "certified"]

VALID_URI_RE = re.compile(r"^(vault|memory|repo)://[^\s]+$")


def normalize_project_tag(value: str | None) -> str | None:
    """Normalize an optional project tag into a lowercase slug or None."""
    if value is None:
        return None
    cleaned = re.sub(r"[^a-z0-9._-]+", "-", str(value).strip().lower()).strip("-")
    if not cleaned:
        return None
    trimmed = cleaned[:64].strip("-")
    return trimmed or None


class EvidenceEnvelope(BaseModel):
    uri: str
    store: StoreKind
    authority_tier: AuthorityTier
    domain: str
    source_id: str
    source_version: str
    clause_path: str
    text: str
    content_sha256: str
    token_count: int = Field(ge=0)
    status: str = "active"
    score: float = 0.0
    stale: bool = False
    truncated: bool = False
    fetch_span_handle: str | None = None
    visual_ref: str | None = None
    file_relpath: str | None = None
    start_line: int | None = None
    end_line: int | None = None

    @field_validator("uri")
    @classmethod
    def _validate_uri(cls, value: str) -> str:
        if not VALID_URI_RE.match(value):
            raise ValueError(f"Invalid typed provenance URI: {value!r}")
        return value


class IndexCoverage(BaseModel):
    fts_chunks: int = 0
    embedded_chunks: int = 0
    pending_chunks: int = 0
    failed_chunks: int = 0
    coverage_pct: float = 100.0
    index_state: Literal["empty", "complete", "partial_embeddings", "fast_bm25_only"] = "empty"


class StoreStatus(BaseModel):
    store: StoreKind
    state: Literal["active", "unconfigured", "degraded", "error"]
    detail: str
    chunk_count: int = 0


class LinkedClauseRef(BaseModel):
    uri: str
    rel_type: EdgeType
    provenance: EdgeProvenance
    clause_path: str | None = None
    source_version: str | None = None
    token_count: int | None = None
    confidence: float | None = None
    direction: Literal["out", "in"] | None = None
    stale: bool = False


class VisualCropNotice(BaseModel):
    visual_id: str
    source_uri: str
    kind: VisualKind
    page_num: int
    crop_path: str
    status: VisualStatus
    action: str = "VISUAL_CROP_REQUIRED"


class FanOutBranch(BaseModel):
    domain: str
    confidence: float
    scoped_query: str
    specialist_uri: str
    source_versions: list[str] = Field(default_factory=list)
    token_budget: int = 2500


class RouteDecision(BaseModel):
    trace_id: str
    primary_domain: str
    selected_domains: list[str]
    probabilities: dict[str, float]
    multi_domain: bool
    gate: Literal["gate0_explicit", "gate0_exact_id", "gate0_single_domain", "laya_system1", "knn_override"]
    source_versions: list[str] = Field(default_factory=list)
    branches: list[FanOutBranch] = Field(default_factory=list)
    aggregate_token_budget: int = 2500
    laya_latency_ms: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class GroundingPack(BaseModel):
    trace_id: str
    query: str
    domain_scope: list[str]
    source_version_filter: str | None = None
    route: RouteDecision | None = None
    index_coverage: IndexCoverage
    store_statuses: list[StoreStatus] = Field(default_factory=list)
    items: list[EvidenceEnvelope] = Field(default_factory=list)
    linked_manifest: list[LinkedClauseRef] = Field(default_factory=list)
    visual_notices: list[VisualCropNotice] = Field(default_factory=list)
    total_tokens: int = 0
    token_budget: int = 2500
    tokenizer: str = "tiktoken:o200k_base"
    stage_latencies_ms: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class MemoryRecord(BaseModel):
    id: str
    domain: str
    ledger_type: Literal["preferences", "decisions", "facts"] = "facts"
    project_tag: str | None = None
    effective_from: str
    effective_to: str | None = None
    version_scope: str | None = None
    status: MemoryStatus = "staged"
    supersedes_id: str | None = None
    content: str
    source_ref: str
    reason_code: str | None = None
    laya_advisory: dict[str, Any] = Field(default_factory=dict)
    created_at: str

    @field_validator("project_tag", mode="before")
    @classmethod
    def _validate_project_tag(cls, value: str | None) -> str | None:
        return normalize_project_tag(value)


class GrepMatch(BaseModel):
    uri: str
    store: StoreKind
    domain: str
    source_id: str
    source_version: str
    authority_tier: AuthorityTier
    clause_path: str
    text: str
    match_count: int = 1
    match_line: int | None = None
    more_matches_in_chunk: int = 0
    token_count: int = 0
    truncated: bool = False
    stale: bool = False


class GrepOverflowEntry(BaseModel):
    uri: str
    clause_path: str
    authority_tier: AuthorityTier
    match_count: int = 1


class GrepResult(BaseModel):
    trace_id: str
    pattern: str
    regex: bool = False
    case_sensitive: bool = False
    render_mode: Literal["full", "snippets"] = "full"
    total_matching_chunks: int = 0
    total_occurrences: int = 0
    returned_chunks: int = 0
    total_tokens: int = 0
    token_budget: int = 2500
    truncated: bool = False
    items: list[GrepMatch] = Field(default_factory=list)
    overflow_manifest: list[GrepOverflowEntry] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    elapsed_ms: float = 0.0


class ClauseLookupResult(BaseModel):
    found: bool
    uri: str
    item: dict[str, Any] | None = None
    text_omitted: bool = False
    neighbors: list[LinkedClauseRef] = Field(default_factory=list)
    children: list[LinkedClauseRef] = Field(default_factory=list)
    cited_by: list[LinkedClauseRef] = Field(default_factory=list)
    inferred_links: list[LinkedClauseRef] = Field(default_factory=list)
    totals: dict[str, int] = Field(default_factory=dict)
    visual_nodes: list[dict[str, Any]] = Field(default_factory=list)


class MemoryRecallItem(BaseModel):
    id: str
    domain: str
    ledger_type: Literal["preferences", "decisions", "facts"] = "facts"
    project_tag: str | None = None
    effective_from: str
    effective_to: str | None = None
    version_scope: str | None = None
    status: MemoryStatus = "active"
    supersedes_id: str | None = None
    content: str
    source_ref: str
    reason_code: str | None = None
    created_at: str
    similarity: float | None = None
    lexical_score: float = 0.0
    hybrid_score: float = 0.0
    conflict_target_id: str | None = None
    certify_hint: str | None = None


class MemoryRecallResult(BaseModel):
    trace_id: str
    status: str = "active"
    project_tag: str | None = None
    total_matched: int = 0
    returned_count: int = 0
    items: list[MemoryRecallItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    elapsed_ms: float = 0.0
