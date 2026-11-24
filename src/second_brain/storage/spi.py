"""Service Provider Interface (SPI) for federated 3-store adapters."""
from __future__ import annotations

from typing import Protocol, runtime_checkable
from second_brain.storage.models import EvidenceEnvelope, StoreStatus


class StoreUnconfiguredError(RuntimeError):
    """Raised when querying an unconfigured adapter such as Milestone 2 Store 3."""


@runtime_checkable
class StoreAdapter(Protocol):
    """Unified interface implemented by Store 1 (Wiki/Parsed), Store 2 (Memory), and Store 3 (Code)."""

    def search(
        self,
        query: str,
        domain_scope: list[str],
        version_filter: str | None,
        limit: int,
    ) -> list[EvidenceEnvelope]: ...

    def get_by_uri(self, uri: str) -> EvidenceEnvelope | None: ...

    def health(self) -> StoreStatus: ...


class CodeAstAdapter:
    """Milestone 2 Store 3 (Per-Repo Code AST Adapter) interface stub.

    Explicitly reports `unconfigured` status and raises StoreUnconfiguredError on direct search
    so callers never receive silent empty hits.
    """

    def __init__(self, workspace_root: str | None = None) -> None:
        self.workspace_root = workspace_root

    def health(self) -> StoreStatus:
        return StoreStatus(
            store="code",
            state="unconfigured",
            detail="Store 3 Per-Repo Code AST Adapter is scheduled for Milestone 2 and not configured.",
            chunk_count=0,
        )

    def search(
        self,
        query: str,
        domain_scope: list[str],
        version_filter: str | None,
        limit: int,
    ) -> list[EvidenceEnvelope]:
        raise StoreUnconfiguredError(
            "Store 3 Per-Repo Code AST Adapter is unconfigured (Milestone 2). "
            "Pass store='wiki'|'parsed'|'memory' or use native workspace code search."
        )

    def get_by_uri(self, uri: str) -> EvidenceEnvelope | None:
        raise StoreUnconfiguredError(
            f"Cannot dereference {uri!r}: Store 3 Per-Repo Code AST Adapter is unconfigured (Milestone 2)."
        )
