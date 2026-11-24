"""Rotating trace_id JSONL logger and `brain explain` lookup."""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


class TraceLogger:
    """Appends structured JSONL trace records to state/logs/traces-YYYY-MM-DD.jsonl."""

    def __init__(self, logs_dir: Path) -> None:
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _today_file(self) -> Path:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.logs_dir / f"traces-{day}.jsonl"

    def write(self, record: dict[str, Any]) -> Path:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **record,
        }
        target = self._today_file()
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        with self._lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(line)
        return target

    def find_trace(self, trace_id: str) -> dict[str, Any] | None:
        if not self.logs_dir.exists():
            return None
        matches: list[dict[str, Any]] = []
        files = sorted(self.logs_dir.glob("traces-*.jsonl"), reverse=True)
        for f in files:
            with f.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("trace_id") == trace_id:
                        matches.append(obj)
        if not matches:
            return None
        merged: dict[str, Any] = {}
        for item in matches:
            merged.update(item)
        merged["events"] = matches
        return merged
