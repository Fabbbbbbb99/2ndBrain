"""Build Laya training, validation, and calibration JSONL datasets from Vault + certified traces."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import yaml

from second_brain.config import load_config
from second_brain.storage.sqlite_store import SqliteStore


def _load_domain_specs(vault_path: Path) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for yf in sorted((vault_path / "specialists").glob("*.yaml")):
        raw = yaml.safe_load(yf.read_text(encoding="utf-8")) or {}
        if "domain_id" in raw:
            specs[str(raw["domain_id"])] = raw
    return specs


def build_laya_datasets(
    *,
    vault_path: Path,
    state_dir: Path,
    out_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    rng = random.Random(seed)
    specs = _load_domain_specs(vault_path)
    domain_ids = sorted(specs.keys())
    if len(domain_ids) < 2:
        raise ValueError(
            f"Laya fine-tuning requires at least 2 domain specialists under {vault_path / 'specialists'}; "
            f"found {len(domain_ids)} ({domain_ids})."
        )
    criteria = {d: str(specs[d].get("description", d))[:120] for d in domain_ids}
    tax = yaml.safe_load((vault_path / "ontology" / "taxonomy.yaml").read_text(encoding="utf-8")) or {}
    tax_domains = tax.get("domains") or {}

    records: list[dict[str, Any]] = []

    # 1. Synthetic domain routing examples from taxonomy aliases + specialist instructions
    for idx, dom in enumerate(domain_ids):
        aliases = list((tax_domains.get(dom) or {}).get("aliases") or [])
        desc = str(specs[dom].get("description", dom))
        y_idx = domain_ids.index(dom)
        prompts = [
            f"Verify compliance and architecture constraints for {dom}: {desc}",
            *[f"Technical specification question regarding {a} in {dom}" for a in aliases],
        ]
        for p in prompts:
            records.append(
                {
                    "state": p,
                    "src": f"synthetic_taxonomy:{dom}",
                    "qs": [
                        {
                            "t": "choice",
                            "ins": "Select the primary engineering domain for this technical request.",
                            "crit": criteria,
                            "y": y_idx,
                        }
                    ],
                }
            )

    # 2. Ingested clause/section headings from SQLite
    db_path = state_dir / "brain.sqlite"
    if db_path.exists():
        store = SqliteStore(db_path, readonly=True)
        try:
            rows = store.conn.execute(
                "SELECT domain, clause_path, text FROM chunks WHERE store IN ('parsed','wiki') LIMIT 2000"
            ).fetchall()
            for r in rows:
                dom = r["domain"]
                if dom not in domain_ids:
                    continue
                snippet = f"{r['clause_path']}: {r['text'][:350]}"
                records.append(
                    {
                        "state": snippet,
                        "src": "ingested_chunk",
                        "qs": [
                            {
                                "t": "choice",
                                "ins": "Select the primary engineering domain for this technical request.",
                                "crit": criteria,
                                "y": domain_ids.index(dom),
                            }
                        ],
                    }
                )
        finally:
            store.close()

    # 3. Real logged human certifications + priority hard negatives
    cert_log = state_dir / "logs" / "certify_traces.jsonl"
    hard_neg_count = 0
    if cert_log.exists():
        for line in cert_log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            dom = item.get("final_domain")
            if dom in domain_ids:
                weight_repeat = 3 if item.get("hard_negative") else 1
                if item.get("hard_negative"):
                    hard_neg_count += 1
                for _ in range(weight_repeat):
                    records.append(
                        {
                            "state": str(item.get("text", ""))[:500],
                            "src": "certify_hard_negative" if item.get("hard_negative") else "certify_trace",
                            "qs": [
                                {
                                    "t": "choice",
                                    "ins": "Select the primary engineering domain for this technical request.",
                                    "crit": criteria,
                                    "y": domain_ids.index(dom),
                                }
                            ],
                        }
                    )

    rng.shuffle(records)
    n = len(records)
    n_train = max(1, int(n * 0.7))
    n_val = max(1, int(n * 0.15)) if n >= 3 else 0
    train_split = records[:n_train]
    val_split = records[n_train : n_train + n_val]
    cal_split = records[n_train + n_val :] or val_split

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train.jsonl", train_split), ("val.jsonl", val_split), ("calibration.jsonl", cal_split)):
        with (out_dir / name).open("w", encoding="utf-8", newline="\n") as fh:
            for row in split:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "total_records": n,
        "train": len(train_split),
        "val": len(val_split),
        "calibration": len(cal_split),
        "hard_negatives": hard_neg_count,
        "choice_labels": domain_ids,
        "out_dir": str(out_dir),
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Laya training JSONL splits from 2ndBrain Vault & traces")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    summary = build_laya_datasets(
        vault_path=cfg.vault_path,
        state_dir=cfg.state_dir,
        out_dir=Path(args.out_dir).expanduser().resolve(),
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
