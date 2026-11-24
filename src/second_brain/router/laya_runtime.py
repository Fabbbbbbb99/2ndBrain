"""Pure-ONNX CPU runtime for convaiinnovations/laya models with signed manifest checks and rollback."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import yaml
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

from second_brain.config import BrainConfig
from second_brain.storage.sqlite_store import atomic_write_lf

QTYPES = {"choice": 0, "score": 1, "noul": 2}
STOCK_LAYA_ONNX_REPO = "yehor-oleksiuk/laya-typed-decisions-onnx"
STOCK_LAYA_CFG_REPO = "convaiinnovations/laya-typed-decisions"
STOCK_LAYA_SHA256 = "78553ff45b2f656a8b6bd412f52d8d3e575fafb95560bc33d75aca8d95161716"


class LayaManifestError(RuntimeError):
    """Raised when model_manifest.json fails integrity or schema verification."""


class LayaModelManifest(BaseModel):
    model_id: str
    onnx_file: str = "model_int8.onnx"
    model_sha256: str
    router_contract_version: str = "1.0.0"
    quant_recipe: str
    choice_labels: list[str]
    max_len: int = Field(default=1024, le=1024)
    head_max_len: int = Field(default=256, le=512)
    temperature_config: dict[str, Any] = Field(default_factory=dict)
    max_prob_delta_vs_fp32: float = 0.0
    ece_score: float | None = None
    is_purpose_trained: bool = False
    allow_laya_auto_commit: bool = False


def active_laya_pointer_path(cfg: BrainConfig) -> Path:
    return cfg.state_dir / "model_pointer.json"


def resolve_active_laya_dir(cfg: BrainConfig) -> Path:
    ptr = active_laya_pointer_path(cfg)
    if ptr.exists():
        data = json.loads(ptr.read_text(encoding="utf-8"))
        active = Path(data["active_dir"])
        if not active.is_absolute():
            active = cfg.models_dir / active
        return active.resolve()
    return cfg.laya_dir


def _vault_specialist_domains(vault_path: Path) -> set[str]:
    spec_dir = vault_path / "specialists"
    if not spec_dir.exists():
        return set()
    domains: set[str] = set()
    for yf in sorted(spec_dir.glob("*.yaml")):
        raw = yaml.safe_load(yf.read_text(encoding="utf-8")) or {}
        if "domain_id" in raw:
            domains.add(str(raw["domain_id"]))
    return domains


def read_laya_manifest(model_dir: Path, vault_path: Path | None = None) -> LayaModelManifest:
    manifest_path = Path(model_dir) / "model_manifest.json"
    if not manifest_path.exists():
        raise LayaManifestError(f"Missing signed model_manifest.json in {model_dir}")
    manifest = LayaModelManifest(**json.loads(manifest_path.read_text(encoding="utf-8")))
    onnx_path = Path(model_dir) / manifest.onnx_file
    if not onnx_path.exists():
        raise LayaManifestError(f"Missing ONNX model referenced by manifest: {onnx_path}")
    actual_sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    if actual_sha != manifest.model_sha256:
        raise LayaManifestError(
            f"SHA-256 mismatch on {onnx_path}: manifest={manifest.model_sha256[:12]} actual={actual_sha[:12]}"
        )
    if vault_path is not None:
        expected_domains = _vault_specialist_domains(vault_path)
        is_stock_wildcard = (not manifest.is_purpose_trained) and ("*" in manifest.choice_labels)
        if expected_domains and (not is_stock_wildcard) and not expected_domains.issubset(set(manifest.choice_labels)):
            missing = sorted(expected_domains - set(manifest.choice_labels))
            raise LayaManifestError(
                f"Laya choice_labels {manifest.choice_labels} diverge from specialists/*.yaml missing={missing}"
            )
    return manifest


def pull_stock_models(
    cfg: BrainConfig,
    *,
    allow_unverified: bool = False,
    skip_fastembed: bool = False,
) -> dict[str, Any]:
    """Download stock Laya INT8 ONNX assets + calibrated temperatures and optional FastEmbed models."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from huggingface_hub import hf_hub_download

    laya_dir = cfg.models_dir / "laya_stock"
    laya_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = Path(hf_hub_download(STOCK_LAYA_ONNX_REPO, "model_int8.onnx", local_dir=str(laya_dir)))
    Path(hf_hub_download(STOCK_LAYA_ONNX_REPO, "tokenizer.json", local_dir=str(laya_dir)))
    Path(hf_hub_download(STOCK_LAYA_ONNX_REPO, "tokenizer_config.json", local_dir=str(laya_dir)))
    cfg_path = Path(hf_hub_download(STOCK_LAYA_CFG_REPO, "rl_agent_config.json", local_dir=str(laya_dir)))

    with onnx_path.open("rb") as fh:
        actual_sha = hashlib.file_digest(fh, "sha256").hexdigest()
    sha_verified = actual_sha == STOCK_LAYA_SHA256
    if (not sha_verified) and (not allow_unverified):
        raise LayaManifestError(
            f"Stock Laya SHA-256 mismatch on {onnx_path}: expected={STOCK_LAYA_SHA256[:12]} actual={actual_sha[:12]}. "
            "Pass --allow-unverified to accept an updated upstream checkpoint."
        )

    rl_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    manifest_payload: dict[str, Any] = {
        "model_id": STOCK_LAYA_ONNX_REPO,
        "onnx_file": "model_int8.onnx",
        "model_sha256": actual_sha,
        "router_contract_version": "1.0.0",
        "quant_recipe": "matmul_nbits_8bit_block64_fp32_head",
        "choice_labels": ["*"],
        "max_len": min(1024, int(rl_cfg.get("max_len", 1024))),
        "head_max_len": min(512, int(rl_cfg.get("head_max_len", 256))),
        "temperature_config": {
            "temperature": list(rl_cfg.get("temperature", [1.0, 1.0, 1.0])),
            "temperature_by_options": dict(rl_cfg.get("temperature_by_options", {})),
        },
        "max_prob_delta_vs_fp32": 0.0,
        "ece_score": None,
        "is_purpose_trained": False,
        "allow_laya_auto_commit": False,
    }
    atomic_write_lf(laya_dir / "model_manifest.json", json.dumps(manifest_payload, indent=2))
    verified = read_laya_manifest(laya_dir, vault_path=cfg.vault_path)

    fastembed_prefetched: list[str] = []
    if not skip_fastembed:
        from fastembed import TextEmbedding
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        cfg.fastembed_dir.mkdir(parents=True, exist_ok=True)
        TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir=str(cfg.fastembed_dir),
            threads=min(6, cfg.intra_op_num_threads),
            providers=["CPUExecutionProvider"],
            enable_cpu_mem_arena=cfg.enable_cpu_mem_arena,
        )
        fastembed_prefetched.append("BAAI/bge-small-en-v1.5")
        TextCrossEncoder(
            model_name="Xenova/ms-marco-MiniLM-L-6-v2",
            cache_dir=str(cfg.fastembed_dir),
            threads=min(6, cfg.intra_op_num_threads),
            providers=["CPUExecutionProvider"],
            enable_cpu_mem_arena=cfg.enable_cpu_mem_arena,
        )
        fastembed_prefetched.append("Xenova/ms-marco-MiniLM-L-6-v2")

    return {
        "laya_dir": str(laya_dir),
        "model_sha256": verified.model_sha256,
        "sha_verified_against_pinned": sha_verified,
        "choice_labels": verified.choice_labels,
        "fastembed_dir": str(cfg.fastembed_dir),
        "fastembed_prefetched": fastembed_prefetched,
    }


def swap_laya_model(cfg: BrainConfig, new_model_dir: Path) -> dict[str, Any]:
    target = Path(new_model_dir).expanduser().resolve()
    manifest = read_laya_manifest(target, cfg.vault_path)
    if not (target / "tokenizer.json").exists():
        raise LayaManifestError(f"Missing tokenizer.json in {target}")
    current = resolve_active_laya_dir(cfg)
    ptr_path = active_laya_pointer_path(cfg)
    history: list[str] = []
    if ptr_path.exists():
        prev_data = json.loads(ptr_path.read_text(encoding="utf-8"))
        history = list(prev_data.get("history", []))
    if str(current) != str(target):
        history.append(str(current))
    payload = {"active_dir": str(target), "history": history[-5:]}
    atomic_write_lf(ptr_path, json.dumps(payload, indent=2))
    return {"swapped": True, "active_dir": str(target), "previous_dir": str(current), "manifest": manifest.model_dump()}


def rollback_laya_model(cfg: BrainConfig) -> dict[str, Any]:
    ptr_path = active_laya_pointer_path(cfg)
    if not ptr_path.exists():
        raise LayaManifestError("No model swap history exists to roll back.")
    data = json.loads(ptr_path.read_text(encoding="utf-8"))
    history = list(data.get("history", []))
    if not history:
        raise LayaManifestError("No previous Laya checkpoint in rollback history.")
    prev_dir = Path(history.pop())
    manifest = read_laya_manifest(prev_dir, cfg.vault_path)
    payload = {"active_dir": str(prev_dir), "history": history}
    atomic_write_lf(ptr_path, json.dumps(payload, indent=2))
    return {"rolled_back": True, "active_dir": str(prev_dir), "manifest": manifest.model_dump()}


def _render_options(q: dict[str, Any]) -> list[str]:
    qtype = q["t"]
    crit = q.get("crit")
    if qtype == "choice":
        return [k if not v else f"{k}: {v}" for k, v in crit.items()]
    if qtype == "score":
        return [f"level {i}: {c}" for i, c in enumerate(crit)]
    crit = crit or {}
    return [
        "false: " + (crit.get("false") or "no, the statement does not hold"),
        "true: " + (crit.get("true") or "yes, the statement holds"),
    ]


def _temp_bucket(qtype_idx: int, k: int) -> str:
    name = {0: "choice", 1: "score", 2: "noul"}[qtype_idx]
    if name == "noul":
        return "noul:2"
    if name == "score":
        return f"score:{k}" if k > 5 else "score:3-5"
    if k <= 2:
        return "choice:2"
    if k <= 5:
        return "choice:3-5"
    if k <= 10:
        return "choice:6-10"
    return "choice:11+"


class LayaOnnxRuntime:
    """CPU-only ONNX Runtime wrapper implementing convaiinnovations/laya typed decisions."""

    def __init__(
        self,
        model_dir: Path,
        manifest: LayaModelManifest,
        intra_op_num_threads: int = 8,
        inter_op_num_threads: int = 1,
        enable_cpu_mem_arena: bool = False,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.manifest = manifest
        tok_path = self.model_dir / "tokenizer.json"
        if not tok_path.exists():
            raise LayaManifestError(f"Missing tokenizer.json in {self.model_dir}")
        self.tok = Tokenizer.from_file(str(tok_path))
        self.cls_id = self.tok.token_to_id("[CLS]")
        self.sep_id = self.tok.token_to_id("[SEP]")
        self.pad_id = self.tok.token_to_id("[PAD]")
        self.mask_id = self.tok.token_to_id("[MASK]")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_num_threads
        opts.inter_op_num_threads = inter_op_num_threads
        opts.enable_cpu_mem_arena = enable_cpu_mem_arena
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        onnx_path = self.model_dir / manifest.onnx_file
        self.session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.temperatures = manifest.temperature_config.get("temperature", [1.0, 1.0, 1.0])
        self.temp_by_opts = manifest.temperature_config.get("temperature_by_options", {})

    @classmethod
    def from_dir(
        cls,
        model_dir: Path,
        vault_path: Path | None = None,
        intra_op_num_threads: int = 8,
        inter_op_num_threads: int = 1,
        enable_cpu_mem_arena: bool = False,
    ) -> "LayaOnnxRuntime":
        manifest = read_laya_manifest(model_dir, vault_path=vault_path)
        return cls(
            model_dir=model_dir,
            manifest=manifest,
            intra_op_num_threads=intra_op_num_threads,
            inter_op_num_threads=inter_op_num_threads,
            enable_cpu_mem_arena=enable_cpu_mem_arena,
        )

    def _build_sequence(self, state: str, q: dict[str, Any], max_len: int) -> tuple[list[int], list[int]]:
        if max_len > 1024:
            raise ValueError(f"Laya hard context ceiling is 1024 tokens, got {max_len}")
        head_max = self.manifest.head_max_len
        opts = _render_options(q)
        ins = str(q["ins"]).replace("[MASK]", " ")
        head_ids = self.tok.encode(f"{q['t']} question: {ins}", add_special_tokens=False).ids
        opt_ids = [
            [self.mask_id] + self.tok.encode(" " + opt.replace("[MASK]", " "), add_special_tokens=False).ids[:48]
            for opt in opts
        ]
        opt_budget = head_max - sum(len(o) for o in opt_ids)
        if opt_budget < 16:
            per = max(4, (head_max - 16) // max(1, len(opt_ids)))
            opt_ids = [o[:per] for o in opt_ids]
            opt_budget = head_max - sum(len(o) for o in opt_ids)

        head_ids = head_ids[: max(8, opt_budget)]
        ids = [self.cls_id] + head_ids + [self.sep_id]
        markers: list[int] = []
        for o in opt_ids:
            markers.append(len(ids))
            ids.extend(o)
        ids.append(self.sep_id)

        room = max(0, max_len - len(ids) - 1)
        st_ids = self.tok.encode(state.replace("[MASK]", " "), add_special_tokens=False).ids[:room]
        ids = ids + st_ids + [self.sep_id]
        return ids[:max_len], [m for m in markers if m < max_len]

    def run_questions(
        self,
        state: str,
        questions: list[dict[str, Any]],
        cap_tokens: int = 256,
    ) -> list[dict[str, Any]]:
        if cap_tokens > self.manifest.max_len or cap_tokens > 1024:
            raise ValueError(f"Requested cap_tokens={cap_tokens} exceeds Laya ceiling 1024.")
        rows = []
        for q in questions:
            ids, markers = self._build_sequence(state, q, max_len=cap_tokens)
            rows.append((ids, markers, QTYPES[q["t"]], q))

        b = len(rows)
        seq_len = max(len(r[0]) for r in rows)
        k_max = max(len(r[1]) for r in rows)
        input_ids = np.full((b, seq_len), self.pad_id, dtype=np.int64)
        attention_mask = np.zeros((b, seq_len), dtype=np.int64)
        marker_pos = np.zeros((b, k_max), dtype=np.int64)
        marker_mask = np.zeros((b, k_max), dtype=bool)
        qtype = np.array([r[2] for r in rows], dtype=np.int64)

        for i, (r_ids, r_markers, _, _) in enumerate(rows):
            input_ids[i, : len(r_ids)] = r_ids
            attention_mask[i, : len(r_ids)] = 1
            marker_pos[i, : len(r_markers)] = r_markers
            marker_mask[i, : len(r_markers)] = True

        logits, _act = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "marker_pos": marker_pos,
                "marker_mask": marker_mask,
                "qtype": qtype,
            },
        )

        outputs: list[dict[str, Any]] = []
        for idx, (_ids, markers, qt, q) in enumerate(rows):
            k = len(markers)
            temp = float(self.temp_by_opts.get(_temp_bucket(qt, k), self.temperatures[qt]))
            z = logits[idx, :k] / max(temp, 1e-4)
            z = z - np.max(z)
            probs = np.exp(z)
            probs = probs / np.sum(probs)
            ent = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
            conf = 1.0 - (ent / math.log(k)) if k >= 2 else 1.0
            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                prob_map = {kk: round(float(vv), 4) for kk, vv in zip(keys, probs)}
                outputs.append(
                    {
                        "type": "choice",
                        "choice": keys[int(np.argmax(probs))],
                        "probabilities": prob_map,
                        "confidence": round(conf, 4),
                    }
                )
            elif q["t"] == "noul":
                outputs.append(
                    {
                        "type": "noul",
                        "p_true": round(float(probs[1]), 4),
                        "p_false": round(float(probs[0]), 4),
                        "confidence": round(conf, 4),
                    }
                )
            else:
                outputs.append(
                    {
                        "type": "score",
                        "score": round(float(np.sum(np.arange(k) * probs)), 4),
                        "probabilities": [round(float(v), 4) for v in probs],
                        "confidence": round(conf, 4),
                    }
                )
        return outputs

    def predict_choice(self, state: str, instructions: str, criteria: dict[str, str], cap_tokens: int = 256) -> dict[str, Any]:
        return self.run_questions(state, [{"t": "choice", "ins": instructions, "crit": criteria}], cap_tokens=cap_tokens)[0]

    def predict_noul(
        self,
        state: str,
        instructions: str,
        true_desc: str = "yes, statement holds",
        false_desc: str = "no, statement does not hold",
        cap_tokens: int = 1024,
    ) -> dict[str, Any]:
        return self.run_questions(
            state,
            [{"t": "noul", "ins": instructions, "crit": {"true": true_desc, "false": false_desc}}],
            cap_tokens=cap_tokens,
        )[0]
