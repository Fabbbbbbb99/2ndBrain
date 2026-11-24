"""External GPU/Cloud export pipeline: PyTorch DecisionModel -> FP32 ONNX -> MatMulNBits INT8 -> Signed Manifest.

Run this script in the Kaggle/Colab environment (where torch, transformers, and onnxruntime are installed)
after fine-tuning convaiinnovations/laya on your 2ndBrain JSONL splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def compute_ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    if len(probs) == 0:
        return 0.0
    conf = np.max(probs, axis=-1)
    pred = np.argmax(probs, axis=-1)
    correct = (pred == labels).astype(np.float32)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if np.any(sel):
            ece += float(np.mean(sel)) * abs(float(np.mean(conf[sel])) - float(np.mean(correct[sel])))
    return round(float(ece), 4)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Grid-search scalar temperature minimizing negative log-likelihood on calibration split."""
    if len(logits) == 0:
        return 1.0
    best_t = 1.0
    best_nll = float("inf")
    for t in np.linspace(0.5, 3.0, 51):
        scaled = logits / float(t)
        scaled -= np.max(scaled, axis=-1, keepdims=True)
        probs = np.exp(scaled)
        probs /= np.sum(probs, axis=-1, keepdims=True)
        nll = -float(np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0))))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return round(best_t, 4)


def quantize_matmul_nbits_excluding_head(fp32_onnx: Path, int8_onnx: Path) -> None:
    """Apply MatMulNBits 8-bit block-64 quantization while preserving fp32 decision head precision."""
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    model = onnx.load(str(fp32_onnx))
    prod: dict[str, Any] = {}
    for node in model.graph.node:
        for out_name in node.output:
            prod[out_name] = node

    excluded: set[str] = set()
    stack = [o.name for o in model.graph.output]
    seen: set[str] = set()
    while stack:
        tensor_name = stack.pop()
        if tensor_name in seen:
            continue
        seen.add(tensor_name)
        node = prod.get(tensor_name)
        if node is None:
            continue
        if node.op_type in ("MatMul", "Gemm"):
            excluded.add(node.name)
            continue
        stack.extend(node.input)

    quantizer = MatMulNBitsQuantizer(
        model,
        bits=8,
        block_size=64,
        is_symmetric=True,
        accuracy_level=4,
        nodes_to_exclude=sorted(excluded),
    )
    quantizer.process()
    quantizer.model.save_model_to_file(str(int8_onnx), use_external_data_format=False)


def export_and_verify(
    *,
    ckpt_dir: Path,
    calibration_jsonl: Path,
    out_dir: Path,
    choice_labels: list[str],
    max_delta_limit: float = 0.05,
    max_ece_limit: float = 0.08,
) -> dict[str, Any]:
    import onnxruntime as ort
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    # Expects convaiinnovations/laya rl_common on PYTHONPATH
    from rl_common import QTYPES, DecisionModel, build_sequence

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ckpt_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(str(ckpt_dir / "tokenizer"))
    ecfg = AutoConfig.from_pretrained(str(ckpt_dir / "encoder"))
    ecfg._attn_implementation = "eager"
    enc = AutoModel.from_config(ecfg, attn_implementation="eager")
    model = DecisionModel(enc, cfg["head_layers"], len(cfg["act_costs"]) + 1)
    model.load_state_dict(load_file(str(ckpt_dir / "model.safetensors")), strict=False)
    model.eval().float()

    dummy_ids = torch.ones((1, 64), dtype=torch.long)
    dummy_att = torch.ones((1, 64), dtype=torch.long)
    dummy_mpos = torch.tensor([[10, 15]], dtype=torch.long)
    dummy_mmask = torch.tensor([[True, True]], dtype=torch.bool)
    dummy_qt = torch.tensor([0], dtype=torch.long)

    fp32_path = out_dir / "model_fp32.onnx"
    int8_path = out_dir / "model_int8.onnx"
    torch.onnx.export(
        model,
        (dummy_ids, dummy_att, dummy_mpos, dummy_mmask, dummy_qt),
        str(fp32_path),
        input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        output_names=["logits", "act_logits"],
        dynamic_axes={
            "input_ids": {0: "B", 1: "L"},
            "attention_mask": {0: "B", 1: "L"},
            "marker_pos": {0: "B", 1: "K"},
            "marker_mask": {0: "B", 1: "K"},
            "qtype": {0: "B"},
            "logits": {0: "B", 1: "K"},
            "act_logits": {0: "B"},
        },
        opset_version=18,
    )
    quantize_matmul_nbits_excluding_head(fp32_path, int8_path)

    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])

    cal_rows = [json.loads(ln) for ln in calibration_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    logits_fp32_list = []
    logits_int8_list = []
    labels = []
    max_delta = 0.0

    for row in cal_rows[:200]:
        q = row["qs"][0]
        ids, markers = build_sequence(tok, row["state"], q, max_len=min(512, cfg["max_len"]), head_max_len=cfg["head_max_len"])
        k = len(markers)
        feeds = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
            "marker_pos": np.array([markers], dtype=np.int64),
            "marker_mask": np.ones((1, k), dtype=bool),
            "qtype": np.array([QTYPES[q["t"]]], dtype=np.int64),
        }
        z32 = sess_fp32.run(None, feeds)[0][0, :k]
        z8 = sess_int8.run(None, feeds)[0][0, :k]
        p32 = np.exp(z32 - z32.max()); p32 /= p32.sum()
        p8 = np.exp(z8 - z8.max()); p8 /= p8.sum()
        max_delta = max(max_delta, float(np.max(np.abs(p32 - p8))))
        logits_fp32_list.append(z32)
        logits_int8_list.append(z8)
        labels.append(int(q["y"]))

    z_arr = np.asarray(logits_int8_list, dtype=np.float32)
    y_arr = np.asarray(labels, dtype=np.int64)
    temp = fit_temperature(z_arr, y_arr)
    scaled = z_arr / temp
    scaled -= np.max(scaled, axis=-1, keepdims=True)
    probs_cal = np.exp(scaled)
    probs_cal /= np.sum(probs_cal, axis=-1, keepdims=True)
    ece = compute_ece(probs_cal, y_arr)

    if max_delta > max_delta_limit:
        raise RuntimeError(f"INT8 probability drift {max_delta:.4f} exceeded limit {max_delta_limit}")
    if ece > max_ece_limit:
        raise RuntimeError(f"Calibration ECE {ece:.4f} exceeded limit {max_ece_limit}")

    tok.save_pretrained(str(out_dir))
    sha = hashlib.sha256(int8_path.read_bytes()).hexdigest()
    manifest = {
        "model_id": f"custom/laya-2ndbrain-{sha[:8]}",
        "onnx_file": "model_int8.onnx",
        "model_sha256": sha,
        "router_contract_version": "1.0.0",
        "quant_recipe": "matmul_nbits_8bit_block64_fp32_head",
        "choice_labels": choice_labels,
        "max_len": cfg["max_len"],
        "head_max_len": cfg["head_max_len"],
        "temperature_config": {"temperature": [temp, temp, temp], "temperature_by_options": {}},
        "max_prob_delta_vs_fp32": round(max_delta, 4),
        "ece_score": ece,
        "is_purpose_trained": True,
        "allow_laya_auto_commit": True,
    }
    (out_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Export and calibrate fine-tuned Laya to CPU INT8 ONNX")
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--calibration-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--choice-labels", required=True, help="Comma-separated specialist domain_ids")
    args = parser.parse_args()

    manifest = export_and_verify(
        ckpt_dir=Path(args.ckpt_dir),
        calibration_jsonl=Path(args.calibration_jsonl),
        out_dir=Path(args.out_dir),
        choice_labels=[s.strip() for s in args.choice_labels.split(",") if s.strip()],
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
