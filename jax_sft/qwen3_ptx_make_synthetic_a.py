#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Make synthetic retention data for the Qwen3 MCQ CNL pipeline.

This creates self-labeled retention rows by running a frozen Qwen3 model over a
bank of multiple-choice prompts and writing the model's own A/B/C/D prediction
as the label. The output JSONL has the same ``question``/``label`` schema used
by the training runners, so it can be passed as ``--retention_jsonl`` to
``qwen3_ptx_ab_train.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax_sft.qwen3_ptx_split_train import (
    candidate_ids,
    ensure_pad_token,
    format_prompt,
    load_jsonl,
    write_jsonl_line,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ptx_dir", type=str, default=None)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--weights_dir", type=str, default="~/weights")
    p.add_argument("--source_jsonl", type=str, nargs="+", required=True)
    p.add_argument("--out_jsonl", type=str, required=True)
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--label_mode", choices=["argmax", "sample"], default="argmax")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--min_confidence", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tp_size", type=int, default=None)
    p.add_argument("--dp_shard", action="store_true")
    return p.parse_args()


def load_backend(args: argparse.Namespace) -> tuple[Any, str]:
    if args.ptx_dir:
        ptx_dir = Path(args.ptx_dir).expanduser()
        sys.path.insert(0, str(ptx_dir))
        from models import qwen

        return qwen, str(ptx_dir)

    from jax_sft.ptx_backend import qwen

    return qwen, "vendored:jax_sft.ptx_backend.qwen"


def tokenize_prompt(tokenizer: Any, question: str, max_length: int) -> dict[str, jax.Array] | None:
    prompt = format_prompt(tokenizer, question)
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not ids or len(ids) >= max_length:
        return None
    pad = max_length - len(ids)
    return {
        "tokens": jnp.asarray([ids + [tokenizer.pad_token_id] * pad], dtype=jnp.int32),
        "last_idx": jnp.asarray([len(ids) - 1], dtype=jnp.int32),
    }


def stable_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def main() -> None:
    args = parse_args()
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")

    qwen, backend = load_backend(args)
    tp_size = args.tp_size or jax.device_count()
    print("JAX devices:", jax.devices())
    print("MODEL:", args.model_name)
    print("QWEN_BACKEND:", backend)
    print("TP_SIZE:", tp_size)
    print("SOURCE_JSONL:", args.source_jsonl)
    print("OUT_JSONL:", args.out_jsonl)

    model = qwen.load(
        args.model_name,
        args.weights_dir,
        tp_size=tp_size,
        dp_shard=args.dp_shard,
        init="pretrained",
    )
    ensure_pad_token(model.tokenizer)
    weights = jax.tree.map(lambda x: x.astype(jnp.float32), model.weights)
    forward = jax.jit(model.forward)
    predict_logits_fn = jax.jit(lambda w, batch: forward(batch["tokens"], w)[0, batch["last_idx"][0], :])
    cand_ids = candidate_ids(model.tokenizer)
    rng = np.random.default_rng(args.seed)

    rows = load_jsonl(args.source_jsonl, args.max_rows)
    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    kept = 0
    skipped = 0
    low_conf = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="synthetic A"):
            question = row.get("question")
            if not question:
                skipped += 1
                continue
            batch = tokenize_prompt(model.tokenizer, question, args.max_length)
            if batch is None:
                skipped += 1
                continue
            logits = np.asarray(predict_logits_fn(weights, batch))[cand_ids]
            probs = stable_softmax(logits / args.temperature)
            if args.label_mode == "sample":
                idx = int(rng.choice(np.arange(4), p=probs))
            else:
                idx = int(np.argmax(probs))
            confidence = float(probs[idx])
            if confidence < args.min_confidence:
                low_conf += 1
                continue
            label = "ABCD"[idx]
            write_jsonl_line(
                f,
                {
                    "label": label,
                    "question": question,
                    "predict_label": label,
                    "synthetic": True,
                    "synthetic_method": "qwen3_pseudo_label",
                    "synthetic_label_mode": args.label_mode,
                    "synthetic_confidence": confidence,
                    "source_label": row.get("label"),
                    "source_predict_label": row.get("predict_label"),
                },
            )
            kept += 1

    print("========== Synthetic A Summary ==========")
    print(f"Total          : {len(rows)}")
    print(f"Kept           : {kept}")
    print(f"Skipped        : {skipped}")
    print(f"Low confidence : {low_conf}")


if __name__ == "__main__":
    main()
