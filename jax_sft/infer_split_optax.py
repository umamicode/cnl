#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Infer A/B/C/D with a Flax causal LM and split JSONL into correct/wrong."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from transformers import FlaxAutoModelForCausalLM
except ImportError:
    try:
        from transformers.models.auto.modeling_flax_auto import FlaxAutoModelForCausalLM
    except ImportError as exc:
        raise ImportError(
            "FlaxAutoModelForCausalLM is unavailable. Install flax and pin "
            "transformers to a 4.x version with both Qwen3 config support and "
            "Flax auto-model classes, for example: "
            "uv pip install 'transformers>=4.51.0,<5'. Transformers 5.x does "
            "not expose the Flax auto-model path used by this runner."
        ) from exc

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax_sft.io import read_jsonl
from jax_sft.sft_optax import array_batch, candidate_ids, ensure_pad_token, tokenize_row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument(
        "--jsonl",
        type=str,
        nargs="+",
        required=True,
        help="One or more JSONL files. Multiple files are concatenated.",
    )
    p.add_argument("--out_wrong_jsonl", type=str, required=True)
    p.add_argument("--out_correct_jsonl", type=str, required=True)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--from_pt", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    return p.parse_args()


def load_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for path in paths:
        for row in read_jsonl(path):
            key = (row.get("question"), row.get("label"))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def predict_abcd(predict_logits_fn: Any, params: Any, batch: dict[str, Any], cand_ids: np.ndarray) -> str:
    logits = np.asarray(predict_logits_fn(params, array_batch(batch)))
    return "ABCD"[int(np.argmax(logits[cand_ids]))]


def main() -> None:
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    ensure_pad_token(tok)
    model = FlaxAutoModelForCausalLM.from_pretrained(
        args.model_name,
        from_pt=args.from_pt,
        trust_remote_code=args.trust_remote_code,
    )
    params = model.params
    cand_ids = candidate_ids(tok)

    rows = load_rows(args.jsonl)
    if args.max_rows is not None:
        rows = rows[:args.max_rows]

    predict_logits_fn = jax.jit(
        lambda p, batch: model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            params=p,
            train=False,
        ).logits[
            0,
            jnp.sum(batch["attention_mask"][0]) - 1,
            :,
        ]
    )

    os.makedirs(os.path.dirname(args.out_wrong_jsonl) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_correct_jsonl) or ".", exist_ok=True)

    correct = 0
    wrong = 0
    with open(args.out_wrong_jsonl, "w", encoding="utf-8") as fw, open(
        args.out_correct_jsonl,
        "w",
        encoding="utf-8",
    ) as fc:
        for row in tqdm(rows, desc="infer & split"):
            batch = tokenize_row(tok, row, args.max_length)
            pred = predict_abcd(predict_logits_fn, params, batch, cand_ids)
            out = {
                "label": row["label"],
                "question": row["question"],
                "predict_label": pred,
            }
            if pred == row["label"]:
                correct += 1
                fc.write(json.dumps(out, ensure_ascii=False) + "\n")
            else:
                wrong += 1
                fw.write(json.dumps(out, ensure_ascii=False) + "\n")

    total = len(rows)
    print("========== Inference Summary ==========")
    print(f"Total    : {total}")
    print(f"Correct  : {correct}")
    print(f"Wrong    : {wrong}")
    print(f"Accuracy : {(correct / total if total else 0.0):.4f}")


if __name__ == "__main__":
    main()
