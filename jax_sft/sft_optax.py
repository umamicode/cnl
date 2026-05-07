#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JAX/Optax SFT runner for Collaborative Neuron Learning.

This script mirrors ``sft/sft.py`` but uses Flax models and Optax optimizers.
It is intentionally simple: one JSONL sample at a time, next-token loss on the
answer letter, and per-sample CNL masking.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from transformers import FlaxAutoModelForCausalLM
except ImportError:
    try:
        from transformers.models.auto.modeling_flax_auto import FlaxAutoModelForCausalLM
    except ImportError as exc:
        raise ImportError(
            "FlaxAutoModelForCausalLM is unavailable. Install Flax in this "
            "environment, for example: uv pip install flax. If you pass --from_pt "
            "to convert PyTorch weights, install torch as well. If Flax is "
            "already installed, pin transformers to a version that includes "
            "Flax model classes, such as transformers==4.48.0."
        ) from exc

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax_sft.cnl import MaskStage, add_trees, cnl_optax_step, divide_tree
from jax_sft.io import append_csv, read_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument(
        "--wrong_jsonl",
        type=str,
        default="./data/csqa_wrong_Qwen2.5-1.5B-Instruct.jsonl",
    )
    p.add_argument(
        "--correct_jsonl",
        type=str,
        default="./data/csqa_correct_Qwen2.5-1.5B-Instruct.jsonl",
    )
    p.add_argument("--out_dir", type=str, default="./jax_ckpts/csqa_optax")
    p.add_argument("--lr", type=float, default=1e-7)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--use_freeze", type=int, choices=[0, 1], default=1)
    p.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default="sgd")
    p.add_argument(
        "--mask_stage",
        choices=["gradient", "update"],
        default="gradient",
        help="gradient matches sft.py; update matches the Adam/momentum variants.",
    )
    p.add_argument(
        "--from_pt",
        action="store_true",
        help="Ask transformers to convert PyTorch checkpoints when Flax weights are unavailable.",
    )
    p.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to Hugging Face loaders.",
    )
    return p.parse_args()


def make_optimizer(name: str, lr: float) -> optax.GradientTransformation:
    if name == "sgd":
        return optax.sgd(lr)
    if name == "adam":
        return optax.adam(lr)
    if name == "adamw":
        return optax.adamw(lr)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_inputs(tok: AutoTokenizer, question: str) -> dict[str, jax.Array]:
    if getattr(tok, "chat_template", None):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = question

    batch = tok(prompt, return_tensors="np")
    return {k: jnp.asarray(v) for k, v in batch.items()}


def label_token_id(tok: AutoTokenizer, label: str) -> jax.Array:
    ids = tok(label, add_special_tokens=False, return_tensors="np").input_ids
    return jnp.asarray([int(ids[0, -1])])


def candidate_ids(tok: AutoTokenizer) -> np.ndarray:
    return np.asarray(
        [
            int(tok(c, add_special_tokens=False, return_tensors="np").input_ids[0, -1])
            for c in "ABCD"
        ],
        dtype=np.int32,
    )


def loss_for_row(model: Any, tok: AutoTokenizer, params: Any, row: dict[str, Any]) -> jax.Array:
    inputs = build_inputs(tok, row["question"])
    labels = label_token_id(tok, row["label"])
    outputs = model(**inputs, params=params, train=True)
    logits = outputs.logits[:, -1, :]
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()


def mean_grad_correct(
    model: Any,
    tok: AutoTokenizer,
    params: Any,
    rows: list[dict[str, Any]],
) -> Any:
    if not rows:
        raise ValueError("correct rows are empty")

    total_grad = None
    loss_grad_fn = jax.value_and_grad(lambda p, row: loss_for_row(model, tok, p, row))

    for row in tqdm(rows, desc="correct mean-grad"):
        _, grad = loss_grad_fn(params, row)
        total_grad = grad if total_grad is None else add_trees(total_grad, grad)

    return divide_tree(total_grad, float(len(rows)))


def train_wrong_epoch(
    model: Any,
    tok: AutoTokenizer,
    params: Any,
    rows: list[dict[str, Any]],
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    reference_grads: Any | None,
    mask_stage: MaskStage,
    desc: str,
) -> tuple[Any, optax.OptState, float]:
    if not rows:
        raise ValueError("wrong rows are empty")

    loss_sum = 0.0
    loss_grad_fn = jax.value_and_grad(lambda p, row: loss_for_row(model, tok, p, row))

    for row in tqdm(rows, desc=desc):
        loss, grads = loss_grad_fn(params, row)
        params, opt_state, _ = cnl_optax_step(
            params,
            grads,
            reference_grads,
            optimizer,
            opt_state,
            mask_stage=mask_stage,
        )
        loss_sum += float(loss)

    return params, opt_state, loss_sum / len(rows)


def predict_abcd(model: Any, tok: AutoTokenizer, params: Any, row: dict[str, Any], cand_ids: np.ndarray) -> str:
    inputs = build_inputs(tok, row["question"])
    outputs = model(**inputs, params=params, train=False)
    logits = np.asarray(outputs.logits[:, -1, :][0])
    return "ABCD"[int(np.argmax(logits[cand_ids]))]


def infer_and_dump(
    model: Any,
    tok: AutoTokenizer,
    params: Any,
    rows: list[dict[str, Any]],
    cand_ids: np.ndarray,
    path: str,
    desc: str,
) -> int:
    ok = 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in tqdm(rows, desc=desc):
            pred = predict_abcd(model, tok, params, row, cand_ids)
            ok += int(pred == row["label"])
            f.write(json.dumps({
                "label": row["label"],
                "predict_label": pred,
                "question": row["question"],
            }, ensure_ascii=False) + "\n")
    return ok


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    jsonl_dir = os.path.join(args.out_dir, "jsonl")
    os.makedirs(jsonl_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    model = FlaxAutoModelForCausalLM.from_pretrained(
        args.model_name,
        from_pt=args.from_pt,
        trust_remote_code=args.trust_remote_code,
    )
    params = model.params

    wrong_rows = read_jsonl(args.wrong_jsonl)
    correct_rows = read_jsonl(args.correct_jsonl)
    cand_ids = candidate_ids(tok)

    optimizer = make_optimizer(args.optimizer, args.lr)
    opt_state = optimizer.init(params)

    header = ["epoch", "train_avg_loss", "wrong_to_correct", "correct_to_wrong"]
    summary_csv = os.path.join(args.out_dir, "summary.csv")

    print("JAX devices:", jax.devices())
    print("MODEL:", args.model_name)
    print("WRONG_JSONL:", args.wrong_jsonl)
    print("CORRECT_JSONL:", args.correct_jsonl)
    print("OUT_DIR:", args.out_dir)
    print("USE_FREEZE:", bool(args.use_freeze))
    print("OPTIMIZER:", args.optimizer)
    print("MASK_STAGE:", args.mask_stage)

    for ep in range(1, args.epochs + 1):
        print(f"\n===== Epoch {ep} =====")

        reference_grads = None
        if args.use_freeze:
            reference_grads = mean_grad_correct(model, tok, params, correct_rows)

        params, opt_state, train_loss = train_wrong_epoch(
            model,
            tok,
            params,
            wrong_rows,
            optimizer,
            opt_state,
            reference_grads,
            args.mask_stage,
            desc=f"train wrong ep{ep}",
        )

        w_ok = infer_and_dump(
            model,
            tok,
            params,
            wrong_rows,
            cand_ids,
            os.path.join(jsonl_dir, f"infer_wrong_ep{ep}.jsonl"),
            f"infer wrong ep{ep}",
        )
        c_ok = infer_and_dump(
            model,
            tok,
            params,
            correct_rows,
            cand_ids,
            os.path.join(jsonl_dir, f"infer_correct_ep{ep}.jsonl"),
            f"infer correct ep{ep}",
        )

        append_csv(
            summary_csv,
            {
                "epoch": ep,
                "train_avg_loss": train_loss,
                "wrong_to_correct": w_ok,
                "correct_to_wrong": len(correct_rows) - c_ok,
            },
            header,
        )

    model.save_pretrained(os.path.join(args.out_dir, "model"), params=params)
    tok.save_pretrained(os.path.join(args.out_dir, "model"))
    print("Done.")


if __name__ == "__main__":
    main()
