#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Qwen3 CNL split+train using the local ptx JAX Qwen backend.

This bypasses Hugging Face FlaxAuto, which does not provide Qwen3 causal-LM
classes. It expects the sibling/local ptx repo to provide models/qwen.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax_sft.cnl import add_trees, cnl_optax_step, divide_tree


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ptx_dir", type=str, default="~/ptx")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--weights_dir", type=str, default="~/weights")
    p.add_argument("--source_jsonl", type=str, nargs="+", required=True)
    p.add_argument("--out_correct_jsonl", type=str, required=True)
    p.add_argument("--out_wrong_jsonl", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--max_wrong", type=int, default=None)
    p.add_argument("--max_correct", type=int, default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-7)
    p.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default="sgd")
    p.add_argument("--use_freeze", type=int, choices=[0, 1], default=1)
    p.add_argument("--mask_stage", choices=["gradient", "update"], default="gradient")
    p.add_argument("--tp_size", type=int, default=None)
    p.add_argument("--dp_shard", action="store_true")
    p.add_argument("--eval_before_train", type=int, choices=[0, 1], default=1)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default=None)
    return p.parse_args()


def load_jsonl(paths: list[str], max_rows: int | None) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row.get("question"), row.get("label"))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
                if max_rows is not None and len(rows) >= max_rows:
                    return rows
    return rows


def write_jsonl_line(f, row: dict[str, Any]) -> None:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_csv(path: str, row: dict[str, Any], header: list[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def format_prompt(tokenizer: Any, question: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return question


def ensure_pad_token(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def label_id(tokenizer: Any, label: str) -> int:
    ids = tokenizer(label, add_special_tokens=False)["input_ids"]
    return int(ids[-1])


def candidate_ids(tokenizer: Any) -> np.ndarray:
    return np.asarray([label_id(tokenizer, c) for c in "ABCD"], dtype=np.int32)


def tokenize_row(tokenizer: Any, row: dict[str, Any], max_length: int) -> dict[str, Any] | None:
    prompt = format_prompt(tokenizer, row["question"])
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(ids) >= max_length:
        return None
    pad = max_length - len(ids)
    return {
        "tokens": jnp.asarray([ids + [tokenizer.pad_token_id] * pad], dtype=jnp.int32),
        "attention_mask": jnp.asarray([[1] * len(ids) + [0] * pad], dtype=bool),
        "last_idx": jnp.asarray([len(ids) - 1], dtype=jnp.int32),
        "label_id": jnp.asarray([label_id(tokenizer, row["label"])], dtype=jnp.int32),
        "label": row["label"],
        "question": row["question"],
    }


def array_batch(batch: dict[str, Any]) -> dict[str, jax.Array]:
    return {
        "tokens": batch["tokens"],
        "attention_mask": batch["attention_mask"],
        "last_idx": batch["last_idx"],
        "label_id": batch["label_id"],
    }


def make_optimizer(name: str, lr: float) -> optax.GradientTransformation:
    if name == "sgd":
        return optax.sgd(lr)
    if name == "adam":
        return optax.adam(lr)
    if name == "adamw":
        return optax.adamw(lr)
    raise ValueError(f"Unsupported optimizer: {name}")


def loss_for_batch(forward: Any, weights: Any, batch: dict[str, jax.Array]) -> jax.Array:
    logits = forward(batch["tokens"], weights)
    logits = logits[jnp.arange(logits.shape[0]), batch["last_idx"], :]
    return optax.softmax_cross_entropy_with_integer_labels(logits, batch["label_id"]).mean()


def predict_abcd(predict_logits_fn: Any, weights: Any, batch: dict[str, Any], cand_ids: np.ndarray) -> str:
    logits = np.asarray(predict_logits_fn(weights, array_batch(batch)))
    return "ABCD"[int(np.argmax(logits[cand_ids]))]


def split_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    weights: Any,
    predict_logits_fn: Any,
    cand_ids: np.ndarray,
    max_length: int,
    out_correct: str,
    out_wrong: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    os.makedirs(os.path.dirname(out_correct) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_wrong) or ".", exist_ok=True)
    correct_batches = []
    wrong_batches = []
    skipped = 0
    with open(out_correct, "w", encoding="utf-8") as fc, open(out_wrong, "w", encoding="utf-8") as fw:
        for row in tqdm(rows, desc="split"):
            batch = tokenize_row(tokenizer, row, max_length)
            if batch is None:
                skipped += 1
                continue
            pred = predict_abcd(predict_logits_fn, weights, batch, cand_ids)
            out = {"label": row["label"], "question": row["question"], "predict_label": pred}
            if pred == row["label"]:
                correct_batches.append(batch)
                write_jsonl_line(fc, out)
            else:
                wrong_batches.append(batch)
                write_jsonl_line(fw, out)
    print("========== Split Summary ==========")
    print(f"Total   : {len(rows)}")
    print(f"Correct : {len(correct_batches)}")
    print(f"Wrong   : {len(wrong_batches)}")
    print(f"Skipped : {skipped}")
    return correct_batches, wrong_batches


def mean_grad_correct(weights: Any, correct_batches: list[dict[str, Any]], loss_grad_fn: Any) -> Any:
    if not correct_batches:
        raise ValueError("correct set is empty")
    total_grad = None
    for batch in tqdm(correct_batches, desc="correct mean-grad"):
        _, grad = loss_grad_fn(weights, array_batch(batch))
        total_grad = grad if total_grad is None else add_trees(total_grad, grad)
    return divide_tree(total_grad, float(len(correct_batches)))


def infer_batches(
    weights: Any,
    batches: list[dict[str, Any]],
    predict_logits_fn: Any,
    cand_ids: np.ndarray,
    path: str,
    desc: str,
) -> int:
    ok = 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for batch in tqdm(batches, desc=desc):
            pred = predict_abcd(predict_logits_fn, weights, batch, cand_ids)
            ok += int(pred == batch["label"])
            write_jsonl_line(f, {"label": batch["label"], "question": batch["question"], "predict_label": pred})
    return ok


def maybe_init_wandb(args: argparse.Namespace, n_wrong: int, n_correct: int) -> Any | None:
    if not args.wandb_project:
        return None
    import wandb

    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config={**vars(args), "wrong_rows": n_wrong, "correct_rows": n_correct, "jax_devices": len(jax.devices())},
    )


def main() -> None:
    args = parse_args()
    ptx_dir = Path(args.ptx_dir).expanduser()
    sys.path.insert(0, str(ptx_dir))
    from models import qwen

    tp_size = args.tp_size or jax.device_count()
    print("JAX devices:", jax.devices())
    print("MODEL:", args.model_name)
    print("PTX_DIR:", ptx_dir)
    print("TP_SIZE:", tp_size)

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

    loss_grad_fn = jax.jit(jax.value_and_grad(lambda w, batch: loss_for_batch(forward, w, batch)))
    predict_logits_fn = jax.jit(
        lambda w, batch: forward(batch["tokens"], w)[0, batch["last_idx"][0], :]
    )

    rows = load_jsonl(args.source_jsonl, args.max_rows)
    cand_ids = candidate_ids(model.tokenizer)
    correct_batches, wrong_batches = split_rows(
        rows,
        model.tokenizer,
        weights,
        predict_logits_fn,
        cand_ids,
        args.max_length,
        args.out_correct_jsonl,
        args.out_wrong_jsonl,
    )
    if args.max_correct is not None:
        correct_batches = correct_batches[: args.max_correct]
    if args.max_wrong is not None:
        wrong_batches = wrong_batches[: args.max_wrong]

    optimizer = make_optimizer(args.optimizer, args.lr)
    opt_state = optimizer.init(weights)

    def train_step(w, state, batch, ref_grads):
        loss, grads = loss_grad_fn(w, batch)
        w, state, _ = cnl_optax_step(
            w,
            grads,
            ref_grads,
            optimizer,
            state,
            mask_stage=args.mask_stage,
        )
        return w, state, loss

    def plain_step(w, state, batch, ref_grads):
        del ref_grads
        loss, grads = loss_grad_fn(w, batch)
        updates, state = optimizer.update(grads, state, w)
        w = optax.apply_updates(w, updates)
        return w, state, loss

    train_step_fn = jax.jit(train_step if args.use_freeze else plain_step, donate_argnums=(0, 1))
    jsonl_dir = Path(args.out_dir) / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = str(Path(args.out_dir) / "summary.csv")
    header = ["epoch", "train_avg_loss", "wrong_to_correct", "correct_to_wrong"]
    wandb_run = maybe_init_wandb(args, len(wrong_batches), len(correct_batches))

    if args.eval_before_train:
        print("\n===== Epoch 0 (before training) =====")
        w0 = infer_batches(weights, wrong_batches, predict_logits_fn, cand_ids, str(jsonl_dir / "infer_wrong_ep0.jsonl"), "infer wrong ep0")
        c0 = infer_batches(weights, correct_batches, predict_logits_fn, cand_ids, str(jsonl_dir / "infer_correct_ep0.jsonl"), "infer correct ep0")
        row = {"epoch": 0, "train_avg_loss": "", "wrong_to_correct": w0, "correct_to_wrong": len(correct_batches) - c0}
        append_csv(summary_csv, row, header)
        if wandb_run is not None:
            wandb_run.log(row | {"wrong_accuracy": w0 / len(wrong_batches), "correct_accuracy": c0 / len(correct_batches)}, step=0)

    for ep in range(1, args.epochs + 1):
        print(f"\n===== Epoch {ep} =====")
        ref_grads = mean_grad_correct(weights, correct_batches, loss_grad_fn) if args.use_freeze else None
        loss_sum = 0.0
        for batch in tqdm(wrong_batches, desc=f"train wrong ep{ep}"):
            weights, opt_state, loss = train_step_fn(weights, opt_state, array_batch(batch), ref_grads)
            loss_sum += float(loss)
        train_loss = loss_sum / len(wrong_batches)
        w_ok = infer_batches(weights, wrong_batches, predict_logits_fn, cand_ids, str(jsonl_dir / f"infer_wrong_ep{ep}.jsonl"), f"infer wrong ep{ep}")
        c_ok = infer_batches(weights, correct_batches, predict_logits_fn, cand_ids, str(jsonl_dir / f"infer_correct_ep{ep}.jsonl"), f"infer correct ep{ep}")
        row = {"epoch": ep, "train_avg_loss": train_loss, "wrong_to_correct": w_ok, "correct_to_wrong": len(correct_batches) - c_ok}
        append_csv(summary_csv, row, header)
        if wandb_run is not None:
            wandb_run.log(
                row | {
                    "wrong_accuracy": w_ok / len(wrong_batches),
                    "correct_accuracy": c_ok / len(correct_batches),
                    "forgetting_rate": (len(correct_batches) - c_ok) / len(correct_batches),
                },
                step=ep,
            )

    if wandb_run is not None:
        wandb_run.finish()
    print("Done.")


if __name__ == "__main__":
    main()
