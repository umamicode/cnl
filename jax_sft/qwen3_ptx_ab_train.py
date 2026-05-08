#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Qwen3 CNL training for explicit A/B retention-vs-injection experiments.

This runner keeps the same multiple-choice next-token objective as
``qwen3_ptx_split_train.py`` but replaces the implicit correct/wrong split with
explicit data roles:

    A = retention/mastered data
    B = injection/new data

Use ``--retention_filter correct`` to make A match the CNL "mastered set"
definition, and ``--train_filter wrong`` to make B match the paper's injection
set definition. Leave filters as ``none`` for a standard A-then-B finetuning
setting.
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
import optax
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax_sft.cnl import add_trees, cnl_optax_step, divide_tree
from jax_sft.qwen3_ptx_split_train import (
    append_csv,
    array_batch,
    candidate_ids,
    ensure_pad_token,
    infer_batches,
    load_jsonl,
    loss_for_batch,
    make_optimizer,
    predict_abcd,
    save_final_summary,
    tokenize_row,
    update_wandb_summary,
    write_jsonl_line,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ptx_dir", type=str, default=None)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--weights_dir", type=str, default="~/weights")

    p.add_argument("--retention_jsonl", type=str, nargs="+", required=True)
    p.add_argument("--train_jsonl", type=str, nargs="+", required=True)
    p.add_argument("--retention_eval_jsonl", type=str, nargs="+", default=None)
    p.add_argument("--train_eval_jsonl", type=str, nargs="+", default=None)
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--retention_filter", choices=["none", "correct", "wrong"], default="correct")
    p.add_argument("--train_filter", choices=["none", "correct", "wrong"], default="none")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--max_retention", type=int, default=None)
    p.add_argument("--max_train", type=int, default=None)
    p.add_argument("--max_retention_eval", type=int, default=None)
    p.add_argument("--max_train_eval", type=int, default=None)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-7)
    p.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default="sgd")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--use_freeze", type=int, choices=[0, 1], default=1)
    p.add_argument("--mask_stage", choices=["gradient", "update"], default="gradient")
    p.add_argument(
        "--ref_refresh_steps",
        type=int,
        default=0,
        help="0 refreshes the A reference gradient once per epoch; N refreshes every N B updates.",
    )

    p.add_argument("--tp_size", type=int, default=None)
    p.add_argument("--dp_shard", action="store_true")
    p.add_argument("--eval_before_train", type=int, choices=[0, 1], default=1)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default=None)
    return p.parse_args()


def maybe_init_wandb(args: argparse.Namespace, n_a: int, n_b: int, n_a_eval: int, n_b_eval: int) -> Any | None:
    if not args.wandb_project:
        return None
    import wandb

    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config={
            **vars(args),
            "retention_rows": n_a,
            "train_rows": n_b,
            "retention_eval_rows": n_a_eval,
            "train_eval_rows": n_b_eval,
            "jax_devices": len(jax.devices()),
        },
    )


def load_backend(args: argparse.Namespace) -> tuple[Any, str]:
    if args.ptx_dir:
        ptx_dir = Path(args.ptx_dir).expanduser()
        sys.path.insert(0, str(ptx_dir))
        from models import qwen

        return qwen, str(ptx_dir)

    from jax_sft.ptx_backend import qwen

    return qwen, "vendored:jax_sft.ptx_backend.qwen"


def filter_batches(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    weights: Any,
    predict_logits_fn: Any,
    cand_ids: Any,
    max_length: int,
    filter_mode: str,
    max_batches: int | None,
    out_jsonl: str,
    desc: str,
) -> list[dict[str, Any]]:
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    batches = []
    skipped = 0
    filtered = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in tqdm(rows, desc=desc):
            batch = tokenize_row(tokenizer, row, max_length)
            if batch is None:
                skipped += 1
                continue
            pred = predict_abcd(predict_logits_fn, weights, batch, cand_ids)
            keep = (
                filter_mode == "none"
                or (filter_mode == "correct" and pred == row["label"])
                or (filter_mode == "wrong" and pred != row["label"])
            )
            if not keep:
                filtered += 1
                continue
            out = {
                "label": row["label"],
                "question": row["question"],
                "predict_label": pred,
                "source_predict_label": row.get("predict_label"),
            }
            batches.append(batch)
            write_jsonl_line(f, out)
            if max_batches is not None and len(batches) >= max_batches:
                break
    print(f"{desc}: kept={len(batches)} filtered={filtered} skipped={skipped}")
    return batches


def tokenize_batches(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    max_batches: int | None,
    desc: str,
) -> list[dict[str, Any]]:
    batches = []
    skipped = 0
    for row in tqdm(rows, desc=desc):
        batch = tokenize_row(tokenizer, row, max_length)
        if batch is None:
            skipped += 1
            continue
        batches.append(batch)
        if max_batches is not None and len(batches) >= max_batches:
            break
    print(f"{desc}: kept={len(batches)} skipped={skipped}")
    return batches


def mean_grad_retention(weights: Any, retention_batches: list[dict[str, Any]], loss_grad_fn: Any) -> Any:
    if not retention_batches:
        raise ValueError("retention set is empty")
    total_grad = None
    for batch in tqdm(retention_batches, desc="retention mean-grad"):
        _, grad = loss_grad_fn(weights, array_batch(batch))
        total_grad = grad if total_grad is None else add_trees(total_grad, grad)
    return divide_tree(total_grad, float(len(retention_batches)))


def eval_loss_batches(weights: Any, batches: list[dict[str, Any]], loss_fn: Any, desc: str) -> float:
    if not batches:
        return 0.0
    total = 0.0
    for batch in tqdm(batches, desc=desc):
        total += float(loss_fn(weights, array_batch(batch)))
    return total / len(batches)


def build_metrics(
    epoch: int,
    train_loss: float | str,
    a_ok: int,
    b_ok: int,
    a_loss: float,
    b_loss: float,
    n_a: int,
    n_b: int,
    base_a_acc: float | None,
    base_b_acc: float | None,
    base_a_loss: float | None,
    base_b_loss: float | None,
) -> dict[str, Any]:
    a_accuracy = a_ok / n_a if n_a else 0.0
    b_accuracy = b_ok / n_b if n_b else 0.0
    a_drop = (base_a_acc - a_accuracy) if base_a_acc is not None else ""
    b_gain = (b_accuracy - base_b_acc) if base_b_acc is not None else ""
    a_loss_delta = (a_loss - base_a_loss) if base_a_loss is not None else ""
    b_loss_delta = (b_loss - base_b_loss) if base_b_loss is not None else ""
    tradeoff_score = (b_gain - max(a_drop, 0.0)) if isinstance(a_drop, float) and isinstance(b_gain, float) else ""
    return {
        "epoch": epoch,
        "train_avg_loss": train_loss,
        "a_correct": a_ok,
        "a_total": n_a,
        "a_accuracy": a_accuracy,
        "a_loss": a_loss,
        "a_drop": a_drop,
        "a_loss_delta": a_loss_delta,
        "b_correct": b_ok,
        "b_total": n_b,
        "b_accuracy": b_accuracy,
        "b_loss": b_loss,
        "b_gain": b_gain,
        "b_loss_delta": b_loss_delta,
        "retention_score": a_accuracy,
        "learning_score": b_accuracy,
        "tradeoff_score": tradeoff_score,
    }


def main() -> None:
    args = parse_args()
    qwen, backend = load_backend(args)
    tp_size = args.tp_size or jax.device_count()

    print("JAX devices:", jax.devices())
    print("MODEL:", args.model_name)
    print("QWEN_BACKEND:", backend)
    print("TP_SIZE:", tp_size)
    print("RETENTION_JSONL:", args.retention_jsonl)
    print("TRAIN_JSONL:", args.train_jsonl)
    print("RETENTION_FILTER:", args.retention_filter)
    print("TRAIN_FILTER:", args.train_filter)

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

    loss_fn = jax.jit(lambda w, batch: loss_for_batch(forward, w, batch))
    loss_grad_fn = jax.jit(jax.value_and_grad(lambda w, batch: loss_for_batch(forward, w, batch)))
    predict_logits_fn = jax.jit(lambda w, batch: forward(batch["tokens"], w)[0, batch["last_idx"][0], :])
    cand_ids = candidate_ids(model.tokenizer)

    jsonl_dir = Path(args.out_dir) / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)

    retention_rows = load_jsonl(args.retention_jsonl, None)
    train_rows = load_jsonl(args.train_jsonl, None)
    retention_eval_rows = load_jsonl(args.retention_eval_jsonl or args.retention_jsonl, None)
    train_eval_rows = load_jsonl(args.train_eval_jsonl or args.train_jsonl, None)

    retention_batches = filter_batches(
        retention_rows,
        model.tokenizer,
        weights,
        predict_logits_fn,
        cand_ids,
        args.max_length,
        args.retention_filter,
        args.max_retention,
        str(jsonl_dir / "retention_train_filtered.jsonl"),
        "filter retention train",
    )
    train_batches = filter_batches(
        train_rows,
        model.tokenizer,
        weights,
        predict_logits_fn,
        cand_ids,
        args.max_length,
        args.train_filter,
        args.max_train,
        str(jsonl_dir / "train_filtered.jsonl"),
        "filter train",
    )
    retention_eval_batches = tokenize_batches(
        retention_eval_rows,
        model.tokenizer,
        args.max_length,
        args.max_retention_eval,
        "tokenize retention eval",
    )
    train_eval_batches = tokenize_batches(
        train_eval_rows,
        model.tokenizer,
        args.max_length,
        args.max_train_eval,
        "tokenize train eval",
    )

    if args.use_freeze and not retention_batches:
        raise ValueError("CNL requested but no retention batches survived filtering")
    if not train_batches:
        raise ValueError("train set is empty")

    optimizer = make_optimizer(args.optimizer, args.lr, weight_decay=args.weight_decay)
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
    summary_csv = str(Path(args.out_dir) / "summary.csv")
    header = [
        "epoch",
        "train_avg_loss",
        "a_correct",
        "a_total",
        "a_accuracy",
        "a_loss",
        "a_drop",
        "a_loss_delta",
        "b_correct",
        "b_total",
        "b_accuracy",
        "b_loss",
        "b_gain",
        "b_loss_delta",
        "retention_score",
        "learning_score",
        "tradeoff_score",
    ]

    wandb_run = maybe_init_wandb(
        args,
        len(retention_batches),
        len(train_batches),
        len(retention_eval_batches),
        len(train_eval_batches),
    )
    base_a_acc = None
    base_b_acc = None
    base_a_loss = None
    base_b_loss = None
    final_metrics = None

    def evaluate(epoch: int, train_loss: float | str) -> dict[str, Any]:
        a_ok = infer_batches(
            weights,
            retention_eval_batches,
            predict_logits_fn,
            cand_ids,
            str(jsonl_dir / f"infer_retention_ep{epoch}.jsonl"),
            f"infer retention ep{epoch}",
        )
        b_ok = infer_batches(
            weights,
            train_eval_batches,
            predict_logits_fn,
            cand_ids,
            str(jsonl_dir / f"infer_train_ep{epoch}.jsonl"),
            f"infer train ep{epoch}",
        )
        a_loss = eval_loss_batches(weights, retention_eval_batches, loss_fn, f"loss retention ep{epoch}")
        b_loss = eval_loss_batches(weights, train_eval_batches, loss_fn, f"loss train ep{epoch}")
        return build_metrics(
            epoch,
            train_loss,
            a_ok,
            b_ok,
            a_loss,
            b_loss,
            len(retention_eval_batches),
            len(train_eval_batches),
            base_a_acc,
            base_b_acc,
            base_a_loss,
            base_b_loss,
        )

    if args.eval_before_train:
        print("\n===== Epoch 0 (before training) =====")
        metrics = evaluate(0, "")
        base_a_acc = float(metrics["a_accuracy"])
        base_b_acc = float(metrics["b_accuracy"])
        base_a_loss = float(metrics["a_loss"])
        base_b_loss = float(metrics["b_loss"])
        metrics = build_metrics(
            0,
            "",
            int(metrics["a_correct"]),
            int(metrics["b_correct"]),
            base_a_loss,
            base_b_loss,
            len(retention_eval_batches),
            len(train_eval_batches),
            base_a_acc,
            base_b_acc,
            base_a_loss,
            base_b_loss,
        )
        append_csv(summary_csv, metrics, header)
        if wandb_run is not None:
            wandb_run.log(metrics, step=0)

    global_step = 0
    for ep in range(1, args.epochs + 1):
        print(f"\n===== Epoch {ep} =====")
        ref_grads = mean_grad_retention(weights, retention_batches, loss_grad_fn) if args.use_freeze else None
        loss_sum = 0.0
        for batch in tqdm(train_batches, desc=f"train B ep{ep}"):
            if args.use_freeze and args.ref_refresh_steps > 0 and global_step > 0 and global_step % args.ref_refresh_steps == 0:
                ref_grads = mean_grad_retention(weights, retention_batches, loss_grad_fn)
            weights, opt_state, loss = train_step_fn(weights, opt_state, array_batch(batch), ref_grads)
            loss_sum += float(loss)
            global_step += 1
        train_loss = loss_sum / len(train_batches)
        metrics = evaluate(ep, train_loss)
        append_csv(summary_csv, metrics, header)
        if wandb_run is not None:
            wandb_run.log(metrics, step=ep)
        final_metrics = metrics

    if wandb_run is not None:
        if final_metrics is not None:
            update_wandb_summary(wandb_run, final_metrics)
            save_final_summary(args.out_dir, final_metrics)
        wandb_run.finish()
    elif final_metrics is not None:
        save_final_summary(args.out_dir, final_metrics)
    print("Done.")


if __name__ == "__main__":
    main()
