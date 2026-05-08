#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Two-stage Qwen3 continual-learning runner.

Practical setting:

1. Train the base model on A with normal SFT.
2. Treat the resulting model as the "old-task" model.
3. Train on B with either CNL or plain SFT.
4. Measure A drop relative to after-A, and B gain relative to before-B.

The B-stage CNL reference data defaults to A train data, but can be replaced
with synthetic A. With ``--synthetic_b_retention 1``, synthetic retention rows
are generated from the A-trained model before B training.
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

from jax_sft.cnl import cnl_optax_step
from jax_sft.qwen3_ptx_ab_train import (
    eval_loss_batches,
    filter_batches,
    load_backend,
    mean_grad_retention,
    tokenize_batches,
)
from jax_sft.qwen3_ptx_split_train import (
    append_csv,
    array_batch,
    candidate_ids,
    configure_wandb_mode,
    ensure_pad_token,
    infer_batches,
    load_jsonl,
    loss_for_batch,
    make_optimizer,
    save_final_summary,
    update_wandb_summary,
    write_jsonl_line,
    format_prompt,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ptx_dir", type=str, default=None)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--weights_dir", type=str, default="~/weights")

    p.add_argument("--a_train_jsonl", type=str, nargs="+", required=True)
    p.add_argument("--b_train_jsonl", type=str, nargs="+", required=True)
    p.add_argument("--a_eval_jsonl", type=str, nargs="+", default=None)
    p.add_argument("--b_eval_jsonl", type=str, nargs="+", default=None)
    p.add_argument(
        "--b_retention_jsonl",
        type=str,
        nargs="+",
        default=None,
        help="Data used for the B-stage CNL reference gradient. Defaults to A train data.",
    )
    p.add_argument("--synthetic_b_retention", type=int, choices=[0, 1], default=0)
    p.add_argument(
        "--synthetic_source_jsonl",
        type=str,
        nargs="+",
        default=None,
        help="Prompt bank for synthetic B-stage retention. Defaults to b_retention_jsonl or A train data.",
    )
    p.add_argument("--synthetic_max_rows", type=int, default=None)
    p.add_argument("--synthetic_label_mode", choices=["argmax", "sample"], default="argmax")
    p.add_argument("--synthetic_temperature", type=float, default=1.0)
    p.add_argument("--synthetic_min_confidence", type=float, default=0.0)
    p.add_argument("--synthetic_seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--max_a_train", type=int, default=None)
    p.add_argument("--max_b_train", type=int, default=None)
    p.add_argument("--max_a_eval", type=int, default=None)
    p.add_argument("--max_b_eval", type=int, default=None)
    p.add_argument("--max_b_retention", type=int, default=None)
    p.add_argument(
        "--b_retention_ratio",
        type=float,
        default=1.0,
        help="Random fraction of B-stage A-retention rows to keep. Values >1 are interpreted as percentages.",
    )
    p.add_argument("--b_retention_seed", type=int, default=0)

    p.add_argument("--a_epochs", type=int, default=1)
    p.add_argument("--a_lr", type=float, default=1e-7)
    p.add_argument("--a_optimizer", choices=["sgd", "adam", "adamw"], default="adamw")
    p.add_argument("--a_weight_decay", type=float, default=1e-4)

    p.add_argument("--b_epochs", type=int, default=1)
    p.add_argument("--b_lr", type=float, default=1e-7)
    p.add_argument("--b_optimizer", choices=["sgd", "adam", "adamw"], default="adamw")
    p.add_argument("--b_weight_decay", type=float, default=1e-4)
    p.add_argument("--b_method", choices=["cnl", "sft"], default="cnl")
    p.add_argument("--mask_stage", choices=["gradient", "update"], default="update")
    p.add_argument(
        "--b_retention_filter",
        choices=["none", "correct", "wrong"],
        default="none",
        help="Filter B-stage retention/reference rows after A training.",
    )
    p.add_argument(
        "--b_train_filter",
        choices=["none", "correct", "wrong"],
        default="none",
        help="Filter B train rows after A training; use wrong for injection-only B training.",
    )
    p.add_argument(
        "--ref_refresh_steps",
        type=int,
        default=0,
        help="0 refreshes the B-stage retention gradient once per epoch; N refreshes every N B updates.",
    )

    p.add_argument("--tp_size", type=int, default=None)
    p.add_argument("--dp_shard", action="store_true")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default=None)
    return p.parse_args()


def maybe_init_wandb(args: argparse.Namespace, counts: dict[str, int]) -> Any | None:
    if not args.wandb_project:
        return None
    import wandb

    configure_wandb_mode(args.wandb_mode)
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config={
            **vars(args),
            **counts,
            "method": "cnl_synth"
            if args.b_method == "cnl" and args.synthetic_b_retention
            else args.b_method,
            "jax_devices": len(jax.devices()),
        },
    )


def write_stage_json(path: str, metrics: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")


def normalize_ratio(ratio: float) -> float:
    ratio = ratio / 100.0 if ratio > 1 else ratio
    if ratio <= 0 or ratio > 1:
        raise ValueError(f"ratio must be in (0, 1] or (0, 100], got {ratio}")
    return ratio


def random_subset_rows(rows: list[dict[str, Any]], ratio: float, seed: int, desc: str) -> list[dict[str, Any]]:
    ratio = normalize_ratio(ratio)
    if ratio >= 1 or not rows:
        print(f"{desc}: ratio=1.0 kept={len(rows)}/{len(rows)}")
        return rows
    n_keep = max(1, int(round(len(rows) * ratio)))
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(rows), size=n_keep, replace=False).tolist())
    print(f"{desc}: ratio={ratio:.4f} seed={seed} kept={n_keep}/{len(rows)}")
    return [rows[i] for i in indices]


def stable_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


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


def make_synthetic_retention_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    weights: Any,
    predict_logits_fn: Any,
    cand_ids: Any,
    *,
    max_length: int,
    max_rows: int | None,
    label_mode: str,
    temperature: float,
    min_confidence: float,
    seed: int,
    out_jsonl: str,
) -> list[dict[str, Any]]:
    if temperature <= 0:
        raise ValueError("--synthetic_temperature must be positive")
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    synthetic_rows = []
    skipped = 0
    low_conf = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="make synthetic retention"):
            question = row.get("question")
            if not question:
                skipped += 1
                continue
            batch = tokenize_prompt(tokenizer, question, max_length)
            if batch is None:
                skipped += 1
                continue
            logits = np.asarray(predict_logits_fn(weights, batch))[cand_ids]
            probs = stable_softmax(logits / temperature)
            if label_mode == "sample":
                idx = int(rng.choice(np.arange(4), p=probs))
            else:
                idx = int(np.argmax(probs))
            confidence = float(probs[idx])
            if confidence < min_confidence:
                low_conf += 1
                continue
            label = "ABCD"[idx]
            out = {
                "label": label,
                "question": question,
                "predict_label": label,
                "synthetic": True,
                "synthetic_method": "a_trained_qwen3_pseudo_label",
                "synthetic_label_mode": label_mode,
                "synthetic_confidence": confidence,
                "source_label": row.get("label"),
                "source_predict_label": row.get("predict_label"),
            }
            synthetic_rows.append(out)
            write_jsonl_line(f, out)
            if max_rows is not None and len(synthetic_rows) >= max_rows:
                break
    print("========== Synthetic B-Retention Summary ==========")
    print(f"Source rows    : {len(rows)}")
    print(f"Kept           : {len(synthetic_rows)}")
    print(f"Skipped        : {skipped}")
    print(f"Low confidence : {low_conf}")
    return synthetic_rows


def build_metrics(
    *,
    stage: str,
    global_step: int,
    a_epoch: int,
    b_epoch: int,
    train_avg_loss: float | str,
    a_ok: int,
    b_ok: int,
    a_loss: float,
    b_loss: float,
    n_a_eval: int,
    n_b_eval: int,
    a_after_a_acc: float | None,
    b_before_b_acc: float | None,
    a_after_a_loss: float | None,
    b_before_b_loss: float | None,
) -> dict[str, Any]:
    a_accuracy = a_ok / n_a_eval if n_a_eval else 0.0
    b_accuracy = b_ok / n_b_eval if n_b_eval else 0.0
    a_drop = (a_after_a_acc - a_accuracy) if a_after_a_acc is not None else ""
    b_gain = (b_accuracy - b_before_b_acc) if b_before_b_acc is not None else ""
    a_loss_delta = (a_loss - a_after_a_loss) if a_after_a_loss is not None else ""
    b_loss_delta = (b_loss - b_before_b_loss) if b_before_b_loss is not None else ""
    tradeoff_score = (b_gain - max(a_drop, 0.0)) if isinstance(a_drop, float) and isinstance(b_gain, float) else ""
    return {
        "stage": stage,
        "global_step": global_step,
        "a_epoch": a_epoch,
        "b_epoch": b_epoch,
        "train_avg_loss": train_avg_loss,
        "a_correct": a_ok,
        "a_total": n_a_eval,
        "a_accuracy": a_accuracy,
        "a_loss": a_loss,
        "a_after_a_accuracy": a_after_a_acc if a_after_a_acc is not None else "",
        "a_drop_from_after_a": a_drop,
        "a_loss_delta_from_after_a": a_loss_delta,
        "b_correct": b_ok,
        "b_total": n_b_eval,
        "b_accuracy": b_accuracy,
        "b_loss": b_loss,
        "b_before_b_accuracy": b_before_b_acc if b_before_b_acc is not None else "",
        "b_gain_from_before_b": b_gain,
        "b_loss_delta_from_before_b": b_loss_delta,
        "retention_score": a_accuracy,
        "learning_score": b_accuracy,
        "tradeoff_score": tradeoff_score,
    }


def main() -> None:
    args = parse_args()
    qwen, backend = load_backend(args)
    tp_size = args.tp_size or jax.device_count()
    b_retention_paths = args.b_retention_jsonl or args.a_train_jsonl
    synthetic_source_paths = args.synthetic_source_jsonl or b_retention_paths

    print("JAX devices:", jax.devices())
    print("MODEL:", args.model_name)
    print("QWEN_BACKEND:", backend)
    print("TP_SIZE:", tp_size)
    print("A_TRAIN_JSONL:", args.a_train_jsonl)
    print("B_TRAIN_JSONL:", args.b_train_jsonl)
    print("B_RETENTION_JSONL:", b_retention_paths)
    print("SYNTHETIC_B_RETENTION:", bool(args.synthetic_b_retention))
    if args.synthetic_b_retention:
        print("SYNTHETIC_SOURCE_JSONL:", synthetic_source_paths)
    print("B_METHOD:", args.b_method)

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
    summary_csv = str(Path(args.out_dir) / "summary.csv")
    header = [
        "stage",
        "method",
        "b_method",
        "synthetic_b_retention",
        "b_retention_ratio",
        "b_retention_seed",
        "a_lr",
        "b_lr",
        "a_optimizer",
        "b_optimizer",
        "mask_stage",
        "global_step",
        "a_epoch",
        "b_epoch",
        "train_avg_loss",
        "a_correct",
        "a_total",
        "a_accuracy",
        "a_loss",
        "a_after_a_accuracy",
        "a_drop_from_after_a",
        "a_loss_delta_from_after_a",
        "b_correct",
        "b_total",
        "b_accuracy",
        "b_loss",
        "b_before_b_accuracy",
        "b_gain_from_before_b",
        "b_loss_delta_from_before_b",
        "retention_score",
        "learning_score",
        "tradeoff_score",
    ]

    a_train_rows = load_jsonl(args.a_train_jsonl, None)
    b_train_rows = load_jsonl(args.b_train_jsonl, None)
    a_eval_rows = load_jsonl(args.a_eval_jsonl or args.a_train_jsonl, None)
    b_eval_rows = load_jsonl(args.b_eval_jsonl or args.b_train_jsonl, None)
    b_retention_rows = [] if args.synthetic_b_retention else load_jsonl(b_retention_paths, None)
    synthetic_source_rows = load_jsonl(synthetic_source_paths, None) if args.synthetic_b_retention else []
    b_retention_ratio = normalize_ratio(args.b_retention_ratio)

    a_train_batches = tokenize_batches(a_train_rows, model.tokenizer, args.max_length, args.max_a_train, "tokenize A train")
    a_eval_batches = tokenize_batches(a_eval_rows, model.tokenizer, args.max_length, args.max_a_eval, "tokenize A eval")
    b_eval_batches = tokenize_batches(b_eval_rows, model.tokenizer, args.max_length, args.max_b_eval, "tokenize B eval")
    if not a_train_batches:
        raise ValueError("A train set is empty")
    if not a_eval_batches:
        raise ValueError("A eval set is empty")
    if not b_eval_batches:
        raise ValueError("B eval set is empty")

    counts = {
        "a_train_rows": len(a_train_batches),
        "a_eval_rows": len(a_eval_batches),
        "b_eval_rows": len(b_eval_batches),
        "b_retention_ratio": b_retention_ratio,
        "b_retention_seed": args.b_retention_seed,
    }
    wandb_run = maybe_init_wandb(args, counts)
    method = "cnl_synth" if args.b_method == "cnl" and args.synthetic_b_retention else args.b_method

    def evaluate(stage: str, global_step: int, a_epoch: int, b_epoch: int, train_avg_loss: float | str) -> dict[str, Any]:
        a_ok = infer_batches(
            weights,
            a_eval_batches,
            predict_logits_fn,
            cand_ids,
            str(jsonl_dir / f"infer_a_{stage}.jsonl"),
            f"infer A {stage}",
        )
        b_ok = infer_batches(
            weights,
            b_eval_batches,
            predict_logits_fn,
            cand_ids,
            str(jsonl_dir / f"infer_b_{stage}.jsonl"),
            f"infer B {stage}",
        )
        a_loss = eval_loss_batches(weights, a_eval_batches, loss_fn, f"loss A {stage}")
        b_loss = eval_loss_batches(weights, b_eval_batches, loss_fn, f"loss B {stage}")
        metrics = build_metrics(
            stage=stage,
            global_step=global_step,
            a_epoch=a_epoch,
            b_epoch=b_epoch,
            train_avg_loss=train_avg_loss,
            a_ok=a_ok,
            b_ok=b_ok,
            a_loss=a_loss,
            b_loss=b_loss,
            n_a_eval=len(a_eval_batches),
            n_b_eval=len(b_eval_batches),
            a_after_a_acc=a_after_a_acc,
            b_before_b_acc=b_before_b_acc,
            a_after_a_loss=a_after_a_loss,
            b_before_b_loss=b_before_b_loss,
        )
        metrics.update(
            {
                "method": method,
                "b_method": args.b_method,
                "synthetic_b_retention": args.synthetic_b_retention,
                "b_retention_ratio": b_retention_ratio,
                "b_retention_seed": args.b_retention_seed,
                "a_lr": args.a_lr,
                "b_lr": args.b_lr,
                "a_optimizer": args.a_optimizer,
                "b_optimizer": args.b_optimizer,
                "mask_stage": args.mask_stage if args.b_method == "cnl" else "none",
            }
        )
        return metrics

    a_after_a_acc = None
    b_before_b_acc = None
    a_after_a_loss = None
    b_before_b_loss = None
    global_step = 0
    final_metrics = None

    print("\n===== Base Evaluation =====")
    metrics = evaluate("base", global_step, 0, 0, "")
    append_csv(summary_csv, metrics, header)
    if wandb_run is not None:
        wandb_run.log(metrics, step=global_step)
    write_stage_json(str(Path(args.out_dir) / "base_summary.json"), metrics)

    a_optimizer = make_optimizer(args.a_optimizer, args.a_lr, weight_decay=args.a_weight_decay)
    a_opt_state = a_optimizer.init(weights)

    def a_train_step(w, state, batch):
        loss, grads = loss_grad_fn(w, batch)
        updates, state = a_optimizer.update(grads, state, w)
        w = optax.apply_updates(w, updates)
        return w, state, loss

    a_train_step_fn = jax.jit(a_train_step, donate_argnums=(0, 1))

    for ep in range(1, args.a_epochs + 1):
        print(f"\n===== Stage A: train on A epoch {ep}/{args.a_epochs} =====")
        loss_sum = 0.0
        for batch in tqdm(a_train_batches, desc=f"train A ep{ep}"):
            weights, a_opt_state, loss = a_train_step_fn(weights, a_opt_state, array_batch(batch))
            loss_sum += float(loss)
            global_step += 1
        train_loss = loss_sum / len(a_train_batches)
        metrics = evaluate(f"after_a_ep{ep}", global_step, ep, 0, train_loss)
        append_csv(summary_csv, metrics, header)
        if wandb_run is not None:
            wandb_run.log(metrics, step=global_step)
        final_metrics = metrics

    if final_metrics is None:
        raise ValueError("--a_epochs must be >= 1 for the practical A-then-B setting")

    a_after_a_acc = float(final_metrics["a_accuracy"])
    b_before_b_acc = float(final_metrics["b_accuracy"])
    a_after_a_loss = float(final_metrics["a_loss"])
    b_before_b_loss = float(final_metrics["b_loss"])
    after_a_metrics = build_metrics(
        stage="after_a",
        global_step=global_step,
        a_epoch=args.a_epochs,
        b_epoch=0,
        train_avg_loss=final_metrics["train_avg_loss"],
        a_ok=int(final_metrics["a_correct"]),
        b_ok=int(final_metrics["b_correct"]),
        a_loss=a_after_a_loss,
        b_loss=b_before_b_loss,
        n_a_eval=len(a_eval_batches),
        n_b_eval=len(b_eval_batches),
        a_after_a_acc=a_after_a_acc,
        b_before_b_acc=b_before_b_acc,
        a_after_a_loss=a_after_a_loss,
        b_before_b_loss=b_before_b_loss,
    )
    write_stage_json(str(Path(args.out_dir) / "after_a_summary.json"), after_a_metrics)

    print("\n===== Build B-stage Train/Retention Sets After A =====")
    if args.synthetic_b_retention:
        synthetic_limit = args.synthetic_max_rows if args.synthetic_max_rows is not None else args.max_b_retention
        b_retention_rows = make_synthetic_retention_rows(
            synthetic_source_rows,
            model.tokenizer,
            weights,
            predict_logits_fn,
            cand_ids,
            max_length=args.max_length,
            max_rows=synthetic_limit,
            label_mode=args.synthetic_label_mode,
            temperature=args.synthetic_temperature,
            min_confidence=args.synthetic_min_confidence,
            seed=args.synthetic_seed,
            out_jsonl=str(jsonl_dir / "synthetic_b_retention_after_a.jsonl"),
        )
    else:
        if args.max_b_retention is not None:
            b_retention_rows = b_retention_rows[: args.max_b_retention]
        b_retention_rows = random_subset_rows(
            b_retention_rows,
            b_retention_ratio,
            args.b_retention_seed,
            "subset B-stage A-retention rows",
        )
    b_train_batches = filter_batches(
        b_train_rows,
        model.tokenizer,
        weights,
        predict_logits_fn,
        cand_ids,
        args.max_length,
        args.b_train_filter,
        args.max_b_train,
        str(jsonl_dir / "b_train_filtered_after_a.jsonl"),
        "filter B train after A",
    )
    b_retention_batches = filter_batches(
        b_retention_rows,
        model.tokenizer,
        weights,
        predict_logits_fn,
        cand_ids,
        args.max_length,
        args.b_retention_filter,
        args.max_b_retention,
        str(jsonl_dir / "b_retention_filtered_after_a.jsonl"),
        "filter B-stage retention after A",
    )
    if not b_train_batches:
        raise ValueError("B train set is empty after filtering")
    if args.b_method == "cnl" and not b_retention_batches:
        raise ValueError("B-stage CNL requested but retention/reference set is empty")
    if wandb_run is not None:
        wandb_run.config.update(
            {"b_train_rows": len(b_train_batches), "b_retention_rows": len(b_retention_batches)},
            allow_val_change=True,
        )

    b_optimizer = make_optimizer(args.b_optimizer, args.b_lr, weight_decay=args.b_weight_decay)
    b_opt_state = b_optimizer.init(weights)

    def b_train_step(w, state, batch, ref_grads):
        loss, grads = loss_grad_fn(w, batch)
        if args.b_method == "cnl":
            w, state, _ = cnl_optax_step(
                w,
                grads,
                ref_grads,
                b_optimizer,
                state,
                mask_stage=args.mask_stage,
            )
        else:
            updates, state = b_optimizer.update(grads, state, w)
            w = optax.apply_updates(w, updates)
        return w, state, loss

    b_train_step_fn = jax.jit(b_train_step, donate_argnums=(0, 1))

    for ep in range(1, args.b_epochs + 1):
        print(f"\n===== Stage B: train on B epoch {ep}/{args.b_epochs} ({args.b_method}) =====")
        ref_grads = mean_grad_retention(weights, b_retention_batches, loss_grad_fn) if args.b_method == "cnl" else None
        loss_sum = 0.0
        for batch in tqdm(b_train_batches, desc=f"train B ep{ep}"):
            if (
                args.b_method == "cnl"
                and args.ref_refresh_steps > 0
                and global_step > 0
                and global_step % args.ref_refresh_steps == 0
            ):
                ref_grads = mean_grad_retention(weights, b_retention_batches, loss_grad_fn)
            weights, b_opt_state, loss = b_train_step_fn(weights, b_opt_state, array_batch(batch), ref_grads)
            loss_sum += float(loss)
            global_step += 1
        train_loss = loss_sum / len(b_train_batches)
        metrics = evaluate(f"after_b_ep{ep}", global_step, args.a_epochs, ep, train_loss)
        append_csv(summary_csv, metrics, header)
        if wandb_run is not None:
            wandb_run.log(metrics, step=global_step)
        final_metrics = metrics

    if final_metrics is not None:
        save_final_summary(args.out_dir, final_metrics)
        write_stage_json(str(Path(args.out_dir) / "after_b_summary.json"), final_metrics)
    if wandb_run is not None:
        if final_metrics is not None:
            update_wandb_summary(wandb_run, final_metrics)
        wandb_run.finish()
    print("Done.")


if __name__ == "__main__":
    main()
