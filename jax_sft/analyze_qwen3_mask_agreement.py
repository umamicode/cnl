#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare real-CNL and synthetic-CNL masks without running training.

This is a standalone diagnostic. It intentionally does not modify the training
scripts, so it can run alongside active sweeps.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax_sft.qwen3_ptx_split_train import (
    add_trees,
    array_batch,
    candidate_ids,
    divide_tree,
    ensure_pad_token,
    load_jsonl,
    loss_for_batch,
    make_optimizer,
    make_random_mcq_rows,
    make_synthetic_correct_batches,
    predict_abcd,
    split_rows,
    subset_items,
    tokenize_row,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ptx_dir", type=str, default=None)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--weights_dir", type=str, default="~/weights")
    p.add_argument("--source_jsonl", type=str, nargs="+", required=True)
    p.add_argument("--out_correct_jsonl", type=str, required=True)
    p.add_argument("--out_wrong_jsonl", type=str, required=True)
    p.add_argument("--skip_split", action="store_true")
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--max_correct", type=int, default=None)
    p.add_argument("--max_wrong", type=int, default=None)
    p.add_argument(
        "--max_wrong_batches",
        type=int,
        default=64,
        help="Number of wrong/injection batches to analyze. Use 0 for all.",
    )
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--correct_ratio", type=float, default=1.0)
    p.add_argument("--correct_seed", type=int, default=0)
    p.add_argument("--correct_subset_mode", choices=["random", "nested"], default="nested")
    p.add_argument("--synthetic_correct_jsonl", type=str, nargs="+", default=None)
    p.add_argument("--synthetic_correct_source_jsonl", type=str, nargs="+", default=None)
    p.add_argument("--synthetic_correct_mode", choices=["source", "random"], default="random")
    p.add_argument("--synthetic_correct_n", type=int, default=512)
    p.add_argument(
        "--synthetic_correct_size_match",
        choices=["fixed", "correct", "wrong"],
        default="correct",
    )
    p.add_argument("--synthetic_correct_max_rows", type=int, default=None)
    p.add_argument("--synthetic_label_mode", choices=["argmax", "sample"], default="argmax")
    p.add_argument("--synthetic_temperature", type=float, default=1.0)
    p.add_argument("--synthetic_min_confidence", type=float, default=0.0)
    p.add_argument("--synthetic_seed", type=int, default=0)
    p.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default="adamw")
    p.add_argument("--lr", type=float, default=1e-7)
    p.add_argument("--weight_decay", type=float, default=1e-1)
    p.add_argument("--mask_stage", choices=["gradient", "update"], default="update")
    p.add_argument(
        "--agreement_margin",
        type=float,
        default=0.0,
        help="Treat a coordinate as allowed when score >= -agreement_margin.",
    )
    p.add_argument(
        "--optimizer_state_policy",
        choices=["init", "scan"],
        default="scan",
        help=(
            "'init' compares each wrong example with a fresh optimizer state. "
            "'scan' advances optimizer moments across wrong examples while "
            "keeping the weights fixed."
        ),
    )
    p.add_argument("--tp_size", type=int, default=None)
    p.add_argument("--dp_shard", action="store_true")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default=None)
    return p.parse_args()


def configure_wandb_mode(mode: str | None) -> None:
    if mode:
        os.environ["WANDB_MODE"] = mode
    elif os.environ.get("WANDB_MODE") == "":
        os.environ.pop("WANDB_MODE", None)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def path_to_name(path: tuple[Any, ...]) -> str:
    parts = []
    for part in path:
        if hasattr(part, "key"):
            parts.append(str(part.key))
        elif hasattr(part, "idx"):
            parts.append(str(part.idx))
        elif hasattr(part, "name"):
            parts.append(str(part.name))
        else:
            parts.append(str(part))
    return ".".join(parts)


def layer_group(name: str) -> str:
    if name.startswith("model.layers."):
        parts = name.split(".")
        if len(parts) >= 3:
            return ".".join(parts[:3])
    if name.startswith("model.embed_tokens"):
        return "model.embed_tokens"
    if name.startswith("model.norm"):
        return "model.norm"
    if name.startswith("lm_head"):
        return "lm_head"
    return name.split(".weight")[0]


def zeros_stats() -> dict[str, float]:
    return {
        "n": 0.0,
        "agree": 0.0,
        "intersection": 0.0,
        "union": 0.0,
        "real_on": 0.0,
        "synth_on": 0.0,
        "real_only": 0.0,
        "synth_only": 0.0,
        "both_off": 0.0,
        "weight_sum": 0.0,
        "w_agree": 0.0,
        "w_intersection": 0.0,
        "w_union": 0.0,
        "w_real_on": 0.0,
        "w_synth_on": 0.0,
    }


def add_stats(dst: dict[str, float], src: dict[str, float]) -> None:
    for key, value in src.items():
        dst[key] += float(value)


def finalize_stats(prefix: str, stats: dict[str, float]) -> dict[str, float]:
    n = stats["n"]
    weight_sum = stats["weight_sum"]
    out = {
        f"{prefix}/n_params_compared": n,
        f"{prefix}/agreement": stats["agree"] / n if n else 0.0,
        f"{prefix}/jaccard": stats["intersection"] / stats["union"] if stats["union"] else 1.0,
        f"{prefix}/real_density": stats["real_on"] / n if n else 0.0,
        f"{prefix}/synth_density": stats["synth_on"] / n if n else 0.0,
        f"{prefix}/real_only_density": stats["real_only"] / n if n else 0.0,
        f"{prefix}/synth_only_density": stats["synth_only"] / n if n else 0.0,
        f"{prefix}/both_off_density": stats["both_off"] / n if n else 0.0,
        f"{prefix}/weighted_agreement": stats["w_agree"] / weight_sum if weight_sum else 0.0,
        f"{prefix}/weighted_jaccard": (
            stats["w_intersection"] / stats["w_union"] if stats["w_union"] else 1.0
        ),
        f"{prefix}/weighted_real_density": stats["w_real_on"] / weight_sum if weight_sum else 0.0,
        f"{prefix}/weighted_synth_density": stats["w_synth_on"] / weight_sum if weight_sum else 0.0,
    }
    return out


def leaf_stats(
    value: jax.Array,
    real_ref: jax.Array,
    synth_ref: jax.Array,
    *,
    mask_stage: str,
    agreement_margin: float,
) -> dict[str, float]:
    direction = -value if mask_stage == "update" else value
    weight = jnp.abs(direction).astype(jnp.float32)
    real_score = direction * real_ref
    synth_score = direction * synth_ref
    real_mask = real_score >= -agreement_margin
    synth_mask = synth_score >= -agreement_margin
    both = jnp.logical_and(real_mask, synth_mask)
    either = jnp.logical_or(real_mask, synth_mask)
    agree = real_mask == synth_mask
    real_only = jnp.logical_and(real_mask, jnp.logical_not(synth_mask))
    synth_only = jnp.logical_and(jnp.logical_not(real_mask), synth_mask)
    both_off = jnp.logical_and(jnp.logical_not(real_mask), jnp.logical_not(synth_mask))
    weighted = lambda mask: jnp.sum(jnp.where(mask, weight, jnp.zeros_like(weight)))
    stats = {
        "n": float(real_mask.size),
        "agree": float(jax.device_get(jnp.sum(agree))),
        "intersection": float(jax.device_get(jnp.sum(both))),
        "union": float(jax.device_get(jnp.sum(either))),
        "real_on": float(jax.device_get(jnp.sum(real_mask))),
        "synth_on": float(jax.device_get(jnp.sum(synth_mask))),
        "real_only": float(jax.device_get(jnp.sum(real_only))),
        "synth_only": float(jax.device_get(jnp.sum(synth_only))),
        "both_off": float(jax.device_get(jnp.sum(both_off))),
        "weight_sum": float(jax.device_get(jnp.sum(weight))),
        "w_agree": float(jax.device_get(weighted(agree))),
        "w_intersection": float(jax.device_get(weighted(both))),
        "w_union": float(jax.device_get(weighted(either))),
        "w_real_on": float(jax.device_get(weighted(real_mask))),
        "w_synth_on": float(jax.device_get(weighted(synth_mask))),
    }
    return stats


def tree_mask_stats(
    value_tree: Any,
    real_ref_tree: Any,
    synth_ref_tree: Any,
    *,
    mask_stage: str,
    agreement_margin: float,
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    value_flat = dict(jax.tree_util.tree_flatten_with_path(value_tree)[0])
    real_flat = dict(jax.tree_util.tree_flatten_with_path(real_ref_tree)[0])
    synth_flat = dict(jax.tree_util.tree_flatten_with_path(synth_ref_tree)[0])
    global_stats = zeros_stats()
    by_param = {}
    by_layer: dict[str, dict[str, float]] = defaultdict(zeros_stats)

    for path, value in value_flat.items():
        name = path_to_name(path)
        stats = leaf_stats(
            value,
            real_flat[path],
            synth_flat[path],
            mask_stage=mask_stage,
            agreement_margin=agreement_margin,
        )
        by_param[name] = stats
        add_stats(global_stats, stats)
        add_stats(by_layer[layer_group(name)], stats)

    return global_stats, by_layer, by_param


def stats_rows(scope: str, grouped: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    for name, stats in sorted(grouped.items()):
        row = {"scope": scope, "name": name}
        row.update(finalize_stats("metrics", stats))
        rows.append(row)
    return rows


def mean_grad(weights: Any, batches: list[dict[str, Any]], loss_grad_fn: Any, desc: str) -> Any:
    if not batches:
        raise ValueError(f"{desc} set is empty")
    total_grad = None
    for batch in tqdm(batches, desc=desc):
        _, grad = loss_grad_fn(weights, array_batch(batch))
        total_grad = grad if total_grad is None else add_trees(total_grad, grad)
    return divide_tree(total_grad, float(len(batches)))


def load_batches_from_jsonl(paths: list[str], tokenizer: Any, max_rows: int | None, max_length: int, desc: str) -> list[dict[str, Any]]:
    rows = load_jsonl(paths, max_rows)
    return [
        batch
        for row in tqdm(rows, desc=desc)
        if (batch := tokenize_row(tokenizer, row, max_length)) is not None
    ]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ptx_dir:
        ptx_dir = Path(args.ptx_dir).expanduser()
        sys.path.insert(0, str(ptx_dir))
        from models import qwen

        backend = str(ptx_dir)
    else:
        from jax_sft.ptx_backend import qwen

        backend = "vendored:jax_sft.ptx_backend.qwen"

    tp_size = args.tp_size or jax.device_count()
    print("JAX devices:", jax.devices())
    print("MODEL:", args.model_name)
    print("QWEN_BACKEND:", backend)
    print("TP_SIZE:", tp_size)
    print("MASK_STAGE:", args.mask_stage)
    print("OPTIMIZER:", args.optimizer)
    print("LR:", args.lr)
    print("OUT_DIR:", args.out_dir)

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
    predict_logits_fn = jax.jit(lambda w, batch: forward(batch["tokens"], w)[0, batch["last_idx"][0], :])
    cand_ids = candidate_ids(model.tokenizer)

    if args.skip_split:
        print("Reusing existing split files.")
        correct_batches = load_batches_from_jsonl(
            [args.out_correct_jsonl],
            model.tokenizer,
            args.max_rows,
            args.max_length,
            "tokenize correct",
        )
        wrong_batches = load_batches_from_jsonl(
            [args.out_wrong_jsonl],
            model.tokenizer,
            args.max_rows,
            args.max_length,
            "tokenize wrong",
        )
    else:
        rows = load_jsonl(args.source_jsonl, args.max_rows)
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
    correct_ref_batches = subset_items(
        correct_batches,
        args.correct_ratio,
        args.correct_seed,
        args.correct_subset_mode,
        "subset real correct/mastered rows",
    )

    if args.synthetic_correct_jsonl:
        synth_ref_batches = load_batches_from_jsonl(
            args.synthetic_correct_jsonl,
            model.tokenizer,
            None,
            args.max_length,
            "tokenize synthetic correct",
        )
    else:
        if args.synthetic_correct_size_match == "correct":
            synth_n = len(correct_ref_batches)
        elif args.synthetic_correct_size_match == "wrong":
            synth_n = len(wrong_batches)
        else:
            synth_n = args.synthetic_correct_max_rows or args.synthetic_correct_n
        if args.synthetic_correct_mode == "random":
            synthetic_rows = make_random_mcq_rows(synth_n, args.synthetic_seed)
        else:
            source_paths = args.synthetic_correct_source_jsonl or args.source_jsonl
            synthetic_rows = load_jsonl(source_paths, synth_n)
        synth_ref_batches = make_synthetic_correct_batches(
            synthetic_rows,
            model.tokenizer,
            weights,
            predict_logits_fn,
            cand_ids,
            args.max_length,
            label_mode=args.synthetic_label_mode,
            temperature=args.synthetic_temperature,
            min_confidence=args.synthetic_min_confidence,
            seed=args.synthetic_seed,
            out_jsonl=str(out_dir / "synthetic_correct_reference.jsonl"),
        )

    if args.max_wrong_batches and args.max_wrong_batches > 0:
        analyzed_wrong_batches = wrong_batches[: args.max_wrong_batches]
    else:
        analyzed_wrong_batches = wrong_batches

    print(f"real correct reference rows : {len(correct_ref_batches)}")
    print(f"synthetic reference rows    : {len(synth_ref_batches)}")
    print(f"wrong batches analyzed      : {len(analyzed_wrong_batches)} / {len(wrong_batches)}")

    real_ref_grads = mean_grad(weights, correct_ref_batches, loss_grad_fn, "real correct mean-grad")
    synth_ref_grads = mean_grad(weights, synth_ref_batches, loss_grad_fn, "synthetic correct mean-grad")

    optimizer = make_optimizer(args.optimizer, args.lr, weight_decay=args.weight_decay)
    opt_state = optimizer.init(weights)

    run = None
    if args.wandb_project:
        import wandb

        configure_wandb_mode(args.wandb_mode)
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            tags=[
                "analysis:mask_agreement",
                f"mask_stage:{args.mask_stage}",
                f"optimizer:{args.optimizer}",
                f"synth_size:{args.synthetic_correct_size_match}",
                f"synth_mode:{args.synthetic_correct_mode}",
            ],
            config={
                **vars(args),
                "analysis": "real_vs_synth_cnl_mask_agreement",
                "backend": backend,
                "jax_devices": len(jax.devices()),
                "n_correct_ref": len(correct_ref_batches),
                "n_synth_ref": len(synth_ref_batches),
                "n_wrong_total": len(wrong_batches),
                "n_wrong_analyzed": len(analyzed_wrong_batches),
            },
        )

    global_total = zeros_stats()
    layer_total: dict[str, dict[str, float]] = defaultdict(zeros_stats)
    param_total: dict[str, dict[str, float]] = defaultdict(zeros_stats)
    batch_rows = []

    for i, batch in enumerate(tqdm(analyzed_wrong_batches, desc="mask agreement")):
        _, grads = loss_grad_fn(weights, array_batch(batch))
        if args.mask_stage == "update":
            updates, new_opt_state = optimizer.update(grads, opt_state, weights)
            compare_tree = updates
            if args.optimizer_state_policy == "scan":
                opt_state = new_opt_state
        else:
            compare_tree = grads

        batch_stats, by_layer, by_param = tree_mask_stats(
            compare_tree,
            real_ref_grads,
            synth_ref_grads,
            mask_stage=args.mask_stage,
            agreement_margin=args.agreement_margin,
        )
        add_stats(global_total, batch_stats)
        for key, stats in by_layer.items():
            add_stats(layer_total[key], stats)
        for key, stats in by_param.items():
            add_stats(param_total[key], stats)

        row = {
            "batch": i,
            "label": batch.get("label", ""),
            "question": batch.get("question", "")[:200],
            **finalize_stats("batch", batch_stats),
        }
        batch_rows.append(row)
        if run is not None:
            run.log(
                {
                    "batch/agreement": row["batch/agreement"],
                    "batch/jaccard": row["batch/jaccard"],
                    "batch/weighted_agreement": row["batch/weighted_agreement"],
                    "batch/weighted_jaccard": row["batch/weighted_jaccard"],
                    "batch/real_density": row["batch/real_density"],
                    "batch/synth_density": row["batch/synth_density"],
                },
                step=i,
            )

    summary = {
        **finalize_stats("global", global_total),
        "n_correct_ref": len(correct_ref_batches),
        "n_synth_ref": len(synth_ref_batches),
        "n_wrong_total": len(wrong_batches),
        "n_wrong_analyzed": len(analyzed_wrong_batches),
        "mask_stage": args.mask_stage,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "agreement_margin": args.agreement_margin,
        "optimizer_state_policy": args.optimizer_state_policy,
        "synthetic_correct_mode": args.synthetic_correct_mode,
        "synthetic_correct_size_match": args.synthetic_correct_size_match,
    }
    layer_rows = stats_rows("layer", dict(layer_total))
    param_rows = stats_rows("param", dict(param_total))

    (out_dir / "mask_agreement_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(out_dir / "mask_agreement_batches.csv", batch_rows)
    write_csv(out_dir / "mask_agreement_layers.csv", layer_rows)
    write_csv(out_dir / "mask_agreement_params.csv", param_rows)

    print("========== Mask Agreement Summary ==========")
    for key in [
        "global/agreement",
        "global/jaccard",
        "global/weighted_agreement",
        "global/weighted_jaccard",
        "global/real_density",
        "global/synth_density",
        "global/real_only_density",
        "global/synth_only_density",
    ]:
        print(f"{key}: {summary[key]:.6f}")
    print(f"Wrote: {out_dir / 'mask_agreement_summary.json'}")
    print(f"Wrote: {out_dir / 'mask_agreement_layers.csv'}")

    if run is not None:
        for key, value in summary.items():
            run.summary[key] = value
        import wandb

        run.log(
            {
                "mask_agreement_layers": wandb_table(wandb, layer_rows),
                "mask_agreement_params": wandb_table(wandb, param_rows),
                "mask_agreement_batches": wandb_table(wandb, batch_rows),
            }
        )
        run.finish()


def wandb_table(wandb: Any, rows: list[dict[str, Any]]) -> Any:
    if not rows:
        return wandb.Table(columns=["empty"], data=[])
    columns = list(rows[0].keys())
    return wandb.Table(columns=columns, data=[[row.get(col) for col in columns] for row in rows])


if __name__ == "__main__":
    main()
