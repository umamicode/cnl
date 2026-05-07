#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JAX/Optax SFT runner for Collaborative Neuron Learning.

This script mirrors ``sft/sft.py`` but uses Flax models and Optax optimizers.
It uses fixed-length tokenization and jitted per-sample steps so TPU smoke tests
do not recompile for every question length.
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
    p.add_argument(
        "--max_wrong",
        type=int,
        default=None,
        help="Optional cap for wrong/injection rows, useful for smoke tests.",
    )
    p.add_argument(
        "--max_correct",
        type=int,
        default=None,
        help="Optional cap for correct/mastered rows, useful for smoke tests.",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Static token length for padding/truncation. Keep fixed for JIT reuse.",
    )
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default=None, help="Examples: online, offline, disabled.")
    p.add_argument(
        "--eval_before_train",
        type=int,
        choices=[0, 1],
        default=1,
        help="Write epoch 0 metrics before any updates.",
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


def format_prompt(tok: AutoTokenizer, question: str) -> str:
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )

    return question


def ensure_pad_token(tok: AutoTokenizer) -> None:
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token


def label_token_id(tok: AutoTokenizer, label: str) -> int:
    ids = tok(label, add_special_tokens=False, return_tensors="np").input_ids
    return int(ids[0, -1])


def tokenize_row(tok: AutoTokenizer, row: dict[str, Any], max_length: int) -> dict[str, Any]:
    batch = tok(
        format_prompt(tok, row["question"]),
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    return {
        "input_ids": jnp.asarray(batch["input_ids"], dtype=jnp.int32),
        "attention_mask": jnp.asarray(batch["attention_mask"], dtype=jnp.int32),
        "label_id": jnp.asarray([label_token_id(tok, row["label"])], dtype=jnp.int32),
        "label": row["label"],
        "question": row["question"],
    }


def tokenize_rows(
    tok: AutoTokenizer,
    rows: list[dict[str, Any]],
    max_length: int,
    desc: str,
) -> list[dict[str, Any]]:
    return [tokenize_row(tok, row, max_length) for row in tqdm(rows, desc=desc)]


def array_batch(batch: dict[str, Any]) -> dict[str, jax.Array]:
    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "label_id": batch["label_id"],
    }


def candidate_ids(tok: AutoTokenizer) -> np.ndarray:
    return np.asarray(
        [
            int(tok(c, add_special_tokens=False, return_tensors="np").input_ids[0, -1])
            for c in "ABCD"
        ],
        dtype=np.int32,
    )


def loss_for_batch(model: Any, params: Any, batch: dict[str, jax.Array]) -> jax.Array:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        params=params,
        train=False,
    )
    lengths = jnp.sum(batch["attention_mask"], axis=1) - 1
    logits = outputs.logits[jnp.arange(outputs.logits.shape[0]), lengths, :]
    return optax.softmax_cross_entropy_with_integer_labels(logits, batch["label_id"]).mean()


def mean_grad_correct(
    model: Any,
    params: Any,
    batches: list[dict[str, Any]],
    loss_grad_fn: Any,
) -> Any:
    if not batches:
        raise ValueError("correct rows are empty")

    total_grad = None

    for batch in tqdm(batches, desc="correct mean-grad"):
        _, grad = loss_grad_fn(params, array_batch(batch))
        total_grad = grad if total_grad is None else add_trees(total_grad, grad)

    return divide_tree(total_grad, float(len(batches)))


def train_wrong_epoch(
    params: Any,
    batches: list[dict[str, Any]],
    opt_state: optax.OptState,
    reference_grads: Any | None,
    train_step_fn: Any,
    desc: str,
) -> tuple[Any, optax.OptState, float]:
    if not batches:
        raise ValueError("wrong rows are empty")

    loss_sum = 0.0

    for batch in tqdm(batches, desc=desc):
        params, opt_state, loss = train_step_fn(params, opt_state, array_batch(batch), reference_grads)
        loss_sum += float(loss)

    return params, opt_state, loss_sum / len(batches)


def predict_abcd(predict_logits_fn: Any, params: Any, batch: dict[str, Any], cand_ids: np.ndarray) -> str:
    logits = np.asarray(predict_logits_fn(params, batch))
    return "ABCD"[int(np.argmax(logits[cand_ids]))]


def infer_and_dump(
    predict_logits_fn: Any,
    params: Any,
    batches: list[dict[str, Any]],
    cand_ids: np.ndarray,
    path: str,
    desc: str,
) -> int:
    ok = 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for batch in tqdm(batches, desc=desc):
            pred = predict_abcd(predict_logits_fn, params, array_batch(batch), cand_ids)
            ok += int(pred == batch["label"])
            f.write(json.dumps({
                "label": batch["label"],
                "predict_label": pred,
                "question": batch["question"],
            }, ensure_ascii=False) + "\n")
    return ok


def make_train_step(
    model: Any,
    optimizer: optax.GradientTransformation,
    mask_stage: MaskStage,
) -> Any:
    def step(
        params: Any,
        opt_state: optax.OptState,
        batch: dict[str, jax.Array],
        reference_grads: Any | None,
    ) -> tuple[Any, optax.OptState, jax.Array]:
        loss, grads = jax.value_and_grad(lambda p: loss_for_batch(model, p, batch))(params)
        params, opt_state, _ = cnl_optax_step(
            params,
            grads,
            reference_grads,
            optimizer,
            opt_state,
            mask_stage=mask_stage,
        )
        return params, opt_state, loss

    return jax.jit(step)


def make_plain_train_step(model: Any, optimizer: optax.GradientTransformation) -> Any:
    def step(
        params: Any,
        opt_state: optax.OptState,
        batch: dict[str, jax.Array],
        reference_grads: Any | None,
    ) -> tuple[Any, optax.OptState, jax.Array]:
        del reference_grads
        loss, grads = jax.value_and_grad(lambda p: loss_for_batch(model, p, batch))(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return jax.jit(step)


def maybe_init_wandb(args: argparse.Namespace, wrong_rows: int, correct_rows: int) -> Any | None:
    if not args.wandb_project:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb logging was requested, but wandb is not installed. "
            "Install it with: uv pip install wandb"
        ) from exc

    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config={
            "model_name": args.model_name,
            "wrong_jsonl": args.wrong_jsonl,
            "correct_jsonl": args.correct_jsonl,
            "out_dir": args.out_dir,
            "lr": args.lr,
            "epochs": args.epochs,
            "use_freeze": bool(args.use_freeze),
            "optimizer": args.optimizer,
            "mask_stage": args.mask_stage,
            "max_wrong": args.max_wrong,
            "max_correct": args.max_correct,
            "max_length": args.max_length,
            "eval_before_train": bool(args.eval_before_train),
            "wrong_rows": wrong_rows,
            "correct_rows": correct_rows,
            "jax_devices": len(jax.devices()),
        },
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    jsonl_dir = os.path.join(args.out_dir, "jsonl")
    os.makedirs(jsonl_dir, exist_ok=True)

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

    wrong_rows = read_jsonl(args.wrong_jsonl)
    correct_rows = read_jsonl(args.correct_jsonl)
    if args.max_wrong is not None:
        wrong_rows = wrong_rows[:args.max_wrong]
    if args.max_correct is not None:
        correct_rows = correct_rows[:args.max_correct]
    cand_ids = candidate_ids(tok)
    wrong_batches = tokenize_rows(tok, wrong_rows, args.max_length, "tokenize wrong")
    correct_batches = tokenize_rows(tok, correct_rows, args.max_length, "tokenize correct")
    wandb_run = maybe_init_wandb(args, len(wrong_rows), len(correct_rows))

    optimizer = make_optimizer(args.optimizer, args.lr)
    opt_state = optimizer.init(params)
    loss_grad_fn = jax.jit(jax.value_and_grad(lambda p, batch: loss_for_batch(model, p, batch)))
    train_step_fn = (
        make_train_step(model, optimizer, args.mask_stage)
        if args.use_freeze
        else make_plain_train_step(model, optimizer)
    )
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
    print("MAX_LENGTH:", args.max_length)
    print("WRONG_ROWS:", len(wrong_rows))
    print("CORRECT_ROWS:", len(correct_rows))

    if args.eval_before_train:
        print("\n===== Epoch 0 (before training) =====")
        w0_ok = infer_and_dump(
            predict_logits_fn,
            params,
            wrong_batches,
            cand_ids,
            os.path.join(jsonl_dir, "infer_wrong_ep0.jsonl"),
            "infer wrong ep0",
        )
        c0_ok = infer_and_dump(
            predict_logits_fn,
            params,
            correct_batches,
            cand_ids,
            os.path.join(jsonl_dir, "infer_correct_ep0.jsonl"),
            "infer correct ep0",
        )
        append_csv(
            summary_csv,
            {
                "epoch": 0,
                "train_avg_loss": "",
                "wrong_to_correct": w0_ok,
                "correct_to_wrong": len(correct_rows) - c0_ok,
            },
            header,
        )
        if wandb_run is not None:
            wandb_run.log({
                "epoch": 0,
                "wrong_to_correct": w0_ok,
                "correct_to_wrong": len(correct_rows) - c0_ok,
                "wrong_accuracy": w0_ok / len(wrong_rows) if wrong_rows else 0.0,
                "correct_accuracy": c0_ok / len(correct_rows) if correct_rows else 0.0,
                "forgetting_rate": (len(correct_rows) - c0_ok) / len(correct_rows) if correct_rows else 0.0,
            }, step=0)

    for ep in range(1, args.epochs + 1):
        print(f"\n===== Epoch {ep} =====")

        reference_grads = None
        if args.use_freeze:
            reference_grads = mean_grad_correct(
                model,
                params,
                correct_batches,
                loss_grad_fn,
            )

        params, opt_state, train_loss = train_wrong_epoch(
            params,
            wrong_batches,
            opt_state,
            reference_grads,
            train_step_fn,
            desc=f"train wrong ep{ep}",
        )

        w_ok = infer_and_dump(
            predict_logits_fn,
            params,
            wrong_batches,
            cand_ids,
            os.path.join(jsonl_dir, f"infer_wrong_ep{ep}.jsonl"),
            f"infer wrong ep{ep}",
        )
        c_ok = infer_and_dump(
            predict_logits_fn,
            params,
            correct_batches,
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
        if wandb_run is not None:
            wandb_run.log({
                "epoch": ep,
                "train_avg_loss": train_loss,
                "wrong_to_correct": w_ok,
                "correct_to_wrong": len(correct_rows) - c_ok,
                "wrong_accuracy": w_ok / len(wrong_rows) if wrong_rows else 0.0,
                "correct_accuracy": c_ok / len(correct_rows) if correct_rows else 0.0,
                "forgetting_rate": (len(correct_rows) - c_ok) / len(correct_rows) if correct_rows else 0.0,
            }, step=ep)

    model.save_pretrained(os.path.join(args.out_dir, "model"), params=params)
    tok.save_pretrained(os.path.join(args.out_dir, "model"))
    if wandb_run is not None:
        wandb_run.finish()
    print("Done.")


if __name__ == "__main__":
    main()
