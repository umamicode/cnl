#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Qwen3 CNL split+train using a vendored ptx JAX Qwen backend.

This bypasses Hugging Face FlaxAuto, which does not provide Qwen3 causal-LM
classes. By default it uses ``jax_sft.ptx_backend.qwen``. Pass ``--ptx_dir`` to
override this with an external ptx checkout.
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
from jax.sharding import PartitionSpec as P
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax_sft.cnl import add_trees, cnl_optax_step, divide_tree


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ptx_dir", type=str, default=None)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--weights_dir", type=str, default="~/weights")
    p.add_argument("--source_jsonl", type=str, nargs="+", required=True)
    p.add_argument("--out_correct_jsonl", type=str, required=True)
    p.add_argument("--out_wrong_jsonl", type=str, required=True)
    p.add_argument(
        "--synthetic_correct_jsonl",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional pseudo-labeled rows to use as the CNL reference set. "
            "Evaluation still uses out_correct_jsonl/source split."
        ),
    )
    p.add_argument(
        "--synthetic_correct_source_jsonl",
        type=str,
        nargs="+",
        default=None,
        help="Prompt bank to pseudo-label for cnl_synth. Defaults to source_jsonl.",
    )
    p.add_argument(
        "--synthetic_correct_mode",
        choices=["source", "random"],
        default="source",
        help=(
            "How cnl_synth gets prompts. 'source' pseudo-labels an existing "
            "prompt bank; 'random' creates task-shaped MCQ prompts from a "
            "small built-in vocabulary before pseudo-labeling."
        ),
    )
    p.add_argument("--synthetic_correct_n", type=int, default=512)
    p.add_argument(
        "--synthetic_correct_size_match",
        choices=["fixed", "correct", "wrong"],
        default="fixed",
        help=(
            "For cnl_synth, choose synthetic reference count. 'fixed' uses "
            "synthetic_correct_n/max_rows; 'correct' matches the real correct "
            "set size; 'wrong' matches the real wrong set size."
        ),
    )
    p.add_argument("--synthetic_correct_max_rows", type=int, default=None)
    p.add_argument("--synthetic_label_mode", choices=["argmax", "sample"], default="argmax")
    p.add_argument("--synthetic_temperature", type=float, default=1.0)
    p.add_argument("--synthetic_min_confidence", type=float, default=0.0)
    p.add_argument("--synthetic_seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument(
        "--skip_split",
        action="store_true",
        help="Reuse out_correct_jsonl/out_wrong_jsonl instead of running model inference split.",
    )
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--max_wrong", type=int, default=None)
    p.add_argument("--max_correct", type=int, default=None)
    p.add_argument(
        "--correct_ratio",
        type=float,
        default=1.0,
        help="Random fraction of correct/mastered rows to keep. Values >1 are interpreted as percentages.",
    )
    p.add_argument("--correct_seed", type=int, default=0)
    p.add_argument(
        "--method_name",
        choices=[
            "sft",
            "cnl",
            "cnl_synth",
            "cnl_margin",
            "cnl_leaky",
            "cnl_margin_synth",
            "cnl_leaky_synth",
        ],
        default=None,
    )
    p.add_argument(
        "--correct_subset_mode",
        choices=["random", "nested"],
        default="random",
        help=(
            "How to select correct/mastered reference rows. 'random' keeps the "
            "original independent subset behavior; 'nested' uses a seed-specific "
            "permutation prefix so larger ratios contain smaller ratios."
        ),
    )
    p.add_argument(
        "--correct_eval_scope",
        choices=["subset", "all"],
        default="subset",
        help=(
            "Which correct/mastered rows to evaluate after training. 'subset' "
            "keeps legacy behavior; 'all' evaluates retention on the full "
            "correct set while using correct_ratio only for the CNL reference set."
        ),
    )
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-7)
    p.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default="sgd")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--use_freeze", type=int, choices=[0, 1], default=1)
    p.add_argument("--mask_stage", choices=["gradient", "update"], default="gradient")
    p.add_argument(
        "--cnl_mask_mode",
        choices=["hard", "margin", "leaky"],
        default="hard",
        help="Relaxation for the CNL mask. Method names cnl_margin/cnl_leaky override this.",
    )
    p.add_argument(
        "--cnl_margin",
        type=float,
        default=0.0,
        help="Margin CNL keeps coordinates with similarity >= -cnl_margin.",
    )
    p.add_argument(
        "--cnl_leak",
        type=float,
        default=0.0,
        help="Leaky CNL scales conflicting coordinates by this factor instead of freezing them.",
    )
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


def normalize_ratio(ratio: float) -> float:
    ratio = ratio / 100.0 if ratio > 1 else ratio
    if ratio <= 0 or ratio > 1:
        raise ValueError(f"ratio must be in (0, 1] or (0, 100], got {ratio}")
    return ratio


def subset_items(items: list[Any], ratio: float, seed: int, mode: str, desc: str) -> list[Any]:
    ratio = normalize_ratio(ratio)
    if ratio >= 1 or not items:
        print(f"{desc}: ratio=1.0 mode={mode} kept={len(items)}/{len(items)}")
        return items
    n_keep = max(1, int(round(len(items) * ratio)))
    rng = np.random.default_rng(seed)
    if mode == "random":
        indices = rng.choice(len(items), size=n_keep, replace=False).tolist()
    elif mode == "nested":
        indices = rng.permutation(len(items))[:n_keep].tolist()
    else:
        raise ValueError(f"unknown subset mode: {mode}")
    indices = sorted(indices)
    print(f"{desc}: ratio={ratio:.4f} seed={seed} mode={mode} kept={n_keep}/{len(items)}")
    return [items[i] for i in indices]


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


def array_batch(batch: dict[str, Any]) -> dict[str, jax.Array]:
    return {
        "tokens": batch["tokens"],
        "attention_mask": batch["attention_mask"],
        "last_idx": batch["last_idx"],
        "label_id": batch["label_id"],
    }


def make_optimizer(name: str, lr: float, weight_decay: float = 1e-4) -> optax.GradientTransformation:
    if name == "sgd":
        return optax.sgd(lr)
    if name == "adam":
        return optax.adam(lr)
    if name == "adamw":
        return optax.adamw(lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def loss_for_batch(forward: Any, weights: Any, batch: dict[str, jax.Array]) -> jax.Array:
    logits = forward(batch["tokens"], weights)
    logits = logits.at[jnp.arange(logits.shape[0]), batch["last_idx"], :].get(
        out_sharding=P("data", "model")
    )
    logprobs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    label_logprobs = logprobs.at[jnp.arange(logprobs.shape[0]), batch["label_id"]].get(
        out_sharding=P("data")
    )
    return -label_logprobs.mean()


def predict_abcd(predict_logits_fn: Any, weights: Any, batch: dict[str, Any], cand_ids: np.ndarray) -> str:
    logits = np.asarray(predict_logits_fn(weights, array_batch(batch)))
    return "ABCD"[int(np.argmax(logits[cand_ids]))]


def stable_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def make_random_mcq_rows(n_rows: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    subjects = [
        "a student",
        "a chef",
        "a doctor",
        "a traveler",
        "a gardener",
        "a teacher",
        "a mechanic",
        "a musician",
        "an athlete",
        "a parent",
        "a scientist",
        "an artist",
    ]
    intents = [
        "needs to write a note",
        "wants to stay dry in rain",
        "is preparing food",
        "needs to measure time",
        "wants to open a locked door",
        "is cleaning a floor",
        "needs to cut paper",
        "wants to drink water",
        "is planting seeds",
        "needs to fix a loose screw",
        "wants to read in the dark",
        "is going to mail a letter",
    ]
    places = [
        "kitchen",
        "classroom",
        "hospital",
        "garden",
        "office",
        "garage",
        "library",
        "airport",
        "store",
        "park",
        "bedroom",
        "workshop",
    ]
    objects = [
        "pencil",
        "umbrella",
        "spoon",
        "clock",
        "key",
        "mop",
        "scissors",
        "cup",
        "shovel",
        "screwdriver",
        "lamp",
        "stamp",
        "blanket",
        "hammer",
        "book",
        "phone",
        "shoe",
        "plate",
        "bottle",
        "ruler",
        "chair",
        "map",
        "wallet",
        "soap",
        "towel",
        "basket",
        "paintbrush",
        "fork",
        "battery",
        "notebook",
    ]
    question_templates = [
        "Question: In a {place}, {subject} {intent}. Which object is most useful?",
        "Question: {subject_cap} {intent} while in a {place}. What should they use?",
        "Question: Which item would best help {subject} who {intent}?",
        "Question: If {subject} is in a {place} and {intent}, which choice makes the most sense?",
    ]
    rows = []
    for i in range(n_rows):
        options = rng.choice(objects, size=4, replace=False).tolist()
        subject = str(rng.choice(subjects))
        intent = str(rng.choice(intents))
        place = str(rng.choice(places))
        template = str(rng.choice(question_templates))
        stem = template.format(
            subject=subject,
            subject_cap=subject[:1].upper() + subject[1:],
            intent=intent,
            place=place,
        )
        question = (
            f"{stem}\n"
            f"A. {options[0]}\n"
            f"B. {options[1]}\n"
            f"C. {options[2]}\n"
            f"D. {options[3]}\n"
            "Answer:"
        )
        rows.append({"question": question, "label": "A", "synthetic_prompt": True, "synthetic_prompt_id": i})
    return rows


def make_synthetic_correct_batches(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    weights: Any,
    predict_logits_fn: Any,
    cand_ids: np.ndarray,
    max_length: int,
    *,
    label_mode: str,
    temperature: float,
    min_confidence: float,
    seed: int,
    out_jsonl: str,
) -> list[dict[str, Any]]:
    if temperature <= 0:
        raise ValueError("--synthetic_temperature must be positive")
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    rng = np.random.default_rng(seed)
    batches = []
    skipped = 0
    low_conf = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="make synthetic correct"):
            question = row.get("question")
            if not question:
                skipped += 1
                continue
            prompt_batch = tokenize_prompt(tokenizer, question, max_length)
            if prompt_batch is None:
                skipped += 1
                continue
            logits = np.asarray(predict_logits_fn(weights, prompt_batch))[cand_ids]
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
            synthetic_row = {
                "label": label,
                "question": question,
                "predict_label": label,
                "synthetic": True,
                "synthetic_method": "qwen3_on_policy_pseudo_label",
                "synthetic_label_mode": label_mode,
                "synthetic_confidence": confidence,
                "source_label": row.get("label"),
                "source_predict_label": row.get("predict_label"),
            }
            batch = tokenize_row(tokenizer, synthetic_row, max_length)
            if batch is None:
                skipped += 1
                continue
            batches.append(batch)
            write_jsonl_line(f, synthetic_row)

    print("========== Synthetic Correct Summary ==========")
    print(f"Source rows    : {len(rows)}")
    print(f"Kept           : {len(batches)}")
    print(f"Skipped        : {skipped}")
    print(f"Low confidence : {low_conf}")
    return batches


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


def configure_wandb_mode(mode: str | None) -> None:
    if mode:
        os.environ["WANDB_MODE"] = mode
    elif os.environ.get("WANDB_MODE") == "":
        os.environ.pop("WANDB_MODE", None)


def effective_method(args: argparse.Namespace) -> str:
    if args.method_name:
        return args.method_name
    if not args.use_freeze:
        return "sft"
    return "cnl_synth" if args.synthetic_correct_jsonl else "cnl"


def method_id(method: str) -> int:
    return {
        "sft": 0,
        "cnl": 1,
        "cnl_synth": 2,
        "cnl_margin": 3,
        "cnl_leaky": 4,
        "cnl_margin_synth": 5,
        "cnl_leaky_synth": 6,
    }.get(method, -1)


def effective_cnl_mask_mode(args: argparse.Namespace) -> str:
    method = effective_method(args)
    if method in ("cnl_margin", "cnl_margin_synth"):
        return "margin"
    if method in ("cnl_leaky", "cnl_leaky_synth"):
        return "leaky"
    return args.cnl_mask_mode


def uses_synthetic_reference(args: argparse.Namespace) -> bool:
    return effective_method(args).endswith("_synth")


def method_variant(args: argparse.Namespace) -> str:
    method = effective_method(args)
    if method == "cnl_margin":
        return f"{method}_m{args.cnl_margin:g}"
    if method == "cnl_leaky":
        return f"{method}_a{args.cnl_leak:g}"
    if method == "cnl_synth":
        return f"{method}_{args.synthetic_correct_size_match}"
    if method == "cnl_margin_synth":
        return f"{method}_{args.synthetic_correct_size_match}_m{args.cnl_margin:g}"
    if method == "cnl_leaky_synth":
        return f"{method}_{args.synthetic_correct_size_match}_a{args.cnl_leak:g}"
    return method


def method_variant_id(args: argparse.Namespace) -> int:
    method = effective_method(args)
    if not uses_synthetic_reference(args):
        return method_id(method)
    size_offset = {
        "fixed": 0,
        "correct": 1,
        "wrong": 2,
    }.get(args.synthetic_correct_size_match, 0)
    return {
        "cnl_synth": 20,
        "cnl_margin_synth": 50,
        "cnl_leaky_synth": 60,
    }.get(method, 90) + size_offset


def wandb_tags(args: argparse.Namespace) -> list[str]:
    method = effective_method(args)
    tags = [f"method:{method}", f"variant:{method_variant(args)}"]
    if uses_synthetic_reference(args):
        tags.append(f"synth_size:{args.synthetic_correct_size_match}")
        tags.append(f"synth_mode:{args.synthetic_correct_mode}")
    if method.startswith("cnl"):
        tags.append(f"cnl_mask:{effective_cnl_mask_mode(args)}")
    return tags


def maybe_init_wandb(
    args: argparse.Namespace,
    n_wrong: int,
    n_correct_eval: int,
    n_correct_ref: int,
) -> Any | None:
    if not args.wandb_project:
        return None
    import wandb

    configure_wandb_mode(args.wandb_mode)
    method = effective_method(args)
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=wandb_tags(args),
        config={
            **vars(args),
            "method": method,
            "method_id": method_id(method),
            "method_variant": method_variant(args),
            "method_variant_id": method_variant_id(args),
            "is_cnl": int(method.startswith("cnl")),
            "effective_cnl_mask_mode": effective_cnl_mask_mode(args),
            "wrong_rows": n_wrong,
            "correct_rows": n_correct_eval,
            "correct_eval_rows": n_correct_eval,
            "correct_ref_rows": n_correct_ref,
            "correct_ratio_normalized": normalize_ratio(args.correct_ratio),
            "jax_devices": len(jax.devices()),
        },
    )


def build_metrics(
    epoch: int,
    train_loss: float | str,
    wrong_ok: int,
    correct_ok: int,
    n_wrong: int,
    n_correct: int,
) -> dict[str, Any]:
    correct_to_wrong = n_correct - correct_ok
    wrong_accuracy = wrong_ok / n_wrong if n_wrong else 0.0
    correct_accuracy = correct_ok / n_correct if n_correct else 0.0
    forgetting_rate = correct_to_wrong / n_correct if n_correct else 0.0
    return {
        "epoch": epoch,
        "train_avg_loss": train_loss,
        "wrong_to_correct": wrong_ok,
        "correct_to_wrong": correct_to_wrong,
        "wrong_accuracy": wrong_accuracy,
        "correct_accuracy": correct_accuracy,
        "forgetting_rate": forgetting_rate,
        "learning_rate": wrong_accuracy,
        "retention_rate": correct_accuracy,
        "n_wrong": n_wrong,
        "n_correct": n_correct,
    }


def update_wandb_summary(wandb_run: Any | None, metrics: dict[str, Any]) -> None:
    if wandb_run is None:
        return
    for key, value in metrics.items():
        if value == "":
            continue
        wandb_run.summary[key] = value
        wandb_run.summary[f"final/{key}"] = value


def save_final_summary(out_dir: str, metrics: dict[str, Any]) -> None:
    path = Path(out_dir) / "final_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    args = parse_args()
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

    cand_ids = candidate_ids(model.tokenizer)
    if args.skip_split:
        print("Reusing existing split files.")
        correct_rows = load_jsonl([args.out_correct_jsonl], args.max_rows)
        wrong_rows = load_jsonl([args.out_wrong_jsonl], args.max_rows)
        correct_batches = [
            batch for row in tqdm(correct_rows, desc="tokenize correct") if (batch := tokenize_row(model.tokenizer, row, args.max_length)) is not None
        ]
        wrong_batches = [
            batch for row in tqdm(wrong_rows, desc="tokenize wrong") if (batch := tokenize_row(model.tokenizer, row, args.max_length)) is not None
        ]
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
    jsonl_dir = Path(args.out_dir) / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    correct_eval_batches = list(correct_batches)
    if args.synthetic_correct_jsonl:
        synthetic_rows = load_jsonl(args.synthetic_correct_jsonl, None)
        correct_ref_source_batches = [
            batch
            for row in tqdm(synthetic_rows, desc="tokenize synthetic correct")
            if (batch := tokenize_row(model.tokenizer, row, args.max_length)) is not None
        ]
        if not correct_ref_source_batches:
            raise ValueError("synthetic correct/reference set is empty")
    elif uses_synthetic_reference(args):
        if args.synthetic_correct_size_match == "correct":
            matched_synth_n = len(correct_batches)
        elif args.synthetic_correct_size_match == "wrong":
            matched_synth_n = len(wrong_batches)
        else:
            matched_synth_n = args.synthetic_correct_max_rows or args.synthetic_correct_n
        if args.synthetic_correct_mode == "random":
            synthetic_rows = make_random_mcq_rows(matched_synth_n, args.synthetic_seed)
        else:
            source_paths = args.synthetic_correct_source_jsonl or args.source_jsonl
            synthetic_rows = load_jsonl(source_paths, matched_synth_n)
        correct_ref_source_batches = make_synthetic_correct_batches(
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
            out_jsonl=str(jsonl_dir / "synthetic_correct_reference.jsonl"),
        )
        if not correct_ref_source_batches:
            raise ValueError("synthetic correct/reference set is empty")
    else:
        correct_ref_source_batches = correct_batches
    correct_ref_batches = subset_items(
        correct_ref_source_batches,
        args.correct_ratio,
        args.correct_seed,
        args.correct_subset_mode,
        "subset correct/mastered rows",
    )
    if args.correct_eval_scope == "subset":
        correct_eval_batches = correct_ref_batches
    print(f"correct reference rows: {len(correct_ref_batches)}")
    print(f"correct eval rows     : {len(correct_eval_batches)} ({args.correct_eval_scope})")

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
            mask_mode=effective_cnl_mask_mode(args),
            margin=args.cnl_margin,
            leak=args.cnl_leak,
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
        "method",
        "method_id",
        "method_variant",
        "method_variant_id",
        "is_cnl",
        "use_freeze",
        "effective_cnl_mask_mode",
        "cnl_margin",
        "cnl_leak",
        "correct_ratio",
        "correct_seed",
        "correct_subset_mode",
        "correct_eval_scope",
        "n_correct_ref",
        "synthetic_correct_mode",
        "synthetic_correct_size_match",
        "synthetic_correct_n_effective",
        "train_avg_loss",
        "wrong_to_correct",
        "correct_to_wrong",
        "wrong_accuracy",
        "correct_accuracy",
        "forgetting_rate",
        "learning_rate",
        "retention_rate",
        "n_wrong",
        "n_correct",
    ]
    final_metrics = None
    method = effective_method(args)
    numeric_method_id = method_id(method)
    variant = method_variant(args)
    numeric_variant_id = method_variant_id(args)
    cnl_mask_mode = effective_cnl_mask_mode(args)
    correct_ratio = normalize_ratio(args.correct_ratio)

    def annotate_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        metrics.update(
            {
                "method": method,
                "method_id": numeric_method_id,
                "method_variant": variant,
                "method_variant_id": numeric_variant_id,
                "is_cnl": int(method.startswith("cnl")),
                "use_freeze": args.use_freeze,
                "effective_cnl_mask_mode": cnl_mask_mode,
                "cnl_margin": args.cnl_margin if method in ("cnl_margin", "cnl_margin_synth") else "",
                "cnl_leak": args.cnl_leak if method in ("cnl_leaky", "cnl_leaky_synth") else "",
                "correct_ratio": correct_ratio,
                "correct_seed": args.correct_seed,
                "correct_subset_mode": args.correct_subset_mode,
                "correct_eval_scope": args.correct_eval_scope,
                "n_correct_ref": len(correct_ref_batches),
                "synthetic_correct_mode": args.synthetic_correct_mode if uses_synthetic_reference(args) else "",
                "synthetic_correct_size_match": args.synthetic_correct_size_match if uses_synthetic_reference(args) else "",
                "synthetic_correct_n_effective": len(correct_ref_source_batches) if uses_synthetic_reference(args) else "",
            }
        )
        return metrics

    wandb_run = maybe_init_wandb(
        args,
        len(wrong_batches),
        len(correct_eval_batches),
        len(correct_ref_batches),
    )

    if args.eval_before_train:
        print("\n===== Epoch 0 (before training) =====")
        w0 = infer_batches(weights, wrong_batches, predict_logits_fn, cand_ids, str(jsonl_dir / "infer_wrong_ep0.jsonl"), "infer wrong ep0")
        c0 = infer_batches(weights, correct_eval_batches, predict_logits_fn, cand_ids, str(jsonl_dir / "infer_correct_ep0.jsonl"), "infer correct ep0")
        metrics = annotate_metrics(build_metrics(0, "", w0, c0, len(wrong_batches), len(correct_eval_batches)))
        append_csv(summary_csv, metrics, header)
        if wandb_run is not None:
            wandb_run.log(metrics, step=0)

    for ep in range(1, args.epochs + 1):
        print(f"\n===== Epoch {ep} =====")
        ref_grads = mean_grad_correct(weights, correct_ref_batches, loss_grad_fn) if args.use_freeze else None
        loss_sum = 0.0
        for batch in tqdm(wrong_batches, desc=f"train wrong ep{ep}"):
            weights, opt_state, loss = train_step_fn(weights, opt_state, array_batch(batch), ref_grads)
            loss_sum += float(loss)
        train_loss = loss_sum / len(wrong_batches)
        w_ok = infer_batches(weights, wrong_batches, predict_logits_fn, cand_ids, str(jsonl_dir / f"infer_wrong_ep{ep}.jsonl"), f"infer wrong ep{ep}")
        c_ok = infer_batches(weights, correct_eval_batches, predict_logits_fn, cand_ids, str(jsonl_dir / f"infer_correct_ep{ep}.jsonl"), f"infer correct ep{ep}")
        metrics = annotate_metrics(build_metrics(ep, train_loss, w_ok, c_ok, len(wrong_batches), len(correct_eval_batches)))
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
