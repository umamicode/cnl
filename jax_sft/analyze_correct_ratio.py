#!/usr/bin/env python
"""Analyze the CNL correct/replay-ratio sweep.

This script answers a deliberately matched question:

  Holding the training hyperparameters fixed, how does CNL change as the
  correct/replay ratio changes?

It can read an existing CSV export or fetch all runs from W&B. The W&B path is
the most useful because normal W&B chart exports often omit SFT rows or config
fields.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import os
import re
from pathlib import Path
from typing import Any


RUN_RE = re.compile(
    r"(?P<prefix>.*?)-(?P<method>cnl|sft)-cr(?P<ratio>\d+)-seed(?P<seed>\d+)"
    r"-lr(?P<lr>.+)-ep(?P<epochs>\d+)-opt(?P<optimizer>.+)-mask(?P<mask_stage>.+)$"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="umamicode/cnl-repro-correct-ratio")
    p.add_argument(
        "--csv",
        nargs="*",
        default=None,
        help="Optional W&B CSV export(s) to read instead of fetching runs. Multiple files are concatenated.",
    )
    p.add_argument("--out_dir", default="outputs/correct_ratio_analysis")
    p.add_argument(
        "--select",
        choices=["hmean", "sum", "correct_then_wrong", "wrong_then_correct"],
        default="hmean",
        help="How to select the best full-replay CNL hyperparameter.",
    )
    p.add_argument("--full_ratio", type=float, default=100.0)
    p.add_argument("--min_correct", type=float, default=0.0)
    p.add_argument("--min_wrong", type=float, default=0.0)
    return p.parse_args()


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def nested_get(dct: dict[str, Any], key: str) -> Any:
    if key in dct:
        return dct[key]
    if key.startswith("final/"):
        alt = key.replace("final/", "", 1)
        return dct.get(alt)
    return None


def parse_name(name: str) -> dict[str, Any]:
    match = RUN_RE.search(name or "")
    if not match:
        return {}
    out = match.groupdict()
    out["correct_ratio"] = float(out.pop("ratio"))
    out["correct_seed"] = int(out.pop("seed"))
    out["epochs"] = int(out["epochs"])
    return out


def row_from_parts(name: str, summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any] | None:
    parsed = parse_name(name)
    method = config.get("method") or parsed.get("method")
    if not method:
        use_freeze = config.get("use_freeze")
        method = "cnl" if str(use_freeze).lower() in {"1", "true"} else "sft"

    correct_acc = as_float(nested_get(summary, "final/correct_accuracy"))
    wrong_acc = as_float(nested_get(summary, "final/wrong_accuracy"))
    if correct_acc is None or wrong_acc is None:
        return None

    ratio = as_float(config.get("correct_ratio"))
    if ratio is None:
        ratio = as_float(summary.get("correct_ratio"))
    if ratio is None:
        ratio = parsed.get("correct_ratio")
    if ratio is not None and ratio <= 1.0:
        ratio *= 100.0

    lr = config.get("lr", parsed.get("lr"))
    epochs = config.get("epochs", parsed.get("epochs"))
    optimizer = config.get("optimizer", parsed.get("optimizer"))
    mask_stage = config.get("mask_stage", parsed.get("mask_stage"))
    if method == "sft":
        mask_stage = "none"

    return {
        "name": name,
        "method": method,
        "correct_ratio": ratio,
        "correct_seed": config.get("correct_seed", parsed.get("correct_seed", 0)),
        "lr": str(lr),
        "epochs": int(epochs) if epochs not in (None, "") else None,
        "optimizer": str(optimizer),
        "mask_stage": str(mask_stage),
        "correct_accuracy": correct_acc,
        "wrong_accuracy": wrong_acc,
        "forgetting_rate": 1.0 - correct_acc,
    }


def fetch_wandb(project: str) -> list[dict[str, Any]]:
    import wandb

    api = wandb.Api()
    rows = []
    for run in api.runs(project):
        summary = dict(run.summary._json_dict)
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        row = row_from_parts(run.name, summary, config)
        if row is not None:
            rows.append(row)
    return rows


def read_csv(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            name = raw.get("Name") or raw.get("name") or ""
            summary: dict[str, Any] = dict(raw)
            config: dict[str, Any] = dict(raw)
            if "summary" in raw:
                try:
                    summary.update(ast.literal_eval(raw["summary"]))
                except (SyntaxError, ValueError):
                    pass
            if "config" in raw:
                try:
                    config.update(ast.literal_eval(raw["config"]))
                except (SyntaxError, ValueError):
                    pass
            row = row_from_parts(name, summary, config)
            if row is not None:
                rows.append(row)
    return rows


def read_csvs(paths: list[str]) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for path in paths:
        for row in read_csv(path):
            key = row["name"]
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def score(row: dict[str, Any], mode: str) -> tuple[float, ...]:
    c = row["correct_accuracy"]
    w = row["wrong_accuracy"]
    if mode == "hmean":
        h = 0.0 if c + w == 0 else 2 * c * w / (c + w)
        return (h, c, w)
    if mode == "sum":
        return (c + w, c, w)
    if mode == "correct_then_wrong":
        return (c, w)
    if mode == "wrong_then_correct":
        return (w, c)
    raise ValueError(mode)


def same_hparams_key(row: dict[str, Any], include_mask: bool = True) -> tuple[Any, ...]:
    base = (row["lr"], row["epochs"], row["optimizer"])
    if include_mask:
        return base + (row["mask_stage"],)
    return base


def select_full_replay(rows: list[dict[str, Any]], full_ratio: float, mode: str, min_correct: float, min_wrong: float) -> dict[str, Any]:
    candidates = [
        r
        for r in rows
        if r["method"] == "cnl"
        and r["correct_ratio"] is not None
        and math.isclose(float(r["correct_ratio"]), full_ratio)
        and r["correct_accuracy"] >= min_correct
        and r["wrong_accuracy"] >= min_wrong
    ]
    if not candidates:
        raise ValueError(f"No full-ratio CNL rows found for correct_ratio={full_ratio}.")
    return max(candidates, key=lambda r: score(r, mode))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = [
        "name",
        "method",
        "correct_ratio",
        "correct_seed",
        "lr",
        "epochs",
        "optimizer",
        "mask_stage",
        "correct_accuracy",
        "wrong_accuracy",
        "forgetting_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_matched(rows: list[dict[str, Any]], selected: dict[str, Any], out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    cnl_rows = [
        r
        for r in rows
        if r["method"] == "cnl"
        and same_hparams_key(r, include_mask=True) == same_hparams_key(selected, include_mask=True)
    ]
    cnl_rows = sorted(cnl_rows, key=lambda r: float(r["correct_ratio"]))

    sft_rows = [
        r
        for r in rows
        if r["method"] == "sft"
        and same_hparams_key(r, include_mask=False) == same_hparams_key(selected, include_mask=False)
    ]
    sft = max(sft_rows, key=lambda r: score(r, "hmean")) if sft_rows else None

    paths = []
    title_hp = (
        f"lr={selected['lr']}, epochs={selected['epochs']}, "
        f"opt={selected['optimizer']}, mask={selected['mask_stage']}"
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=180)
    ratios = [r["correct_ratio"] for r in cnl_rows]
    ax.plot(ratios, [r["correct_accuracy"] for r in cnl_rows], marker="o", linewidth=2.2, label="CNL correct_accuracy")
    ax.plot(ratios, [r["wrong_accuracy"] for r in cnl_rows], marker="o", linewidth=2.2, label="CNL wrong_accuracy")
    if sft is not None:
        ax.axhline(sft["correct_accuracy"], color="#1f2937", linestyle="--", linewidth=1.6, label="SFT correct_accuracy")
        ax.axhline(sft["wrong_accuracy"], color="#6b7280", linestyle="--", linewidth=1.6, label="SFT wrong_accuracy")
    ax.set_title("Matched Hyperparameters: Replay Ratio Ablation", fontsize=15, weight="bold", pad=12)
    ax.text(
        0.5,
        0.965,
        title_hp,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
        color="#4b5563",
        bbox={"facecolor": "white", "edgecolor": "#e5e7eb", "alpha": 0.85, "pad": 3},
    )
    ax.set_xlabel("correct/replay ratio (%)")
    ax.set_ylabel("final accuracy")
    ax.set_xticks(ratios)
    ax.set_ylim(0, 1.03)
    ax.grid(True, color="#d8dde8", alpha=0.85)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", loc="lower right")
    fig.tight_layout()
    path = out_dir / "matched_hparams_replay_ratio_lines.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7.2, 6.3), dpi=180)
    scatter = ax.scatter(
        [r["correct_accuracy"] for r in cnl_rows],
        [r["wrong_accuracy"] for r in cnl_rows],
        c=[r["correct_ratio"] for r in cnl_rows],
        s=92,
        cmap="plasma",
        edgecolors="white",
        linewidths=0.8,
        label="CNL ratios",
    )
    for r in cnl_rows:
        ax.annotate(f"{int(r['correct_ratio'])}%", (r["correct_accuracy"], r["wrong_accuracy"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    if sft is not None:
        ax.scatter([sft["correct_accuracy"]], [sft["wrong_accuracy"]], marker="*", s=230, color="#111827", edgecolors="white", linewidths=0.9, label="SFT matched")
    ax.set_title("Matched Hyperparameters: Pareto View", fontsize=15, weight="bold", pad=12)
    ax.text(
        0.5,
        0.965,
        title_hp,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
        color="#4b5563",
        bbox={"facecolor": "white", "edgecolor": "#e5e7eb", "alpha": 0.85, "pad": 3},
    )
    ax.set_xlabel("final/correct_accuracy")
    ax.set_ylabel("final/wrong_accuracy")
    ax.set_xlim(max(0, min([r["correct_accuracy"] for r in cnl_rows] + ([sft["correct_accuracy"]] if sft else [])) - 0.05), 1.04)
    ax.set_ylim(max(0, min([r["wrong_accuracy"] for r in cnl_rows] + ([sft["wrong_accuracy"]] if sft else [])) - 0.05), 1.01)
    ax.grid(True, color="#d8dde8", alpha=0.85)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", loc="lower left")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("correct/replay ratio (%)")
    fig.tight_layout()
    path = out_dir / "matched_hparams_replay_ratio_pareto.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csvs(args.csv) if args.csv else fetch_wandb(args.project)
    if not rows:
        raise ValueError("No usable rows found.")

    all_path = out_dir / "correct_ratio_runs_flat.csv"
    write_csv(all_path, rows)

    selected = select_full_replay(rows, args.full_ratio, args.select, args.min_correct, args.min_wrong)
    matched = [
        r
        for r in rows
        if (
            r["method"] == "cnl"
            and same_hparams_key(r, include_mask=True) == same_hparams_key(selected, include_mask=True)
        )
        or (
            r["method"] == "sft"
            and same_hparams_key(r, include_mask=False) == same_hparams_key(selected, include_mask=False)
        )
    ]
    matched_path = out_dir / "matched_hparams_rows.csv"
    write_csv(matched_path, matched)
    plot_paths = plot_matched(rows, selected, out_dir)

    print("Selected full-replay CNL hyperparameter:")
    print(
        f"  lr={selected['lr']} epochs={selected['epochs']} optimizer={selected['optimizer']} "
        f"mask_stage={selected['mask_stage']} correct_ratio={selected['correct_ratio']}"
    )
    print(
        f"  correct_accuracy={selected['correct_accuracy']:.6f} "
        f"wrong_accuracy={selected['wrong_accuracy']:.6f}"
    )
    print(f"Wrote: {all_path}")
    print(f"Wrote: {matched_path}")
    for path in plot_paths:
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
