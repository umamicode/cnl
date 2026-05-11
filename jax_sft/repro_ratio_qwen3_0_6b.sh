#!/usr/bin/env bash
set -euo pipefail

# Reproduce-setting correct/reference-ratio experiment.
#
# Experiment:
#   1. Split into initially-correct and initially-wrong examples.
#   2. Train on initially-wrong examples.
#   3. Change how much of the initially-correct set CNL can use as its
#      reference-gradient data.
#   4. Always evaluate final/correct_accuracy on the full initially-correct
#      set, so ratios are comparable.
#
# Defaults use the same paper-style hyperparameter neighborhood as the
# reproduce search. Override LRS/EPOCHS_LIST/OPTIMIZERS/MASK_STAGES to anchor
# on a smaller set of selected runs.
#
# Example:
#   bash jax_sft/repro_ratio_qwen3_0_6b.sh csqa

DATASET="${1:-csqa}"

export WANDB_PROJECT="${WANDB_PROJECT:-cnl-repro-ratio-paper-hparams}"
export SWEEP_NAME="${SWEEP_NAME:-qwen3-0.6b-${DATASET}-ratio-paper-hparams}"
export METHODS="${METHODS:-sft cnl cnl_synth}"
export SYNTHETIC_CORRECT_MODE="${SYNTHETIC_CORRECT_MODE:-random}"
export SYNTHETIC_CORRECT_N="${SYNTHETIC_CORRECT_N:-512}"

export LRS="${LRS:-1e-9 2e-9 1e-7 2e-7 1e-6 2e-6}"
export EPOCHS_LIST="${EPOCHS_LIST:-25}"
export OPTIMIZERS="${OPTIMIZERS:-adamw sgd}"
export MASK_STAGES="${MASK_STAGES:-update}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-1}"

export CORRECT_RATIOS="${CORRECT_RATIOS:-10 20 40 60 80 100}"
export CORRECT_SEEDS="${CORRECT_SEEDS:-0}"
export CORRECT_SUBSET_MODE="${CORRECT_SUBSET_MODE:-nested}"
export CORRECT_EVAL_SCOPE="${CORRECT_EVAL_SCOPE:-all}"
export RUN_SFT_PER_RATIO="${RUN_SFT_PER_RATIO:-0}"

export MAX_LENGTH="${MAX_LENGTH:-256}"

bash jax_sft/_scripts/sweep_qwen3_0_6b_correct_ratio.sh "${DATASET}"
