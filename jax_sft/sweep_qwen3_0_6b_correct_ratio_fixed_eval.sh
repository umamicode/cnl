#!/usr/bin/env bash
set -euo pipefail

# Confirmation sweep for the correct/mastered-data ratio question.
#
# Unlike the broader correct-ratio sweep, this keeps the retention evaluation
# fixed on the full initially-correct set. correct_ratio only changes the CNL
# reference-gradient subset. Subsets are nested by default: for a given seed,
# 20% contains the same examples as 10%, 40% contains 20%, and so on.
#
# Example:
#   WANDB_PROJECT=cnl-repro-correct-ratio-fixed-eval \
#   bash jax_sft/sweep_qwen3_0_6b_correct_ratio_fixed_eval.sh csqa

DATASET="${1:-csqa}"

export WANDB_PROJECT="${WANDB_PROJECT:-cnl-repro-correct-ratio-fixed-eval}"
export SWEEP_NAME="${SWEEP_NAME:-qwen3-0.6b-correct-ratio-fixed-eval}"
export METHODS="${METHODS:-cnl sft}"

export CORRECT_RATIOS="${CORRECT_RATIOS:-10 20 40 60 80 100}"
export CORRECT_SEEDS="${CORRECT_SEEDS:-0 1 2 3 4}"
export CORRECT_SUBSET_MODE="${CORRECT_SUBSET_MODE:-nested}"
export CORRECT_EVAL_SCOPE="${CORRECT_EVAL_SCOPE:-all}"

# Two focused settings:
# - 5e-7/ep3/adamw/update: best full-reference CNL setting in the new export.
# - 1e-6/ep3/adamw/update: the old near-perfect setting from cnl-repro.
export LRS="${LRS:-5e-7 1e-6}"
export EPOCHS_LIST="${EPOCHS_LIST:-3}"
export OPTIMIZERS="${OPTIMIZERS:-adamw}"
export MASK_STAGES="${MASK_STAGES:-update}"
export RUN_SFT_PER_RATIO="${RUN_SFT_PER_RATIO:-0}"

bash jax_sft/sweep_qwen3_0_6b_correct_ratio.sh "${DATASET}"
