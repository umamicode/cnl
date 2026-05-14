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
# Defaults are anchored on the CNL full-data settings that previously reached
# the top-right corner in the reproduce search:
#
#   cnl-cr100-lr1e-7-ep25-optadamw-maskupdate
#   cnl-cr100-lr2e-7-ep25-optadamw-maskupdate
#   cnl-cr100-lr1e-6-ep25-optadamw-maskupdate
#   cnl-cr100-lr2e-6-ep25-optadamw-maskupdate
#
# The ratio order starts at 100 so each known full-data candidate appears
# immediately, followed by smaller reference-data ratios for the same
# hyperparameter setting.
#
# Example:
#   bash jax_sft/repro_ratio_qwen3_0_6b.sh csqa

DATASET="${1:-csqa}"

export WANDB_PROJECT="${WANDB_PROJECT:-cnl-repro-ratio-candidate-hparams}"
export SWEEP_NAME="${SWEEP_NAME:-qwen3-0.6b-${DATASET}-ratio-candidate-hparams}"
export METHODS="${METHODS:-cnl sft}"
export SYNTHETIC_CORRECT_MODE="${SYNTHETIC_CORRECT_MODE:-random}"
export SYNTHETIC_CORRECT_N="${SYNTHETIC_CORRECT_N:-512}"

export LRS="${LRS:-1e-7 2e-7 1e-6 2e-6}"
export EPOCHS_LIST="${EPOCHS_LIST:-25}"
export OPTIMIZERS="${OPTIMIZERS:-adamw}"
export MASK_STAGES="${MASK_STAGES:-update}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-1}"

export CORRECT_RATIOS="${CORRECT_RATIOS:-100 80 60 40 20 10}"
export CORRECT_SEEDS="${CORRECT_SEEDS:-0}"
export CORRECT_SUBSET_MODE="${CORRECT_SUBSET_MODE:-nested}"
export CORRECT_EVAL_SCOPE="${CORRECT_EVAL_SCOPE:-all}"
export RUN_SFT_PER_RATIO="${RUN_SFT_PER_RATIO:-0}"

export MAX_LENGTH="${MAX_LENGTH:-256}"

bash jax_sft/_scripts/sweep_qwen3_0_6b_correct_ratio.sh "${DATASET}"
