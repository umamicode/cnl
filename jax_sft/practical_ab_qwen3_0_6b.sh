#!/usr/bin/env bash
set -euo pipefail

# Practical A-then-B continual-learning sweep.
#
# Experiment:
#   1. Train on old task/data A with SFT.
#   2. Train on new task/data B with SFT, CNL, or CNL using synthetic A.
#   3. Evaluate A retention on the full A eval set and B learning on the full
#      B eval set.
#
# Training policy:
#   - Default: fixed epochs for A and B.
#   - Optional A_TARGET_LOSS: stop A early once A eval loss reaches target.
#   - Optional B_TARGET_LOSS: stop B early once B eval loss reaches target.
#
# Example:
#   bash jax_sft/practical_ab_qwen3_0_6b.sh csqa medqa
#
# Target-A policy example:
#   A_EPOCHS_LIST=10 A_TARGET_LOSS=0.5 \
#   bash jax_sft/practical_ab_qwen3_0_6b.sh csqa medqa

A_DATASET="${1:-csqa}"
B_DATASET="${2:-medqa}"

export WANDB_PROJECT="${WANDB_PROJECT:-cnl-practical}"
export SWEEP_NAME="${SWEEP_NAME:-qwen3-0.6b-a_then_b-${A_DATASET}-to-${B_DATASET}}"
export METHODS="${METHODS:-sft cnl cnl_synth}"

# Start near the learning rates that worked in the reproduce setting and sweep
# a small neighborhood for safety.
LR_GRID="${LR_GRID:-5e-7 1e-6 2e-6 5e-6}"
export A_LRS="${A_LRS:-${LR_GRID}}"
export B_LRS="${B_LRS:-${LR_GRID}}"
export A_EPOCHS_LIST="${A_EPOCHS_LIST:-3}"
export B_EPOCHS_LIST="${B_EPOCHS_LIST:-1 2 3}"
export A_OPTIMIZERS="${A_OPTIMIZERS:-adamw}"
export B_OPTIMIZERS="${B_OPTIMIZERS:-adamw}"
export MASK_STAGES="${MASK_STAGES:-update}"

export A_TARGET_LOSS="${A_TARGET_LOSS:-}"
export B_TARGET_LOSS="${B_TARGET_LOSS:-}"

export B_RETENTION_RATIO="${B_RETENTION_RATIO:-100}"
export B_RETENTION_SEED="${B_RETENTION_SEED:-0}"
export B_RETENTION_SUBSET_MODE="${B_RETENTION_SUBSET_MODE:-nested}"
export B_RETENTION_FILTER="${B_RETENTION_FILTER:-none}"
export B_TRAIN_FILTER="${B_TRAIN_FILTER:-none}"
export REF_REFRESH_STEPS="${REF_REFRESH_STEPS:-0}"

export MAX_LENGTH="${MAX_LENGTH:-256}"

bash jax_sft/_scripts/sweep_qwen3_0_6b_train_a_then_b.sh "${A_DATASET}" "${B_DATASET}"
