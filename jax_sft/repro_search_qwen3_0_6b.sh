#!/usr/bin/env bash
set -euo pipefail

# Paper-style reproduce-setting hyperparameter search.
#
# Experiment:
#   1. Split one dataset into initially-correct and initially-wrong examples.
#   2. Train on initially-wrong examples.
#   3. Compare CNL against plain SFT on final/correct_accuracy and
#      final/wrong_accuracy.
#
# Defaults follow the paper where the details are specified: 25 epochs and the
# reported SGD learning-rate magnitudes, with a small neighborhood including
# the Qwen3 AdamW settings that previously worked well. This runs on the full
# available split; set MAX_ROWS/MAX_WRONG/MAX_CORRECT only for smoke tests.
#
# Example:
#   bash jax_sft/repro_search_qwen3_0_6b.sh csqa

DATASET="${1:-csqa}"

export WANDB_PROJECT="${WANDB_PROJECT:-cnl-repro-paper-hparams}"
export SWEEP_NAME="${SWEEP_NAME:-qwen3-0.6b-${DATASET}-paper-hparam-sweep}"
export METHODS="${METHODS:-sft cnl cnl_synth}"
export SYNTHETIC_CORRECT_MODE="${SYNTHETIC_CORRECT_MODE:-random}"
export SYNTHETIC_CORRECT_N="${SYNTHETIC_CORRECT_N:-512}"
export SYNTHETIC_CORRECT_SIZE_MATCHES="${SYNTHETIC_CORRECT_SIZE_MATCHES:-correct wrong}"

export LRS="${LRS:-1e-9 2e-9 1e-7 2e-7 1e-6 2e-6}"
export EPOCHS_LIST="${EPOCHS_LIST:-25}"
export OPTIMIZERS="${OPTIMIZERS:-adamw sgd}"
export MASK_STAGES="${MASK_STAGES:-update}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-1}"

export MAX_LENGTH="${MAX_LENGTH:-256}"

bash jax_sft/_scripts/sweep_qwen3_0_6b.sh "${DATASET}"
