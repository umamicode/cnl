#!/usr/bin/env bash
set -euo pipefail

# Paper-style reproduce-setting search for relaxed CNL variants only.
#
# This is intentionally separate from repro_search_qwen3_0_6b.sh so a running
# SFT/CNL/CNL-synth sweep can continue undisturbed.
#
# Methods:
#   cnl_margin      : real mastered data + margin relaxation
#   cnl_leaky       : real mastered data + leaky relaxation
#   cnl_margin_synth: synthetic mastered data + margin relaxation
#   cnl_leaky_synth : synthetic mastered data + leaky relaxation
#
# Example:
#   bash jax_sft/repro_search_relaxed_qwen3_0_6b.sh csqa

DATASET="${1:-csqa}"
export DATA_ROOT="${DATA_ROOT:-data}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
export MODEL_TAG="${MODEL_TAG:-$(printf '%s' "${MODEL_NAME}" | tr '/:' '__')}"

export WANDB_PROJECT="${WANDB_PROJECT:-cnl-repro-synth}"
export SWEEP_NAME="${SWEEP_NAME:-qwen3-0.6b-${DATASET}-relaxed-paper-hparam-sweep}"
export METHODS="${METHODS:-cnl_margin cnl_leaky cnl_margin_synth cnl_leaky_synth}"
export SYNTHETIC_CORRECT_MODE="${SYNTHETIC_CORRECT_MODE:-random}"
export SYNTHETIC_CORRECT_N="${SYNTHETIC_CORRECT_N:-512}"
export SYNTHETIC_CORRECT_SIZE_MATCHES="${SYNTHETIC_CORRECT_SIZE_MATCHES:-correct wrong}"

# Keep relaxed runs from rewriting the split files used by any currently
# running reproduce sweep.
export OUT_CORRECT="${OUT_CORRECT:-${DATA_ROOT}/${DATASET}_correct_${MODEL_TAG}_relaxed.jsonl}"
export OUT_WRONG="${OUT_WRONG:-${DATA_ROOT}/${DATASET}_wrong_${MODEL_TAG}_relaxed.jsonl}"

export LRS="${LRS:-1e-9 2e-9 1e-7 2e-7 1e-6 2e-6}"
export EPOCHS_LIST="${EPOCHS_LIST:-25}"
export OPTIMIZERS="${OPTIMIZERS:-adamw sgd}"
export MASK_STAGES="${MASK_STAGES:-update}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-1}"

# Real relaxed variants are 48 runs. Including synthetic-correct/wrong
# variants makes the full default 144 runs.
export CNL_MARGINS="${CNL_MARGINS:-1e-12 1e-10}"
export CNL_LEAKS="${CNL_LEAKS:-0.01 0.05}"

export MAX_LENGTH="${MAX_LENGTH:-256}"

bash jax_sft/_scripts/sweep_qwen3_0_6b.sh "${DATASET}"
