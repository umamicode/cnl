#!/usr/bin/env bash
set -euo pipefail

# Synthetic-reference reproduction sweep.
#
# Compares:
#   sft           = train wrong/injection set with no CNL
#   cnl           = CNL with the actual mastered/correct set
#   cnl_synth     = memory-free CNL; generates synthetic MCQ prompts, then pseudo-labels them
#   cnl_synth_icl = saves a few mastered examples as ICL generation demos,
#                   generates same-format prompts across varied topics, then pseudo-labels them
#
# Example:
#   bash jax_sft/repro_synth_qwen3_0_6b.sh csqa

DATASET="${1:-csqa}"

export WANDB_PROJECT="${WANDB_PROJECT:-cnl-repro-synth}"
export SWEEP_NAME="${SWEEP_NAME:-qwen3-0.6b-${DATASET}-synth-reference-sweep}"
export METHODS="${METHODS:-sft cnl cnl_synth cnl_synth_icl}"

export SYNTHETIC_CORRECT_N="${SYNTHETIC_CORRECT_N:-512}"
export SYNTHETIC_CORRECT_SIZE_MATCHES="${SYNTHETIC_CORRECT_SIZE_MATCHES:-correct wrong}"
export SYNTHETIC_ICL_EXAMPLES="${SYNTHETIC_ICL_EXAMPLES:-4}"
export SYNTH_LABEL_MODE="${SYNTH_LABEL_MODE:-argmax}"
export SYNTH_TEMPERATURES="${SYNTH_TEMPERATURES:-0.7 1.0}"
export SYNTHETIC_GENERATION_MAX_LENGTH="${SYNTHETIC_GENERATION_MAX_LENGTH:-768}"
export SYNTHETIC_GENERATION_MAX_NEW_TOKENS="${SYNTHETIC_GENERATION_MAX_NEW_TOKENS:-192}"
export SYNTHETIC_GENERATION_BATCH_SIZE="${SYNTHETIC_GENERATION_BATCH_SIZE:-8}"
export SYNTHETIC_GENERATION_RETRIES="${SYNTHETIC_GENERATION_RETRIES:-3}"
export SYNTH_MIN_CONFIDENCE="${SYNTH_MIN_CONFIDENCE:-0.0}"
export SYNTH_SEED="${SYNTH_SEED:-0}"

export LRS="${LRS:-1e-7 2e-7 1e-6 2e-6}"
export EPOCHS_LIST="${EPOCHS_LIST:-25}"
export OPTIMIZERS="${OPTIMIZERS:-adamw}"
export MASK_STAGES="${MASK_STAGES:-update}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-1}"
export MAX_LENGTH="${MAX_LENGTH:-256}"
export CORRECT_EVAL_SCOPE="${CORRECT_EVAL_SCOPE:-all}"

bash jax_sft/_scripts/sweep_qwen3_0_6b.sh "${DATASET}"
