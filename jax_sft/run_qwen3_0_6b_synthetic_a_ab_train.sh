#!/usr/bin/env bash
set -euo pipefail

# Synthetic-A A/B experiment for Qwen3-0.6B.
#
# Step 1: create synthetic retention rows by pseudo-labeling a prompt bank with
#         the frozen base model.
# Step 2: train on B while using synthetic A as the CNL retention set.
#
# If A_EVAL_JSONLS is provided, evaluation still measures real A retention.
# Otherwise the synthetic A rows are also used for A evaluation.

A_DATASET="${A_DATASET:-${1:-csqa}}"
B_DATASET="${B_DATASET:-${2:-medqa}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
MODEL_TAG="${MODEL_TAG:-$(printf '%s' "${MODEL_NAME}" | tr '/:' '__')}"
DATA_ROOT="${DATA_ROOT:-data}"
OUT_ROOT="${OUT_ROOT:-jax_ckpts/synthetic_a_ab}"
SYNTH_ROOT="${SYNTH_ROOT:-${DATA_ROOT}/synthetic}"

SYNTH_MAX_ROWS="${SYNTH_MAX_ROWS:-512}"
SYNTH_LABEL_MODE="${SYNTH_LABEL_MODE:-argmax}"
SYNTH_TEMPERATURE="${SYNTH_TEMPERATURE:-1.0}"
SYNTH_MIN_CONFIDENCE="${SYNTH_MIN_CONFIDENCE:-0.0}"

if [[ -n "${SYNTH_SOURCE_JSONLS:-}" ]]; then
  # shellcheck disable=SC2206
  SYNTH_SOURCE_FILES=(${SYNTH_SOURCE_JSONLS})
elif [[ -f "${DATA_ROOT}/all_correct_Qwen2.5-1.5B-Instruct.jsonl" && -f "${DATA_ROOT}/all_wrong_Qwen2.5-1.5B-Instruct.jsonl" ]]; then
  SYNTH_SOURCE_FILES=(
    "${DATA_ROOT}/all_correct_Qwen2.5-1.5B-Instruct.jsonl"
    "${DATA_ROOT}/all_wrong_Qwen2.5-1.5B-Instruct.jsonl"
  )
elif [[ -f "${DATA_ROOT}/${A_DATASET}_correct_Qwen2.5-1.5B-Instruct.jsonl" && -f "${DATA_ROOT}/${A_DATASET}_wrong_Qwen2.5-1.5B-Instruct.jsonl" ]]; then
  SYNTH_SOURCE_FILES=(
    "${DATA_ROOT}/${A_DATASET}_correct_Qwen2.5-1.5B-Instruct.jsonl"
    "${DATA_ROOT}/${A_DATASET}_wrong_Qwen2.5-1.5B-Instruct.jsonl"
  )
else
  echo "Could not find a prompt bank for synthetic A. Set SYNTH_SOURCE_JSONLS='path/a.jsonl path/b.jsonl'."
  exit 1
fi

mkdir -p "${SYNTH_ROOT}"
SYNTH_A_JSONL="${SYNTH_A_JSONL:-${SYNTH_ROOT}/${A_DATASET}_syntheticA_${MODEL_TAG}_${SYNTH_LABEL_MODE}_n${SYNTH_MAX_ROWS}.jsonl}"

echo "================ Qwen3 Synthetic A ================"
echo "A_DATASET          : ${A_DATASET}"
echo "B_DATASET          : ${B_DATASET}"
echo "MODEL_NAME         : ${MODEL_NAME}"
echo "SYNTH_SOURCE_JSONLS: ${SYNTH_SOURCE_FILES[*]}"
echo "SYNTH_A_JSONL      : ${SYNTH_A_JSONL}"
echo "SYNTH_MAX_ROWS     : ${SYNTH_MAX_ROWS}"
echo "SYNTH_LABEL_MODE   : ${SYNTH_LABEL_MODE}"
echo "==================================================="

SYNTH_FLAGS=(
  --weights_dir "${WEIGHTS_DIR:-${HOME}/weights}"
  --model_name "${MODEL_NAME}"
  --source_jsonl "${SYNTH_SOURCE_FILES[@]}"
  --out_jsonl "${SYNTH_A_JSONL}"
  --max_rows "${SYNTH_MAX_ROWS}"
  --max_length "${MAX_LENGTH:-256}"
  --label_mode "${SYNTH_LABEL_MODE}"
  --temperature "${SYNTH_TEMPERATURE}"
  --min_confidence "${SYNTH_MIN_CONFIDENCE}"
)
if [[ -n "${PTX_DIR:-}" ]]; then
  SYNTH_FLAGS+=(--ptx_dir "${PTX_DIR}")
fi

python jax_sft/qwen3_ptx_make_synthetic_a.py "${SYNTH_FLAGS[@]}"

METHOD="${METHOD:-cnl}" \
A_DATASET="${A_DATASET}_synthetic" \
B_DATASET="${B_DATASET}" \
A_JSONLS="${SYNTH_A_JSONL}" \
RETENTION_FILTER="${RETENTION_FILTER:-none}" \
TRAIN_FILTER="${TRAIN_FILTER:-none}" \
OUT_ROOT="${OUT_ROOT}" \
WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3-0.6b-synthA-${A_DATASET}-to-${B_DATASET}-${METHOD:-cnl}}" \
bash jax_sft/run_qwen3_0_6b_ab_train.sh "${A_DATASET}_synthetic" "${B_DATASET}"

echo "Synthetic-A A/B run done."
