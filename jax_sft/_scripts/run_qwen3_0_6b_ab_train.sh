#!/usr/bin/env bash
set -euo pipefail

# Explicit A/B retention-vs-injection experiment for Qwen3-0.6B.
#
# Example:
#   WANDB_PROJECT=cnl-repro bash jax_sft/run_qwen3_0_6b_ab_train.sh csqa medqa
#
# A is the retention/mastered data. B is the new/injection data.

A_DATASET="${A_DATASET:-${1:-csqa}}"
B_DATASET="${B_DATASET:-${2:-medqa}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
MODEL_TAG="${MODEL_TAG:-$(printf '%s' "${MODEL_NAME}" | tr '/:' '__')}"
WEIGHTS_DIR="${WEIGHTS_DIR:-${HOME}/weights}"
PTX_DIR="${PTX_DIR:-}"
DATA_ROOT="${DATA_ROOT:-data}"
OUT_ROOT="${OUT_ROOT:-jax_ckpts/ab}"

METHOD="${METHOD:-cnl}"
USE_FREEZE="${USE_FREEZE:-1}"
if [[ "${METHOD}" == "sft" ]]; then
  USE_FREEZE=0
fi

LR="${LR:-1e-7}"
EPOCHS="${EPOCHS:-1}"
OPTIMIZER="${OPTIMIZER:-sgd}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
MASK_STAGE="${MASK_STAGE:-gradient}"
REF_REFRESH_STEPS="${REF_REFRESH_STEPS:-0}"
MAX_LENGTH="${MAX_LENGTH:-256}"
RETENTION_FILTER="${RETENTION_FILTER:-correct}"
TRAIN_FILTER="${TRAIN_FILTER:-none}"
EVAL_BEFORE_TRAIN="${EVAL_BEFORE_TRAIN:-1}"

MAX_A="${MAX_A:-}"
MAX_B="${MAX_B:-}"
MAX_A_EVAL="${MAX_A_EVAL:-}"
MAX_B_EVAL="${MAX_B_EVAL:-}"

WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3-0.6b-ab-${A_DATASET}-to-${B_DATASET}-${METHOD}}"
WANDB_MODE="${WANDB_MODE:-}"

if [[ -n "${A_JSONLS:-}" ]]; then
  # shellcheck disable=SC2206
  A_FILES=(${A_JSONLS})
elif [[ -f "${DATA_ROOT}/${A_DATASET}_4options.jsonl" ]]; then
  A_FILES=("${DATA_ROOT}/${A_DATASET}_4options.jsonl")
elif [[ -f "${DATA_ROOT}/${A_DATASET}_correct_Qwen2.5-1.5B-Instruct.jsonl" && -f "${DATA_ROOT}/${A_DATASET}_wrong_Qwen2.5-1.5B-Instruct.jsonl" ]]; then
  A_FILES=(
    "${DATA_ROOT}/${A_DATASET}_correct_Qwen2.5-1.5B-Instruct.jsonl"
    "${DATA_ROOT}/${A_DATASET}_wrong_Qwen2.5-1.5B-Instruct.jsonl"
  )
else
  echo "Could not find A data for ${A_DATASET}. Set A_JSONLS='path/a.jsonl path/b.jsonl'."
  exit 1
fi

if [[ -n "${B_JSONLS:-}" ]]; then
  # shellcheck disable=SC2206
  B_FILES=(${B_JSONLS})
elif [[ -f "${DATA_ROOT}/${B_DATASET}_4options.jsonl" ]]; then
  B_FILES=("${DATA_ROOT}/${B_DATASET}_4options.jsonl")
elif [[ -f "${DATA_ROOT}/${B_DATASET}_correct_Qwen2.5-1.5B-Instruct.jsonl" && -f "${DATA_ROOT}/${B_DATASET}_wrong_Qwen2.5-1.5B-Instruct.jsonl" ]]; then
  B_FILES=(
    "${DATA_ROOT}/${B_DATASET}_correct_Qwen2.5-1.5B-Instruct.jsonl"
    "${DATA_ROOT}/${B_DATASET}_wrong_Qwen2.5-1.5B-Instruct.jsonl"
  )
else
  echo "Could not find B data for ${B_DATASET}. Set B_JSONLS='path/a.jsonl path/b.jsonl'."
  exit 1
fi

if [[ -n "${A_EVAL_JSONLS:-}" ]]; then
  # shellcheck disable=SC2206
  A_EVAL_FILES=(${A_EVAL_JSONLS})
else
  A_EVAL_FILES=("${A_FILES[@]}")
fi

if [[ -n "${B_EVAL_JSONLS:-}" ]]; then
  # shellcheck disable=SC2206
  B_EVAL_FILES=(${B_EVAL_JSONLS})
else
  B_EVAL_FILES=("${B_FILES[@]}")
fi

OUT_DIR="${OUT_DIR:-${OUT_ROOT}/${A_DATASET}_to_${B_DATASET}_${MODEL_TAG}_${METHOD}_lr${LR}_ep${EPOCHS}_opt${OPTIMIZER}_mask${MASK_STAGE}}"

echo "================ Qwen3 A/B CNL Train ================"
echo "A_DATASET        : ${A_DATASET}"
echo "B_DATASET        : ${B_DATASET}"
echo "MODEL_NAME       : ${MODEL_NAME}"
echo "METHOD           : ${METHOD}"
echo "USE_FREEZE       : ${USE_FREEZE}"
echo "A_JSONLS         : ${A_FILES[*]}"
echo "B_JSONLS         : ${B_FILES[*]}"
echo "A_EVAL_JSONLS    : ${A_EVAL_FILES[*]}"
echo "B_EVAL_JSONLS    : ${B_EVAL_FILES[*]}"
echo "RETENTION_FILTER : ${RETENTION_FILTER}"
echo "TRAIN_FILTER     : ${TRAIN_FILTER}"
echo "LR               : ${LR}"
echo "EPOCHS           : ${EPOCHS}"
echo "OPTIMIZER        : ${OPTIMIZER}"
echo "WEIGHT_DECAY     : ${WEIGHT_DECAY}"
echo "MASK_STAGE       : ${MASK_STAGE}"
echo "REF_REFRESH_STEPS: ${REF_REFRESH_STEPS}"
echo "OUT_DIR          : ${OUT_DIR}"
echo "======================================================"

FLAGS=(
  --weights_dir "${WEIGHTS_DIR}"
  --model_name "${MODEL_NAME}"
  --retention_jsonl "${A_FILES[@]}"
  --train_jsonl "${B_FILES[@]}"
  --retention_eval_jsonl "${A_EVAL_FILES[@]}"
  --train_eval_jsonl "${B_EVAL_FILES[@]}"
  --out_dir "${OUT_DIR}"
  --retention_filter "${RETENTION_FILTER}"
  --train_filter "${TRAIN_FILTER}"
  --optimizer "${OPTIMIZER}"
  --weight_decay "${WEIGHT_DECAY}"
  --lr "${LR}"
  --epochs "${EPOCHS}"
  --use_freeze "${USE_FREEZE}"
  --mask_stage "${MASK_STAGE}"
  --ref_refresh_steps "${REF_REFRESH_STEPS}"
  --max_length "${MAX_LENGTH}"
  --eval_before_train "${EVAL_BEFORE_TRAIN}"
)

if [[ -n "${PTX_DIR}" ]]; then
  FLAGS+=(--ptx_dir "${PTX_DIR}")
fi
if [[ -n "${MAX_A}" ]]; then
  FLAGS+=(--max_retention "${MAX_A}")
fi
if [[ -n "${MAX_B}" ]]; then
  FLAGS+=(--max_train "${MAX_B}")
fi
if [[ -n "${MAX_A_EVAL}" ]]; then
  FLAGS+=(--max_retention_eval "${MAX_A_EVAL}")
fi
if [[ -n "${MAX_B_EVAL}" ]]; then
  FLAGS+=(--max_train_eval "${MAX_B_EVAL}")
fi
if [[ -n "${WANDB_PROJECT}" ]]; then
  FLAGS+=(--wandb_project "${WANDB_PROJECT}" --wandb_run_name "${WANDB_RUN_NAME}")
fi
if [[ -n "${WANDB_ENTITY}" ]]; then
  FLAGS+=(--wandb_entity "${WANDB_ENTITY}")
fi
if [[ -n "${WANDB_MODE}" ]]; then
  FLAGS+=(--wandb_mode "${WANDB_MODE}")
fi

python jax_sft/qwen3_ptx_ab_train.py "${FLAGS[@]}"

echo "A/B train done."
