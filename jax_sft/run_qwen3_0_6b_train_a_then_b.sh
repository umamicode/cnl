#!/usr/bin/env bash
set -euo pipefail

# Practical continual-learning experiment for Qwen3-0.6B:
#   1. Train on A with normal SFT.
#   2. Train on B with CNL or SFT.
#   3. Measure A drop from after-A and B gain from before-B.
#
# Example:
#   WANDB_PROJECT=cnl-practical bash jax_sft/run_qwen3_0_6b_train_a_then_b.sh csqa medqa

A_DATASET="${A_DATASET:-${1:-csqa}}"
B_DATASET="${B_DATASET:-${2:-medqa}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
MODEL_TAG="${MODEL_TAG:-$(printf '%s' "${MODEL_NAME}" | tr '/:' '__')}"
WEIGHTS_DIR="${WEIGHTS_DIR:-${HOME}/weights}"
PTX_DIR="${PTX_DIR:-}"
DATA_ROOT="${DATA_ROOT:-data}"
OUT_ROOT="${OUT_ROOT:-jax_ckpts/a_then_b}"

A_EPOCHS="${A_EPOCHS:-1}"
A_LR="${A_LR:-1e-6}"
A_OPTIMIZER="${A_OPTIMIZER:-adamw}"
A_WEIGHT_DECAY="${A_WEIGHT_DECAY:-1e-4}"

B_EPOCHS="${B_EPOCHS:-1}"
B_LR="${B_LR:-1e-6}"
B_OPTIMIZER="${B_OPTIMIZER:-adamw}"
B_WEIGHT_DECAY="${B_WEIGHT_DECAY:-1e-4}"
B_METHOD="${B_METHOD:-${METHOD:-cnl}}"
MASK_STAGE="${MASK_STAGE:-update}"
REF_REFRESH_STEPS="${REF_REFRESH_STEPS:-0}"
B_RETENTION_FILTER="${B_RETENTION_FILTER:-none}"
B_TRAIN_FILTER="${B_TRAIN_FILTER:-none}"
SYNTHETIC_B_RETENTION="${SYNTHETIC_B_RETENTION:-0}"
SYNTHETIC_MAX_ROWS="${SYNTHETIC_MAX_ROWS:-}"
SYNTHETIC_LABEL_MODE="${SYNTHETIC_LABEL_MODE:-argmax}"
SYNTHETIC_TEMPERATURE="${SYNTHETIC_TEMPERATURE:-1.0}"
SYNTHETIC_MIN_CONFIDENCE="${SYNTHETIC_MIN_CONFIDENCE:-0.0}"
SYNTHETIC_SEED="${SYNTHETIC_SEED:-0}"

MAX_LENGTH="${MAX_LENGTH:-256}"
MAX_A_TRAIN="${MAX_A_TRAIN:-}"
MAX_B_TRAIN="${MAX_B_TRAIN:-}"
MAX_A_EVAL="${MAX_A_EVAL:-}"
MAX_B_EVAL="${MAX_B_EVAL:-}"
MAX_B_RETENTION="${MAX_B_RETENTION:-}"

WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3-0.6b-a_then_b-${A_DATASET}-to-${B_DATASET}-${B_METHOD}}"
WANDB_MODE="${WANDB_MODE:-}"

resolve_dataset() {
  local dataset="$1"
  local env_value="$2"

  if [[ -n "${env_value}" ]]; then
    # shellcheck disable=SC2206
    local files=(${env_value})
    printf '%s\n' "${files[@]}"
  elif [[ -f "${DATA_ROOT}/${dataset}_4options.jsonl" ]]; then
    printf '%s\n' "${DATA_ROOT}/${dataset}_4options.jsonl"
  elif [[ -f "${DATA_ROOT}/${dataset}_correct_Qwen2.5-1.5B-Instruct.jsonl" && -f "${DATA_ROOT}/${dataset}_wrong_Qwen2.5-1.5B-Instruct.jsonl" ]]; then
    printf '%s\n' "${DATA_ROOT}/${dataset}_correct_Qwen2.5-1.5B-Instruct.jsonl"
    printf '%s\n' "${DATA_ROOT}/${dataset}_wrong_Qwen2.5-1.5B-Instruct.jsonl"
  else
    echo "Could not find data for ${dataset}."
    echo "Set the corresponding JSONL env var, e.g. A_JSONLS='path/a.jsonl path/b.jsonl'."
    exit 1
  fi
}

declare -a A_FILES
declare -a B_FILES
declare -a A_EVAL_FILES
declare -a B_EVAL_FILES
declare -a B_RETENTION_FILES
declare -a SYNTHETIC_SOURCE_FILES

while IFS= read -r file; do A_FILES+=("${file}"); done < <(resolve_dataset "${A_DATASET}" "${A_JSONLS:-}")
while IFS= read -r file; do B_FILES+=("${file}"); done < <(resolve_dataset "${B_DATASET}" "${B_JSONLS:-}")

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

if [[ -n "${B_RETENTION_JSONLS:-}" ]]; then
  # shellcheck disable=SC2206
  B_RETENTION_FILES=(${B_RETENTION_JSONLS})
else
  B_RETENTION_FILES=("${A_FILES[@]}")
fi

if [[ -n "${SYNTHETIC_SOURCE_JSONLS:-}" ]]; then
  # shellcheck disable=SC2206
  SYNTHETIC_SOURCE_FILES=(${SYNTHETIC_SOURCE_JSONLS})
fi

OUT_DIR="${OUT_DIR:-${OUT_ROOT}/${A_DATASET}_then_${B_DATASET}_${MODEL_TAG}_${B_METHOD}_a${A_EPOCHS}_b${B_EPOCHS}_blr${B_LR}_opt${B_OPTIMIZER}_mask${MASK_STAGE}}"

echo "================ Qwen3 Train A Then B ================"
echo "A_DATASET         : ${A_DATASET}"
echo "B_DATASET         : ${B_DATASET}"
echo "MODEL_NAME        : ${MODEL_NAME}"
echo "A_JSONLS          : ${A_FILES[*]}"
echo "B_JSONLS          : ${B_FILES[*]}"
echo "A_EVAL_JSONLS     : ${A_EVAL_FILES[*]}"
echo "B_EVAL_JSONLS     : ${B_EVAL_FILES[*]}"
echo "B_RETENTION_JSONLS: ${B_RETENTION_FILES[*]}"
echo "A_EPOCHS/LR/OPT   : ${A_EPOCHS} / ${A_LR} / ${A_OPTIMIZER}"
echo "B_EPOCHS/LR/OPT   : ${B_EPOCHS} / ${B_LR} / ${B_OPTIMIZER}"
echo "B_METHOD          : ${B_METHOD}"
echo "MASK_STAGE        : ${MASK_STAGE}"
echo "B_RETENTION_FILTER: ${B_RETENTION_FILTER}"
echo "B_TRAIN_FILTER    : ${B_TRAIN_FILTER}"
echo "SYNTHETIC_B_RET   : ${SYNTHETIC_B_RETENTION}"
if [[ -n "${SYNTHETIC_SOURCE_JSONLS:-}" ]]; then
  echo "SYNTH_SOURCE_JSONLS: ${SYNTHETIC_SOURCE_FILES[*]}"
fi
echo "REF_REFRESH_STEPS : ${REF_REFRESH_STEPS}"
echo "OUT_DIR           : ${OUT_DIR}"
echo "======================================================"

FLAGS=(
  --weights_dir "${WEIGHTS_DIR}"
  --model_name "${MODEL_NAME}"
  --a_train_jsonl "${A_FILES[@]}"
  --b_train_jsonl "${B_FILES[@]}"
  --a_eval_jsonl "${A_EVAL_FILES[@]}"
  --b_eval_jsonl "${B_EVAL_FILES[@]}"
  --b_retention_jsonl "${B_RETENTION_FILES[@]}"
  --out_dir "${OUT_DIR}"
  --a_epochs "${A_EPOCHS}"
  --a_lr "${A_LR}"
  --a_optimizer "${A_OPTIMIZER}"
  --a_weight_decay "${A_WEIGHT_DECAY}"
  --b_epochs "${B_EPOCHS}"
  --b_lr "${B_LR}"
  --b_optimizer "${B_OPTIMIZER}"
  --b_weight_decay "${B_WEIGHT_DECAY}"
  --b_method "${B_METHOD}"
  --mask_stage "${MASK_STAGE}"
  --b_retention_filter "${B_RETENTION_FILTER}"
  --b_train_filter "${B_TRAIN_FILTER}"
  --synthetic_b_retention "${SYNTHETIC_B_RETENTION}"
  --synthetic_label_mode "${SYNTHETIC_LABEL_MODE}"
  --synthetic_temperature "${SYNTHETIC_TEMPERATURE}"
  --synthetic_min_confidence "${SYNTHETIC_MIN_CONFIDENCE}"
  --synthetic_seed "${SYNTHETIC_SEED}"
  --ref_refresh_steps "${REF_REFRESH_STEPS}"
  --max_length "${MAX_LENGTH}"
)

if [[ -n "${PTX_DIR}" ]]; then
  FLAGS+=(--ptx_dir "${PTX_DIR}")
fi
if [[ -n "${MAX_A_TRAIN}" ]]; then
  FLAGS+=(--max_a_train "${MAX_A_TRAIN}")
fi
if [[ -n "${MAX_B_TRAIN}" ]]; then
  FLAGS+=(--max_b_train "${MAX_B_TRAIN}")
fi
if [[ -n "${MAX_A_EVAL}" ]]; then
  FLAGS+=(--max_a_eval "${MAX_A_EVAL}")
fi
if [[ -n "${MAX_B_EVAL}" ]]; then
  FLAGS+=(--max_b_eval "${MAX_B_EVAL}")
fi
if [[ -n "${MAX_B_RETENTION}" ]]; then
  FLAGS+=(--max_b_retention "${MAX_B_RETENTION}")
fi
if [[ -n "${SYNTHETIC_MAX_ROWS}" ]]; then
  FLAGS+=(--synthetic_max_rows "${SYNTHETIC_MAX_ROWS}")
fi
if [[ -n "${SYNTHETIC_SOURCE_JSONLS:-}" ]]; then
  FLAGS+=(--synthetic_source_jsonl "${SYNTHETIC_SOURCE_FILES[@]}")
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

python jax_sft/qwen3_ptx_train_a_then_b.py "${FLAGS[@]}"

echo "Train A then B done."
