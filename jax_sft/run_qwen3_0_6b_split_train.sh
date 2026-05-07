#!/usr/bin/env bash
set -euo pipefail

# Split data with the exact model, then train CNL on that split.
#
# Defaults target Qwen3-0.6B:
#   MODEL_NAME=Qwen/Qwen3-0.6B
#
# Important: this wrapper uses jax_sft/infer_split_optax.py and
# jax_sft/sft_optax.py. Those scripts currently load models through
# transformers.FlaxAutoModelForCausalLM. If your transformers/flax stack does
# not expose a Flax Qwen3 causal-LM implementation, this script will stop at
# the model-loading step. In that case, wire jax_sft/cnl.py into a Qwen3-capable
# JAX backend such as EasyDeL or MaxText, then keep this pipeline shape.

DATASET="${DATASET:-${1:-csqa}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
MODEL_TAG="${MODEL_TAG:-$(printf '%s' "${MODEL_NAME}" | tr '/:' '__')}"

LR="${LR:-1e-7}"
EPOCHS="${EPOCHS:-1}"
OPTIMIZER="${OPTIMIZER:-sgd}"
MASK_STAGE="${MASK_STAGE:-gradient}"
MAX_LENGTH="${MAX_LENGTH:-256}"
USE_FREEZE="${USE_FREEZE:-1}"
OUT_ROOT="${OUT_ROOT:-jax_ckpts}"
DATA_ROOT="${DATA_ROOT:-data}"

TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
FROM_PT="${FROM_PT:-0}"
EVAL_BEFORE_TRAIN="${EVAL_BEFORE_TRAIN:-1}"

# Optional smoke-test caps.
MAX_ROWS="${MAX_ROWS:-}"
MAX_WRONG="${MAX_WRONG:-}"
MAX_CORRECT="${MAX_CORRECT:-}"

# Optional W&B settings.
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${MODEL_TAG}_${DATASET}_cnl}"
WANDB_MODE="${WANDB_MODE:-}"

if [[ -n "${SOURCE_JSONLS:-}" ]]; then
  # shellcheck disable=SC2206
  SOURCE_FILES=(${SOURCE_JSONLS})
elif [[ -f "${DATA_ROOT}/${DATASET}_4options.jsonl" ]]; then
  SOURCE_FILES=("${DATA_ROOT}/${DATASET}_4options.jsonl")
elif [[ -f "${DATA_ROOT}/${DATASET}_correct_Qwen2.5-1.5B-Instruct.jsonl" && -f "${DATA_ROOT}/${DATASET}_wrong_Qwen2.5-1.5B-Instruct.jsonl" ]]; then
  SOURCE_FILES=(
    "${DATA_ROOT}/${DATASET}_correct_Qwen2.5-1.5B-Instruct.jsonl"
    "${DATA_ROOT}/${DATASET}_wrong_Qwen2.5-1.5B-Instruct.jsonl"
  )
else
  echo "Could not find source data for ${DATASET}."
  echo "Set SOURCE_JSONLS='path/a.jsonl path/b.jsonl' or provide ${DATA_ROOT}/${DATASET}_4options.jsonl."
  exit 1
fi

OUT_CORRECT="${OUT_CORRECT:-${DATA_ROOT}/${DATASET}_correct_${MODEL_TAG}.jsonl}"
OUT_WRONG="${OUT_WRONG:-${DATA_ROOT}/${DATASET}_wrong_${MODEL_TAG}.jsonl}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/${DATASET}_${MODEL_TAG}_cnl_lr${LR}_freeze${USE_FREEZE}}"

COMMON_MODEL_FLAGS=()
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  COMMON_MODEL_FLAGS+=(--trust_remote_code)
fi
if [[ "${FROM_PT}" == "1" ]]; then
  COMMON_MODEL_FLAGS+=(--from_pt)
fi

SPLIT_FLAGS=(
  --model_name "${MODEL_NAME}"
  --jsonl "${SOURCE_FILES[@]}"
  --out_correct_jsonl "${OUT_CORRECT}"
  --out_wrong_jsonl "${OUT_WRONG}"
  --max_length "${MAX_LENGTH}"
  "${COMMON_MODEL_FLAGS[@]}"
)
if [[ -n "${MAX_ROWS}" ]]; then
  SPLIT_FLAGS+=(--max_rows "${MAX_ROWS}")
fi

TRAIN_FLAGS=(
  --model_name "${MODEL_NAME}"
  --wrong_jsonl "${OUT_WRONG}"
  --correct_jsonl "${OUT_CORRECT}"
  --out_dir "${OUT_DIR}"
  --optimizer "${OPTIMIZER}"
  --lr "${LR}"
  --epochs "${EPOCHS}"
  --use_freeze "${USE_FREEZE}"
  --mask_stage "${MASK_STAGE}"
  --max_length "${MAX_LENGTH}"
  --eval_before_train "${EVAL_BEFORE_TRAIN}"
  "${COMMON_MODEL_FLAGS[@]}"
)
if [[ -n "${MAX_WRONG}" ]]; then
  TRAIN_FLAGS+=(--max_wrong "${MAX_WRONG}")
fi
if [[ -n "${MAX_CORRECT}" ]]; then
  TRAIN_FLAGS+=(--max_correct "${MAX_CORRECT}")
fi
if [[ -n "${WANDB_PROJECT}" ]]; then
  TRAIN_FLAGS+=(--wandb_project "${WANDB_PROJECT}" --wandb_run_name "${WANDB_RUN_NAME}")
fi
if [[ -n "${WANDB_ENTITY}" ]]; then
  TRAIN_FLAGS+=(--wandb_entity "${WANDB_ENTITY}")
fi
if [[ -n "${WANDB_MODE}" ]]; then
  TRAIN_FLAGS+=(--wandb_mode "${WANDB_MODE}")
fi

echo "================ Qwen3 Split + Train ================"
echo "DATASET       : ${DATASET}"
echo "MODEL_NAME    : ${MODEL_NAME}"
echo "MODEL_TAG     : ${MODEL_TAG}"
echo "SOURCE_JSONLS : ${SOURCE_FILES[*]}"
echo "OUT_CORRECT   : ${OUT_CORRECT}"
echo "OUT_WRONG     : ${OUT_WRONG}"
echo "OUT_DIR       : ${OUT_DIR}"
echo "LR            : ${LR}"
echo "EPOCHS        : ${EPOCHS}"
echo "USE_FREEZE    : ${USE_FREEZE}"
echo "OPTIMIZER     : ${OPTIMIZER}"
echo "MASK_STAGE    : ${MASK_STAGE}"
echo "MAX_LENGTH    : ${MAX_LENGTH}"
echo "====================================================="

python jax_sft/infer_split_optax.py "${SPLIT_FLAGS[@]}"
python jax_sft/sft_optax.py "${TRAIN_FLAGS[@]}"

echo "Split + train done."

