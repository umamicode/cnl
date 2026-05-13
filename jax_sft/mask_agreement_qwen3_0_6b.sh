#!/usr/bin/env bash
set -euo pipefail

# Standalone real-vs-synthetic CNL mask agreement diagnostic.
#
# This does not train and does not modify the sweep scripts. It loads the same
# Qwen3/PTX backend, computes a real mastered-set reference gradient, computes a
# synthetic reference gradient, and compares the masks induced on wrong examples.
#
# Example:
#   WANDB_PROJECT=cnl-mask-agreement bash jax_sft/mask_agreement_qwen3_0_6b.sh csqa

DATASET="${DATASET:-${1:-csqa}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
MODEL_TAG="${MODEL_TAG:-$(printf '%s' "${MODEL_NAME}" | tr '/:' '__')}"
WEIGHTS_DIR="${WEIGHTS_DIR:-${HOME}/weights}"
DATA_ROOT="${DATA_ROOT:-data}"
OUT_ROOT="${OUT_ROOT:-jax_ckpts/analysis/mask-agreement}"
PTX_DIR="${PTX_DIR:-}"

LR="${LR:-1e-7}"
OPTIMIZER="${OPTIMIZER:-adamw}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-1}"
MASK_STAGE="${MASK_STAGE:-update}"
AGREEMENT_MARGIN="${AGREEMENT_MARGIN:-0.0}"
OPTIMIZER_STATE_POLICY="${OPTIMIZER_STATE_POLICY:-scan}"
MAX_LENGTH="${MAX_LENGTH:-256}"
MAX_ROWS="${MAX_ROWS:-}"
MAX_CORRECT="${MAX_CORRECT:-}"
MAX_WRONG="${MAX_WRONG:-}"
MAX_WRONG_BATCHES="${MAX_WRONG_BATCHES:-64}"
CORRECT_RATIO="${CORRECT_RATIO:-100}"
CORRECT_SEED="${CORRECT_SEED:-0}"
CORRECT_SUBSET_MODE="${CORRECT_SUBSET_MODE:-nested}"

SYNTHETIC_CORRECT_JSONLS="${SYNTHETIC_CORRECT_JSONLS:-}"
SYNTHETIC_CORRECT_SOURCE_JSONLS="${SYNTHETIC_CORRECT_SOURCE_JSONLS:-}"
SYNTHETIC_CORRECT_MODE="${SYNTHETIC_CORRECT_MODE:-random}"
SYNTHETIC_CORRECT_N="${SYNTHETIC_CORRECT_N:-512}"
SYNTHETIC_CORRECT_SIZE_MATCH="${SYNTHETIC_CORRECT_SIZE_MATCH:-correct}"
SYNTHETIC_CORRECT_MAX_ROWS="${SYNTHETIC_CORRECT_MAX_ROWS:-}"
SYNTHETIC_LABEL_MODE="${SYNTHETIC_LABEL_MODE:-argmax}"
SYNTHETIC_TEMPERATURE="${SYNTHETIC_TEMPERATURE:-1.0}"
SYNTHETIC_MIN_CONFIDENCE="${SYNTHETIC_MIN_CONFIDENCE:-0.0}"
SYNTHETIC_SEED="${SYNTHETIC_SEED:-0}"

WANDB_PROJECT="${WANDB_PROJECT:-cnl-mask-agreement}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
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

SKIP_SPLIT="${SKIP_SPLIT:-auto}"
if [[ "${SKIP_SPLIT}" == "auto" ]]; then
  if [[ -f "${OUT_CORRECT}" && -f "${OUT_WRONG}" ]]; then
    SKIP_SPLIT="1"
  else
    SKIP_SPLIT="0"
  fi
fi

OUT_DIR="${OUT_DIR:-${OUT_ROOT}/${DATASET}_${MODEL_TAG}_real_vs_synth_${SYNTHETIC_CORRECT_SIZE_MATCH}_lr${LR}_${OPTIMIZER}_${MASK_STAGE}}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-mask-agreement-${DATASET}-${SYNTHETIC_CORRECT_SIZE_MATCH}-lr${LR}-${OPTIMIZER}-${MASK_STAGE}}"

echo "================ Qwen3 Mask Agreement ================"
echo "DATASET          : ${DATASET}"
echo "MODEL_NAME       : ${MODEL_NAME}"
echo "SOURCE_JSONLS    : ${SOURCE_FILES[*]}"
echo "OUT_CORRECT      : ${OUT_CORRECT}"
echo "OUT_WRONG        : ${OUT_WRONG}"
echo "OUT_DIR          : ${OUT_DIR}"
echo "SKIP_SPLIT       : ${SKIP_SPLIT}"
echo "OPTIMIZER/LR     : ${OPTIMIZER} / ${LR}"
echo "MASK_STAGE       : ${MASK_STAGE}"
echo "STATE_POLICY     : ${OPTIMIZER_STATE_POLICY}"
echo "MAX_WRONG_BATCHES: ${MAX_WRONG_BATCHES}"
echo "SYNTH_MODE       : ${SYNTHETIC_CORRECT_MODE}"
echo "SYNTH_SIZE       : ${SYNTHETIC_CORRECT_SIZE_MATCH}"
echo "WANDB_PROJECT    : ${WANDB_PROJECT}"
echo "======================================================"

FLAGS=(
  --weights_dir "${WEIGHTS_DIR}"
  --model_name "${MODEL_NAME}"
  --source_jsonl "${SOURCE_FILES[@]}"
  --out_correct_jsonl "${OUT_CORRECT}"
  --out_wrong_jsonl "${OUT_WRONG}"
  --out_dir "${OUT_DIR}"
  --optimizer "${OPTIMIZER}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --mask_stage "${MASK_STAGE}"
  --agreement_margin "${AGREEMENT_MARGIN}"
  --optimizer_state_policy "${OPTIMIZER_STATE_POLICY}"
  --max_length "${MAX_LENGTH}"
  --max_wrong_batches "${MAX_WRONG_BATCHES}"
  --correct_ratio "${CORRECT_RATIO}"
  --correct_seed "${CORRECT_SEED}"
  --correct_subset_mode "${CORRECT_SUBSET_MODE}"
  --synthetic_correct_mode "${SYNTHETIC_CORRECT_MODE}"
  --synthetic_correct_n "${SYNTHETIC_CORRECT_N}"
  --synthetic_correct_size_match "${SYNTHETIC_CORRECT_SIZE_MATCH}"
  --synthetic_label_mode "${SYNTHETIC_LABEL_MODE}"
  --synthetic_temperature "${SYNTHETIC_TEMPERATURE}"
  --synthetic_min_confidence "${SYNTHETIC_MIN_CONFIDENCE}"
  --synthetic_seed "${SYNTHETIC_SEED}"
)

if [[ "${SKIP_SPLIT}" == "1" ]]; then
  FLAGS+=(--skip_split)
fi
if [[ -n "${MAX_ROWS}" ]]; then
  FLAGS+=(--max_rows "${MAX_ROWS}")
fi
if [[ -n "${MAX_CORRECT}" ]]; then
  FLAGS+=(--max_correct "${MAX_CORRECT}")
fi
if [[ -n "${MAX_WRONG}" ]]; then
  FLAGS+=(--max_wrong "${MAX_WRONG}")
fi
if [[ -n "${PTX_DIR}" ]]; then
  FLAGS+=(--ptx_dir "${PTX_DIR}")
fi
if [[ -n "${SYNTHETIC_CORRECT_JSONLS}" ]]; then
  # shellcheck disable=SC2206
  SYNTHETIC_CORRECT_FILES=(${SYNTHETIC_CORRECT_JSONLS})
  FLAGS+=(--synthetic_correct_jsonl "${SYNTHETIC_CORRECT_FILES[@]}")
fi
if [[ -n "${SYNTHETIC_CORRECT_SOURCE_JSONLS}" ]]; then
  # shellcheck disable=SC2206
  SYNTHETIC_CORRECT_SOURCE_FILES=(${SYNTHETIC_CORRECT_SOURCE_JSONLS})
  FLAGS+=(--synthetic_correct_source_jsonl "${SYNTHETIC_CORRECT_SOURCE_FILES[@]}")
fi
if [[ -n "${SYNTHETIC_CORRECT_MAX_ROWS}" ]]; then
  FLAGS+=(--synthetic_correct_max_rows "${SYNTHETIC_CORRECT_MAX_ROWS}")
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

python jax_sft/analyze_qwen3_mask_agreement.py "${FLAGS[@]}"

echo "Mask agreement analysis done."
