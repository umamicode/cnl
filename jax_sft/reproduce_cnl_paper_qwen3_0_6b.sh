#!/usr/bin/env bash
set -euo pipefail

# Paper-style CNL reproduction track on Qwen3-0.6B.
#
# This follows the CNL paper's correct/wrong experiment shape:
#   M = examples the model initially answers correctly
#   I = examples the model initially answers incorrectly
#   train on I, protect M with CNL
#
# It is "paper-style" rather than exact-paper by default because it uses
# Qwen/Qwen3-0.6B and the vendored JAX/PTX backend. Override MODEL_NAME and
# data paths if you want a different model/backend.

DATASETS="${DATASETS:-csqa medqa arc_c mmlu}"
METHODS="${METHODS:-cnl sft}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
MODEL_TAG="${MODEL_TAG:-$(printf '%s' "${MODEL_NAME}" | tr '/:' '__')}"
OUT_ROOT="${OUT_ROOT:-jax_ckpts/paper_style_qwen3_0_6b}"

LR="${LR:-1e-7}"
EPOCHS="${EPOCHS:-25}"
OPTIMIZER="${OPTIMIZER:-sgd}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
MASK_STAGE="${MASK_STAGE:-gradient}"
MAX_LENGTH="${MAX_LENGTH:-256}"

WANDB_PROJECT="${WANDB_PROJECT:-cnl-repro}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-}"

# Optional smoke caps.
MAX_ROWS="${MAX_ROWS:-}"
MAX_WRONG="${MAX_WRONG:-}"
MAX_CORRECT="${MAX_CORRECT:-}"

echo "================ Qwen3 Paper-Style CNL ================"
echo "DATASETS      : ${DATASETS}"
echo "METHODS       : ${METHODS}"
echo "MODEL_NAME    : ${MODEL_NAME}"
echo "LR            : ${LR}"
echo "EPOCHS        : ${EPOCHS}"
echo "OPTIMIZER     : ${OPTIMIZER}"
echo "WEIGHT_DECAY  : ${WEIGHT_DECAY}"
echo "MASK_STAGE    : ${MASK_STAGE}"
echo "OUT_ROOT      : ${OUT_ROOT}"
echo "WANDB_PROJECT : ${WANDB_PROJECT}"
echo "======================================================="

for dataset in ${DATASETS}; do
  first_for_dataset=1
  for method in ${METHODS}; do
    use_freeze=1
    run_mask="${MASK_STAGE}"
    if [[ "${method}" == "sft" ]]; then
      use_freeze=0
      run_mask="none"
    elif [[ "${method}" != "cnl" ]]; then
      echo "Unknown method: ${method}" >&2
      exit 1
    fi

    run_name="paper-style-qwen3-${dataset}-${method}-lr${LR}-ep${EPOCHS}-opt${OPTIMIZER}-mask${run_mask}"
    out_dir="${OUT_ROOT}/${dataset}/${method}_lr${LR}_ep${EPOCHS}_opt${OPTIMIZER}_mask${run_mask}"
    skip_split=$([[ "${first_for_dataset}" == "1" ]] && echo 0 || echo 1)

    echo
    echo "================ Paper-Style Run ================"
    echo "DATASET   : ${dataset}"
    echo "METHOD    : ${method}"
    echo "RUN_NAME  : ${run_name}"
    echo "OUT_DIR   : ${out_dir}"
    echo "SKIP_SPLIT: ${skip_split}"
    echo "==============================================="

    MODEL_NAME="${MODEL_NAME}" \
    MODEL_TAG="${MODEL_TAG}" \
    OUT_ROOT="${OUT_ROOT}" \
    OUT_DIR="${out_dir}" \
    LR="${LR}" \
    EPOCHS="${EPOCHS}" \
    OPTIMIZER="${OPTIMIZER}" \
    WEIGHT_DECAY="${WEIGHT_DECAY}" \
    MASK_STAGE="${MASK_STAGE}" \
    MAX_LENGTH="${MAX_LENGTH}" \
    USE_FREEZE="${use_freeze}" \
    SKIP_SPLIT="${skip_split}" \
    MAX_ROWS="${MAX_ROWS}" \
    MAX_WRONG="${MAX_WRONG}" \
    MAX_CORRECT="${MAX_CORRECT}" \
    WANDB_PROJECT="${WANDB_PROJECT}" \
    WANDB_RUN_NAME="${run_name}" \
    bash jax_sft/run_qwen3_0_6b_split_train.sh "${dataset}"

    first_for_dataset=0
  done
done

echo "Paper-style runs done."
