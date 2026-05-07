#!/usr/bin/env bash
set -euo pipefail

# Matched CNL/baseline sweep for Qwen3-0.6B.
#
# Example:
#   WANDB_PROJECT=cnl-repro bash jax_sft/sweep_qwen3_0_6b.sh csqa
#
# Smoke sweep:
#   MAX_ROWS=64 MAX_WRONG=32 MAX_CORRECT=32 \
#   LRS="1e-8 5e-8" EPOCHS_LIST="1" \
#   WANDB_PROJECT=cnl-repro bash jax_sft/sweep_qwen3_0_6b.sh csqa

DATASET="${DATASET:-${1:-csqa}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
MODEL_TAG="${MODEL_TAG:-$(printf '%s' "${MODEL_NAME}" | tr '/:' '__')}"

LRS="${LRS:-1e-8 2e-8 5e-8 1e-7}"
EPOCHS_LIST="${EPOCHS_LIST:-1}"
OPTIMIZERS="${OPTIMIZERS:-sgd}"
MASK_STAGES="${MASK_STAGES:-gradient update}"
METHODS="${METHODS:-cnl sft}"

WANDB_PROJECT="${WANDB_PROJECT:-cnl-repro}"
SWEEP_NAME="${SWEEP_NAME:-qwen3-0.6b-${DATASET}-sweep}"
OUT_ROOT="${OUT_ROOT:-jax_ckpts/sweeps/${SWEEP_NAME}}"
DATA_ROOT="${DATA_ROOT:-data}"
MAX_LENGTH="${MAX_LENGTH:-256}"

OUT_CORRECT="${OUT_CORRECT:-${DATA_ROOT}/${DATASET}_correct_${MODEL_TAG}.jsonl}"
OUT_WRONG="${OUT_WRONG:-${DATA_ROOT}/${DATASET}_wrong_${MODEL_TAG}.jsonl}"

run_one() {
  local method="$1"
  local lr="$2"
  local epochs="$3"
  local optimizer="$4"
  local mask_stage="$5"
  local use_freeze="1"
  local run_mask="${mask_stage}"

  if [[ "${method}" == "sft" ]]; then
    use_freeze="0"
    run_mask="none"
  fi

  local run_name="${SWEEP_NAME}-${method}-lr${lr}-ep${epochs}-opt${optimizer}-mask${run_mask}"
  local out_dir="${OUT_ROOT}/${method}_lr${lr}_ep${epochs}_opt${optimizer}_mask${run_mask}"

  echo
  echo "================ Sweep Run ================"
  echo "RUN_NAME  : ${run_name}"
  echo "METHOD    : ${method}"
  echo "LR        : ${lr}"
  echo "EPOCHS    : ${epochs}"
  echo "OPTIMIZER : ${optimizer}"
  echo "MASK      : ${run_mask}"
  echo "OUT_DIR   : ${out_dir}"
  echo "==========================================="

  MODEL_NAME="${MODEL_NAME}" \
  DATA_ROOT="${DATA_ROOT}" \
  OUT_CORRECT="${OUT_CORRECT}" \
  OUT_WRONG="${OUT_WRONG}" \
  OUT_DIR="${out_dir}" \
  OUT_ROOT="${OUT_ROOT}" \
  LR="${lr}" \
  EPOCHS="${epochs}" \
  OPTIMIZER="${optimizer}" \
  MASK_STAGE="${mask_stage}" \
  USE_FREEZE="${use_freeze}" \
  MAX_LENGTH="${MAX_LENGTH}" \
  SKIP_SPLIT="${SKIP_SPLIT_FOR_RUN}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_RUN_NAME="${run_name}" \
  bash jax_sft/run_qwen3_0_6b_split_train.sh "${DATASET}"
}

echo "================ Qwen3 CNL Sweep ================"
echo "DATASET       : ${DATASET}"
echo "MODEL_NAME    : ${MODEL_NAME}"
echo "LRS           : ${LRS}"
echo "EPOCHS_LIST   : ${EPOCHS_LIST}"
echo "OPTIMIZERS    : ${OPTIMIZERS}"
echo "MASK_STAGES   : ${MASK_STAGES}"
echo "METHODS       : ${METHODS}"
echo "WANDB_PROJECT : ${WANDB_PROJECT}"
echo "OUT_ROOT      : ${OUT_ROOT}"
echo "================================================="

first_run=1
for lr in ${LRS}; do
  for epochs in ${EPOCHS_LIST}; do
    for optimizer in ${OPTIMIZERS}; do
      for method in ${METHODS}; do
        if [[ "${method}" == "cnl" ]]; then
          for mask_stage in ${MASK_STAGES}; do
            SKIP_SPLIT_FOR_RUN=$([[ "${first_run}" == "1" ]] && echo 0 || echo 1)
            run_one "${method}" "${lr}" "${epochs}" "${optimizer}" "${mask_stage}"
            first_run=0
          done
        elif [[ "${method}" == "sft" ]]; then
          SKIP_SPLIT_FOR_RUN=$([[ "${first_run}" == "1" ]] && echo 0 || echo 1)
          run_one "${method}" "${lr}" "${epochs}" "${optimizer}" "gradient"
          first_run=0
        else
          echo "Unknown method: ${method}" >&2
          exit 1
        fi
      done
    done
  done
done

echo "Sweep done."
