#!/usr/bin/env bash
set -euo pipefail

# Practical A-then-B sweep for Qwen3-0.6B.
#
# Compares:
#   sft       = train A, then plain finetune on B
#   cnl       = train A, then CNL on B using real A as B-stage retention
#   cnl_synth = train A, then CNL on B using synthetic A generated after A
#
# Example:
#   WANDB_PROJECT=cnl-practical bash jax_sft/sweep_qwen3_0_6b_train_a_then_b.sh csqa medqa

A_DATASET="${A_DATASET:-${1:-csqa}}"
B_DATASET="${B_DATASET:-${2:-medqa}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
MODEL_TAG="${MODEL_TAG:-$(printf '%s' "${MODEL_NAME}" | tr '/:' '__')}"

METHODS="${METHODS:-sft cnl cnl_synth}"
A_LRS="${A_LRS:-1e-6}"
B_LRS="${B_LRS:-5e-7 1e-6 2e-6}"
A_EPOCHS_LIST="${A_EPOCHS_LIST:-1}"
B_EPOCHS_LIST="${B_EPOCHS_LIST:-1 2 3}"
A_OPTIMIZERS="${A_OPTIMIZERS:-adamw}"
B_OPTIMIZERS="${B_OPTIMIZERS:-adamw}"
MASK_STAGES="${MASK_STAGES:-update}"

WANDB_PROJECT="${WANDB_PROJECT:-cnl-practical}"
SWEEP_NAME="${SWEEP_NAME:-qwen3-0.6b-a_then_b-${A_DATASET}-to-${B_DATASET}}"
OUT_ROOT="${OUT_ROOT:-jax_ckpts/sweeps/${SWEEP_NAME}}"
MAX_LENGTH="${MAX_LENGTH:-256}"

# Optional smoke caps.
MAX_A_TRAIN="${MAX_A_TRAIN:-}"
MAX_B_TRAIN="${MAX_B_TRAIN:-}"
MAX_A_EVAL="${MAX_A_EVAL:-}"
MAX_B_EVAL="${MAX_B_EVAL:-}"
MAX_B_RETENTION="${MAX_B_RETENTION:-}"
SYNTHETIC_MAX_ROWS="${SYNTHETIC_MAX_ROWS:-${MAX_B_RETENTION}}"

REF_REFRESH_STEPS="${REF_REFRESH_STEPS:-0}"
B_RETENTION_FILTER="${B_RETENTION_FILTER:-none}"
B_TRAIN_FILTER="${B_TRAIN_FILTER:-none}"
B_RETENTION_RATIO="${B_RETENTION_RATIO:-100}"
B_RETENTION_SEED="${B_RETENTION_SEED:-0}"
B_RETENTION_SUBSET_MODE="${B_RETENTION_SUBSET_MODE:-nested}"
SYNTHETIC_LABEL_MODE="${SYNTHETIC_LABEL_MODE:-argmax}"
SYNTHETIC_TEMPERATURE="${SYNTHETIC_TEMPERATURE:-1.0}"
SYNTHETIC_MIN_CONFIDENCE="${SYNTHETIC_MIN_CONFIDENCE:-0.0}"
SYNTHETIC_SEED="${SYNTHETIC_SEED:-0}"

run_one() {
  local method="$1"
  local a_lr="$2"
  local b_lr="$3"
  local a_epochs="$4"
  local b_epochs="$5"
  local a_optimizer="$6"
  local b_optimizer="$7"
  local mask_stage="$8"

  local b_method="cnl"
  local synthetic_b_retention="0"
  local run_mask="${mask_stage}"
  if [[ "${method}" == "sft" ]]; then
    b_method="sft"
    run_mask="none"
  elif [[ "${method}" == "cnl_synth" ]]; then
    synthetic_b_retention="1"
  elif [[ "${method}" != "cnl" ]]; then
    echo "Unknown method: ${method}" >&2
    exit 1
  fi

  local run_name="${SWEEP_NAME}-${method}-alr${a_lr}-blr${b_lr}-aep${a_epochs}-bep${b_epochs}-aopt${a_optimizer}-bopt${b_optimizer}-mask${run_mask}"
  local out_dir="${OUT_ROOT}/${method}_alr${a_lr}_blr${b_lr}_aep${a_epochs}_bep${b_epochs}_aopt${a_optimizer}_bopt${b_optimizer}_mask${run_mask}"

  echo
  echo "================ Practical Sweep Run ================"
  echo "RUN_NAME : ${run_name}"
  echo "METHOD   : ${method}"
  echo "A_LR/EP  : ${a_lr} / ${a_epochs}"
  echo "B_LR/EP  : ${b_lr} / ${b_epochs}"
  echo "A_OPT    : ${a_optimizer}"
  echo "B_OPT    : ${b_optimizer}"
  echo "MASK     : ${run_mask}"
  echo "OUT_DIR  : ${out_dir}"
  echo "====================================================="

  MODEL_NAME="${MODEL_NAME}" \
  MODEL_TAG="${MODEL_TAG}" \
  OUT_ROOT="${OUT_ROOT}" \
  OUT_DIR="${out_dir}" \
  A_EPOCHS="${a_epochs}" \
  A_LR="${a_lr}" \
  A_OPTIMIZER="${a_optimizer}" \
  B_EPOCHS="${b_epochs}" \
  B_LR="${b_lr}" \
  B_OPTIMIZER="${b_optimizer}" \
  B_METHOD="${b_method}" \
  MASK_STAGE="${mask_stage}" \
  B_RETENTION_RATIO="${B_RETENTION_RATIO}" \
  B_RETENTION_SEED="${B_RETENTION_SEED}" \
  B_RETENTION_SUBSET_MODE="${B_RETENTION_SUBSET_MODE}" \
  SYNTHETIC_B_RETENTION="${synthetic_b_retention}" \
  SYNTHETIC_MAX_ROWS="${SYNTHETIC_MAX_ROWS}" \
  SYNTHETIC_LABEL_MODE="${SYNTHETIC_LABEL_MODE}" \
  SYNTHETIC_TEMPERATURE="${SYNTHETIC_TEMPERATURE}" \
  SYNTHETIC_MIN_CONFIDENCE="${SYNTHETIC_MIN_CONFIDENCE}" \
  SYNTHETIC_SEED="${SYNTHETIC_SEED}" \
  REF_REFRESH_STEPS="${REF_REFRESH_STEPS}" \
  B_RETENTION_FILTER="${B_RETENTION_FILTER}" \
  B_TRAIN_FILTER="${B_TRAIN_FILTER}" \
  MAX_LENGTH="${MAX_LENGTH}" \
  MAX_A_TRAIN="${MAX_A_TRAIN}" \
  MAX_B_TRAIN="${MAX_B_TRAIN}" \
  MAX_A_EVAL="${MAX_A_EVAL}" \
  MAX_B_EVAL="${MAX_B_EVAL}" \
  MAX_B_RETENTION="${MAX_B_RETENTION}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_RUN_NAME="${run_name}" \
  bash jax_sft/run_qwen3_0_6b_train_a_then_b.sh "${A_DATASET}" "${B_DATASET}"
}

echo "================ Qwen3 Practical A-then-B Sweep ================"
echo "A_DATASET      : ${A_DATASET}"
echo "B_DATASET      : ${B_DATASET}"
echo "MODEL_NAME     : ${MODEL_NAME}"
echo "METHODS        : ${METHODS}"
echo "A_LRS          : ${A_LRS}"
echo "B_LRS          : ${B_LRS}"
echo "A_EPOCHS_LIST  : ${A_EPOCHS_LIST}"
echo "B_EPOCHS_LIST  : ${B_EPOCHS_LIST}"
echo "A_OPTIMIZERS   : ${A_OPTIMIZERS}"
echo "B_OPTIMIZERS   : ${B_OPTIMIZERS}"
echo "MASK_STAGES    : ${MASK_STAGES}"
echo "B_RET_RATIO    : ${B_RETENTION_RATIO}"
echo "B_RET_SEED     : ${B_RETENTION_SEED}"
echo "B_RET_SUBSET   : ${B_RETENTION_SUBSET_MODE}"
echo "WANDB_PROJECT  : ${WANDB_PROJECT}"
echo "OUT_ROOT       : ${OUT_ROOT}"
echo "==============================================================="

for a_lr in ${A_LRS}; do
  for b_lr in ${B_LRS}; do
    for a_epochs in ${A_EPOCHS_LIST}; do
      for b_epochs in ${B_EPOCHS_LIST}; do
        for a_optimizer in ${A_OPTIMIZERS}; do
          for b_optimizer in ${B_OPTIMIZERS}; do
            for method in ${METHODS}; do
              if [[ "${method}" == "sft" ]]; then
                run_one "${method}" "${a_lr}" "${b_lr}" "${a_epochs}" "${b_epochs}" "${a_optimizer}" "${b_optimizer}" "update"
              else
                for mask_stage in ${MASK_STAGES}; do
                  run_one "${method}" "${a_lr}" "${b_lr}" "${a_epochs}" "${b_epochs}" "${a_optimizer}" "${b_optimizer}" "${mask_stage}"
                done
              fi
            done
          done
        done
      done
    done
  done
done

echo "Practical sweep done."
