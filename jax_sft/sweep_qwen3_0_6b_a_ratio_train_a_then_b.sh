#!/usr/bin/env bash
set -euo pipefail

# Practical A-then-B sweep over how much real A data CNL can use during B.
#
# Stage A always trains on A first. A_RETENTION_RATIOS controls the random
# subset of A available as the B-stage CNL retention/reference set.
#
# Methods:
#   sft = train A, then plain finetune on B
#   cnl = train A, then CNL on B using a random ratio of real A

A_DATASET="${A_DATASET:-${1:-csqa}}"
B_DATASET="${B_DATASET:-${2:-medqa}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
MODEL_TAG="${MODEL_TAG:-$(printf '%s' "${MODEL_NAME}" | tr '/:' '__')}"

METHODS="${METHODS:-sft cnl}"
A_RETENTION_RATIOS="${A_RETENTION_RATIOS:-10 20 40 60 80 100}"
A_RETENTION_SEEDS="${A_RETENTION_SEEDS:-0}"

LR_GRID="${LR_GRID:-5e-7 1e-6 2e-6}"
A_LRS="${A_LRS:-${LR_GRID}}"
B_LRS="${B_LRS:-${LR_GRID}}"
A_EPOCHS_LIST="${A_EPOCHS_LIST:-3}"
B_EPOCHS_LIST="${B_EPOCHS_LIST:-1 3}"
A_OPTIMIZERS="${A_OPTIMIZERS:-adamw}"
B_OPTIMIZERS="${B_OPTIMIZERS:-adamw}"
MASK_STAGES="${MASK_STAGES:-update}"

# By default, SFT does not rerun for every A-retention ratio because the ratio
# is unused by plain B-stage finetuning. Set to 1 if you want matched duplicate
# SFT runs per ratio for plotting convenience.
RUN_SFT_PER_RATIO="${RUN_SFT_PER_RATIO:-0}"

WANDB_PROJECT="${WANDB_PROJECT:-cnl-practical-a-ratio}"
SWEEP_NAME="${SWEEP_NAME:-qwen3-0.6b-a_ratio-${A_DATASET}-to-${B_DATASET}}"
OUT_ROOT="${OUT_ROOT:-jax_ckpts/sweeps/${SWEEP_NAME}}"
MAX_LENGTH="${MAX_LENGTH:-256}"

# Optional smoke caps.
MAX_A_TRAIN="${MAX_A_TRAIN:-}"
MAX_B_TRAIN="${MAX_B_TRAIN:-}"
MAX_A_EVAL="${MAX_A_EVAL:-}"
MAX_B_EVAL="${MAX_B_EVAL:-}"
MAX_B_RETENTION="${MAX_B_RETENTION:-}"

REF_REFRESH_STEPS="${REF_REFRESH_STEPS:-0}"
B_RETENTION_FILTER="${B_RETENTION_FILTER:-none}"
B_TRAIN_FILTER="${B_TRAIN_FILTER:-none}"

run_one() {
  local method="$1"
  local ratio="$2"
  local seed="$3"
  local a_lr="$4"
  local b_lr="$5"
  local a_epochs="$6"
  local b_epochs="$7"
  local a_optimizer="$8"
  local b_optimizer="$9"
  local mask_stage="${10}"

  local b_method="cnl"
  local run_mask="${mask_stage}"
  if [[ "${method}" == "sft" ]]; then
    b_method="sft"
    run_mask="none"
  elif [[ "${method}" != "cnl" ]]; then
    echo "Unknown method: ${method}" >&2
    exit 1
  fi

  local run_name="${SWEEP_NAME}-${method}-ar${ratio}-seed${seed}-alr${a_lr}-blr${b_lr}-aep${a_epochs}-bep${b_epochs}-aopt${a_optimizer}-bopt${b_optimizer}-mask${run_mask}"
  local out_dir="${OUT_ROOT}/${method}_ar${ratio}_seed${seed}_alr${a_lr}_blr${b_lr}_aep${a_epochs}_bep${b_epochs}_aopt${a_optimizer}_bopt${b_optimizer}_mask${run_mask}"

  echo
  echo "================ A-Ratio Sweep Run ================"
  echo "RUN_NAME : ${run_name}"
  echo "METHOD   : ${method}"
  echo "A_RATIO  : ${ratio}"
  echo "SEED     : ${seed}"
  echo "A_LR/EP  : ${a_lr} / ${a_epochs}"
  echo "B_LR/EP  : ${b_lr} / ${b_epochs}"
  echo "A_OPT    : ${a_optimizer}"
  echo "B_OPT    : ${b_optimizer}"
  echo "MASK     : ${run_mask}"
  echo "OUT_DIR  : ${out_dir}"
  echo "==================================================="

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
  B_RETENTION_RATIO="${ratio}" \
  B_RETENTION_SEED="${seed}" \
  SYNTHETIC_B_RETENTION="0" \
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

echo "================ Qwen3 A-Ratio Practical Sweep ================"
echo "A_DATASET         : ${A_DATASET}"
echo "B_DATASET         : ${B_DATASET}"
echo "MODEL_NAME        : ${MODEL_NAME}"
echo "METHODS           : ${METHODS}"
echo "A_RETENTION_RATIOS: ${A_RETENTION_RATIOS}"
echo "A_RETENTION_SEEDS : ${A_RETENTION_SEEDS}"
echo "A_LRS             : ${A_LRS}"
echo "B_LRS             : ${B_LRS}"
echo "A_EPOCHS_LIST     : ${A_EPOCHS_LIST}"
echo "B_EPOCHS_LIST     : ${B_EPOCHS_LIST}"
echo "A_OPTIMIZERS      : ${A_OPTIMIZERS}"
echo "B_OPTIMIZERS      : ${B_OPTIMIZERS}"
echo "MASK_STAGES       : ${MASK_STAGES}"
echo "RUN_SFT_PER_RATIO : ${RUN_SFT_PER_RATIO}"
echo "WANDB_PROJECT     : ${WANDB_PROJECT}"
echo "OUT_ROOT          : ${OUT_ROOT}"
echo "==============================================================="

for a_lr in ${A_LRS}; do
  for b_lr in ${B_LRS}; do
    for a_epochs in ${A_EPOCHS_LIST}; do
      for b_epochs in ${B_EPOCHS_LIST}; do
        for a_optimizer in ${A_OPTIMIZERS}; do
          for b_optimizer in ${B_OPTIMIZERS}; do
            for method in ${METHODS}; do
              if [[ "${method}" == "sft" && "${RUN_SFT_PER_RATIO}" != "1" ]]; then
                run_one "sft" "100" "0" "${a_lr}" "${b_lr}" "${a_epochs}" "${b_epochs}" "${a_optimizer}" "${b_optimizer}" "update"
              elif [[ "${method}" == "sft" ]]; then
                for ratio in ${A_RETENTION_RATIOS}; do
                  for seed in ${A_RETENTION_SEEDS}; do
                    run_one "sft" "${ratio}" "${seed}" "${a_lr}" "${b_lr}" "${a_epochs}" "${b_epochs}" "${a_optimizer}" "${b_optimizer}" "update"
                  done
                done
              elif [[ "${method}" == "cnl" ]]; then
                for ratio in ${A_RETENTION_RATIOS}; do
                  for seed in ${A_RETENTION_SEEDS}; do
                    for mask_stage in ${MASK_STAGES}; do
                      run_one "cnl" "${ratio}" "${seed}" "${a_lr}" "${b_lr}" "${a_epochs}" "${b_epochs}" "${a_optimizer}" "${b_optimizer}" "${mask_stage}"
                    done
                  done
                done
              else
                echo "Unknown method: ${method}" >&2
                exit 1
              fi
            done
          done
        done
      done
    done
  done
done

echo "A-ratio practical sweep done."
