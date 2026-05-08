#!/usr/bin/env bash
set -euo pipefail

# Paper-style correct/mastered-data ratio sweep for Qwen3-0.6B.
#
# This keeps the reproduction setting fixed and only varies how much of the
# initially-correct set CNL can use as its reference set.
#
# Example:
#   WANDB_PROJECT=cnl-repro-correct-ratio \
#   bash jax_sft/sweep_qwen3_0_6b_correct_ratio.sh csqa

DATASETS="${DATASETS:-${1:-csqa}}"
METHODS="${METHODS:-cnl sft}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
MODEL_TAG="${MODEL_TAG:-$(printf '%s' "${MODEL_NAME}" | tr '/:' '__')}"

CORRECT_RATIOS="${CORRECT_RATIOS:-10 20 40 60 80 100}"
CORRECT_SEEDS="${CORRECT_SEEDS:-0}"

LR="${LR:-1e-7}"
EPOCHS="${EPOCHS:-25}"
OPTIMIZER="${OPTIMIZER:-sgd}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
MASK_STAGE="${MASK_STAGE:-gradient}"
MAX_LENGTH="${MAX_LENGTH:-256}"

# By default, SFT runs once because correct_ratio is unused when CNL is off.
# Set RUN_SFT_PER_RATIO=1 for matched duplicate SFT points per ratio.
RUN_SFT_PER_RATIO="${RUN_SFT_PER_RATIO:-0}"

WANDB_PROJECT="${WANDB_PROJECT:-cnl-repro-correct-ratio}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-}"
SWEEP_NAME="${SWEEP_NAME:-qwen3-0.6b-correct-ratio}"
OUT_ROOT="${OUT_ROOT:-jax_ckpts/sweeps/${SWEEP_NAME}}"
DATA_ROOT="${DATA_ROOT:-data}"

# Optional smoke caps.
MAX_ROWS="${MAX_ROWS:-}"
MAX_WRONG="${MAX_WRONG:-}"
MAX_CORRECT="${MAX_CORRECT:-}"

run_one() {
  local dataset="$1"
  local method="$2"
  local ratio="$3"
  local seed="$4"
  local skip_split="$5"

  local use_freeze="1"
  local run_mask="${MASK_STAGE}"
  if [[ "${method}" == "sft" ]]; then
    use_freeze="0"
    run_mask="none"
  elif [[ "${method}" != "cnl" ]]; then
    echo "Unknown method: ${method}" >&2
    exit 1
  fi

  local run_name="${SWEEP_NAME}-${dataset}-${method}-cr${ratio}-seed${seed}-lr${LR}-ep${EPOCHS}-opt${OPTIMIZER}-mask${run_mask}"
  local out_dir="${OUT_ROOT}/${dataset}/${method}_cr${ratio}_seed${seed}_lr${LR}_ep${EPOCHS}_opt${OPTIMIZER}_mask${run_mask}"

  echo
  echo "================ Correct-Ratio Repro Run ================"
  echo "DATASET      : ${dataset}"
  echo "RUN_NAME     : ${run_name}"
  echo "METHOD       : ${method}"
  echo "CORRECT_RATIO: ${ratio}"
  echo "CORRECT_SEED : ${seed}"
  echo "LR/EPOCHS    : ${LR} / ${EPOCHS}"
  echo "OPT/MASK     : ${OPTIMIZER} / ${run_mask}"
  echo "SKIP_SPLIT   : ${skip_split}"
  echo "OUT_DIR      : ${out_dir}"
  echo "========================================================="

  MODEL_NAME="${MODEL_NAME}" \
  MODEL_TAG="${MODEL_TAG}" \
  DATA_ROOT="${DATA_ROOT}" \
  OUT_ROOT="${OUT_ROOT}" \
  OUT_DIR="${out_dir}" \
  LR="${LR}" \
  EPOCHS="${EPOCHS}" \
  OPTIMIZER="${OPTIMIZER}" \
  WEIGHT_DECAY="${WEIGHT_DECAY}" \
  MASK_STAGE="${MASK_STAGE}" \
  MAX_LENGTH="${MAX_LENGTH}" \
  USE_FREEZE="${use_freeze}" \
  CORRECT_RATIO="${ratio}" \
  CORRECT_SEED="${seed}" \
  SKIP_SPLIT="${skip_split}" \
  MAX_ROWS="${MAX_ROWS}" \
  MAX_WRONG="${MAX_WRONG}" \
  MAX_CORRECT="${MAX_CORRECT}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_ENTITY="${WANDB_ENTITY}" \
  WANDB_MODE="${WANDB_MODE}" \
  WANDB_RUN_NAME="${run_name}" \
  bash jax_sft/run_qwen3_0_6b_split_train.sh "${dataset}"
}

echo "================ Qwen3 Correct-Ratio Repro Sweep ================"
echo "DATASETS       : ${DATASETS}"
echo "METHODS        : ${METHODS}"
echo "MODEL_NAME     : ${MODEL_NAME}"
echo "CORRECT_RATIOS : ${CORRECT_RATIOS}"
echo "CORRECT_SEEDS  : ${CORRECT_SEEDS}"
echo "LR             : ${LR}"
echo "EPOCHS         : ${EPOCHS}"
echo "OPTIMIZER      : ${OPTIMIZER}"
echo "WEIGHT_DECAY   : ${WEIGHT_DECAY}"
echo "MASK_STAGE     : ${MASK_STAGE}"
echo "RUN_SFT_PER_RATIO: ${RUN_SFT_PER_RATIO}"
echo "WANDB_PROJECT  : ${WANDB_PROJECT}"
echo "OUT_ROOT       : ${OUT_ROOT}"
echo "================================================================="

for dataset in ${DATASETS}; do
  first_for_dataset=1
  for method in ${METHODS}; do
    if [[ "${method}" == "sft" && "${RUN_SFT_PER_RATIO}" != "1" ]]; then
      skip_split=$([[ "${first_for_dataset}" == "1" ]] && echo 0 || echo 1)
      run_one "${dataset}" "sft" "100" "0" "${skip_split}"
      first_for_dataset=0
    elif [[ "${method}" == "sft" ]]; then
      for ratio in ${CORRECT_RATIOS}; do
        for seed in ${CORRECT_SEEDS}; do
          skip_split=$([[ "${first_for_dataset}" == "1" ]] && echo 0 || echo 1)
          run_one "${dataset}" "sft" "${ratio}" "${seed}" "${skip_split}"
          first_for_dataset=0
        done
      done
    elif [[ "${method}" == "cnl" ]]; then
      for ratio in ${CORRECT_RATIOS}; do
        for seed in ${CORRECT_SEEDS}; do
          skip_split=$([[ "${first_for_dataset}" == "1" ]] && echo 0 || echo 1)
          run_one "${dataset}" "cnl" "${ratio}" "${seed}" "${skip_split}"
          first_for_dataset=0
        done
      done
    else
      echo "Unknown method: ${method}" >&2
      exit 1
    fi
  done
done

echo "Correct-ratio reproduction sweep done."
