#!/usr/bin/env bash
set -euo pipefail

# Paper-style correct/mastered-data ratio sweep for Qwen3-0.6B.
#
# This sweeps normal training hyperparameters plus how much of the initially
# correct/mastered set CNL can use as its reference set. By default, retention
# is always evaluated on the full initially-correct set, so correct_ratio
# changes only CNL's reference data and points are comparable across ratios.
#
# Example:
#   WANDB_PROJECT=cnl-repro-correct-ratio \
#   bash jax_sft/sweep_qwen3_0_6b_correct_ratio.sh csqa

DATASETS="${DATASETS:-${1:-csqa}}"
METHODS="${METHODS:-cnl sft}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
MODEL_TAG="${MODEL_TAG:-$(printf '%s' "${MODEL_NAME}" | tr '/:' '__')}"
SYNTHETIC_CORRECT_SOURCE_JSONLS="${SYNTHETIC_CORRECT_SOURCE_JSONLS:-}"
SYNTHETIC_CORRECT_MODE="${SYNTHETIC_CORRECT_MODE:-random}"
SYNTHETIC_CORRECT_N="${SYNTHETIC_CORRECT_N:-512}"
SYNTHETIC_CORRECT_SIZE_MATCHES="${SYNTHETIC_CORRECT_SIZE_MATCHES:-fixed}"
SYNTHETIC_CORRECT_MAX_ROWS="${SYNTHETIC_CORRECT_MAX_ROWS:-}"
SYNTH_LABEL_MODE="${SYNTH_LABEL_MODE:-argmax}"
SYNTH_TEMPERATURE="${SYNTH_TEMPERATURE:-1.0}"
SYNTH_MIN_CONFIDENCE="${SYNTH_MIN_CONFIDENCE:-0.0}"
SYNTH_SEED="${SYNTH_SEED:-0}"

CORRECT_RATIOS="${CORRECT_RATIOS:-10 20 40 60 80 100}"
CORRECT_SEEDS="${CORRECT_SEEDS:-0}"
CORRECT_SUBSET_MODE="${CORRECT_SUBSET_MODE:-nested}"
CORRECT_EVAL_SCOPE="${CORRECT_EVAL_SCOPE:-all}"

LRS="${LRS:-${LR:-1e-8 2e-8 5e-8 1e-7 2e-7 5e-7 1e-6 2e-6 5e-6 1e-5 2e-5 5e-5 1e-4}}"
EPOCHS_LIST="${EPOCHS_LIST:-${EPOCHS:-1 2 3}}"
OPTIMIZERS="${OPTIMIZERS:-${OPTIMIZER:-adamw sgd}}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
MASK_STAGES="${MASK_STAGES:-${MASK_STAGE:-gradient update}}"
CNL_MARGINS="${CNL_MARGINS:-1e-12 1e-10}"
CNL_LEAKS="${CNL_LEAKS:-0.01 0.05}"
MAX_LENGTH="${MAX_LENGTH:-256}"

# By default, SFT runs once because correct_ratio is unused when CNL is off.
# Set RUN_SFT_PER_RATIO=1 for matched duplicate SFT points per ratio.
RUN_SFT_PER_RATIO="${RUN_SFT_PER_RATIO:-0}"

WANDB_PROJECT="${WANDB_PROJECT:-cnl-repro-correct-ratio-fixed-eval}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-}"
SWEEP_NAME="${SWEEP_NAME:-qwen3-0.6b-correct-ratio-fixed-eval}"
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
  local lr="$5"
  local epochs="$6"
  local optimizer="$7"
  local mask_stage="$8"
  local skip_split="$9"
  local synth_size_match="${10:-fixed}"
  local cnl_margin="${11:-0.0}"
  local cnl_leak="${12:-0.0}"

  local use_freeze="1"
  local run_mask="${mask_stage}"
  local synth_suffix=""
  local cnl_mask_mode="hard"
  local relax_suffix=""
  if [[ "${method}" == "sft" ]]; then
    use_freeze="0"
    run_mask="none"
  elif [[ "${method}" == "cnl_synth" ]]; then
    use_freeze="1"
    synth_suffix="-synth${synth_size_match}"
  elif [[ "${method}" == "cnl_margin" ]]; then
    use_freeze="1"
    cnl_mask_mode="margin"
    relax_suffix="-margin${cnl_margin}"
  elif [[ "${method}" == "cnl_leaky" ]]; then
    use_freeze="1"
    cnl_mask_mode="leaky"
    relax_suffix="-leak${cnl_leak}"
  elif [[ "${method}" == "cnl_margin_synth" ]]; then
    use_freeze="1"
    cnl_mask_mode="margin"
    synth_suffix="-synth${synth_size_match}"
    relax_suffix="-margin${cnl_margin}"
  elif [[ "${method}" == "cnl_leaky_synth" ]]; then
    use_freeze="1"
    cnl_mask_mode="leaky"
    synth_suffix="-synth${synth_size_match}"
    relax_suffix="-leak${cnl_leak}"
  elif [[ "${method}" != "cnl" ]]; then
    echo "Unknown method: ${method}" >&2
    exit 1
  fi

  local run_name="${SWEEP_NAME}-${dataset}-${method}${synth_suffix}${relax_suffix}-cr${ratio}-seed${seed}-lr${lr}-ep${epochs}-opt${optimizer}-mask${run_mask}"
  local out_dir="${OUT_ROOT}/${dataset}/${method}${synth_suffix}${relax_suffix}_cr${ratio}_seed${seed}_lr${lr}_ep${epochs}_opt${optimizer}_mask${run_mask}"

  echo
  echo "================ Correct-Ratio Repro Run ================"
  echo "DATASET      : ${dataset}"
  echo "RUN_NAME     : ${run_name}"
  echo "METHOD       : ${method}"
  echo "CORRECT_RATIO: ${ratio}"
  echo "CORRECT_SEED : ${seed}"
  echo "SYNTH_SIZE   : ${synth_size_match}"
  echo "CNL_MODE     : ${cnl_mask_mode}"
  echo "CNL_MARGIN   : ${cnl_margin}"
  echo "CNL_LEAK     : ${cnl_leak}"
  echo "CORRECT_SUBSET: ${CORRECT_SUBSET_MODE}"
  echo "CORRECT_EVAL  : ${CORRECT_EVAL_SCOPE}"
  echo "LR/EPOCHS    : ${lr} / ${epochs}"
  echo "OPT/MASK     : ${optimizer} / ${run_mask}"
  echo "SKIP_SPLIT   : ${skip_split}"
  echo "OUT_DIR      : ${out_dir}"
  echo "========================================================="

  MODEL_NAME="${MODEL_NAME}" \
  MODEL_TAG="${MODEL_TAG}" \
  DATA_ROOT="${DATA_ROOT}" \
  OUT_ROOT="${OUT_ROOT}" \
  OUT_DIR="${out_dir}" \
  LR="${lr}" \
  EPOCHS="${epochs}" \
  OPTIMIZER="${optimizer}" \
  METHOD_NAME="${method}" \
  WEIGHT_DECAY="${WEIGHT_DECAY}" \
  MASK_STAGE="${mask_stage}" \
  CNL_MASK_MODE="${cnl_mask_mode}" \
  CNL_MARGIN="${cnl_margin}" \
  CNL_LEAK="${cnl_leak}" \
  MAX_LENGTH="${MAX_LENGTH}" \
  USE_FREEZE="${use_freeze}" \
  CORRECT_RATIO="${ratio}" \
  CORRECT_SEED="${seed}" \
  CORRECT_SUBSET_MODE="${CORRECT_SUBSET_MODE}" \
  CORRECT_EVAL_SCOPE="${CORRECT_EVAL_SCOPE}" \
  SKIP_SPLIT="${skip_split}" \
  MAX_ROWS="${MAX_ROWS}" \
  MAX_WRONG="${MAX_WRONG}" \
  MAX_CORRECT="${MAX_CORRECT}" \
  SYNTHETIC_CORRECT_SOURCE_JSONLS="${SYNTHETIC_CORRECT_SOURCE_JSONLS}" \
  SYNTHETIC_CORRECT_MODE="${SYNTHETIC_CORRECT_MODE}" \
  SYNTHETIC_CORRECT_N="${SYNTHETIC_CORRECT_N}" \
  SYNTHETIC_CORRECT_SIZE_MATCH="${synth_size_match}" \
  SYNTHETIC_CORRECT_MAX_ROWS="${SYNTHETIC_CORRECT_MAX_ROWS}" \
  SYNTHETIC_LABEL_MODE="${SYNTH_LABEL_MODE}" \
  SYNTHETIC_TEMPERATURE="${SYNTH_TEMPERATURE}" \
  SYNTHETIC_MIN_CONFIDENCE="${SYNTH_MIN_CONFIDENCE}" \
  SYNTHETIC_SEED="${SYNTH_SEED}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_RUN_NAME="${run_name}" \
  bash jax_sft/_scripts/run_qwen3_0_6b_split_train.sh "${dataset}"
}

echo "================ Qwen3 Correct-Ratio Repro Sweep ================"
echo "DATASETS       : ${DATASETS}"
echo "METHODS        : ${METHODS}"
echo "MODEL_NAME     : ${MODEL_NAME}"
echo "SYNTH_MODE     : ${SYNTHETIC_CORRECT_MODE}"
echo "SYNTH_SOURCE   : ${SYNTHETIC_CORRECT_SOURCE_JSONLS:-source_jsonl}"
echo "SYNTH_N        : ${SYNTHETIC_CORRECT_N}"
echo "SYNTH_SIZES    : ${SYNTHETIC_CORRECT_SIZE_MATCHES}"
echo "CORRECT_RATIOS : ${CORRECT_RATIOS}"
echo "CORRECT_SEEDS  : ${CORRECT_SEEDS}"
echo "CORRECT_SUBSET : ${CORRECT_SUBSET_MODE}"
echo "CORRECT_EVAL   : ${CORRECT_EVAL_SCOPE}"
echo "LRS            : ${LRS}"
echo "EPOCHS_LIST    : ${EPOCHS_LIST}"
echo "OPTIMIZERS     : ${OPTIMIZERS}"
echo "WEIGHT_DECAY   : ${WEIGHT_DECAY}"
echo "MASK_STAGES    : ${MASK_STAGES}"
echo "CNL_MARGINS    : ${CNL_MARGINS}"
echo "CNL_LEAKS      : ${CNL_LEAKS}"
echo "RUN_SFT_PER_RATIO: ${RUN_SFT_PER_RATIO}"
echo "WANDB_PROJECT  : ${WANDB_PROJECT}"
echo "OUT_ROOT       : ${OUT_ROOT}"
echo "================================================================="

for dataset in ${DATASETS}; do
  first_for_dataset=1
  for lr in ${LRS}; do
    for epochs in ${EPOCHS_LIST}; do
      for optimizer in ${OPTIMIZERS}; do
        for method in ${METHODS}; do
          if [[ "${method}" == "sft" && "${RUN_SFT_PER_RATIO}" != "1" ]]; then
            skip_split=$([[ "${first_for_dataset}" == "1" ]] && echo 0 || echo 1)
            run_one "${dataset}" "sft" "100" "0" "${lr}" "${epochs}" "${optimizer}" "gradient" "${skip_split}" "fixed" "0.0" "0.0"
            first_for_dataset=0
          elif [[ "${method}" == "sft" ]]; then
            for ratio in ${CORRECT_RATIOS}; do
              for seed in ${CORRECT_SEEDS}; do
                skip_split=$([[ "${first_for_dataset}" == "1" ]] && echo 0 || echo 1)
                run_one "${dataset}" "sft" "${ratio}" "${seed}" "${lr}" "${epochs}" "${optimizer}" "gradient" "${skip_split}" "fixed" "0.0" "0.0"
                first_for_dataset=0
              done
            done
          elif [[ "${method}" == "cnl" ]]; then
            for ratio in ${CORRECT_RATIOS}; do
              for seed in ${CORRECT_SEEDS}; do
                for mask_stage in ${MASK_STAGES}; do
                  skip_split=$([[ "${first_for_dataset}" == "1" ]] && echo 0 || echo 1)
                  run_one "${dataset}" "${method}" "${ratio}" "${seed}" "${lr}" "${epochs}" "${optimizer}" "${mask_stage}" "${skip_split}" "fixed" "0.0" "0.0"
                  first_for_dataset=0
                done
              done
            done
          elif [[ "${method}" == "cnl_synth" ]]; then
            for synth_size_match in ${SYNTHETIC_CORRECT_SIZE_MATCHES}; do
              for ratio in ${CORRECT_RATIOS}; do
                for seed in ${CORRECT_SEEDS}; do
                  for mask_stage in ${MASK_STAGES}; do
                    skip_split=$([[ "${first_for_dataset}" == "1" ]] && echo 0 || echo 1)
                    run_one "${dataset}" "${method}" "${ratio}" "${seed}" "${lr}" "${epochs}" "${optimizer}" "${mask_stage}" "${skip_split}" "${synth_size_match}" "0.0" "0.0"
                    first_for_dataset=0
                  done
                done
              done
            done
          elif [[ "${method}" == "cnl_margin" ]]; then
            for cnl_margin in ${CNL_MARGINS}; do
              for ratio in ${CORRECT_RATIOS}; do
                for seed in ${CORRECT_SEEDS}; do
                  for mask_stage in ${MASK_STAGES}; do
                    skip_split=$([[ "${first_for_dataset}" == "1" ]] && echo 0 || echo 1)
                    run_one "${dataset}" "${method}" "${ratio}" "${seed}" "${lr}" "${epochs}" "${optimizer}" "${mask_stage}" "${skip_split}" "fixed" "${cnl_margin}" "0.0"
                    first_for_dataset=0
                  done
                done
              done
            done
          elif [[ "${method}" == "cnl_leaky" ]]; then
            for cnl_leak in ${CNL_LEAKS}; do
              for ratio in ${CORRECT_RATIOS}; do
                for seed in ${CORRECT_SEEDS}; do
                  for mask_stage in ${MASK_STAGES}; do
                    skip_split=$([[ "${first_for_dataset}" == "1" ]] && echo 0 || echo 1)
                    run_one "${dataset}" "${method}" "${ratio}" "${seed}" "${lr}" "${epochs}" "${optimizer}" "${mask_stage}" "${skip_split}" "fixed" "0.0" "${cnl_leak}"
                    first_for_dataset=0
                  done
                done
              done
            done
          elif [[ "${method}" == "cnl_margin_synth" ]]; then
            for synth_size_match in ${SYNTHETIC_CORRECT_SIZE_MATCHES}; do
              for cnl_margin in ${CNL_MARGINS}; do
                for ratio in ${CORRECT_RATIOS}; do
                  for seed in ${CORRECT_SEEDS}; do
                    for mask_stage in ${MASK_STAGES}; do
                      skip_split=$([[ "${first_for_dataset}" == "1" ]] && echo 0 || echo 1)
                      run_one "${dataset}" "${method}" "${ratio}" "${seed}" "${lr}" "${epochs}" "${optimizer}" "${mask_stage}" "${skip_split}" "${synth_size_match}" "${cnl_margin}" "0.0"
                      first_for_dataset=0
                    done
                  done
                done
              done
            done
          elif [[ "${method}" == "cnl_leaky_synth" ]]; then
            for synth_size_match in ${SYNTHETIC_CORRECT_SIZE_MATCHES}; do
              for cnl_leak in ${CNL_LEAKS}; do
                for ratio in ${CORRECT_RATIOS}; do
                  for seed in ${CORRECT_SEEDS}; do
                    for mask_stage in ${MASK_STAGES}; do
                      skip_split=$([[ "${first_for_dataset}" == "1" ]] && echo 0 || echo 1)
                      run_one "${dataset}" "${method}" "${ratio}" "${seed}" "${lr}" "${epochs}" "${optimizer}" "${mask_stage}" "${skip_split}" "${synth_size_match}" "0.0" "${cnl_leak}"
                      first_for_dataset=0
                    done
                  done
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

echo "Correct-ratio reproduction sweep done."
