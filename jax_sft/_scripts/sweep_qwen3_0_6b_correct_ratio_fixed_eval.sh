#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible alias. Prefer:
#   bash jax_sft/repro_ratio_qwen3_0_6b.sh csqa

bash jax_sft/repro_ratio_qwen3_0_6b.sh "${1:-csqa}"
