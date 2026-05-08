# JAX/Optax CNL

This directory contains a JAX/Optax implementation path for the core
Collaborative Neuron Learning update rule.

## What Is Implemented

- `cnl.py`: pure JAX tree utilities for CNL masking.
- `infer_split_optax.py`: split data into correct/wrong for the exact Flax
  model you plan to train.
- `sft_optax.py`: a Flax/Hugging Face SFT runner that mirrors `sft/sft.py`.
- `requirements-jax.txt`: optional JAX/Flax/Optax dependencies.

The training loop keeps the original experiment shape:

1. Load `wrong_jsonl` and `correct_jsonl`.
2. Compute a mean reference gradient on the correct/mastered set.
3. Train one wrong/injection sample at a time.
4. Mask conflicting gradient or optimizer-update elements.
5. Write per-epoch inference JSONL and `summary.csv`.

## Data Defaults

The runnable defaults use the checked-in CSQA split generated for
`Qwen2.5-1.5B-Instruct`:

```bash
data/csqa_wrong_Qwen2.5-1.5B-Instruct.jsonl
data/csqa_correct_Qwen2.5-1.5B-Instruct.jsonl
```

Each row is expected to contain:

```json
{
  "label": "A",
  "question": "Question: ...\nOptions:\nA: ...\nB: ...\nC: ...\nD: ...",
  "predict_label": "D"
}
```

`predict_label` is useful for provenance but is not required by the SFT loss.

For valid learning/forgetting metrics, the correct/wrong split must come from
the same model you train. For example, to make a GPT-2 CSQA split from the
checked-in Qwen split files:

```bash
python jax_sft/infer_split_optax.py \
  --model_name openai-community/gpt2 \
  --jsonl data/csqa_correct_Qwen2.5-1.5B-Instruct.jsonl data/csqa_wrong_Qwen2.5-1.5B-Instruct.jsonl \
  --out_correct_jsonl data/csqa_correct_openai-community-gpt2.jsonl \
  --out_wrong_jsonl data/csqa_wrong_openai-community-gpt2.jsonl \
  --max_length 256
```

Then train on those same-model files.

For Qwen3-0.6B, the intended one-command pipeline is:

```bash
bash jax_sft/run_qwen3_0_6b_split_train.sh csqa
```

By default this uses the vendored `jax_sft/ptx_backend/qwen.py` Qwen3 JAX
backend. Set `PTX_DIR=/path/to/ptx` only if you want to override it with an
external PTX checkout.

Useful overrides:

```bash
WANDB_PROJECT=cnl-repro \
EPOCHS=1 \
MAX_LENGTH=256 \
bash jax_sft/run_qwen3_0_6b_split_train.sh csqa
```

Run a matched CNL/baseline sweep:

```bash
WANDB_PROJECT=cnl-repro \
bash jax_sft/sweep_qwen3_0_6b.sh csqa
```

Small smoke sweep:

```bash
MAX_ROWS=64 \
MAX_WRONG=32 \
MAX_CORRECT=32 \
LRS="1e-8 5e-8" \
EPOCHS_LIST="1" \
WANDB_PROJECT=cnl-repro \
bash jax_sft/sweep_qwen3_0_6b.sh csqa
```

Sweep knobs:

```text
LRS="1e-9 2e-9 5e-9 1e-8 2e-8 5e-8 1e-7 2e-7 5e-7 1e-6 2e-6 5e-6 1e-5 2e-5 5e-5 1e-4"
EPOCHS_LIST="1 2 3"
OPTIMIZERS="adamw sgd"
MASK_STAGES="gradient update"
METHODS="cnl sft"
MAX_ROWS=512
MAX_WRONG=256
MAX_CORRECT=256
```

`MASK_STAGES` only affects CNL runs. `gradient` masks raw gradients before the
optimizer update; `update` masks the final optimizer update direction. SFT
ignores this setting and is logged as `masknone`.

## Three Experiment Tracks

### 1. Paper-Style CNL Reproduction

This keeps the paper's original experiment shape: split one dataset into
examples the current model answers correctly and incorrectly, train on the
wrong/injection set, and use the correct/mastered set for CNL masking.

```bash
WANDB_PROJECT=cnl-repro \
bash jax_sft/reproduce_cnl_paper_qwen3_0_6b.sh
```

Useful smoke version:

```bash
DATASETS="csqa" \
METHODS="cnl sft" \
MAX_ROWS=64 \
MAX_WRONG=32 \
MAX_CORRECT=32 \
EPOCHS=1 \
WANDB_PROJECT=cnl-repro \
bash jax_sft/reproduce_cnl_paper_qwen3_0_6b.sh
```

### 2. Explicit A/B Retention-vs-Injection

This is the continual-learning setting:

```text
A = old / retained data
B = new / injection data
```

The runner logs A/B accuracy and loss before and after training, plus
`a_drop`, `b_gain`, and `tradeoff_score`.

```bash
WANDB_PROJECT=cnl-repro \
bash jax_sft/run_qwen3_0_6b_ab_train.sh csqa medqa
```

By default, A is filtered to examples the base model currently answers
correctly (`RETENTION_FILTER=correct`) and B is trained as provided
(`TRAIN_FILTER=none`). For a paper-like B injection set, use:

```bash
RETENTION_FILTER=correct \
TRAIN_FILTER=wrong \
WANDB_PROJECT=cnl-repro \
bash jax_sft/run_qwen3_0_6b_ab_train.sh csqa medqa
```

To compare against plain SFT on B:

```bash
METHOD=sft \
WANDB_PROJECT=cnl-repro \
bash jax_sft/run_qwen3_0_6b_ab_train.sh csqa medqa
```

### 3. Synthetic A

This creates synthetic retention rows by pseudo-labeling a prompt bank with the
frozen base model, then uses those rows as A for the A/B runner.

```bash
WANDB_PROJECT=cnl-repro \
bash jax_sft/run_qwen3_0_6b_synthetic_a_ab_train.sh csqa medqa
```

If you still have a real A evaluation set, pass it separately so training uses
synthetic A but retention is measured on real A:

```bash
A_EVAL_JSONLS="data/csqa_correct_Qwen2.5-1.5B-Instruct.jsonl data/csqa_wrong_Qwen2.5-1.5B-Instruct.jsonl" \
WANDB_PROJECT=cnl-repro \
bash jax_sft/run_qwen3_0_6b_synthetic_a_ab_train.sh csqa medqa
```

Reference gradient refresh can be tightened with `REF_REFRESH_STEPS`. The
default `0` means once per epoch. `REF_REFRESH_STEPS=1` is closest to the
current-gradient assumption but much slower.

The Qwen3 wrappers default to `Qwen/Qwen3-0.6B` and use the vendored
`jax_sft/ptx_backend/qwen.py` backend, so they do not depend on Hugging Face
FlaxAuto support for Qwen3. The older `sft_optax.py` and `infer_split_optax.py`
scripts still use `FlaxAutoModelForCausalLM`; keep those for GPT-2-style smoke
tests or models with native Flax classes.

## Install

```bash
pip install -r requirements-jax.txt
```

Use the JAX installation command appropriate for your accelerator if you need
GPU/TPU support.

For a TPU VM using `uv`, include `optax`, `wandb`, `transformers`,
`safetensors`, and `huggingface-hub`:

```bash
uv venv ~/.venvs/py312 --python 3.12 -q
source ~/.venvs/py312/bin/activate
uv pip install "jax[tpu]" flax optax wandb fire hydra-core omegaconf datasets grain "transformers>=4.51.0,<5" safetensors "huggingface-hub[hf-xet]" zstandard jinja2
```

## Run

Use a model that has a Flax causal-LM implementation in `transformers`:

```bash
python jax_sft/sft_optax.py \
  --model_name openai-community/gpt2 \
  --wrong_jsonl data/csqa_wrong_Qwen2.5-1.5B-Instruct.jsonl \
  --correct_jsonl data/csqa_correct_Qwen2.5-1.5B-Instruct.jsonl \
  --out_dir jax_ckpts/csqa_gpt2_cnl \
  --optimizer sgd \
  --lr 1e-7 \
  --epochs 25 \
  --use_freeze 1
```

For a quick TPU smoke test, cap both splits first:

```bash
python jax_sft/sft_optax.py \
  --model_name openai-community/gpt2 \
  --wrong_jsonl data/csqa_wrong_Qwen2.5-1.5B-Instruct.jsonl \
  --correct_jsonl data/csqa_correct_Qwen2.5-1.5B-Instruct.jsonl \
  --out_dir jax_ckpts/smoke_gpt2_cnl \
  --optimizer sgd \
  --lr 1e-7 \
  --epochs 1 \
  --use_freeze 1 \
  --max_correct 4 \
  --max_wrong 4
```

The runner pads/truncates prompts to a static `--max_length` and jit-compiles
the per-sample gradient/update path. This is intended to validate correctness on
TPU. It still uses one sample at a time and does not yet shard parameters or
batches across all TPU devices.

To log epoch metrics to W&B, add:

```bash
--wandb_project cnl-repro \
--wandb_run_name smoke-gpt2-csqa-cnl
```

If you only have PyTorch weights for a model with a compatible Flax
architecture, add:

```bash
--from_pt
```

## Important Model Constraint

The original scripts target Qwen2.5 and Llama-3.2 PyTorch checkpoints. Those
architectures may not have native Flax implementations in the installed
`transformers` version. The JAX code is Optax-native, but the model must still
be loadable by `FlaxAutoModelForCausalLM`.

For Qwen/Llama-scale production runs in JAX, the next step is usually to wire
these `cnl.py` update primitives into a JAX-native model stack such as MaxText,
EasyLM, or another internal Flax/Linen implementation.
