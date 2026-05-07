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

This wrapper defaults to `Qwen/Qwen3-0.6B`, whose Hugging Face model card
requires recent `transformers` support for Qwen3. Use `transformers>=4.51,<5`
for this runner: 4.51+ recognizes Qwen3 configs, while 5.x no longer exposes
the Flax auto-model path used here. The current Python runner uses
`FlaxAutoModelForCausalLM`; if that stack recognizes Qwen3 but cannot load a
Flax Qwen3 causal LM, use the same pipeline shape with a Qwen3-capable JAX
backend such as EasyDeL or MaxText.

## Install

```bash
pip install -r requirements-jax.txt
```

Use the JAX installation command appropriate for your accelerator if you need
GPU/TPU support.

For a TPU VM using `uv`, include `flax` alongside `jax[tpu]` and `optax`:

```bash
uv venv ~/.venvs/py312 --python 3.12 -q
source ~/.venvs/py312/bin/activate
uv pip install "jax[tpu]" flax optax wandb fire hydra-core omegaconf datasets grain "transformers>=4.51.0,<5" safetensors "huggingface-hub[hf-xet]" zstandard jinja2
```

The script does not require PyTorch unless you pass `--from_pt` to convert
PyTorch checkpoints into Flax parameters.

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
