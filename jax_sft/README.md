# JAX/Optax CNL

This directory contains a JAX/Optax implementation path for the core
Collaborative Neuron Learning update rule.

## What Is Implemented

- `cnl.py`: pure JAX tree utilities for CNL masking.
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
uv pip install "jax[tpu]" flax optax wandb fire hydra-core omegaconf datasets grain transformers safetensors huggingface-hub zstandard jinja2
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
