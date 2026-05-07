# Collaborative Neuron Learning (CNL)

A PyTorch implementation for mitigating catastrophic forgetting in LLM fine-tuning through gradient-based neuron selection.

## Overview

When fine-tuning LLMs on new knowledge, the model often forgets previously learned information — a phenomenon known as **catastrophic forgetting**. This project implements **Collaborative Neuron Learning (CNL)**, which selectively updates only the neurons that contribute positively to both learning new knowledge and retaining old knowledge.

### Core Idea

![Framework](frame.jpg)

**Key Concepts:**

| Term | Definition |
|------|------------|
| **Mastered Set (M)** | Samples the model already answers correctly |
| **Injection Set (I)** | Samples the model currently answers incorrectly (new knowledge to learn) |
| **Gradient Similarity** | $S(\mathcal{M}, \mathcal{I}) = \nabla_{\theta} \mathcal{L}^{\mathcal{M}}(\theta)^{\top} \nabla_{\theta} \mathcal{L}^{\mathcal{I}}(\theta)$ |
| **Neuron Contribution** | $s_{\theta_j}(\mathcal{M}, \mathcal{I}) = \nabla_{\theta_j} \mathcal{L}^{\mathcal{M}}(\theta) \cdot \nabla_{\theta_j} \mathcal{L}^{\mathcal{I}}(\theta)$ |

**Neuron Classification:**

- **Collaborative Neuron** ($s_{\theta_j} \geq 0$): Gradient directions align → update allowed
- **Conflicting Neuron** ($s_{\theta_j} < 0$): Gradient directions conflict → frozen during training

**CNL Update Rule:**

$$\theta \gets \theta - \eta \cdot \mathbb{I}[s_{\theta}(\mathcal{M}, \mathcal{I}) \geq 0] \odot \nabla_{\theta} L^{\mathcal{I}}(\theta)$$

## Project Structure

```
├── infer/                  # Inference & data splitting
│   ├── infer.py
│   └── infer_*.sh
│
├── sft/                    # Main SFT with CNL
│   ├── sft.py
│   └── sft_*.sh
│
├── neuron/                 # Neuron-level gradient analysis
│   ├── neuron.py
│   └── neuron_*.sh
│
├── distance/               # Gradient dot product analysis
│   ├── distance.py
│   ├── neg_distance.py
│   └── distance_*.sh
│
├── sft_lora/               # LoRA baseline
├── sft_moment/             # CNL + Momentum
├── sft_adam/               # CNL + Adam optimizer
├── sft_adamw/              # CNL + AdamW optimizer
├── sft_replay/             # Replay baseline (mixed training)
├── sft_ood_correct/        # OOD experiment on correct samples
├── sft_ood_wrong/          # OOD experiment on wrong samples
└── jax_sft/                # JAX/Optax CNL primitives and Flax runner
```

## Quick Start

### 0. Data Format

The input data should be a JSONL file where each line follows this format:

```json
{
  "label": "A",
  "question": "Question: [Your Question here]\nOptions:\nA: [Option A]\nB: [Option B]\nC: [Option C]\nD: [Option D]\n\nPlease reply with only A, B, C, or D. Do not provide any explanation."
}
```

### 1. Inference & Split

Split dataset into correct (mastered) and wrong (injection) samples:

```bash
bash infer/infer_qwen_1.5b.sh csqa
```

Then, split the datasets into two parts (required for Replay or OOD experiments):

```bash
# Split Mastered Set (for Replay / OOD-Correct)
python infer/split_data.py \
  --input_jsonl ./data/csqa_correct_Qwen2.5-1.5B-Instruct.jsonl \
  --output1 ./data/csqa_correct1_Qwen2.5-1.5B-Instruct.jsonl \
  --output2 ./data/csqa_correct2_Qwen2.5-1.5B-Instruct.jsonl

# Split Injection Set (for OOD-Wrong)
python infer/split_data.py \
  --input_jsonl ./data/csqa_wrong_Qwen2.5-1.5B-Instruct.jsonl \
  --output1 ./data/csqa_wrong1_Qwen2.5-1.5B-Instruct.jsonl \
  --output2 ./data/csqa_wrong2_Qwen2.5-1.5B-Instruct.jsonl
```

This creates:
- `data/{dataset}_correct_{model}.jsonl` — Mastered Set
- `data/{dataset}_wrong_{model}.jsonl` — Injection Set
- `data/{dataset}_correct1/2_{model}.jsonl` — Mastered Set Splits
- `data/{dataset}_wrong1/2_{model}.jsonl` — Injection Set Splits

### 2. Train with CNL

```bash
# With CNL (use_freeze=1)
bash sft/sft_qwen_1.5b.sh csqa

# Without CNL - conventional fine-tuning (use_freeze=0)
# Modify USE_FREEZE=0 in the script
```

### 3. Analyze Results

Check `summary.csv` in the output directory:

| Column | Description |
|--------|-------------|
| `epoch` | Training epoch |
| `train_avg_loss` | Average training loss |
| `wrong_to_correct` | Injection set: wrong → correct = **Learning** |
| `correct_to_wrong` | Mastered set: correct → wrong = **Forgetting** |

### 4. Gradient Analysis

Analyze neuron-level gradient statistics:

```bash
bash neuron/neuron_qwen_1.5b.sh csqa
```

Analyze gradient dot product distribution:

```bash
bash distance/distance_qwen_1.5b.sh csqa
python distance/neg_distance.py
```

## Supported Models

- Qwen2.5-1.5B/3B/7B-Instruct
- Llama-3.2-1B/3B-Instruct

## Command Line Arguments

All scripts accept a dataset name as the first argument (default: `csqa`):

```bash
bash sft/sft_qwen_1.5b.sh arc_c      # ARC-Challenge
bash sft/sft_qwen_1.5b.sh csqa       # CommonsenseQA
bash sft/sft_qwen_1.5b.sh medqa      # MedQA
bash sft/sft_qwen_1.5b.sh mmlu       # MMLU
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--use_freeze` | 1: CNL enabled, 0: CNL disabled | 1 |
| `--epochs` | Number of training epochs | 25 |

## Experimental Environment

- **Python**: 3.12.2
- **Precision**: FP32
- **Reproducibility**: Temperature set to 0

## Requirements

- `torch==2.2.0`
- `transformers==4.48.0`
- `tqdm==4.67.1`
- `numpy==2.2.6`
- `peft==0.12.0`

Optional JAX/Optax path:

```bash
pip install -r requirements-jax.txt
```

The JAX runner is in `jax_sft/sft_optax.py`. It defaults to the checked-in CSQA
correct/wrong split, but still requires a model supported by
`FlaxAutoModelForCausalLM`.

## License

 Apache-2.0 license
