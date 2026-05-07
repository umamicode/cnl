#!/usr/bin/env python
"""Preflight checks for the temporary FlaxAuto runner."""

from __future__ import annotations

import importlib.util
import sys

import transformers
from packaging.version import Version


def main() -> int:
    version = Version(transformers.__version__)
    print(f"transformers={transformers.__version__}")

    if version.major >= 5:
        print(
            "ERROR: transformers 5.x does not expose the FlaxAutoModelForCausalLM "
            "path used by jax_sft. Install a 4.x version instead:\n"
            "  uv pip install --reinstall 'transformers>=4.51.0,<5'",
            file=sys.stderr,
        )
        return 1

    if version < Version("4.51.0"):
        print(
            "ERROR: Qwen3 configs need transformers>=4.51.0. Install:\n"
            "  uv pip install --reinstall 'transformers>=4.51.0,<5'",
            file=sys.stderr,
        )
        return 1

    if importlib.util.find_spec("flax") is None:
        print("ERROR: flax is not installed. Install: uv pip install flax", file=sys.stderr)
        return 1

    try:
        from transformers.models.auto.modeling_flax_auto import FLAX_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    except Exception as exc:
        print(f"ERROR: could not import Flax auto-model mapping: {exc}", file=sys.stderr)
        return 1

    supported = sorted(FLAX_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
    print("Flax causal-LM model types:", ", ".join(supported))
    if "qwen3" not in supported:
        print(
            "ERROR: this transformers/flax stack has no Flax Qwen3 causal-LM. "
            "The temporary FlaxAuto runner cannot train Qwen3. Use EasyDeL or "
            "MaxText for the Qwen3 model backend.",
            file=sys.stderr,
        )
        return 2

    print("Flax Qwen3 support detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

