# Vendored PTX Qwen Backend

This directory contains the minimal Qwen3 JAX backend needed by the CNL
Qwen3 split/train pipeline.

Source:

```text
/Users/dongkyu/ptx/models/qwen.py
```

Only the Qwen model loader/forward/save code is vendored here. The rest of the
PTX training repo is intentionally not copied so this repository stays focused
on CNL experiments.

