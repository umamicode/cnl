from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, reshard


class SamplingState(NamedTuple):
    step: int
    key: jax.Array
    tokens: jax.Array
    kv: jax.Array
    done: jax.Array


def _sample_step(state, forward, weights, tokenizer, temperature, eos_id):
    pad_id = tokenizer.pad_token_id
    key, key_sampling = jax.random.split(state.key)
    input_token = state.tokens[:, state.step, None]
    logits, kv = forward(input_token, weights, state.kv, state.step)
    sampled_token = jax.random.categorical(key_sampling, logits[:, 0, :] / temperature)

    next_token = state.tokens[:, state.step + 1]
    update_token = jnp.where((~state.done) & (next_token == pad_id), sampled_token, next_token)
    tokens = state.tokens.at[:, state.step + 1].set(update_token)
    done = state.done | ((next_token == pad_id) & (sampled_token == eos_id))
    return SamplingState(state.step + 1, key, tokens, kv, done)


@partial(jax.jit, static_argnames=("forward", "init_kv", "tokenizer"))
def _sample(key, forward, init_kv, weights, tokenizer, tokens, temperature, eos_id):
    batch_size, seq_len = tokens.shape
    tokens = reshard(tokens, P("data", None))
    state = SamplingState(
        step=0,
        key=key,
        tokens=tokens,
        kv=init_kv(batch_size, seq_len),
        done=jnp.zeros([batch_size], dtype=bool, out_sharding=P("data")),
    )
    step_fn = lambda state: _sample_step(state, forward, weights, tokenizer, temperature, eos_id)
    cond_fn = lambda state: (state.step + 1 < seq_len) & jnp.any(~state.done)
    state = jax.lax.while_loop(cond_fn, step_fn, state)
    return state.tokens


def sample(key, model, tokens, temperature=1.0, weights=None, eos_id=None):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if weights is None:
        weights = model.weights
    if eos_id is None:
        eos_id = model.tokenizer.eos_token_id or -1
    return _sample(key, model.forward, model.init_kv, weights, model.tokenizer, tokens, temperature, eos_id)
