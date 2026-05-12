"""CNL masking primitives for JAX and Optax.

The original PyTorch implementation updates a parameter element only when the
training gradient and the mastered-set reference gradient have nonnegative
product. These helpers keep that rule explicit and reusable with any Optax
optimizer.
"""

from __future__ import annotations

from typing import Any, Literal

import jax
import jax.numpy as jnp
import optax

PyTree = Any
MaskStage = Literal["gradient", "update"]
MaskMode = Literal["hard", "margin", "leaky"]


def _validate_mask_args(mask_mode: MaskMode, margin: float, leak: float) -> None:
    if mask_mode not in ("hard", "margin", "leaky"):
        raise ValueError("mask_mode must be 'hard', 'margin', or 'leaky'")
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    if not 0 <= leak <= 1:
        raise ValueError("leak must be in [0, 1]")


def _apply_mask(value: jax.Array, score: jax.Array, mask_mode: MaskMode, margin: float, leak: float) -> jax.Array:
    if mask_mode == "margin":
        return jnp.where(score >= -margin, value, jnp.zeros_like(value))
    if mask_mode == "leaky":
        return jnp.where(score >= 0, value, value * leak)
    return jnp.where(score >= 0, value, jnp.zeros_like(value))


def mask_gradients(
    grads: PyTree,
    reference_grads: PyTree | None,
    *,
    mask_mode: MaskMode = "hard",
    margin: float = 0.0,
    leak: float = 0.0,
) -> PyTree:
    """Relax or zero gradient elements that conflict with reference gradients."""
    if reference_grads is None:
        return grads

    _validate_mask_args(mask_mode, margin, leak)
    return jax.tree_util.tree_map(
        lambda g, r: _apply_mask(g, g * r, mask_mode, margin, leak),
        grads,
        reference_grads,
    )


def mask_optax_updates(
    updates: PyTree,
    reference_grads: PyTree | None,
    *,
    mask_mode: MaskMode = "hard",
    margin: float = 0.0,
    leak: float = 0.0,
) -> PyTree:
    """Relax or zero Optax update elements whose gradient-like direction conflicts.

    Optax updates already include the descent sign, so the gradient-like
    direction is ``-updates``. This makes the mask condition match the PyTorch
    Adam/momentum variants, where the mask is computed on the positive
    optimizer direction and then subtracted from the parameters.
    """
    if reference_grads is None:
        return updates

    _validate_mask_args(mask_mode, margin, leak)
    return jax.tree_util.tree_map(
        lambda u, r: _apply_mask(u, (-u) * r, mask_mode, margin, leak),
        updates,
        reference_grads,
    )


def add_trees(a: PyTree, b: PyTree) -> PyTree:
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def divide_tree(tree: PyTree, denom: float) -> PyTree:
    return jax.tree_util.tree_map(lambda x: x / denom, tree)


def cnl_optax_step(
    params: PyTree,
    grads: PyTree,
    reference_grads: PyTree | None,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    *,
    mask_stage: MaskStage = "gradient",
    mask_mode: MaskMode = "hard",
    margin: float = 0.0,
    leak: float = 0.0,
) -> tuple[PyTree, optax.OptState, PyTree]:
    """Apply one Optax step with optional CNL masking.

    Args:
        params: Current model parameters.
        grads: Gradients for the injection/wrong sample or batch.
        reference_grads: Mean gradients from the mastered/correct set. If
            ``None``, this is a normal Optax update.
        optimizer: Any Optax gradient transformation, such as SGD or AdamW.
        opt_state: Current optimizer state.
        mask_stage: ``"gradient"`` matches the main PyTorch ``sft.py`` rule.
            ``"update"`` matches the Adam/momentum variants, masking the final
            optimizer update direction instead.
        mask_mode: ``"hard"`` is original CNL. ``"margin"`` accepts mildly
            negative similarity scores ``>= -margin``. ``"leaky"`` scales
            conflicting coordinates by ``leak`` instead of freezing them.
    """
    if mask_stage not in ("gradient", "update"):
        raise ValueError("mask_stage must be 'gradient' or 'update'")
    _validate_mask_args(mask_mode, margin, leak)

    opt_grads = (
        mask_gradients(grads, reference_grads, mask_mode=mask_mode, margin=margin, leak=leak)
        if mask_stage == "gradient"
        else grads
    )
    updates, opt_state = optimizer.update(opt_grads, opt_state, params)

    if mask_stage == "update":
        updates = mask_optax_updates(updates, reference_grads, mask_mode=mask_mode, margin=margin, leak=leak)

    params = optax.apply_updates(params, updates)
    return params, opt_state, updates
