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


def mask_gradients(grads: PyTree, reference_grads: PyTree | None) -> PyTree:
    """Zero gradient elements that conflict with the reference gradients."""
    if reference_grads is None:
        return grads

    return jax.tree_util.tree_map(
        lambda g, r: jnp.where((g * r) >= 0, g, jnp.zeros_like(g)),
        grads,
        reference_grads,
    )


def mask_optax_updates(updates: PyTree, reference_grads: PyTree | None) -> PyTree:
    """Zero Optax update elements whose gradient-like direction conflicts.

    Optax updates already include the descent sign, so the gradient-like
    direction is ``-updates``. This makes the mask condition match the PyTorch
    Adam/momentum variants, where the mask is computed on the positive
    optimizer direction and then subtracted from the parameters.
    """
    if reference_grads is None:
        return updates

    return jax.tree_util.tree_map(
        lambda u, r: jnp.where(((-u) * r) >= 0, u, jnp.zeros_like(u)),
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
    """
    if mask_stage not in ("gradient", "update"):
        raise ValueError("mask_stage must be 'gradient' or 'update'")

    opt_grads = mask_gradients(grads, reference_grads) if mask_stage == "gradient" else grads
    updates, opt_state = optimizer.update(opt_grads, opt_state, params)

    if mask_stage == "update":
        updates = mask_optax_updates(updates, reference_grads)

    params = optax.apply_updates(params, updates)
    return params, opt_state, updates

