from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn


class LinOSSLayer(nn.Module):
    """Placeholder LinOSS model layer."""

    num_oscillators: int = 250
    readout_dim: int = 1
    nonlin: str = "glu"
    implicit: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(self.readout_dim)(x)
        return x
