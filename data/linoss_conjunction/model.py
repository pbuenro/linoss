# collision_mvp/linoss_conjunction/model.py
from __future__ import annotations
from functools import partial
import jax, jax.numpy as jnp
from linoss.models.LinOSS import LinOSSLayer


class LinOSSPredictor:
    """
    Wrapper around LinOSSLayer.
      • input : float32 [720, 5]
      • output: scalar miss-distance (km)
    """

    def __init__(
        self,
        num_oscillators: int = 250,
        hidden_dim: int = 128,
        seed: int = 0,
    ):
        key = jax.random.PRNGKey(seed)
        self.layer = LinOSSLayer(
            num_oscillators=num_oscillators,
            readout_dim=hidden_dim,
            nonlin="glu",
            implicit=True,
            key=key,
        )

    # ---- JIT-compiled forward -------------------------------------------------
    @partial(jax.jit, static_argnums=0)   # ‘self’ is static, input is traced
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.shape != (720, 5):
            raise ValueError("Input must have shape [720, 5]")
        if x.dtype != jnp.float32:
            x = x.astype(jnp.float32)

        return jnp.squeeze(self.layer(x))        # scalar
