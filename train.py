#!/usr/bin/env python3
"""
train.py – LinOSS training loop
-----------------------------------------

Reads every *.parquet tensor in sequences/parquet/, splits 90 / 10,
and trains until the validation MAE stops improving.

This version is updated to work with the final project structure,
importing the model directly from the `linoss` package.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Final, Iterator, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import polars as pl
import typer
from flax.training import train_state
from tqdm.auto import tqdm

# --- Add project root to path to enable finding sibling packages ---
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# -----------------------------------------------------------------------

# Assumes this script is run from the repository root
from linoss.models.LinOSS import LinOSSLayer


# ── paths ─────────────────────────────────────────────────────────────
PARQUET_DIR: Final[Path] = ROOT / "sequences" / "parquet"
OUTPUTS_DIR: Final[Path] = ROOT / "outputs"
CKPT_DIR: Final[Path] = OUTPUTS_DIR / "checkpoints"
OUTPUTS_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)
CKPT_FILE: Final[Path] = CKPT_DIR / "best_model.msgpack"
METRIC_FILE: Final[Path] = OUTPUTS_DIR / "train_history.json"


# ── hyper-params (CLI overrides) ──────────────────────────────────────
app = typer.Typer(add_completion=False, help="Train the LinOSS conjunction model.")


def data_generator(
    files: list[Path],
    batch_size: int,
    *,
    shuffle: bool = True,
) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Yields shuffled batches (x, y) from a list of Parquet files.
    This version streams from disk to keep memory usage low.
    """
    if shuffle:
        random.shuffle(files)

    buffer = []
    for p in files:
        x = pl.read_parquet(p).to_numpy(dtype=np.float32)
        y = np.linalg.norm(x[-1, :3])
        buffer.append((x, y))
        
        if len(buffer) == batch_size:
            x_batch, y_batch = zip(*buffer)
            yield np.stack(x_batch), np.array(y_batch)
            buffer.clear()
            
    if buffer: # Yield any remaining items
        x_batch, y_batch = zip(*buffer)
        yield np.stack(x_batch), np.array(y_batch)


@app.command()
def main(
    epochs: int = typer.Option(50, help="Maximum number of training epochs."),
    batch_size: int = typer.Option(512, help="Number of samples per batch."),
    lr: float = typer.Option(3e-4, help="Learning rate for the AdamW optimizer."),
    patience: int = typer.Option(5, help="Epochs to wait for improvement before early stopping."),
    seed: int = typer.Option(0, help="Random seed for reproducibility."),
):
    """Train the LinOSS model on generated sequence data."""
    files = sorted(PARQUET_DIR.glob("*.parquet"))
    if len(files) < 100:
        typer.secho(f"⛔️ Not enough tensors found in {PARQUET_DIR}", fg=typer.colors.RED)
        raise typer.Exit(1)
        
    random.seed(seed)
    random.shuffle(files)

    split_idx = math.floor(0.9 * len(files))
    train_files, val_files = files[:split_idx], files[split_idx:]
    typer.echo(f"Found {len(files)} total samples.")
    typer.echo(f"Training on {len(train_files)} samples, validating on {len(val_files)}.")

    # --- Initialize model, optimizer, and state ---
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    
    model = LinOSSLayer(
        num_oscillators=250,
        readout_dim=1, # Direct scalar output for regression
        nonlin="glu",
        implicit=True
    )

    dummy_x = jnp.zeros((batch_size, 720, 5), dtype=jnp.float32)
    params = model.init(init_rng, dummy_x)

    tx = optax.adamw(lr)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # --- Define JIT-compiled train and validation steps ---
    @jax.jit
    def train_step(state, x, y):
        def loss_fn(params):
            y_pred = state.apply_fn(params, x)
            return jnp.mean(jnp.abs(y_pred.squeeze() - y))
        
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    @jax.jit
    def validation_step(state, x, y):
        y_pred = state.apply_fn({'params': state.params}, x)
        return jnp.mean(jnp.abs(y_pred.squeeze() - y))

    # --- Training Loop ---
    history = {"train_mae": [], "val_mae": []}
    best_val_mae = float('inf')
    epochs_no_improve = 0
    
    for ep in range(1, epochs + 1):
        # -- Train --
        train_loss, train_steps = 0.0, 0
        train_gen = data_generator(train_files, batch_size, shuffle=True)
        with tqdm(total=len(train_files), desc=f"Epoch {ep}/{epochs} [Train]", leave=False) as pbar:
            for x, y in train_gen:
                state, loss = train_step(state, x, y)
                train_loss += loss.item() * len(x)
                train_steps += len(x)
                pbar.update(len(x))
        epoch_train_mae = train_loss / train_steps
        history["train_mae"].append(epoch_train_mae)

        # -- Validate --
        val_loss, val_steps = 0.0, 0
        val_gen = data_generator(val_files, batch_size, shuffle=False)
        for x, y in tqdm(val_gen, total=len(val_files), desc="Validating", leave=False):
            loss = validation_step(state, x, y)
            val_loss += loss.item() * len(x)
            val_steps += len(x)
        epoch_val_mae = val_loss / val_steps
        history["val_mae"].append(epoch_val_mae)
        
        tqdm.write(f"Epoch {ep:02d} | Train MAE: {epoch_train_mae:.3f} km | Val MAE: {epoch_val_mae:.3f} km")

        # -- Checkpoint and Early Stopping --
        if epoch_val_mae < best_val_mae:
            best_val_mae = epoch_val_mae
            epochs_no_improve = 0
            from flax.serialization import to_bytes
            CKPT_FILE.write_bytes(to_bytes(state.params))
            tqdm.write(f"  ✅ New best validation MAE. Checkpoint saved to {CKPT_FILE}")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            tqdm.write(f"🛑 Early stopping triggered after {patience} epochs with no improvement.")
            break
            
    METRIC_FILE.write_text(json.dumps(history, indent=2))
    typer.secho(f"\n🏅 Best validation MAE: {best_val_mae:.3f} km", fg=typer.colors.BRIGHT_GREEN)


if __name__ == "__main__":
    app()
