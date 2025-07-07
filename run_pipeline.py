#!/usr/bin/env python3
"""run_pipeline.py - Main orchestrator for the data pipeline.
============================================================

This script automates the entire two-step process of fetching orbital
data and generating training sequences. It acts as a single entry
point that calls the other scripts in the correct order.

This version is hardened for unattended runs with key features:
1.  **Graceful Shutdown**: Pressing CTRL-C will now cleanly terminate
    the running sub-process before exiting.
2.  **Unbuffered I/O**: Ensures real-time output streaming even in
    non-interactive environments like CI/CD runners.
3.  **Runtime Logging**: Reports the total execution time of the
    pipeline upon completion.

Requirements
------------
* All requirements from `fetch_tle.py` and `make_sequences.py`.
* This script should be located in the root of the `collision_mvp`
  project directory.

Usage example
-------------
# Run the full pipeline for the last 30 days, sampling 5000 pairs per day.
python run_pipeline.py --sample 5000
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Optional

import typer

app = typer.Typer(add_completion=False, help="Run the full data fetch and sequence generation pipeline.")

# --- Path Configuration ---
# This script assumes it's being run from the root of the 'collision_mvp' project.
SCRIPT_DIR = Path(__file__).parent.resolve()
FETCH_SCRIPT_PATH: Final[Path] = SCRIPT_DIR / "linoss" / "orbit-data" / "fetch_tle.py"
SEQUENCE_SCRIPT_PATH: Final[Path] = SCRIPT_DIR / "sequences" / "make_sequences.py"


def _run_command(cmd: list[str], script_name: str) -> None:
    """Run a command as a subprocess, streaming its output."""
    typer.secho(f"▶️  Running {script_name}...", fg=typer.colors.BLUE)
    typer.secho(f"   Command: {' '.join(cmd)}", fg=typer.colors.CYAN)

    proc = None
    try:
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        
        if proc.stdout:
            for line in proc.stdout:
                sys.stdout.write(f"   {line}")
        
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)

    except KeyboardInterrupt:
        typer.secho(f"\n🛑  Pipeline interrupted by user (CTRL-C).", fg=typer.colors.YELLOW)
        if proc:
            typer.secho(f"   Sending shutdown signal to {script_name}...", fg=typer.colors.YELLOW)
            proc.send_signal(signal.SIGINT)
            proc.wait()
        raise typer.Exit(130)
    except FileNotFoundError:
        typer.secho(f"❌ ERROR: Script not found at {cmd[1]}", fg=typer.colors.RED)
        typer.secho("   Please ensure the path configuration at the top of this script is correct.", fg=typer.colors.YELLOW)
        raise typer.Exit(1)
    except subprocess.CalledProcessError as e:
        typer.secho(f"\n❌ ERROR: {script_name} failed with exit code {e.returncode}.", fg=typer.colors.RED)
        # Raising typer.Exit with a non-zero code ensures the CI pipeline fails.
        raise typer.Exit(e.returncode)
        
    typer.secho(f"✅ {script_name} completed successfully.", fg=typer.colors.GREEN)


@app.command()
def main(
    start_days_ago: int = typer.Option(
        30,
        "--start-days-ago",
        help="Start date for the fetch range, in days ago.",
    ),
    end_days_ago: int = typer.Option(
        1,
        "--end-days-ago",
        help="End date for the fetch range (inclusive).",
    ),
    sample: Optional[int] = typer.Option(
        None,
        "--sample",
        min=1,
        help="Randomly sample N pairs per day during sequence generation.",
    ),
) -> None:
    """
    Run the complete data pipeline: fetch TLE/solar data, then generate sequence tensors.
    """
    start_time = datetime.now(timezone.utc)
    typer.secho("🚀  Starting the data processing pipeline...", fg=typer.colors.MAGENTA)
    
    # --- Step 1: Fetch Data ---
    fetch_cmd = [
        sys.executable, str(FETCH_SCRIPT_PATH),
        "--start-days-ago", str(start_days_ago),
        "--end-days-ago", str(end_days_ago),
    ]
    _run_command(fetch_cmd, "fetch_tle.py")

    # --- Step 2: Generate Sequences ---
    sequence_cmd = [
        sys.executable, str(SEQUENCE_SCRIPT_PATH),
        "--all",
    ]
    if sample:
        sequence_cmd.extend(["--sample", str(sample)])
    _run_command(sequence_cmd, "make_sequences.py")

    elapsed = datetime.now(timezone.utc) - start_time
    typer.secho("\n🎉  Pipeline finished successfully!", fg=typer.colors.BRIGHT_GREEN)
    typer.secho(f"⏱️  Total runtime: {elapsed}", fg=typer.colors.CYAN)


if __name__ == "__main__":
    app()
