"""Submit training commands as SLURM batch jobs."""

import re
import shlex
import shutil
import subprocess
from pathlib import Path

import typer

from eh30pt.logging import logger


def build_sbatch(
    command: str,
    args: list[str],
    *,
    job_name: str,
    partition: str | None,
    gres: str | None,
    cpus: int,
    mem: str,
    time: str,
    project_dir: str,
    output: str,
) -> str:
    """Render the batch script running command with args."""
    directives = [f"#SBATCH --job-name={job_name}"]
    if partition is not None:
        directives.append(f"#SBATCH --partition={partition}")

    if gres is not None:
        directives.append(f"#SBATCH --gres={gres}")

    directives.extend(
        [
            f"#SBATCH --cpus-per-task={cpus}",
            f"#SBATCH --mem={mem}",
            f"#SBATCH --time={time}",
            f"#SBATCH --output={output}",
        ]
    )

    inner = shlex.join(
        [
            "uv",
            "run",
            "--no-sync",
            "eh30pt",
            command,
            *args,
        ]
    )

    body = [
        "set -euo pipefail",
        f"cd {shlex.quote(project_dir)}",
        inner,
    ]

    return "\n".join(["#!/usr/bin/env bash", *directives, "", *body, ""])


def main(
    command: str,
    args: list[str],
    *,
    job_name: str,
    output: str,
    partition: str | None,
    gres: str | None,
    cpus: int,
    mem: str,
    time: str,
    dry_run: bool,
) -> None:
    """Render the batch script for command, then print or submit it."""
    project_dir = str(Path.cwd())
    script = build_sbatch(
        command,
        args,
        job_name=job_name,
        partition=partition,
        gres=gres,
        cpus=cpus,
        mem=mem,
        time=time,
        project_dir=project_dir,
        output=output,
    )

    path = Path(project_dir) / "log" / "slurm" / f"{job_name}.sbatch"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script)
    logger.info("wrote batch script to '%s'", path)

    if dry_run:
        typer.echo(script, nl=False)
        return

    sbatch = shutil.which("sbatch")
    if sbatch is None:
        msg = "sbatch executable not found on PATH"
        raise RuntimeError(msg)

    completed = subprocess.run(  # noqa: S603
        [sbatch, str(path)],
        capture_output=True,
        check=True,
        text=True,
    )

    logger.info("sbatch: %s", completed.stdout.strip())

    match = re.search(r"Submitted batch job (\d+)", completed.stdout)
    if match is not None:
        logger.info("submitted slurm job %s", match.group(1))
