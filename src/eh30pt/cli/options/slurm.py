"""SLURM job submission options."""

import rich_click as click

# Unannotated on purpose: click decorators are generic over the decorated
# callable, which a variable annotation cannot express, so the precise type
# is left for inference.

inner_command_argument = click.argument("command", type=str)

slurm_job_name_option = click.option(
    "--job-name",
    "job_name",
    type=str,
    default=None,
    help="SLURM job name, defaults to the training command",
)

slurm_output_option = click.option(
    "--output",
    "output",
    type=str,
    default="log/slurm/%j-%x.out",
    show_default=True,
    help="SLURM output path, supports %j and %x placeholders",
)

slurm_partition_option = click.option(
    "--partition",
    "partition",
    type=str,
    default=None,
    help="SLURM partition to submit to",
)

slurm_gres_option = click.option(
    "--gres",
    "gres",
    type=str,
    default=None,
    help="Generic resources to request per job",
)

slurm_cpus_option = click.option(
    "--cpus",
    "cpus",
    type=click.IntRange(min=1),
    default=8,
    show_default=True,
    help="CPUs to request per job",
)

slurm_mem_option = click.option(
    "--mem",
    "mem",
    type=str,
    default="64G",
    show_default=True,
    help="Memory to request per job",
)

slurm_time_option = click.option(
    "--time",
    "time",
    type=str,
    default="12:00:00",
    show_default=True,
    help="Wall-clock limit per job",
)

dry_run_option = click.option(
    "--dry-run",
    "dry_run",
    is_flag=True,
    default=False,
    help="Print the batch script without submitting",
)
