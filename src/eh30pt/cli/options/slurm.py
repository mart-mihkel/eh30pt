"""SLURM job submission options."""

import rich_click as click

# Unannotated on purpose: click decorators are generic over the decorated
# callable, which a variable annotation cannot express, so the precise type
# is left for inference.

InnerCommandArgument = click.argument("command", type=str)

SlurmJobNameOption = click.option(
    "--job-name",
    "job_name",
    type=str,
    default=None,
    help="SLURM job name, defaults to the training command",
)

SlurmOutputOption = click.option(
    "--output",
    "output",
    type=str,
    default="log/slurm/%j-%x.out",
    show_default=True,
    help="SLURM output path, supports %j and %x placeholders",
)

SlurmPartitionOption = click.option(
    "--partition",
    "partition",
    type=str,
    default=None,
    help="SLURM partition to submit to",
)

SlurmGresOption = click.option(
    "--gres",
    "gres",
    type=str,
    default=None,
    help="Generic resources to request per job",
)

SlurmCpusOption = click.option(
    "--cpus",
    "cpus",
    type=click.IntRange(min=1),
    default=8,
    show_default=True,
    help="CPUs to request per job",
)

SlurmMemOption = click.option(
    "--mem",
    "mem",
    type=str,
    default="64G",
    show_default=True,
    help="Memory to request per job",
)

SlurmTimeOption = click.option(
    "--time",
    "time",
    type=str,
    default="12:00:00",
    show_default=True,
    help="Wall-clock limit per job",
)

DryRunOption = click.option(
    "--dry-run",
    "dry_run",
    is_flag=True,
    default=False,
    help="Print the batch script without submitting",
)
