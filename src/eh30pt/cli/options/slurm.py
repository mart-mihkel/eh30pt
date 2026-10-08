"""SLURM job submission options."""

from typing import Annotated

from typer import Argument, Option

InnerCommandArgument = Annotated[
    str,
    Argument(help="Training command to submit"),
]

SlurmJobNameOption = Annotated[
    str | None,
    Option("--job-name", help="SLURM job name, defaults to the training command"),
]

SlurmOutputOption = Annotated[
    str,
    Option("--output", help="SLURM output path, supports %j and %x placeholders"),
]

SlurmPartitionOption = Annotated[
    str | None,
    Option("--partition", help="SLURM partition to submit to"),
]

SlurmGresOption = Annotated[
    str | None,
    Option("--gres", help="Generic resources to request per job"),
]

SlurmCpusOption = Annotated[
    int,
    Option("--cpus", min=1, help="CPUs to request per job"),
]

SlurmMemOption = Annotated[
    str,
    Option("--mem", help="Memory to request per job"),
]

SlurmTimeOption = Annotated[
    str,
    Option("--time", help="Wall-clock limit per job"),
]

DryRunOption = Annotated[
    bool,
    Option("--dry-run", help="Print the batch script without submitting"),
]
