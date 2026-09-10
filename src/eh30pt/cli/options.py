"""Shared Typer option and argument types."""

from pathlib import Path
from typing import Annotated

from typer import Argument, Option

from eh30pt.types import (
    Architecture,
    DatasetName,
    DataSource,
    LogLevel,
    PrefixInit,
    Provider,
)

ModelOption = Annotated[
    str,
    Option("--model", "-m", help="HuggingFace model or path to checkpoint"),
]

DatasetOption = Annotated[
    DatasetName.__value__,
    Option("--dataset", "-d", help="Dataset name"),
]

NShotOption = Annotated[
    int,
    Option("--n-shot", "-n", min=0, help="Number of examples in system prompt"),
]

OutputOption = Annotated[
    Path | None,
    Option("--output", "-o", help="CSV path for the Label Studio export"),
]

LangExtractNShotOption = Annotated[
    int,
    Option("--n-shot", "-n", min=1, help="Number of examples in system prompt"),
]

TrainStepsOption = Annotated[
    int,
    Option("--train-steps", "-t", min=1, help="Number of training steps"),
]

EvalStepsOption = Annotated[
    int,
    Option("--eval-steps", "-e", min=0, help="Evaluate every N steps, 0 disables"),
]

BatchSizeOption = Annotated[
    int,
    Option("--batch-size", "-b", min=1, help="Training/eval batch size"),
]

PatienceOption = Annotated[
    int,
    Option(
        "--patience",
        min=0,
        help="Evals without improvement before stopping, 0 disables",
    ),
]

ToleranceOption = Annotated[
    float,
    Option(
        "--tolerance",
        min=0,
        help="Minimum eval improvement resetting the patience counter",
    ),
]

LearningRateOption = Annotated[
    float,
    Option("--learning-rate", "-r", min=0, help="Optimizer learning rate"),
]

WarmupRatioOption = Annotated[
    float,
    Option("--warmup-ratio", "-w", min=0, help="Fraction of steps for lr warmup"),
]

ArchOption = Annotated[
    Architecture.__value__ | None,
    Option("--arch", "-a", help="Override auto-detected model architecture"),
]

PrefixInitOption = Annotated[
    PrefixInit.__value__,
    Option("--prefix-init", "-p", help="Prefix initialization method"),
]

HeadOnlyOption = Annotated[
    bool,
    Option(help="Freeze all parameters except for classifier head"),
]

LoraRankOption = Annotated[
    int,
    Option("--lora-rank", min=1, help="LoRA adapter rank"),
]

LoraAlphaOption = Annotated[
    int,
    Option("--lora-alpha", min=1, help="LoRA scaling factor"),
]

LoraDropoutOption = Annotated[
    float,
    Option("--lora-dropout", min=0, max=1, help="LoRA dropout probability"),
]

TrackingUriOption = Annotated[
    str,
    Option(envvar="MLFLOW_TRACKING_URI", help="MLflow tracking URI"),
]

ExperimentOption = Annotated[
    str,
    Option(envvar="MLFLOW_EXPERIMENT", help="Experiment name for mlflow tracking"),
]

ExportRunOption = Annotated[
    str,
    Option(
        "--mlflow-run-id",
        envvar="MLFLOW_RUN_ID",
        help="Run whose prediction shards to export",
    ),
]

RunNameOption = Annotated[
    str | None,
    Option(
        envvar="MLFLOW_RUN_NAME",
        help="Run name for mlflow tracking, inferred from parameters by default",
    ),
]

LogLevelOption = Annotated[
    LogLevel.__value__,
    Option("--log-level", "-l", help="Log level"),
]

SeedOption = Annotated[
    int | None,
    Option("--seed", "-s", min=0, help="Random seed"),
]

PromptArgument = Annotated[
    str,
    Argument(help="User prompt to send to the server"),
]

SystemOption = Annotated[
    str | None,
    Option("--system", help="System prompt, defaults to the dataset prompt"),
]

BaseUrlOption = Annotated[
    str,
    Option("--base-url", help="Base URL of the server"),
]

ShardSizeOption = Annotated[
    int,
    Option("--shard-size", min=1, help="Records per prediction shard"),
]

LogIntervalOption = Annotated[
    int,
    Option("--log-interval", min=1, help="Samples between running metric logs"),
]

ConcurrencyOption = Annotated[
    int,
    Option("--concurrency", "-c", min=1, help="Concurrent server requests"),
]

ProviderOption = Annotated[
    Provider.__value__,
    Option("--provider", help="Server provider"),
]

SourceOption = Annotated[
    DataSource.__value__,
    Option("--source", help="Est-Health 30 query for discontinuation events"),
]

CompletionDatasetOption = Annotated[
    DatasetName.__value__ | None,
    Option("--dataset", "-d", help="Constrain output to the dataset label set"),
]

FromIdOption = Annotated[
    int | None,
    Option(
        "--from-id",
        min=0,
        help="Only stream rows with id greater than this value",
    ),
]

RunIdOption = Annotated[
    str | None,
    Option(
        "--mlflow-run-id",
        envvar="MLFLOW_RUN_ID",
        help="Reattach to an existing run and append results",
    ),
]

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
