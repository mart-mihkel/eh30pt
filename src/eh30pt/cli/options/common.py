"""Shared model, dataset and run-shape options."""

from pathlib import Path
from typing import Annotated

from typer import Option

from eh30pt.types import Architecture, DatasetName, LogLevel

ModelOption = Annotated[
    str,
    Option("--model", "-m", help="HuggingFace model or path to checkpoint"),
]

DatasetOption = Annotated[
    DatasetName.__value__,
    Option("--dataset", "-d", help="Dataset name"),
]

InferredDatasetOption = Annotated[
    DatasetName.__value__ | None,
    Option(
        "--dataset",
        "-d",
        help="Dataset name, inferred from the run when omitted",
    ),
]

NShotOption = Annotated[
    int,
    Option("--n-shot", "-n", min=0, help="Number of examples in system prompt"),
]

LangExtractNShotOption = Annotated[
    int,
    Option("--n-shot", "-n", min=1, help="Number of examples in system prompt"),
]

OutputOption = Annotated[
    Path | None,
    Option("--output", "-o", help="Path for the generated export"),
]

LogLevelOption = Annotated[
    LogLevel.__value__,
    Option("--log-level", "-l", help="Log level"),
]

SeedOption = Annotated[
    int | None,
    Option("--seed", "-s", min=0, help="Random seed"),
]

ArchOption = Annotated[
    Architecture.__value__ | None,
    Option("--arch", "-a", help="Override auto-detected model architecture"),
]
