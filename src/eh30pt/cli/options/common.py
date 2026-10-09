"""Shared model, dataset and run-shape options."""

from pathlib import Path

import rich_click as click

# Unannotated on purpose: click decorators are generic over the decorated
# callable, which a variable annotation cannot express, so the precise type
# is left for inference.

ModelOption = click.option(
    "--model",
    "-m",
    "model",
    required=True,
    type=str,
    help="HuggingFace model or path to checkpoint",
)

DatasetOption = click.option(
    "--dataset",
    "-d",
    "dataset",
    required=True,
    type=click.Choice(
        [
            "multinerd",
            "oblique",
            "discont-classifier",
            "discont-decider",
            "discont-extract",
            "discont-multiple",
        ]
    ),
    help="Dataset name",
)

InferredDatasetOption = click.option(
    "--dataset",
    "-d",
    "dataset",
    required=False,
    default=None,
    type=click.Choice(
        [
            "multinerd",
            "oblique",
            "discont-classifier",
            "discont-decider",
            "discont-extract",
            "discont-multiple",
        ]
    ),
    help="Dataset name, inferred from the run when omitted",
)

NShotOption = click.option(
    "--n-shot",
    "-n",
    "n_shot",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Number of examples in system prompt",
)

LangExtractNShotOption = click.option(
    "--n-shot",
    "-n",
    "n_shot",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Number of examples in system prompt",
)

OutputOption = click.option(
    "--output",
    "-o",
    "output",
    type=click.Path(path_type=Path),
    default=None,
    help="Path for the generated export",
)

LogLevelOption = click.option(
    "--log-level",
    "-l",
    "log_level",
    type=click.Choice(["debug", "info", "warning", "error"]),
    default="info",
    show_default=True,
    help="Log level",
)

SeedOption = click.option(
    "--seed",
    "-s",
    "seed",
    type=click.IntRange(min=0),
    default=None,
    help="Random seed",
)

ArchOption = click.option(
    "--arch",
    "-a",
    "arch",
    type=click.Choice(["encoder", "decoder", "encoder-decoder"]),
    default=None,
    help="Override auto-detected model architecture",
)
