"""Training loop options."""

from typing import TYPE_CHECKING

import rich_click as click

if TYPE_CHECKING:
    from collections.abc import Callable

    from rich_click.decorators import FC

# Unannotated on purpose: click decorators are generic over the decorated
# callable, which a variable annotation cannot express, so the precise type
# is left for inference.

TrainStepsOption = click.option(
    "--train-steps",
    "-t",
    "train_steps",
    required=True,
    type=click.IntRange(min=1),
    help="Number of training steps",
)

EvalStepsOption = click.option(
    "--eval-steps",
    "-e",
    "eval_steps",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Evaluate every N steps, 0 disables",
)

BatchSizeOption = click.option(
    "--batch-size",
    "-b",
    "batch_size",
    type=click.IntRange(min=1),
    default=8,
    show_default=True,
    help="Training/eval batch size",
)

PatienceOption = click.option(
    "--patience",
    "patience",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Evals without improvement before stopping, 0 disables",
)

ToleranceOption = click.option(
    "--tolerance",
    "tolerance",
    type=click.FloatRange(min=0),
    default=0.01,
    show_default=True,
    help="Minimum eval improvement resetting the patience counter",
)


def learning_rate_option(default: float) -> Callable[[FC], FC]:
    """Build the optimizer learning rate option with a per-command default."""
    return click.option(
        "--learning-rate",
        "-r",
        "learning_rate",
        type=click.FloatRange(min=0),
        default=default,
        show_default=True,
        help="Optimizer learning rate",
    )


WarmupRatioOption = click.option(
    "--warmup-ratio",
    "-w",
    "warmup_ratio",
    type=click.FloatRange(min=0),
    default=0.06,
    show_default=True,
    help="Fraction of steps for lr warmup",
)

PrefixInitOption = click.option(
    "--prefix-init",
    "prefix_init",
    type=click.Choice(["pretrained", "random"]),
    default="pretrained",
    show_default=True,
    help="Prefix initialization method",
)

HeadOnlyOption = click.option(
    "--head-only",
    "head_only",
    is_flag=True,
    default=False,
    help="Freeze all parameters except for classifier head",
)
