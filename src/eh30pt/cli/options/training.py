"""Training loop options."""

from typing import TYPE_CHECKING

import rich_click as click

if TYPE_CHECKING:
    from collections.abc import Callable

    from rich_click.decorators import FC

train_steps_option = click.option(
    "--train-steps",
    "-t",
    "train_steps",
    required=True,
    type=click.IntRange(min=1),
    help="Number of training steps",
)

eval_steps_option = click.option(
    "--eval-steps",
    "-e",
    "eval_steps",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Evaluate every N steps, 0 disables",
)

batch_size_option = click.option(
    "--batch-size",
    "-b",
    "batch_size",
    type=click.IntRange(min=1),
    default=8,
    show_default=True,
    help="Training/eval batch size",
)

patience_option = click.option(
    "--patience",
    "patience",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Evals without improvement before stopping, 0 disables",
)

tolerance_option = click.option(
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


def warmup_ratio_option(default: float) -> Callable[[FC], FC]:
    """Build the optimizer warmup ratio option with a per-command default."""
    return click.option(
        "--warmup-ratio",
        "-w",
        "warmup_ratio",
        type=click.FloatRange(min=0),
        default=default,
        show_default=True,
        help="Fraction of steps for lr warmup",
    )


prefix_init_option = click.option(
    "--prefix-init",
    "prefix_init",
    type=click.Choice(["pretrained", "random"]),
    default="pretrained",
    show_default=True,
    help="Prefix initialization method",
)

head_only_option = click.option(
    "--head-only",
    "head_only",
    is_flag=True,
    default=False,
    help="Freeze all parameters except for classifier head",
)

constrain_option = click.option(
    "--constrain/--no-constrain",
    "constrain",
    default=False,
    show_default=True,
    help="Constrain eval predictions to legal label token sequences",
)
