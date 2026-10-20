"""Training loop options."""

from typing import TYPE_CHECKING

import rich_click as click

if TYPE_CHECKING:
    from collections.abc import Callable

    from rich_click.decorators import FC

    from eh30pt.types import EvalSteps


class _EvalStepsType(click.ParamType):
    """Accept a non-negative step count or the literal 'epoch'."""

    name = "steps"

    def convert(
        self,
        value: object,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> EvalSteps:
        if value == "epoch":
            return "epoch"

        try:
            steps = int(str(value))
        except ValueError:
            self.fail(f"'{value}' is not a step count or 'epoch'", param, ctx)

        if steps < 0:
            self.fail(f"'{value}' is not a step count or 'epoch'", param, ctx)

        return steps


epochs_option = click.option(
    "--epochs",
    "-e",
    "epochs",
    default=3,
    type=click.FloatRange(min=0, min_open=True),
    help="Number of passes over the train split",
)

train_size_option = click.option(
    "--train-size",
    "-t",
    "train_size",
    type=click.IntRange(min=1),
    default=None,
    help="Use only the first N train examples, full split when omitted",
)

save_model_option = click.option(
    "--save-model/--no-save-model",
    "save_model",
    default=False,
    show_default=True,
    help="Save HF checkpoints and log the model as an artifact",
)

eval_steps_option = click.option(
    "--eval-steps",
    "-es",
    "eval_steps",
    type=_EvalStepsType(),
    default=0,
    show_default=True,
    help="Evaluate every N steps, 'epoch' evaluates once per epoch, 0 disables",
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
    "-pat",
    "patience",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Evals without improvement before stopping, 0 disables",
)

tolerance_option = click.option(
    "--tolerance",
    "-tol",
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
        "-lr",
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
        "-wr",
        "warmup_ratio",
        type=click.FloatRange(min=0),
        default=default,
        show_default=True,
        help="Fraction of steps for lr warmup",
    )


prefix_init_option = click.option(
    "--prefix-init",
    "-pi",
    "prefix_init",
    type=click.Choice(["pretrained", "random"]),
    default="pretrained",
    show_default=True,
    help="Prefix initialization method",
)

num_virtual_option = click.option(
    "--num-virtual",
    "-nv",
    "num_virtual",
    type=click.IntRange(min=1),
    default=None,
    help="Virtual token count, defaults to the system prompt length",
)

head_only_option = click.option(
    "--head-only",
    "-ho",
    "head_only",
    is_flag=True,
    default=False,
    help="Freeze all parameters except for classifier head",
)

sequence_classification_option = click.option(
    "--sequence-classification",
    "-sc",
    "sequence_classification",
    is_flag=True,
    default=False,
    help="Load decoder and encoder-decoder models with a classification head",
)
