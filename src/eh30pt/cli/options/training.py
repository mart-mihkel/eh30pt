"""Training loop options."""

from typing import Annotated

from typer import Option

from eh30pt.types import PrefixInit

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

PrefixInitOption = Annotated[
    PrefixInit.__value__,
    Option("--prefix-init", "-p", help="Prefix initialization method"),
]

HeadOnlyOption = Annotated[
    bool,
    Option(help="Freeze all parameters except for classifier head"),
]
