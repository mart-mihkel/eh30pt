"""Command line interface."""

import asyncio
from typing import Annotated

from typer import Argument, Option, Typer

from eh30pt.logging import setup_logging
from eh30pt.types import Architecture, DatasetName, LogLevel, PrefixInit

app = Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Interface for running scripts",
    no_args_is_help=True,
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

ResumeOption = Annotated[
    bool,
    Option(
        "--resume/--no-resume",
        help="Reattach to the latest unfinished run with identical config",
    ),
]


TrackingUriOption = Annotated[
    str,
    Option(
        envvar="MLFLOW_TRACKING_URI",
        help="Can be overriden with envrionment variables",
    ),
]

ExperimentOption = Annotated[
    str,
    Option(help="Experiment name for mlflow tracking"),
]

RunNameOption = Annotated[
    str | None,
    Option(help="Run name for mlflow tracking, inferred from parameters by default"),
]

MetricsOption = Annotated[
    list[str] | None,
    Option("--metric", help="Per-step metric to export the history of, repeatable"),
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
    str,
    Option("--system", help="System prompt steering the completion"),
]

BaseUrlOption = Annotated[
    str,
    Option("--base-url", help="Base URL of the llama server"),
]

CompletionDatasetOption = Annotated[
    DatasetName.__value__ | None,
    Option("--dataset", "-d", help="Constrain output to the dataset label set"),
]

OneConvOption = Annotated[
    bool,
    Option(
        "--one-conv/--no-one-conv",
        help="Keep all samples in one conversation instead of restarting per sample",
    ),
]


def _set_seed(seed: int | None) -> None:
    import random

    import numpy as np
    import torch

    from eh30pt.logging import logger

    if seed is None:
        logger.warning("randomness is not fixed")
        return

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.mps.is_available():
        torch.mps.manual_seed(seed)


def _log_accelerator() -> None:
    import torch

    from eh30pt.logging import logger

    if torch.cuda.is_available():
        logger.info("cuda available")
    elif torch.mps.is_available():
        logger.info("metal available")
    else:
        logger.warning("no acceleator available")


@app.callback()
def _setup() -> None:
    """Set up logging before running a command."""
    setup_logging()


@app.command(help="Fine-tune and run test evaluation")
def fine_tune(
    *,
    model: ModelOption,
    dataset: DatasetOption,
    head_only: HeadOnlyOption = False,
    n_shot: NShotOption = 0,
    train_steps: TrainStepsOption,
    eval_steps: EvalStepsOption = 0,
    batch_size: BatchSizeOption = 8,
    learning_rate: LearningRateOption = 5e-5,
    warmup_ratio: WarmupRatioOption = 0.06,
    patience: PatienceOption = 0,
    tolerance: ToleranceOption = 0.01,
    arch: ArchOption = None,
    resume: ResumeOption = True,
    mlflow_experiment: ExperimentOption = "eh30pt",
    mlflow_run_name: RunNameOption = None,
    mlflow_tracking_uri: TrackingUriOption = "sqlite:///mlflow.db",
    log_level: LogLevelOption = "info",
    seed: SeedOption = None,
) -> None:
    """Fine-tune and run test evaluation."""
    from eh30pt.cli.scripts.finetune import main
    from eh30pt.logging import logger

    logger.setLevel(log_level.upper())
    _log_accelerator()
    _set_seed(seed)

    main(
        model_path=model,
        dataset=dataset,
        arch=arch,
        head_only=head_only,
        n_shot=n_shot,
        eval_steps=eval_steps,
        patience=patience,
        tolerance=tolerance,
        resume=resume,
        train_steps=train_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        mlflow_experiment=mlflow_experiment,
        mlflow_run_name=mlflow_run_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
        seed=seed,
    )


@app.command(help="Prompt-tune and run test evaluation")
def prompt_tune(
    *,
    model: ModelOption,
    dataset: DatasetOption,
    prefix_init: PrefixInitOption = "pretrained",
    n_shot: NShotOption = 0,
    train_steps: TrainStepsOption,
    eval_steps: EvalStepsOption = 0,
    batch_size: BatchSizeOption = 8,
    learning_rate: LearningRateOption = 1e-3,
    warmup_ratio: WarmupRatioOption = 0.06,
    patience: PatienceOption = 0,
    tolerance: ToleranceOption = 0.01,
    arch: ArchOption = None,
    resume: ResumeOption = True,
    mlflow_experiment: ExperimentOption = "eh30pt",
    mlflow_run_name: RunNameOption = None,
    mlflow_tracking_uri: TrackingUriOption = "sqlite:///mlflow.db",
    log_level: LogLevelOption = "info",
    seed: SeedOption = None,
) -> None:
    """Prompt-tune and run test evaluation."""
    from eh30pt.cli.scripts.ptune import main
    from eh30pt.logging import logger

    logger.setLevel(log_level.upper())
    _log_accelerator()
    _set_seed(seed)

    main(
        model_path=model,
        dataset=dataset,
        prefix_init=prefix_init,
        arch=arch,
        n_shot=n_shot,
        eval_steps=eval_steps,
        patience=patience,
        tolerance=tolerance,
        resume=resume,
        train_steps=train_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        mlflow_experiment=mlflow_experiment,
        mlflow_run_name=mlflow_run_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
        seed=seed,
    )


@app.command(help="Run test evaluation with few-shot learning")
def few_shot(
    *,
    model: ModelOption,
    dataset: DatasetOption,
    n_shot: NShotOption = 0,
    batch_size: BatchSizeOption = 8,
    arch: ArchOption = None,
    mlflow_experiment: ExperimentOption = "eh30pt",
    mlflow_run_name: RunNameOption = None,
    mlflow_tracking_uri: TrackingUriOption = "sqlite:///mlflow.db",
    log_level: LogLevelOption = "info",
    seed: SeedOption = None,
) -> None:
    """Run test evaluation with few-shot learning."""
    from eh30pt.cli.scripts.fewshot import main
    from eh30pt.logging import logger

    logger.setLevel(log_level.upper())
    _log_accelerator()
    _set_seed(seed)

    main(
        model_path=model,
        arch=arch,
        dataset=dataset,
        n_shot=n_shot,
        batch_size=batch_size,
        mlflow_experiment=mlflow_experiment,
        mlflow_run_name=mlflow_run_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
        seed=seed,
    )


@app.command(help="Query a llama server and print the completion")
def completion(
    *,
    user_prompt: PromptArgument,
    system_prompt: SystemOption,
    base_url: BaseUrlOption = "http://localhost:8080",
    dataset: CompletionDatasetOption = None,
    log_level: LogLevelOption = "info",
) -> None:
    """Query a llama server and print the completion."""
    from eh30pt.cli.scripts.completion import main
    from eh30pt.datasets.registry import get_labels
    from eh30pt.logging import logger

    logger.setLevel(log_level.upper())
    labels = get_labels(dataset) if dataset is not None else None
    asyncio.run(main(user_prompt, system_prompt, base_url, labels=labels))


@app.command(help="Run few-shot test evaluation against a llama server")
def evaluate(
    *,
    system_prompt: SystemOption,
    base_url: BaseUrlOption = "http://localhost:8080",
    dataset: DatasetOption,
    n_shot: NShotOption = 0,
    one_conv: OneConvOption = False,
    seed: SeedOption = None,
    mlflow_run_name: RunNameOption = None,
    mlflow_experiment: ExperimentOption = "eh30pt",
    mlflow_tracking_uri: TrackingUriOption = "sqlite:///mlflow.db",
    log_level: LogLevelOption = "info",
) -> None:
    """Run few-shot test evaluation against a llama server."""
    from eh30pt.cli.scripts.evaluate import main
    from eh30pt.logging import logger

    logger.setLevel(log_level.upper())

    asyncio.run(
        main(
            system_prompt=system_prompt,
            base_url=base_url,
            dataset=dataset,
            n_shot=n_shot,
            one_conv=one_conv,
            mlflow_run_name=mlflow_run_name,
            mlflow_experiment=mlflow_experiment,
            mlflow_tracking_uri=mlflow_tracking_uri,
            seed=seed,
        ),
    )


@app.command(help="Export MLflow experiments to csv")
def collect(
    *,
    mlflow_experiment: ExperimentOption = "eh30pt",
    mlflow_tracking_uri: TrackingUriOption = "sqlite:///mlflow.db",
    metrics: MetricsOption = None,
    log_level: LogLevelOption = "info",
) -> None:
    """Export MLflow experiments to csv."""
    from eh30pt.logging import logger
    from eh30pt.tracking import collect_runs

    logger.setLevel(log_level.upper())
    collect_runs(
        mlflow_experiment,
        mlflow_tracking_uri,
        tuple(metrics or ()),
        write_csv=True,
    )


if __name__ == "__main__":
    app()
