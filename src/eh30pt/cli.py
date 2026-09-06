"""Command line interface."""

from typing import Annotated

from typer import Option, Typer

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

ArchOption = Annotated[
    Architecture.__value__ | None,
    Option("--arch", "-a", help="Override auto-detected model architecture"),
]

NShotOption = Annotated[
    int,
    Option("--n-shot", "-n", min=0, help="Number of examples in system prompt"),
]

TrainSamplesOption = Annotated[
    int | None,
    Option(
        "--train-samples",
        "-t",
        min=0,
        help="If present take a subset of tokenized train data",
    ),
]

ValSamplesOption = Annotated[
    int | None,
    Option(
        "--val-samples",
        "-v",
        min=0,
        help="If present take a subset of tokenized validation data",
    ),
]

DoEvalOption = Annotated[
    bool,
    Option(help="Run evalutaion during training"),
]

PatienceOption = Annotated[
    int,
    Option(
        "--patience",
        min=0,
        help="Eval epochs without improvement before stopping, 0 disables",
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

ResumeOption = Annotated[
    bool,
    Option(
        "--resume/--no-resume",
        help="Reattach to the latest unfinished run with identical config",
    ),
]

EpochsOption = Annotated[
    int,
    Option("--epochs", "-e", min=0, help="Number of training epochs"),
]

BatchSizeOption = Annotated[
    int,
    Option("--batch-size", "-b", min=1, help="Training/eval batch size"),
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

LogLevelOption = Annotated[
    LogLevel.__value__,
    Option("--log-level", "-l", help="Log level"),
]

SeedOption = Annotated[
    int | None,
    Option(min=0, help="Random seed"),
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
    model: ModelOption,
    dataset: DatasetOption,
    *,
    arch: ArchOption = None,
    head_only: Annotated[
        bool,
        Option(help="Freeze all parameters except for classifier head"),
    ] = False,
    n_shot: NShotOption = 0,
    train_samples: TrainSamplesOption = None,
    val_samples: ValSamplesOption = None,
    do_eval: DoEvalOption = False,
    patience: PatienceOption = 0,
    tolerance: ToleranceOption = 0.01,
    resume: ResumeOption = True,
    epochs: EpochsOption = 5,
    batch_size: BatchSizeOption = 8,
    learning_rate: Annotated[
        float,
        Option("--learning-rate", "-r", min=0, help="Optimizer learning rate"),
    ] = 5e-5,
    mlflow_experiment: ExperimentOption = "eh30pt",
    mlflow_run_name: RunNameOption = None,
    mlflow_tracking_uri: TrackingUriOption = "sqlite:///mlflow.db",
    log_level: LogLevelOption = "info",
    seed: SeedOption = None,
) -> None:
    """Fine-tune and run test evaluation."""
    from eh30pt.logging import logger
    from eh30pt.scripts.finetune import main

    logger.setLevel(log_level.upper())
    _log_accelerator()
    _set_seed(seed)

    main(
        model_path=model,
        dataset=dataset,
        arch=arch,
        head_only=head_only,
        n_shot=n_shot,
        train_samples=train_samples,
        val_samples=val_samples,
        do_eval=do_eval,
        patience=patience,
        tolerance=tolerance,
        resume=resume,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mlflow_experiment=mlflow_experiment,
        mlflow_run_name=mlflow_run_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
        seed=seed,
    )


@app.command(help="Prompt-tune and run test evaluation")
def prompt_tune(
    model: ModelOption,
    dataset: DatasetOption,
    *,
    prefix_init: Annotated[
        PrefixInit.__value__,
        Option("--prefix-init", "-p", help="Prefix initialization method"),
    ] = "pretrained",
    arch: ArchOption = None,
    n_shot: NShotOption = 0,
    train_samples: TrainSamplesOption = None,
    val_samples: ValSamplesOption = None,
    do_eval: DoEvalOption = False,
    patience: PatienceOption = 0,
    tolerance: ToleranceOption = 0.01,
    resume: ResumeOption = True,
    epochs: EpochsOption = 5,
    batch_size: BatchSizeOption = 8,
    learning_rate: Annotated[
        float,
        Option("--learning-rate", "-r", min=0, help="Optimizer learning rate"),
    ] = 1e-3,
    mlflow_experiment: ExperimentOption = "eh30pt",
    mlflow_run_name: RunNameOption = None,
    mlflow_tracking_uri: TrackingUriOption = "sqlite:///mlflow.db",
    log_level: LogLevelOption = "info",
    seed: SeedOption = None,
) -> None:
    """Prompt-tune and run test evaluation."""
    from eh30pt.logging import logger
    from eh30pt.scripts.ptune import main

    logger.setLevel(log_level.upper())
    _log_accelerator()
    _set_seed(seed)

    main(
        model_path=model,
        dataset=dataset,
        prefix_init=prefix_init,
        arch=arch,
        n_shot=n_shot,
        train_samples=train_samples,
        val_samples=val_samples,
        do_eval=do_eval,
        patience=patience,
        tolerance=tolerance,
        resume=resume,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mlflow_experiment=mlflow_experiment,
        mlflow_run_name=mlflow_run_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
        seed=seed,
    )


@app.command(help="Run test evaluation with few-shot learning")
def few_shot(
    model: ModelOption,
    dataset: DatasetOption,
    *,
    arch: ArchOption = None,
    n_shot: NShotOption = 0,
    batch_size: BatchSizeOption = 8,
    mlflow_experiment: ExperimentOption = "eh30pt",
    mlflow_run_name: RunNameOption = None,
    mlflow_tracking_uri: TrackingUriOption = "sqlite:///mlflow.db",
    log_level: LogLevelOption = "info",
    seed: SeedOption = None,
) -> None:
    """Run test evaluation with few-shot learning."""
    from eh30pt.logging import logger
    from eh30pt.scripts.fewshot import main

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


@app.command(help="Export MLflow experiments to csv")
def collect(
    *,
    mlflow_experiment: ExperimentOption = "eh30pt",
    mlflow_tracking_uri: TrackingUriOption = "sqlite:///mlflow.db",
    metrics: Annotated[
        list[str] | None,
        Option("--metric", help="Per-step metric to export the history of, repeatable"),
    ] = None,
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
