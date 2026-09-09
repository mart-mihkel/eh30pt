"""Est-Health 30 workflows."""

import asyncio
from typing import Annotated

from typer import Argument, Option, Typer

from eh30pt.logging import setup_logging
from eh30pt.types import DatasetName, LogLevel

app = Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Est-Health 30 workflows",
    no_args_is_help=True,
)

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

DatasetOption = Annotated[
    DatasetName.__value__ | None,
    Option("--dataset", "-d", help="Constrain output to the dataset label set"),
]

EvalDatasetOption = Annotated[
    DatasetName.__value__,
    Option("--dataset", "-d", help="Dataset to evaluate on"),
]

NShotOption = Annotated[
    int,
    Option("--n-shot", "-n", min=0, help="Number of few-shot examples"),
]

RunNameOption = Annotated[
    str | None,
    Option(help="Run name for mlflow tracking, inferred from parameters by default"),
]

ExperimentOption = Annotated[
    str,
    Option(help="Experiment name for mlflow tracking"),
]

TrackingUriOption = Annotated[
    str,
    Option(
        envvar="MLFLOW_TRACKING_URI",
        help="Can be overriden with envrionment variables",
    ),
]

SeedOption = Annotated[
    int | None,
    Option("--seed", "-s", min=0, help="Sampling seed"),
]

OneConvOption = Annotated[
    bool,
    Option(
        "--one-conv/--no-one-conv",
        help="Keep all samples in one conversation instead of restarting per sample",
    ),
]

LogLevelOption = Annotated[
    LogLevel.__value__,
    Option("--log-level", "-l", help="Log level"),
]


@app.callback()
def _setup() -> None:
    """Set up logging before running a command."""
    setup_logging()


@app.command(help="Query a llama server and print the completion")
def completion(
    *,
    user_prompt: PromptArgument,
    system_prompt: SystemOption,
    base_url: BaseUrlOption = "http://localhost:8080",
    dataset: DatasetOption = None,
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
def infer(
    *,
    system_prompt: SystemOption,
    base_url: BaseUrlOption = "http://localhost:8080",
    dataset: EvalDatasetOption,
    n_shot: NShotOption = 0,
    one_conv: OneConvOption = False,
    seed: SeedOption = None,
    mlflow_run_name: RunNameOption = None,
    mlflow_experiment: ExperimentOption = "eh30pt",
    mlflow_tracking_uri: TrackingUriOption = "sqlite:///mlflow.db",
    log_level: LogLevelOption = "info",
) -> None:
    """Run few-shot test evaluation against a llama server."""
    from eh30pt.cli.scripts.infer import main
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


if __name__ == "__main__":
    app()
