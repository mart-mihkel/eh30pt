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

NShotOption = Annotated[
    int,
    Option("--n-shot", "-n", min=0, help="Number of few-shot examples"),
]

DatasetOption = Annotated[
    DatasetName.__value__ | None,
    Option(
        "--dataset",
        "-d",
        help="Constrain output to the dataset label set",
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
    user_prompt: PromptArgument,
    system_prompt: SystemOption,
    *,
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


if __name__ == "__main__":
    app()
